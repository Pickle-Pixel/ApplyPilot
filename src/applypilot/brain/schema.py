"""Install and deeply verify the additive Postgres canonical-brain schema."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from types import MappingProxyType

from psycopg import sql
from psycopg.pq import TransactionStatus

_SCHEMA_SQL = Path(__file__).with_name("schema_v1.sql")
_SCHEMA_V2_SQL = Path(__file__).with_name("schema_v2.sql")
_SCHEMA_V3_SQL = Path(__file__).with_name("schema_v3.sql")
_SCHEMA_V4_SQL = Path(__file__).with_name("schema_v4.sql")
_SCHEMA_V5_SQL = Path(__file__).with_name("schema_v5.sql")
_SCHEMA_V6_SQL = Path(__file__).with_name("schema_v6.sql")
_SCHEMA_V7_SQL = Path(__file__).with_name("schema_v7.sql")
_SCHEMA_V8_SQL = Path(__file__).with_name("schema_v8.sql")
_SCHEMA_LOCK_KEY = "applypilot:brain:schema:v1"
_MIGRATION_NAME = "brain schema v1"
_MIGRATION_V2_NAME = "brain schema v2 lifecycle principals"
_MIGRATION_V3_NAME = "brain schema v3 lane canary pins"
_MIGRATION_V4_NAME = "brain schema v4 scoped candidate authority"
_MIGRATION_V5_NAME = "brain schema v5 durable factual graph authority"
_MIGRATION_V6_NAME = "brain schema v6 immutable artifact authority"
_MIGRATION_V7_NAME = "brain schema v7 replay-safe artifact authority"
_MIGRATION_V8_NAME = "brain schema v8 incarnation and retention authority"
_EXPECTED_V4_CHECKSUM = "51c61d0035cd7e503a7824539b56a505727bce8496f70e48f9d8e2576f1267d7"
_EXPECTED_V5_CHECKSUM = "52d1726bb13df54591fcd6343884ec8f6f7d18ab7fbf74b4dc33ba509cb0e559"
_EXPECTED_V6_CHECKSUM = "74503db87872670bb7db61498fe0f870f0de13928286dc58c6049d57f8dd2955"
_EXPECTED_V7_CHECKSUM = "d76b037dee1bfd285026ab2503b16be2a39b0c6726eac80a099e3f825ccdfc9f"
_EXPECTED_V8_CHECKSUM = "9af16b97aea9da1ad83755543cbe5da40062f37d942ad64fcf918e8bd70271f3"
_LEGACY_V1_CHECKSUMS = frozenset({
    # Canonical V1 ledger value emitted before the PG18 catalog fingerprint
    # pin correction. The deep catalog verifier below remains mandatory.
    "1976fdc4dbeac4cf853ce6e445efb071651696ae47ee67b61e58aab98ff4e2a5",
})
_MIGRATION_ROLE = "brain_schema_migrator"
_VERIFIER_ROLE = "brain_schema_verifier"
_UNPINNED_PG18_CATALOG_HASH = "PG18_PIN_REQUIRED"
_PG_CATALOG_HASHES = MappingProxyType({
    18: MappingProxyType({
        "base": "22cfe2c608305d89dd773a0f0a4ed91a35be066aab0468154de08be2361b2c6a",
        "current_base": "bc4b812202b72029bd2422fff11b2d803bfe4f59283ae0fe8faddcf79dfe4648",
        "v5": "c84d51b987a3a6915d3af9f186c3c179f09828c2e1e98f5fe04be60e9c063ad7",
        "current_v5": "903d96ec8c9f0da427771ceae78a5c4d4b39fb38bb72a39bd8894e75b3b05bb1",
        "v6": "413fdaacc60793dab624b7d12def0362d9539521be055d716bc9320450362329",
        "v7": "b8bcc3e3fedea82fe0149d3dd0372332e24b94579c692e949d9b9c3dce8a0ee5",
        "v8": "b1ae7f1df9e7c902d4954b4f1fa29c3fdf9d4c55516ff1f5011375b23163cebf",
    })
})
_PG18_CATALOG_SHAPES = MappingProxyType({
    "pg_catalog.pg_constraint": (
        ("oid", "oid"),
        ("conname", "name"),
        ("connamespace", "oid"),
        ("contype", '"char"'),
        ("condeferrable", "boolean"),
        ("condeferred", "boolean"),
        ("conenforced", "boolean"),
        ("convalidated", "boolean"),
        ("conrelid", "oid"),
        ("contypid", "oid"),
        ("conindid", "oid"),
        ("conparentid", "oid"),
        ("confrelid", "oid"),
        ("confupdtype", '"char"'),
        ("confdeltype", '"char"'),
        ("confmatchtype", '"char"'),
        ("conislocal", "boolean"),
        ("coninhcount", "smallint"),
        ("connoinherit", "boolean"),
        ("conperiod", "boolean"),
        ("conkey", "smallint[]"),
        ("confkey", "smallint[]"),
        ("conpfeqop", "oid[]"),
        ("conppeqop", "oid[]"),
        ("conffeqop", "oid[]"),
        ("confdelsetcols", "smallint[]"),
        ("conexclop", "oid[]"),
        ("conbin", "pg_node_tree"),
    ),
    "pg_catalog.pg_auth_members": (
        ("oid", "oid"),
        ("roleid", "oid"),
        ("member", "oid"),
        ("grantor", "oid"),
        ("admin_option", "boolean"),
        ("inherit_option", "boolean"),
        ("set_option", "boolean"),
    ),
})
_STATUS_ROLE = "brain_status_reader"
_CURRENT_V6_CHECKSUM = "74503db87872670bb7db61498fe0f870f0de13928286dc58c6049d57f8dd2955"
_POLICY_CONTROLLER_ROLE = "brain_policy_controller"
_GRAPH_AUTHORITY_ROLE = "brain_graph_authority"
_CANDIDATE_READER_ROLE = "brain_candidate_reader"
_CANDIDATE_WRITER_ROLE = "brain_candidate_writer"
_V4_PUBLISH_SIGNATURE = (
    "public.brain_publish_v4_candidate(text,text,text,text,text,bigint,uuid,text,text,text,text,text,bigint)"
)
_V4_PUBLISH_ARGUMENTS = (
    "requested_owner_id text, requested_campaign_id text, requested_recommendation_lane text, "
    "requested_execution_channel text, requested_execution_scope text, requested_authority_epoch bigint, "
    "requested_database_incarnation_id uuid, requested_candidate_decision_id text, "
    "requested_semantic_content_hash text, requested_candidate_artifact_hash text, "
    "requested_envelope_id text, requested_envelope_artifact_hash text, "
    "requested_graph_approval_receipt_id bigint"
)
_V4_PUBLISH_ARGUMENT_NAMES = tuple(argument.split()[0] for argument in _V4_PUBLISH_ARGUMENTS.split(", "))
_V5_PUBLISH_SIGNATURE = (
    "public.brain_publish_v5_candidate(text,text,text,text,text,bigint,uuid,text,text,text,text,text,bigint)"
)
_V5_PUBLISH_ARGUMENTS = _V4_PUBLISH_ARGUMENTS
_V5_RELATIONS = {
    "brain_authority_epoch_events",
    "brain_factual_ontology_manifests",
    "brain_factual_ontology_terms",
    "brain_factual_ontology_closures",
    "brain_factual_generations",
    "brain_factual_generation_members",
    "brain_factual_generation_closures",
    "brain_factual_approval_receipts",
    "brain_graph_fact_events",
    "brain_factual_approval_consumptions",
    "brain_factual_generation_coverage",
    "brain_factual_contradictions",
    "brain_factual_contradiction_events",
    "brain_factual_graph_snapshots",
    "brain_factual_snapshot_approval_bindings",
    "brain_v5_candidate_publication_events",
}
_V7_RELATIONS = {"brain_v7_topology_contract"}
_V8_RELATIONS = {"applypilot_database_identity", "brain_artifact_retention_attestations"}
# The exact 16 canonical projection keys emitted by
# applypilot.brain.parity_adapters.compute_import_parity, in emission order.
_V8_PARITY_DEFINITION_VERSION = 2
_V8_PARITY_CHECK_KEYS = (
    "jobs", "aliases", "observations", "applications", "application_events",
    "email_events", "email_event_reviews", "reviewed_outcomes", "research_labels",
    "research_label_confidence", "research_pairwise_labels", "research_kg_artifacts",
    "research_kg_runs", "research_scores", "decision_policy_versions", "job_decisions",
)
_V5_CATALOG_RELATIONS = _V5_RELATIONS | {
    "brain_factual_contradiction_state",
    "brain_graph_approval_consumptions",
    "brain_graph_approval_receipts",
}
_V5_SEQUENCES = {
    "brain_authority_epoch_events_authority_epoch_event_id_seq",
    "brain_factual_contradiction_events_contradiction_event_id_seq",
}
_V5_FUNCTIONS = {
    "brain_v5_sha256_text",
    "brain_v5_frame",
    "brain_compute_factual_ontology_root",
    "brain_compute_factual_membership_root",
    "brain_compute_factual_semantic_root",
    "brain_reject_closed_ontology_term",
    "brain_reject_closed_generation_member",
    "brain_create_factual_ontology",
    "brain_add_factual_ontology_term",
    "brain_close_factual_ontology",
    "brain_create_factual_generation",
    "brain_add_factual_generation_member",
    "brain_close_factual_generation",
    "brain_admit_factual_event",
    "brain_record_factual_assertion_coverage",
    "brain_review_factual_exclusion",
    "brain_create_factual_contradiction",
    "brain_append_factual_contradiction_event",
    "brain_publish_factual_snapshot",
    "brain_record_authority_epoch_event",
    "brain_record_graph_approval_v5",
    "brain_bind_factual_snapshot_approval",
    "brain_publish_v5_candidate",
}
_V5_POLICY_FUNCTIONS = {
    "brain_record_authority_epoch_event",
    "brain_record_graph_approval_v5",
    "brain_bind_factual_snapshot_approval",
}
_V5_GRAPH_AUTHORITY_FUNCTIONS = _V5_FUNCTIONS - _V5_POLICY_FUNCTIONS - {
    "brain_reject_closed_ontology_term",
    "brain_reject_closed_generation_member",
    "brain_publish_v5_candidate",
}
_V4_RELATIONS = {
    "brain_authority_scope_state",
    "brain_authority_transition_events",
    "brain_graph_approval_receipts",
    "brain_v4_candidate_decisions",
    "brain_v4_decision_envelopes",
    "brain_graph_approval_consumptions",
    "brain_immutable_artifact_references",
}
_V5_CATALOG_RELATIONS |= _V4_RELATIONS
_V5_CATALOG_FUNCTIONS = _V5_FUNCTIONS | {"brain_publish_v4_candidate"}
_V4_READ_RELATIONS = {
    "brain_authority_scope_state",
    "brain_v4_candidate_decisions",
    "brain_v4_decision_envelopes",
    "brain_immutable_artifact_references",
}
_V4_COLUMN_CONTRACTS = {
    "brain_authority_scope_state": (
        ("authority_scope_id", "bigint", True, "a", ""),
        ("owner_id", "text", True, "", ""),
        ("campaign_id", "text", True, "", ""),
        ("recommendation_lane", "text", True, "", ""),
        ("execution_channel", "text", True, "", ""),
        ("execution_scope", "text", True, "", ""),
        ("state", "text", True, "", "'active'::text"),
        ("authority_epoch", "bigint", True, "", ""),
        ("database_incarnation_id", "uuid", True, "", ""),
        ("migration_run_id", "bigint", True, "", ""),
        ("source_namespace", "text", True, "", ""),
        ("parity_run_id", "bigint", True, "", ""),
        ("definition_version", "integer", True, "", ""),
        ("report_artifact_hash", "text", True, "", ""),
        ("created_at", "timestamp with time zone", True, "", "now()"),
    ),
    "brain_authority_transition_events": (
        ("authority_transition_event_id", "bigint", True, "a", ""),
        ("authority_scope_id", "bigint", True, "", ""),
        ("event_type", "text", True, "", ""),
        ("authority_epoch", "bigint", True, "", ""),
        ("database_incarnation_id", "uuid", True, "", ""),
        ("actor_id", "text", True, "", ""),
        ("occurred_at", "timestamp with time zone", True, "", "now()"),
        ("created_at", "timestamp with time zone", True, "", "now()"),
    ),
    "brain_graph_approval_receipts": (
        ("graph_approval_receipt_id", "bigint", True, "a", ""),
        ("authority_scope_id", "bigint", True, "", ""),
        ("authority_epoch", "bigint", True, "", ""),
        ("database_incarnation_id", "uuid", True, "", ""),
        ("graph_snapshot_id", "text", True, "", ""),
        ("approval_state", "text", True, "", ""),
        ("approval_artifact_hash", "text", True, "", ""),
        ("predecessor_deny_receipt_hash", "text", False, "", ""),
        ("created_at", "timestamp with time zone", True, "", "now()"),
    ),
    "brain_v4_candidate_decisions": (
        ("candidate_decision_id", "text", True, "", ""),
        ("authority_scope_id", "bigint", True, "", ""),
        ("semantic_content_hash", "text", True, "", ""),
        ("candidate_artifact_hash", "text", True, "", ""),
        ("graph_approval_receipt_id", "bigint", True, "", ""),
        ("published_at", "timestamp with time zone", True, "", "now()"),
    ),
    "brain_v4_decision_envelopes": (
        ("envelope_id", "text", True, "", ""),
        ("candidate_decision_id", "text", True, "", ""),
        ("envelope_artifact_hash", "text", True, "", ""),
        ("created_at", "timestamp with time zone", True, "", "now()"),
    ),
    "brain_graph_approval_consumptions": (
        ("graph_approval_consumption_id", "bigint", True, "a", ""),
        ("graph_approval_receipt_id", "bigint", True, "", ""),
        ("candidate_decision_id", "text", True, "", ""),
        ("authority_scope_id", "bigint", True, "", ""),
        ("consumed_at", "timestamp with time zone", True, "", "now()"),
    ),
    "brain_immutable_artifact_references": (
        ("artifact_reference_id", "bigint", True, "a", ""),
        ("artifact_hash", "text", True, "", ""),
        ("reference_type", "text", True, "", ""),
        ("subject_id", "text", True, "", ""),
        ("created_at", "timestamp with time zone", True, "", "now()"),
    ),
}
_V4_KEY_CONTRACTS = {
    ("brain_authority_scope_state", "p", ("authority_scope_id",)),
    ("brain_authority_scope_state", "u", (
        "owner_id", "campaign_id", "recommendation_lane", "execution_channel", "execution_scope",
    )),
    ("brain_authority_transition_events", "p", ("authority_transition_event_id",)),
    ("brain_authority_transition_events", "u", (
        "authority_scope_id", "event_type", "authority_epoch", "database_incarnation_id",
    )),
    ("brain_graph_approval_receipts", "p", ("graph_approval_receipt_id",)),
    ("brain_graph_approval_receipts", "u", (
        "authority_scope_id", "authority_epoch", "database_incarnation_id", "graph_snapshot_id",
    )),
    ("brain_v4_candidate_decisions", "p", ("candidate_decision_id",)),
    ("brain_v4_candidate_decisions", "u", ("authority_scope_id", "semantic_content_hash")),
    ("brain_v4_decision_envelopes", "p", ("envelope_id",)),
    ("brain_v4_decision_envelopes", "u", ("candidate_decision_id",)),
    ("brain_graph_approval_consumptions", "p", ("graph_approval_consumption_id",)),
    ("brain_graph_approval_consumptions", "u", ("graph_approval_receipt_id",)),
    ("brain_graph_approval_consumptions", "u", ("candidate_decision_id",)),
    ("brain_immutable_artifact_references", "p", ("artifact_reference_id",)),
    ("brain_immutable_artifact_references", "u", ("artifact_hash", "reference_type", "subject_id")),
}
_V4_FOREIGN_KEY_CONTRACTS = {
    ("brain_authority_scope_state", (
        "migration_run_id", "source_namespace", "parity_run_id", "definition_version", "report_artifact_hash",
    ), "brain_parity_runs", (
        "migration_run_id", "source_namespace", "parity_run_id", "definition_version", "report_artifact_hash",
    )),
    ("brain_authority_transition_events", ("authority_scope_id",), "brain_authority_scope_state", ("authority_scope_id",)),
    ("brain_graph_approval_receipts", ("authority_scope_id",), "brain_authority_scope_state", ("authority_scope_id",)),
    ("brain_graph_approval_receipts", ("approval_artifact_hash",), "brain_artifacts", ("artifact_hash",)),
    ("brain_graph_approval_receipts", ("predecessor_deny_receipt_hash",), "brain_artifacts", ("artifact_hash",)),
    ("brain_v4_candidate_decisions", ("authority_scope_id",), "brain_authority_scope_state", ("authority_scope_id",)),
    ("brain_v4_candidate_decisions", ("candidate_artifact_hash",), "brain_artifacts", ("artifact_hash",)),
    ("brain_v4_candidate_decisions", ("graph_approval_receipt_id",), "brain_graph_approval_receipts", ("graph_approval_receipt_id",)),
    ("brain_v4_decision_envelopes", ("candidate_decision_id",), "brain_v4_candidate_decisions", ("candidate_decision_id",)),
    ("brain_v4_decision_envelopes", ("envelope_artifact_hash",), "brain_artifacts", ("artifact_hash",)),
    ("brain_graph_approval_consumptions", ("graph_approval_receipt_id",), "brain_graph_approval_receipts", ("graph_approval_receipt_id",)),
    ("brain_graph_approval_consumptions", ("candidate_decision_id",), "brain_v4_candidate_decisions", ("candidate_decision_id",)),
    ("brain_graph_approval_consumptions", ("authority_scope_id",), "brain_authority_scope_state", ("authority_scope_id",)),
    ("brain_immutable_artifact_references", ("artifact_hash",), "brain_artifacts", ("artifact_hash",)),
}
_V4_SCOPE_CHECK_FRAGMENTS = (
    ("btrim(owner_id)", "<>''"),
    ("btrim(campaign_id)", "<>''"),
    ("recommendation_lane", "core_fit", "strategic_stretch", "qualified_fallback", "review", "reject_hold"),
    ("execution_channel", "ats", "linkedin"),
    ("btrim(execution_scope)", "lower(btrim(execution_scope))", "global"),
    ("state", "active"),
    ("authority_epoch>0",),
)
_STATUS_READ_RELATIONS = {
    "brain_decision_policies",
    "brain_policy_artifacts",
    "brain_policy_approvals",
    "brain_parity_runs",
    "brain_parity_run_events",
}
_STATUS_READ_COLUMNS = {
    "fleet_config": {
        "id", "paused", "ats_paused", "ats_pause_source", "ats_apply_mode",
        "linkedin_apply_mode", "canary_enabled", "linkedin_canary_enabled",
        "canary_remaining", "linkedin_canary_remaining", "ats_policy_version",
        "linkedin_policy_version", "pinned_worker_version", "canary_worker_id",
        "canary_version", "ats_canary_worker_id", "ats_canary_version",
        "linkedin_canary_worker_id", "linkedin_canary_version", "approval_threshold",
        "spend_cap_usd", "linkedin_owner_ip",
    },
    "fleet_decision_policies": {
        "policy_version", "lane", "status", "activated_at", "retired_at",
    },
    "apply_queue": {
        "status", "lease_owner", "lease_expires_at", "decision_id", "policy_version",
        "decision_action", "qualification_verdict", "qualification_score",
        "qualification_floor", "preference_score", "outcome_score", "final_score",
        "decision_confidence", "decision_created_at", "decision_expires_at", "input_hash",
    },
    "linkedin_queue": {
        "status", "lease_owner", "lease_expires_at", "decision_id", "policy_version",
        "decision_action", "qualification_verdict", "qualification_score",
        "qualification_floor", "preference_score", "outcome_score", "final_score",
        "decision_confidence", "decision_created_at", "decision_expires_at", "input_hash",
    },
}
_CONTROLLER_FUNCTIONS = {
    "brain_controller_transition_policy",
    "brain_controller_arm_canary",
    "brain_controller_stop_canary",
}
_LIFECYCLE_LOCK_RELATIONS = {
    "apply_queue",
    "fleet_config",
    "fleet_decision_policies",
    "fleet_desired_state",
    "fleet_worker_principals",
    "linkedin_queue",
    "rate_governor",
    "worker_heartbeat",
    "workers",
}
_LIFECYCLE_OWNER_READ_RELATIONS = _LIFECYCLE_LOCK_RELATIONS | {
    "applied_set",
    "apply_attempts",
    "apply_result_events",
    "fleet_worker_blocklist",
}

_FUNCTIONS = {
    "brain_reject_mutation": "",
    "brain_register_decision": "",
    "brain_require_controller": "",
    "brain_artifact_is_authoritative": "candidate_hash text",
    "brain_check_controller_insert": "",
    "brain_check_migration_run_event": "",
    "brain_check_migration_batch_definition": "",
    "brain_check_migration_batch_event": "",
    "brain_check_migration_checkpoint": "",
    "brain_check_policy_lifecycle": "",
    "brain_bind_policy_gates": (
        "candidate text, candidate_lane text, target text, candidate_definition_version integer"
    ),
    "brain_check_supersession": "",
    "brain_check_parity_pass": "",
    "brain_check_parity_result": "",
    "brain_check_archive_manifest": "",
    "brain_transition_policy": "requested_policy_version text, requested_lifecycle text",
    "brain_controller_transition_policy": (
        "requested_policy_version text, requested_lifecycle text, expected_lane text"
    ),
    "brain_controller_arm_canary": (
        "requested_policy_version text, requested_lane text, requested_capacity integer, "
        "expected_ats_pause_source text, expect_null_ats_pause_source boolean, "
        "heartbeat_max_age_seconds integer"
    ),
    "brain_controller_stop_canary": "requested_lane text",
}


def _bounded_timeout(value: object, default: float = 30.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) and parsed >= 0 else default


_LOCK_TIMEOUT_SECONDS = _bounded_timeout(os.environ.get("APPLYPILOT_BRAIN_SCHEMA_LOCK_TIMEOUT_SECONDS"))

_RELATIONS = {
    "brain_schema_versions": "r",
    "brain_artifacts": "r",
    "brain_artifact_locations": "r",
    "brain_jobs": "r",
    "brain_job_aliases": "r",
    "brain_job_observations": "r",
    "brain_label_events": "r",
    "brain_pairwise_events": "r",
    "brain_email_events": "r",
    "brain_reviewed_outcomes": "r",
    "brain_applications": "r",
    "brain_application_events": "r",
    "brain_decision_policies": "r",
    "brain_policy_artifacts": "r",
    "brain_policy_approvals": "r",
    "brain_policy_gate_definitions": "r",
    "brain_policy_release_gate_events": "r",
    "brain_policy_transition_receipts": "r",
    "brain_policy_activation_receipts": "r",
    "brain_canary_lifecycle_events": "r",
    "brain_decision_identities": "r",
    "brain_job_decisions": "p",
    "brain_migration_sources": "r",
    "brain_migration_runs": "r",
    "brain_migration_run_events": "r",
    "brain_migration_batches": "r",
    "brain_migration_batch_events": "r",
    "brain_migration_checkpoints": "r",
    "brain_migration_quarantine": "r",
    "brain_parity_definitions": "r",
    "brain_parity_runs": "r",
    "brain_parity_results": "r",
    "brain_parity_run_events": "r",
}

# Key columns are verified by exact formatted type and nullability. Constraint and
# trigger checks below cover the remainder of each table's behavioral contract.
_KEY_COLUMNS = {
    ("brain_schema_versions", "migration_checksum"): ("text", True),
    ("brain_artifacts", "artifact_hash"): ("text", True),
    ("brain_artifacts", "request_id"): ("text", True),
    ("brain_artifact_locations", "durability"): ("text", True),
    ("brain_jobs", "job_id"): ("text", True),
    ("brain_job_aliases", "source_database_fingerprint"): ("text", True),
    ("brain_label_events", "job_id"): ("text", False),
    ("brain_pairwise_events", "left_job_id"): ("text", False),
    ("brain_reviewed_outcomes", "email_event_id"): ("bigint", True),
    ("brain_reviewed_outcomes", "weight"): ("numeric", False),
    ("brain_applications", "lane"): ("text", False),
    ("brain_application_events", "application_id"): ("text", True),
    ("brain_policy_artifacts", "artifact_role"): ("text", True),
    ("brain_canary_lifecycle_events", "prior_ats_pause_source"): ("text", False),
    ("brain_job_decisions", "decision_id"): ("text", True),
    ("brain_job_decisions", "policy_version"): ("text", True),
    ("brain_job_decisions", "confidence"): ("numeric", False),
    ("brain_job_decisions", "expires_at"): ("timestamp with time zone", False),
    ("brain_migration_batch_events", "lease_expires_at"): ("timestamp with time zone", False),
    ("brain_parity_results", "mismatch_count"): ("bigint", True),
}

_CONSTRAINTS = {
    "brain_canary_lifecycle_events_lane_check": ("lane", "ats", "linkedin"),
    "brain_canary_lifecycle_events_type_check": ("event_type", "armed", "stopped"),
    "brain_job_aliases_idempotent": ("UNIQUE NULLS NOT DISTINCT", "source_namespace", "source_database_fingerprint"),
    "brain_label_events_endpoint": (
        "CHECK",
        "job_id IS NOT NULL",
        "source_item_id IS NOT NULL",
        "source_item_url IS NOT NULL",
    ),
    "brain_pairwise_events_left_endpoint": ("CHECK", "left_job_id IS NOT NULL", "left_source_item_id IS NOT NULL"),
    "brain_pairwise_events_right_endpoint": ("CHECK", "right_job_id IS NOT NULL", "right_source_item_id IS NOT NULL"),
    "brain_reviewed_outcomes_email_job_fk": (
        "FOREIGN KEY (email_event_id, job_id)",
        "brain_email_events(email_event_id, job_id)",
    ),
    "brain_application_events_application_id_fkey": (
        "FOREIGN KEY (application_id)",
        "brain_applications(application_id)",
    ),
    "brain_policy_artifacts_pkey": ("PRIMARY KEY (policy_version, artifact_role)",),
    "brain_policy_release_gate_truth": (
        "CHECK",
        "gate_state <> 'passed'::text",
        "mismatch_count = 0",
        "unresolved_count = 0",
    ),
    "brain_policy_transition_receipts_gate_fk": (
        "FOREIGN KEY (policy_version, lane, lifecycle, definition_version, gate_name, gate_event_id, gate_state)",
        "brain_policy_release_gate_events(policy_version, lane, lifecycle, definition_version, gate_name, gate_event_id, gate_state)",
    ),
    "brain_job_decisions_identity_fk": (
        "FOREIGN KEY (policy_version, decision_id)",
        "brain_decision_identities(policy_version, decision_id)",
    ),
    "brain_job_decisions_policy_lane_fk": (
        "FOREIGN KEY (policy_version, lane)",
        "brain_decision_policies(policy_version, lane)",
    ),
    "brain_job_decisions_apply_expiry": (
        "CHECK",
        "action = 'apply'::text",
        "expires_at IS NOT NULL",
        "expires_at > created_at",
    ),
    "brain_job_decisions_policy_job_key": ("UNIQUE (policy_version, job_id)",),
    "brain_parity_results_definition_fk": (
        "FOREIGN KEY (definition_version, check_key, table_name, check_type)",
        "brain_parity_definitions(definition_version, check_key, relation_name, check_type)",
    ),
    "brain_migration_batches_lineage_key": (
        "UNIQUE (migration_run_id, source_namespace, source_table, migration_batch_id, batch_ordinal)",
    ),
    "brain_migration_batch_events_batch_fk": (
        "FOREIGN KEY (migration_run_id, source_namespace, source_table, migration_batch_id, batch_ordinal)",
        "brain_migration_batches(migration_run_id, source_namespace, source_table, migration_batch_id, batch_ordinal)",
    ),
    "brain_migration_checkpoints_completed_event_fk": (
        "FOREIGN KEY (migration_run_id, source_namespace, source_table, migration_batch_id, batch_ordinal, migration_batch_event_id, committed_event_type)",
        "brain_migration_batch_events",
    ),
    "brain_migration_quarantine_batch_fk": (
        "FOREIGN KEY (migration_run_id, source_namespace, source_table, migration_batch_id, batch_ordinal)",
        "brain_migration_batches",
    ),
    "brain_parity_results_run_fk": (
        "FOREIGN KEY (migration_run_id, source_namespace, parity_run_id, definition_version, report_artifact_hash)",
        "brain_parity_runs(migration_run_id, source_namespace, parity_run_id, definition_version, report_artifact_hash)",
    ),
    "brain_parity_results_check_key": ("UNIQUE (parity_run_id, definition_version, check_key)",),
}

_INDEXES = {
    "brain_decision_policies_one_active_per_lane": (True, "(lifecycle = 'active'::text)"),
    "brain_job_observations_one_successor": (True, "(supersedes_observation_id IS NOT NULL)"),
    "brain_label_events_one_successor": (True, "(supersedes_label_event_id IS NOT NULL)"),
    "brain_pairwise_events_one_successor": (True, "(supersedes_pairwise_event_id IS NOT NULL)"),
    "brain_job_decisions_job_created_idx": (False, None),
    "brain_job_decisions_policy_action_idx": (False, None),
    "brain_jobs_canonical_url_key": (True, "(canonical_url IS NOT NULL)"),
    "brain_migration_batch_events_claim_idx": (False, None),
    "brain_migration_batches_range_idx": (False, None),
    "brain_migration_checkpoints_run_idx": (False, None),
    "brain_migration_quarantine_run_idx": (False, None),
    "brain_parity_results_run_idx": (False, None),
}

_IMMUTABLE_TABLES = {
    "brain_schema_versions",
    "brain_artifacts",
    "brain_artifact_locations",
    "brain_job_aliases",
    "brain_job_observations",
    "brain_label_events",
    "brain_pairwise_events",
    "brain_email_events",
    "brain_reviewed_outcomes",
    "brain_application_events",
    "brain_policy_artifacts",
    "brain_policy_approvals",
    "brain_policy_gate_definitions",
    "brain_policy_release_gate_events",
    "brain_policy_transition_receipts",
    "brain_policy_activation_receipts",
    "brain_canary_lifecycle_events",
    "brain_decision_identities",
    "brain_job_decisions",
    "brain_migration_sources",
    "brain_migration_runs",
    "brain_migration_run_events",
    "brain_migration_batches",
    "brain_migration_batch_events",
    "brain_migration_checkpoints",
    "brain_migration_quarantine",
    "brain_parity_definitions",
    "brain_parity_runs",
    "brain_parity_results",
    "brain_parity_run_events",
}


def _schema_bytes() -> bytes:
    return _SCHEMA_SQL.read_bytes()


def _schema_checksum() -> str:
    return hashlib.sha256(_schema_bytes()).hexdigest()


def _schema_v2_bytes() -> bytes:
    return _SCHEMA_V2_SQL.read_bytes()


def _schema_v2_checksum() -> str:
    return hashlib.sha256(_schema_v2_bytes()).hexdigest()


def _schema_v3_bytes() -> bytes:
    return _SCHEMA_V3_SQL.read_bytes()


def _schema_v3_checksum() -> str:
    return hashlib.sha256(_schema_v3_bytes()).hexdigest()


def _schema_v4_bytes() -> bytes:
    return _SCHEMA_V4_SQL.read_bytes()


def _schema_v4_checksum() -> str:
    return hashlib.sha256(_schema_v4_bytes()).hexdigest()


def _schema_v5_bytes() -> bytes:
    return _SCHEMA_V5_SQL.read_bytes()


def _schema_v5_checksum() -> str:
    return hashlib.sha256(_schema_v5_bytes()).hexdigest()


def _schema_v6_bytes() -> bytes:
    return _SCHEMA_V6_SQL.read_bytes()


def _schema_v6_checksum() -> str:
    return hashlib.sha256(_schema_v6_bytes()).hexdigest()

def _schema_v7_bytes() -> bytes:
    return _SCHEMA_V7_SQL.read_bytes()


def _schema_v7_checksum() -> str:
    return hashlib.sha256(_schema_v7_bytes()).hexdigest()


def _assert_schema_v7_bytes_immutable() -> None:
    actual_checksum = _schema_v7_checksum()
    if actual_checksum != _EXPECTED_V7_CHECKSUM:
        raise RuntimeError(
            f"immutable schema v7 file checksum mismatch: expected {_EXPECTED_V7_CHECKSUM}, got {actual_checksum}"
        )


def _schema_v8_bytes() -> bytes:
    return _SCHEMA_V8_SQL.read_bytes()


def _schema_v8_checksum() -> str:
    return hashlib.sha256(_schema_v8_bytes()).hexdigest()


def _assert_schema_v8_bytes_immutable() -> None:
    actual_checksum = _schema_v8_checksum()
    if actual_checksum != _EXPECTED_V8_CHECKSUM:
        raise RuntimeError(
            f"immutable schema v8 file checksum mismatch: expected {_EXPECTED_V8_CHECKSUM}, got {actual_checksum}"
        )

def _assert_current_schema_v6_bytes_immutable() -> None:
    actual_checksum = _schema_v6_checksum()
    if actual_checksum != _CURRENT_V6_CHECKSUM:
        raise RuntimeError(
            f"immutable committed schema v6 file checksum mismatch: "
            f"expected {_CURRENT_V6_CHECKSUM}, got {actual_checksum}"
        )



def _assert_schema_v6_bytes_immutable() -> None:
    _assert_current_schema_v6_bytes_immutable()


def _assert_schema_v5_bytes_immutable() -> None:
    actual_checksum = _schema_v5_checksum()
    if actual_checksum != _EXPECTED_V5_CHECKSUM:
        raise RuntimeError(
            f"immutable schema v5 file checksum mismatch: expected {_EXPECTED_V5_CHECKSUM}, "
            f"got {actual_checksum}"
        )


def _compact_sql(value: str) -> str:
    return re.sub(r"\s+", "", value).lower().replace("::text", "")


def _function_body_fingerprint(definition: str) -> str | None:
    body = re.search(r"\bAS\s+(\$[A-Za-z0-9_]*\$)(.*?)\1", definition, re.IGNORECASE | re.DOTALL)
    if body is None:
        return None
    return hashlib.sha256(body.group(2).encode("utf-8")).hexdigest()


def _raw_function_body_fingerprint(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _v6_function_body_fingerprint(name: str) -> str:
    definition = _schema_v6_bytes().decode("utf-8")
    match = re.search(
        rf"CREATE(?: OR REPLACE)? FUNCTION public\.{re.escape(name)}\b.*?"
        r"\bAS\s+(\$[A-Za-z0-9_]*\$)(.*?)\1",
        definition,
        re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"schema v6 function body missing: {name}")
    return _raw_function_body_fingerprint(match.group(2))


def _stable_catalog_records(records) -> list[dict[str, object]]:
    normalized = [dict(record) for record in records]
    return sorted(
        normalized,
        key=lambda record: json.dumps(record, sort_keys=True, separators=(",", ":"), default=str),
    )


def _expected_v4_publish_body_fingerprint() -> str:
    fingerprint = _function_body_fingerprint(_schema_v4_bytes().decode("utf-8"))
    if fingerprint is None:
        raise RuntimeError("schema v4 publish function body could not be fingerprinted")
    return fingerprint


def _expected_v5_function_body_fingerprints() -> dict[str, str]:
    definition = _schema_v5_bytes().decode("utf-8")
    fingerprints: dict[str, str] = {}
    for name in _V5_FUNCTIONS:
        match = re.search(
            rf"CREATE FUNCTION public\.{re.escape(name)}\(.*?\)\s+RETURNS\s+.*?\bAS\s+(\$\$)(.*?)\1;",
            definition,
            re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            raise RuntimeError(f"schema v5 function body could not be fingerprinted: {name}")
        fingerprints[name] = _raw_function_body_fingerprint(match.group(2))
    return fingerprints


def _verify_v4_catalog_contract(cur, problems: list[str]) -> None:
    cur.execute(
        "SELECT c.relname,a.attname,format_type(a.atttypid,a.atttypmod) AS data_type,a.attnotnull,"
        "a.attidentity,COALESCE(pg_get_expr(default_value.adbin,default_value.adrelid),'') AS default_expression "
        "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "JOIN pg_attribute a ON a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped "
        "LEFT JOIN pg_attrdef default_value ON default_value.adrelid=a.attrelid AND default_value.adnum=a.attnum "
        "WHERE n.nspname='public' AND c.relname=ANY(%s) ORDER BY c.relname,a.attnum",
        (list(_V4_COLUMN_CONTRACTS),),
    )
    actual_columns: dict[str, tuple[tuple[str, str, bool, str, str], ...]] = {}
    for row in cur.fetchall():
        actual_columns.setdefault(row["relname"], tuple())
        actual_columns[row["relname"]] += ((
            row["attname"], row["data_type"], bool(row["attnotnull"]), row["attidentity"], row["default_expression"],
        ),)
    for relation, expected in _V4_COLUMN_CONTRACTS.items():
        if actual_columns.get(relation) != expected:
            problems.append(
                f"v4 column contract mismatch for {relation}: "
                f"expected={expected!r}, actual={actual_columns.get(relation)!r}"
            )

    cur.execute(
        "SELECT relation.relname,con.contype,ARRAY(SELECT attribute.attname "
        "FROM unnest(con.conkey) WITH ORDINALITY key_column(attnum,position) "
        "JOIN pg_attribute attribute ON attribute.attrelid=con.conrelid AND attribute.attnum=key_column.attnum "
        "ORDER BY key_column.position) AS columns "
        "FROM pg_constraint con JOIN pg_class relation ON relation.oid=con.conrelid "
        "JOIN pg_namespace namespace ON namespace.oid=relation.relnamespace "
        "WHERE namespace.nspname='public' AND relation.relname=ANY(%s) AND con.contype IN ('p','u')",
        (list(_V4_RELATIONS),),
    )
    actual_keys = {
        (row["relname"], row["contype"], tuple(row["columns"]))
        for row in cur.fetchall()
    }
    if actual_keys != _V4_KEY_CONTRACTS:
        problems.append(
            "v4 primary/unique constraint contract mismatch: "
            f"expected={sorted(_V4_KEY_CONTRACTS)!r}, actual={sorted(actual_keys)!r}"
        )

    cur.execute(
        "SELECT relation.relname AS relation_name,target_namespace.nspname AS target_schema,target.relname AS target_name,"
        "ARRAY(SELECT attribute.attname FROM unnest(con.conkey) WITH ORDINALITY key_column(attnum,position) "
        "JOIN pg_attribute attribute ON attribute.attrelid=con.conrelid AND attribute.attnum=key_column.attnum "
        "ORDER BY key_column.position) AS local_columns,"
        "ARRAY(SELECT attribute.attname FROM unnest(con.confkey) WITH ORDINALITY key_column(attnum,position) "
        "JOIN pg_attribute attribute ON attribute.attrelid=con.confrelid AND attribute.attnum=key_column.attnum "
        "ORDER BY key_column.position) AS target_columns "
        "FROM pg_constraint con JOIN pg_class relation ON relation.oid=con.conrelid "
        "JOIN pg_namespace namespace ON namespace.oid=relation.relnamespace "
        "JOIN pg_class target ON target.oid=con.confrelid "
        "JOIN pg_namespace target_namespace ON target_namespace.oid=target.relnamespace "
        "WHERE namespace.nspname='public' AND relation.relname=ANY(%s) AND con.contype='f'",
        (list(_V4_RELATIONS),),
    )
    actual_foreign_keys = {
        (
            row["relation_name"],
            tuple(row["local_columns"]),
            f"{row['target_schema']}.{row['target_name']}",
            tuple(row["target_columns"]),
        )
        for row in cur.fetchall()
    }
    expected_foreign_keys = {
        (relation, local_columns, f"public.{target_name}", target_columns)
        for relation, local_columns, target_name, target_columns in _V4_FOREIGN_KEY_CONTRACTS
    }
    if actual_foreign_keys != expected_foreign_keys:
        problems.append(
            "v4 foreign-key lineage contract mismatch: "
            f"expected={sorted(expected_foreign_keys)!r}, actual={sorted(actual_foreign_keys)!r}"
        )

    cur.execute(
        "SELECT pg_get_constraintdef(con.oid,true) AS definition FROM pg_constraint con "
        "WHERE con.conrelid='public.brain_authority_scope_state'::regclass AND con.contype='c'"
    )
    scope_checks = tuple(_compact_sql(row["definition"]) for row in cur.fetchall())
    for fragments in _V4_SCOPE_CHECK_FRAGMENTS:
        normalized = tuple(_compact_sql(fragment) for fragment in fragments)
        if not any(all(fragment in definition for fragment in normalized) for definition in scope_checks):
            problems.append(
                "v4 authority scope constraint mismatch: required=" + ",".join(fragments)
            )

    cur.execute(
        "SELECT relation.relname,t.tgname,t.tgenabled,t.tgtype,function.proname FROM pg_trigger t "
        "JOIN pg_class relation ON relation.oid=t.tgrelid JOIN pg_namespace namespace ON namespace.oid=relation.relnamespace "
        "JOIN pg_proc function ON function.oid=t.tgfoid "
        "WHERE NOT t.tgisinternal AND namespace.nspname='public' AND relation.relname=ANY(%s)",
        (list(_V4_RELATIONS),),
    )
    actual_triggers = {
        (row["relname"], row["tgname"], row["tgenabled"], row["tgtype"], row["proname"])
        for row in cur.fetchall()
    }
    expected_triggers = {
        (relation, f"{relation}_immutable", "O", 27, "brain_reject_mutation")
        for relation in _V4_RELATIONS
    }
    if actual_triggers != expected_triggers:
        problems.append(
            "v4 immutable trigger contract mismatch: "
            f"expected={sorted(expected_triggers)!r}, actual={sorted(actual_triggers)!r}"
        )


def _verify_v4_publish_function(cur, problems: list[str]) -> None:
    cur.execute(
        "SELECT p.oid,owner.rolname AS owner_name,p.prokind,p.prosecdef,p.proconfig,p.proargnames,"
        "pg_get_function_identity_arguments(p.oid) AS arguments,pg_get_function_result(p.oid) AS result,"
        "language.lanname,p.prosrc AS function_body "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace JOIN pg_roles owner ON owner.oid=p.proowner "
        "JOIN pg_language language ON language.oid=p.prolang "
        "WHERE n.nspname='public' AND p.oid=to_regprocedure(%s)",
        (_V4_PUBLISH_SIGNATURE,),
    )
    function = cur.fetchone()
    if function is None:
        problems.append("v4 candidate publish procedure missing or identity signature mismatch")
        return
    if function["owner_name"] != _MIGRATION_ROLE:
        problems.append("v4 candidate publish procedure ownership mismatch")
    if function["prokind"] != "f" or function["arguments"] != _V4_PUBLISH_ARGUMENTS or function["result"] != "text":
        problems.append("v4 candidate publish procedure identity signature mismatch")
    if tuple(function["proargnames"] or ()) != _V4_PUBLISH_ARGUMENT_NAMES:
        problems.append("v4 candidate publish procedure argument-name signature mismatch")
    if function["lanname"] != "plpgsql" or not function["prosecdef"]:
        problems.append("v4 candidate publish procedure language/security-definer mismatch")
    if tuple(function["proconfig"] or ()) != ("search_path=pg_catalog, public",):
        problems.append("v4 candidate publish procedure search_path mismatch")
    if _raw_function_body_fingerprint(function["function_body"]) != _expected_v4_publish_body_fingerprint():
        problems.append("v4 candidate publish procedure body fingerprint mismatch")


def _verify_v4_effective_candidate_authority(cur, problems: list[str]) -> None:
    for role_name, expected_tables, expected_function in (
        (_CANDIDATE_READER_ROLE, {("public", relation, "SELECT") for relation in _V4_READ_RELATIONS}, None),
        (_CANDIDATE_WRITER_ROLE, set(), _V4_PUBLISH_SIGNATURE),
    ):
        cur.execute(
            "SELECT n.nspname,c.relname,privilege.privilege_type FROM pg_class c "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "CROSS JOIN unnest(ARRAY['SELECT','INSERT','UPDATE','DELETE','TRUNCATE','REFERENCES','TRIGGER']) "
            "AS privilege(privilege_type) WHERE n.nspname<>'information_schema' AND n.nspname!~'^pg_' "
            "AND c.relkind IN ('r','p','v','m','f') "
            "AND has_table_privilege(%s,c.oid,privilege.privilege_type)",
            (role_name,),
        )
        actual_tables = {(row["nspname"], row["relname"], row["privilege_type"]) for row in cur.fetchall()}
        if actual_tables != expected_tables:
            exposures = sorted(actual_tables - expected_tables)
            missing = sorted(expected_tables - actual_tables)
            problems.append(
                f"candidate {role_name} effective table authority exposure: "
                f"extra={exposures!r}, missing={missing!r}"
            )
        cur.execute(
            "SELECT n.nspname,p.proname,pg_get_function_identity_arguments(p.oid) AS arguments "
            "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
            "WHERE n.nspname<>'information_schema' AND n.nspname!~'^pg_' "
            "AND has_function_privilege(%s,p.oid,'EXECUTE')",
            (role_name,),
        )
        actual_functions = {
            (row["nspname"], row["proname"], row["arguments"])
            for row in cur.fetchall()
        }
        expected_functions = set()
        if expected_function is not None:
            expected_functions.add(("public", "brain_publish_v4_candidate", _V4_PUBLISH_ARGUMENTS))
        if actual_functions != expected_functions:
            exposures = sorted(actual_functions - expected_functions)
            missing = sorted(expected_functions - actual_functions)
            problems.append(
                f"candidate {role_name} effective function authority exposure: "
                f"extra={exposures!r}, missing={missing!r}"
            )


def _require_idle(conn) -> None:
    if conn.info.transaction_status != TransactionStatus.IDLE:
        raise RuntimeError("brain schema helpers require an idle connection and never commit or roll back caller work")


def _identity(cur) -> tuple[str, bool]:
    cur.execute("SELECT current_user AS name, rolsuper FROM pg_roles WHERE rolname = current_user")
    row = cur.fetchone()
    return row["name"], bool(row["rolsuper"])


def _activate_migration_identity(cur) -> str:
    cur.execute(
        "WITH RECURSIVE memberships(roleid) AS ("
        "SELECT oid FROM pg_roles WHERE rolname=session_user UNION "
        "SELECT membership.roleid FROM pg_auth_members membership "
        "JOIN memberships prior ON prior.roleid=membership.member) "
        "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname=%s) AS present, "
        "session_user AS session_name, EXISTS (SELECT 1 FROM memberships WHERE roleid="
        "(SELECT oid FROM pg_roles WHERE rolname=%s)) AS member,"
        "EXISTS (SELECT 1 FROM pg_roles WHERE oid=10 AND rolname=session_user "
        "AND rolsuper) AS bootstrap_superuser",
        (_MIGRATION_ROLE, _MIGRATION_ROLE),
    )
    row = cur.fetchone()
    if not row["present"] or not (row["member"] or row["bootstrap_superuser"]):
        raise RuntimeError(
            "brain schema migration requires explicit session_user membership in fixed role "
            f"{_MIGRATION_ROLE} or bootstrap superuser OID 10"
        )
    cur.execute(
        "SELECT COALESCE(grantee.rolname,'PUBLIC') AS grantee FROM pg_namespace n "
        "CROSS JOIN LATERAL aclexplode(n.nspacl) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE n.nspname='public' AND acl.privilege_type='CREATE' "
        "AND COALESCE(grantee.rolname,'PUBLIC') NOT IN (%s,'pg_database_owner')",
        (_MIGRATION_ROLE,),
    )
    for grant in cur.fetchall():
        grantee = sql.SQL("PUBLIC") if grant["grantee"] == "PUBLIC" else sql.Identifier(grant["grantee"])
        cur.execute(sql.SQL("REVOKE CREATE ON SCHEMA public FROM {}").format(grantee))
    cur.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(_MIGRATION_ROLE)))
    current_user, _ = _identity(cur)
    if current_user != _MIGRATION_ROLE:
        raise RuntimeError("failed to activate fixed brain migration role")
    return current_user


def _ensure_v5_graph_authority_role(cur) -> None:
    """Create or validate the fixed graph-writer capability before SET ROLE."""
    cur.execute("RESET ROLE")
    cur.execute(
        "SELECT rolcanlogin,rolsuper,rolcreatedb,rolcreaterole,rolinherit,rolreplication,rolbypassrls "
        "FROM pg_roles WHERE rolname=%s",
        (_GRAPH_AUTHORITY_ROLE,),
    )
    role = cur.fetchone()
    if role is None:
        cur.execute(
            "SELECT rolsuper OR rolcreaterole AS can_create FROM pg_roles WHERE rolname=session_user"
        )
        if not cur.fetchone()["can_create"]:
            raise RuntimeError(
                f"brain schema v5 requires {_GRAPH_AUTHORITY_ROLE} to be pre-provisioned "
                "or a migration session with CREATEROLE"
            )
        cur.execute(
            sql.SQL(
                "CREATE ROLE {} NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE "
                "NOREPLICATION NOBYPASSRLS"
            ).format(sql.Identifier(_GRAPH_AUTHORITY_ROLE))
        )
        return
    if any(role.values()):
        raise RuntimeError(f"unsafe fixed graph authority role attributes: {_GRAPH_AUTHORITY_ROLE}")


def _verifier_identity() -> str:
    return _VERIFIER_ROLE


def _apply_acl_contract(cur, migration_identity: str) -> None:
    verifier_identity = _verifier_identity()
    cur.execute("SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname=%s) AS present", (verifier_identity,))
    if not cur.fetchone()["present"]:
        raise RuntimeError(f"fixed brain verifier role does not exist: {verifier_identity}")

    cur.execute(
        "SELECT COALESCE(grantee.rolname, 'PUBLIC') AS grantee "
        "FROM pg_namespace n CROSS JOIN LATERAL aclexplode(n.nspacl) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE n.nspname='public' AND acl.privilege_type='CREATE' "
        "AND COALESCE(grantee.rolname, 'PUBLIC') NOT IN (%s, 'pg_database_owner')",
        (migration_identity,),
    )
    for row in cur.fetchall():
        grantee = sql.SQL("PUBLIC") if row["grantee"] == "PUBLIC" else sql.Identifier(row["grantee"])
        cur.execute(sql.SQL("REVOKE CREATE ON SCHEMA public FROM {}").format(grantee))

    verifier = sql.Identifier(verifier_identity)
    cur.execute(sql.SQL("REVOKE ALL PRIVILEGES ON SCHEMA public, brain_archive FROM {}").format(verifier))
    cur.execute(sql.SQL("GRANT USAGE ON SCHEMA public, brain_archive TO {}").format(verifier))
    for relation in _RELATIONS:
        cur.execute("SELECT to_regclass(%s) AS relation", (f"public.{relation}",))
        if cur.fetchone()["relation"] is None:
            continue
        cur.execute(
            sql.SQL("GRANT SELECT ON TABLE {}.{} TO {}").format(
                sql.Identifier("public"), sql.Identifier(relation), verifier
            )
        )
    cur.execute(
        sql.SQL("GRANT SELECT ON TABLE {}.{} TO {}").format(
            sql.Identifier("brain_archive"), sql.Identifier("brain_archive_manifests"), verifier
        )
    )
    for relation in ("fleet_config", "fleet_decision_policies", "apply_queue", "linkedin_queue"):
        cur.execute("SELECT to_regclass(%s) AS relation", (f"public.{relation}",))
        if cur.fetchone()["relation"] is not None:
            cur.execute(
                "SELECT has_table_privilege(%s,%s,'SELECT,INSERT,UPDATE') AS allowed",
                (migration_identity, f"public.{relation}"),
            )
            if not cur.fetchone()["allowed"]:
                raise RuntimeError(
                    f"fixed controller role {migration_identity} requires SELECT, INSERT, UPDATE on public.{relation}"
                )


def _version_rows(cur):
    cur.execute("SELECT to_regclass('public.brain_schema_versions') AS relation")
    if cur.fetchone()["relation"] is None:
        return []
    cur.execute(
        "SELECT version, migration_name, migration_checksum, applied_at, applied_by "
        "FROM public.brain_schema_versions ORDER BY version"
    )
    rows = cur.fetchall()
    if not rows:
        raise RuntimeError("brain schema version ledger exists but is empty")
    versions = [row["version"] for row in rows]
    if versions not in (
        [1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8],
    ):
        raise RuntimeError(
            "unsupported or non-contiguous brain schema version ledger: "
            + ", ".join(str(version) for version in versions)
        )
    if len(rows) >= 4 and (
        rows[3]["migration_name"] != _MIGRATION_V4_NAME
        or rows[3]["migration_checksum"] != _EXPECTED_V4_CHECKSUM
        or rows[3]["applied_by"] != _MIGRATION_ROLE
    ):
        raise RuntimeError("unsupported or non-contiguous brain schema version ledger: invalid version 4 contract")
    if len(rows) >= 5 and (
        rows[4]["migration_name"] != _MIGRATION_V5_NAME
        or rows[4]["migration_checksum"] != _EXPECTED_V5_CHECKSUM
        or rows[4]["applied_by"] != _MIGRATION_ROLE
    ):
        raise RuntimeError("unsupported or non-contiguous brain schema version ledger: invalid version 5 contract")
    if len(rows) >= 6 and (
        rows[5]["migration_name"] != _MIGRATION_V6_NAME
        or rows[5]["migration_checksum"] not in {_EXPECTED_V6_CHECKSUM, _CURRENT_V6_CHECKSUM}
        or rows[5]["applied_by"] != _MIGRATION_ROLE
    ):
        raise RuntimeError("unsupported or non-contiguous brain schema version ledger: invalid version 6 contract")
    if len(rows) >= 7 and (
        rows[6]["migration_name"] != _MIGRATION_V7_NAME
        or rows[6]["migration_checksum"] != _EXPECTED_V7_CHECKSUM
        or rows[6]["applied_by"] != _MIGRATION_ROLE
    ):
        raise RuntimeError("unsupported or non-contiguous brain schema version ledger: invalid version 7 contract")
    if len(rows) == 8 and (
        rows[7]["migration_name"] != _MIGRATION_V8_NAME
        or rows[7]["migration_checksum"] != _EXPECTED_V8_CHECKSUM
        or rows[7]["applied_by"] != _MIGRATION_ROLE
    ):
        raise RuntimeError("unsupported or non-contiguous brain schema version ledger: invalid version 8 contract")
    return rows


def _version_row(cur):
    rows = _version_rows(cur)
    return rows[0] if rows else None


def _assert_existing_ownership(cur, migration_identity: str) -> None:
    cur.execute(
        "SELECT n.nspname, c.relname, owner.rolname AS owner_name "
        "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "JOIN pg_roles owner ON owner.oid = c.relowner "
        "WHERE (n.nspname = 'public' AND left(c.relname, 6) = 'brain_') "
        "OR n.nspname = 'brain_archive'"
    )
    owned_objects = [(f"{row['nspname']}.{row['relname']}", row["owner_name"]) for row in cur.fetchall()]
    cur.execute(
        "SELECT n.nspname, p.proname, owner.rolname AS owner_name "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
        "JOIN pg_roles owner ON owner.oid = p.proowner "
        "WHERE n.nspname = 'public' AND left(p.proname, 6) = 'brain_'"
    )
    owned_objects.extend((f"{row['nspname']}.{row['proname']}()", row["owner_name"]) for row in cur.fetchall())
    cur.execute(
        "SELECT n.nspname, owner.rolname AS owner_name FROM pg_namespace n "
        "JOIN pg_roles owner ON owner.oid = n.nspowner WHERE n.nspname = 'brain_archive'"
    )
    owned_objects.extend((row["nspname"], row["owner_name"]) for row in cur.fetchall())
    invalid = []
    for object_name, owner_name in owned_objects:
        if owner_name != migration_identity:
            invalid.append(f"{object_name} owned by {owner_name}")
    if invalid:
        raise RuntimeError(
            "existing brain objects are not owned by the migration identity or one of its roles: "
            + ", ".join(sorted(invalid))
        )


def require_pg18_authority_catalog(cur) -> None:
    """Reject unsupported authority databases before any transaction-visible mutation."""
    cur.execute(
        "SELECT current_setting('server_version_num')::integer AS server_version_num"
    )
    version = cur.fetchone()["server_version_num"]
    if version // 10000 != 18:
        raise RuntimeError(
            "PostgreSQL 18 authority catalog contract required: "
            f"server_version_num={version}"
        )
    for relation, expected in _PG18_CATALOG_SHAPES.items():
        cur.execute(
            "SELECT attname,format_type(atttypid,atttypmod) AS data_type "
            "FROM pg_attribute WHERE attrelid=%s::regclass AND attnum>0 "
            "AND NOT attisdropped ORDER BY attnum",
            (relation,),
        )
        actual = tuple((row["attname"], row["data_type"]) for row in cur.fetchall())
        if actual != expected:
            raise RuntimeError(
                f"PostgreSQL 18 authority catalog shape mismatch for {relation}: "
                f"expected={expected!r}, actual={actual!r}"
            )


def _pg18_catalog_hash(name: str) -> str:
    value = _PG_CATALOG_HASHES[18][name]
    if value == _UNPINNED_PG18_CATALOG_HASH:
        raise RuntimeError(f"PostgreSQL 18 catalog hash pin is not installed: {name}")
    return value


def _acquire_xact_lock(cur, timeout_seconds: float) -> None:
    deadline = time.monotonic() + _bounded_timeout(timeout_seconds)
    while True:
        cur.execute(
            "SELECT pg_try_advisory_xact_lock(hashtext(%s)) AS acquired",
            (_SCHEMA_LOCK_KEY,),
        )
        if cur.fetchone()["acquired"]:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for the brain schema v1 migration lock; another owner may be stalled while migrating"
            )
        time.sleep(0.05)


def _catalog_constraint_records(
    cur,
    relation_names: list[str],
    *,
    include_archive: bool,
):
    cur.execute(
        "SELECT constraint_namespace.nspname AS constraint_schema,"
        "format('%%I.%%I',relation_namespace.nspname,relation.relname) AS relation_identity,"
        "con.conname,con.contype,con.condeferrable,con.condeferred,con.conenforced,"
        "con.convalidated,"
        "CASE WHEN con.contypid=0 THEN NULL ELSE "
        "format('%%I.%%I',constraint_type_namespace.nspname,constraint_type.typname) "
        "END AS type_identity,"
        "CASE WHEN con.conindid=0 THEN NULL ELSE "
        "format('%%I.%%I',backing_index_namespace.nspname,backing_index.relname) "
        "END AS index_identity,"
        "CASE WHEN con.conparentid=0 THEN NULL ELSE "
        "format('%%I.%%I.%%I',parent_namespace.nspname,parent_relation.relname,parent_constraint.conname) "
        "END AS parent_constraint_identity,"
        "CASE WHEN con.confrelid=0 THEN NULL ELSE "
        "format('%%I.%%I',referenced_namespace.nspname,referenced_relation.relname) "
        "END AS referenced_relation_identity,"
        "con.confupdtype,con.confdeltype,con.confmatchtype,con.conislocal,con.coninhcount,"
        "con.connoinherit,con.conperiod,con.conkey,con.confkey,"
        "ARRAY(SELECT format('%%I.%%I(%%s,%%s)',operator_namespace.nspname,operator.oprname,"
        "CASE WHEN operator.oprleft=0 THEN 'NONE' ELSE "
        "format('%%I.%%I',left_type_namespace.nspname,left_type.typname) END,"
        "CASE WHEN operator.oprright=0 THEN 'NONE' ELSE "
        "format('%%I.%%I',right_type_namespace.nspname,right_type.typname) END) FROM "
        "unnest(COALESCE(con.conpfeqop,ARRAY[]::oid[])) WITH ORDINALITY item(operator_oid,position) "
        "JOIN pg_operator operator ON operator.oid=item.operator_oid "
        "JOIN pg_namespace operator_namespace ON operator_namespace.oid=operator.oprnamespace "
        "LEFT JOIN pg_type left_type ON left_type.oid=operator.oprleft "
        "LEFT JOIN pg_namespace left_type_namespace ON left_type_namespace.oid=left_type.typnamespace "
        "LEFT JOIN pg_type right_type ON right_type.oid=operator.oprright "
        "LEFT JOIN pg_namespace right_type_namespace ON right_type_namespace.oid=right_type.typnamespace "
        "ORDER BY position) AS parent_fk_equality_operators,"
        "ARRAY(SELECT format('%%I.%%I(%%s,%%s)',operator_namespace.nspname,operator.oprname,"
        "CASE WHEN operator.oprleft=0 THEN 'NONE' ELSE "
        "format('%%I.%%I',left_type_namespace.nspname,left_type.typname) END,"
        "CASE WHEN operator.oprright=0 THEN 'NONE' ELSE "
        "format('%%I.%%I',right_type_namespace.nspname,right_type.typname) END) FROM "
        "unnest(COALESCE(con.conppeqop,ARRAY[]::oid[])) WITH ORDINALITY item(operator_oid,position) "
        "JOIN pg_operator operator ON operator.oid=item.operator_oid "
        "JOIN pg_namespace operator_namespace ON operator_namespace.oid=operator.oprnamespace "
        "LEFT JOIN pg_type left_type ON left_type.oid=operator.oprleft "
        "LEFT JOIN pg_namespace left_type_namespace ON left_type_namespace.oid=left_type.typnamespace "
        "LEFT JOIN pg_type right_type ON right_type.oid=operator.oprright "
        "LEFT JOIN pg_namespace right_type_namespace ON right_type_namespace.oid=right_type.typnamespace "
        "ORDER BY position) AS parent_pk_equality_operators,"
        "ARRAY(SELECT format('%%I.%%I(%%s,%%s)',operator_namespace.nspname,operator.oprname,"
        "CASE WHEN operator.oprleft=0 THEN 'NONE' ELSE "
        "format('%%I.%%I',left_type_namespace.nspname,left_type.typname) END,"
        "CASE WHEN operator.oprright=0 THEN 'NONE' ELSE "
        "format('%%I.%%I',right_type_namespace.nspname,right_type.typname) END) FROM "
        "unnest(COALESCE(con.conffeqop,ARRAY[]::oid[])) WITH ORDINALITY item(operator_oid,position) "
        "JOIN pg_operator operator ON operator.oid=item.operator_oid "
        "JOIN pg_namespace operator_namespace ON operator_namespace.oid=operator.oprnamespace "
        "LEFT JOIN pg_type left_type ON left_type.oid=operator.oprleft "
        "LEFT JOIN pg_namespace left_type_namespace ON left_type_namespace.oid=left_type.typnamespace "
        "LEFT JOIN pg_type right_type ON right_type.oid=operator.oprright "
        "LEFT JOIN pg_namespace right_type_namespace ON right_type_namespace.oid=right_type.typnamespace "
        "ORDER BY position) AS foreign_fk_equality_operators,"
        "con.confdelsetcols,"
        "ARRAY(SELECT format('%%I.%%I(%%s,%%s)',operator_namespace.nspname,operator.oprname,"
        "CASE WHEN operator.oprleft=0 THEN 'NONE' ELSE "
        "format('%%I.%%I',left_type_namespace.nspname,left_type.typname) END,"
        "CASE WHEN operator.oprright=0 THEN 'NONE' ELSE "
        "format('%%I.%%I',right_type_namespace.nspname,right_type.typname) END) FROM "
        "unnest(COALESCE(con.conexclop,ARRAY[]::oid[])) WITH ORDINALITY item(operator_oid,position) "
        "JOIN pg_operator operator ON operator.oid=item.operator_oid "
        "JOIN pg_namespace operator_namespace ON operator_namespace.oid=operator.oprnamespace "
        "LEFT JOIN pg_type left_type ON left_type.oid=operator.oprleft "
        "LEFT JOIN pg_namespace left_type_namespace ON left_type_namespace.oid=left_type.typnamespace "
        "LEFT JOIN pg_type right_type ON right_type.oid=operator.oprright "
        "LEFT JOIN pg_namespace right_type_namespace ON right_type_namespace.oid=right_type.typnamespace "
        "ORDER BY position) AS exclusion_operators,"
        "pg_get_constraintdef(con.oid,false) AS definition "
        "FROM pg_constraint con "
        "JOIN pg_namespace constraint_namespace ON constraint_namespace.oid=con.connamespace "
        "JOIN pg_class relation ON relation.oid=con.conrelid "
        "JOIN pg_namespace relation_namespace ON relation_namespace.oid=relation.relnamespace "
        "LEFT JOIN pg_type constraint_type ON constraint_type.oid=con.contypid "
        "LEFT JOIN pg_namespace constraint_type_namespace "
        "ON constraint_type_namespace.oid=constraint_type.typnamespace "
        "LEFT JOIN pg_class backing_index ON backing_index.oid=con.conindid "
        "LEFT JOIN pg_namespace backing_index_namespace "
        "ON backing_index_namespace.oid=backing_index.relnamespace "
        "LEFT JOIN pg_constraint parent_constraint ON parent_constraint.oid=con.conparentid "
        "LEFT JOIN pg_class parent_relation ON parent_relation.oid=parent_constraint.conrelid "
        "LEFT JOIN pg_namespace parent_namespace ON parent_namespace.oid=parent_relation.relnamespace "
        "LEFT JOIN pg_class referenced_relation ON referenced_relation.oid=con.confrelid "
        "LEFT JOIN pg_namespace referenced_namespace "
        "ON referenced_namespace.oid=referenced_relation.relnamespace "
        "WHERE (relation_namespace.nspname='public' AND relation.relname=ANY(%s)) "
        "OR (%s AND relation_namespace.nspname='brain_archive' "
        "AND relation.relname='brain_archive_manifests') "
        "ORDER BY relation_namespace.nspname,relation.relname,con.conname",
        (relation_names, include_archive),
    )
    return cur.fetchall()


def _catalog_contract_hash(
    cur,
    *,
    include_v5: bool = False,
    include_v7: bool = False,
    include_v8: bool = False,
) -> str:
    # pg_get_* output is search_path-sensitive; pin it so the immutable catalog
    # fingerprint reflects catalog state rather than a prior helper's session setting.
    cur.execute("SELECT current_setting('search_path') AS current_search_path")
    prior_search_path = cur.fetchone()["current_search_path"]
    cur.execute("SET LOCAL search_path=pg_catalog, public")
    relations = (
        list(_RELATIONS)
        + (sorted(_V7_RELATIONS) if include_v7 else [])
        + (sorted(_V8_RELATIONS) if include_v8 else [])
    )
    cur.execute(
        "SELECT n.nspname AS schema_name,c.relname,a.attnum,a.attname,"
        "format_type(a.atttypid,a.atttypmod) AS data_type,a.attnotnull,a.attidentity,a.attgenerated,"
        "pg_get_expr(d.adbin,d.adrelid,false) AS default_expr "
        "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "JOIN pg_attribute a ON a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped "
        "LEFT JOIN pg_attrdef d ON d.adrelid=c.oid AND d.adnum=a.attnum "
        "WHERE (n.nspname='public' AND c.relname=ANY(%s)) "
        "OR (n.nspname='brain_archive' AND c.relname='brain_archive_manifests') "
        "ORDER BY n.nspname,c.relname,a.attnum",
        (relations,),
    )
    columns = cur.fetchall()
    constraints = _catalog_constraint_records(cur, relations, include_archive=True)
    cur.execute(
        "SELECT n.nspname AS schema_name,t.relname AS relation_name,i.relname AS index_name,"
        "x.indisunique,x.indisprimary,x.indisvalid,x.indisready,"
        "pg_get_indexdef(i.oid,0,false) AS definition,pg_get_expr(x.indpred,x.indrelid,false) AS predicate,"
        "ARRAY(SELECT opc.opcname FROM unnest(x.indclass::oid[]) WITH ORDINALITY value(opcoid,ordinal) "
        "JOIN pg_opclass opc ON opc.oid=value.opcoid ORDER BY value.ordinal) AS opclasses "
        "FROM pg_index x JOIN pg_class i ON i.oid=x.indexrelid JOIN pg_class t ON t.oid=x.indrelid "
        "JOIN pg_namespace n ON n.oid=t.relnamespace "
        "WHERE (n.nspname='public' AND t.relname=ANY(%s)) "
        "OR (n.nspname='brain_archive' AND t.relname='brain_archive_manifests') "
        "ORDER BY n.nspname,t.relname,i.relname",
        (relations,),
    )
    indexes = cur.fetchall()
    cur.execute(
        "SELECT n.nspname AS schema_name,c.relname,t.tgname,t.tgenabled,"
        "pg_get_triggerdef(t.oid,false) AS definition,p.proname AS function_name,"
        "pg_get_function_identity_arguments(p.oid) AS function_arguments "
        "FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid JOIN pg_namespace n ON n.oid=c.relnamespace "
        "JOIN pg_proc p ON p.oid=t.tgfoid WHERE NOT t.tgisinternal AND "
        "((n.nspname='public' AND c.relname=ANY(%s)) "
        "OR (n.nspname='brain_archive' AND c.relname='brain_archive_manifests')) "
        "ORDER BY n.nspname,c.relname,t.tgname",
        (relations,),
    )
    triggers = cur.fetchall()
    cur.execute(
        "SELECT p.proname,pg_get_function_identity_arguments(p.oid) AS arguments,"
        "pg_get_function_result(p.oid) AS result,l.lanname AS language,p.prosecdef,p.provolatile,"
        "p.proparallel,p.proisstrict,COALESCE(p.proconfig,ARRAY[]::text[]) AS config,"
        "regexp_replace(p.prosrc,'\\s+',' ','g') AS normalized_body "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace JOIN pg_language l ON l.oid=p.prolang "
        "WHERE n.nspname='public' AND left(p.proname,6)='brain_' "
        "AND p.proname <> 'brain_publish_v4_candidate' AND NOT (p.proname=ANY(%s)) "
        "ORDER BY p.proname,pg_get_function_identity_arguments(p.oid)",
        (list(_V5_FUNCTIONS) if include_v5 else [],),
    )
    functions = cur.fetchall()
    payload = {
        "columns": _stable_catalog_records(columns),
        "constraints": _stable_catalog_records(constraints),
        "indexes": _stable_catalog_records(indexes),
        "triggers": _stable_catalog_records(triggers),
        "functions": _stable_catalog_records(functions),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    cur.execute("SELECT set_config('search_path', %s, true)", (prior_search_path,))
    return hashlib.sha256(encoded).hexdigest()


def _v5_catalog_contract_hash(cur) -> str:
    """Fingerprint inherited V4 authority plus every surface introduced or changed by V5."""
    cur.execute("SELECT current_setting('search_path') AS current_search_path")
    prior_search_path = cur.fetchone()["current_search_path"]
    cur.execute("SET LOCAL search_path=pg_catalog, public")
    relation_names = sorted(_V5_CATALOG_RELATIONS | _V5_SEQUENCES)
    try:
        cur.execute(
            "SELECT c.relname,c.relkind,c.relpersistence,c.relreplident,c.relrowsecurity,"
            "c.relforcerowsecurity,c.relispopulated,c.reloptions,owner.rolname AS owner_name "
            "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "JOIN pg_roles owner ON owner.oid=c.relowner "
            "WHERE n.nspname='public' AND c.relname=ANY(%s) ORDER BY c.relname",
            (relation_names,),
        )
        relations = cur.fetchall()
        cur.execute(
            "SELECT c.relname,a.attnum,a.attname,format_type(a.atttypid,a.atttypmod) AS data_type,"
            "a.attnotnull,a.attidentity,a.attgenerated,a.attstorage,a.attcompression,"
            "coll.collname AS collation_name,"
            "pg_get_expr(default_value.adbin,default_value.adrelid,false) AS default_expression "
            "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "JOIN pg_attribute a ON a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped "
            "LEFT JOIN pg_attrdef default_value ON default_value.adrelid=a.attrelid "
            "AND default_value.adnum=a.attnum "
            "LEFT JOIN pg_collation coll ON coll.oid=a.attcollation "
            "WHERE n.nspname='public' AND c.relname=ANY(%s) ORDER BY c.relname,a.attnum",
            (sorted(_V5_CATALOG_RELATIONS),),
        )
        columns = cur.fetchall()
        constraints = _catalog_constraint_records(
            cur,
            sorted(_V5_CATALOG_RELATIONS),
            include_archive=False,
        )
        cur.execute(
            "SELECT relation.relname,index_class.relname AS index_name,index_data.indisunique,"
            "index_data.indisprimary,index_data.indisexclusion,index_data.indimmediate,"
            "index_data.indisvalid,index_data.indisready,index_data.indislive,"
            "pg_get_indexdef(index_class.oid,0,false) AS definition,"
            "pg_get_expr(index_data.indpred,index_data.indrelid,false) AS predicate "
            "FROM pg_index index_data JOIN pg_class index_class ON index_class.oid=index_data.indexrelid "
            "JOIN pg_class relation ON relation.oid=index_data.indrelid "
            "JOIN pg_namespace namespace ON namespace.oid=relation.relnamespace "
            "WHERE namespace.nspname='public' AND relation.relname=ANY(%s) "
            "ORDER BY relation.relname,index_class.relname",
            (sorted(_V5_CATALOG_RELATIONS),),
        )
        indexes = cur.fetchall()
        cur.execute(
            "SELECT relation.relname,trigger.tgname,trigger.tgenabled,trigger.tgtype,"
            "function.proname AS function_name,pg_get_triggerdef(trigger.oid,false) AS definition "
            "FROM pg_trigger trigger JOIN pg_class relation ON relation.oid=trigger.tgrelid "
            "JOIN pg_namespace namespace ON namespace.oid=relation.relnamespace "
            "JOIN pg_proc function ON function.oid=trigger.tgfoid "
            "WHERE NOT trigger.tgisinternal AND namespace.nspname='public' "
            "AND relation.relname=ANY(%s) ORDER BY relation.relname,trigger.tgname",
            (sorted(_V5_CATALOG_RELATIONS),),
        )
        triggers = cur.fetchall()
        cur.execute(
            "SELECT c.relname,pg_get_viewdef(c.oid,false) AS definition "
            "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname='public' AND c.relname='brain_factual_contradiction_state'"
        )
        views = cur.fetchall()
        cur.execute(
            "SELECT c.relname,format_type(sequence.seqtypid,NULL) AS data_type,sequence.seqstart,"
            "sequence.seqincrement,sequence.seqmax,sequence.seqmin,sequence.seqcache,sequence.seqcycle "
            "FROM pg_sequence sequence JOIN pg_class c ON c.oid=sequence.seqrelid "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname='public' AND c.relname=ANY(%s) ORDER BY c.relname",
            (sorted(_V5_SEQUENCES),),
        )
        sequences = cur.fetchall()
        cur.execute(
            "SELECT p.proname,pg_get_function_identity_arguments(p.oid) AS arguments,"
            "pg_get_function_result(p.oid) AS result,p.prokind,p.prosecdef,p.proleakproof,"
            "p.proisstrict,p.provolatile,p.proparallel,p.proconfig,p.proargnames,"
            "language.lanname AS language_name,owner.rolname AS owner_name,"
            "p.prosrc AS function_body "
            "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
            "JOIN pg_language language ON language.oid=p.prolang "
            "JOIN pg_roles owner ON owner.oid=p.proowner "
            "WHERE n.nspname='public' AND p.proname=ANY(%s) "
            "ORDER BY p.proname,pg_get_function_identity_arguments(p.oid)",
            (sorted(_V5_CATALOG_FUNCTIONS),),
        )
        functions = cur.fetchall()
        cur.execute(
            "SELECT c.relname,COALESCE(grantee.rolname,'PUBLIC') AS grantee,"
            "grantor.rolname AS grantor,acl.privilege_type,acl.is_grantable "
            "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "CROSS JOIN LATERAL aclexplode(COALESCE(c.relacl,CASE WHEN c.relkind='S' "
            "THEN acldefault('S',c.relowner) ELSE acldefault('r',c.relowner) END)) acl "
            "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
            "JOIN pg_roles grantor ON grantor.oid=acl.grantor "
            "WHERE n.nspname='public' AND c.relname=ANY(%s) AND acl.grantee<>c.relowner "
            "ORDER BY c.relname,grantee,acl.privilege_type",
            (relation_names,),
        )
        relation_acls = cur.fetchall()
        cur.execute(
            "SELECT p.proname,pg_get_function_identity_arguments(p.oid) AS arguments,"
            "COALESCE(grantee.rolname,'PUBLIC') AS grantee,grantor.rolname AS grantor,"
            "acl.privilege_type,acl.is_grantable "
            "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
            "CROSS JOIN LATERAL aclexplode(COALESCE(p.proacl,acldefault('f',p.proowner))) acl "
            "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
            "JOIN pg_roles grantor ON grantor.oid=acl.grantor "
            "WHERE n.nspname='public' AND p.proname=ANY(%s) AND acl.grantee<>p.proowner "
            "ORDER BY p.proname,arguments,grantee,acl.privilege_type",
            (sorted(_V5_CATALOG_FUNCTIONS),),
        )
        function_acls = cur.fetchall()
        payload = {
            "columns": _stable_catalog_records(columns),
            "constraints": _stable_catalog_records(constraints),
            "function_acls": _stable_catalog_records(function_acls),
            "functions": _stable_catalog_records(functions),
            "indexes": _stable_catalog_records(indexes),
            "relation_acls": _stable_catalog_records(relation_acls),
            "relations": _stable_catalog_records(relations),
            "sequences": _stable_catalog_records(sequences),
            "triggers": _stable_catalog_records(triggers),
            "views": _stable_catalog_records(views),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(encoded).hexdigest()
    finally:
        if cur.connection.info.transaction_status != TransactionStatus.INERROR:
            cur.execute("SELECT set_config('search_path', %s, true)", (prior_search_path,))


def _verify_contract(cur) -> None:
    require_pg18_authority_catalog(cur)
    checksum = _schema_checksum()
    versions = _version_rows(cur)
    version = versions[0] if versions else None
    problems: list[str] = []
    if version is None:
        problems.append("missing schema version 1")
        migration_identity = None
    else:
        migration_identity = _MIGRATION_ROLE
        if version["migration_name"] != _MIGRATION_NAME:
            problems.append("migration name mismatch")
        if version["migration_checksum"] not in {checksum, *_LEGACY_V1_CHECKSUMS}:
            problems.append("migration checksum mismatch")
        if version["applied_by"] != _MIGRATION_ROLE:
            problems.append(f"migration ledger owner mismatch: expected {_MIGRATION_ROLE}, got {version['applied_by']}")
    version_v2 = versions[1] if len(versions) > 1 else None
    if version_v2 is None:
        problems.append("missing schema version 2 lifecycle principal contract")
    else:
        if version_v2["migration_name"] != _MIGRATION_V2_NAME:
            problems.append("migration v2 name mismatch")
        if version_v2["migration_checksum"] != _schema_v2_checksum():
            problems.append("migration v2 checksum mismatch")
        if version_v2["applied_by"] != _MIGRATION_ROLE:
            problems.append(
                f"migration v2 ledger owner mismatch: expected {_MIGRATION_ROLE}, "
                f"got {version_v2['applied_by']}"
            )
    version_v3 = versions[2] if len(versions) > 2 else None
    if version_v3 is None:
        problems.append("missing schema version 3 lane canary pin contract")
    else:
        if version_v3["migration_name"] != _MIGRATION_V3_NAME:
            problems.append("migration v3 name mismatch")
        if version_v3["migration_checksum"] != _schema_v3_checksum():
            problems.append("migration v3 checksum mismatch")
        if version_v3["applied_by"] != _MIGRATION_ROLE:
            problems.append(
                f"migration v3 ledger owner mismatch: expected {_MIGRATION_ROLE}, "
                f"got {version_v3['applied_by']}"
            )

    cur.execute(
        "SELECT c.relname, c.relkind FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' AND c.relname = ANY(%s)",
        (list(_RELATIONS),),
    )
    relations = {row["relname"]: row["relkind"] for row in cur.fetchall()}
    for name, kind in _RELATIONS.items():
        if relations.get(name) != kind:
            problems.append(f"relation {name} expected kind {kind}, got {relations.get(name)!r}")

    cur.execute(
        "SELECT c.relname, a.attname, format_type(a.atttypid, a.atttypmod) AS data_type, "
        "a.attnotnull FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped "
        "WHERE n.nspname = 'public' AND c.relname = ANY(%s)",
        (list(_RELATIONS),),
    )
    columns = {(row["relname"], row["attname"]): (row["data_type"], bool(row["attnotnull"])) for row in cur.fetchall()}
    for key, signature in _KEY_COLUMNS.items():
        if columns.get(key) != signature:
            problems.append(f"column {key[0]}.{key[1]} expected {signature}, got {columns.get(key)!r}")

    cur.execute(
        "SELECT conname, convalidated, pg_get_constraintdef(con.oid, true) AS definition "
        "FROM pg_constraint con JOIN pg_namespace n ON n.oid = con.connamespace "
        "WHERE n.nspname IN ('public', 'brain_archive') AND conname = ANY(%s)",
        (list(_CONSTRAINTS),),
    )
    constraints = {row["conname"]: row for row in cur.fetchall()}
    for name, required_fragments in _CONSTRAINTS.items():
        constraint = constraints.get(name)
        if constraint is None:
            problems.append(f"missing constraint: {name}")
        elif not constraint["convalidated"]:
            problems.append(f"constraint {name} is not validated")
        elif any(fragment not in constraint["definition"] for fragment in required_fragments):
            problems.append(f"constraint {name} definition mismatch: {constraint['definition']}")

    cur.execute(
        "SELECT c.relname AS index_name, i.indisunique, i.indisvalid, i.indisready, "
        "pg_get_expr(i.indpred, i.indrelid) AS predicate "
        "FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid "
        "JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname='public' AND c.relname=ANY(%s)",
        (list(_INDEXES),),
    )
    indexes = {row["index_name"]: row for row in cur.fetchall()}
    for name, (unique, predicate) in _INDEXES.items():
        index = indexes.get(name)
        if index is None:
            problems.append(f"missing index: {name}")
        elif not index["indisvalid"] or not index["indisready"]:
            problems.append(f"index {name} is not valid and ready")
        elif bool(index["indisunique"]) != unique or index["predicate"] != predicate:
            problems.append(
                f"index {name} signature mismatch: unique={index['indisunique']} predicate={index['predicate']!r}"
            )

    cur.execute(
        "SELECT c.relname, t.tgname, t.tgenabled, t.tgtype, p.proname, pg_get_triggerdef(t.oid, true) AS definition "
        "FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "JOIN pg_proc p ON p.oid=t.tgfoid "
        "WHERE NOT t.tgisinternal AND n.nspname IN ('public', 'brain_archive')"
    )
    triggers = {(row["relname"], row["tgname"]): row for row in cur.fetchall()}
    for table in _IMMUTABLE_TABLES:
        for suffix, trigger_type in (("append_only", 27), ("append_only_truncate", 34)):
            trigger = triggers.get((table, f"{table}_{suffix}"))
            if trigger is None:
                problems.append(f"missing {suffix} trigger on {table}")
            elif trigger["tgenabled"] != "O" or trigger["proname"] != "brain_reject_mutation":
                problems.append(f"trigger {table}_{suffix} binding or enabled state mismatch")
            elif trigger["tgtype"] != trigger_type:
                problems.append(f"trigger {table}_{suffix} event definition mismatch")
    for suffix, trigger_type in (("append_only", 27), ("append_only_truncate", 34)):
        trigger = triggers.get(("brain_archive_manifests", f"brain_archive_manifests_{suffix}"))
        if trigger is None or trigger["tgenabled"] != "O" or trigger["proname"] != "brain_reject_mutation":
            problems.append(f"archive manifest {suffix} trigger binding or enabled state mismatch")
        elif trigger["tgtype"] != trigger_type:
            problems.append(f"archive manifest {suffix} trigger event definition mismatch")
    for table in (
        "brain_job_observations",
        "brain_label_events",
        "brain_pairwise_events",
        "brain_email_events",
        "brain_reviewed_outcomes",
        "brain_application_events",
        "brain_migration_run_events",
        "brain_migration_batch_events",
    ):
        trigger = triggers.get((table, f"{table}_supersession"))
        if (
            trigger is None
            or trigger["tgenabled"] != "O"
            or trigger["proname"] != "brain_check_supersession"
            or trigger["tgtype"] != 5
        ):
            problems.append(f"supersession trigger on {table} has an invalid binding or enabled state")
    for table, name, function, trigger_type in (
        ("brain_job_decisions", "brain_job_decisions_register", "brain_register_decision", 7),
        ("brain_decision_policies", "brain_decision_policies_lifecycle", "brain_check_policy_lifecycle", 23),
    ):
        trigger = triggers.get((table, name))
        if (
            trigger is None
            or trigger["tgenabled"] != "O"
            or trigger["proname"] != function
            or trigger["tgtype"] != trigger_type
        ):
            problems.append(f"trigger {name} binding or enabled state mismatch")
    for table, name, function in (
        ("brain_parity_run_events", "brain_parity_run_events_pass", "brain_check_parity_pass"),
        ("brain_parity_results", "brain_parity_results_guard", "brain_check_parity_result"),
        ("brain_archive_manifests", "brain_archive_manifests_guard", "brain_check_archive_manifest"),
    ):
        trigger = triggers.get((table, name))
        if trigger is None or trigger["tgenabled"] != "O" or trigger["proname"] != function:
            problems.append(f"trigger {name} binding or enabled state mismatch")

    cur.execute(
        "SELECT p.proname, pg_get_function_identity_arguments(p.oid) AS arguments, p.prosecdef "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "WHERE n.nspname='public' AND p.proname=ANY(%s)",
        (list(_FUNCTIONS),),
    )
    functions = {row["proname"]: row for row in cur.fetchall()}
    for name, arguments in _FUNCTIONS.items():
        function = functions.get(name)
        expected_security_definer = name == "brain_transition_policy" or name in _CONTROLLER_FUNCTIONS
        if (
            function is None
            or function["arguments"] != arguments
            or bool(function["prosecdef"]) != expected_security_definer
        ):
            problems.append(f"function {name} signature or security mode mismatch")

    cur.execute(
        "SELECT pt.partstrat, pg_get_partkeydef(pt.partrelid) AS partkey "
        "FROM pg_partitioned_table pt WHERE pt.partrelid = "
        "to_regclass('public.brain_job_decisions')"
    )
    partition = cur.fetchone()
    if partition is None or partition["partstrat"] != "l" or partition["partkey"] != "LIST (policy_version)":
        problems.append("brain_job_decisions is not LIST partitioned by policy_version")
    cur.execute(
        "SELECT c.relname, pg_get_expr(c.relpartbound, c.oid) AS bound "
        "FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid "
        "WHERE i.inhparent = to_regclass('public.brain_job_decisions')"
    )
    children = {row["relname"]: row["bound"] for row in cur.fetchall()}
    if "DEFAULT" in children.values():
        problems.append("default policy decision partition is forbidden")
    for child in children:
        for trigger_name, function_name, trigger_type in (
            ("brain_job_decisions_register", "brain_register_decision", 7),
            ("brain_job_decisions_append_only", "brain_reject_mutation", 27),
            ("brain_job_decisions_append_only_truncate", "brain_reject_mutation", 34),
        ):
            trigger = triggers.get((child, trigger_name))
            if (
                trigger is None
                or trigger["tgenabled"] != "O"
                or trigger["proname"] != function_name
                or trigger["tgtype"] != trigger_type
            ):
                problems.append(f"partition {child} trigger {trigger_name} contract mismatch")
        cur.execute(
            "SELECT conname,contype,convalidated,pg_get_constraintdef(oid,true) AS definition "
            "FROM pg_constraint WHERE conrelid='public.brain_job_decisions'::regclass "
            "ORDER BY conname"
        )
        parent_constraint_rows = cur.fetchall()
        cur.execute(
            "SELECT conname,contype,convalidated,pg_get_constraintdef(oid,true) AS definition "
            "FROM pg_constraint WHERE conrelid=%s::regclass ORDER BY conname",
            (f"public.{child}",),
        )
        child_constraint_rows = cur.fetchall()
        parent_constraints = sorted(
            (row["contype"], row["convalidated"], row["definition"]) for row in parent_constraint_rows
        )
        child_constraints = sorted(
            (row["contype"], row["convalidated"], row["definition"]) for row in child_constraint_rows
        )
        if child_constraints != parent_constraints:
            problems.append(
                f"partition {child} constraint contract mismatch: "
                f"expected={parent_constraints!r} actual={child_constraints!r}"
            )

        cur.execute(
            "SELECT index_class.relname AS parent_index,idx.indisunique,idx.indisprimary,idx.indisvalid,"
            "idx.indisready,idx.indnkeyatts,idx.indnatts,idx.indkey::text AS keys,"
            "idx.indclass::text AS opclasses,idx.indoption::text AS options,"
            "pg_get_expr(idx.indexprs,idx.indrelid,true) AS expressions,"
            "pg_get_expr(idx.indpred,idx.indrelid,true) AS predicate "
            "FROM pg_index idx JOIN pg_class index_class ON index_class.oid=idx.indexrelid "
            "WHERE idx.indrelid='public.brain_job_decisions'::regclass ORDER BY index_class.relname"
        )
        parent_indexes = {row["parent_index"]: tuple(row.values())[1:] for row in cur.fetchall()}
        cur.execute(
            "SELECT parent_class.relname AS parent_index,child_class.relname AS child_index,"
            "idx.indisunique,idx.indisprimary,idx.indisvalid,idx.indisready,idx.indnkeyatts,idx.indnatts,"
            "idx.indkey::text AS keys,idx.indclass::text AS opclasses,idx.indoption::text AS options,"
            "pg_get_expr(idx.indexprs,idx.indrelid,true) AS expressions,"
            "pg_get_expr(idx.indpred,idx.indrelid,true) AS predicate "
            "FROM pg_index idx JOIN pg_class child_class ON child_class.oid=idx.indexrelid "
            "LEFT JOIN pg_inherits inheritance ON inheritance.inhrelid=idx.indexrelid "
            "LEFT JOIN pg_class parent_class ON parent_class.oid=inheritance.inhparent "
            "WHERE idx.indrelid=%s::regclass ORDER BY child_class.relname",
            (f"public.{child}",),
        )
        child_index_rows = cur.fetchall()
        child_indexes = {
            row["parent_index"]: tuple(row.values())[2:] for row in child_index_rows if row["parent_index"] is not None
        }
        if (
            any(row["parent_index"] is None for row in child_index_rows)
            or set(child_indexes) != set(parent_indexes)
            or any(child_indexes[name] != parent_indexes[name] for name in parent_indexes)
        ):
            problems.append(f"partition {child} index contract mismatch")
    allowed_public_relations = set(_RELATIONS) | _V4_RELATIONS | set(children)
    if len(versions) >= 5:
        allowed_public_relations |= _V5_RELATIONS
    if len(versions) >= 7:
        allowed_public_relations |= _V7_RELATIONS
    cur.execute(
        "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname='public' AND left(c.relname,6)='brain_' AND c.relkind IN ('r','p')"
    )
    unknown_relations = sorted(
        row["relname"] for row in cur.fetchall() if row["relname"] not in allowed_public_relations
    )
    if unknown_relations:
        problems.append("unenumerated brain relations: " + ", ".join(unknown_relations))
    cur.execute(
        "SELECT c.relkind FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'brain_archive' AND c.relname = 'brain_archive_manifests'"
    )
    archive_relation = cur.fetchone()
    if archive_relation is None or archive_relation["relkind"] != "r":
        problems.append("brain_archive_manifests is missing or not an ordinary table")

    if migration_identity is not None:
        cur.execute(
            "SELECT n.nspname, c.relname, owner.rolname AS owner_name "
            "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "JOIN pg_roles owner ON owner.oid = c.relowner "
            "WHERE (n.nspname = 'public' AND left(c.relname, 6) = 'brain_') "
            "OR n.nspname = 'brain_archive'"
        )
        for row in cur.fetchall():
            if row["owner_name"] != migration_identity:
                problems.append(f"ownership mismatch: {row['nspname']}.{row['relname']} owned by {row['owner_name']}")
        cur.execute(
            "SELECT n.nspname, p.proname, owner.rolname AS owner_name "
            "FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
            "JOIN pg_roles owner ON owner.oid = p.proowner "
            "WHERE n.nspname = 'public' AND left(p.proname, 6) = 'brain_'"
        )
        for row in cur.fetchall():
            if row["owner_name"] != migration_identity:
                problems.append(f"ownership mismatch: {row['nspname']}.{row['proname']}() owned by {row['owner_name']}")
        cur.execute(
            "SELECT owner.rolname AS owner_name FROM pg_namespace n "
            "JOIN pg_roles owner ON owner.oid = n.nspowner WHERE n.nspname = 'brain_archive'"
        )
        archive_owner = cur.fetchone()
        if archive_owner is None:
            problems.append("missing brain_archive schema")
        else:
            if archive_owner["owner_name"] != migration_identity:
                problems.append(f"ownership mismatch: brain_archive owned by {archive_owner['owner_name']}")

    catalog_hash = _catalog_contract_hash(cur, include_v5=len(versions) >= 5)
    expected_base_catalog_hash = _pg18_catalog_hash(
        "current_base" if len(versions) >= 3 else "base"
    )
    if catalog_hash != expected_base_catalog_hash:
        problems.append(f"catalog contract hash mismatch: expected {expected_base_catalog_hash}, got {catalog_hash}")

    cur.execute(
        "SELECT rolname,rolcanlogin,rolsuper,rolcreatedb,rolcreaterole,rolreplication,"
        "rolbypassrls,rolinherit FROM pg_roles WHERE rolname=ANY(%s) ORDER BY rolname",
        ([_POLICY_CONTROLLER_ROLE, _STATUS_ROLE],),
    )
    capability_roles = cur.fetchall()
    if len(capability_roles) != 2:
        problems.append("missing fixed lifecycle capability roles")
    for role in capability_roles:
        unsafe = any(
            role[key]
            for key in (
                "rolcanlogin",
                "rolsuper",
                "rolcreatedb",
                "rolcreaterole",
                "rolreplication",
                "rolbypassrls",
                "rolinherit",
            )
        )
        if unsafe:
            problems.append(f"unsafe lifecycle capability role attributes: {role['rolname']}")
    cur.execute(
        "SELECT member_role.rolname AS member_role,parent_role.rolname AS parent_role "
        "FROM pg_auth_members membership "
        "JOIN pg_roles member_role ON member_role.oid=membership.member "
        "JOIN pg_roles parent_role ON parent_role.oid=membership.roleid "
        "WHERE member_role.rolname=ANY(%s) OR parent_role.rolname=ANY(%s)",
        (
            [_POLICY_CONTROLLER_ROLE, _STATUS_ROLE],
            [_MIGRATION_ROLE, _POLICY_CONTROLLER_ROLE, _STATUS_ROLE],
        ),
    )
    unsafe_memberships = [
        f"{row['member_role']}->{row['parent_role']}"
        for row in cur.fetchall()
        if row["member_role"] in {_POLICY_CONTROLLER_ROLE, _STATUS_ROLE}
    ]
    if unsafe_memberships:
        problems.append("unsafe lifecycle role membership: " + ", ".join(unsafe_memberships))
    cur.execute(
        "WITH RECURSIVE memberships(login_oid,roleid) AS ("
        "SELECT oid,oid FROM pg_roles WHERE rolcanlogin UNION "
        "SELECT prior.login_oid,membership.roleid FROM memberships prior "
        "JOIN pg_auth_members membership ON membership.member=prior.roleid) "
        "SELECT DISTINCT login.rolname AS login_name,capability.rolname AS capability "
        "FROM memberships JOIN pg_roles login ON login.oid=memberships.login_oid "
        "JOIN pg_roles capability ON capability.oid=memberships.roleid "
        "WHERE capability.rolname=ANY(%s)",
        ([_POLICY_CONTROLLER_ROLE, _STATUS_ROLE],),
    )
    for lifecycle_login in cur.fetchall():
        login_name = lifecycle_login["login_name"]
        capability_name = lifecycle_login["capability"]
        cur.execute(
            "WITH RECURSIVE memberships(roleid) AS ("
            "SELECT oid FROM pg_roles WHERE rolname=%s UNION "
            "SELECT member.roleid FROM pg_auth_members member "
            "JOIN memberships prior ON prior.roleid=member.member) "
            "SELECT EXISTS(SELECT 1 FROM memberships JOIN pg_roles role ON role.oid=memberships.roleid "
            "WHERE role.rolname<>%s AND role.rolname<>%s) AS extra_membership,"
            "EXISTS(SELECT 1 FROM pg_roles role WHERE role.rolname=%s AND "
            "(role.rolsuper OR role.rolcreatedb OR role.rolcreaterole OR role.rolreplication "
            "OR role.rolbypassrls)) AS unsafe_identity,"
            "has_schema_privilege(%s,'public','CREATE') AS public_create,"
            "EXISTS(SELECT 1 FROM pg_class relation JOIN pg_namespace namespace "
            "ON namespace.oid=relation.relnamespace WHERE namespace.nspname='public' "
            "AND relation.relkind IN ('r','p','v','m','f') AND (has_table_privilege(%s,relation.oid,"
            "'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') OR "
            "has_any_column_privilege(%s,relation.oid,'INSERT,UPDATE,REFERENCES'))) AS mutation_authority,"
            "EXISTS(SELECT 1 FROM pg_proc function JOIN pg_namespace namespace "
            "ON namespace.oid=function.pronamespace CROSS JOIN LATERAL aclexplode(function.proacl) acl "
            "WHERE namespace.nspname='public' AND acl.grantee="
            "(SELECT oid FROM pg_roles WHERE rolname=%s)) AS direct_function_grant",
            (
                login_name,
                login_name,
                capability_name,
                login_name,
                login_name,
                login_name,
                login_name,
                login_name,
            ),
        )
        login_contract = cur.fetchone()
        if login_contract is None or any(login_contract.values()):
            problems.append(
                f"lifecycle login {login_name} exceeds {capability_name} capability"
            )
    cur.execute(
        "SELECT p.proname,has_function_privilege(%s,p.oid,'EXECUTE') AS controller_execute,"
        "has_function_privilege(%s,p.oid,'EXECUTE') AS status_execute "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "WHERE n.nspname='public' AND p.proname=ANY(%s) ORDER BY p.proname",
        (
            _POLICY_CONTROLLER_ROLE,
            _STATUS_ROLE,
            sorted(_CONTROLLER_FUNCTIONS | {"brain_transition_policy"}),
        ),
    )
    lifecycle_function_acl = cur.fetchall()
    if len(lifecycle_function_acl) != 4 or any(
        row["status_execute"]
        or row["controller_execute"] != (row["proname"] in _CONTROLLER_FUNCTIONS)
        for row in lifecycle_function_acl
    ):
        problems.append("lifecycle function ACL contract mismatch")
    missing_owner_privileges: list[str] = []
    for privilege, relations in (
        ("SELECT", _LIFECYCLE_OWNER_READ_RELATIONS),
        ("UPDATE", _LIFECYCLE_LOCK_RELATIONS),
        ("INSERT", {"fleet_decision_policies"}),
    ):
        cur.execute(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname='public' AND c.relname=ANY(%s) "
            "AND NOT has_table_privilege(%s,c.oid,%s) ORDER BY c.relname",
            (sorted(relations), _MIGRATION_ROLE, privilege),
        )
        missing_owner_privileges.extend(
            f"{row['relname']}:{privilege}" for row in cur.fetchall()
        )
    if missing_owner_privileges:
        problems.append(
            "lifecycle function owner lacks required privileges: "
            + ", ".join(missing_owner_privileges)
        )
    for role_name in (_STATUS_ROLE, _POLICY_CONTROLLER_ROLE):
        cur.execute(
            "SELECT has_schema_privilege(%s,'public','USAGE') AS usage,"
            "has_schema_privilege(%s,'public','CREATE') AS create",
            (role_name, role_name),
        )
        if cur.fetchone() != {"usage": True, "create": False}:
            problems.append(f"lifecycle role {role_name} schema ACL mismatch")
    cur.execute(
        "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname='public' AND c.relkind IN ('r','p') AND "
        "has_table_privilege(%s,c.oid,'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')",
        (_STATUS_ROLE,),
    )
    status_mutations = [row["relname"] for row in cur.fetchall()]
    if status_mutations:
        problems.append("status role has mutation privileges: " + ", ".join(status_mutations))
    cur.execute(
        "SELECT c.relname,NULL::text AS column_name,acl.privilege_type FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace CROSS JOIN LATERAL aclexplode(c.relacl) acl "
        "WHERE n.nspname='public' AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s) "
        "UNION ALL "
        "SELECT c.relname,a.attname,acl.privilege_type FROM pg_attribute a "
        "JOIN pg_class c ON c.oid=a.attrelid JOIN pg_namespace n ON n.oid=c.relnamespace "
        "CROSS JOIN LATERAL aclexplode(a.attacl) acl WHERE n.nspname='public' "
        "AND a.attnum>0 AND NOT a.attisdropped "
        "AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s)",
        (_STATUS_ROLE, _STATUS_ROLE),
    )
    actual_status_acls = {
        (row["relname"], row["column_name"], row["privilege_type"])
        for row in cur.fetchall()
    }
    expected_status_acls = {
        (relation, None, "SELECT") for relation in _STATUS_READ_RELATIONS
    } | {
        (relation, column, "SELECT")
        for relation, columns in _STATUS_READ_COLUMNS.items()
        for column in columns
    }
    if actual_status_acls != expected_status_acls:
        problems.append(
            "status role direct ACL contract mismatch: "
            f"missing={sorted(expected_status_acls - actual_status_acls)!r} "
            f"extra={sorted(actual_status_acls - expected_status_acls)!r}"
        )
    cur.execute(
        "SELECT c.relname,acl.privilege_type FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace "
        "CROSS JOIN LATERAL aclexplode(c.relacl) acl "
        "WHERE n.nspname='public' AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s) "
        "UNION ALL "
        "SELECT c.relname||'.'||a.attname,acl.privilege_type FROM pg_attribute a "
        "JOIN pg_class c ON c.oid=a.attrelid JOIN pg_namespace n ON n.oid=c.relnamespace "
        "CROSS JOIN LATERAL aclexplode(a.attacl) acl "
        "WHERE n.nspname='public' AND a.attnum>0 AND NOT a.attisdropped "
        "AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s)",
        (_POLICY_CONTROLLER_ROLE, _POLICY_CONTROLLER_ROLE),
    )
    controller_table_acls = [
        f"{row['relname']}:{row['privilege_type']}" for row in cur.fetchall()
    ]
    if controller_table_acls:
        problems.append(
            "policy controller has forbidden direct table privileges: "
            + ", ".join(sorted(controller_table_acls))
        )

    verifier_identity = _verifier_identity()
    cur.execute(
        "SELECT has_schema_privilege(%s,'public','USAGE') AS public_usage,"
        "has_schema_privilege(%s,'public','CREATE') AS public_create,"
        "has_schema_privilege(%s,'brain_archive','USAGE') AS archive_usage,"
        "has_schema_privilege(%s,'brain_archive','CREATE') AS archive_create",
        (verifier_identity, verifier_identity, verifier_identity, verifier_identity),
    )
    verifier_schema_acl = cur.fetchone()
    if verifier_schema_acl != {
        "public_usage": True,
        "public_create": False,
        "archive_usage": True,
        "archive_create": False,
    }:
        problems.append(f"verifier role {verifier_identity} schema ACL contract mismatch")
    cur.execute(
        "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname='public' AND left(c.relname,6)<>'brain_' AND c.relkind IN ('r','p','v','m','f') "
        "AND has_table_privilege(%s,c.oid,'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')",
        (verifier_identity,),
    )
    unrelated_verifier_relations = [row["relname"] for row in cur.fetchall()]
    if unrelated_verifier_relations:
        problems.append(
            "verifier role has unrelated public relation access: " + ", ".join(sorted(unrelated_verifier_relations))
        )
    cur.execute(
        "SELECT n.nspname, c.relname, COALESCE(grantee.rolname, 'PUBLIC') AS grantee, acl.privilege_type "
        "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "CROSS JOIN LATERAL aclexplode(c.relacl) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee "
        "WHERE ((n.nspname = 'public' AND left(c.relname, 6) = 'brain_') "
        "OR n.nspname = 'brain_archive') AND acl.grantee <> c.relowner"
    )
    invalid_grants = []
    for row in cur.fetchall():
        allowed = row["grantee"] == verifier_identity and row["privilege_type"] == "SELECT"
        allowed = allowed or (
            row["nspname"] == "public"
            and row["relname"] in _STATUS_READ_RELATIONS
            and row["grantee"] == _STATUS_ROLE
            and row["privilege_type"] == "SELECT"
        )
        allowed = allowed or (
            row["nspname"] == "public"
            and row["relname"] in {
                "brain_authority_scope_state",
                "brain_v4_candidate_decisions",
                "brain_v4_decision_envelopes",
                "brain_immutable_artifact_references",
            }
            and row["grantee"] == _CANDIDATE_READER_ROLE
            and row["privilege_type"] == "SELECT"
        )
        if not allowed:
            invalid_grants.append(f"{row['nspname']}.{row['relname']}->{row['grantee']}:{row['privilege_type']}")
    if invalid_grants:
        problems.append("invalid non-owner object ACLs: " + ", ".join(sorted(set(invalid_grants))))
    if verifier_identity is not None:
        cur.execute(
            "SELECT count(*) AS count FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE ((n.nspname='public' AND left(c.relname, 6)='brain_') OR n.nspname='brain_archive') "
            "AND c.relkind IN ('r','p') AND has_table_privilege(%s, c.oid, 'SELECT')",
            (verifier_identity,),
        )
        readable = cur.fetchone()["count"]
        cur.execute(
            "SELECT count(*) AS count FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE ((n.nspname='public' AND left(c.relname, 6)='brain_') OR n.nspname='brain_archive') "
            "AND c.relkind IN ('r','p')"
        )
        if readable != cur.fetchone()["count"]:
            problems.append(f"verifier role {verifier_identity} lacks complete read-only table access")
    cur.execute(
        "SELECT p.proname, COALESCE(grantee.rolname, 'PUBLIC') AS grantee "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
        "CROSS JOIN LATERAL aclexplode(p.proacl) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee "
        "WHERE n.nspname = 'public' AND left(p.proname, 6) = 'brain_' "
        "AND acl.grantee <> p.proowner"
    )
    function_grants = [
        f"{row['proname']}()->{row['grantee']}"
        for row in cur.fetchall()
            if not (
                row["proname"] in _CONTROLLER_FUNCTIONS
                and row["grantee"] == _POLICY_CONTROLLER_ROLE
            ) and not (
                row["proname"] in _V5_POLICY_FUNCTIONS
                and row["grantee"] == _POLICY_CONTROLLER_ROLE
            ) and not (
                row["proname"] in _V5_GRAPH_AUTHORITY_FUNCTIONS
                and row["grantee"] == _GRAPH_AUTHORITY_ROLE
            ) and not (
                row["proname"] in {"brain_publish_v4_candidate", "brain_publish_v5_candidate"}
                and row["grantee"] == _CANDIDATE_WRITER_ROLE
            )
    ]
    if function_grants:
        problems.append("non-owner function ACLs: " + ", ".join(sorted(set(function_grants))))
    cur.execute(
        "SELECT COALESCE(grantee.rolname, 'PUBLIC') AS grantee "
        "FROM pg_namespace n CROSS JOIN LATERAL aclexplode(n.nspacl) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee "
        "WHERE n.nspname = 'brain_archive' AND acl.grantee <> n.nspowner"
    )
    schema_grants = [row["grantee"] for row in cur.fetchall()]
    expected_schema_grants = [] if verifier_identity is None else [verifier_identity]
    if sorted(schema_grants) != expected_schema_grants:
        problems.append("brain_archive schema ACL contract mismatch: " + ", ".join(sorted(schema_grants)))
    cur.execute(
        "SELECT COALESCE(grantee.rolname, 'PUBLIC') AS grantee, acl.privilege_type "
        "FROM pg_namespace n CROSS JOIN LATERAL aclexplode(n.nspacl) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE n.nspname='public' AND acl.privilege_type='CREATE' "
        "AND COALESCE(grantee.rolname, 'PUBLIC') NOT IN (%s, 'pg_database_owner')",
        (migration_identity or "",),
    )
    create_grants = [row["grantee"] for row in cur.fetchall()]
    if create_grants:
        problems.append("non-controller public schema CREATE authority: " + ", ".join(sorted(create_grants)))
    if migration_identity is not None:
        cur.execute(
            "SELECT da.defaclobjtype, COALESCE(grantee.rolname, 'PUBLIC') AS grantee "
            "FROM pg_default_acl da JOIN pg_namespace n ON n.oid = da.defaclnamespace "
            "CROSS JOIN LATERAL aclexplode(da.defaclacl) acl "
            "LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee "
            "WHERE da.defaclrole = (SELECT oid FROM pg_roles WHERE rolname = %s) "
            "AND n.nspname = 'public' AND acl.grantee <> da.defaclrole",
            (migration_identity,),
        )
        default_grants = [f"{row['defaclobjtype']}->{row['grantee']}" for row in cur.fetchall()]
        if default_grants:
            problems.append("non-owner default ACLs: " + ", ".join(sorted(set(default_grants))))

    cur.execute("SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fleet_worker') AS present")
    if cur.fetchone()["present"]:
        cur.execute(
            "SELECT n.nspname, c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE ((n.nspname = 'public' AND left(c.relname, 6) = 'brain_') OR n.nspname = 'brain_archive') "
            "AND c.relkind IN ('r', 'p', 'S') AND CASE WHEN c.relkind = 'S' "
            "THEN has_sequence_privilege('fleet_worker', c.oid, 'USAGE,SELECT,UPDATE') "
            "ELSE has_table_privilege('fleet_worker', c.oid, 'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') END"
        )
        exposed = [f"{row['nspname']}.{row['relname']}" for row in cur.fetchall()]
        if exposed:
            problems.append("fleet_worker authority exposure: " + ", ".join(sorted(exposed)))
        cur.execute(
            "SELECT p.proname FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
            "WHERE n.nspname = 'public' AND left(p.proname, 6) = 'brain_' "
            "AND has_function_privilege('fleet_worker', p.oid, 'EXECUTE')"
        )
        exposed_functions = [row["proname"] for row in cur.fetchall()]
        cur.execute("SELECT has_schema_privilege('fleet_worker', 'brain_archive', 'USAGE,CREATE') AS exposed")
        if exposed_functions or cur.fetchone()["exposed"]:
            problems.append("fleet_worker function/schema exposure: " + ", ".join(sorted(exposed_functions)))

    if problems:
        raise RuntimeError("brain schema v1 verification failed: " + "; ".join(problems))


def _verify_v4_contract(cur) -> None:
    """Verify the additive V4 authority objects without perturbing V1-V3 hashes."""
    versions = _version_rows(cur)
    problems: list[str] = []
    if len(versions) != 4:
        problems.append("missing schema version 4 scoped candidate authority contract")
    else:
        version = versions[3]
        if version["migration_name"] != _MIGRATION_V4_NAME:
            problems.append("migration v4 name mismatch")
        if version["migration_checksum"] != _schema_v4_checksum():
            problems.append("migration v4 checksum mismatch")
        if version["applied_by"] != _MIGRATION_ROLE:
            problems.append("migration v4 ledger owner mismatch")

    cur.execute(
        "SELECT c.relname, owner.rolname AS owner_name FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace JOIN pg_roles owner ON owner.oid=c.relowner "
        "WHERE n.nspname='public' AND c.relname=ANY(%s)",
        (list(_V4_RELATIONS),),
    )
    objects = {row["relname"]: row["owner_name"] for row in cur.fetchall()}
    for relation in _V4_RELATIONS:
        if objects.get(relation) != _MIGRATION_ROLE:
            problems.append(f"v4 ownership mismatch for {relation}: {objects.get(relation)!r}")
    _verify_v4_catalog_contract(cur, problems)
    _verify_v4_publish_function(cur, problems)

    for role_name in (_CANDIDATE_READER_ROLE, _CANDIDATE_WRITER_ROLE, _MIGRATION_ROLE):
        cur.execute(
            "SELECT rolcanlogin,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls,rolinherit "
            "FROM pg_roles WHERE rolname=%s", (role_name,)
        )
        role = cur.fetchone()
        if role is None:
            problems.append(f"missing v4 role {role_name}")
        elif any(
            role[name]
            for name in (
                "rolcanlogin",
                "rolsuper",
                "rolcreatedb",
                "rolcreaterole",
                "rolreplication",
                "rolbypassrls",
            )
        ) or (role_name != _MIGRATION_ROLE and role["rolinherit"]):
            problems.append(f"v4 role attributes invalid for {role_name}")

    cur.execute(
        "SELECT member.rolname AS member_role,parent.rolname AS parent_role FROM pg_auth_members membership "
        "JOIN pg_roles member ON member.oid=membership.member JOIN pg_roles parent ON parent.oid=membership.roleid "
        "WHERE member.rolname IN (%s,%s)",
        (_CANDIDATE_READER_ROLE, _CANDIDATE_WRITER_ROLE),
    )
    memberships = [f"{row['member_role']}->{row['parent_role']}" for row in cur.fetchall()]
    if memberships:
        problems.append("candidate role memberships retained: " + ", ".join(sorted(memberships)))
    cur.execute(
        "SELECT n.nspname,c.relname,COALESCE(role.rolname,'PUBLIC') AS grantee,acl.privilege_type FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace CROSS JOIN LATERAL aclexplode(c.relacl) acl "
        "JOIN pg_roles role ON role.oid=acl.grantee WHERE n.nspname <> 'information_schema' AND n.nspname !~ '^pg_' "
        "AND c.relkind IN ('r','p','v','m','f') "
        "AND role.rolname IN (%s,%s)",
        (_CANDIDATE_READER_ROLE, _CANDIDATE_WRITER_ROLE),
    )
    expected_table_acls = {
        ("public", relation, _CANDIDATE_READER_ROLE, "SELECT")
        for relation in (
            "brain_authority_scope_state",
            "brain_v4_candidate_decisions",
            "brain_v4_decision_envelopes",
            "brain_immutable_artifact_references",
        )
    }
    actual_table_acls = {
        (row["nspname"], row["relname"], row["grantee"], row["privilege_type"])
        for row in cur.fetchall()
    }
    if actual_table_acls != expected_table_acls:
        problems.append("candidate table ACL contract mismatch")
    cur.execute(
        "SELECT n.nspname,c.relname,role.rolname AS grantee,acl.privilege_type FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace CROSS JOIN LATERAL aclexplode(c.relacl) acl "
        "JOIN pg_roles role ON role.oid=acl.grantee WHERE c.relkind='S' "
        "AND role.rolname IN (%s,%s)",
        (_CANDIDATE_READER_ROLE, _CANDIDATE_WRITER_ROLE),
    )
    if cur.fetchall():
        problems.append("candidate sequence ACLs retained")
    cur.execute(
        "SELECT n.nspname,t.typname,role.rolname AS grantee,acl.privilege_type FROM pg_type t "
        "JOIN pg_namespace n ON n.oid=t.typnamespace CROSS JOIN LATERAL aclexplode(t.typacl) acl "
        "JOIN pg_roles role ON role.oid=acl.grantee WHERE t.typtype IN ('c','d','e','r','m') "
        "AND role.rolname IN (%s,%s)",
        (_CANDIDATE_READER_ROLE, _CANDIDATE_WRITER_ROLE),
    )
    if cur.fetchall():
        problems.append("candidate type ACLs retained")
    cur.execute(
        "SELECT n.nspname,p.proname,role.rolname AS grantee,acl.privilege_type FROM pg_proc p "
        "JOIN pg_namespace n ON n.oid=p.pronamespace CROSS JOIN LATERAL aclexplode(p.proacl) acl "
        "JOIN pg_roles role ON role.oid=acl.grantee WHERE role.rolname IN (%s,%s)",
        (_CANDIDATE_READER_ROLE, _CANDIDATE_WRITER_ROLE),
    )
    actual_function_acls = {(row["nspname"], row["proname"], row["grantee"], row["privilege_type"]) for row in cur.fetchall()}
    if actual_function_acls != {("public", "brain_publish_v4_candidate", _CANDIDATE_WRITER_ROLE, "EXECUTE")}:
        problems.append("candidate function ACL contract mismatch")
    cur.execute(
        "SELECT n.nspname,role.rolname AS grantee,acl.privilege_type FROM pg_namespace n "
        "CROSS JOIN LATERAL aclexplode(n.nspacl) acl JOIN pg_roles role ON role.oid=acl.grantee "
        "WHERE role.rolname IN (%s,%s)",
        (_CANDIDATE_READER_ROLE, _CANDIDATE_WRITER_ROLE),
    )
    schema_acls = {(row["nspname"], row["grantee"], row["privilege_type"]) for row in cur.fetchall()}
    allowed_schema_acls = {
        ("public", _CANDIDATE_READER_ROLE, "USAGE"),
        ("public", _CANDIDATE_WRITER_ROLE, "USAGE"),
    }
    if not schema_acls <= allowed_schema_acls:
        problems.append(f"candidate schema ACL contract mismatch: {sorted(schema_acls)!r}")
    for role_name in (_CANDIDATE_READER_ROLE, _CANDIDATE_WRITER_ROLE):
        cur.execute(
            "SELECT has_schema_privilege(%s,'public','USAGE') AS usage,has_schema_privilege(%s,'public','CREATE') AS create",
            (role_name, role_name),
        )
        if cur.fetchone() != {"usage": True, "create": False}:
            problems.append(f"candidate schema capability mismatch for {role_name}")
    cur.execute(
        "SELECT role.rolname FROM pg_database d CROSS JOIN LATERAL aclexplode(d.datacl) acl "
        "JOIN pg_roles role ON role.oid=acl.grantee WHERE d.datname=current_database() "
        "AND role.rolname IN (%s,%s)",
        (_CANDIDATE_READER_ROLE, _CANDIDATE_WRITER_ROLE),
    )
    if cur.fetchall():
        problems.append("candidate database ACLs retained")
    cur.execute(
        "SELECT 1 FROM pg_default_acl da CROSS JOIN LATERAL aclexplode(da.defaclacl) acl "
        "WHERE acl.grantee IN (SELECT oid FROM pg_roles WHERE rolname IN (%s,%s))",
        (_CANDIDATE_READER_ROLE, _CANDIDATE_WRITER_ROLE),
    )
    if cur.fetchone() is not None:
        problems.append("candidate default ACLs retained")

    cur.execute(
        "SELECT c.relname, COALESCE(grantee.rolname,'PUBLIC') AS grantee, acl.privilege_type "
        "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "CROSS JOIN LATERAL aclexplode(COALESCE(c.relacl,acldefault('r',c.relowner))) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE n.nspname='public' AND c.relname=ANY(%s) AND acl.grantee=0",
        (list(_V4_RELATIONS),),
    )
    public_grants = [f"{row['relname']}:{row['privilege_type']}" for row in cur.fetchall()]
    if public_grants:
        problems.append("v4 PUBLIC table grants: " + ", ".join(sorted(public_grants)))
    cur.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_proc p CROSS JOIN LATERAL aclexplode(p.proacl) acl "
        "WHERE p.oid=to_regprocedure(%s) "
        "AND acl.grantee=0) AS allowed",
        (_V4_PUBLISH_SIGNATURE,),
    )
    if cur.fetchone()["allowed"]:
        problems.append("v4 PUBLIC candidate publish grant")
    for relation in _V4_RELATIONS:
        cur.execute(
            "SELECT has_table_privilege(%s,%s,'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') AS allowed",
            (_CANDIDATE_WRITER_ROLE, f"public.{relation}"),
        )
        if cur.fetchone()["allowed"]:
            problems.append(f"candidate writer direct mutation grant on {relation}")
    cur.execute(
        "SELECT has_function_privilege(%s,to_regprocedure(%s),'EXECUTE') AS allowed",
        (_CANDIDATE_WRITER_ROLE, _V4_PUBLISH_SIGNATURE),
    )
    if not cur.fetchone()["allowed"]:
        problems.append("candidate writer lacks bounded publish capability")
    _verify_v4_effective_candidate_authority(cur, problems)
    if problems:
        raise RuntimeError("brain schema v4 verification failed: " + "; ".join(problems))


def _verify_v5_contract(cur) -> None:
    """Verify V5's additive graph authority and superseding candidate ACL."""
    _assert_schema_v5_bytes_immutable()
    versions = _version_rows(cur)
    problems: list[str] = []
    if _schema_v4_checksum() != _EXPECTED_V4_CHECKSUM:
        problems.append(
            f"immutable schema v4 file checksum mismatch: expected {_EXPECTED_V4_CHECKSUM}, "
            f"got {_schema_v4_checksum()}"
        )
    if len(versions) < 5:
        problems.append("missing schema version 5 durable factual graph authority contract")
    else:
        version = versions[4]
        if version["migration_name"] != _MIGRATION_V5_NAME:
            problems.append("migration v5 name mismatch")
        if version["migration_checksum"] != _EXPECTED_V5_CHECKSUM:
            problems.append("migration v5 checksum mismatch")
        if version["applied_by"] != _MIGRATION_ROLE:
            problems.append("migration v5 ledger owner mismatch")

    actual_v5_catalog_hash = _v5_catalog_contract_hash(cur)
    expected_v5_catalog_hash = _pg18_catalog_hash(
        "current_v5" if len(versions) >= 5 else "v5"
    )
    if actual_v5_catalog_hash != expected_v5_catalog_hash:
        problems.append(
            "v5 exact catalog contract mismatch: "
            f"expected {expected_v5_catalog_hash}, got {actual_v5_catalog_hash}"
        )

    cur.execute(
        "SELECT rolcanlogin,rolsuper,rolcreatedb,rolcreaterole,rolinherit,rolreplication,rolbypassrls "
        "FROM pg_roles WHERE rolname=%s",
        (_GRAPH_AUTHORITY_ROLE,),
    )
    graph_role = cur.fetchone()
    if graph_role is None or any(graph_role.values()):
        problems.append(f"unsafe or missing fixed graph authority role: {_GRAPH_AUTHORITY_ROLE}")
    cur.execute(
        "SELECT pg_has_role(%s,%s,'USAGE') AS policy_has_graph,"
        "pg_has_role(%s,%s,'USAGE') AS candidate_has_graph,"
        "has_schema_privilege(%s,'public','USAGE') AS graph_usage,"
        "has_schema_privilege(%s,'public','CREATE') AS graph_create",
        (_POLICY_CONTROLLER_ROLE, _GRAPH_AUTHORITY_ROLE, _CANDIDATE_WRITER_ROLE,
         _GRAPH_AUTHORITY_ROLE, _GRAPH_AUTHORITY_ROLE, _GRAPH_AUTHORITY_ROLE),
    )
    graph_boundary = cur.fetchone()
    if graph_boundary != {
        "policy_has_graph": False,
        "candidate_has_graph": False,
        "graph_usage": True,
        "graph_create": False,
    }:
        problems.append(f"graph authority role boundary mismatch: {graph_boundary!r}")

    cur.execute(
        "WITH RECURSIVE inherited(roleid,path) AS ("
        "SELECT oid,ARRAY[oid] FROM pg_roles WHERE rolname=%s UNION ALL "
        "SELECT membership.roleid,prior.path || membership.roleid "
        "FROM inherited prior JOIN pg_auth_members membership ON membership.member=prior.roleid "
        "WHERE NOT membership.roleid=ANY(prior.path)) "
        "SELECT role.rolname FROM inherited JOIN pg_roles role ON role.oid=inherited.roleid "
        "WHERE role.rolname<>%s ORDER BY role.rolname",
        (_GRAPH_AUTHORITY_ROLE, _GRAPH_AUTHORITY_ROLE),
    )
    graph_inherited_roles = [row["rolname"] for row in cur.fetchall()]
    if graph_inherited_roles:
        problems.append(
            "unsafe graph authority role membership: " + ", ".join(graph_inherited_roles)
        )

    capability_domains = {
        _GRAPH_AUTHORITY_ROLE: "graph",
        _CANDIDATE_READER_ROLE: "candidate",
        _CANDIDATE_WRITER_ROLE: "candidate",
        _POLICY_CONTROLLER_ROLE: "policy",
        _MIGRATION_ROLE: "migration",
        _STATUS_ROLE: "status",
        _VERIFIER_ROLE: "verifier",
    }
    cur.execute(
        "WITH RECURSIVE memberships(login_oid,roleid) AS ("
        "SELECT oid,oid FROM pg_roles WHERE rolcanlogin UNION "
        "SELECT prior.login_oid,membership.roleid FROM memberships prior "
        "JOIN pg_auth_members membership ON membership.member=prior.roleid) "
        "SELECT login.rolname AS login_name,capability.rolname AS capability "
        "FROM memberships JOIN pg_roles login ON login.oid=memberships.login_oid "
        "JOIN pg_roles capability ON capability.oid=memberships.roleid "
        "WHERE capability.rolname=ANY(%s) ORDER BY login.rolname,capability.rolname",
        (sorted(capability_domains),),
    )
    login_domains: dict[str, set[str]] = {}
    for membership in cur.fetchall():
        login_domains.setdefault(membership["login_name"], set()).add(
            capability_domains[membership["capability"]]
        )
    conflicting_logins = {
        login_name: sorted(domains)
        for login_name, domains in login_domains.items()
        if len(domains) > 1
    }
    if conflicting_logins:
        problems.append(
            "conflicting brain capability memberships: "
            + ", ".join(
                f"{login_name}={domains!r}"
                for login_name, domains in sorted(conflicting_logins.items())
            )
        )

    cur.execute(
        "SELECT c.relname,c.relkind,owner.rolname AS owner_name FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace JOIN pg_roles owner ON owner.oid=c.relowner "
        "WHERE n.nspname='public' AND (c.relname=ANY(%s) OR c.relname='brain_factual_contradiction_state')",
        (list(_V5_RELATIONS),),
    )
    objects = {row["relname"]: (row["relkind"], row["owner_name"]) for row in cur.fetchall()}
    for relation in _V5_RELATIONS:
        if objects.get(relation) != ("r", _MIGRATION_ROLE):
            problems.append(f"v5 relation contract mismatch for {relation}: {objects.get(relation)!r}")
    if objects.get("brain_factual_contradiction_state") != ("v", _MIGRATION_ROLE):
        problems.append("v5 contradiction state view contract mismatch")

    cur.execute(
        "SELECT relation.relname,con.contype,ARRAY(SELECT attribute.attname "
        "FROM unnest(con.conkey) WITH ORDINALITY key_column(attnum,position) "
        "JOIN pg_attribute attribute ON attribute.attrelid=con.conrelid "
        "AND attribute.attnum=key_column.attnum ORDER BY key_column.position) AS columns,"
        "target.relname AS target_name,ARRAY(SELECT attribute.attname "
        "FROM unnest(con.confkey) WITH ORDINALITY key_column(attnum,position) "
        "JOIN pg_attribute attribute ON attribute.attrelid=con.confrelid "
        "AND attribute.attnum=key_column.attnum ORDER BY key_column.position) AS target_columns "
        "FROM pg_constraint con JOIN pg_class relation ON relation.oid=con.conrelid "
        "JOIN pg_namespace namespace ON namespace.oid=relation.relnamespace "
        "LEFT JOIN pg_class target ON target.oid=con.confrelid "
        "WHERE namespace.nspname='public' AND relation.relname=ANY(%s) AND con.contype IN ('p','f')",
        (list(_V5_RELATIONS),),
    )
    constraints = cur.fetchall()
    primary_keys = {
        row["relname"]: tuple(row["columns"])
        for row in constraints
        if row["contype"] == "p"
    }
    expected_primary_keys = {
        "brain_authority_epoch_events": ("authority_epoch_event_id",),
        "brain_factual_ontology_manifests": ("owner_id", "ontology_version"),
        "brain_factual_ontology_terms": ("owner_id", "ontology_version", "predicate", "term_id"),
        "brain_factual_ontology_closures": ("owner_id", "ontology_version"),
        "brain_factual_generations": ("owner_id", "generation_id"),
        "brain_factual_generation_members": ("owner_id", "generation_id", "source_span_id"),
        "brain_factual_generation_closures": ("owner_id", "generation_id"),
        "brain_factual_approval_receipts": ("owner_id", "human_approval_id"),
        "brain_graph_fact_events": ("owner_id", "event_id"),
        "brain_factual_approval_consumptions": ("owner_id", "human_approval_id"),
        "brain_factual_generation_coverage": ("owner_id", "generation_id", "source_span_id"),
        "brain_factual_contradictions": ("owner_id", "contradiction_id"),
        "brain_factual_contradiction_events": ("contradiction_event_id",),
        "brain_factual_graph_snapshots": ("owner_id", "graph_snapshot_id"),
        "brain_factual_snapshot_approval_bindings": ("graph_approval_receipt_id",),
        "brain_v5_candidate_publication_events": ("candidate_decision_id",),
    }
    if primary_keys != expected_primary_keys:
        problems.append(f"v5 primary-key contract mismatch: {primary_keys!r}")
    foreign_keys = {
        (row["relname"], tuple(row["columns"]), row["target_name"], tuple(row["target_columns"]))
        for row in constraints
        if row["contype"] == "f"
    }
    required_foreign_keys = {
        ("brain_authority_epoch_events", ("authority_scope_id",), "brain_authority_scope_state", ("authority_scope_id",)),
        ("brain_authority_epoch_events", ("predecessor_event_id",), "brain_authority_epoch_events", ("authority_epoch_event_id",)),
        ("brain_authority_epoch_events", ("transition_receipt_hash",), "brain_artifacts", ("artifact_hash",)),
        ("brain_factual_ontology_manifests", ("ontology_manifest_hash",), "brain_artifacts", ("artifact_hash",)),
        ("brain_factual_ontology_terms", ("ontology_manifest_hash",), "brain_artifacts", ("artifact_hash",)),
        ("brain_factual_ontology_terms", ("term_artifact_hash",), "brain_artifacts", ("artifact_hash",)),
        ("brain_factual_ontology_closures", ("owner_id", "ontology_version", "ontology_manifest_hash"), "brain_factual_ontology_manifests", ("owner_id", "ontology_version", "ontology_manifest_hash")),
        ("brain_factual_ontology_closures", ("close_receipt_hash",), "brain_artifacts", ("artifact_hash",)),
        ("brain_factual_generations", ("owner_id", "ontology_version", "ontology_manifest_hash", "ontology_root_hash"), "brain_factual_ontology_closures", ("owner_id", "ontology_version", "ontology_manifest_hash", "ontology_root_hash")),
        ("brain_factual_generations", ("membership_manifest_hash",), "brain_artifacts", ("artifact_hash",)),
        ("brain_factual_generation_members", ("owner_id", "generation_id"), "brain_factual_generations", ("owner_id", "generation_id")),
        ("brain_factual_generation_members", ("source_artifact_hash",), "brain_artifacts", ("artifact_hash",)),
        ("brain_factual_generation_closures", ("owner_id", "generation_id"), "brain_factual_generations", ("owner_id", "generation_id")),
        ("brain_factual_generation_closures", ("close_receipt_hash",), "brain_artifacts", ("artifact_hash",)),
        ("brain_factual_ontology_terms", ("owner_id", "ontology_version", "ontology_manifest_hash"), "brain_factual_ontology_manifests", ("owner_id", "ontology_version", "ontology_manifest_hash")),
        ("brain_factual_approval_receipts", ("owner_id", "ontology_version", "predicate", "term_id"), "brain_factual_ontology_terms", ("owner_id", "ontology_version", "predicate", "term_id")),
        ("brain_factual_approval_receipts", ("owner_id", "generation_id", "source_span_id", "source_artifact_hash", "source_class"), "brain_factual_generation_members", ("owner_id", "generation_id", "source_span_id", "source_artifact_hash", "source_class")),
        ("brain_factual_approval_receipts", ("approval_receipt_hash",), "brain_artifacts", ("artifact_hash",)),
        ("brain_factual_approval_receipts", ("source_artifact_hash",), "brain_artifacts", ("artifact_hash",)),
        ("brain_graph_fact_events", ("event_artifact_hash",), "brain_artifacts", ("artifact_hash",)),
        ("brain_graph_fact_events", ("owner_id", "human_approval_id", "approval_receipt_hash", "claim_projection_hash", "ontology_version", "predicate", "term_id", "mutation_action"), "brain_factual_approval_receipts", ("owner_id", "human_approval_id", "approval_receipt_hash", "claim_projection_hash", "ontology_version", "predicate", "term_id", "mutation_action")),
        ("brain_graph_fact_events", ("owner_id", "generation_id", "source_span_id", "supersedes_event_id"), "brain_graph_fact_events", ("owner_id", "generation_id", "source_span_id", "event_id")),
        ("brain_factual_approval_consumptions", ("owner_id", "human_approval_id", "approval_receipt_hash"), "brain_factual_approval_receipts", ("owner_id", "human_approval_id", "approval_receipt_hash")),
        ("brain_factual_approval_consumptions", ("owner_id", "event_id", "approval_receipt_hash"), "brain_graph_fact_events", ("owner_id", "event_id", "approval_receipt_hash")),
        ("brain_factual_generation_coverage", ("owner_id", "generation_id", "source_span_id"), "brain_factual_generation_members", ("owner_id", "generation_id", "source_span_id")),
        ("brain_factual_generation_coverage", ("owner_id", "generation_id", "source_span_id", "event_id"), "brain_graph_fact_events", ("owner_id", "generation_id", "source_span_id", "event_id")),
        ("brain_factual_generation_coverage", ("review_receipt_hash",), "brain_artifacts", ("artifact_hash",)),
        ("brain_factual_contradictions", ("owner_id", "generation_id"), "brain_factual_generations", ("owner_id", "generation_id")),
        ("brain_factual_contradictions", ("contradiction_artifact_hash",), "brain_artifacts", ("artifact_hash",)),
        ("brain_factual_contradiction_events", ("owner_id", "contradiction_id"), "brain_factual_contradictions", ("owner_id", "contradiction_id")),
        ("brain_factual_contradiction_events", ("previous_event_id",), "brain_factual_contradiction_events", ("contradiction_event_id",)),
        ("brain_factual_contradiction_events", ("review_receipt_hash",), "brain_artifacts", ("artifact_hash",)),
        ("brain_factual_graph_snapshots", ("owner_id", "generation_id"), "brain_factual_generation_closures", ("owner_id", "generation_id")),
        ("brain_factual_graph_snapshots", ("coverage_receipt_hash",), "brain_artifacts", ("artifact_hash",)),
        ("brain_factual_graph_snapshots", ("snapshot_artifact_hash",), "brain_artifacts", ("artifact_hash",)),
        ("brain_factual_snapshot_approval_bindings", ("owner_id", "graph_snapshot_id"), "brain_factual_graph_snapshots", ("owner_id", "graph_snapshot_id")),
        ("brain_factual_snapshot_approval_bindings", ("graph_approval_receipt_id",), "brain_graph_approval_receipts", ("graph_approval_receipt_id",)),
        ("brain_factual_snapshot_approval_bindings", ("authority_epoch_event_id",), "brain_authority_epoch_events", ("authority_epoch_event_id",)),
        ("brain_v5_candidate_publication_events", ("candidate_decision_id",), "brain_v4_candidate_decisions", ("candidate_decision_id",)),
        ("brain_v5_candidate_publication_events", ("graph_approval_receipt_id",), "brain_graph_approval_receipts", ("graph_approval_receipt_id",)),
        ("brain_v5_candidate_publication_events", ("authority_scope_id",), "brain_authority_scope_state", ("authority_scope_id",)),
        ("brain_v5_candidate_publication_events", ("authority_epoch_event_id",), "brain_authority_epoch_events", ("authority_epoch_event_id",)),
    }
    if foreign_keys != required_foreign_keys:
        problems.append(
            "v5 foreign-key contract mismatch: "
            f"missing={sorted(required_foreign_keys - foreign_keys)!r}, "
            f"extra={sorted(foreign_keys - required_foreign_keys)!r}"
        )

    cur.execute(
        "SELECT relation.relname,ARRAY(SELECT attribute.attname FROM unnest(con.conkey) "
        "WITH ORDINALITY key_column(attnum,position) JOIN pg_attribute attribute "
        "ON attribute.attrelid=con.conrelid AND attribute.attnum=key_column.attnum "
        "ORDER BY key_column.position) AS columns FROM pg_constraint con "
        "JOIN pg_class relation ON relation.oid=con.conrelid "
        "JOIN pg_namespace namespace ON namespace.oid=relation.relnamespace "
        "WHERE namespace.nspname='public' AND relation.relname=ANY(%s) AND con.contype='u'",
        (list(_V5_RELATIONS),),
    )
    unique_keys = {(row["relname"], tuple(row["columns"])) for row in cur.fetchall()}
    expected_unique_keys = {
        ("brain_authority_epoch_events", ("predecessor_event_id",)),
        ("brain_authority_epoch_events", ("authority_scope_id", "event_sequence")),
        ("brain_authority_epoch_events", ("authority_scope_id", "authority_epoch", "database_incarnation_id", "event_type")),
        ("brain_factual_ontology_manifests", ("owner_id", "ontology_version", "ontology_manifest_hash")),
        ("brain_factual_ontology_manifests", ("owner_id", "ontology_manifest_hash")),
        ("brain_factual_ontology_terms", ("owner_id", "ontology_version", "term_id")),
        ("brain_factual_ontology_terms", ("owner_id", "ontology_version", "predicate", "term_namespace", "term_digest")),
        ("brain_factual_ontology_closures", ("owner_id", "ontology_version", "ontology_manifest_hash", "ontology_root_hash")),
        ("brain_factual_generation_members", ("owner_id", "generation_id", "member_ordinal")),
        ("brain_factual_generation_members", ("owner_id", "generation_id", "source_span_id", "source_artifact_hash", "source_class")),
        ("brain_factual_approval_receipts", ("owner_id", "approval_receipt_hash")),
        ("brain_factual_approval_receipts", ("owner_id", "human_approval_id", "approval_receipt_hash")),
        ("brain_factual_approval_receipts", ("owner_id", "human_approval_id", "approval_receipt_hash", "claim_projection_hash", "ontology_version", "predicate", "term_id", "mutation_action")),
        ("brain_graph_fact_events", ("owner_id", "generation_id", "system_receipt_sequence")),
        ("brain_graph_fact_events", ("owner_id", "generation_id", "source_span_id", "event_id")),
        ("brain_graph_fact_events", ("owner_id", "event_id", "approval_receipt_hash")),
        ("brain_factual_approval_consumptions", ("owner_id", "event_id")),
        ("brain_factual_contradiction_events", ("previous_event_id",)),
        ("brain_factual_contradiction_events", ("owner_id", "contradiction_id", "event_sequence")),
        ("brain_factual_graph_snapshots", ("owner_id", "generation_id", "semantic_root_hash")),
        ("brain_v5_candidate_publication_events", ("authority_scope_id", "candidate_decision_id")),
    }
    if unique_keys != expected_unique_keys:
        problems.append(
            "v5 unique-key contract mismatch: "
            f"missing={sorted(expected_unique_keys - unique_keys)!r}, "
            f"extra={sorted(unique_keys - expected_unique_keys)!r}"
        )

    cur.execute(
        "SELECT relation.relname,pg_get_constraintdef(con.oid,true) AS definition "
        "FROM pg_constraint con JOIN pg_class relation ON relation.oid=con.conrelid "
        "JOIN pg_namespace namespace ON namespace.oid=relation.relnamespace "
        "WHERE namespace.nspname='public' AND relation.relname=ANY(%s) AND con.contype='c'",
        (list(_V5_RELATIONS),),
    )
    checks: dict[str, tuple[str, ...]] = {}
    for row in cur.fetchall():
        checks.setdefault(row["relname"], tuple())
        checks[row["relname"]] += (_compact_sql(row["definition"]),)
    required_checks = {
        "brain_authority_epoch_events": (
            ("event_sequence>0",),
            ("event_type=ANY", "granted", "revoked"),
            ("event_sequence=1", "event_type='granted'", "predecessor_event_idisnull"),
            ("event_sequence>1", "predecessor_event_idisnotnull"),
        ),
        "brain_factual_ontology_terms": (
            ("term_id", "term_namespace", "term_digest"),
            ("term_digest", "^[0-9a-f]{64}$"),
            ("predicate='has_skill'", "term_namespace='skill'"),
            ("predicate='has_work_authorization'", "term_namespace='work-authorization'"),
        ),
        "brain_factual_ontology_closures": (
            ("term_count>=0",),
            ("ontology_root_hash", "^[0-9a-f]{64}$"),
        ),
        "brain_factual_generations": (("ontology_root_hash", "^[0-9a-f]{64}$"),),
        "brain_factual_generation_closures": (
            ("membership_count>=0",),
            ("membership_root_hash", "^[0-9a-f]{64}$"),
        ),
        "brain_graph_fact_events": (
            ("mutation_action='assert'", "supersedes_event_idisnull"),
            ("mutation_action='supersede'", "supersedes_event_idisnotnull"),
            ("system_receipt_sequence>0",),
        ),
        "brain_factual_generation_coverage": (
            ("disposition='assertion'", "event_idisnotnull", "review_receipt_hashisnull"),
            ("disposition='exclusion'", "event_idisnull", "review_receipt_hashisnotnull"),
        ),
        "brain_factual_contradiction_events": (
            ("state_after='active'",),
            ("state_after='resolved'", "review_receipt_hashisnotnull"),
        ),
        "brain_factual_graph_snapshots": (("valid_toisnull", "valid_to>valid_from"),),
    }
    for relation, fragment_groups in required_checks.items():
        definitions = checks.get(relation, ())
        for fragments in fragment_groups:
            normalized = tuple(_compact_sql(fragment) for fragment in fragments)
            if not any(all(fragment in definition for fragment in normalized) for definition in definitions):
                problems.append(f"v5 check contract mismatch for {relation}: required={fragments!r}")

    cur.execute(
        "SELECT a.attname FROM pg_attribute a "
        "WHERE a.attrelid='public.brain_graph_approval_receipts'::regclass "
        "AND a.attname='predecessor_deny_graph_approval_receipt_id' AND NOT a.attisdropped"
    )
    if cur.fetchone() is None:
        problems.append("v5 predecessor-denial identity column missing")
    cur.execute(
        "SELECT con.contype,ARRAY(SELECT attribute.attname FROM unnest(con.conkey) "
        "WITH ORDINALITY key_column(attnum,position) JOIN pg_attribute attribute "
        "ON attribute.attrelid=con.conrelid AND attribute.attnum=key_column.attnum "
        "ORDER BY key_column.position) AS columns,target.relname AS target_name "
        "FROM pg_constraint con LEFT JOIN pg_class target ON target.oid=con.confrelid "
        "WHERE con.conrelid='public.brain_graph_approval_receipts'::regclass AND con.contype IN ('u','f')"
    )
    approval_constraints = {
        (row["contype"], tuple(row["columns"]), row["target_name"])
        for row in cur.fetchall()
    }
    if (
        "u",
        ("authority_scope_id", "authority_epoch", "database_incarnation_id", "graph_snapshot_id", "approval_state"),
        None,
    ) not in approval_constraints:
        problems.append("v5 graph approval state identity key missing")
    if (
        "f",
        ("predecessor_deny_graph_approval_receipt_id",),
        "brain_graph_approval_receipts",
    ) not in approval_constraints:
        problems.append("v5 predecessor-denial self lineage missing")

    cur.execute(
        "SELECT relation.relname,con.contype,ARRAY(SELECT attribute.attname FROM unnest(con.conkey) "
        "WITH ORDINALITY key_column(attnum,position) JOIN pg_attribute attribute "
        "ON attribute.attrelid=con.conrelid AND attribute.attnum=key_column.attnum "
        "ORDER BY key_column.position) AS columns FROM pg_constraint con "
        "JOIN pg_class relation ON relation.oid=con.conrelid "
        "WHERE con.conrelid IN ('public.brain_authority_transition_events'::regclass,"
        "'public.brain_graph_approval_consumptions'::regclass) AND con.contype='u'"
    )
    superseded_uniques = {
        (row["relname"], tuple(row["columns"])) for row in cur.fetchall()
    }
    if (
        "brain_authority_transition_events",
        ("authority_scope_id", "event_type", "authority_epoch", "database_incarnation_id"),
    ) not in superseded_uniques:
        problems.append("v5 altered the immutable V4 transition identity contract")
    if (
        "brain_graph_approval_consumptions",
        ("graph_approval_receipt_id",),
    ) in superseded_uniques:
        problems.append("v5 still limits an approved factual snapshot to one candidate")
    if (
        "brain_graph_approval_consumptions",
        ("candidate_decision_id",),
    ) not in superseded_uniques:
        problems.append("v5 per-candidate graph approval usage identity is missing")

    cur.execute(
        "SELECT index_class.relname,i.indisunique,relation.relname AS table_name,"
        "ARRAY(SELECT attribute.attname FROM unnest(i.indkey) WITH ORDINALITY key_column(attnum,position) "
        "JOIN pg_attribute attribute ON attribute.attrelid=i.indrelid AND attribute.attnum=key_column.attnum "
        "ORDER BY key_column.position) AS columns,pg_get_expr(i.indpred,i.indrelid) AS predicate "
        "FROM pg_index i JOIN pg_class index_class ON index_class.oid=i.indexrelid "
        "JOIN pg_class relation ON relation.oid=i.indrelid WHERE index_class.relname=ANY(%s)",
        ([
            "brain_authority_epoch_events_latest_v5",
            "brain_graph_approval_consumptions_receipt_v5",
            "brain_v5_candidate_publication_receipt_idx",
        ],),
    )
    superseding_indexes = {row["relname"]: row for row in cur.fetchall()}
    expected_indexes = {
        "brain_authority_epoch_events_latest_v5": (
            False, "brain_authority_epoch_events", ("authority_scope_id", "event_sequence"), None
        ),
        "brain_graph_approval_consumptions_receipt_v5": (
            False, "brain_graph_approval_consumptions", ("graph_approval_receipt_id",), None
        ),
        "brain_v5_candidate_publication_receipt_idx": (
            False, "brain_v5_candidate_publication_events", ("graph_approval_receipt_id",), None
        ),
    }
    actual_indexes = {
        name: (row["indisunique"], row["table_name"], tuple(row["columns"]), row["predicate"])
        for name, row in superseding_indexes.items()
    }
    if actual_indexes != expected_indexes:
        problems.append(f"v5 explicit index contract mismatch: {actual_indexes!r}")

    cur.execute(
        "SELECT relation.relname,t.tgname,t.tgenabled,t.tgtype,function.proname "
        "FROM pg_trigger t JOIN pg_class relation ON relation.oid=t.tgrelid "
        "JOIN pg_namespace namespace ON namespace.oid=relation.relnamespace "
        "JOIN pg_proc function ON function.oid=t.tgfoid "
        "WHERE NOT t.tgisinternal AND namespace.nspname='public' AND relation.relname=ANY(%s)",
        (list(_V5_RELATIONS),),
    )
    triggers = {
        (row["relname"], row["tgname"], row["tgenabled"], row["tgtype"], row["proname"])
        for row in cur.fetchall()
    }
    expected_triggers = {
        (relation, f"{relation}_immutable", "O", 27, "brain_reject_mutation")
        for relation in _V5_RELATIONS
    }
    expected_triggers.add((
        "brain_factual_generation_members",
        "brain_factual_generation_members_closed",
        "O",
        7,
        "brain_reject_closed_generation_member",
    ))
    expected_triggers.add((
        "brain_factual_ontology_terms",
        "brain_factual_ontology_terms_closed",
        "O",
        7,
        "brain_reject_closed_ontology_term",
    ))
    if triggers != expected_triggers:
        problems.append("v5 immutable/closure trigger contract mismatch")

    expected_bodies = _expected_v5_function_body_fingerprints()
    cur.execute(
        "SELECT p.proname,owner.rolname AS owner_name,p.prosecdef,p.proconfig,"
        "pg_get_function_identity_arguments(p.oid) AS identity_arguments,"
        "l.lanname AS language_name,p.prosrc AS function_body FROM pg_proc p "
        "JOIN pg_namespace n ON n.oid=p.pronamespace JOIN pg_roles owner ON owner.oid=p.proowner "
        "JOIN pg_language l ON l.oid=p.prolang "
        "WHERE n.nspname='public' AND p.proname=ANY(%s)",
        (list(_V5_FUNCTIONS),),
    )
    functions = {row["proname"]: row for row in cur.fetchall()}
    for name, fingerprint in expected_bodies.items():
        function = functions.get(name)
        if function is None:
            problems.append(f"v5 bounded function missing: {name}")
            continue
        expected_security_definer = name not in {"brain_v5_sha256_text", "brain_v5_frame"}
        if (
            function["owner_name"] != _MIGRATION_ROLE
            or function["prosecdef"] != expected_security_definer
        ):
            problems.append(f"v5 bounded function ownership/security mismatch: {name}")
        if tuple(function["proconfig"] or ()) != ("search_path=pg_catalog, public",):
            problems.append(f"v5 bounded function search_path mismatch: {name}")
        if _raw_function_body_fingerprint(function["function_body"]) != fingerprint:
            problems.append(f"v5 bounded function body fingerprint mismatch: {name}")
    expected_function_arguments = {
        "brain_v5_sha256_text": "requested_value text",
        "brain_v5_frame": "requested_value text",
        "brain_compute_factual_ontology_root": "requested_owner_id text, requested_ontology_version text",
        "brain_compute_factual_membership_root": "requested_owner_id text, requested_generation_id text",
        "brain_compute_factual_semantic_root": "requested_owner_id text, requested_generation_id text",
        "brain_reject_closed_ontology_term": "",
        "brain_reject_closed_generation_member": "",
        "brain_create_factual_ontology": "requested_owner_id text, requested_ontology_version text, requested_manifest_hash text",
        "brain_add_factual_ontology_term": "requested_owner_id text, requested_ontology_version text, requested_manifest_hash text, requested_predicate text, requested_term_namespace text, requested_term_digest text, requested_term_id text, requested_canonical_label text, requested_term_artifact_hash text",
        "brain_close_factual_ontology": "requested_owner_id text, requested_ontology_version text, requested_term_count bigint, requested_ontology_root_hash text, requested_close_receipt_hash text",
        "brain_create_factual_generation": "requested_owner_id text, requested_generation_id text, requested_membership_manifest_hash text, requested_ontology_version text, requested_ontology_root_hash text",
        "brain_add_factual_generation_member": "requested_owner_id text, requested_generation_id text, requested_source_span_id text, requested_source_artifact_hash text, requested_source_class text, requested_member_ordinal bigint",
        "brain_close_factual_generation": "requested_owner_id text, requested_generation_id text, requested_membership_count bigint, requested_membership_root_hash text, requested_close_receipt_hash text",
        "brain_admit_factual_event": "requested_owner_id text, requested_generation_id text, requested_source_span_id text, requested_human_approval_id text, requested_approval_receipt_hash text, requested_claim_projection_hash text, requested_source_artifact_hash text, requested_source_class text, requested_ontology_version text, requested_predicate text, requested_term_id text, requested_event_id text, requested_event_artifact_hash text, requested_system_receipt_sequence bigint, requested_mutation_action text, requested_supersedes_event_id text, requested_issued_at timestamp with time zone",
        "brain_record_factual_assertion_coverage": "requested_owner_id text, requested_generation_id text, requested_source_span_id text, requested_event_id text",
        "brain_review_factual_exclusion": "requested_owner_id text, requested_generation_id text, requested_source_span_id text, requested_reason text, requested_review_receipt_hash text, requested_reviewer_id text, requested_reviewed_at timestamp with time zone",
        "brain_create_factual_contradiction": "requested_owner_id text, requested_contradiction_id text, requested_generation_id text, requested_contradiction_artifact_hash text",
        "brain_append_factual_contradiction_event": "requested_owner_id text, requested_contradiction_id text, requested_event_sequence bigint, requested_event_type text, requested_state_after text, requested_severity text, requested_previous_event_id bigint, requested_review_receipt_hash text",
        "brain_publish_factual_snapshot": "requested_owner_id text, requested_graph_snapshot_id text, requested_generation_id text, requested_semantic_root_hash text, requested_coverage_receipt_hash text, requested_membership_root_hash text, requested_event_high_water bigint, requested_valid_from timestamp with time zone, requested_valid_to timestamp with time zone, requested_snapshot_artifact_hash text",
        "brain_record_authority_epoch_event": "requested_authority_scope_id bigint, requested_event_sequence bigint, requested_event_type text, requested_authority_epoch bigint, requested_database_incarnation_id uuid, requested_predecessor_event_id bigint, requested_actor_id text, requested_transition_receipt_hash text",
        "brain_record_graph_approval_v5": "requested_authority_scope_id bigint, requested_authority_epoch bigint, requested_database_incarnation_id uuid, requested_graph_snapshot_id text, requested_approval_state text, requested_approval_artifact_hash text, requested_predecessor_deny_graph_approval_receipt_id bigint, requested_predecessor_deny_receipt_hash text",
        "brain_bind_factual_snapshot_approval": "requested_owner_id text, requested_graph_snapshot_id text, requested_graph_approval_receipt_id bigint",
        "brain_publish_v5_candidate": _V5_PUBLISH_ARGUMENTS,
    }
    actual_function_arguments = {
        name: row["identity_arguments"] for name, row in functions.items()
    }
    if actual_function_arguments != expected_function_arguments:
        problems.append(f"v5 bounded function signature mismatch: {actual_function_arguments!r}")
    _verify_v4_publish_function(cur, problems)

    cur.execute(
        "SELECT c.relname,acl.privilege_type FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "CROSS JOIN LATERAL aclexplode(COALESCE(c.relacl,acldefault('r',c.relowner))) acl "
        "WHERE n.nspname='public' AND c.relname=ANY(%s) AND acl.grantee=0",
        (list(_V5_RELATIONS | {"brain_factual_contradiction_state"}),),
    )
    public_grants = cur.fetchall()
    if public_grants:
        problems.append(f"v5 PUBLIC relation grants: {public_grants!r}")
    for relation in _V5_RELATIONS:
        cur.execute(
            "SELECT has_table_privilege(%s,%s,'SELECT') AS verifier_read,"
            "has_table_privilege(%s,%s,'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') AS candidate_access,"
            "has_table_privilege(%s,%s,'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') AS controller_access,"
            "has_table_privilege(%s,%s,'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') AS graph_access",
            (_VERIFIER_ROLE, f"public.{relation}", _CANDIDATE_WRITER_ROLE, f"public.{relation}",
             _POLICY_CONTROLLER_ROLE, f"public.{relation}", _GRAPH_AUTHORITY_ROLE,
             f"public.{relation}"),
        )
        access = cur.fetchone()
        if not access["verifier_read"]:
            problems.append(f"v5 verifier lacks SELECT on {relation}")
        if access["candidate_access"]:
            problems.append(f"candidate writer direct graph DML on {relation}")
        if access["controller_access"]:
            problems.append(f"policy controller direct graph DML on {relation}")
        if access["graph_access"]:
            problems.append(f"graph authority has forbidden direct graph DML on {relation}")
    cur.execute(
        "SELECT has_table_privilege(%s,'public.brain_factual_contradiction_state','SELECT') AS verifier_read",
        (_VERIFIER_ROLE,),
    )
    if not cur.fetchone()["verifier_read"]:
        problems.append("v5 verifier lacks SELECT on brain_factual_contradiction_state")
    cur.execute(
        "SELECT has_function_privilege(%s,to_regprocedure(%s),'EXECUTE') AS old_allowed,"
        "has_function_privilege(%s,to_regprocedure(%s),'EXECUTE') AS new_allowed",
        (_CANDIDATE_WRITER_ROLE, _V4_PUBLISH_SIGNATURE, _CANDIDATE_WRITER_ROLE, _V5_PUBLISH_SIGNATURE),
    )
    publish_acl = cur.fetchone()
    if publish_acl != {"old_allowed": False, "new_allowed": True}:
        problems.append(f"v5 superseding candidate function ACL mismatch: {publish_acl!r}")
    cur.execute(
        "SELECT p.proname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "WHERE n.nspname='public' AND p.proname=ANY(%s) AND has_function_privilege('public',p.oid,'EXECUTE')",
        (list(_V5_FUNCTIONS),),
    )
    public_functions = [row["proname"] for row in cur.fetchall()]
    if public_functions:
        problems.append(f"v5 PUBLIC function grants: {sorted(public_functions)!r}")
    cur.execute(
        "SELECT p.proname,grantee.rolname AS grantee,acl.privilege_type "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "CROSS JOIN LATERAL aclexplode(COALESCE(p.proacl,acldefault('f',p.proowner))) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE n.nspname='public' AND p.proname=ANY(%s) AND acl.grantee<>p.proowner",
        (list(_V5_FUNCTIONS),),
    )
    actual_function_grants = {
        (row["proname"], row["grantee"], row["privilege_type"])
        for row in cur.fetchall()
    }
    expected_function_grants = {
        (name, _POLICY_CONTROLLER_ROLE, "EXECUTE") for name in _V5_POLICY_FUNCTIONS
    } | {
        (name, _GRAPH_AUTHORITY_ROLE, "EXECUTE")
        for name in _V5_GRAPH_AUTHORITY_FUNCTIONS
    } | {("brain_publish_v5_candidate", _CANDIDATE_WRITER_ROLE, "EXECUTE")}
    if actual_function_grants != expected_function_grants:
        problems.append(
            "v5 bounded function ACL mismatch: "
            f"missing={sorted(expected_function_grants - actual_function_grants)!r}, "
            f"extra={sorted(actual_function_grants - expected_function_grants)!r}"
        )
    if problems:
        raise RuntimeError("brain schema v5 verification failed: " + "; ".join(problems))


def verify_brain_schema_v1_in_transaction(cur) -> None:
    """Deeply verify the immutable V1-V3 prefix in the caller's transaction."""
    require_pg18_authority_catalog(cur)
    _verify_contract(cur)


def verify_brain_schema_v1(conn) -> None:
    """Deeply verify schema v1 without changing caller transaction state."""
    _require_idle(conn)
    with conn.transaction():
        with conn.cursor() as cur:
            verify_brain_schema_v1_in_transaction(cur)


def ensure_brain_schema_v1_in_transaction(
    cur,
    *,
    lock_timeout_seconds: float = _LOCK_TIMEOUT_SECONDS,
) -> None:
    """Install or verify V1-V3 using the caller's active transaction."""
    require_pg18_authority_catalog(cur)
    versions = _version_rows(cur)
    if len(versions) >= 3:
        _verify_contract(cur)
        return

    migration_identity = _activate_migration_identity(cur)
    _assert_existing_ownership(cur, migration_identity)
    _acquire_xact_lock(cur, lock_timeout_seconds)

    # Another installer may have completed while this transaction waited.
    versions = _version_rows(cur)
    if len(versions) >= 3:
        _verify_contract(cur)
        return

    if not versions:
        cur.execute(_schema_bytes().decode("utf-8"))
        _apply_acl_contract(cur, migration_identity)
        cur.execute(
            "INSERT INTO public.brain_schema_versions "
            "(version, migration_name, migration_checksum, applied_by) "
            "VALUES (1, %s, %s, %s)",
            (_MIGRATION_NAME, _schema_checksum(), migration_identity),
        )
    if len(versions) < 2:
        cur.execute(_schema_v2_bytes().decode("utf-8"))
        _apply_acl_contract(cur, migration_identity)
        cur.execute(
            "INSERT INTO public.brain_schema_versions "
            "(version, migration_name, migration_checksum, applied_by) "
            "VALUES (2, %s, %s, %s)",
            (_MIGRATION_V2_NAME, _schema_v2_checksum(), migration_identity),
        )
    cur.execute(_schema_v3_bytes().decode("utf-8"))
    _apply_acl_contract(cur, migration_identity)
    cur.execute(
        "INSERT INTO public.brain_schema_versions "
        "(version, migration_name, migration_checksum, applied_by) "
        "VALUES (3, %s, %s, %s)",
        (_MIGRATION_V3_NAME, _schema_v3_checksum(), migration_identity),
    )
    _verify_contract(cur)


def ensure_brain_schema_v1(
    conn,
    *,
    lock_timeout_seconds: float = _LOCK_TIMEOUT_SECONDS,
) -> None:
    """Install every immutable brain schema migration, or verify the latest contract."""
    _require_idle(conn)
    with conn.transaction():
        with conn.cursor() as cur:
            ensure_brain_schema_v1_in_transaction(cur, lock_timeout_seconds=lock_timeout_seconds)


def verify_brain_schema_v4_in_transaction(cur) -> None:
    """Verify the V1-V4 contract using the caller's active transaction."""
    require_pg18_authority_catalog(cur)
    _verify_contract(cur)
    _verify_v4_contract(cur)


def verify_brain_schema_v4(conn) -> None:
    """Verify the V1-V4 immutable migration ledger and V4 authority boundary."""
    _require_idle(conn)
    with conn.transaction():
        with conn.cursor() as cur:
            verify_brain_schema_v4_in_transaction(cur)


def ensure_brain_schema_v4_in_transaction(
    cur,
    *,
    lock_timeout_seconds: float = _LOCK_TIMEOUT_SECONDS,
) -> None:
    """Install or verify the V1-V4 ledger using the caller's active transaction."""
    require_pg18_authority_catalog(cur)
    ensure_brain_schema_v1_in_transaction(cur, lock_timeout_seconds=lock_timeout_seconds)
    versions = _version_rows(cur)
    if len(versions) == 4:
        verify_brain_schema_v4_in_transaction(cur)
        return
    migration_identity = _activate_migration_identity(cur)
    _assert_existing_ownership(cur, migration_identity)
    _acquire_xact_lock(cur, lock_timeout_seconds)
    versions = _version_rows(cur)
    if len(versions) == 4:
        verify_brain_schema_v4_in_transaction(cur)
        return
    if len(versions) != 3:
        raise RuntimeError("brain schema v4 requires contiguous V1-V3 ledger")
    migration_v4 = _schema_v4_bytes()
    cur.execute(migration_v4.decode("utf-8"))
    migration_v4_checksum = hashlib.sha256(migration_v4).hexdigest()
    if migration_v4_checksum != _EXPECTED_V4_CHECKSUM:
        raise RuntimeError(
            f"immutable schema v4 file checksum mismatch: expected {_EXPECTED_V4_CHECKSUM}, "
            f"got {migration_v4_checksum}"
        )
    cur.execute(
        "INSERT INTO public.brain_schema_versions "
        "(version, migration_name, migration_checksum, applied_by) VALUES (4, %s, %s, %s)",
        (_MIGRATION_V4_NAME, _schema_v4_checksum(), migration_identity),
    )
    verify_brain_schema_v4_in_transaction(cur)


def ensure_brain_schema_v4(
    conn,
    *,
    lock_timeout_seconds: float = _LOCK_TIMEOUT_SECONDS,
) -> None:
    """Atomically install or verify every immutable V1-V4 brain schema migration."""
    _require_idle(conn)
    with conn.transaction():
        with conn.cursor() as cur:
            ensure_brain_schema_v4_in_transaction(cur, lock_timeout_seconds=lock_timeout_seconds)


def verify_brain_schema_v5_in_transaction(cur) -> None:
    """Verify V1-V5 and V5's superseding authority boundary in one transaction."""
    require_pg18_authority_catalog(cur)
    _verify_contract(cur)
    _verify_v5_contract(cur)


def verify_brain_schema_v5(conn) -> None:
    """Verify the immutable V1-V5 ledger and durable factual graph authority."""
    _require_idle(conn)
    with conn.transaction():
        with conn.cursor() as cur:
            verify_brain_schema_v5_in_transaction(cur)


def ensure_brain_schema_v5_in_transaction(
    cur,
    *,
    lock_timeout_seconds: float = _LOCK_TIMEOUT_SECONDS,
) -> None:
    """Install or verify the immutable V1-V5 ledger in the caller's transaction."""
    require_pg18_authority_catalog(cur)
    _assert_schema_v5_bytes_immutable()
    versions = _version_rows(cur)
    if len(versions) == 5:
        verify_brain_schema_v5_in_transaction(cur)
        return
    ensure_brain_schema_v4_in_transaction(cur, lock_timeout_seconds=lock_timeout_seconds)
    _ensure_v5_graph_authority_role(cur)
    migration_identity = _activate_migration_identity(cur)
    _assert_existing_ownership(cur, migration_identity)
    _acquire_xact_lock(cur, lock_timeout_seconds)
    versions = _version_rows(cur)
    if len(versions) == 5:
        verify_brain_schema_v5_in_transaction(cur)
        return
    if len(versions) != 4:
        raise RuntimeError("brain schema v5 requires contiguous V1-V4 ledger")
    cur.execute(_schema_v5_bytes().decode("utf-8"))
    cur.execute(
        "INSERT INTO public.brain_schema_versions "
        "(version,migration_name,migration_checksum,applied_by) VALUES (5,%s,%s,%s)",
        (_MIGRATION_V5_NAME, _EXPECTED_V5_CHECKSUM, migration_identity),
    )
    verify_brain_schema_v5_in_transaction(cur)


def ensure_brain_schema_v5(
    conn,
    *,
    lock_timeout_seconds: float = _LOCK_TIMEOUT_SECONDS,
) -> None:
    """Atomically install or verify every immutable V1-V5 migration."""
    _require_idle(conn)
    with conn.transaction():
        with conn.cursor() as cur:
            ensure_brain_schema_v5_in_transaction(cur, lock_timeout_seconds=lock_timeout_seconds)


def _verify_v6_contract(cur, *, upgraded_to_v7: bool = False, upgraded_to_v8: bool = False) -> None:
    versions = _version_rows(cur)
    expected_count = 8 if upgraded_to_v8 else (7 if upgraded_to_v7 else 6)
    if len(versions) != expected_count:
        raise RuntimeError("brain schema v6 verification requires the expected contiguous ledger")
    cur.execute(
        "SELECT c.relname, owner.rolname AS owner_name FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace "
        "JOIN pg_roles owner ON owner.oid=c.relowner "
        "WHERE n.nspname='public' AND c.relname=ANY(%s)",
        (["brain_artifact_authority_requests", "brain_artifact_authority_registrations"],),
    )
    owners = {row["relname"]: row["owner_name"] for row in cur.fetchall()}
    if owners != {
        "brain_artifact_authority_requests": "brain_artifact_authority_owner",
        "brain_artifact_authority_registrations": "brain_artifact_authority_owner",
    }:
        raise RuntimeError("brain schema v6 authority table ownership mismatch")
    expected_columns = {
        "brain_artifact_authority_requests": (
            ("request_id", "uuid", True), ("manifest_sha256", "text", True), ("purpose", "text", True),
            ("key_id", "text", True), ("issued_at", "timestamp with time zone", True),
            ("expires_at", "timestamp with time zone", True), ("destination_system_id", "text", True),
            ("destination_database_name", "text", True), ("artifact_count", "integer", True),
            ("receipt", "jsonb", True), ("registered_at", "timestamp with time zone", True),
        ),
        "brain_artifact_authority_registrations": (
            ("request_id", "uuid", True), ("artifact_ordinal", "integer", True),
            ("artifact_hash", "text", True), ("byte_length", "bigint", True), ("media_type", "text", True),
            ("backend", "text", True), ("bucket", "text", True), ("object_key", "text", True),
            ("provider_version_id", "text", True), ("provider_checksum", "text", True),
            ("encryption_mode", "text", True), ("encryption_key_id", "text", True),
            ("policy_source_id", "text", False), ("registered_at", "timestamp with time zone", True),
        ),
    }
    cur.execute(
        "SELECT c.relname,a.attname,format_type(a.atttypid,a.atttypmod) AS data_type,a.attnotnull "
        "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "JOIN pg_attribute a ON a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped "
        "WHERE n.nspname='public' AND c.relname=ANY(%s) ORDER BY c.relname,a.attnum",
        (list(expected_columns),),
    )
    actual_columns: dict[str, list[tuple[str, str, bool]]] = {name: [] for name in expected_columns}
    for column in cur.fetchall():
        actual_columns[column["relname"]].append(
            (column["attname"], column["data_type"], column["attnotnull"])
        )
    if {name: tuple(columns) for name, columns in actual_columns.items()} != expected_columns:
        raise RuntimeError("brain schema v6 column contract mismatch")
    cur.execute(
        "SELECT c.relname,con.contype,count(*) AS count FROM pg_constraint con "
        "JOIN pg_class c ON c.oid=con.conrelid JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname='public' AND c.relname=ANY(%s) GROUP BY c.relname,con.contype",
        (list(expected_columns),),
    )
    constraint_counts = {(row["relname"], row["contype"]): row["count"] for row in cur.fetchall()}
    if constraint_counts != {
        ("brain_artifact_authority_requests", "c"): 8,
        ("brain_artifact_authority_requests", "n"): 11,
        ("brain_artifact_authority_requests", "p"): 1,
        ("brain_artifact_authority_registrations", "c"): 11,
        ("brain_artifact_authority_registrations", "f"): 2,
        ("brain_artifact_authority_registrations", "n"): 13,
        ("brain_artifact_authority_registrations", "p"): 1,
        ("brain_artifact_authority_registrations", "u"): 1,
    }:
        raise RuntimeError(f"brain schema v6 constraint contract mismatch: {constraint_counts!r}")
    cur.execute(
        "SELECT c.relname,t.tgname,p.proname AS function_name,t.tgenabled "
        "FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid "
        "JOIN pg_namespace n ON n.oid=c.relnamespace JOIN pg_proc p ON p.oid=t.tgfoid "
        "WHERE NOT t.tgisinternal AND n.nspname='public' AND c.relname=ANY(%s) "
        "ORDER BY c.relname,t.tgname",
        (list(expected_columns),),
    )
    triggers = [(row["relname"], row["tgname"], row["function_name"], row["tgenabled"]) for row in cur.fetchall()]
    expected_triggers = sorted(
        (relation, f"{relation}_append_only", "brain_reject_mutation", "O")
        for relation in expected_columns
    ) + sorted(
        (relation, f"{relation}_no_truncate", "brain_reject_mutation", "O")
        for relation in expected_columns
    )
    if sorted(triggers) != sorted(expected_triggers):
        raise RuntimeError("brain schema v6 trigger contract mismatch")
    cur.execute(
        "SELECT owner.rolname AS owner_name, p.prosecdef, p.proconfig "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "JOIN pg_roles owner ON owner.oid=p.proowner "
        "WHERE n.nspname='public' AND p.proname='brain_register_authoritative_artifact_manifest'"
    )
    function = cur.fetchone()
    if (
        function is None
        or function["owner_name"] != "brain_artifact_authority_owner"
        or not function["prosecdef"]
        or function["proconfig"] != ["search_path=pg_catalog, public"]
    ):
        raise RuntimeError("brain schema v6 registration function contract mismatch")
    cur.execute(
        "SELECT p.proname,p.prosrc FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "WHERE n.nspname='public' AND p.proname=ANY(%s)",
        ([
            "brain_register_authoritative_artifact_manifest",
            "brain_artifact_is_authoritative",
            "brain_check_policy_lifecycle",
        ],),
    )
    bodies = {row["proname"]: _raw_function_body_fingerprint(row["prosrc"]) for row in cur.fetchall()}
    if not (upgraded_to_v7 or upgraded_to_v8) and bodies != {
        name: _v6_function_body_fingerprint(name)
        for name in (
            "brain_register_authoritative_artifact_manifest",
            "brain_artifact_is_authoritative",
            "brain_check_policy_lifecycle",
        )
    }:
        raise RuntimeError("brain schema v6 function body contract mismatch")
    cur.execute(
        "SELECT owner.rolname AS owner_name FROM pg_proc p "
        "JOIN pg_namespace n ON n.oid=p.pronamespace JOIN pg_roles owner ON owner.oid=p.proowner "
        "WHERE n.nspname='public' AND p.proname='brain_check_policy_lifecycle'"
    )
    lifecycle = cur.fetchone()
    if lifecycle is None or lifecycle["owner_name"] != _MIGRATION_ROLE:
        raise RuntimeError("brain schema v6 lifecycle function owner mismatch")
    cur.execute(
        "SELECT owner.rolname AS owner_name,p.proconfig,p.prosrc,p.prosecdef FROM pg_proc p "
        "JOIN pg_namespace n ON n.oid=p.pronamespace JOIN pg_roles owner ON owner.oid=p.proowner "
        "WHERE n.nspname='public' AND p.proname='brain_artifact_is_authoritative'"
    )
    authority_predicate = cur.fetchone()
    if (
        authority_predicate is None
        or authority_predicate["owner_name"] != "brain_artifact_authority_owner"
        or not authority_predicate["prosecdef"]
        or authority_predicate["proconfig"] != ["search_path=pg_catalog, public"]
        or "brain_artifact_authority_registrations" not in authority_predicate["prosrc"]
    ):
        raise RuntimeError("brain schema v6 authority predicate contract mismatch")
    for role_name in ("brain_artifact_authority_owner", "brain_artifact_authority_writer"):
        cur.execute(
            "SELECT rolsuper, rolcreaterole, rolcreatedb, rolcanlogin FROM pg_roles WHERE rolname=%s",
            (role_name,),
        )
        role = cur.fetchone()
        if role is None or role["rolsuper"] or role["rolcreaterole"] or role["rolcreatedb"] or role["rolcanlogin"]:
            raise RuntimeError(f"brain schema v6 unsafe or missing role: {role_name}")
        cur.execute("SELECT has_schema_privilege(%s,'public','CREATE') AS allowed", (role_name,))
        if cur.fetchone()["allowed"]:
            raise RuntimeError(f"brain schema v6 role retains public schema CREATE: {role_name}")
    cur.execute(
        "SELECT has_function_privilege('brain_artifact_authority_writer', "
        "'public.brain_register_authoritative_artifact_manifest(uuid,text,text,text,timestamptz,timestamptz,text,text,jsonb)', "
        "'EXECUTE') AS writer_execute, EXISTS ("
        "SELECT 1 FROM pg_proc p CROSS JOIN LATERAL aclexplode(COALESCE(p.proacl,acldefault('f',p.proowner))) acl "
        "WHERE p.oid='public.brain_register_authoritative_artifact_manifest"
        "(uuid,text,text,text,timestamptz,timestamptz,text,text,jsonb)'::regprocedure "
        "AND acl.grantee=0 AND acl.privilege_type='EXECUTE') AS public_execute"
    )
    grants = cur.fetchone()
    if not grants["writer_execute"] or grants["public_execute"]:
        raise RuntimeError("brain schema v6 registration function ACL mismatch")
    cur.execute(
        "SELECT parent.rolname AS parent_role, member.rolname AS member_role "
        "FROM pg_auth_members membership JOIN pg_roles parent ON parent.oid=membership.roleid "
        "JOIN pg_roles member ON member.oid=membership.member "
        "WHERE parent.rolname=ANY(%s) OR member.rolname=ANY(%s) "
        "ORDER BY parent.rolname,member.rolname",
        (["brain_artifact_authority_owner", "brain_artifact_authority_writer"],) * 2,
    )
    memberships = [(row["parent_role"], row["member_role"]) for row in cur.fetchall()]
    if memberships != [("brain_artifact_authority_owner", _MIGRATION_ROLE)]:
        raise RuntimeError("brain schema v6 role membership contract mismatch")
    cur.execute(
        "SELECT has_table_privilege('brain_artifact_authority_owner','public.brain_artifacts','SELECT,INSERT') "
        "AS artifacts, has_table_privilege('brain_artifact_authority_owner',"
        "'public.brain_artifact_locations','SELECT,INSERT') AS locations, "
        "has_sequence_privilege('brain_artifact_authority_owner',"
        "'public.brain_artifact_locations_artifact_location_id_seq','USAGE,SELECT') AS identity_sequence"
    )
    owner_grants = cur.fetchone()
    if not all(owner_grants.values()):
        raise RuntimeError("brain schema v6 owner dependency ACL mismatch")
    cur.execute(
        "SELECT has_table_privilege('brain_artifact_authority_owner','public.brain_artifacts',"
        "'UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') AS artifacts_extra,"
        "has_table_privilege('brain_artifact_authority_owner','public.brain_artifact_locations',"
        "'UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') AS locations_extra,"
        "has_sequence_privilege('brain_artifact_authority_owner',"
        "'public.brain_artifact_locations_artifact_location_id_seq','UPDATE') AS sequence_update,"
        "has_table_privilege('brain_artifact_authority_writer','public.brain_artifact_authority_requests',"
        "'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') AS writer_requests,"
        "has_table_privilege('brain_artifact_authority_writer','public.brain_artifact_authority_registrations',"
        "'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') AS writer_registrations"
    )
    forbidden_grants = cur.fetchone()
    if any(forbidden_grants.values()):
        raise RuntimeError("brain schema v6 excessive table or sequence ACL")


def verify_brain_schema_v6_in_transaction(cur) -> None:
    """Verify V1-V6 and the immutable artifact registration authority."""
    require_pg18_authority_catalog(cur)
    _assert_current_schema_v6_bytes_immutable()
    _verify_v6_contract(cur)
    actual_catalog_hash = _catalog_contract_hash(cur, include_v5=True)
    expected_catalog_hash = _pg18_catalog_hash("v6")
    if actual_catalog_hash != expected_catalog_hash:
        raise RuntimeError(
            "brain schema v6 catalog contract hash mismatch: "
            f"expected {expected_catalog_hash}, got {actual_catalog_hash}"
        )


def verify_brain_schema_v6(conn) -> None:
    _require_idle(conn)
    with conn.transaction():
        with conn.cursor() as cur:
            verify_brain_schema_v6_in_transaction(cur)


def ensure_brain_schema_v6_in_transaction(
    cur,
    *,
    lock_timeout_seconds: float = _LOCK_TIMEOUT_SECONDS,
) -> None:
    require_pg18_authority_catalog(cur)
    _assert_schema_v6_bytes_immutable()
    versions = _version_rows(cur)
    if len(versions) == 6:
        verify_brain_schema_v6_in_transaction(cur)
        return
    ensure_brain_schema_v5_in_transaction(cur, lock_timeout_seconds=lock_timeout_seconds)
    migration_identity = _activate_migration_identity(cur)
    _acquire_xact_lock(cur, lock_timeout_seconds)
    versions = _version_rows(cur)
    if len(versions) == 6:
        verify_brain_schema_v6_in_transaction(cur)
        return
    if len(versions) != 5:
        raise RuntimeError("brain schema v6 requires contiguous V1-V5 ledger")
    cur.execute(_schema_v6_bytes().decode("utf-8"))
    cur.execute(
        "INSERT INTO public.brain_schema_versions "
        "(version,migration_name,migration_checksum,applied_by) VALUES (6,%s,%s,%s)",
        (_MIGRATION_V6_NAME, _CURRENT_V6_CHECKSUM, migration_identity),
    )
    verify_brain_schema_v6_in_transaction(cur)


def ensure_brain_schema_v6(
    conn,
    *,
    lock_timeout_seconds: float = _LOCK_TIMEOUT_SECONDS,
) -> None:
    _require_idle(conn)
    with conn.transaction():
        with conn.cursor() as cur:
            ensure_brain_schema_v6_in_transaction(cur, lock_timeout_seconds=lock_timeout_seconds)


def ensure_policy_partition(conn, policy_version: str) -> str:
    """Create and verify the owner-only LIST partition for one policy version."""
    if not isinstance(policy_version, str) or not policy_version.strip():
        raise ValueError("policy_version must be a non-empty string")
    _require_idle(conn)
    suffix = hashlib.sha256(policy_version.encode("utf-8")).hexdigest()[:16]
    partition_name = f"brain_job_decisions_policy_{suffix}"
    with conn.transaction():
        with conn.cursor() as cur:
            version = _version_row(cur)
            if version is None:
                raise RuntimeError("brain schema v1 must be installed before creating a policy partition")
            current_user = _activate_migration_identity(cur)
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))",
                ("applypilot:brain:policy-partition", policy_version),
            )
            cur.execute(
                "SELECT owner.rolname AS owner_name FROM pg_class c "
                "JOIN pg_roles owner ON owner.oid = c.relowner "
                "WHERE c.oid = 'public.brain_job_decisions'::regclass"
            )
            owner = cur.fetchone()["owner_name"]
            cur.execute(
                "SELECT %s = %s OR pg_has_role(%s, %s, 'MEMBER') AS allowed",
                (current_user, owner, current_user, owner),
            )
            if not cur.fetchone()["allowed"]:
                raise RuntimeError("policy partitions may only be created by the brain schema owner")
            cur.execute(
                "SELECT 1 FROM public.brain_decision_policies WHERE policy_version = %s",
                (policy_version,),
            )
            if cur.fetchone() is None:
                raise RuntimeError(f"unknown policy_version: {policy_version}")
            cur.execute(
                "SELECT c.oid::regclass::text AS relation FROM pg_inherits i "
                "JOIN pg_class c ON c.oid=i.inhrelid "
                "WHERE i.inhparent='public.brain_job_decisions'::regclass "
                "AND pg_get_expr(c.relpartbound, c.oid)='DEFAULT'"
            )
            default_partition = cur.fetchone()
            if default_partition is not None:
                cur.execute("SET LOCAL ROLE NONE")
                cur.execute(
                    sql.SQL("SELECT count(*) AS count FROM {} WHERE policy_version=%s").format(
                        sql.Identifier(*default_partition["relation"].split(".", 1))
                    ),
                    (policy_version,),
                )
                if cur.fetchone()["count"]:
                    raise RuntimeError(
                        f"default partition contains rows for policy_version {policy_version}; refusing to route silently"
                    )
                raise RuntimeError(
                    "default decision partition is forbidden; detach it before creating policy partitions"
                )
            cur.execute("SELECT to_regclass(%s) AS relation", (f"public.{partition_name}",))
            if cur.fetchone()["relation"] is None:
                cur.execute(
                    sql.SQL("CREATE TABLE {}.{} PARTITION OF {}.{} FOR VALUES IN ({})").format(
                        sql.Identifier("public"),
                        sql.Identifier(partition_name),
                        sql.Identifier("public"),
                        sql.Identifier("brain_job_decisions"),
                        sql.Literal(policy_version),
                    )
                )
            cur.execute(
                "SELECT t.tgname FROM pg_trigger t WHERE t.tgrelid=%s::regclass AND NOT t.tgisinternal",
                (f"public.{partition_name}",),
            )
            partition_triggers = {row["tgname"] for row in cur.fetchall()}
            for trigger_name, timing, granularity in (
                ("brain_job_decisions_register", "BEFORE INSERT", "FOR EACH ROW"),
                ("brain_job_decisions_append_only", "BEFORE UPDATE OR DELETE", "FOR EACH ROW"),
                ("brain_job_decisions_append_only_truncate", "BEFORE TRUNCATE", "FOR EACH STATEMENT"),
            ):
                if trigger_name not in partition_triggers:
                    function_name = (
                        "brain_register_decision" if trigger_name.endswith("register") else "brain_reject_mutation"
                    )
                    cur.execute(
                        sql.SQL("CREATE TRIGGER {} {} ON {}.{} {} EXECUTE FUNCTION public.{}()").format(
                            sql.Identifier(trigger_name),
                            sql.SQL(timing),
                            sql.Identifier("public"),
                            sql.Identifier(partition_name),
                            sql.SQL(granularity),
                            sql.Identifier(function_name),
                        )
                    )
            cur.execute(
                "SELECT DISTINCT COALESCE(role.rolname,'PUBLIC') AS grantee "
                "FROM pg_class c CROSS JOIN LATERAL aclexplode(c.relacl) acl "
                "LEFT JOIN pg_roles role ON role.oid=acl.grantee "
                "WHERE c.oid=%s::regclass AND acl.grantee<>c.relowner",
                (f"public.{partition_name}",),
            )
            for grant in cur.fetchall():
                grantee = sql.SQL("PUBLIC") if grant["grantee"] == "PUBLIC" else sql.Identifier(grant["grantee"])
                cur.execute(
                    sql.SQL("REVOKE ALL PRIVILEGES ON TABLE {}.{} FROM {}").format(
                        sql.Identifier("public"), sql.Identifier(partition_name), grantee
                    )
                )
            cur.execute(
                sql.SQL("GRANT SELECT ON TABLE {}.{} TO {}").format(
                    sql.Identifier("public"), sql.Identifier(partition_name), sql.Identifier(_VERIFIER_ROLE)
                )
            )
            cur.execute(
                "SELECT pg_get_expr(c.relpartbound, c.oid) AS bound, owner.rolname AS owner_name "
                "FROM pg_class c JOIN pg_roles owner ON owner.oid = c.relowner "
                "WHERE c.oid = %s::regclass",
                (f"public.{partition_name}",),
            )
            row = cur.fetchone()
            expected_bound = f"FOR VALUES IN ('{policy_version.replace(chr(39), chr(39) * 2)}')"
            if row is None or row["bound"] != expected_bound or row["owner_name"] != owner:
                raise RuntimeError("existing policy partition has an incompatible bound or owner")
            _verify_contract(cur)
    return partition_name


ensure_schema_v1 = ensure_brain_schema_v1
verify_schema_v1 = verify_brain_schema_v1
def _verify_v7_membership_contract(cur) -> None:
    cur.execute(
        "SELECT owner.oid,owner.rolname,owner.rolcanlogin,owner.rolinherit,"
        "owner.rolsuper,owner.rolcreatedb,owner.rolcreaterole,"
        "owner.rolreplication,owner.rolbypassrls "
        "FROM pg_database database JOIN pg_roles owner ON owner.oid=database.datdba "
        "WHERE database.datname=current_database()"
    )
    database_owner_role = cur.fetchone()
    database_owner = database_owner_role["rolname"]
    cur.execute(
        "SELECT parent.rolname AS parent_role,member.rolname AS member_role,"
        "grantor.oid AS grantor_oid,grantor.rolsuper AS grantor_is_superuser,"
        "membership.admin_option,"
        "membership.inherit_option,membership.set_option "
        "FROM pg_auth_members membership "
        "JOIN pg_roles parent ON parent.oid=membership.roleid "
        "JOIN pg_roles member ON member.oid=membership.member "
        "JOIN pg_roles grantor ON grantor.oid=membership.grantor "
        "WHERE parent.rolname=ANY(%s) "
        "ORDER BY parent.rolname,member.rolname,grantor.rolname",
        (["brain_artifact_authority_owner", "brain_artifact_authority_writer"],),
    )
    rows = cur.fetchall()
    exact = (
        len(rows) == 1
        and rows[0]["parent_role"] == "brain_artifact_authority_owner"
        and rows[0]["member_role"] == _MIGRATION_ROLE
        and rows[0]["grantor_oid"] == 10
        and rows[0]["grantor_is_superuser"] is True
        and rows[0]["admin_option"] is False
        and rows[0]["inherit_option"] is False
        and rows[0]["set_option"] is True
    )
    if not exact:
        raise RuntimeError(
            f"brain schema v7 exact authority membership contract mismatch: {rows!r}"
        )
    database_owner_is_bootstrap = database_owner_role["oid"] == 10
    if not database_owner_is_bootstrap:
        expected_attributes = {
            "rolcanlogin": True,
            "rolinherit": False,
            "rolsuper": False,
            "rolcreatedb": False,
            "rolcreaterole": False,
            "rolreplication": False,
            "rolbypassrls": False,
        }
        actual_attributes = {
            name: bool(database_owner_role[name]) for name in expected_attributes
        }
        if actual_attributes != expected_attributes:
            raise RuntimeError(
                "brain schema v7 dedicated database owner role attribute contract mismatch"
            )
        cur.execute(
            "SELECT parent.rolname AS parent_role,grantor.oid AS grantor_oid,"
            "grantor.rolsuper AS grantor_is_superuser,membership.admin_option,"
            "membership.inherit_option,membership.set_option "
            "FROM pg_auth_members membership "
            "JOIN pg_roles parent ON parent.oid=membership.roleid "
            "JOIN pg_roles member ON member.oid=membership.member "
            "JOIN pg_roles grantor ON grantor.oid=membership.grantor "
            "WHERE member.rolname=%s",
            (database_owner,),
        )
        provider_edges = cur.fetchall()
        provider_exact = (
            len(provider_edges) == 1
            and provider_edges[0]["parent_role"] == _MIGRATION_ROLE
            and provider_edges[0]["grantor_oid"] == 10
            and provider_edges[0]["grantor_is_superuser"] is True
            and provider_edges[0]["admin_option"] is False
            and provider_edges[0]["inherit_option"] is False
            and provider_edges[0]["set_option"] is True
        )
        if not provider_exact:
            raise RuntimeError(
                "brain schema v7 exact migrator-to-database-owner membership contract mismatch"
            )
    cur.execute(
        "WITH RECURSIVE closure(root_oid,member_oid,path) AS ("
        "SELECT root.oid,membership.member,ARRAY[root.oid,membership.member] "
        "FROM pg_roles root JOIN pg_auth_members membership ON membership.roleid=root.oid "
        "WHERE root.rolname=ANY(%s) UNION ALL "
        "SELECT closure.root_oid,membership.member,closure.path||membership.member "
        "FROM closure JOIN pg_auth_members membership ON membership.roleid=closure.member_oid "
        "WHERE NOT membership.member=ANY(closure.path)) "
        "SELECT root.rolname AS root_role,member.rolname AS member_role "
        "FROM closure JOIN pg_roles root ON root.oid=closure.root_oid "
        "JOIN pg_roles member ON member.oid=closure.member_oid "
        "ORDER BY root.rolname,member.rolname",
        (["brain_artifact_authority_owner", "brain_artifact_authority_writer"],),
    )
    reachable = {(row["root_role"], row["member_role"]) for row in cur.fetchall()}
    allowed = {("brain_artifact_authority_owner", _MIGRATION_ROLE)}
    if not database_owner_is_bootstrap:
        allowed.add(("brain_artifact_authority_owner", database_owner))
    if reachable != allowed:
        raise RuntimeError(
            f"brain schema v7 recursive authority membership contract mismatch: {reachable!r}"
        )

def _verify_v7_acl_contract(cur) -> None:
    cur.execute(
        "SELECT c.relname,COALESCE(grantee.rolname,'PUBLIC') AS grantee,acl.privilege_type "
        "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "CROSS JOIN LATERAL aclexplode(c.relacl) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE n.nspname='public' AND left(c.relname,6)='brain_' "
        "AND acl.grantee<>c.relowner"
    )
    invalid_relation_acls = []
    authority_dependencies = {
        ("brain_artifacts", "SELECT"),
        ("brain_artifacts", "INSERT"),
        ("brain_artifact_locations", "SELECT"),
        ("brain_artifact_locations", "INSERT"),
        ("brain_artifact_locations_artifact_location_id_seq", "SELECT"),
        ("brain_artifact_locations_artifact_location_id_seq", "USAGE"),
    }
    for row in cur.fetchall():
        allowed = row["grantee"] == _VERIFIER_ROLE and row["privilege_type"] == "SELECT"
        allowed = allowed or (
            row["grantee"] == _STATUS_ROLE
            and row["relname"] in _STATUS_READ_RELATIONS
            and row["privilege_type"] == "SELECT"
        )
        allowed = allowed or (
            row["grantee"] == _CANDIDATE_READER_ROLE
            and row["relname"] in _V4_READ_RELATIONS
            and row["privilege_type"] == "SELECT"
        )
        allowed = allowed or (
            row["grantee"] == "brain_artifact_authority_owner"
            and (row["relname"], row["privilege_type"]) in authority_dependencies
        )
        # V8: the retention ledger is owned by the migrator but read and appended
        # by the SECURITY DEFINER artifact-authority owner.
        allowed = allowed or (
            row["relname"] == "brain_artifact_retention_attestations"
            and row["grantee"] == "brain_artifact_authority_owner"
            and row["privilege_type"] in ("SELECT", "INSERT")
        )
        if not allowed:
            invalid_relation_acls.append(
                f"{row['relname']}->{row['grantee']}:{row['privilege_type']}"
            )
    if invalid_relation_acls:
        raise RuntimeError(
            "brain schema v7 relation ACL contract mismatch: "
            + ", ".join(sorted(invalid_relation_acls))
        )

    cur.execute(
        "SELECT p.proname,COALESCE(grantee.rolname,'PUBLIC') AS grantee,acl.privilege_type "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "CROSS JOIN LATERAL aclexplode(COALESCE(p.proacl,acldefault('f',p.proowner))) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE n.nspname='public' AND left(p.proname,6)='brain_' "
        "AND acl.grantee<>p.proowner"
    )
    invalid_function_acls = []
    migration_functions = {
        "brain_artifact_is_authoritative",
        "brain_snapshot_reference_provenance_matches",
        "brain_snapshot_binding_is_authoritative",
    }
    for row in cur.fetchall():
        allowed = row["privilege_type"] == "EXECUTE" and (
            (row["proname"] in _CONTROLLER_FUNCTIONS and row["grantee"] == _POLICY_CONTROLLER_ROLE)
            or (row["proname"] in _V5_POLICY_FUNCTIONS and row["grantee"] == _POLICY_CONTROLLER_ROLE)
            or (row["proname"] in _V5_GRAPH_AUTHORITY_FUNCTIONS and row["grantee"] == _GRAPH_AUTHORITY_ROLE)
            or (
                row["proname"] in {"brain_publish_v4_candidate", "brain_publish_v5_candidate"}
                and row["grantee"] == _CANDIDATE_WRITER_ROLE
            )
            or (
                row["proname"] == "brain_register_authoritative_artifact_manifest"
                and row["grantee"] == "brain_artifact_authority_writer"
            )
            or (
                row["proname"] == "brain_attest_artifact_retention"
                and row["grantee"] == "brain_artifact_authority_writer"
            )
            or (row["proname"] in migration_functions and row["grantee"] == _MIGRATION_ROLE)
        )
        if not allowed:
            invalid_function_acls.append(
                f"{row['proname']}()->{row['grantee']}:{row['privilege_type']}"
            )
    if invalid_function_acls:
        raise RuntimeError(
            "brain schema v7 function ACL contract mismatch: "
            + ", ".join(sorted(invalid_function_acls))
        )

def _verify_v7_topology_relation_contract(cur) -> int:
    cur.execute(
        "SELECT c.relkind,owner.rolname AS owner_name "
        "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "JOIN pg_roles owner ON owner.oid=c.relowner "
        "WHERE n.nspname='public' AND c.relname='brain_v7_topology_contract'"
    )
    relation = cur.fetchone()
    if relation != {"relkind": "r", "owner_name": _MIGRATION_ROLE}:
        raise RuntimeError(f"brain schema v7 topology relation mismatch: {relation!r}")
    cur.execute(
        "SELECT a.attname,format_type(a.atttypid,a.atttypmod) AS data_type,"
        "a.attnotnull,pg_get_expr(d.adbin,d.adrelid,true) AS default_expr "
        "FROM pg_attribute a LEFT JOIN pg_attrdef d "
        "ON d.adrelid=a.attrelid AND d.adnum=a.attnum "
        "WHERE a.attrelid='public.brain_v7_topology_contract'::regclass "
        "AND a.attnum>0 AND NOT a.attisdropped ORDER BY a.attnum"
    )
    if cur.fetchall() != [
        {
            "attname": "singleton_id",
            "data_type": "smallint",
            "attnotnull": True,
            "default_expr": None,
        },
        {
            "attname": "controller_role_oid",
            "data_type": "oid",
            "attnotnull": True,
            "default_expr": None,
        },
    ]:
        raise RuntimeError("brain schema v7 topology column contract mismatch")
    cur.execute(
        "SELECT conname,contype,convalidated,conenforced,"
        "pg_get_constraintdef(oid,true) AS definition "
        "FROM pg_constraint WHERE conrelid='public.brain_v7_topology_contract'::regclass "
        "ORDER BY conname"
    )
    constraints = cur.fetchall()
    if constraints != [
        {
            "conname": "brain_v7_topology_contract_controller_role_oid_not_null",
            "contype": "n",
            "convalidated": True,
            "conenforced": True,
            "definition": "NOT NULL controller_role_oid",
        },
        {
            "conname": "brain_v7_topology_contract_pkey",
            "contype": "p",
            "convalidated": True,
            "conenforced": True,
            "definition": "PRIMARY KEY (singleton_id)",
        },
        {
            "conname": "brain_v7_topology_contract_singleton_check",
            "contype": "c",
            "convalidated": True,
            "conenforced": True,
            "definition": "CHECK (singleton_id = 1)",
        },
        {
            "conname": "brain_v7_topology_contract_singleton_id_not_null",
            "contype": "n",
            "convalidated": True,
            "conenforced": True,
            "definition": "NOT NULL singleton_id",
        },
    ]:
        raise RuntimeError(
            f"brain schema v7 topology constraint contract mismatch: {constraints!r}"
        )
    cur.execute(
        "SELECT tgname,tgenabled,tgtype,p.proname AS function_name "
        "FROM pg_trigger t JOIN pg_proc p ON p.oid=t.tgfoid "
        "WHERE t.tgrelid='public.brain_v7_topology_contract'::regclass "
        "AND NOT t.tgisinternal ORDER BY tgname"
    )
    if cur.fetchall() != [
        {
            "tgname": "brain_v7_topology_contract_append_only",
            "tgenabled": "O",
            "tgtype": 27,
            "function_name": "brain_reject_mutation",
        },
        {
            "tgname": "brain_v7_topology_contract_append_only_truncate",
            "tgenabled": "O",
            "tgtype": 34,
            "function_name": "brain_reject_mutation",
        },
    ]:
        raise RuntimeError("brain schema v7 topology trigger contract mismatch")
    cur.execute(
        "SELECT COALESCE(grantee.rolname,'PUBLIC') AS grantee,acl.privilege_type "
        "FROM pg_class c CROSS JOIN LATERAL aclexplode(c.relacl) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE c.oid='public.brain_v7_topology_contract'::regclass "
        "AND acl.grantee<>c.relowner ORDER BY grantee,acl.privilege_type"
    )
    topology_acls = cur.fetchall()
    if topology_acls != [{"grantee": _VERIFIER_ROLE, "privilege_type": "SELECT"}]:
        raise RuntimeError(
            f"brain schema v7 topology ACL contract mismatch: {topology_acls!r}"
        )
    cur.execute(
        "SELECT singleton_id,controller_role_oid FROM public.brain_v7_topology_contract"
    )
    topology_rows = cur.fetchall()
    if len(topology_rows) != 1 or topology_rows[0]["singleton_id"] != 1:
        raise RuntimeError(
            f"brain schema v7 topology singleton contract mismatch: {topology_rows!r}"
        )
    return topology_rows[0]["controller_role_oid"]


def _resolve_safe_v7_controller_role_oid(cur, controller_role: str) -> int:
    if not controller_role:
        raise RuntimeError("brain schema v7 requires an expected controller role")
    cur.execute(
        "SELECT oid,rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,"
        "rolreplication,rolbypassrls FROM pg_roles WHERE rolname=%s",
        (controller_role,),
    )
    role = cur.fetchone()
    if not (
        role is not None
        and role["rolcanlogin"] is True
        and role["rolinherit"] is False
        and role["rolsuper"] is False
        and role["rolcreatedb"] is False
        and role["rolcreaterole"] is False
        and role["rolreplication"] is False
        and role["rolbypassrls"] is False
    ):
        raise RuntimeError(
            f"brain schema v7 expected controller role attributes mismatch: {controller_role!r}"
        )
    return role["oid"]


def _verify_v7_controller_membership_oid(cur, controller_role_oid: int) -> None:
    cur.execute(
        "SELECT member.oid AS member_oid,member.rolname AS member_role,"
        "member.rolcanlogin,member.rolinherit,member.rolsuper,member.rolcreatedb,"
        "member.rolcreaterole,member.rolreplication,member.rolbypassrls,"
        "grantor.oid AS grantor_oid,"
        "grantor.rolsuper AS grantor_is_superuser,membership.admin_option,"
        "membership.inherit_option,membership.set_option "
        "FROM pg_auth_members membership "
        "JOIN pg_roles parent ON parent.oid=membership.roleid "
        "JOIN pg_roles member ON member.oid=membership.member "
        "JOIN pg_roles grantor ON grantor.oid=membership.grantor "
        "WHERE parent.rolname=%s "
        "ORDER BY member.rolname,grantor.rolname",
        (_POLICY_CONTROLLER_ROLE,),
    )
    rows = cur.fetchall()
    exact = (
        len(rows) == 1
        and rows[0]["member_oid"] == controller_role_oid
        and rows[0]["rolcanlogin"] is True
        and rows[0]["rolinherit"] is False
        and rows[0]["rolsuper"] is False
        and rows[0]["rolcreatedb"] is False
        and rows[0]["rolcreaterole"] is False
        and rows[0]["rolreplication"] is False
        and rows[0]["rolbypassrls"] is False
        and rows[0]["grantor_oid"] == 10
        and rows[0]["grantor_is_superuser"] is True
        and rows[0]["admin_option"] is False
        and rows[0]["inherit_option"] is False
        and rows[0]["set_option"] is True
    )
    if not exact:
        raise RuntimeError(
            f"brain schema v7 exact controller membership contract mismatch: {rows!r}"
        )
    cur.execute(
        "WITH RECURSIVE closure(member_oid,path) AS ("
        "SELECT membership.member,ARRAY[parent.oid,membership.member] "
        "FROM pg_roles parent JOIN pg_auth_members membership "
        "ON membership.roleid=parent.oid "
        "WHERE parent.rolname=%s UNION ALL "
        "SELECT membership.member,closure.path||membership.member "
        "FROM closure JOIN pg_auth_members membership "
        "ON membership.roleid=closure.member_oid "
        "WHERE NOT membership.member=ANY(closure.path)) "
        "SELECT DISTINCT member_oid FROM closure ORDER BY member_oid",
        (_POLICY_CONTROLLER_ROLE,),
    )
    closure = [row["member_oid"] for row in cur.fetchall()]
    if closure != [controller_role_oid]:
        raise RuntimeError(
            f"brain schema v7 controller membership closure mismatch: {closure!r}"
        )


def _verify_v7_controller_membership_contract(cur) -> None:
    controller_role_oid = _verify_v7_topology_relation_contract(cur)
    _verify_v7_controller_membership_oid(cur, controller_role_oid)


def _expected_v7_controller_role_oid(cur, controller_role: str) -> int:
    controller_role_oid = _resolve_safe_v7_controller_role_oid(cur, controller_role)
    _verify_v7_controller_membership_oid(cur, controller_role_oid)
    return controller_role_oid


def _require_existing_v7_controller_identity(cur, controller_role: str) -> int:
    controller_role_oid = _resolve_safe_v7_controller_role_oid(cur, controller_role)
    cur.execute(
        "SELECT singleton_id,controller_role_oid "
        "FROM public.brain_v7_topology_contract ORDER BY singleton_id"
    )
    rows = cur.fetchall()
    if not (
        len(rows) == 1
        and rows[0]["singleton_id"] == 1
        and rows[0]["controller_role_oid"] == controller_role_oid
    ):
        raise RuntimeError(
            "brain schema v7 supplied controller topology mismatch: "
            f"expected role {controller_role!r} OID {controller_role_oid}, got {rows!r}"
        )
    return controller_role_oid


def _verify_v7_inherited_contracts(cur) -> None:
    _verify_v7_membership_contract(cur)
    _verify_v7_controller_membership_contract(cur)
    _verify_v5_contract(cur)
    _verify_v6_contract(cur, upgraded_to_v7=True)
    _verify_v7_acl_contract(cur)
    cur.execute(
        "SELECT c.relname,owner.rolname AS owner_name FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace "
        "JOIN pg_roles owner ON owner.oid=c.relowner "
        "WHERE n.nspname='public' AND c.relname=ANY(%s)",
        (list(_RELATIONS) + sorted(_V7_RELATIONS),),
    )
    inherited_owners = {row["relname"]: row["owner_name"] for row in cur.fetchall()}
    invalid_owners = {
        name: inherited_owners.get(name)
        for name in [*list(_RELATIONS), *sorted(_V7_RELATIONS)]
        if inherited_owners.get(name) != _MIGRATION_ROLE
    }
    if invalid_owners:
        raise RuntimeError(
            f"brain schema v7 inherited relation ownership mismatch: {invalid_owners!r}"
        )

def _verify_v7_contract(cur, *, check_catalog_hash: bool = True, expected_version_count: int = 7) -> None:
    if check_catalog_hash:
        actual_catalog_hash = _catalog_contract_hash(cur, include_v5=True, include_v7=True)
        expected_catalog_hash = _pg18_catalog_hash("v7")
        if actual_catalog_hash != expected_catalog_hash:
            raise RuntimeError(
                "brain schema v7 catalog contract hash mismatch: "
                f"expected {expected_catalog_hash}, got {actual_catalog_hash}"
            )
    versions = _version_rows(cur)
    if len(versions) != expected_version_count:
        raise RuntimeError("brain schema v7 verification requires a contiguous V1-V7 ledger")
    version = versions[6]
    if (
        version["version"] != 7 or version["migration_name"] != _MIGRATION_V7_NAME
        or version["migration_checksum"] != _EXPECTED_V7_CHECKSUM
        or version["applied_by"] != _MIGRATION_ROLE
    ):
        raise RuntimeError("brain schema v7 ledger contract mismatch")
    cur.execute(
        "SELECT p.proname,owner.rolname AS owner_name,p.prosecdef,p.proconfig,p.prosrc "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "JOIN pg_roles owner ON owner.oid=p.proowner "
        "WHERE n.nspname='public' AND p.proname=ANY(%s)",
        (["brain_register_authoritative_artifact_manifest", "brain_register_authoritative_artifact_manifest_v6"],),
    )
    functions = {row["proname"]: row for row in cur.fetchall()}
    wrapper = functions.get("brain_register_authoritative_artifact_manifest")
    legacy = functions.get("brain_register_authoritative_artifact_manifest_v6")
    if (
        wrapper is None or legacy is None
        or wrapper["owner_name"] != "brain_artifact_authority_owner"
        or legacy["owner_name"] != "brain_artifact_authority_owner"
        or not wrapper["prosecdef"] or not legacy["prosecdef"]
        or wrapper["proconfig"] != ["search_path=pg_catalog, public"]
        or legacy["proconfig"] != ["search_path=pg_catalog, public"]
        or "pg_advisory_xact_lock" not in wrapper["prosrc"]
        or "replay destination mismatch" not in wrapper["prosrc"]
    ):
        raise RuntimeError("brain schema v7 registration function contract mismatch")
    cur.execute(
        "SELECT p.proname,COALESCE(grantee.rolname,'PUBLIC') AS grantee,acl.privilege_type "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "CROSS JOIN LATERAL aclexplode(COALESCE(p.proacl,acldefault('f',p.proowner))) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE n.nspname='public' AND p.proname=ANY(%s) AND acl.grantee<>p.proowner",
        (["brain_register_authoritative_artifact_manifest", "brain_register_authoritative_artifact_manifest_v6"],),
    )
    grants = {(row["proname"], row["grantee"], row["privilege_type"]) for row in cur.fetchall()}
    if grants != {(
        "brain_register_authoritative_artifact_manifest",
        "brain_artifact_authority_writer",
        "EXECUTE",
    )}:
        raise RuntimeError("brain schema v7 function ACL contract mismatch")
    cur.execute(
        "SELECT p.proname,owner.rolname AS owner_name,p.prosecdef,p.proconfig,p.prosrc "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "JOIN pg_roles owner ON owner.oid=p.proowner "
        "WHERE n.nspname='public' AND p.proname=ANY(%s)",
        ([
            "brain_artifact_is_authoritative",
            "brain_snapshot_reference_provenance_matches",
            "brain_snapshot_binding_is_authoritative",
            "brain_check_policy_lifecycle",
        ],),
    )
    support = {row["proname"]: row for row in cur.fetchall()}
    expected_support = {
        "brain_artifact_is_authoritative": ("brain_artifact_authority_owner", True),
        "brain_snapshot_reference_provenance_matches": ("brain_artifact_authority_owner", False),
        "brain_snapshot_binding_is_authoritative": ("brain_artifact_authority_owner", True),
        "brain_check_policy_lifecycle": (_MIGRATION_ROLE, False),
    }
    if set(support) != set(expected_support):
        raise RuntimeError("brain schema v7 support function set mismatch")
    for name, (owner_name, security_definer) in expected_support.items():
        function = support[name]
        if function["owner_name"] != owner_name or function["prosecdef"] is not security_definer:
            raise RuntimeError(f"brain schema v7 support function ownership mismatch: {name}")
    for name in (
        "brain_artifact_is_authoritative",
        "brain_snapshot_reference_provenance_matches",
        "brain_snapshot_binding_is_authoritative",
    ):
        if support[name]["proconfig"] != ["search_path=pg_catalog, public"]:
            raise RuntimeError(f"brain schema v7 support function search_path mismatch: {name}")
    if (
        "applypilot.policy.snapshot-reference" not in support["brain_snapshot_reference_provenance_matches"]["prosrc"]
        or "brain_artifact_authority_registrations" not in support["brain_snapshot_binding_is_authoritative"]["prosrc"]
        or "brain_snapshot_binding_is_authoritative" not in support["brain_check_policy_lifecycle"]["prosrc"]
    ):
        raise RuntimeError("brain schema v7 snapshot support function body mismatch")
    cur.execute(
        "SELECT p.proname,COALESCE(grantee.rolname,'PUBLIC') AS grantee,acl.privilege_type "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "CROSS JOIN LATERAL aclexplode(COALESCE(p.proacl,acldefault('f',p.proowner))) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE n.nspname='public' AND p.proname=ANY(%s) AND acl.grantee<>p.proowner",
        ([
            "brain_artifact_is_authoritative",
            "brain_snapshot_reference_provenance_matches",
            "brain_snapshot_binding_is_authoritative",
        ],),
    )
    support_grants = {
        (row["proname"], row["grantee"], row["privilege_type"]) for row in cur.fetchall()
    }
    if support_grants != {
        ("brain_artifact_is_authoritative", _MIGRATION_ROLE, "EXECUTE"),
        ("brain_snapshot_reference_provenance_matches", _MIGRATION_ROLE, "EXECUTE"),
        ("brain_snapshot_binding_is_authoritative", _MIGRATION_ROLE, "EXECUTE"),
    }:
        raise RuntimeError("brain schema v7 support function ACL mismatch")
    cur.execute(
        "SELECT c.relname,c.relkind,owner.rolname AS owner_name FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace JOIN pg_roles owner ON owner.oid=c.relowner "
        "WHERE n.nspname='public' AND c.relname=ANY(%s)",
        ([
            "brain_artifact_authority_requests",
            "brain_artifact_authority_registrations",
            "brain_artifact_locations_artifact_location_id_seq",
        ],),
    )
    relations = {
        row["relname"]: (row["relkind"], row["owner_name"]) for row in cur.fetchall()
    }
    if relations != {
        "brain_artifact_authority_requests": ("r", "brain_artifact_authority_owner"),
        "brain_artifact_authority_registrations": ("r", "brain_artifact_authority_owner"),
        "brain_artifact_locations_artifact_location_id_seq": ("S", _MIGRATION_ROLE),
    }:
        raise RuntimeError("brain schema v7 authority relation ownership mismatch")
    cur.execute(
        "SELECT c.relname,grantee.rolname AS grantee,acl.privilege_type "
        "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "CROSS JOIN LATERAL aclexplode(c.relacl) acl "
        "JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE n.nspname='public' AND c.relname=ANY(%s) "
        "AND grantee.rolname=ANY(%s) AND acl.grantee<>c.relowner",
        (["brain_artifacts", "brain_artifact_locations", "brain_artifact_locations_artifact_location_id_seq"],
         ["brain_artifact_authority_owner", "brain_artifact_authority_writer"]),
    )
    dependency_grants = {
        (row["relname"], row["grantee"], row["privilege_type"]) for row in cur.fetchall()
    }
    if dependency_grants != {
        ("brain_artifacts", "brain_artifact_authority_owner", "SELECT"),
        ("brain_artifacts", "brain_artifact_authority_owner", "INSERT"),
        ("brain_artifact_locations", "brain_artifact_authority_owner", "SELECT"),
        ("brain_artifact_locations", "brain_artifact_authority_owner", "INSERT"),
        ("brain_artifact_locations_artifact_location_id_seq", "brain_artifact_authority_owner", "SELECT"),
        ("brain_artifact_locations_artifact_location_id_seq", "brain_artifact_authority_owner", "USAGE"),
    }:
        raise RuntimeError("brain schema v7 authority dependency ACL mismatch")
    cur.execute(
        "SELECT grantee.rolname AS grantee,acl.privilege_type FROM pg_namespace n "
        "CROSS JOIN LATERAL aclexplode(COALESCE(n.nspacl,acldefault('n',n.nspowner))) acl "
        "JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE n.nspname='public' AND grantee.rolname=ANY(%s)",
        (["brain_artifact_authority_owner", "brain_artifact_authority_writer"],),
    )
    schema_grants = {(row["grantee"], row["privilege_type"]) for row in cur.fetchall()}
    if schema_grants != {
        ("brain_artifact_authority_owner", "USAGE"),
        ("brain_artifact_authority_writer", "USAGE"),
    }:
        raise RuntimeError("brain schema v7 authority schema ACL mismatch")
    cur.execute(
        "SELECT 1 FROM pg_default_acl defaults "
        "CROSS JOIN LATERAL aclexplode(defaults.defaclacl) acl "
        "LEFT JOIN pg_roles owner ON owner.oid=defaults.defaclrole "
        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE owner.rolname=ANY(%s) OR grantee.rolname=ANY(%s) LIMIT 1",
        (["brain_artifact_authority_owner", "brain_artifact_authority_writer"],) * 2,
    )
    if cur.fetchone() is not None:
        raise RuntimeError("brain schema v7 authority default ACL leakage")


def verify_brain_schema_v7_in_transaction(cur) -> None:
    require_pg18_authority_catalog(cur)
    _assert_current_schema_v6_bytes_immutable()
    _assert_schema_v7_bytes_immutable()
    _verify_v7_inherited_contracts(cur)
    _verify_v7_contract(cur)


def verify_brain_schema_v7(conn) -> None:
    _require_idle(conn)
    with conn.transaction():
        with conn.cursor() as cur:
            verify_brain_schema_v7_in_transaction(cur)


def ensure_brain_schema_v7_in_transaction(
    cur,
    *,
    lock_timeout_seconds: float = _LOCK_TIMEOUT_SECONDS,
    expected_controller_role: str | None = None,
) -> None:
    require_pg18_authority_catalog(cur)
    _assert_current_schema_v6_bytes_immutable()
    _assert_schema_v7_bytes_immutable()
    versions = _version_rows(cur)
    if len(versions) == 7:
        if expected_controller_role is not None:
            _require_existing_v7_controller_identity(cur, expected_controller_role)
        verify_brain_schema_v7_in_transaction(cur)
        return
    if expected_controller_role is None:
        raise RuntimeError(
            "brain schema v7 first installation requires expected_controller_role"
        )
    expected_controller_role_oid = _resolve_safe_v7_controller_role_oid(
        cur, expected_controller_role
    )
    if len(versions) < 5:
        ensure_brain_schema_v5_in_transaction(cur, lock_timeout_seconds=lock_timeout_seconds)
    migration_identity = _activate_migration_identity(cur)
    _acquire_xact_lock(cur, lock_timeout_seconds)
    versions = _version_rows(cur)
    if len(versions) == 5:
        cur.execute(_schema_v6_bytes().decode("utf-8"))
        cur.execute(
            "INSERT INTO public.brain_schema_versions "
            "(version,migration_name,migration_checksum,applied_by) VALUES (6,%s,%s,%s)",
            (_MIGRATION_V6_NAME, _CURRENT_V6_CHECKSUM, migration_identity),
        )
        versions = [*versions, {"version": 6}]
    if len(versions) != 6:
        raise RuntimeError("brain schema v7 requires contiguous V1-V6 ledger")
    expected_controller_role_oid = _expected_v7_controller_role_oid(
        cur, expected_controller_role
    )
    cur.execute("GRANT CREATE ON SCHEMA public TO brain_artifact_authority_owner")
    cur.execute(_schema_v7_bytes().decode("utf-8"))
    cur.execute(
        "INSERT INTO public.brain_v7_topology_contract "
        "(singleton_id,controller_role_oid) VALUES (1,%s)",
        (expected_controller_role_oid,),
    )
    cur.execute("REVOKE CREATE ON SCHEMA public FROM brain_artifact_authority_owner")
    cur.execute(
        "INSERT INTO public.brain_schema_versions "
        "(version,migration_name,migration_checksum,applied_by) VALUES (7,%s,%s,%s)",
        (_MIGRATION_V7_NAME, _EXPECTED_V7_CHECKSUM, migration_identity),
    )
    verify_brain_schema_v7_in_transaction(cur)


def ensure_brain_schema_v7(
    conn,
    *,
    lock_timeout_seconds: float = _LOCK_TIMEOUT_SECONDS,
    expected_controller_role: str | None = None,
) -> None:
    _require_idle(conn)
    with conn.transaction():
        with conn.cursor() as cur:
            ensure_brain_schema_v7_in_transaction(
                cur,
                lock_timeout_seconds=lock_timeout_seconds,
                expected_controller_role=expected_controller_role,
            )


def _verify_v8_inherited_contracts(cur) -> None:
    _verify_v7_membership_contract(cur)
    _verify_v7_controller_membership_contract(cur)
    _verify_v5_contract(cur)
    _verify_v6_contract(cur, upgraded_to_v8=True)
    _verify_v7_acl_contract(cur)
    # Re-run the V7 structural contract (registration functions, snapshot support,
    # authority relations) but skip its catalog hash: V8 adds objects that would
    # legitimately change the V7-scoped fingerprint.
    _verify_v7_contract(cur, check_catalog_hash=False, expected_version_count=8)
    cur.execute(
        "SELECT c.relname,owner.rolname AS owner_name FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace "
        "JOIN pg_roles owner ON owner.oid=c.relowner "
        "WHERE n.nspname='public' AND c.relname=ANY(%s)",
        (list(_RELATIONS) + sorted(_V7_RELATIONS) + sorted(_V8_RELATIONS),),
    )
    inherited_owners = {row["relname"]: row["owner_name"] for row in cur.fetchall()}
    invalid_owners = {
        name: inherited_owners.get(name)
        for name in [*list(_RELATIONS), *sorted(_V7_RELATIONS), *sorted(_V8_RELATIONS)]
        if inherited_owners.get(name) != _MIGRATION_ROLE
    }
    if invalid_owners:
        raise RuntimeError(
            f"brain schema v8 inherited relation ownership mismatch: {invalid_owners!r}"
        )


def _verify_v8_contract(cur) -> None:
    actual_catalog_hash = _catalog_contract_hash(
        cur, include_v5=True, include_v7=True, include_v8=True
    )
    expected_catalog_hash = _pg18_catalog_hash("v8")
    if actual_catalog_hash != expected_catalog_hash:
        raise RuntimeError(
            "brain schema v8 catalog contract hash mismatch: "
            f"expected {expected_catalog_hash}, got {actual_catalog_hash}"
        )
    versions = _version_rows(cur)
    if len(versions) != 8:
        raise RuntimeError("brain schema v8 verification requires a contiguous V1-V8 ledger")
    version = versions[7]
    if (
        version["version"] != 8 or version["migration_name"] != _MIGRATION_V8_NAME
        or version["migration_checksum"] != _EXPECTED_V8_CHECKSUM
        or version["applied_by"] != _MIGRATION_ROLE
    ):
        raise RuntimeError("brain schema v8 ledger contract mismatch")
    # The early V1-V3 ledger checksums are not pinned by _version_rows; the deep V8
    # verifier reconciles them against the committed migration bytes (accepting the
    # published legacy V1 variant) so a historical checksum substituted over this
    # catalog is still rejected.
    accepted_early = {
        1: {_schema_checksum(), *_LEGACY_V1_CHECKSUMS},
        2: {_schema_v2_checksum()},
        3: {_schema_v3_checksum()},
    }
    for early_version, accepted in accepted_early.items():
        recorded = versions[early_version - 1]["migration_checksum"]
        if recorded not in accepted:
            raise RuntimeError(
                f"brain schema v8 ledger records an unrecognized version {early_version} checksum: {recorded}"
            )

    # Durable database incarnation: singleton, server-generated, append-only.
    cur.execute(
        "SELECT c.relkind,owner.rolname AS owner_name FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace JOIN pg_roles owner ON owner.oid=c.relowner "
        "WHERE n.nspname='public' AND c.relname='applypilot_database_identity'"
    )
    identity_relation = cur.fetchone()
    if identity_relation != {"relkind": "r", "owner_name": _MIGRATION_ROLE}:
        raise RuntimeError(f"brain schema v8 identity relation mismatch: {identity_relation!r}")
    cur.execute(
        "SELECT a.attname,format_type(a.atttypid,a.atttypmod) AS data_type,a.attnotnull "
        "FROM pg_attribute a WHERE a.attrelid='public.applypilot_database_identity'::regclass "
        "AND a.attnum>0 AND NOT a.attisdropped ORDER BY a.attnum"
    )
    if [
        (row["attname"], row["data_type"], bool(row["attnotnull"])) for row in cur.fetchall()
    ] != [
        ("singleton_id", "smallint", True),
        ("database_incarnation_id", "uuid", True),
        ("initialized_at", "timestamp with time zone", True),
    ]:
        raise RuntimeError("brain schema v8 identity column contract mismatch")
    cur.execute(
        "SELECT database_incarnation_id FROM public.applypilot_database_identity "
        "WHERE singleton_id=1"
    )
    identity_rows = cur.fetchall()
    if len(identity_rows) != 1 or identity_rows[0]["database_incarnation_id"] is None:
        raise RuntimeError("brain schema v8 database incarnation is not a non-null singleton")
    cur.execute("SELECT count(*) AS count FROM public.applypilot_database_identity")
    if cur.fetchone()["count"] != 1:
        raise RuntimeError("brain schema v8 database incarnation must be a single row")
    cur.execute(
        "SELECT t.tgname,t.tgtype,p.proname AS function_name FROM pg_trigger t "
        "JOIN pg_proc p ON p.oid=t.tgfoid "
        "WHERE t.tgrelid='public.applypilot_database_identity'::regclass "
        "AND NOT t.tgisinternal ORDER BY t.tgname"
    )
    if [
        (row["tgname"], row["tgtype"], row["function_name"]) for row in cur.fetchall()
    ] != [
        ("applypilot_database_identity_append_only", 27, "brain_reject_mutation"),
        ("applypilot_database_identity_append_only_truncate", 34, "brain_reject_mutation"),
    ]:
        raise RuntimeError("brain schema v8 identity append-only trigger contract mismatch")
    cur.execute(
        "SELECT COALESCE(grantee.rolname,'PUBLIC') AS grantee,acl.privilege_type "
        "FROM pg_class c CROSS JOIN LATERAL aclexplode(c.relacl) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE c.oid='public.applypilot_database_identity'::regclass "
        "AND acl.grantee<>c.relowner ORDER BY grantee,acl.privilege_type"
    )
    if [
        (row["grantee"], row["privilege_type"]) for row in cur.fetchall()
    ] != [
        ("brain_artifact_authority_owner", "SELECT"),
        (_VERIFIER_ROLE, "SELECT"),
    ]:
        raise RuntimeError("brain schema v8 identity ACL contract mismatch")

    # Append-only storage-retention attestations.
    cur.execute(
        "SELECT c.relkind,owner.rolname AS owner_name FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace JOIN pg_roles owner ON owner.oid=c.relowner "
        "WHERE n.nspname='public' AND c.relname='brain_artifact_retention_attestations'"
    )
    retention_relation = cur.fetchone()
    if retention_relation != {"relkind": "r", "owner_name": _MIGRATION_ROLE}:
        raise RuntimeError(f"brain schema v8 retention relation mismatch: {retention_relation!r}")
    cur.execute(
        "SELECT con.contype,count(*) AS count FROM pg_constraint con "
        "WHERE con.conrelid='public.brain_artifact_retention_attestations'::regclass "
        "GROUP BY con.contype ORDER BY con.contype"
    )
    retention_constraints = {row["contype"]: row["count"] for row in cur.fetchall()}
    if retention_constraints.get("f", 0) != 1 or retention_constraints.get("u", 0) != 1 \
       or retention_constraints.get("p", 0) != 1 or retention_constraints.get("c", 0) < 5:
        raise RuntimeError(
            f"brain schema v8 retention constraint contract mismatch: {retention_constraints!r}"
        )
    cur.execute(
        "SELECT t.tgname,t.tgtype,p.proname AS function_name FROM pg_trigger t "
        "JOIN pg_proc p ON p.oid=t.tgfoid "
        "WHERE t.tgrelid='public.brain_artifact_retention_attestations'::regclass "
        "AND NOT t.tgisinternal ORDER BY t.tgname"
    )
    if [
        (row["tgname"], row["tgtype"], row["function_name"]) for row in cur.fetchall()
    ] != [
        ("brain_artifact_retention_attestations_append_only", 27, "brain_reject_mutation"),
        ("brain_artifact_retention_attestations_append_only_truncate", 34, "brain_reject_mutation"),
        ("brain_artifact_retention_attestations_validate", 7, "brain_check_retention_attestation"),
    ]:
        raise RuntimeError("brain schema v8 retention trigger contract mismatch")
    cur.execute(
        "SELECT COALESCE(grantee.rolname,'PUBLIC') AS grantee,acl.privilege_type "
        "FROM pg_class c CROSS JOIN LATERAL aclexplode(c.relacl) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE c.oid='public.brain_artifact_retention_attestations'::regclass "
        "AND acl.grantee<>c.relowner ORDER BY grantee,acl.privilege_type"
    )
    if [
        (row["grantee"], row["privilege_type"]) for row in cur.fetchall()
    ] != [
        ("brain_artifact_authority_owner", "INSERT"),
        ("brain_artifact_authority_owner", "SELECT"),
        (_VERIFIER_ROLE, "SELECT"),
    ]:
        raise RuntimeError("brain schema v8 retention ACL contract mismatch")

    # Authority predicate now requires an exact still-current retention attestation
    # bound to the current incarnation, and the retention writer is a locked-down
    # SECURITY DEFINER granted only to the artifact-authority writer.
    cur.execute(
        "SELECT p.proname,owner.rolname AS owner_name,p.prosecdef,p.proconfig,p.prosrc "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "JOIN pg_roles owner ON owner.oid=p.proowner "
        "WHERE n.nspname='public' AND p.proname=ANY(%s)",
        (["brain_artifact_is_authoritative", "brain_attest_artifact_retention",
          "brain_check_retention_attestation"],),
    )
    v8_functions = {row["proname"]: row for row in cur.fetchall()}
    predicate = v8_functions.get("brain_artifact_is_authoritative")
    if (
        predicate is None
        or predicate["owner_name"] != "brain_artifact_authority_owner"
        or not predicate["prosecdef"]
        or predicate["proconfig"] != ["search_path=pg_catalog, public"]
        or "brain_artifact_retention_attestations" not in predicate["prosrc"]
        or "applypilot_database_identity" not in predicate["prosrc"]
        or "retain_until" not in predicate["prosrc"]
        or "now()" not in predicate["prosrc"]
        # The live verification lease must gate authority alongside retention.
        or "verification_lease_expires_at" not in predicate["prosrc"]
        # Authority is bound to the exact still-valid storage object the attestation
        # names, not merely to any verified location for the artifact hash.
        or "location.object_key = attestation.object_key" not in predicate["prosrc"]
        or "location.backend = attestation.backend" not in predicate["prosrc"]
        or "location.provider_checksum = attestation.provider_checksum" not in predicate["prosrc"]
    ):
        raise RuntimeError("brain schema v8 authority predicate contract mismatch")
    attest = v8_functions.get("brain_attest_artifact_retention")
    if (
        attest is None
        or attest["owner_name"] != "brain_artifact_authority_owner"
        or not attest["prosecdef"]
        or attest["proconfig"] != ["search_path=pg_catalog, public"]
        # Idempotent replay is scoped to the current incarnation and the observed
        # payload, and a same-identity conflict fails closed.
        or "database_incarnation_id = current_incarnation" not in attest["prosrc"]
        or "IS DISTINCT FROM brain_attest_artifact_retention.verified_at" not in attest["prosrc"]
        or "verification_lease_id" not in attest["prosrc"]
        or "conflicting retention attestation replay" not in attest["prosrc"]
        or "unique_violation" not in attest["prosrc"]
    ):
        raise RuntimeError("brain schema v8 retention writer contract mismatch")
    validator = v8_functions.get("brain_check_retention_attestation")
    if (
        validator is None
        or validator["owner_name"] != _MIGRATION_ROLE
        # verified_at must be non-future and bound to the location observation.
        or "cannot be in the future" not in validator["prosrc"]
        or "location.verified_at <= NEW.verified_at" not in validator["prosrc"]
    ):
        raise RuntimeError("brain schema v8 retention validator contract mismatch")
    # The lease-within-retention guard must be present as an enforced CHECK so an
    # attestation whose lease outlives its retention deadline is rejected at write time.
    cur.execute(
        "SELECT pg_get_constraintdef(con.oid) AS definition FROM pg_constraint con "
        "WHERE con.conrelid='public.brain_artifact_retention_attestations'::regclass "
        "AND con.conname='brain_artifact_retention_attestations_lease_within_retention'"
    )
    lease_constraint = cur.fetchone()
    if (
        lease_constraint is None
        or "verification_lease_expires_at <= retain_until" not in lease_constraint["definition"]
    ):
        raise RuntimeError("brain schema v8 lease-within-retention constraint contract mismatch")
    cur.execute(
        "SELECT p.proname,COALESCE(grantee.rolname,'PUBLIC') AS grantee,acl.privilege_type "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "CROSS JOIN LATERAL aclexplode(COALESCE(p.proacl,acldefault('f',p.proowner))) acl "
        "LEFT JOIN pg_roles grantee ON grantee.oid=acl.grantee "
        "WHERE n.nspname='public' AND p.proname='brain_attest_artifact_retention' "
        "AND acl.grantee<>p.proowner"
    )
    if {(row["grantee"], row["privilege_type"]) for row in cur.fetchall()} != {
        ("brain_artifact_authority_writer", "EXECUTE")
    }:
        raise RuntimeError("brain schema v8 retention writer ACL mismatch")

    # Parity definition version 2 contains exactly the 16 canonical projection keys.
    cur.execute(
        "SELECT check_key FROM public.brain_parity_definitions WHERE definition_version=%s "
        "ORDER BY check_key",
        (_V8_PARITY_DEFINITION_VERSION,),
    )
    actual_keys = tuple(row["check_key"] for row in cur.fetchall())
    if actual_keys != tuple(sorted(_V8_PARITY_CHECK_KEYS)):
        raise RuntimeError(
            "brain schema v8 parity definition version 2 key contract mismatch: "
            f"{actual_keys!r}"
        )


def verify_brain_schema_v8_in_transaction(cur) -> None:
    require_pg18_authority_catalog(cur)
    _assert_current_schema_v6_bytes_immutable()
    _assert_schema_v7_bytes_immutable()
    _assert_schema_v8_bytes_immutable()
    _verify_v8_inherited_contracts(cur)
    _verify_v8_contract(cur)


def verify_brain_schema_v8(conn) -> None:
    _require_idle(conn)
    with conn.transaction():
        with conn.cursor() as cur:
            verify_brain_schema_v8_in_transaction(cur)


def ensure_brain_schema_v8_in_transaction(
    cur,
    *,
    lock_timeout_seconds: float = _LOCK_TIMEOUT_SECONDS,
    expected_controller_role: str | None = None,
) -> None:
    require_pg18_authority_catalog(cur)
    _assert_current_schema_v6_bytes_immutable()
    _assert_schema_v7_bytes_immutable()
    _assert_schema_v8_bytes_immutable()
    versions = _version_rows(cur)
    if len(versions) == 8:
        verify_brain_schema_v8_in_transaction(cur)
        return
    if len(versions) < 7:
        ensure_brain_schema_v7_in_transaction(
            cur,
            lock_timeout_seconds=lock_timeout_seconds,
            expected_controller_role=expected_controller_role,
        )
    migration_identity = _activate_migration_identity(cur)
    _acquire_xact_lock(cur, lock_timeout_seconds)
    versions = _version_rows(cur)
    if len(versions) != 7:
        raise RuntimeError("brain schema v8 requires a contiguous V1-V7 ledger")
    cur.execute("GRANT CREATE ON SCHEMA public TO brain_artifact_authority_owner")
    cur.execute(_schema_v8_bytes().decode("utf-8"))
    cur.execute("REVOKE CREATE ON SCHEMA public FROM brain_artifact_authority_owner")
    cur.execute(
        "INSERT INTO public.brain_schema_versions "
        "(version,migration_name,migration_checksum,applied_by) VALUES (8,%s,%s,%s)",
        (_MIGRATION_V8_NAME, _EXPECTED_V8_CHECKSUM, migration_identity),
    )
    verify_brain_schema_v8_in_transaction(cur)


def ensure_brain_schema_v8(
    conn,
    *,
    lock_timeout_seconds: float = _LOCK_TIMEOUT_SECONDS,
    expected_controller_role: str | None = None,
) -> None:
    _require_idle(conn)
    with conn.transaction():
        with conn.cursor() as cur:
            ensure_brain_schema_v8_in_transaction(
                cur,
                lock_timeout_seconds=lock_timeout_seconds,
                expected_controller_role=expected_controller_role,
            )


ensure_schema_v4 = ensure_brain_schema_v4
verify_schema_v4 = verify_brain_schema_v4
ensure_schema_v5 = ensure_brain_schema_v5
verify_schema_v5 = verify_brain_schema_v5
ensure_schema_v6 = ensure_brain_schema_v6
verify_schema_v6 = verify_brain_schema_v6
ensure_schema_v7 = ensure_brain_schema_v7
verify_schema_v7 = verify_brain_schema_v7
ensure_schema_v8 = ensure_brain_schema_v8
verify_schema_v8 = verify_brain_schema_v8
