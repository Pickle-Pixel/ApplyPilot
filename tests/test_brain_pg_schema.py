from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import io
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
from threading import Barrier

import psycopg
import pytest
from psycopg import sql as pg_sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.pq import TransactionStatus
from psycopg.rows import dict_row

from applypilot.brain import schema
from applypilot.brain.artifact_authority import (
    PURPOSE as ARTIFACT_AUTHORITY_PURPOSE,
    RetentionAttestation,
    canonical_manifest_bytes,
    coordinate_artifact_authority,
    retention_signing_message,
    signing_message,
    validate_retention_attestation,
)
from applypilot.brain.lifecycle import arm_canary, authority_status, stop_canary
from applypilot.brain.policy_artifacts import compile_policy_artifacts
from applypilot.brain.sqlite_to_postgres import (
    _acquire_import_lock,
    _activate_controller,
    _insert_decisions,
    _insert_policies,
    _release_import_context,
    BrainImportError,
)
from applypilot.fleet import pg_roles, schema as fleet_schema
from conftest import (
    DISPOSABLE_CLUSTER_COMMENT_PREFIX,
    DISPOSABLE_CLUSTER_MARKER_ENV,
    DISPOSABLE_CLUSTER_MARKER_ROLE,
    DISPOSABLE_DATABASE_COMMENT,
    require_disposable_postgres,
)


def test_v6_clean_install_registration_replay_and_direct_dml_denial(brain_pg) -> None:
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        with connection.transaction():
            connection.execute("SET LOCAL ROLE brain_schema_migrator")
            migrations = (
                (1, schema._MIGRATION_NAME, schema._schema_bytes(), schema._schema_checksum()),
                (2, schema._MIGRATION_V2_NAME, schema._schema_v2_bytes(), schema._schema_v2_checksum()),
                (3, schema._MIGRATION_V3_NAME, schema._schema_v3_bytes(), schema._schema_v3_checksum()),
                (4, schema._MIGRATION_V4_NAME, schema._schema_v4_bytes(), schema._EXPECTED_V4_CHECKSUM),
                (5, schema._MIGRATION_V5_NAME, schema._schema_v5_bytes(), schema._EXPECTED_V5_CHECKSUM),
                (6, schema._MIGRATION_V6_NAME, schema._schema_v6_bytes(), schema._EXPECTED_V6_CHECKSUM),
            )
            for version, name, migration, checksum in migrations:
                statements = [migration.decode("utf-8")]
                if version == 6:
                    connection.execute("SET LOCAL ROLE NONE")
                    connection.execute(
                        "GRANT USAGE, CREATE ON SCHEMA public TO brain_schema_migrator WITH GRANT OPTION"
                    )
                    connection.execute("SET LOCAL ROLE brain_schema_migrator")
                    sql_text = statements[0]
                    markers = (
                        "CREATE FUNCTION public.brain_register_authoritative_artifact_manifest",
                        "ALTER FUNCTION public.brain_register_authoritative_artifact_manifest",
                        "CREATE OR REPLACE FUNCTION public.brain_check_policy_lifecycle",
                        "SET ROLE brain_artifact_authority_owner;",
                    )
                    offsets = [sql_text.index(marker) for marker in markers]
                    statements = [
                        sql_text[: offsets[0]],
                        sql_text[offsets[0] : offsets[1]],
                        sql_text[offsets[1] : offsets[2]],
                        sql_text[offsets[2] : offsets[3]],
                        sql_text[offsets[3] :],
                    ]
                try:
                    for section, statement in enumerate(statements, start=1):
                        connection.execute(statement)
                except Exception as exc:
                    position = getattr(getattr(exc, "diag", None), "statement_position", None)
                    raise AssertionError(
                        f"raw migration v{version} section {section} failed at position {position}"
                    ) from exc
                connection.execute(
                    "INSERT INTO brain_schema_versions "
                    "(version,migration_name,migration_checksum,applied_by) VALUES (%s,%s,%s,%s)",
                    (version, name, checksum, "brain_schema_migrator"),
                )
        assert connection.execute(
            "SELECT array_agg(version ORDER BY version) AS versions FROM public.brain_schema_versions"
        ).fetchone()["versions"] == [1, 2, 3, 4, 5, 6]
        system_id = connection.execute("SELECT system_identifier::text AS id FROM pg_control_system()").fetchone()["id"]
        database_name = connection.execute("SELECT current_database() AS name").fetchone()["name"]
        payload = json.dumps([{
            "artifact_hash": "a" * 64, "byte_length": 7, "media_type": "application/json", "backend": "s3",
            "bucket": "immutable", "object_key": "sha256/aa", "provider_version_id": "version-1",
            "provider_checksum": "checksum-1", "storage_immutable": True, "encryption_mode": "customer_managed",
            "encryption_key_id": "kms-key-1", "policy_source_id": "policy-1",
        }])
        params = (
            "11111111-1111-1111-1111-111111111111", "b" * 64,
            "brain-artifact-authority-registration-v1", "key-1", "2026-07-18T12:00:00Z",
            "2099-07-18T13:00:00Z", system_id, database_name, payload,
        )
        connection.execute("SET ROLE brain_artifact_authority_writer")
        statement = (
            "SELECT brain_register_authoritative_artifact_manifest("
            "%s::uuid,%s,%s,%s,%s::timestamptz,%s::timestamptz,%s,%s,%s::jsonb) AS receipt"
        )
        receipt = connection.execute(statement, params).fetchone()["receipt"]
        assert connection.execute(statement, params).fetchone()["receipt"] == receipt
        with pytest.raises(psycopg.errors.UniqueViolation, match="different manifest digest"):
            with connection.transaction():
                connection.execute(statement, (params[0], "c" * 64, *params[2:]))
        partial_payload = json.dumps(
            [
                dict(json.loads(payload)[0], artifact_hash="d" * 64),
                dict(json.loads(payload)[0], artifact_hash="not-a-digest"),
            ]
        )
        partial_request = "33333333-3333-3333-3333-333333333333"
        with pytest.raises(psycopg.errors.InvalidParameterValue, match="invalid artifact"):
            with connection.transaction():
                connection.execute(statement, (partial_request, "d" * 64, *params[2:-1], partial_payload))
        connection.execute("RESET ROLE")
        connection.execute(
            "INSERT INTO brain_artifacts "
            "(request_id,artifact_hash,media_type,byte_length,schema_version,location) "
            "VALUES ('unregistered','f' || repeat('f',63),'application/json',1,1,'memory://unregistered')"
        )
        connection.execute(
            "INSERT INTO brain_artifact_locations "
            "(artifact_hash,backend,bucket_or_container,object_key,provider_version_id,provider_checksum,"
            "storage_immutable,encryption_mode,encryption_key_id,durability,verified_at) "
            "VALUES (repeat('f',64),'s3','immutable','unregistered','version-x','checksum-x',TRUE,"
            "'customer_managed','kms-key-1','verified',now())"
        )
        assert connection.execute(
            "SELECT brain_artifact_is_authoritative(%s) AS registered,"
            "brain_artifact_is_authoritative(%s) AS unregistered",
            ("a" * 64, "f" * 64),
        ).fetchone() == {"registered": True, "unregistered": False}
        assert connection.execute(
            "SELECT (SELECT count(*) FROM brain_artifact_authority_requests WHERE request_id=%s) AS requests,"
            "(SELECT count(*) FROM brain_artifacts WHERE artifact_hash=%s) AS artifacts",
            (partial_request, "d" * 64),
        ).fetchone() == {"requests": 0, "artifacts": 0}
        connection.execute("SET ROLE brain_artifact_authority_writer")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with connection.transaction():
                connection.execute(
                    "INSERT INTO brain_artifact_authority_requests "
                    "(request_id,manifest_sha256,purpose,key_id,issued_at,expires_at,destination_system_id,"
                    "destination_database_name,artifact_count,receipt) VALUES "
                    "('22222222-2222-2222-2222-222222222222',%s,%s,'key',now(),now()+interval '1 hour',"
                    "%s,%s,1,'{}')",
                    ("c" * 64, "brain-artifact-authority-registration-v1", system_id, database_name),
                )
        connection.execute("RESET ROLE")
        with connection.cursor() as cursor:
            schema._verify_v6_contract(cursor)
        connection.commit()

        concurrent_payload = json.dumps([dict(json.loads(payload)[0], artifact_hash="e" * 64)])
        concurrent_params = (
            "44444444-4444-4444-4444-444444444444", "e" * 64, *params[2:-1], concurrent_payload
        )

        def register_concurrently():
            with psycopg.connect(brain_pg, row_factory=dict_row) as contender:
                contender.execute("SET ROLE brain_artifact_authority_writer")
                return contender.execute(statement, concurrent_params).fetchone()["receipt"]

        with ThreadPoolExecutor(max_workers=2) as pool:
            receipts = [future.result() for future in (pool.submit(register_concurrently), pool.submit(register_concurrently))]
        assert receipts[0] == receipts[1]
        assert connection.execute(
            "SELECT count(*) AS count FROM brain_artifact_authority_requests WHERE request_id=%s",
            (concurrent_params[0],),
        ).fetchone()["count"] == 1
        connection.commit()

        with connection.transaction():
            for artifact_hash in HASHES:
                connection.execute(
                    "INSERT INTO brain_artifacts "
                    "(request_id,artifact_hash,media_type,byte_length,schema_version,location) "
                    "VALUES (%s,%s,'application/json',1,1,%s) ON CONFLICT (artifact_hash) DO NOTHING",
                    (f"policy-{artifact_hash}", artifact_hash, f"memory://{artifact_hash}"),
                )
        _insert_policy(connection, version="v6-nonauthoritative")
        _attach_policy_contract(connection, "v6-nonauthoritative")
        connection.execute(
            "UPDATE fleet_config SET paused=TRUE,ats_paused=TRUE,ats_apply_mode='stopped',"
            "linkedin_apply_mode='stopped',canary_enabled=FALSE,linkedin_canary_enabled=FALSE WHERE id=1"
        )
        connection.commit()
        _create_lifecycle_login(connection, brain_pg, "v6_controller_login", "brain_policy_controller")
        connection.commit()
        with psycopg.connect(_dsn_for(brain_pg, "v6_controller_login"), row_factory=dict_row) as controller:
            with pytest.raises(psycopg.errors.CheckViolation, match="authoritative artifacts"):
                controller.execute(
                    "SELECT brain_controller_transition_policy('v6-nonauthoritative','validated',NULL)"
                )
            controller.rollback()
        with pytest.raises(RuntimeError, match="retains public schema CREATE"):
            with connection.transaction():
                connection.execute("GRANT CREATE ON SCHEMA public TO brain_artifact_authority_writer")
                with connection.cursor() as cursor:
                    schema._verify_v6_contract(cursor)


_V7_REGISTER_SQL = (
    "SELECT brain_register_authoritative_artifact_manifest("
    "%s::uuid,%s,%s,%s,%s::timestamptz,%s::timestamptz,%s,%s,%s::jsonb) AS receipt"
)


def _install_v7_authority(connection) -> None:
    schema.ensure_brain_schema_v1(connection)
    require_disposable_postgres(connection)
    connection.execute(
        "CREATE ROLE brain_controller_test_login LOGIN NOINHERIT NOSUPERUSER "
        "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
    )
    connection.commit()
    topology = pg_roles.BootstrapTopology(
        database_owner_role="postgres",
        controller_role="brain_controller_test_login",
        verifier_role="brain_schema_verifier",
        migrator_role="brain_schema_migrator",
        retired_admin_roles=(),
        infrastructure_superuser_roles=("postgres",),
    )
    with connection.transaction():
        with connection.cursor() as cursor:
            pg_roles._install_brain_authority_in_transaction(cursor, topology=topology)


def _v7_registration_params(connection, *, request_id: str, digest: str, artifact_hash: str):
    identity = connection.execute(
        "SELECT system_identifier::text AS system_id,current_database() AS database_name,"
        "clock_timestamp()-interval '1 second' AS issued_at,"
        "clock_timestamp()+interval '750 milliseconds' AS expires_at FROM pg_control_system()"
    ).fetchone()
    payload = json.dumps([{
        "artifact_hash": artifact_hash,
        "byte_length": 7,
        "media_type": "application/json",
        "backend": "s3",
        "bucket": "immutable",
        "object_key": f"sha256/{artifact_hash}",
        "provider_version_id": f"version-{artifact_hash[:8]}",
        "provider_checksum": f"checksum-{artifact_hash[:8]}",
        "storage_immutable": True,
        "encryption_mode": "customer_managed",
        "encryption_key_id": "kms-key-1",
        "policy_source_id": "opaque-for-nonsnapshot",
    }])
    return (
        request_id,
        digest,
        "brain-artifact-authority-registration-v1",
        "key-1",
        identity["issued_at"],
        identity["expires_at"],
        identity["system_id"],
        identity["database_name"],
        payload,
    )


def test_v7_exact_replay_survives_expiry_but_new_expired_and_changed_requests_fail(brain_pg) -> None:
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v7_authority(connection)
        params = _v7_registration_params(
            connection,
            request_id="71111111-1111-1111-1111-111111111111",
            digest="1" * 64,
            artifact_hash="2" * 64,
        )
        connection.execute("SET ROLE brain_artifact_authority_writer")
        receipt = connection.execute(_V7_REGISTER_SQL, params).fetchone()["receipt"]
        connection.commit()
        connection.execute("SELECT pg_sleep(0.9)")
        assert connection.execute(_V7_REGISTER_SQL, params).fetchone()["receipt"] == receipt
        connection.commit()

        with pytest.raises(psycopg.errors.InvalidParameterValue, match="destination mismatch"):
            connection.execute(_V7_REGISTER_SQL, (*params[:7], "altered-database", params[8]))
        connection.rollback()
        connection.execute("RESET ROLE")
        counts_before = connection.execute(
            "SELECT (SELECT count(*) FROM brain_artifact_authority_requests) AS requests,"
            "(SELECT count(*) FROM brain_artifact_authority_registrations) AS registrations,"
            "(SELECT count(*) FROM brain_artifacts) AS artifacts"
        ).fetchone()
        connection.execute("SET ROLE brain_artifact_authority_writer")
        with pytest.raises(psycopg.errors.UniqueViolation, match="different manifest digest"):
            connection.execute(_V7_REGISTER_SQL, (params[0], "3" * 64, *params[2:]))
        connection.rollback()
        connection.execute("RESET ROLE")
        assert connection.execute(
            "SELECT (SELECT count(*) FROM brain_artifact_authority_requests) AS requests,"
            "(SELECT count(*) FROM brain_artifact_authority_registrations) AS registrations,"
            "(SELECT count(*) FROM brain_artifacts) AS artifacts"
        ).fetchone() == counts_before

        expired_new = list(params)
        expired_new[0] = "72222222-2222-2222-2222-222222222222"
        expired_new[1] = "4" * 64
        expired_new[8] = json.dumps([dict(json.loads(params[8])[0], artifact_hash="5" * 64)])
        connection.execute("SET ROLE brain_artifact_authority_writer")
        with pytest.raises(psycopg.errors.InvalidParameterValue, match="invalid or expired"):
            connection.execute(_V7_REGISTER_SQL, tuple(expired_new))
        connection.rollback()

        connection.execute("RESET ROLE")
        connection.execute("SET session_replication_role=replica")
        connection.execute(
            "UPDATE brain_artifact_authority_requests SET destination_database_name='tampered' "
            "WHERE request_id=%s",
            (params[0],),
        )
        connection.execute("SET session_replication_role=origin")
        connection.commit()
        connection.execute("SET ROLE brain_artifact_authority_writer")
        with pytest.raises(psycopg.errors.InvalidParameterValue, match="destination mismatch"):
            connection.execute(_V7_REGISTER_SQL, params)
        connection.rollback()


def test_v7_same_digest_replay_rejects_invalid_structural_shape(brain_pg) -> None:
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v7_authority(connection)
        params = _v7_registration_params(
            connection,
            request_id="71212121-1212-1212-1212-121212121212",
            digest="1" * 64,
            artifact_hash="2" * 64,
        )
        connection.execute("SET ROLE brain_artifact_authority_writer")
        connection.execute(_V7_REGISTER_SQL, params)
        connection.commit()
        invalid = (
            (*params[:2], "wrong-purpose", *params[3:]),
            (*params[:3], "", *params[4:]),
            (*params[:5], params[4], *params[6:]),
            (*params[:8], "[]"),
            (*params[:8], json.dumps([{"artifact_hash": "bad"}])),
        )
        for candidate in invalid:
            with pytest.raises(psycopg.errors.InvalidParameterValue):
                connection.execute(_V7_REGISTER_SQL, candidate)
            connection.rollback()


def test_v7_snapshot_binding_requires_digest_of_exact_canonical_provenance(brain_pg) -> None:
    policy = "snapshot-digest-v7"
    provenance = _snapshot_provenance(
        policy=policy, lane="ats", role="label_snapshot", source_hash="a" * 64
    )
    exact_hash = hashlib.sha256(provenance.encode("utf-8")).hexdigest()
    unrelated_hash = "f" * 64
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v7_authority(connection)
        params = list(_v7_registration_params(
            connection,
            request_id="71313131-1313-1313-1313-131313131313",
            digest="3" * 64,
            artifact_hash=exact_hash,
        ))
        artifact = json.loads(params[8])[0]
        artifact["policy_source_id"] = provenance
        unrelated = dict(
            artifact,
            artifact_hash=unrelated_hash,
            object_key=f"sha256/{unrelated_hash}",
            provider_version_id="version-unrelated",
            provider_checksum="checksum-unrelated",
        )
        params[8] = json.dumps([artifact, unrelated])
        connection.execute("SET ROLE brain_artifact_authority_writer")
        connection.execute(_V7_REGISTER_SQL, tuple(params))
        connection.commit()
        connection.execute("RESET ROLE")
        assert connection.execute(
            "SELECT brain_snapshot_binding_is_authoritative(%s,%s,%s,%s) AS matched",
            (policy, "ats", "label_snapshot", exact_hash),
        ).fetchone()["matched"] is True
        assert connection.execute(
            "SELECT brain_snapshot_binding_is_authoritative(%s,%s,%s,%s) AS matched",
            (policy, "ats", "label_snapshot", unrelated_hash),
        ).fetchone()["matched"] is False

def test_v7_concurrent_different_digest_has_one_winner_and_no_loser_writes(brain_pg) -> None:
    with psycopg.connect(brain_pg, row_factory=dict_row) as owner:
        _install_v7_authority(owner)
        first = _v7_registration_params(
            owner,
            request_id="73333333-3333-3333-3333-333333333333",
            digest="6" * 64,
            artifact_hash="7" * 64,
        )
        second = (first[0], "8" * 64, *first[2:-1], json.dumps([
            dict(json.loads(first[8])[0], artifact_hash="9" * 64, object_key="sha256/" + "9" * 64)
        ]))
        owner.commit()

    barrier = Barrier(2)

    def register(params):
        with psycopg.connect(brain_pg, row_factory=dict_row) as contender:
            contender.execute("SET ROLE brain_artifact_authority_writer")
            barrier.wait()
            try:
                receipt = contender.execute(_V7_REGISTER_SQL, params).fetchone()["receipt"]
                contender.commit()
                return "registered", receipt
            except psycopg.errors.UniqueViolation:
                contender.rollback()
                return "conflict", None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result() for future in (pool.submit(register, first), pool.submit(register, second))]
    assert sorted(status for status, _receipt in results) == ["conflict", "registered"]
    with psycopg.connect(brain_pg, row_factory=dict_row) as verifier:
        assert verifier.execute(
            "SELECT (SELECT count(*) FROM brain_artifact_authority_requests WHERE request_id=%s) AS requests,"
            "(SELECT count(*) FROM brain_artifact_authority_registrations WHERE request_id=%s) AS registrations,"
            "(SELECT count(*) FROM brain_artifacts WHERE artifact_hash IN (%s,%s)) AS artifacts",
            (first[0], first[0], "7" * 64, "9" * 64),
        ).fetchone() == {"requests": 1, "registrations": 1, "artifacts": 1}


def _snapshot_provenance(*, policy: str, lane: str, role: str, source_hash: str, **extra) -> str:
    value = {
        "kind": "applypilot.policy.snapshot-reference",
        "lane": lane,
        "policyVersion": policy,
        "role": role,
        "schemaVersion": 1,
        "sourceField": role,
        "sourceSha256": source_hash,
        **extra,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_v7_snapshot_reference_provenance_is_closed_and_canonical(brain_pg) -> None:
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v7_authority(connection)
        valid = _snapshot_provenance(
            policy="snapshot-v7", lane="ats", role="label_snapshot", source_hash="a" * 64
        )
        statement = (
            "SELECT brain_snapshot_reference_provenance_matches(%s,%s,%s,%s) AS matches"
        )
        assert connection.execute(
            statement, (valid, "snapshot-v7", "ats", "label_snapshot")
        ).fetchone()["matches"] is True
        invalid = (
            _snapshot_provenance(
                policy="snapshot-v7", lane="ats", role="label_snapshot", source_hash="a" * 64,
                extra="forbidden",
            ),
            valid.replace('"sourceField":"label_snapshot",', ""),
            valid.replace('"label_snapshot"', '"labelSnapshot"'),
            valid.replace("a" * 64, "A" * 64),
            valid.replace('"lane":"ats"', '"lane":"linkedin"'),
            valid.replace('"policyVersion":"snapshot-v7"', '"policyVersion":"other"'),
            valid.replace('"schemaVersion":1', '"schemaVersion":"1"'),
            "{" + valid[1:].replace(",", ", ", 1),
            "not-json",
        )
        for provenance in invalid:
            assert connection.execute(
                statement, (provenance, "snapshot-v7", "ats", "label_snapshot")
            ).fetchone()["matches"] is False


def test_v7_ten_binding_lifecycle_rejects_opaque_snapshot_then_accepts_canonical_provenance(brain_pg) -> None:
    roles = (
        "qualification_model", "preference_model", "outcome_model", "knowledge_graph",
        "label_snapshot", "pairwise_snapshot", "outcome_snapshot", "config", "metrics", "replay",
    )
    policy = "snapshot-lifecycle-v7"
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v7_authority(connection)
        identity = connection.execute(
            "SELECT system_identifier::text AS system_id,current_database() AS database_name,"
            "clock_timestamp()-interval '1 second' AS issued_at,"
            "clock_timestamp()+interval '1 hour' AS expires_at FROM pg_control_system()"
        ).fetchone()
        payload = []
        snapshot_hashes = {}
        pairwise_corrected = _snapshot_provenance(
            policy=policy, lane="ats", role="pairwise_snapshot", source_hash="c" * 64
        )
        for index, role in enumerate(roles):
            policy_source_id = f"opaque-{role}"
            if role in {"label_snapshot", "outcome_snapshot"}:
                policy_source_id = _snapshot_provenance(
                    policy=policy, lane="ats", role=role, source_hash=("abcdef"[index % 6] * 64)
                )
            artifact_hash = HASHES[index]
            if role.endswith("_snapshot"):
                hash_source = pairwise_corrected if role == "pairwise_snapshot" else policy_source_id
                artifact_hash = hashlib.sha256(hash_source.encode("utf-8")).hexdigest()
                snapshot_hashes[role] = artifact_hash
            payload.append({
                "artifact_hash": artifact_hash, "byte_length": index + 1,
                "media_type": "application/json", "backend": "s3", "bucket": "immutable",
                "object_key": f"sha256/{artifact_hash}",
                "provider_version_id": f"version-{index}", "provider_checksum": f"checksum-{index}",
                "storage_immutable": True, "encryption_mode": "customer_managed",
                "encryption_key_id": "kms-key-1", "policy_source_id": policy_source_id,
            })
        params = (
            "74444444-4444-4444-4444-444444444444", "b" * 64,
            "brain-artifact-authority-registration-v1", "key-1", identity["issued_at"],
            identity["expires_at"], identity["system_id"], identity["database_name"], json.dumps(payload),
        )
        connection.execute("SET ROLE brain_artifact_authority_writer")
        connection.execute(_V7_REGISTER_SQL, params)
        connection.commit()
        connection.execute("RESET ROLE")
        _insert_policy(connection, policy)
        _attach_policy_contract(connection, policy, artifact_hashes=snapshot_hashes)
        connection.execute(
            "UPDATE fleet_config SET paused=TRUE,ats_paused=TRUE,"
            "ats_apply_mode='stopped',linkedin_apply_mode='stopped',"
            "canary_enabled=FALSE,linkedin_canary_enabled=FALSE WHERE id=1"
        )
        _set_role_password(connection, brain_pg, "brain_controller_test_login")
        connection.commit()
        with psycopg.connect(
            _dsn_for(brain_pg, "brain_controller_test_login"),
            row_factory=dict_row,
        ) as controller:
            controller.execute("SET ROLE brain_policy_controller")
            with pytest.raises(psycopg.errors.CheckViolation, match="authoritative artifacts"):
                controller.execute(
                    "SELECT brain_controller_transition_policy(%s,'validated',NULL)",
                    (policy,),
                )
            controller.rollback()
        corrected = pairwise_corrected
        connection.execute("SET session_replication_role=replica")
        connection.execute(
            "UPDATE brain_artifact_authority_registrations SET policy_source_id=%s "
            "WHERE artifact_hash=%s",
            (corrected, snapshot_hashes["pairwise_snapshot"]),
        )
        connection.execute("SET session_replication_role=origin")
        connection.commit()
        with psycopg.connect(
            _dsn_for(brain_pg, "brain_controller_test_login"),
            row_factory=dict_row,
        ) as controller:
            controller.execute("SET ROLE brain_policy_controller")
            controller.execute(
                "SELECT brain_controller_transition_policy(%s,'validated',NULL)",
                (policy,),
            )
        assert connection.execute(
            "SELECT lifecycle FROM brain_decision_policies WHERE policy_version=%s", (policy,)
        ).fetchone()["lifecycle"] == "validated"
        assert connection.execute(
            "SELECT count(*) AS count FROM brain_policy_artifacts WHERE policy_version=%s", (policy,)
        ).fetchone()["count"] == 10


def test_v7_ledger_rejects_exact_version_substitution(brain_pg) -> None:
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v7_authority(connection)
        for version, message in ((6, "invalid version 6 contract"), (7, "invalid version 7 contract")):
            with pytest.raises(RuntimeError, match=message):
                with connection.transaction():
                    connection.execute("SET LOCAL session_replication_role=replica")
                    connection.execute(
                        "UPDATE brain_schema_versions SET migration_checksum=%s WHERE version=%s",
                        ("0" * 64, version),
                    )
                    with connection.cursor() as cursor:
                        schema.verify_brain_schema_v7_in_transaction(cursor)
            connection.rollback()
        schema.verify_brain_schema_v7(connection)


def test_current_v6_database_upgrades_to_v7_without_rewriting_v6_ledger(brain_pg) -> None:
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        schema.ensure_brain_schema_v1(connection)
        connection.execute(
            "CREATE ROLE brain_controller_test_login LOGIN NOINHERIT NOSUPERUSER "
            "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
        )
        connection.commit()
        with connection.transaction():
            connection.execute("ALTER SCHEMA public OWNER TO brain_schema_migrator")
            connection.execute("SET ROLE brain_schema_migrator")
            with connection.cursor() as cursor:
                schema.ensure_brain_schema_v6_in_transaction(cursor)
            connection.execute("RESET ROLE")
            connection.execute("ALTER SCHEMA public OWNER TO postgres")
        v6_row = connection.execute(
            "SELECT migration_checksum FROM public.brain_schema_versions WHERE version=6"
        ).fetchone()
        assert v6_row["migration_checksum"] == schema._CURRENT_V6_CHECKSUM
        connection.commit()
        with pytest.raises(RuntimeError, match="first installation requires expected_controller_role"):
            schema.ensure_brain_schema_v7(connection)
        topology = pg_roles.BootstrapTopology(
            database_owner_role="postgres",
            controller_role="brain_controller_test_login",
            verifier_role="brain_schema_verifier",
            migrator_role="brain_schema_migrator",
            retired_admin_roles=(),
            infrastructure_superuser_roles=("postgres",),
        )
        with connection.transaction():
            with connection.cursor() as cursor:
                pg_roles._install_brain_authority_in_transaction(cursor, topology=topology)
        assert connection.execute(
            "SELECT array_agg(version ORDER BY version) AS versions FROM public.brain_schema_versions"
        ).fetchone()["versions"] == [1, 2, 3, 4, 5, 6, 7]
        assert connection.execute(
            "SELECT migration_checksum FROM public.brain_schema_versions WHERE version=6"
        ).fetchone()["migration_checksum"] == schema._CURRENT_V6_CHECKSUM
        connection.commit()
        schema.verify_brain_schema_v7(connection)


def test_v7_install_preserves_permanent_set_edge_for_non_superuser_database_owner(brain_pg) -> None:
    password = "applypilot-v7-provider-test-only"
    with psycopg.connect(brain_pg, row_factory=dict_row) as root:
        require_disposable_postgres(root)
        for authority_role in (
            "brain_artifact_authority_writer",
            "brain_artifact_authority_owner",
        ):
            _drop_fixed_role(root, authority_role)
        root.execute("DROP OWNED BY brain_schema_migrator")
        root.execute("DROP ROLE brain_schema_migrator")
        root.execute(
            pg_sql.SQL(
                "CREATE ROLE brain_v7_provider LOGIN NOINHERIT NOSUPERUSER "
                "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD {}"
            ).format(pg_sql.Literal(password))
        )
        root.execute(
            "CREATE ROLE brain_schema_migrator NOLOGIN NOINHERIT NOSUPERUSER "
            "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
        )
        root.execute(
            "CREATE ROLE brain_artifact_authority_owner NOLOGIN NOINHERIT NOSUPERUSER "
            "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
        )
        root.execute(
            "CREATE ROLE brain_artifact_authority_writer NOLOGIN NOINHERIT NOSUPERUSER "
            "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
        )
        root.execute(
            "GRANT brain_artifact_authority_owner TO brain_schema_migrator "
            "WITH ADMIN FALSE, INHERIT FALSE, SET TRUE GRANTED BY postgres"
        )
        root.execute(
            "GRANT brain_schema_migrator TO brain_v7_provider "
            "WITH ADMIN FALSE, INHERIT FALSE, SET TRUE GRANTED BY postgres"
        )
        root.execute(
            "CREATE ROLE brain_controller_test_login LOGIN NOINHERIT NOSUPERUSER "
            "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
        )
        root.execute("GRANT USAGE, CREATE ON SCHEMA public TO brain_schema_migrator WITH GRANT OPTION")
        root.execute("GRANT CREATE ON DATABASE postgres TO brain_schema_migrator")
        root.execute(
            "GRANT brain_schema_migrator TO postgres "
            "WITH ADMIN FALSE, INHERIT FALSE, SET TRUE GRANTED BY postgres"
        )
        for relation in (
            "fleet_config", "fleet_decision_policies", "apply_queue", "linkedin_queue",
            "workers", "worker_heartbeat", "fleet_worker_principals", "fleet_desired_state",
            "rate_governor", "applied_set", "apply_attempts", "apply_result_events",
            "fleet_worker_blocklist",
        ):
            root.execute(
                pg_sql.SQL(
                    "GRANT SELECT, INSERT, UPDATE ON TABLE public.{} "
                    "TO brain_schema_migrator WITH GRANT OPTION"
                ).format(pg_sql.Identifier(relation))
            )
        root.commit()
        schema.ensure_brain_schema_v1(root)
        root.execute(
            "GRANT brain_policy_controller TO brain_controller_test_login "
            "WITH ADMIN FALSE, INHERIT FALSE, SET TRUE GRANTED BY postgres"
        )
        root.execute(
            "REVOKE brain_schema_migrator FROM postgres GRANTED BY postgres RESTRICT"
        )
        root.execute("ALTER DATABASE postgres OWNER TO brain_v7_provider")
        root.execute("ALTER SCHEMA public OWNER TO pg_database_owner")
        root.commit()
        parameters = conninfo_to_dict(brain_pg)
        parameters.update(user="brain_v7_provider", password=password)
        topology = pg_roles.BootstrapTopology(
            database_owner_role="brain_v7_provider",
            controller_role="brain_controller_test_login",
            verifier_role="brain_schema_verifier",
            migrator_role="brain_schema_migrator",
            retired_admin_roles=(),
            infrastructure_superuser_roles=("postgres",),
        )
        try:
            with psycopg.connect(make_conninfo(**parameters), row_factory=dict_row) as provider:
                assert provider.execute(
                    "SELECT rolcanlogin,rolinherit,rolsuper,rolcreaterole "
                    "FROM pg_roles WHERE rolname=current_user"
                ).fetchone() == {
                    "rolcanlogin": True,
                    "rolinherit": False,
                    "rolsuper": False,
                    "rolcreaterole": False,
                }
                provider.commit()
                root.execute(
                    "GRANT brain_policy_controller TO brain_controller_test_login "
                    "WITH ADMIN TRUE, INHERIT TRUE, SET FALSE"
                )
                root.commit()
                versions_before = root.execute(
                    "SELECT array_agg(version ORDER BY version) AS versions "
                    "FROM brain_schema_versions"
                ).fetchone()["versions"]
                root.commit()
                with pytest.raises(
                    RuntimeError,
                    match="exact brain_policy_controller membership",
                ):
                    with provider.transaction():
                        with provider.cursor() as cursor:
                            pg_roles._install_brain_authority_in_transaction(
                                cursor, topology=topology
                            )
                provider.rollback()
                assert provider.execute(
                    "SELECT admin_option,inherit_option,set_option "
                    "FROM pg_auth_members "
                    "WHERE roleid='brain_policy_controller'::regrole "
                    "AND member='brain_controller_test_login'::regrole"
                ).fetchall() == [
                    {
                        "admin_option": True,
                        "inherit_option": True,
                        "set_option": False,
                    }
                ]
                assert root.execute(
                    "SELECT array_agg(version ORDER BY version) AS versions "
                    "FROM brain_schema_versions"
                ).fetchone()["versions"] == versions_before
                root.commit()
                root.execute(
                    "GRANT brain_policy_controller TO brain_controller_test_login "
                    "WITH ADMIN FALSE, INHERIT FALSE, SET TRUE"
                )
                root.commit()
                root.execute("CREATE TABLE v7_unrelated_guard(id integer)")
                root.execute("GRANT SELECT ON v7_unrelated_guard TO brain_candidate_reader")
                root.commit()
                with pytest.raises(
                    RuntimeError,
                    match="cross-owner ACL leakage|candidate.*authority exposure",
                ):
                    with provider.transaction():
                        with provider.cursor() as cursor:
                            pg_roles._install_brain_authority_in_transaction(
                                cursor, topology=topology
                            )
                provider.rollback()
                assert provider.execute(
                    "SELECT pg_get_userbyid(nspowner) AS owner_name "
                    "FROM pg_namespace WHERE nspname='public'"
                ).fetchone()["owner_name"] == "pg_database_owner"
                assert provider.execute(
                    "SELECT admin_option,inherit_option,set_option,"
                    "pg_get_userbyid(grantor) AS grantor FROM pg_auth_members "
                    "WHERE roleid='brain_schema_migrator'::regrole "
                    "AND member='brain_v7_provider'::regrole"
                    " ORDER BY grantor"
                ).fetchall() == [
                    {
                        "admin_option": False,
                        "inherit_option": False,
                        "set_option": True,
                        "grantor": "postgres",
                    },
                ]
                assert provider.execute(
                    "SELECT has_table_privilege('brain_candidate_reader',"
                    "'v7_unrelated_guard','SELECT') AS allowed"
                ).fetchone()["allowed"] is True
                assert root.execute(
                    "SELECT array_agg(version ORDER BY version) AS versions "
                    "FROM brain_schema_versions"
                ).fetchone()["versions"] == [1, 2, 3]
                root.commit()
                provider.commit()
                root.execute("REVOKE SELECT ON v7_unrelated_guard FROM brain_candidate_reader")
                root.execute("DROP TABLE v7_unrelated_guard")
                root.commit()
                with provider.transaction():
                    with provider.cursor() as cursor:
                        pg_roles._install_brain_authority_in_transaction(
                            cursor, topology=topology
                        )
                assert provider.execute(
                    "SELECT pg_get_userbyid(nspowner) AS owner_name "
                    "FROM pg_namespace WHERE nspname='public'"
                ).fetchone()["owner_name"] == "pg_database_owner"
                assert provider.execute(
                    "SELECT admin_option,inherit_option,set_option,"
                    "pg_get_userbyid(grantor) AS grantor "
                    "FROM pg_auth_members "
                    "WHERE roleid='brain_schema_migrator'::regrole "
                    "AND member='brain_v7_provider'::regrole"
                    " ORDER BY grantor"
                ).fetchall() == [
                    {
                        "admin_option": False,
                        "inherit_option": False,
                        "set_option": True,
                        "grantor": "postgres",
                    },
                ]
                provider.commit()
                schema.verify_brain_schema_v7(root)
        finally:
            root.rollback()
            root.execute("ALTER DATABASE postgres OWNER TO postgres")
            root.execute("ALTER SCHEMA public OWNER TO pg_database_owner")
            root.commit()

def test_v7_install_fails_closed_on_automatic_creator_membership_grant(brain_pg) -> None:
    password = "applypilot-v7-provider-test-only"
    with psycopg.connect(brain_pg, row_factory=dict_row) as root:
        require_disposable_postgres(root)
        root.execute("DROP OWNED BY brain_schema_migrator")
        root.execute("DROP ROLE brain_schema_migrator")
        root.execute(
            pg_sql.SQL(
                "CREATE ROLE brain_v7_provider LOGIN NOINHERIT NOSUPERUSER "
                "NOCREATEDB CREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD {}"
            ).format(pg_sql.Literal(password))
        )
        root.execute(
            "CREATE ROLE brain_controller_test_login LOGIN NOINHERIT NOSUPERUSER "
            "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
        )
        root.execute(
            "GRANT brain_policy_controller TO brain_controller_test_login "
            "WITH ADMIN FALSE, INHERIT FALSE, SET TRUE GRANTED BY postgres"
        )
        root.execute("ALTER DATABASE postgres OWNER TO brain_v7_provider")
        root.commit()
        parameters = conninfo_to_dict(brain_pg)
        parameters.update(user="brain_v7_provider", password=password)
        topology = pg_roles.BootstrapTopology(
            database_owner_role="brain_v7_provider",
            controller_role="brain_controller_test_login",
            verifier_role="brain_schema_verifier",
            migrator_role="brain_schema_migrator",
            retired_admin_roles=(),
            infrastructure_superuser_roles=("postgres",),
        )
        try:
            with psycopg.connect(make_conninfo(**parameters), row_factory=dict_row) as provider:
                provider.execute(
                    "CREATE ROLE brain_schema_migrator NOLOGIN NOINHERIT NOSUPERUSER "
                    "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                )
                provider.commit()
                expected_edge = [
                    {
                        "admin_option": True,
                        "inherit_option": False,
                        "set_option": False,
                        "grantor": "postgres",
                    }
                ]
                query = (
                    "SELECT admin_option,inherit_option,set_option,"
                    "pg_get_userbyid(grantor) AS grantor FROM pg_auth_members "
                    "WHERE roleid='brain_schema_migrator'::regrole "
                    "AND member='brain_v7_provider'::regrole"
                )
                assert provider.execute(query).fetchall() == expected_edge
                provider.commit()
                with pytest.raises(RuntimeError, match="privileged pre-provisioning"):
                    with provider.transaction():
                        with provider.cursor() as cursor:
                            pg_roles._install_brain_authority_in_transaction(
                                cursor, topology=topology
                            )
                provider.rollback()
                assert provider.execute(query).fetchall() == expected_edge
        finally:
            root.rollback()
            root.execute("ALTER DATABASE postgres OWNER TO postgres")
            root.commit()


def test_v7_verifier_rejects_membership_option_grantor_and_multiplicity_drift(brain_pg) -> None:
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v7_authority(connection)
        rename_password = "applypilot-v7-bootstrap-rename-test-only"
        connection.execute(
            pg_sql.SQL("CREATE ROLE brain_v7_renamer LOGIN SUPERUSER PASSWORD {}").format(
                pg_sql.Literal(rename_password)
            )
        )
        connection.commit()
        rename_parameters = conninfo_to_dict(brain_pg)
        rename_parameters.update(user="brain_v7_renamer", password=rename_password)
        with psycopg.connect(make_conninfo(**rename_parameters), row_factory=dict_row) as renamer:
            renamer.execute("ALTER ROLE postgres RENAME TO brain_v7_bootstrap_renamed")
            renamer.commit()
            try:
                schema.verify_brain_schema_v7(connection)
            finally:
                renamer.execute("ALTER ROLE brain_v7_bootstrap_renamed RENAME TO postgres")
                renamer.commit()
        connection.execute("DROP ROLE brain_v7_renamer")
        connection.commit()

        with pytest.raises(psycopg.errors.FeatureNotSupported, match="bootstrap superuser"):
            with connection.transaction():
                connection.execute("ALTER ROLE postgres NOSUPERUSER")
        connection.rollback()
        schema.verify_brain_schema_v7(connection)

        with pytest.raises(RuntimeError, match="database[- ]owner"):
            with connection.transaction():
                connection.execute(
                    "CREATE ROLE brain_v7_owner_drift LOGIN NOINHERIT NOSUPERUSER "
                    "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                )
                connection.execute("ALTER DATABASE postgres OWNER TO brain_v7_owner_drift")
                with connection.cursor() as cursor:
                    schema.verify_brain_schema_v7_in_transaction(cursor)
        connection.rollback()

        mutations = (
            "GRANT brain_artifact_authority_owner TO brain_schema_migrator "
            "WITH ADMIN TRUE, INHERIT FALSE, SET TRUE",
            "GRANT brain_artifact_authority_owner TO brain_schema_migrator "
            "WITH ADMIN FALSE, INHERIT TRUE, SET TRUE",
        )
        for statement in mutations:
            with pytest.raises(RuntimeError, match="membership"):
                with connection.transaction():
                    connection.execute(statement)
                    with connection.cursor() as cursor:
                        schema.verify_brain_schema_v7_in_transaction(cursor)
            connection.rollback()

        with pytest.raises(RuntimeError, match="membership"):
            with connection.transaction():
                connection.execute("CREATE ROLE brain_v7_alternate_grantor SUPERUSER NOLOGIN")
                connection.execute(
                    "GRANT brain_artifact_authority_owner TO brain_v7_alternate_grantor "
                    "WITH ADMIN OPTION"
                )
                connection.execute("SET LOCAL ROLE brain_v7_alternate_grantor")
                connection.execute(
                    "GRANT brain_artifact_authority_owner TO brain_schema_migrator "
                    "WITH ADMIN FALSE, INHERIT FALSE, SET TRUE"
                )
                connection.execute("RESET ROLE")
                with connection.cursor() as cursor:
                    schema.verify_brain_schema_v7_in_transaction(cursor)
        connection.rollback()
        schema.verify_brain_schema_v7(connection)

        with pytest.raises(RuntimeError, match="controller membership"):
            with connection.transaction():
                connection.execute(
                    "GRANT brain_policy_controller TO brain_controller_test_login "
                    "WITH ADMIN TRUE, INHERIT TRUE, SET FALSE"
                )
                with connection.cursor() as cursor:
                    schema.verify_brain_schema_v7_in_transaction(cursor)
        connection.rollback()

        connection.execute(
            "GRANT brain_policy_controller TO brain_controller_test_login "
            "WITH ADMIN TRUE, INHERIT TRUE, SET FALSE"
        )
        connection.commit()
        topology = pg_roles.BootstrapTopology(
            database_owner_role="brain_v7_provider",
            controller_role="brain_controller_test_login",
            verifier_role="brain_schema_verifier",
            migrator_role="brain_schema_migrator",
            retired_admin_roles=(),
            infrastructure_superuser_roles=("postgres",),
        )
        with connection.transaction():
            with connection.cursor() as cursor:
                pg_roles._install_brain_authority_in_transaction(
                    cursor,
                    topology=topology,
                )
        schema.verify_brain_schema_v7(connection)
        assert connection.execute(
            "SELECT admin_option,inherit_option,set_option,grantor "
            "FROM pg_auth_members "
            "WHERE roleid='brain_policy_controller'::regrole"
        ).fetchall() == [
            {
                "admin_option": False,
                "inherit_option": False,
                "set_option": True,
                "grantor": 10,
            }
        ]
        connection.commit()

        with pytest.raises(RuntimeError, match="controller (membership|topology)"):
            with connection.transaction():
                connection.execute(
                    "CREATE ROLE brain_v7_wrong_controller LOGIN NOINHERIT NOSUPERUSER "
                    "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                )
                connection.execute(
                    "REVOKE brain_policy_controller FROM brain_controller_test_login "
                    "GRANTED BY postgres RESTRICT"
                )
                connection.execute(
                    "GRANT brain_policy_controller TO brain_v7_wrong_controller "
                    "WITH ADMIN FALSE, INHERIT FALSE, SET TRUE GRANTED BY postgres"
                )
                with connection.cursor() as cursor:
                    schema.verify_brain_schema_v7_in_transaction(cursor)
        connection.rollback()

        with pytest.raises(RuntimeError, match="controller membership"):
            with connection.transaction():
                connection.execute("CREATE ROLE brain_v7_controller_grantor SUPERUSER NOLOGIN")
                connection.execute(
                    "GRANT brain_policy_controller TO brain_v7_controller_grantor "
                    "WITH ADMIN TRUE, INHERIT FALSE, SET TRUE"
                )
                connection.execute(
                    "REVOKE brain_policy_controller FROM brain_controller_test_login "
                    "GRANTED BY postgres RESTRICT"
                )
                connection.execute("SET LOCAL ROLE brain_v7_controller_grantor")
                connection.execute(
                    "GRANT brain_policy_controller TO brain_controller_test_login "
                    "WITH ADMIN FALSE, INHERIT FALSE, SET TRUE "
                    "GRANTED BY brain_v7_controller_grantor"
                )
                connection.execute("RESET ROLE")
                with connection.cursor() as cursor:
                    schema.verify_brain_schema_v7_in_transaction(cursor)
        connection.rollback()

        with pytest.raises(RuntimeError, match="controller membership"):
            with connection.transaction():
                connection.execute(
                    "CREATE ROLE brain_v7_extra_controller LOGIN NOINHERIT NOSUPERUSER "
                    "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                )
                connection.execute(
                    "GRANT brain_policy_controller TO brain_v7_extra_controller "
                    "WITH ADMIN FALSE, INHERIT FALSE, SET TRUE"
                )
                with connection.cursor() as cursor:
                    schema.verify_brain_schema_v7_in_transaction(cursor)
        connection.rollback()


def test_v7_controller_topology_is_oid_bound_immutable_and_closed(brain_pg) -> None:
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v7_authority(connection)
        persisted_oid = connection.execute(
            "SELECT controller_role_oid FROM brain_v7_topology_contract WHERE singleton_id=1"
        ).fetchone()["controller_role_oid"]
        connection.commit()

        with pytest.raises(RuntimeError, match="invalid version 7 contract"):
            with connection.transaction():
                connection.execute("SET LOCAL session_replication_role=replica")
                connection.execute(
                    "UPDATE brain_schema_versions SET migration_checksum=%s WHERE version=7",
                    ("c1b8143bb676201d62dd4ed6da74a70034c9b83af3c40cea27c0e93e2b28026a",),
                )
                with connection.cursor() as cursor:
                    schema.verify_brain_schema_v7_in_transaction(cursor)
        connection.rollback()

        for statement in (
            "UPDATE brain_v7_topology_contract SET controller_role_oid=controller_role_oid",
            "DELETE FROM brain_v7_topology_contract",
            "TRUNCATE brain_v7_topology_contract",
        ):
            with pytest.raises(
                psycopg.errors.ObjectNotInPrerequisiteState,
                match="append-only",
            ):
                with connection.transaction():
                    connection.execute(statement)
            connection.rollback()

        for statement in (
            "ALTER ROLE brain_controller_test_login INHERIT",
            "ALTER ROLE brain_controller_test_login CREATEROLE",
        ):
            with pytest.raises(RuntimeError, match="controller membership"):
                with connection.transaction():
                    connection.execute(statement)
                    with connection.cursor() as cursor:
                        schema.verify_brain_schema_v7_in_transaction(cursor)
            connection.rollback()

        with pytest.raises(RuntimeError, match="controller membership closure"):
            with connection.transaction():
                connection.execute(
                    "CREATE ROLE brain_v7_controller_attacker LOGIN NOINHERIT NOSUPERUSER "
                    "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                )
                connection.execute(
                    "GRANT brain_controller_test_login TO brain_v7_controller_attacker"
                )
                with connection.cursor() as cursor:
                    schema.verify_brain_schema_v7_in_transaction(cursor)
        connection.rollback()

        with pytest.raises(RuntimeError, match="topology ACL"):
            with connection.transaction():
                connection.execute(
                    "GRANT UPDATE ON brain_v7_topology_contract "
                    "TO brain_artifact_authority_writer"
                )
                with connection.cursor() as cursor:
                    schema.verify_brain_schema_v7_in_transaction(cursor)
        connection.rollback()

        with connection.transaction():
            connection.execute(
                "ALTER ROLE brain_controller_test_login RENAME TO brain_controller_oid_renamed"
            )
            with connection.cursor() as cursor:
                schema.verify_brain_schema_v7_in_transaction(cursor)
            connection.execute(
                "ALTER ROLE brain_controller_oid_renamed RENAME TO brain_controller_test_login"
            )

        with pytest.raises(RuntimeError, match="controller (membership|topology)"):
            with connection.transaction():
                connection.execute(
                    "REVOKE brain_policy_controller FROM brain_controller_test_login "
                    "GRANTED BY postgres RESTRICT"
                )
                connection.execute("DROP ROLE brain_controller_test_login")
                connection.execute(
                    "CREATE ROLE brain_controller_test_login LOGIN NOINHERIT NOSUPERUSER "
                    "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                )
                connection.execute(
                    "GRANT brain_policy_controller TO brain_controller_test_login "
                    "WITH ADMIN FALSE, INHERIT FALSE, SET TRUE GRANTED BY postgres"
                )
                assert connection.execute(
                    "SELECT 'brain_controller_test_login'::regrole::oid AS oid"
                ).fetchone()["oid"] != persisted_oid
                with connection.cursor() as cursor:
                    schema.verify_brain_schema_v7_in_transaction(cursor)
        connection.rollback()

        with pytest.raises(RuntimeError, match="controller (membership|topology)"):
            with connection.transaction():
                connection.execute(
                    "CREATE ROLE brain_v7_wrong_topology LOGIN NOINHERIT NOSUPERUSER "
                    "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                )
                wrong_oid = connection.execute(
                    "SELECT 'brain_v7_wrong_topology'::regrole::oid AS oid"
                ).fetchone()["oid"]
                connection.execute("SET LOCAL session_replication_role=replica")
                connection.execute(
                    "UPDATE brain_v7_topology_contract SET controller_role_oid=%s",
                    (wrong_oid,),
                )
                connection.execute("SET LOCAL session_replication_role=origin")
                with connection.cursor() as cursor:
                    schema.verify_brain_schema_v7_in_transaction(cursor)
        connection.rollback()
        schema.verify_brain_schema_v7(connection)


def test_v7_existing_install_rejects_supplied_controller_topology_before_repair(brain_pg) -> None:
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v7_authority(connection)
        connection.execute(
            "CREATE ROLE brain_v7_supplied_wrong_controller LOGIN NOINHERIT NOSUPERUSER "
            "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
        )
        connection.commit()
        before_topology = connection.execute(
            "SELECT singleton_id,controller_role_oid FROM brain_v7_topology_contract"
        ).fetchall()
        before_memberships = connection.execute(
            "SELECT member,grantor,admin_option,inherit_option,set_option "
            "FROM pg_auth_members WHERE roleid='brain_policy_controller'::regrole"
        ).fetchall()
        topology = pg_roles.BootstrapTopology(
            database_owner_role="postgres",
            controller_role="brain_v7_supplied_wrong_controller",
            verifier_role="brain_schema_verifier",
            migrator_role="brain_schema_migrator",
            retired_admin_roles=(),
            infrastructure_superuser_roles=("postgres",),
        )
        with pytest.raises(RuntimeError, match="persisted.*controller identity mismatch"):
            with connection.transaction():
                with connection.cursor() as cursor:
                    pg_roles._install_brain_authority_in_transaction(cursor, topology=topology)
        connection.rollback()
        assert connection.execute(
            "SELECT singleton_id,controller_role_oid FROM brain_v7_topology_contract"
        ).fetchall() == before_topology
        assert connection.execute(
            "SELECT member,grantor,admin_option,inherit_option,set_option "
            "FROM pg_auth_members WHERE roleid='brain_policy_controller'::regrole"
        ).fetchall() == before_memberships
        connection.commit()
        schema.ensure_brain_schema_v7(
            connection,
            expected_controller_role="brain_controller_test_login",
        )
        with pytest.raises(RuntimeError, match="supplied controller topology mismatch"):
            schema.ensure_brain_schema_v7(
                connection,
                expected_controller_role="brain_v7_supplied_wrong_controller",
            )
        schema.verify_brain_schema_v7(connection)


def test_v7_topology_relation_rejects_structural_and_default_acl_drift(brain_pg) -> None:
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v7_authority(connection)
        controller_oid = connection.execute(
            "SELECT controller_role_oid FROM brain_v7_topology_contract"
        ).fetchone()["controller_role_oid"]
        connection.commit()

        with pytest.raises(psycopg.errors.CheckViolation):
            with connection.transaction():
                connection.execute(
                    "INSERT INTO brain_v7_topology_contract(singleton_id,controller_role_oid) "
                    "VALUES (2,%s)",
                    (controller_oid,),
                )
        connection.rollback()

        mutations = (
            ("ALTER TABLE brain_v7_topology_contract OWNER TO postgres", "topology relation"),
            (
                "ALTER TABLE brain_v7_topology_contract "
                "DROP CONSTRAINT brain_v7_topology_contract_singleton_check",
                "topology constraint",
            ),
            (
                "DROP TRIGGER brain_v7_topology_contract_append_only "
                "ON brain_v7_topology_contract",
                "topology trigger",
            ),
            (
                "ALTER TABLE brain_v7_topology_contract "
                "ALTER COLUMN controller_role_oid DROP NOT NULL",
                "topology column",
            ),
        )
        for statement, message in mutations:
            with pytest.raises(RuntimeError, match=message):
                with connection.transaction():
                    connection.execute(statement)
                    with connection.cursor() as cursor:
                        schema.verify_brain_schema_v7_in_transaction(cursor)
            connection.rollback()

        with pytest.raises(RuntimeError, match="authority default ACL leakage"):
            with connection.transaction():
                connection.execute(
                    "ALTER DEFAULT PRIVILEGES FOR ROLE brain_schema_migrator IN SCHEMA public "
                    "GRANT UPDATE ON TABLES TO brain_artifact_authority_writer"
                )
                with connection.cursor() as cursor:
                    schema.verify_brain_schema_v7_in_transaction(cursor)
        connection.rollback()
        schema.verify_brain_schema_v7(connection)


def test_v7_verifier_rejects_public_and_arbitrary_inherited_authority_acls(brain_pg) -> None:
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v7_authority(connection)
        connection.execute("CREATE ROLE brain_v7_rogue NOLOGIN")
        connection.commit()
        mutations = (
            "GRANT SELECT ON brain_artifact_authority_requests TO brain_v7_rogue",
            "GRANT SELECT ON brain_schema_versions TO PUBLIC",
            "GRANT EXECUTE ON FUNCTION brain_v5_frame(text) TO brain_v7_rogue",
        )
        for statement in mutations:
            with pytest.raises(RuntimeError, match="ACL"):
                with connection.transaction():
                    connection.execute(statement)
                    with connection.cursor() as cursor:
                        schema.verify_brain_schema_v7_in_transaction(cursor)
            connection.rollback()
        schema.verify_brain_schema_v7(connection)

def test_v7_verifier_deeply_rejects_inherited_v1_v5_v6_drift(brain_pg) -> None:
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v7_authority(connection)

        mutations = (
            (
                "CREATE OR REPLACE FUNCTION brain_v5_frame(requested_value text) RETURNS text "
                "LANGUAGE sql IMMUTABLE SET search_path=pg_catalog, public "
                "AS 'SELECT requested_value || ''tampered'''",
                "v5",
            ),
            (
                "ALTER TABLE brain_artifact_authority_requests "
                "DROP CONSTRAINT brain_artifact_authority_requests_manifest_sha256_check",
                "v6",
            ),
            (
                "DROP TRIGGER brain_artifact_authority_registrations_append_only "
                "ON brain_artifact_authority_registrations",
                "v6",
            ),
            ("ALTER TABLE brain_schema_versions OWNER TO postgres", "inherited"),
        )
        for statement, message in mutations:
            with pytest.raises(RuntimeError, match=message):
                with connection.transaction():
                    connection.execute(statement)
                    with connection.cursor() as cursor:
                        schema.verify_brain_schema_v7_in_transaction(cursor)
            connection.rollback()
        schema.verify_brain_schema_v7(connection)

def test_v7_verifier_rejects_owner_acl_sequence_and_default_acl_drift(brain_pg) -> None:
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v7_authority(connection)

        with pytest.raises(RuntimeError, match="support function ownership mismatch"):
            with connection.transaction():
                connection.execute(
                    "ALTER FUNCTION brain_snapshot_binding_is_authoritative(text,text,text,text) "
                    "OWNER TO postgres"
                )
                with connection.cursor() as cursor:
                    schema.verify_brain_schema_v7_in_transaction(cursor)
        connection.rollback()

        with pytest.raises(
            RuntimeError,
            match="registration function ACL mismatch|function ACL contract mismatch",
        ):
            with connection.transaction():
                connection.execute(
                    "GRANT EXECUTE ON FUNCTION brain_register_authoritative_artifact_manifest("
                    "uuid,text,text,text,timestamptz,timestamptz,text,text,jsonb) TO PUBLIC"
                )
                with connection.cursor() as cursor:
                    schema.verify_brain_schema_v7_in_transaction(cursor)
        connection.rollback()

        with pytest.raises(
            RuntimeError,
            match="authority relation ownership mismatch|inherited relation ownership mismatch",
        ):
            with connection.transaction():
                connection.execute("ALTER TABLE brain_artifact_locations OWNER TO postgres")
                with connection.cursor() as cursor:
                    schema.verify_brain_schema_v7_in_transaction(cursor)
        connection.rollback()

        with pytest.raises(RuntimeError, match="authority default ACL leakage"):
            with connection.transaction():
                connection.execute(
                    "ALTER DEFAULT PRIVILEGES FOR ROLE brain_artifact_authority_owner IN SCHEMA public "
                    "GRANT SELECT ON TABLES TO brain_artifact_authority_writer"
                )
                with connection.cursor() as cursor:
                    schema.verify_brain_schema_v7_in_transaction(cursor)
        connection.rollback()
        schema.verify_brain_schema_v7(connection)


def _bootstrap_script_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap-fleet-pg-roles.py"
    specification = importlib.util.spec_from_file_location("bootstrap_fleet_pg_roles", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


AUTHORITY_TABLES = {
    "brain_schema_versions",
    "brain_artifacts",
    "brain_artifact_locations",
    "brain_jobs",
    "brain_job_aliases",
    "brain_job_observations",
    "brain_label_events",
    "brain_pairwise_events",
    "brain_email_events",
    "brain_reviewed_outcomes",
    "brain_applications",
    "brain_application_events",
    "brain_decision_policies",
    "brain_policy_artifacts",
    "brain_policy_approvals",
    "brain_policy_gate_definitions",
    "brain_policy_release_gate_events",
    "brain_policy_transition_receipts",
    "brain_policy_activation_receipts",
    "brain_canary_lifecycle_events",
    "brain_decision_identities",
    "brain_job_decisions",
    "brain_job_decisions_default",
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
FUNCTIONS = (
    "brain_reject_mutation",
    "brain_register_decision",
    "brain_check_policy_lifecycle",
    "brain_check_supersession",
    "brain_reject_default_decision",
    "brain_check_parity_pass",
    "brain_check_archive_manifest",
)
TEST_ROLES = (
    "brain_schema_reader",
    "brain_create_role",
    "brain_broad_role",
    "brain_proxy_owner",
    "fleet_worker",
    "brain_status_test_login",
    "brain_controller_test_login",
    "v6_controller_login",
    "ats_worker_role",
    "linkedin_worker_role",
    "brain_rogue_writer",
    "brain_v7_provider",
)
FIXED_BRAIN_ROLES = (
    "brain_artifact_authority_owner",
    "brain_artifact_authority_writer",
    "brain_candidate_reader",
    "brain_candidate_writer",
    "brain_graph_authority",
    "brain_status_reader",
    "brain_policy_controller",
    "brain_schema_verifier",
    "brain_schema_migrator",
)
HASHES = tuple(character * 64 for character in "abcdef0123456789")


def _drop_fixed_role(conn, role: str) -> None:
    if conn.execute("SELECT 1 FROM pg_namespace WHERE nspname='public'").fetchone() is not None:
        schema_grantors = conn.execute(
            "SELECT DISTINCT grantor.rolname AS grantor FROM pg_namespace n "
            "CROSS JOIN LATERAL aclexplode(n.nspacl) acl JOIN pg_roles grantor ON grantor.oid=acl.grantor "
            "WHERE n.nspname='public' AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s)",
            (role,),
        ).fetchall()
        for grantor in schema_grantors:
            conn.execute(pg_sql.SQL("SET LOCAL ROLE {}").format(pg_sql.Identifier(grantor["grantor"])))
            conn.execute(
                pg_sql.SQL("REVOKE ALL PRIVILEGES ON SCHEMA public FROM {} CASCADE").format(
                    pg_sql.Identifier(role)
                )
            )
            conn.execute("RESET ROLE")
    column_acls = conn.execute(
        "SELECT n.nspname,c.relname,a.attname,acl.privilege_type,grantor.rolname AS grantor "
        "FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid "
        "JOIN pg_namespace n ON n.oid=c.relnamespace "
        "CROSS JOIN LATERAL aclexplode(a.attacl) acl "
        "JOIN pg_roles grantor ON grantor.oid=acl.grantor "
        "WHERE a.attnum>0 AND NOT a.attisdropped "
        "AND acl.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s)",
        (role,),
    ).fetchall()
    for acl in column_acls:
        conn.execute(pg_sql.SQL("SET LOCAL ROLE {}").format(pg_sql.Identifier(acl["grantor"])))
        conn.execute(
            pg_sql.SQL("REVOKE {} ({}) ON TABLE {}.{} FROM {} CASCADE").format(
                pg_sql.SQL(acl["privilege_type"]),
                pg_sql.Identifier(acl["attname"]),
                pg_sql.Identifier(acl["nspname"]),
                pg_sql.Identifier(acl["relname"]),
                pg_sql.Identifier(role),
            )
        )
        conn.execute("RESET ROLE")
    conn.execute(pg_sql.SQL("DROP OWNED BY {} CASCADE").format(pg_sql.Identifier(role)))
    conn.execute(pg_sql.SQL("DROP ROLE {}").format(pg_sql.Identifier(role)))


def test_fixed_role_cleanup_requires_environment_and_database_markers(fleet_db, monkeypatch):
    with psycopg.connect(fleet_db, row_factory=dict_row) as conn:
        monkeypatch.delenv("APPLYPILOT_PGTEST_DISPOSABLE", raising=False)
        with pytest.raises(RuntimeError, match="APPLYPILOT_PGTEST_DISPOSABLE"):
            require_disposable_postgres(conn)

        monkeypatch.setenv("APPLYPILOT_PGTEST_DISPOSABLE", "1")
        database_name = conn.execute("SELECT current_database() AS name").fetchone()["name"]
        conn.execute(pg_sql.SQL("COMMENT ON DATABASE {} IS 'wrong-marker'").format(pg_sql.Identifier(database_name)))
        conn.commit()
        try:
            with pytest.raises(RuntimeError, match="exact disposable database comment"):
                require_disposable_postgres(conn)
        finally:
            conn.execute(
                pg_sql.SQL("COMMENT ON DATABASE {} IS {}").format(
                    pg_sql.Identifier(database_name),
                    pg_sql.Literal(DISPOSABLE_DATABASE_COMMENT),
                )
            )
            conn.commit()
        cluster_nonce = os.environ[DISPOSABLE_CLUSTER_MARKER_ENV]
        monkeypatch.delenv(DISPOSABLE_CLUSTER_MARKER_ENV)
        with pytest.raises(RuntimeError, match="cluster marker nonce"):
            require_disposable_postgres(conn)
        monkeypatch.setenv(DISPOSABLE_CLUSTER_MARKER_ENV, cluster_nonce)
        require_disposable_postgres(conn)


def test_cluster_cleanup_rejects_another_connectable_database(fleet_db):
    other_database = "applypilot_shared_cluster_probe"
    with psycopg.connect(fleet_db, row_factory=dict_row, autocommit=True) as conn:
        require_disposable_postgres(conn)
        conn.execute(pg_sql.SQL("CREATE DATABASE {}").format(pg_sql.Identifier(other_database)))
        try:
            with pytest.raises(RuntimeError, match="no other connectable databases"):
                require_disposable_postgres(conn)
        finally:
            conn.execute(pg_sql.SQL("DROP DATABASE {} WITH (FORCE)").format(pg_sql.Identifier(other_database)))
        require_disposable_postgres(conn)


def test_unprivileged_login_cannot_forge_disposable_cluster_or_database_markers(fleet_db):
    role_name = "disposable_marker_forgery_test"
    password = "disposable-marker-forgery-password"
    forged_role = "forged_disposable_cluster_marker"
    with psycopg.connect(fleet_db, row_factory=dict_row) as provider:
        require_disposable_postgres(provider)
        database_name = provider.execute("SELECT current_database() AS name").fetchone()["name"]
        provider.execute(
            pg_sql.SQL(
                "CREATE ROLE {} LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE "
                "NOREPLICATION NOBYPASSRLS PASSWORD {}"
            ).format(pg_sql.Identifier(role_name), pg_sql.Literal(password))
        )
        provider.execute(
            pg_sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                pg_sql.Identifier(database_name), pg_sql.Identifier(role_name)
            )
        )
        provider.commit()
        parameters = conninfo_to_dict(fleet_db)
        parameters.update(user=role_name, password=password)
        try:
            with psycopg.connect(make_conninfo(**parameters), row_factory=dict_row) as attacker:
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    attacker.execute(
                        pg_sql.SQL("COMMENT ON DATABASE {} IS {}").format(
                            pg_sql.Identifier(database_name),
                            pg_sql.Literal(DISPOSABLE_DATABASE_COMMENT),
                        )
                    )
                attacker.rollback()
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    attacker.execute(
                        pg_sql.SQL("COMMENT ON ROLE {} IS {}").format(
                            pg_sql.Identifier(DISPOSABLE_CLUSTER_MARKER_ROLE),
                            pg_sql.Literal(DISPOSABLE_CLUSTER_COMMENT_PREFIX + "forged"),
                        )
                    )
                attacker.rollback()
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    attacker.execute(pg_sql.SQL("CREATE ROLE {}").format(pg_sql.Identifier(forged_role)))
        finally:
            provider.rollback()
            require_disposable_postgres(provider)
            provider.execute(pg_sql.SQL("DROP ROLE IF EXISTS {}").format(pg_sql.Identifier(forged_role)))
            provider.execute(
                pg_sql.SQL("REVOKE CONNECT ON DATABASE {} FROM {}").format(
                    pg_sql.Identifier(database_name), pg_sql.Identifier(role_name)
                )
            )
            provider.execute(pg_sql.SQL("DROP OWNED BY {}").format(pg_sql.Identifier(role_name)))
            provider.execute(pg_sql.SQL("DROP ROLE {}").format(pg_sql.Identifier(role_name)))
            provider.commit()
        require_disposable_postgres(provider)


def _cleanup(dsn: str, *, drop_fixed_roles: bool = False) -> None:
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        require_disposable_postgres(conn)
        conn.execute("DROP SCHEMA IF EXISTS brain_archive CASCADE")
        conn.execute("DROP TABLE IF EXISTS public.brain_job_decisions CASCADE")
        relations = conn.execute(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname='public' AND left(c.relname,6)='brain_' "
            "AND c.relkind IN ('r','p','v','m','f') ORDER BY c.relname DESC"
        ).fetchall()
        for relation in relations:
            conn.execute(
                pg_sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                    pg_sql.Identifier("public"), pg_sql.Identifier(relation["relname"])
                )
            )
        for table in sorted(AUTHORITY_TABLES, reverse=True):
            conn.execute(f"DROP TABLE IF EXISTS public.{table} CASCADE")
        functions = conn.execute(
            "SELECT p.proname, pg_get_function_identity_arguments(p.oid) AS arguments "
            "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
            "WHERE n.nspname='public' AND left(p.proname,6)='brain_'"
        ).fetchall()
        for function in functions:
            conn.execute(
                pg_sql.SQL("DROP FUNCTION {}.{}({}) CASCADE").format(
                    pg_sql.Identifier("public"),
                    pg_sql.Identifier(function["proname"]),
                    pg_sql.SQL(function["arguments"]),
                )
            )
        for role in TEST_ROLES:
            if conn.execute("SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = %s) AS present", (role,)).fetchone()[
                "present"
            ]:
                conn.execute(f"REVOKE ALL PRIVILEGES ON SCHEMA public FROM {role}")
                conn.execute(f"REVOKE CREATE ON DATABASE postgres FROM {role}")
                conn.execute(
                    f"ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public REVOKE ALL ON TABLES FROM {role}"
                )
                conn.execute(
                    f"ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public REVOKE ALL ON SEQUENCES FROM {role}"
                )
                conn.execute(
                    f"ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public REVOKE ALL ON FUNCTIONS FROM {role}"
                )
                conn.execute(f"DROP OWNED BY {role}")
                conn.execute(f"DROP ROLE {role}")
        if drop_fixed_roles:
            for role in FIXED_BRAIN_ROLES:
                if conn.execute(
                    "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname=%s) AS present",
                    (role,),
                ).fetchone()["present"]:
                    _drop_fixed_role(conn, role)
        conn.execute("ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public REVOKE ALL ON TABLES FROM PUBLIC")
        conn.execute("ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public REVOKE ALL ON SEQUENCES FROM PUBLIC")
        conn.execute("ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public REVOKE ALL ON FUNCTIONS FROM PUBLIC")
        conn.commit()


@pytest.fixture
def brain_pg(fleet_db):
    _cleanup(fleet_db, drop_fixed_roles=True)
    with psycopg.connect(fleet_db, row_factory=dict_row) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS public.fleet_desired_state("
            "machine_owner text primary key,desired_workers integer NOT NULL,agent text,model text,"
            "generation integer,updated_at timestamptz NOT NULL DEFAULT now())"
        )
        for column in (
            "ats_canary_worker_id",
            "ats_canary_version",
            "linkedin_canary_worker_id",
            "linkedin_canary_version",
        ):
            conn.execute(f"ALTER TABLE public.fleet_config ADD COLUMN IF NOT EXISTS {column} text")
        conn.execute(
            "DO $$ BEGIN CREATE ROLE brain_schema_migrator NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$"
        )
        conn.execute(
            "DO $$ BEGIN CREATE ROLE brain_schema_verifier LOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$"
        )
        for role in (
            "brain_status_reader",
            "brain_policy_controller",
            "brain_candidate_reader",
            "brain_candidate_writer",
            "brain_graph_authority",
            "brain_artifact_authority_owner",
            "brain_artifact_authority_writer",
        ):
            conn.execute(
                pg_sql.SQL(
                    "DO $$ BEGIN CREATE ROLE {} NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB "
                    "NOCREATEROLE NOREPLICATION NOBYPASSRLS; "
                    "EXCEPTION WHEN duplicate_object THEN NULL; END $$"
                ).format(pg_sql.Identifier(role))
            )
        conn.execute(
            "GRANT brain_artifact_authority_owner TO brain_schema_migrator WITH INHERIT FALSE, SET TRUE"
        )
        conn.execute(
            "GRANT USAGE ON SCHEMA public TO brain_artifact_authority_owner, brain_artifact_authority_writer"
        )
        _set_role_password(conn, fleet_db, "brain_schema_verifier")
        conn.execute("GRANT brain_schema_migrator TO postgres")
        conn.execute("GRANT USAGE, CREATE ON SCHEMA public TO brain_schema_migrator")
        conn.execute("GRANT CREATE ON DATABASE postgres TO brain_schema_migrator")
        conn.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO brain_schema_migrator WITH GRANT OPTION")
        for relation in (
            "fleet_config",
            "fleet_decision_policies",
            "apply_queue",
            "linkedin_queue",
            "workers",
            "worker_heartbeat",
            "fleet_worker_principals",
            "fleet_desired_state",
            "rate_governor",
        ):
            if conn.execute("SELECT to_regclass(%s) AS relation", (f"public.{relation}",)).fetchone()["relation"]:
                conn.execute(
                    pg_sql.SQL(
                        "GRANT SELECT, INSERT, UPDATE ON TABLE {}.{} TO brain_schema_migrator WITH GRANT OPTION"
                    ).format(pg_sql.Identifier("public"), pg_sql.Identifier(relation))
                )
        conn.commit()
    yield fleet_db
    _cleanup(fleet_db, drop_fixed_roles=True)


class _Pg18CatalogShapeCursor:
    def __init__(
        self,
        *,
        version: int = 180004,
        shapes: dict[str, tuple[tuple[str, str], ...]] | None = None,
    ) -> None:
        self.version = version
        self.shapes = dict(schema._PG18_CATALOG_SHAPES if shapes is None else shapes)
        self._one = None
        self._many = []

    def execute(self, statement: str, parameters=None) -> None:
        if "server_version_num" in statement:
            self._one = {"server_version_num": self.version}
            self._many = []
            return
        relation = parameters[0]
        self._one = None
        self._many = [
            {"attname": name, "data_type": data_type}
            for name, data_type in self.shapes[relation]
        ]

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._many


class _CatalogConstraintQueryCursor:
    def __init__(self) -> None:
        self.statement = ""

    def execute(self, statement: str, _parameters=None) -> None:
        self.statement = statement

    def fetchall(self):
        return []


def test_unsupported_major_pg18_catalog_guard_rejects_wrong_major_double() -> None:
    with pytest.raises(
        RuntimeError,
        match="PostgreSQL 18 authority catalog contract required",
    ):
        schema.require_pg18_authority_catalog(_Pg18CatalogShapeCursor(version=170999))


@pytest.mark.parametrize("defect", ("extra", "missing", "reordered", "retyped"))
def test_pg18_catalog_shape_guard_rejects_adversarial_double(defect: str) -> None:
    shapes = dict(schema._PG18_CATALOG_SHAPES)
    relation = "pg_catalog.pg_constraint"
    columns = list(shapes[relation])
    if defect == "extra":
        columns.append(("unexpected_column", "text"))
    elif defect == "missing":
        columns.pop()
    elif defect == "reordered":
        columns[0], columns[1] = columns[1], columns[0]
    else:
        name, _data_type = columns[0]
        columns[0] = (name, "text")
    shapes[relation] = tuple(columns)

    with pytest.raises(
        RuntimeError,
        match=r"PostgreSQL 18 authority catalog shape mismatch for pg_catalog\.pg_constraint",
    ):
        schema.require_pg18_authority_catalog(_Pg18CatalogShapeCursor(shapes=shapes))


def test_pg18_catalog_shape_guard_accepts_exact_double() -> None:
    schema.require_pg18_authority_catalog(_Pg18CatalogShapeCursor())


def test_pg18_catalog_pins_match_authoritative_fresh_fixtures() -> None:
    assert dict(schema._PG_CATALOG_HASHES[18]) == {
        "base": "22cfe2c608305d89dd773a0f0a4ed91a35be066aab0468154de08be2361b2c6a",
        "current_base": "bc4b812202b72029bd2422fff11b2d803bfe4f59283ae0fe8faddcf79dfe4648",
        "v5": "c84d51b987a3a6915d3af9f186c3c179f09828c2e1e98f5fe04be60e9c063ad7",
        "current_v5": "903d96ec8c9f0da427771ceae78a5c4d4b39fb38bb72a39bd8894e75b3b05bb1",
        "v6": "413fdaacc60793dab624b7d12def0362d9539521be055d716bc9320450362329",
        "v7": "b8bcc3e3fedea82fe0149d3dd0372332e24b94579c692e949d9b9c3dce8a0ee5",
        "v8": "b1ae7f1df9e7c902d4954b4f1fa29c3fdf9d4c55516ff1f5011375b23163cebf",
        "v9": "322c814379eb24c36a09a4f154efe15b986055f71433733d59d7bcbfc49a6102",
    }


def test_pg18_catalog_constraint_query_uses_explicit_qualified_identities() -> None:
    cursor = _CatalogConstraintQueryCursor()
    schema._catalog_constraint_records(
        cursor,
        ["brain_schema_versions"],
        include_archive=True,
    )

    for oid_text_cast in ("::regtype::text", "::regclass::text", "::regoperator::text"):
        assert oid_text_cast not in cursor.statement
    for catalog_join in (
        "JOIN pg_type constraint_type",
        "JOIN pg_class backing_index",
        "JOIN pg_class referenced_relation",
        "JOIN pg_operator operator",
        "JOIN pg_type left_type",
        "JOIN pg_type right_type",
    ):
        assert catalog_join in cursor.statement


def _pg18_boundary_snapshot(connection) -> dict[str, object]:
    roles = connection.execute(
        "SELECT rolname,rolcanlogin,rolinherit,rolsuper,rolcreatedb,rolcreaterole,"
        "rolreplication,rolbypassrls FROM pg_roles "
        "WHERE rolname LIKE 'brain\\_%' ESCAPE '\\' ORDER BY rolname"
    ).fetchall()
    relations = connection.execute(
        "SELECT n.nspname,c.relname,c.relkind,owner.rolname AS owner_name "
        "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "JOIN pg_roles owner ON owner.oid=c.relowner "
        "WHERE n.nspname IN ('public','brain_archive') "
        "ORDER BY n.nspname,c.relname,c.relkind"
    ).fetchall()
    version_relation = connection.execute(
        "SELECT to_regclass('public.brain_schema_versions') AS relation"
    ).fetchone()["relation"]
    versions = []
    if version_relation is not None:
        versions = connection.execute(
            "SELECT version,migration_name,migration_checksum,applied_by "
            "FROM public.brain_schema_versions ORDER BY version"
        ).fetchall()
    advisory_locks = connection.execute(
        "SELECT classid,objid,objsubid,mode,granted FROM pg_locks "
        "WHERE pid=pg_backend_pid() AND locktype='advisory' "
        "ORDER BY classid,objid,objsubid,mode"
    ).fetchall()
    connection.commit()
    return {
        "roles": roles,
        "relations": relations,
        "versions": versions,
        "advisory_locks": advisory_locks,
    }


def test_unsupported_major_pg16_rejects_every_direct_authority_boundary_without_database_drift(
    brain_pg,
) -> None:
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        version = int(connection.execute("SHOW server_version_num").fetchone()["server_version_num"])
        connection.commit()
        if version // 10000 == 18:
            pytest.skip("unsupported-major rejection requires the fixed PostgreSQL 16 fixture")
        before = _pg18_boundary_snapshot(connection)
        topology = pg_roles.BootstrapTopology(
            database_owner_role="postgres",
            controller_role="brain_controller_test_login",
            verifier_role="brain_schema_verifier",
            migrator_role="brain_schema_migrator",
            retired_admin_roles=(),
            infrastructure_superuser_roles=("postgres",),
        )

        connection_boundaries = (
            schema.ensure_brain_schema_v1,
            schema.verify_brain_schema_v1,
            schema.ensure_brain_schema_v4,
            schema.verify_brain_schema_v4,
            schema.ensure_brain_schema_v5,
            schema.verify_brain_schema_v5,
            schema.ensure_brain_schema_v6,
            schema.verify_brain_schema_v6,
            lambda conn: schema.ensure_brain_schema_v7(
                conn, expected_controller_role="brain_controller_test_login"
            ),
            schema.verify_brain_schema_v7,
        )
        in_transaction_boundaries = (
            schema.ensure_brain_schema_v1_in_transaction,
            schema.ensure_brain_schema_v4_in_transaction,
            schema.verify_brain_schema_v4_in_transaction,
            schema.ensure_brain_schema_v5_in_transaction,
            schema.verify_brain_schema_v5_in_transaction,
            schema.ensure_brain_schema_v6_in_transaction,
            schema.verify_brain_schema_v6_in_transaction,
            lambda cur: schema.ensure_brain_schema_v7_in_transaction(
                cur, expected_controller_role="brain_controller_test_login"
            ),
            schema.verify_brain_schema_v7_in_transaction,
            pg_roles.ensure_brain_candidate_roles_in_transaction,
            pg_roles.ensure_brain_artifact_authority_roles_in_transaction,
            lambda cur: pg_roles._install_brain_authority_in_transaction(
                cur, topology=topology
            ),
        )

        for boundary in connection_boundaries:
            with pytest.raises(
                RuntimeError,
                match="PostgreSQL 18 authority catalog contract required",
            ):
                boundary(connection)
            connection.rollback()
            assert _pg18_boundary_snapshot(connection) == before
        for boundary in in_transaction_boundaries:
            with pytest.raises(
                RuntimeError,
                match="PostgreSQL 18 authority catalog contract required",
            ):
                with connection.transaction():
                    with connection.cursor() as cursor:
                        boundary(cursor)
            connection.rollback()
            assert _pg18_boundary_snapshot(connection) == before


def _dsn_for(dsn: str, role: str) -> str:
    return make_conninfo(dsn, user=role)


def _set_role_password(conn, dsn: str, role: str) -> None:
    password = conninfo_to_dict(dsn).get("password")
    if password:
        conn.execute(
            pg_sql.SQL("ALTER ROLE {} PASSWORD {}").format(
                pg_sql.Identifier(role),
                pg_sql.Literal(password),
            )
        )


def _create_lifecycle_login(conn, dsn: str, login: str, capability: str) -> None:
    conn.execute(
        pg_sql.SQL("CREATE ROLE {} LOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS").format(
            pg_sql.Identifier(login)
        )
    )
    _set_role_password(conn, dsn, login)
    conn.execute(
        pg_sql.SQL("GRANT {} TO {}").format(
            pg_sql.Identifier(capability),
            pg_sql.Identifier(login),
        )
    )


def test_v3_lifecycle_principals_enforce_read_and_mutation_boundaries(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as owner:
        schema.ensure_schema_v1(owner)
        _create_lifecycle_login(
            owner,
            brain_pg,
            "brain_status_test_login",
            "brain_status_reader",
        )
        _create_lifecycle_login(
            owner,
            brain_pg,
            "brain_controller_test_login",
            "brain_policy_controller",
        )
        owner.commit()

    with psycopg.connect(
        _dsn_for(brain_pg, "brain_status_test_login"),
        row_factory=dict_row,
    ) as status_conn:
        result = authority_status(status_conn)
        assert result["authority"] == "postgres_staging_candidate"
        assert status_conn.execute(
            "SELECT ats_canary_worker_id,ats_canary_version,linkedin_canary_worker_id,"
            "linkedin_canary_version FROM public.fleet_config WHERE id=1"
        ).fetchone() == {
            "ats_canary_worker_id": None,
            "ats_canary_version": None,
            "linkedin_canary_worker_id": None,
            "linkedin_canary_version": None,
        }
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            status_conn.execute("UPDATE public.fleet_config SET paused=TRUE WHERE id=1")
        status_conn.rollback()

    with psycopg.connect(
        _dsn_for(brain_pg, "brain_controller_test_login"),
        row_factory=dict_row,
    ) as controller:
        with pytest.raises(psycopg.errors.NoDataFound, match="unknown policy_version"):
            controller.execute("SELECT public.brain_controller_transition_policy('missing-policy','validated',NULL)")
        controller.rollback()
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            controller.execute("SELECT public.brain_transition_policy('missing-policy','validated')")
        controller.rollback()
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            controller.execute("UPDATE public.fleet_config SET paused=TRUE WHERE id=1")
        controller.rollback()
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            controller.execute(
                "INSERT INTO public.brain_decision_policies(policy_version,lane,lifecycle) "
                "VALUES('forbidden','ats','draft')"
            )
        controller.rollback()

    with psycopg.connect(brain_pg, row_factory=dict_row) as owner:
        owner.execute("CREATE ROLE brain_rogue_writer NOLOGIN")
        owner.execute("GRANT UPDATE ON public.fleet_config TO brain_rogue_writer")
        owner.execute("GRANT brain_rogue_writer TO brain_controller_test_login")
        owner.commit()
        with pytest.raises(RuntimeError, match="lifecycle login brain_controller_test_login exceeds"):
            schema.ensure_schema_v1(owner)
    with psycopg.connect(
        _dsn_for(brain_pg, "brain_controller_test_login"),
        row_factory=dict_row,
    ) as controller:
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="outside its lifecycle capability"):
            controller.execute("SELECT public.brain_controller_transition_policy('missing-policy','validated',NULL)")
        controller.rollback()
    with psycopg.connect(brain_pg, row_factory=dict_row) as owner:
        owner.execute("REVOKE brain_rogue_writer FROM brain_controller_test_login")
        owner.execute("GRANT UPDATE (paused) ON public.fleet_config TO brain_controller_test_login")
        owner.commit()
        with pytest.raises(RuntimeError, match="lifecycle login brain_controller_test_login exceeds"):
            schema.ensure_schema_v1(owner)
    with psycopg.connect(
        _dsn_for(brain_pg, "brain_controller_test_login"),
        row_factory=dict_row,
    ) as controller:
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="outside its lifecycle capability"):
            controller.execute("SELECT public.brain_controller_transition_policy('missing-policy','validated',NULL)")


def test_v3_controller_arms_and_stops_real_postgres_ats_canary(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as owner:
        schema.ensure_schema_v1(owner)
        _create_lifecycle_login(
            owner,
            brain_pg,
            "brain_controller_test_login",
            "brain_policy_controller",
        )
        owner.commit()
        _activate_controller(owner)
        owner.execute(
            "INSERT INTO public.brain_decision_policies(policy_version,lane,lifecycle) "
            "VALUES('ats-canary-real','ats','draft')"
        )
        owner.execute("SET LOCAL ROLE NONE")
        owner.execute("ALTER TABLE public.brain_decision_policies DISABLE TRIGGER brain_decision_policies_lifecycle")
        owner.execute(
            "UPDATE public.brain_decision_policies SET lifecycle='canary',"
            "validated_at=now(),canary_at=now() "
            "WHERE policy_version='ats-canary-real'"
        )
        owner.execute("ALTER TABLE public.brain_decision_policies ENABLE TRIGGER brain_decision_policies_lifecycle")
        owner.commit()
        owner.execute("RESET ROLE")
        owner.execute(
            "INSERT INTO public.fleet_decision_policies(policy_version,lane,status) "
            "VALUES('ats-canary-real','ats','canary')"
        )
        owner.execute(
            "INSERT INTO public.fleet_desired_state("
            "machine_owner,desired_workers,agent,model,generation,updated_at) "
            "VALUES('GGGTower',1,'codex','test-model',1,now())"
        )
        owner.execute(
            "INSERT INTO public.workers(worker_id,machine_owner,public_ip,validated,capabilities) "
            "VALUES('ats-canary-worker','GGGTower','203.0.113.10',TRUE,'{\"can_ats\":true}')"
        )
        owner.execute(
            "INSERT INTO public.worker_heartbeat(worker_id,machine_owner,home_ip,role,state,sw_version,last_beat) "
            "VALUES('ats-canary-worker','GGGTower','203.0.113.10','apply','idle','release-v2',now())"
        )
        owner.execute("CREATE ROLE ats_worker_role LOGIN INHERIT")
        _set_role_password(owner, brain_pg, "ats_worker_role")
        owner.execute("GRANT USAGE ON SCHEMA public TO ats_worker_role")
        owner.execute("GRANT USAGE ON TYPE public.apply_queue TO ats_worker_role")
        owner.execute(
            "GRANT EXECUTE ON FUNCTION public.fleet_worker_lease_ats(TEXT,TEXT,INTEGER,TEXT,INTEGER) TO ats_worker_role"
        )
        owner.execute(
            "INSERT INTO public.fleet_worker_principals(role_name,worker_id,contract) "
            "VALUES('ats_worker_role','ats-canary-worker','apply')"
        )
        owner.execute(
            "INSERT INTO public.rate_governor(scope_key,daily_cap,count_24h,min_gap_seconds) VALUES "
            "('global',100,0,0),('home_ip:203.0.113.10',10,0,0),('host:example.test',10,0,0)"
        )
        owner.execute(
            "INSERT INTO public.apply_queue("
            "url,application_url,company,score,status,lane,approved_batch,dedup_key,apply_domain,"
            "target_host,decision_id,policy_version,decision_action,qualification_verdict,"
            "qualification_score,qualification_floor,preference_score,outcome_score,final_score,"
            "decision_confidence,decision_created_at,decision_expires_at,input_hash) VALUES("
            "'https://example.test/job','https://example.test/apply','Example',9,'queued','ats',"
            "'reviewed','ats-real-dedup','example.test','example.test','decision-real',"
            "'ats-canary-real','apply','qualified',9,7,9,9,9,.9,now(),now()+interval '1 day','hash-real')"
        )
        owner.execute(
            "UPDATE public.fleet_config SET paused=TRUE,ats_paused=TRUE,"
            "ats_pause_source='operator_test',ats_apply_mode='stopped',"
            "linkedin_apply_mode='stopped',canary_enabled=FALSE,"
            "linkedin_canary_enabled=FALSE,ats_policy_version=NULL,"
            "linkedin_policy_version=NULL,pinned_worker_version='release-v1',"
            "canary_worker_id='ats-canary-worker',canary_version='release-v2',"
            "ats_canary_worker_id='ats-canary-worker',ats_canary_version='release-v2',"
            "linkedin_canary_worker_id='preserved-linkedin-worker',"
            "linkedin_canary_version='preserved-linkedin-version',spend_cap_usd=0 "
            "WHERE id=1"
        )
        owner.commit()

    with psycopg.connect(
        _dsn_for(brain_pg, "brain_controller_test_login"),
        row_factory=dict_row,
    ) as controller:
        with pytest.raises(psycopg.errors.CheckViolation, match="invalid canary lane"):
            controller.execute(
                "SELECT public.brain_controller_arm_canary(%s,%s,%s,%s,%s,%s)",
                ("ats-canary-real", "ats", 1, "operator_test", False, 121),
            )
        controller.rollback()
        armed = arm_canary(
            controller,
            "ats-canary-real",
            "ats",
            1,
            expected_ats_pause_source="operator_test",
        )
        assert armed["worker_id"] == "ats-canary-worker"
        assert armed["expected_worker_version"] == "release-v2"
        assert armed["candidate_url"] == "https://example.test/job"
        with psycopg.connect(
            _dsn_for(brain_pg, "ats_worker_role"),
            row_factory=dict_row,
        ) as worker:
            leased = worker.execute("SELECT url FROM public.fleet_worker_lease_ats(NULL,NULL,1200,NULL,900)").fetchone()
            assert leased["url"] == "https://example.test/job"
            worker.commit()
        stopped = stop_canary(controller, "ats")
        assert stopped["fleet_config"]["paused"] is True
        assert stopped["fleet_config"]["ats_apply_mode"] == "stopped"
        assert stopped["fleet_config"]["ats_pause_source"] == "operator_test"
        assert stopped["fleet_config"]["ats_canary_worker_id"] is None
        assert stopped["fleet_config"]["ats_canary_version"] is None
        assert stopped["fleet_config"]["linkedin_canary_worker_id"] == "preserved-linkedin-worker"
        assert stopped["fleet_config"]["linkedin_canary_version"] == "preserved-linkedin-version"
    with psycopg.connect(brain_pg, row_factory=dict_row) as owner:
        events = owner.execute(
            "SELECT event_type,prior_ats_pause_source FROM public.brain_canary_lifecycle_events "
            "WHERE policy_version='ats-canary-real' ORDER BY event_id"
        ).fetchall()
        assert events == [
            {"event_type": "armed", "prior_ats_pause_source": "operator_test"},
            {"event_type": "stopped", "prior_ats_pause_source": "operator_test"},
        ]


def test_v3_controller_arms_and_stops_real_postgres_linkedin_canary(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as owner:
        schema.ensure_schema_v1(owner)
        _create_lifecycle_login(
            owner,
            brain_pg,
            "brain_controller_test_login",
            "brain_policy_controller",
        )
        owner.commit()
        _activate_controller(owner)
        owner.execute(
            "INSERT INTO public.brain_decision_policies(policy_version,lane,lifecycle) "
            "VALUES('linkedin-canary-real','linkedin','draft')"
        )
        owner.execute("SET LOCAL ROLE NONE")
        owner.execute("ALTER TABLE public.brain_decision_policies DISABLE TRIGGER brain_decision_policies_lifecycle")
        owner.execute(
            "UPDATE public.brain_decision_policies SET lifecycle='canary',"
            "validated_at=now(),canary_at=now() "
            "WHERE policy_version='linkedin-canary-real'"
        )
        owner.execute("ALTER TABLE public.brain_decision_policies ENABLE TRIGGER brain_decision_policies_lifecycle")
        owner.commit()
        owner.execute("RESET ROLE")
        owner.execute(
            "INSERT INTO public.fleet_decision_policies(policy_version,lane,status) "
            "VALUES('linkedin-canary-real','linkedin','canary')"
        )
        owner.execute(
            "INSERT INTO public.fleet_desired_state("
            "machine_owner,desired_workers,agent,model,generation,updated_at) "
            "VALUES('Tarpon',1,'codex','test-model',1,now())"
        )
        owner.execute(
            "INSERT INTO public.workers(worker_id,machine_owner,public_ip,validated,capabilities) "
            "VALUES('linkedin-canary-worker','Tarpon','203.0.113.20',TRUE,"
            "'{\"can_linkedin\":true}')"
        )
        owner.execute(
            "INSERT INTO public.worker_heartbeat(worker_id,machine_owner,home_ip,role,state,sw_version,last_beat) "
            "VALUES('linkedin-canary-worker','Tarpon','203.0.113.20','linkedin','idle','release-v2',now())"
        )
        owner.execute("CREATE ROLE linkedin_worker_role LOGIN INHERIT")
        _set_role_password(owner, brain_pg, "linkedin_worker_role")
        owner.execute("GRANT USAGE ON SCHEMA public TO linkedin_worker_role")
        owner.execute("GRANT USAGE ON TYPE public.linkedin_queue TO linkedin_worker_role")
        owner.execute(
            "GRANT EXECUTE ON FUNCTION "
            "public.fleet_worker_lease_linkedin(TEXT,TEXT,TEXT,INTEGER,TEXT) "
            "TO linkedin_worker_role"
        )
        owner.execute(
            "INSERT INTO public.fleet_worker_principals(role_name,worker_id,contract) "
            "VALUES('linkedin_worker_role','linkedin-canary-worker','linkedin')"
        )
        owner.execute(
            "INSERT INTO public.rate_governor(scope_key,daily_cap,count_24h,min_gap_seconds) VALUES "
            "('global',100,0,0),('account:linkedin',10,0,0)"
        )
        owner.execute(
            "INSERT INTO public.linkedin_queue("
            "url,application_url,company,score,status,lane,approved_batch,dedup_key,"
            "linkedin_resolve_status,linkedin_resolved_at,decision_id,policy_version,"
            "decision_action,qualification_verdict,qualification_score,qualification_floor,"
            "preference_score,outcome_score,final_score,decision_confidence,"
            "decision_created_at,decision_expires_at,input_hash) VALUES("
            "'https://linkedin.test/job','https://linkedin.test/apply','Example',9,'queued',"
            "'linkedin','reviewed','linkedin-real-dedup','easy_apply',now(),"
            "'decision-linkedin-real','linkedin-canary-real','apply','qualified',9,7,9,9,9,.9,"
            "now(),now()+interval '1 day','hash-linkedin-real')"
        )
        owner.execute(
            "UPDATE public.fleet_config SET paused=TRUE,ats_paused=TRUE,"
            "ats_pause_source='incident_hold',ats_apply_mode='stopped',"
            "linkedin_apply_mode='stopped',canary_enabled=FALSE,"
            "linkedin_canary_enabled=FALSE,ats_policy_version=NULL,"
            "linkedin_policy_version=NULL,pinned_worker_version='release-v1',"
            "canary_worker_id='linkedin-canary-worker',canary_version='release-v2',"
            "ats_canary_worker_id='preserved-ats-worker',"
            "ats_canary_version='preserved-ats-version',"
            "linkedin_canary_worker_id='linkedin-canary-worker',"
            "linkedin_canary_version='release-v2',"
            "linkedin_owner_ip='203.0.113.20',spend_cap_usd=0 WHERE id=1"
        )
        owner.commit()

    with psycopg.connect(
        _dsn_for(brain_pg, "brain_controller_test_login"),
        row_factory=dict_row,
    ) as controller:
        armed = arm_canary(controller, "linkedin-canary-real", "linkedin", 1)
        assert armed["worker_id"] == "linkedin-canary-worker"
        assert armed["candidate_url"] == "https://linkedin.test/job"
        assert armed["fleet_config"]["ats_pause_source"] == "incident_hold"
        with psycopg.connect(
            _dsn_for(brain_pg, "linkedin_worker_role"),
            row_factory=dict_row,
        ) as worker:
            leased = worker.execute(
                "SELECT url FROM public.fleet_worker_lease_linkedin(NULL,NULL,NULL,1200,NULL)"
            ).fetchone()
            assert leased["url"] == "https://linkedin.test/job"
            worker.rollback()
        stopped = stop_canary(controller, "linkedin")
        assert stopped["fleet_config"]["paused"] is True
        assert stopped["fleet_config"]["ats_pause_source"] == "incident_hold"
        assert stopped["fleet_config"]["linkedin_canary_worker_id"] is None
        assert stopped["fleet_config"]["linkedin_canary_version"] is None
        assert stopped["fleet_config"]["ats_canary_worker_id"] == "preserved-ats-worker"
        assert stopped["fleet_config"]["ats_canary_version"] == "preserved-ats-version"


def _install_brain_schema_through_v2(conn) -> None:
    with conn.transaction():
        with conn.cursor() as cur:
            migration_identity = schema._activate_migration_identity(cur)
            cur.execute(schema._schema_bytes().decode("utf-8"))
            schema._apply_acl_contract(cur, migration_identity)
            cur.execute(
                "INSERT INTO public.brain_schema_versions("
                "version,migration_name,migration_checksum,applied_by) VALUES(1,%s,%s,%s)",
                (schema._MIGRATION_NAME, schema._schema_checksum(), migration_identity),
            )
            cur.execute(schema._schema_v2_bytes().decode("utf-8"))
            schema._apply_acl_contract(cur, migration_identity)
            cur.execute(
                "INSERT INTO public.brain_schema_versions("
                "version,migration_name,migration_checksum,applied_by) VALUES(2,%s,%s,%s)",
                (schema._MIGRATION_V2_NAME, schema._schema_v2_checksum(), migration_identity),
            )


def test_schema_v2_bytes_remain_immutable():
    assert schema._schema_v2_checksum() == "0d4f29cf873e7e6c5051851be3a1ecf9701fdd8d0570af6f9503e83bb783577a"


def test_existing_v2_database_upgrades_atomically_to_v3(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_brain_schema_through_v2(conn)
        before_v2 = conn.execute(
            "SELECT migration_name,migration_checksum,applied_at,applied_by "
            "FROM public.brain_schema_versions WHERE version=2"
        ).fetchone()
        conn.rollback()

        schema.ensure_schema_v1(conn)

        versions = conn.execute(
            "SELECT version,migration_name,migration_checksum FROM public.brain_schema_versions ORDER BY version"
        ).fetchall()
        assert [row["version"] for row in versions] == [1, 2, 3]
        assert versions[2]["migration_name"] == schema._MIGRATION_V3_NAME
        assert versions[2]["migration_checksum"] == schema._schema_v3_checksum()
        assert (
            conn.execute(
                "SELECT migration_name,migration_checksum,applied_at,applied_by "
                "FROM public.brain_schema_versions WHERE version=2"
            ).fetchone()
            == before_v2
        )


def test_standalone_v4_install_is_atomic_from_v1_through_v4(brain_pg, monkeypatch):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        with monkeypatch.context() as patch:
            patch.setattr(schema, "_schema_v4_bytes", lambda: b"CREATE TABLE broken (")
            with pytest.raises(psycopg.errors.SyntaxError):
                schema.ensure_brain_schema_v4(conn)
        assert (
            conn.execute("SELECT to_regclass('public.brain_schema_versions') AS relation").fetchone()["relation"]
            is None
        )
        conn.rollback()

        schema.ensure_brain_schema_v4(conn)
        assert [
            row["version"]
            for row in conn.execute("SELECT version FROM public.brain_schema_versions ORDER BY version").fetchall()
        ] == [1, 2, 3, 4]


def test_v4_catalog_verification_is_stable_under_changed_search_path(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        schema.ensure_brain_schema_v4(conn)
        conn.execute("SET search_path=pg_catalog")
        conn.commit()

        schema.verify_brain_schema_v4(conn)

        assert conn.execute("SELECT current_setting('search_path') AS value").fetchone()["value"] == "pg_catalog"


def test_failed_v2_to_v3_upgrade_rolls_back_and_retries(brain_pg, monkeypatch):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_brain_schema_through_v2(conn)
        with monkeypatch.context() as patch:
            patch.setattr(schema, "_schema_v3_bytes", lambda: b"CREATE TABLE broken (")
            with pytest.raises(psycopg.errors.SyntaxError):
                schema.ensure_schema_v1(conn)
        assert [
            row["version"]
            for row in conn.execute("SELECT version FROM public.brain_schema_versions ORDER BY version").fetchall()
        ] == [1, 2]
        conn.rollback()

        schema.ensure_schema_v1(conn)
        assert [
            row["version"]
            for row in conn.execute("SELECT version FROM public.brain_schema_versions ORDER BY version").fetchall()
        ] == [1, 2, 3]


def test_existing_v1_database_upgrades_atomically_to_v3(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                migration_identity = schema._activate_migration_identity(cur)
                cur.execute(schema._schema_bytes().decode("utf-8"))
                schema._apply_acl_contract(cur, migration_identity)
                cur.execute(
                    "INSERT INTO public.brain_schema_versions("
                    "version,migration_name,migration_checksum,applied_by) VALUES(1,%s,%s,%s)",
                    (
                        schema._MIGRATION_NAME,
                        schema._schema_checksum(),
                        migration_identity,
                    ),
                )
        before = conn.execute("SELECT version FROM public.brain_schema_versions ORDER BY version").fetchall()
        conn.rollback()
        assert [row["version"] for row in before] == [1]

        schema.ensure_schema_v1(conn)
        first = conn.execute(
            "SELECT version,migration_checksum,applied_at FROM public.brain_schema_versions ORDER BY version"
        ).fetchall()
        conn.rollback()
        assert [row["version"] for row in first] == [1, 2, 3]

        schema.ensure_schema_v1(conn)
        second = conn.execute(
            "SELECT version,migration_checksum,applied_at FROM public.brain_schema_versions ORDER BY version"
        ).fetchall()
        conn.rollback()
        assert second == first


def test_failed_v1_to_v2_upgrade_rolls_back_and_retries(brain_pg, monkeypatch):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                migration_identity = schema._activate_migration_identity(cur)
                cur.execute(schema._schema_bytes().decode("utf-8"))
                schema._apply_acl_contract(cur, migration_identity)
                cur.execute(
                    "INSERT INTO public.brain_schema_versions("
                    "version,migration_name,migration_checksum,applied_by) VALUES(1,%s,%s,%s)",
                    (
                        schema._MIGRATION_NAME,
                        schema._schema_checksum(),
                        migration_identity,
                    ),
                )
        with monkeypatch.context() as patch:
            patch.setattr(schema, "_schema_v2_bytes", lambda: b"CREATE TABLE broken (")
            with pytest.raises(psycopg.errors.SyntaxError):
                schema.ensure_schema_v1(conn)
        versions = conn.execute("SELECT version FROM public.brain_schema_versions ORDER BY version").fetchall()
        assert [row["version"] for row in versions] == [1]
        assert (
            conn.execute("SELECT to_regclass('public.brain_canary_lifecycle_events') AS relation").fetchone()[
                "relation"
            ]
            is None
        )
        conn.rollback()

        schema.ensure_schema_v1(conn)
        versions = conn.execute("SELECT version FROM public.brain_schema_versions ORDER BY version").fetchall()
        assert [row["version"] for row in versions] == [1, 2, 3]


def _fleet_snapshot(conn):
    tables = [
        row["table_name"]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name NOT LIKE 'brain_%' ORDER BY table_name"
        ).fetchall()
    ]
    rows = {}
    for table in tables:
        rows[table] = conn.execute(
            pg_sql.SQL(
                "SELECT COALESCE(jsonb_agg(value ORDER BY value::text), '[]'::jsonb) AS rows "
                "FROM (SELECT to_jsonb(t) AS value FROM {}.{} t) snapshot_rows"
            ).format(pg_sql.Identifier("public"), pg_sql.Identifier(table))
        ).fetchone()["rows"]
    acls = conn.execute(
        "SELECT c.relname, c.relacl::text AS acl FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' "
        "AND left(c.relname, 6) <> 'brain_' ORDER BY c.relname",
    ).fetchall()
    return tables, rows, acls


def _grant_fleet_controller_contract(conn) -> None:
    for relation in ("fleet_config", "fleet_decision_policies", "apply_queue", "linkedin_queue"):
        if conn.execute("SELECT to_regclass(%s) AS relation", (f"public.{relation}",)).fetchone()["relation"]:
            conn.execute(
                pg_sql.SQL("GRANT SELECT, INSERT, UPDATE ON TABLE {}.{} TO brain_schema_migrator").format(
                    pg_sql.Identifier("public"), pg_sql.Identifier(relation)
                )
            )


def _install_fixture_data(conn) -> None:
    schema.ensure_schema_v1(conn)
    with conn.transaction():
        for artifact_hash in HASHES:
            conn.execute(
                "INSERT INTO brain_artifacts "
                "(request_id, artifact_hash, media_type, byte_length, schema_version, location) "
                "VALUES (%s, %s, 'application/json', 1, 1, %s)",
                (f"fixture-{artifact_hash}", artifact_hash, f"memory://{artifact_hash}"),
            )
        conn.execute(
            "INSERT INTO brain_jobs (job_id, source_namespace, source_job_id, title) VALUES "
            "('job-1', 'legacy', '1', 'Engineer'), ('job-2', 'legacy', '2', 'Analyst')"
        )


def _insert_policy(conn, version: str = "ats-v1", lane: str = "ats") -> None:
    with conn.transaction():
        conn.execute(
            "INSERT INTO brain_decision_policies (policy_version, lane, lifecycle) VALUES (%s, %s, 'draft')",
            (version, lane),
        )


def _insert_decision(
    conn,
    *,
    decision_id: str = "decision-1",
    source_id: str = "source-decision-1",
    policy: str = "ats-v1",
    lane: str = "ats",
    job_id: str = "job-1",
    input_hash: str = HASHES[1],
    verdict: str = "qualified",
    action: str = "apply",
) -> None:
    with conn.transaction():
        conn.execute(
            "INSERT INTO brain_job_decisions ("
            "decision_id, source_namespace, source_decision_id, job_id, policy_version, lane, "
            "qualification_score, qualification_floor, preference_score, outcome_score, final_score, "
            "qualification_verdict, action, confidence, input_hash, uncertainty, blockers, "
            "requirements, evidence_nodes, title_signals, explanation, expires_at) "
            "VALUES (%s, 'test', %s, %s, %s, %s, 0.8, 0.6, 0.7, 0.9, 0.8, %s, %s, 0.8, %s, "
            "'[]', '[]', '[]', '[]', '[]', 'because', "
            "CASE WHEN %s = 'apply' THEN now() + interval '1 day' ELSE NULL END)",
            (decision_id, source_id, job_id, policy, lane, verdict, action, input_hash, action),
        )


def test_fresh_install_true_noop_and_fleet_unchanged(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        fleet_schema.ensure_schema_v3(conn)
        _grant_fleet_controller_contract(conn)
        conn.commit()
        fleet_before = _fleet_snapshot(conn)
        conn.rollback()

        schema.ensure_schema_v1(conn)
        conn.execute(
            "INSERT INTO brain_jobs (job_id, source_namespace, source_job_id) VALUES ('preserved', 'test', 'preserved')"
        )
        conn.commit()
        fleet_after_lifecycle_acl = _fleet_snapshot(conn)
        conn.rollback()
        before = conn.execute(
            "SELECT applied_at, migration_checksum FROM brain_schema_versions WHERE version = 1"
        ).fetchone()
        acl_before = conn.execute("SELECT relacl FROM pg_class WHERE oid = 'public.brain_jobs'::regclass").fetchone()[
            "relacl"
        ]
        conn.rollback()

        schema.ensure_schema_v1(conn)

        after = conn.execute(
            "SELECT applied_at, migration_checksum FROM brain_schema_versions WHERE version = 1"
        ).fetchone()
        acl_after = conn.execute("SELECT relacl FROM pg_class WHERE oid = 'public.brain_jobs'::regclass").fetchone()[
            "relacl"
        ]
        assert len(after["migration_checksum"]) == 64
        assert before == after
        assert acl_before == acl_after
        assert conn.execute("SELECT count(*) AS n FROM brain_jobs").fetchone()["n"] == 1
        fleet_after_noop = _fleet_snapshot(conn)
        assert fleet_after_lifecycle_acl == fleet_after_noop
        assert fleet_after_lifecycle_acl[:2] == fleet_before[:2]


def test_failed_install_rolls_back_completely_and_retries(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        conn.execute("CREATE TABLE brain_artifacts (wrong_column integer)")
        conn.commit()
        with pytest.raises(RuntimeError, match="existing brain objects"):
            schema.ensure_schema_v1(conn)
        assert conn.info.transaction_status == TransactionStatus.IDLE
        assert conn.execute("SELECT to_regclass('public.brain_schema_versions') AS name").fetchone()["name"] is None
        conn.rollback()
        conn.execute("DROP TABLE brain_artifacts")
        conn.commit()
        schema.ensure_schema_v1(conn)
        assert conn.info.transaction_status == TransactionStatus.IDLE


def test_helpers_require_idle_and_timeout_leaves_connection_reusable(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        conn.execute("CREATE TEMP TABLE caller_work (value integer)")
        conn.execute("INSERT INTO caller_work VALUES (1)")
        with pytest.raises(RuntimeError, match="idle connection"):
            schema.ensure_schema_v1(conn)
        assert conn.execute("SELECT value FROM caller_work").fetchone()["value"] == 1
        conn.rollback()

    with psycopg.connect(brain_pg, row_factory=dict_row) as holder:
        holder.execute("SELECT pg_advisory_lock(hashtext('applypilot:brain:schema:v1'))")
        with psycopg.connect(brain_pg, row_factory=dict_row) as waiter:
            with pytest.raises(TimeoutError, match="migration lock"):
                schema.ensure_schema_v1(waiter, lock_timeout_seconds=0.05)
            assert waiter.info.transaction_status == TransactionStatus.IDLE
            assert waiter.execute("SELECT 1 AS value").fetchone()["value"] == 1
            waiter.rollback()
        holder.execute("SELECT pg_advisory_unlock(hashtext('applypilot:brain:schema:v1'))")


def test_generic_create_role_cannot_migrate(brain_pg, monkeypatch):
    monkeypatch.delenv("APPLYPILOT_BRAIN_MIGRATION_ROLE", raising=False)
    with psycopg.connect(brain_pg, row_factory=dict_row) as owner:
        owner.execute("CREATE ROLE brain_create_role LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE")
        _set_role_password(owner, brain_pg, "brain_create_role")
        owner.execute("GRANT USAGE, CREATE ON SCHEMA public TO brain_create_role")
        owner.execute("GRANT CREATE ON DATABASE postgres TO brain_create_role")
        owner.commit()
    with psycopg.connect(_dsn_for(brain_pg, "brain_create_role"), row_factory=dict_row) as conn:
        with pytest.raises(RuntimeError, match="migration requires"):
            schema.ensure_schema_v1(conn)
        assert conn.info.transaction_status == TransactionStatus.IDLE
    with psycopg.connect(brain_pg, row_factory=dict_row) as owner:
        assert owner.execute("SELECT to_regclass('public.brain_schema_versions') AS name").fetchone()["name"] is None


def test_exact_configured_migration_role_is_allowed(brain_pg, monkeypatch):
    with psycopg.connect(brain_pg, row_factory=dict_row) as owner:
        owner.execute("CREATE ROLE brain_schema_reader LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE")
        _set_role_password(owner, brain_pg, "brain_schema_reader")
        owner.execute("GRANT brain_schema_migrator TO brain_schema_reader")
        owner.commit()
    monkeypatch.setenv("APPLYPILOT_BRAIN_MIGRATION_ROLE", "brain_schema_reader")
    with psycopg.connect(_dsn_for(brain_pg, "brain_schema_reader"), row_factory=dict_row) as conn:
        schema.ensure_schema_v1(conn)
        assert conn.info.transaction_status == TransactionStatus.IDLE


def test_importer_activates_controller_and_serializes_source_ownership(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as owner:
        schema.ensure_schema_v1(owner)
    source_hash = "1" * 64
    with (
        psycopg.connect(brain_pg, row_factory=dict_row) as holder,
        psycopg.connect(brain_pg, row_factory=dict_row) as contender,
    ):
        _activate_controller(holder)
        _acquire_import_lock(holder, source_hash)
        assert holder.execute("SELECT current_user AS role").fetchone()["role"] == "brain_schema_migrator"

        _activate_controller(contender)
        with pytest.raises(BrainImportError, match="already owns source"):
            _acquire_import_lock(contender, source_hash)

        _release_import_context(contender, source_hash, lock_acquired=False)
        _release_import_context(holder, source_hash, lock_acquired=True)
        assert holder.execute("SELECT current_user AS role").fetchone()["role"] == "postgres"


def test_importer_refuses_to_commit_caller_transaction(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        schema.ensure_schema_v1(conn)
        conn.execute("CREATE TEMP TABLE importer_caller_work (value integer)")
        conn.execute("INSERT INTO importer_caller_work VALUES (1)")

        with pytest.raises(BrainImportError, match="idle dedicated"):
            _activate_controller(conn)

        assert conn.info.transaction_status == TransactionStatus.INTRANS
        assert conn.execute("SELECT value FROM importer_caller_work").fetchone()["value"] == 1
        conn.rollback()


def test_policy_import_uses_compiled_artifacts_and_compact_graph_binding(brain_pg):
    source = sqlite3.connect(":memory:")
    source.row_factory = sqlite3.Row
    source.execute(
        """CREATE TABLE decision_policy_versions (
            policy_version TEXT PRIMARY KEY, lane TEXT, status TEXT,
            qualification_model TEXT, preference_model TEXT, outcome_model TEXT,
            kg_version TEXT, label_snapshot TEXT, pairwise_snapshot TEXT, outcome_snapshot TEXT,
            config_json TEXT, metrics_json TEXT, created_at TEXT, validated_at TEXT,
            activated_at TEXT, retired_at TEXT)"""
    )
    source.execute(
        "INSERT INTO decision_policy_versions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "policy-import-test",
            "ats",
            "draft",
            "qualification-v1",
            None,
            None,
            HASHES[0],
            None,
            None,
            None,
            '{"models":{"qualificationEvidence":{"weights":[1]}}}',
            None,
            "2026-07-16T00:00:00Z",
            None,
            None,
            None,
        ),
    )
    compiled = compile_policy_artifacts(dict(source.execute("SELECT * FROM decision_policy_versions").fetchone()))

    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        schema.ensure_schema_v1(conn)
        with conn.transaction():
            conn.execute(
                "INSERT INTO brain_artifacts "
                "(request_id,artifact_hash,media_type,byte_length,schema_version,location) "
                "VALUES ('graph-import-test',%s,'application/json',1,1,'memory://graph')",
                (HASHES[0],),
            )
            for artifact in compiled.artifacts:
                conn.execute(
                    "INSERT INTO brain_artifacts "
                    "(request_id,artifact_hash,media_type,byte_length,schema_version,provenance,location) "
                    "VALUES (%s,%s,%s,%s,1,%s,%s)",
                    (
                        f"policy-import-test:{artifact.sha256}",
                        artifact.sha256,
                        artifact.media_type,
                        artifact.byte_length,
                        json.dumps({"policy_source_id": artifact.policy_source_id}),
                        f"memory://{artifact.sha256}",
                    ),
                )
        _activate_controller(conn)
        try:
            assert _insert_policies(conn, source) == 1
        finally:
            _release_import_context(conn, "0" * 64, lock_acquired=False)

        policy = conn.execute(
            "SELECT lane,lifecycle,policy_metadata FROM brain_decision_policies "
            "WHERE policy_version='policy-import-test'"
        ).fetchone()
        assert policy["lane"] == "ats"
        assert policy["lifecycle"] == "draft"
        assert policy["policy_metadata"] == compiled.metadata_object()
        bindings = conn.execute(
            "SELECT artifact_role,artifact_hash FROM brain_policy_artifacts "
            "WHERE policy_version='policy-import-test' ORDER BY artifact_role"
        ).fetchall()
        assert bindings == [
            {"artifact_role": "config", "artifact_hash": compiled.artifact("config").sha256},
            {"artifact_role": "knowledge_graph", "artifact_hash": HASHES[0]},
            {
                "artifact_role": "qualification_model",
                "artifact_hash": compiled.artifact("qualification_model").sha256,
            },
        ]


def test_decision_delta_accepts_new_ids_that_sort_before_existing_ids(brain_pg):
    source = sqlite3.connect(":memory:")
    source.row_factory = sqlite3.Row
    source.execute(
        """CREATE TABLE job_decisions (
            decision_id TEXT PRIMARY KEY, job_url TEXT, policy_version TEXT, lane TEXT,
            qualification_score REAL, preference_score REAL, outcome_score REAL, final_score REAL,
            qualification_verdict TEXT, action TEXT, confidence REAL, uncertainty_json TEXT,
            blockers_json TEXT, requirements_json TEXT, evidence_node_ids_json TEXT,
            title_signals_json TEXT, explanation TEXT, input_hash TEXT, created_at TEXT,
            expires_at TEXT)"""
    )
    source.executemany(
        "INSERT INTO job_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                decision_id,
                job_url,
                "policy-delta-test",
                "ats",
                0.8,
                0.7,
                0.6,
                0.75,
                "qualified",
                "review",
                0.9,
                "[]",
                "[]",
                "[]",
                "[]",
                "[]",
                "test",
                input_hash,
                "2026-07-16T00:00:00Z",
                None,
            )
            for decision_id, job_url, input_hash in (
                ("a-new", "https://jobs/new", "a" * 64),
                ("m-existing", "https://jobs/existing", "b" * 64),
            )
        ],
    )

    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        schema.ensure_schema_v1(conn)
        _activate_controller(conn)
        conn.execute(
            "INSERT INTO brain_decision_policies (policy_version,lane,lifecycle) "
            "VALUES ('policy-delta-test','ats','draft')"
        )
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO brain_jobs (job_id,source_namespace,source_job_id,canonical_url) "
                "VALUES (%s,'applypilot-sqlite',%s,%s)",
                [
                    ("job-new", "https://jobs/new", "https://jobs/new"),
                    ("job-existing", "https://jobs/existing", "https://jobs/existing"),
                ],
            )
        conn.commit()
        schema.ensure_policy_partition(conn, "policy-delta-test")
        _activate_controller(conn)
        conn.execute(
            """INSERT INTO brain_job_decisions
               (decision_id,source_namespace,source_decision_id,job_id,policy_version,lane,
                qualification_score,preference_score,outcome_score,final_score,
                qualification_verdict,action,confidence,uncertainty,blockers,requirements,
                evidence_nodes,title_signals,explanation,input_hash,created_at)
               VALUES ('m-existing','applypilot-sqlite','m-existing','job-existing',
                       'policy-delta-test','ats',0.8,0.7,0.6,0.75,'qualified','review',0.9,
                       '[]','[]','[]','[]','[]','test',%s,'2026-07-16T00:00:00Z')""",
            ("b" * 64,),
        )
        conn.commit()

        assert (
            _insert_decisions(
                conn,
                source,
                {"https://jobs/new": "job-new", "https://jobs/existing": "job-existing"},
                1,
            )
            == 2
        )
        assert conn.execute(
            "SELECT source_decision_id FROM brain_job_decisions "
            "WHERE source_namespace='applypilot-sqlite' ORDER BY source_decision_id"
        ).fetchall() == [{"source_decision_id": "a-new"}, {"source_decision_id": "m-existing"}]


def test_deep_verification_rejects_malformed_same_column_schema(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        schema.ensure_schema_v1(conn)
        conn.execute("ALTER TABLE brain_job_decisions DROP CONSTRAINT brain_job_decisions_apply_expiry")
        conn.commit()
        with pytest.raises(RuntimeError, match="missing constraint.*apply_expiry"):
            schema.verify_schema_v1(conn)
        assert conn.info.transaction_status == TransactionStatus.IDLE


def test_deep_verification_rejects_same_name_constraint_and_index_tampering(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        schema.ensure_schema_v1(conn)
        conn.execute("ALTER TABLE brain_job_decisions DROP CONSTRAINT brain_job_decisions_apply_expiry")
        conn.execute("ALTER TABLE brain_job_decisions ADD CONSTRAINT brain_job_decisions_apply_expiry CHECK (true)")
        conn.commit()
        with pytest.raises(RuntimeError, match="apply_expiry definition mismatch"):
            schema.verify_schema_v1(conn)

    _cleanup(brain_pg)
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        schema.ensure_schema_v1(conn)
        conn.execute("DROP INDEX brain_decision_policies_one_active_per_lane")
        conn.execute(
            "CREATE UNIQUE INDEX brain_decision_policies_one_active_per_lane "
            "ON brain_decision_policies(lane, lifecycle)"
        )
        conn.commit()
        with pytest.raises(RuntimeError, match="index.*signature mismatch"):
            schema.verify_schema_v1(conn)


def test_deep_verification_rejects_not_valid_constraint_and_wrong_trigger_binding(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        schema.ensure_schema_v1(conn)
        conn.execute("ALTER TABLE brain_job_decisions DROP CONSTRAINT brain_job_decisions_apply_expiry")
        conn.execute(
            "ALTER TABLE brain_job_decisions ADD CONSTRAINT brain_job_decisions_apply_expiry "
            "CHECK ((action <> 'apply') OR expires_at > created_at) NOT VALID"
        )
        conn.commit()
        with pytest.raises(RuntimeError, match="apply_expiry is not validated"):
            schema.verify_schema_v1(conn)

    _cleanup(brain_pg)
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        schema.ensure_schema_v1(conn)
        conn.execute("DROP TRIGGER brain_artifacts_append_only ON brain_artifacts")
        conn.execute(
            "CREATE TRIGGER brain_artifacts_append_only BEFORE UPDATE OR DELETE ON brain_artifacts "
            "FOR EACH ROW EXECUTE FUNCTION brain_check_archive_manifest()"
        )
        conn.commit()
        with pytest.raises(RuntimeError, match="binding or enabled state mismatch"):
            schema.verify_schema_v1(conn)


def test_deep_verification_rejects_non_migration_object_owner(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as owner:
        schema.ensure_schema_v1(owner)
        owner.execute("CREATE ROLE brain_schema_reader LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE")
        owner.commit()
    with psycopg.connect(brain_pg, row_factory=dict_row) as owner:
        owner.execute("ALTER TABLE brain_jobs OWNER TO postgres")
        owner.commit()
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        with pytest.raises(RuntimeError, match="ownership mismatch.*brain_jobs"):
            schema.verify_schema_v1(conn)
        assert conn.info.transaction_status == TransactionStatus.IDLE


def test_exact_enums_nullable_expiry_aliases_and_lossless_decision_fields(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        _insert_policy(conn)
        schema.ensure_policy_partition(conn, "ats-v1")
        conn.execute(
            "INSERT INTO brain_job_aliases "
            "(job_id, source_namespace, source_database_fingerprint, source_item_id, source_url, alias_type) "
            "VALUES ('job-1', 'review-db', %s, '42', 'https://example.test/42', 'source_id')",
            (HASHES[0],),
        )
        conn.execute(
            "INSERT INTO brain_jobs (job_id,source_namespace,source_job_id,title) VALUES "
            "('job-3','legacy','3','Third'),('job-4','legacy','4','Fourth')"
        )
        conn.commit()
        for decision_id, job_id, verdict, action, input_hash in (
            ("qualified", "job-1", "qualified", "apply", HASHES[2]),
            ("unqualified", "job-2", "unqualified", "reject", HASHES[3]),
            ("uncertain", "job-3", "uncertain", "review", HASHES[4]),
        ):
            _insert_decision(
                conn,
                decision_id=decision_id,
                source_id=decision_id,
                job_id=job_id,
                input_hash=input_hash,
                verdict=verdict,
                action=action,
            )
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.transaction():
                conn.execute(
                    "INSERT INTO brain_job_decisions (decision_id, source_namespace, source_decision_id, "
                    "job_id, policy_version, lane, qualification_verdict, action, input_hash, expires_at) "
                    "VALUES ('bad-expiry', 'test', 'bad-expiry', 'job-4', 'ats-v1', 'ats', "
                    "'qualified', 'apply', %s, now() - interval '1 second')",
                    (HASHES[5],),
                )
        assert (
            conn.execute("SELECT count(*) AS n FROM brain_job_decisions WHERE expires_at IS NULL").fetchone()["n"] == 2
        )


def test_label_and_pairwise_events_preserve_unresolved_source_endpoints(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        conn.execute(
            "INSERT INTO brain_label_events "
            "(source_namespace, source_event_id, source_item_id, source_item_url, project, method, "
            "confidence, weight, label_name, label_value, occurred_at, raw_artifact_hash, event_metadata) "
            "VALUES ('review', 'label-1', 'missing-1', 'https://missing/1', 'applypilot', 'manual', "
            "0.9, 2, 'fit', 'true', now(), %s, '{\"raw\":true}')",
            (HASHES[0],),
        )
        conn.execute(
            "INSERT INTO brain_pairwise_events "
            "(source_namespace, source_event_id, left_source_item_id, left_source_url, "
            "right_source_item_id, right_source_url, project, method, confidence, weight, "
            "preference, occurred_at, raw_artifact_hash, event_metadata) "
            "VALUES ('review', 'pair-1', 'missing-left', 'https://missing/left', "
            "'missing-right', 'https://missing/right', 'applypilot', 'manual', 0.8, 3, "
            "'left', now(), %s, '{\"raw\":true}')",
            (HASHES[1],),
        )
        conn.commit()
        assert conn.execute("SELECT job_id, source_item_id FROM brain_label_events").fetchone() == {
            "job_id": None,
            "source_item_id": "missing-1",
        }


def test_outcome_email_fk_and_stable_applications(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        email_id = conn.execute(
            "INSERT INTO brain_email_events "
            "(source_namespace, source_event_id, job_id, event_type, occurred_at) "
            "VALUES ('gmail', 'mail-1', 'job-1', 'reply', now()) RETURNING email_event_id"
        ).fetchone()["email_event_id"]
        conn.execute(
            "INSERT INTO brain_reviewed_outcomes "
            "(source_namespace, source_event_id, job_id, email_event_id, review_status, normalized_stage, "
            "weight, reviewer, reason, created_at, reviewed_at, updated_at) "
            "VALUES ('review', 'outcome-1', 'job-1', %s, 'confirmed', 'interview', 1.5, "
            "'owner', 'manual', now(), now(), now())",
            (email_id,),
        )
        conn.execute(
            "INSERT INTO brain_applications "
            "(application_id, job_id, source_namespace, source_application_id, source_channel, lane) "
            "VALUES ('app-1', 'job-1', 'legacy', '1', 'email', NULL)"
        )
        conn.execute(
            "INSERT INTO brain_application_events "
            "(application_id, source_namespace, source_event_id, source_channel, event_type, occurred_at) "
            "VALUES ('app-1', 'legacy', 'submitted-1', 'email', 'submitted', now())"
        )
        conn.commit()
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            with conn.transaction():
                conn.execute(
                    "INSERT INTO brain_reviewed_outcomes "
                    "(source_namespace, source_event_id, job_id, email_event_id, review_status, "
                    "normalized_stage, weight, reviewer, created_at, reviewed_at, updated_at) "
                    "VALUES ('review', 'bad', 'job-1', -1, 'confirmed', 'offer', 1, 'owner', now(), now(), now())"
                )


def _attach_policy_contract(
    conn,
    policy: str,
    *,
    artifact_hashes: dict[str, str] | None = None,
) -> dict[str, int]:
    roles = (
        "qualification_model",
        "preference_model",
        "outcome_model",
        "knowledge_graph",
        "label_snapshot",
        "pairwise_snapshot",
        "outcome_snapshot",
        "config",
        "metrics",
        "replay",
    )
    with conn.transaction():
        conn.execute(
            "INSERT INTO brain_artifact_locations "
            "(artifact_hash, backend, bucket_or_container, object_key, provider_version_id, provider_checksum, "
            "storage_immutable, encryption_mode, encryption_key_id, durability, verified_at) "
            "VALUES (%s, 's3', 'protected', %s, %s, %s, TRUE, 'customer_managed', 'key-v1', 'verified', now()) "
            "ON CONFLICT DO NOTHING",
            (HASHES[0], f"reports/{policy}", f"version-{policy}", HASHES[0]),
        )
        for index, role in enumerate(roles):
            conn.execute(
                "INSERT INTO brain_policy_artifacts (policy_version, artifact_role, artifact_hash) VALUES (%s, %s, %s)",
                (policy, role, (artifact_hashes or {}).get(role, HASHES[index])),
            )
        for approval in ("validated", "canary", "active", "retired"):
            conn.execute(
                "INSERT INTO brain_policy_approvals (policy_version, approval_type, approved_by) "
                "VALUES (%s, %s, 'owner')",
                (policy, approval),
            )
        receipts = {}
        definitions = conn.execute(
            "SELECT lifecycle, gate_name FROM brain_policy_gate_definitions "
            "WHERE definition_version=1 AND lane=(SELECT lane FROM brain_decision_policies WHERE policy_version=%s) "
            "ORDER BY lifecycle, gate_name",
            (policy,),
        ).fetchall()
        lane = conn.execute("SELECT lane FROM brain_decision_policies WHERE policy_version=%s", (policy,)).fetchone()[
            "lane"
        ]
        for definition in definitions:
            receipts[definition["lifecycle"]] = conn.execute(
                "INSERT INTO brain_policy_release_gate_events "
                "(policy_version, lane, lifecycle, gate_name, gate_state, checked_by, report_artifact_hash) "
                "VALUES (%s, %s, %s, %s, 'passed', 'owner', %s) RETURNING gate_event_id",
                (policy, lane, definition["lifecycle"], definition["gate_name"], HASHES[0]),
            ).fetchone()["gate_event_id"]
    return receipts


def test_policy_artifact_roles_lifecycle_gates_and_one_active_per_lane(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        fleet_schema.ensure_schema_v3(conn)
        _grant_fleet_controller_contract(conn)
        conn.commit()
        _install_fixture_data(conn)
        _insert_policy(conn)
        conn.execute("SET LOCAL ROLE brain_schema_migrator")
        with pytest.raises(psycopg.errors.CheckViolation, match="latest mandatory gate"):
            with conn.transaction():
                conn.execute("SELECT brain_transition_policy('ats-v1', 'validated')")
        _attach_policy_contract(conn, "ats-v1")
        conn.execute("RESET ROLE")
        conn.execute("UPDATE fleet_config SET paused=TRUE WHERE id=1")
        conn.commit()
        with conn.transaction():
            conn.execute("SET LOCAL ROLE brain_schema_migrator")
            for lifecycle in ("validated", "canary", "active"):
                conn.execute("SELECT brain_transition_policy(%s, %s)", ("ats-v1", lifecycle))
        _insert_policy(conn, "ats-v2", "ats")
        _attach_policy_contract(conn, "ats-v2")
        conn.commit()
        with conn.transaction():
            conn.execute("SET LOCAL ROLE brain_schema_migrator")
            for lifecycle in ("validated", "canary", "active"):
                conn.execute("SELECT brain_transition_policy(%s, %s)", ("ats-v2", lifecycle))
        assert conn.execute(
            "SELECT policy_version,lifecycle FROM brain_decision_policies ORDER BY policy_version"
        ).fetchall() == [
            {"policy_version": "ats-v1", "lifecycle": "retired"},
            {"policy_version": "ats-v2", "lifecycle": "active"},
        ]


def test_artifacts_evidence_migration_parity_and_archive_are_immutable(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        run_id, batch_ids = _insert_migration_run_and_batches(conn, 1)
        batch_id = batch_ids[0]
        _claim_batch(conn, batch_id)
        completed_event_id = _complete_batch(conn, batch_id)
        _checkpoint_batch(conn, batch_id, completed_event_id)
        conn.execute(
            "INSERT INTO brain_migration_quarantine "
            "(migration_run_id,source_namespace,migration_batch_id,batch_ordinal,source_table,source_key,"
            "reason_code,unresolved_evidence) "
            "VALUES (%s,'sqlite',%s,1,'jobs','42','missing_alias','{\"source_id\":42}')",
            (run_id, batch_id),
        )
        conn.execute(
            "INSERT INTO brain_artifact_locations "
            "(artifact_hash,backend,bucket_or_container,object_key,provider_version_id,provider_checksum,"
            "storage_immutable,encryption_mode,encryption_key_id,durability,verified_at) "
            "VALUES (%s,'s3','archive','objects/a','v1',%s,TRUE,'provider_managed','aws/s3','verified',now())",
            (HASHES[0], HASHES[0]),
        )
        conn.execute(
            "INSERT INTO brain_archive.brain_archive_manifests "
            "(retry_identity, source_relation, artifact_hash, row_count) "
            "VALUES ('immutable-archive', 'brain_job_decisions', %s, 0)",
            (HASHES[0],),
        )
        conn.commit()
        for relation in (
            "brain_artifacts",
            "brain_migration_sources",
            "brain_migration_runs",
            "brain_migration_run_events",
            "brain_migration_batches",
            "brain_migration_batch_events",
            "brain_migration_checkpoints",
            "brain_migration_quarantine",
            "brain_archive.brain_archive_manifests",
        ):
            with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState, match="append-only"):
                with conn.transaction():
                    conn.execute(f"DELETE FROM {relation}")


def test_ledgers_decisions_identities_and_policy_receipts_are_immutable(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        _insert_policy(conn)
        _attach_policy_contract(conn, "ats-v1")
        schema.ensure_policy_partition(conn, "ats-v1")
        _insert_decision(conn)
        conn.execute(
            "INSERT INTO brain_label_events "
            "(source_namespace, source_event_id, job_id, project, method, label_name, label_value, occurred_at) "
            "VALUES ('review', 'immutable-label', 'job-1', 'applypilot', 'manual', 'fit', 'true', now())"
        )
        conn.commit()
        for relation in (
            "brain_label_events",
            "brain_job_decisions",
            "brain_decision_identities",
            "brain_policy_artifacts",
            "brain_policy_approvals",
            "brain_policy_release_gate_events",
        ):
            with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState, match="append-only"):
                with conn.transaction():
                    conn.execute(f"DELETE FROM {relation}")


def test_supersession_requires_same_subject_and_prevents_forks(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        first = conn.execute(
            "INSERT INTO brain_label_events "
            "(source_namespace, source_event_id, job_id, project, method, label_name, label_value, occurred_at) "
            "VALUES ('review', 'first', 'job-1', 'applypilot', 'manual', 'fit', 'true', now()) "
            "RETURNING label_event_id"
        ).fetchone()["label_event_id"]
        conn.commit()
        with pytest.raises(psycopg.errors.CheckViolation, match="subject/source mismatch"):
            with conn.transaction():
                conn.execute(
                    "INSERT INTO brain_label_events "
                    "(source_namespace, source_event_id, job_id, project, method, label_name, label_value, "
                    "occurred_at, supersedes_label_event_id) "
                    "VALUES ('review', 'wrong', 'job-2', 'applypilot', 'manual', 'fit', 'false', now(), %s)",
                    (first,),
                )
        with conn.transaction():
            conn.execute(
                "INSERT INTO brain_label_events "
                "(source_namespace, source_event_id, job_id, project, method, label_name, label_value, "
                "occurred_at, supersedes_label_event_id) "
                "VALUES ('review', 'second', 'job-1', 'applypilot', 'manual', 'fit', 'false', now(), %s)",
                (first,),
            )
        with pytest.raises(psycopg.errors.UniqueViolation):
            with conn.transaction():
                conn.execute(
                    "INSERT INTO brain_label_events "
                    "(source_namespace, source_event_id, job_id, project, method, label_name, label_value, "
                    "occurred_at, supersedes_label_event_id) "
                    "VALUES ('review', 'fork', 'job-1', 'applypilot', 'manual', 'fit', 'null', now(), %s)",
                    (first,),
                )


def test_policy_partition_routing_detach_and_global_decision_identity(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        _insert_policy(conn, "ats-v1", "ats")
        _insert_policy(conn, "ats-v2", "ats")
        first = schema.ensure_policy_partition(conn, "ats-v1")
        second = schema.ensure_policy_partition(conn, "ats-v2")
        assert first != second
        assert first == schema.ensure_policy_partition(conn, "ats-v1")
        schema.verify_schema_v1(conn)
        with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState, match="append-only"):
            with conn.transaction():
                conn.execute(
                    pg_sql.SQL("TRUNCATE TABLE {}.{}").format(pg_sql.Identifier("public"), pg_sql.Identifier(first))
                )
        _insert_decision(conn)
        assert (
            conn.execute("SELECT tableoid::regclass::text AS partition FROM brain_job_decisions").fetchone()[
                "partition"
            ]
            == first
        )
        conn.rollback()
        with pytest.raises(psycopg.errors.UniqueViolation):
            _insert_decision(
                conn,
                decision_id="decision-1",
                source_id="other-source",
                policy="ats-v2",
                input_hash=HASHES[2],
            )
        with conn.transaction():
            conn.execute(f"ALTER TABLE brain_job_decisions DETACH PARTITION {second}")
        assert (
            conn.execute("SELECT relispartition FROM pg_class WHERE oid = %s::regclass", (second,)).fetchone()[
                "relispartition"
            ]
            is False
        )


def test_migration_batches_support_skip_locked_disjoint_claims(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        run_id, _ = _insert_migration_run_and_batches(conn)
        conn.commit()
        with conn.transaction():
            first = conn.execute(
                "SELECT migration_batch_id FROM brain_migration_batches WHERE migration_run_id=%s "
                "ORDER BY migration_batch_id FOR UPDATE SKIP LOCKED LIMIT 1",
                (run_id,),
            ).fetchone()["migration_batch_id"]
            assert first is not None


def test_public_fleet_worker_and_broad_default_grants_are_removed(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        conn.execute("CREATE ROLE fleet_worker LOGIN")
        conn.execute("CREATE ROLE brain_broad_role LOGIN")
        conn.execute("GRANT ALL ON SCHEMA public TO fleet_worker, brain_broad_role")
        conn.execute(
            "ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public "
            "GRANT ALL ON TABLES TO fleet_worker, brain_broad_role"
        )
        conn.execute("ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT SELECT ON TABLES TO PUBLIC")
        conn.execute(
            "ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public "
            "GRANT EXECUTE ON FUNCTIONS TO fleet_worker, brain_broad_role"
        )
        conn.commit()
        schema.ensure_schema_v1(conn)
        rows = conn.execute(
            "SELECT c.relname, "
            "has_table_privilege('fleet_worker', c.oid, 'SELECT,INSERT,UPDATE,DELETE') AS fleet, "
            "has_table_privilege('brain_broad_role', c.oid, 'SELECT,INSERT,UPDATE,DELETE') AS broad "
            "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname='public' AND left(c.relname, 6)='brain_' AND c.relkind IN ('r','p')"
        ).fetchall()
        assert rows
        assert all(not row["fleet"] and not row["broad"] for row in rows)
        assert conn.execute(
            "SELECT bool_and(acl.privilege_type='SELECT' AND ("
            "COALESCE(r.rolname,'PUBLIC')='brain_schema_verifier' OR ("
            "COALESCE(r.rolname,'PUBLIC') IN ('brain_status_reader','brain_policy_controller') "
            "AND c.relname IN ('brain_decision_policies','brain_policy_artifacts',"
            "'brain_policy_approvals','brain_parity_runs','brain_parity_run_events')))) AS valid "
            "FROM pg_class c "
            "JOIN pg_namespace n ON n.oid=c.relnamespace CROSS JOIN LATERAL aclexplode(c.relacl) acl "
            "LEFT JOIN pg_roles r ON r.oid=acl.grantee WHERE n.nspname='public' "
            "AND left(c.relname,6)='brain_' AND acl.grantee<>c.relowner"
        ).fetchone()["valid"]
        assert (
            conn.execute(
                "SELECT count(*) AS n FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
                "WHERE n.nspname='public' AND left(p.proname, 6)='brain_' AND "
                "(has_function_privilege('fleet_worker', p.oid, 'EXECUTE') OR "
                "has_function_privilege('brain_broad_role', p.oid, 'EXECUTE'))"
            ).fetchone()["n"]
            == 0
        )
        assert (
            conn.execute(
                "SELECT has_schema_privilege('fleet_worker', 'brain_archive', 'USAGE,CREATE') AS exposed"
            ).fetchone()["exposed"]
            is False
        )
        conn.rollback()
        schema.verify_schema_v1(conn)


def _insert_migration_run_and_batches(conn, batch_count: int = 2) -> tuple[int, list[int]]:
    conn.execute("SET LOCAL ROLE brain_schema_migrator")
    source_id = conn.execute(
        "INSERT INTO brain_migration_sources "
        "(source_namespace, source_fingerprint, byte_length, schema_metadata) "
        "VALUES ('sqlite', %s, 1, '{}') RETURNING migration_source_id",
        (HASHES[0],),
    ).fetchone()["migration_source_id"]
    run_id = conn.execute(
        "INSERT INTO brain_migration_runs (migration_source_id, source_namespace, run_key, metadata) "
        "VALUES (%s, 'sqlite', 'phase1-import', '{}') RETURNING migration_run_id",
        (source_id,),
    ).fetchone()["migration_run_id"]
    conn.execute(
        "INSERT INTO brain_migration_run_events "
        "(migration_run_id, source_namespace, event_type, actor_id) "
        "VALUES (%s, 'sqlite', 'started', 'controller')",
        (run_id,),
    )
    batch_ids = []
    for index in range(batch_count):
        batch_ids.append(
            conn.execute(
                "INSERT INTO brain_migration_batches "
                "(migration_run_id, source_namespace, source_table, batch_ordinal, key_start, key_end) "
                "VALUES (%s, 'sqlite', 'jobs', %s, %s, %s) "
                "RETURNING migration_batch_id",
                (run_id, index + 1, f"{index * 100 + 1:03}", f"{(index + 1) * 100:03}"),
            ).fetchone()["migration_batch_id"]
        )
        conn.execute(
            "INSERT INTO brain_migration_batch_events "
            "(migration_run_id, source_namespace, source_table, migration_batch_id, batch_ordinal, "
            "event_type, attempt) VALUES (%s, 'sqlite', 'jobs', %s, %s, 'pending', 0)",
            (run_id, batch_ids[-1], index + 1),
        )
    return run_id, batch_ids


def _batch_row(conn, batch_id: int):
    return conn.execute(
        "SELECT b.*, e.migration_batch_event_id AS head_event_id, e.attempt AS head_attempt "
        "FROM brain_migration_batches b JOIN LATERAL ("
        "SELECT migration_batch_event_id, attempt FROM brain_migration_batch_events "
        "WHERE migration_batch_id=b.migration_batch_id ORDER BY migration_batch_event_id DESC LIMIT 1"
        ") e ON TRUE WHERE b.migration_batch_id=%s",
        (batch_id,),
    ).fetchone()


def _claim_batch(conn, batch_id: int, worker: str = "worker-1") -> int:
    batch = _batch_row(conn, batch_id)
    return conn.execute(
        "INSERT INTO brain_migration_batch_events "
        "(migration_run_id, source_namespace, source_table, migration_batch_id, batch_ordinal, event_type, "
        "attempt, worker_id, lease_expires_at, supersedes_batch_event_id) "
        "VALUES (%s,%s,%s,%s,%s,'claimed',%s,%s,now()+interval '1 minute',%s) "
        "RETURNING migration_batch_event_id",
        (
            batch["migration_run_id"],
            batch["source_namespace"],
            batch["source_table"],
            batch_id,
            batch["batch_ordinal"],
            batch["head_attempt"] + 1,
            worker,
            batch["head_event_id"],
        ),
    ).fetchone()["migration_batch_event_id"]


def _complete_batch(conn, batch_id: int, worker: str = "worker-1") -> int:
    batch = _batch_row(conn, batch_id)
    return conn.execute(
        "INSERT INTO brain_migration_batch_events "
        "(migration_run_id, source_namespace, source_table, migration_batch_id, batch_ordinal, event_type, "
        "attempt, worker_id, source_count, target_count, canonical_batch_hash, supersedes_batch_event_id) "
        "VALUES (%s,%s,%s,%s,%s,'completed',%s,%s,100,100,%s,%s) RETURNING migration_batch_event_id",
        (
            batch["migration_run_id"],
            batch["source_namespace"],
            batch["source_table"],
            batch_id,
            batch["batch_ordinal"],
            batch["head_attempt"],
            worker,
            HASHES[1],
            batch["head_event_id"],
        ),
    ).fetchone()["migration_batch_event_id"]


def _checkpoint_batch(conn, batch_id: int, completed_event_id: int) -> None:
    batch = _batch_row(conn, batch_id)
    conn.execute(
        "INSERT INTO brain_migration_checkpoints "
        "(migration_run_id,source_namespace,source_table,batch_ordinal,last_key,migration_batch_id,"
        "migration_batch_event_id,canonical_checkpoint_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        (
            batch["migration_run_id"],
            batch["source_namespace"],
            batch["source_table"],
            batch["batch_ordinal"],
            batch["key_end"],
            batch_id,
            completed_event_id,
            HASHES[1],
        ),
    )


def _complete_run(conn, run_id: int) -> int:
    head = conn.execute(
        "SELECT migration_run_event_id FROM brain_migration_run_events WHERE migration_run_id=%s "
        "ORDER BY migration_run_event_id DESC LIMIT 1",
        (run_id,),
    ).fetchone()["migration_run_event_id"]
    return conn.execute(
        "INSERT INTO brain_migration_run_events "
        "(migration_run_id,source_namespace,event_type,actor_id,supersedes_run_event_id) "
        "VALUES (%s,'sqlite','completed','controller',%s) RETURNING migration_run_event_id",
        (run_id, head),
    ).fetchone()["migration_run_event_id"]


def _protect_artifact(conn, artifact_hash: str, suffix: str) -> None:
    conn.execute(
        "INSERT INTO brain_artifact_locations "
        "(artifact_hash,backend,bucket_or_container,object_key,provider_version_id,provider_checksum,"
        "storage_immutable,encryption_mode,encryption_key_id,durability,verified_at) "
        "VALUES (%s,'s3','protected',%s,%s,%s,TRUE,'customer_managed','key-v1','verified',now()) "
        "ON CONFLICT DO NOTHING",
        (artifact_hash, f"receipts/{suffix}", f"version-{suffix}", artifact_hash),
    )


def test_real_shape_import_preserves_unclear_pairwise_and_null_outcome_weight(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        pairwise_id = conn.execute(
            "INSERT INTO brain_pairwise_events "
            "(source_namespace, source_event_id, left_job_id, right_job_id, project, method, preference, occurred_at) "
            "VALUES ('sqlite', 'pair-unclear', 'job-1', 'job-2', 'applypilot', 'import', 'unclear', now()) "
            "RETURNING pairwise_event_id"
        ).fetchone()["pairwise_event_id"]
        email_id = conn.execute(
            "INSERT INTO brain_email_events "
            "(source_namespace, source_event_id, job_id, event_type, occurred_at) "
            "VALUES ('gmail', 'mail-null-weight', 'job-1', 'reply', now()) RETURNING email_event_id"
        ).fetchone()["email_event_id"]
        outcome_id = conn.execute(
            "INSERT INTO brain_reviewed_outcomes "
            "(source_namespace, source_event_id, job_id, email_event_id, review_status, normalized_stage, "
            "weight, reviewer, created_at, reviewed_at, updated_at) "
            "VALUES ('sqlite', 'outcome-null-weight', 'job-1', %s, 'confirmed', 'interview', "
            "NULL, 'owner', now(), now(), now()) RETURNING reviewed_outcome_id",
            (email_id,),
        ).fetchone()["reviewed_outcome_id"]
        assert (
            conn.execute(
                "SELECT preference FROM brain_pairwise_events WHERE pairwise_event_id=%s", (pairwise_id,)
            ).fetchone()["preference"]
            == "unclear"
        )
        assert (
            conn.execute(
                "SELECT weight FROM brain_reviewed_outcomes WHERE reviewed_outcome_id=%s", (outcome_id,)
            ).fetchone()["weight"]
            is None
        )


def test_migration_definitions_are_immutable_and_events_record_run_and_batch_progress(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        run_id, batch_ids = _insert_migration_run_and_batches(conn, 1)
        _claim_batch(conn, batch_ids[0])
        completed_id = _complete_batch(conn, batch_ids[0])
        _checkpoint_batch(conn, batch_ids[0], completed_id)
        _complete_run(conn, run_id)
        conn.commit()
        for relation in (
            "brain_migration_runs",
            "brain_migration_run_events",
            "brain_migration_batches",
            "brain_migration_batch_events",
        ):
            with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState, match="append-only"):
                with conn.transaction():
                    conn.execute(f"UPDATE {relation} SET created_at=created_at")


def test_composite_migration_lineage_and_checkpoint_completion_are_enforced(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        run_id, batch_ids = _insert_migration_run_and_batches(conn, 1)
        other_source = conn.execute(
            "INSERT INTO brain_migration_sources "
            "(source_namespace, source_fingerprint, byte_length, schema_metadata) "
            "VALUES ('sqlite', %s, 1, '{}') RETURNING migration_source_id",
            (HASHES[2],),
        ).fetchone()["migration_source_id"]
        other_run = conn.execute(
            "INSERT INTO brain_migration_runs (migration_source_id, source_namespace, run_key, metadata) "
            "VALUES (%s, 'sqlite', 'other', '{}') RETURNING migration_run_id",
            (other_source,),
        ).fetchone()["migration_run_id"]
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            with conn.transaction():
                conn.execute(
                    "INSERT INTO brain_migration_batch_events "
                    "(migration_run_id,source_namespace,source_table,migration_batch_id,batch_ordinal,event_type,"
                    "attempt,worker_id,lease_expires_at,supersedes_batch_event_id) "
                    "VALUES (%s,'sqlite','jobs',%s,1,'claimed',1,'worker',now()+interval '1 minute',1)",
                    (other_run, batch_ids[0]),
                )
        pending_id = _batch_row(conn, batch_ids[0])["head_event_id"]
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.transaction():
                conn.execute(
                    "INSERT INTO brain_migration_checkpoints "
                    "(migration_run_id,source_namespace,source_table,batch_ordinal,last_key,migration_batch_id,"
                    "migration_batch_event_id,committed_event_type,canonical_checkpoint_hash) "
                    "VALUES (%s,'sqlite','jobs',1,'100',%s,%s,'completed',%s)",
                    (run_id, batch_ids[0], pending_id, HASHES[1]),
                )


def test_parity_pass_requires_clean_results_and_exact_pass_event(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        run_id, batch_ids = _insert_migration_run_and_batches(conn, 1)
        _claim_batch(conn, batch_ids[0])
        completed_batch = _complete_batch(conn, batch_ids[0])
        _checkpoint_batch(conn, batch_ids[0], completed_batch)
        completed_run = _complete_run(conn, run_id)
        for artifact_hash, suffix in zip(HASHES[:3], ("report", "delta", "freeze"), strict=True):
            _protect_artifact(conn, artifact_hash, suffix)
        parity_id = conn.execute(
            "INSERT INTO brain_parity_runs "
            "(migration_run_id,source_namespace,definition_version,completed_run_event_id,report_artifact_hash,"
            "final_delta_receipt_hash,writer_freeze_receipt_hash,started_at) "
            "VALUES (%s,'sqlite',1,%s,%s,%s,%s,now()) RETURNING parity_run_id",
            (run_id, completed_run, HASHES[0], HASHES[1], HASHES[2]),
        ).fetchone()["parity_run_id"]
        definitions = conn.execute(
            "SELECT check_key,relation_name FROM brain_parity_definitions WHERE definition_version=1 ORDER BY check_key"
        ).fetchall()
        for definition in definitions:
            mismatch = definition["check_key"] == "jobs"
            conn.execute(
                "INSERT INTO brain_parity_results "
                "(migration_run_id,source_namespace,parity_run_id,definition_version,check_key,table_name,"
                "check_type,source_count,target_count,source_hash,target_hash,mismatch_count,unresolved_count,"
                "report_artifact_hash) VALUES (%s,'sqlite',%s,1,%s,%s,'canonical',2,%s,%s,%s,%s,%s,%s)",
                (
                    run_id,
                    parity_id,
                    definition["check_key"],
                    definition["relation_name"],
                    1 if mismatch else 2,
                    HASHES[1],
                    HASHES[2] if mismatch else HASHES[1],
                    1 if mismatch else 0,
                    1 if mismatch else 0,
                    HASHES[0],
                ),
            )
        with pytest.raises(psycopg.errors.CheckViolation, match="parity pass requires"):
            with conn.transaction():
                conn.execute(
                    "INSERT INTO brain_parity_run_events "
                    "(migration_run_id,source_namespace,parity_run_id,event_type,actor_id) "
                    "VALUES (%s,'sqlite',%s,'passed','verifier')",
                    (run_id, parity_id),
                )
        clean_parity_id = conn.execute(
            "INSERT INTO brain_parity_runs "
            "(migration_run_id,source_namespace,definition_version,completed_run_event_id,report_artifact_hash,"
            "final_delta_receipt_hash,writer_freeze_receipt_hash,started_at) "
            "VALUES (%s,'sqlite',1,%s,%s,%s,%s,now()) RETURNING parity_run_id",
            (run_id, completed_run, HASHES[0], HASHES[1], HASHES[2]),
        ).fetchone()["parity_run_id"]
        for definition in definitions:
            conn.execute(
                "INSERT INTO brain_parity_results "
                "(migration_run_id,source_namespace,parity_run_id,definition_version,check_key,table_name,"
                "check_type,source_count,target_count,source_hash,target_hash,report_artifact_hash) "
                "VALUES (%s,'sqlite',%s,1,%s,%s,'canonical',2,2,%s,%s,%s)",
                (
                    run_id,
                    clean_parity_id,
                    definition["check_key"],
                    definition["relation_name"],
                    HASHES[1],
                    HASHES[1],
                    HASHES[0],
                ),
            )
        passed_event = conn.execute(
            "INSERT INTO brain_parity_run_events "
            "(migration_run_id,source_namespace,parity_run_id,event_type,actor_id) "
            "VALUES (%s,'sqlite',%s,'passed','verifier') RETURNING parity_run_event_id",
            (run_id, clean_parity_id),
        ).fetchone()["parity_run_event_id"]
        assert passed_event > 0


def test_policy_transition_graph_metadata_immutability_gate_reruns_and_retirement(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        _insert_policy(conn)
        _attach_policy_contract(conn, "ats-v1")
        failed = conn.execute(
            "INSERT INTO brain_policy_release_gate_events "
            "(policy_version,lane,lifecycle,gate_name,gate_state,checked_by,report_artifact_hash,mismatch_count) "
            "VALUES ('ats-v1','ats','validated','parity','failed','owner',%s,1) RETURNING gate_event_id",
            (HASHES[0],),
        ).fetchone()["gate_event_id"]
        passed = conn.execute(
            "INSERT INTO brain_policy_release_gate_events "
            "(policy_version,lane,lifecycle,gate_name,gate_state,checked_by,report_artifact_hash) "
            "VALUES ('ats-v1','ats','validated','parity','passed','owner',%s) RETURNING gate_event_id",
            (HASHES[0],),
        ).fetchone()["gate_event_id"]
        assert failed != passed
        conn.execute("SET LOCAL ROLE brain_schema_migrator")
        with conn.transaction():
            conn.execute("SELECT brain_transition_policy('ats-v1', 'validated')")
        with pytest.raises(psycopg.errors.CheckViolation, match="immutable after draft"):
            with conn.transaction():
                conn.execute("UPDATE brain_decision_policies SET lane='linkedin' WHERE policy_version='ats-v1'")
        with pytest.raises(psycopg.errors.CheckViolation, match="transition"):
            with conn.transaction():
                conn.execute(
                    "UPDATE brain_decision_policies SET lifecycle='active', activated_at=now() "
                    "WHERE policy_version='ats-v1'"
                )


def test_append_only_truncate_and_deep_catalog_tampering_are_rejected(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        schema.ensure_schema_v1(conn)
        with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState, match="append-only"):
            with conn.transaction():
                conn.execute("TRUNCATE brain_artifact_locations")
        conn.execute("ALTER TABLE brain_artifacts DISABLE TRIGGER brain_artifacts_append_only")
        conn.commit()
        with pytest.raises(RuntimeError, match="trigger.*enabled"):
            schema.verify_schema_v1(conn)


def test_deep_verifier_rejects_noop_function_replacement(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        schema.ensure_schema_v1(conn)
        conn.execute(
            "CREATE OR REPLACE FUNCTION brain_reject_mutation() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN PERFORM 1; RAISE EXCEPTION '% is append-only', TG_TABLE_NAME USING ERRCODE='55000'; END $$"
        )
        conn.commit()
        with pytest.raises(RuntimeError, match="catalog contract hash mismatch"):
            schema.verify_schema_v1(conn)


def test_schema_version_ledger_rejects_future_and_interrupted_versions(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        conn.execute("SET LOCAL ROLE brain_schema_migrator")
        conn.execute(
            "CREATE TABLE brain_schema_versions (version integer PRIMARY KEY, migration_name text, "
            "migration_checksum text, applied_at timestamptz, applied_by text)"
        )
        conn.commit()
        with pytest.raises(RuntimeError, match="ledger exists but is empty"):
            schema.ensure_schema_v1(conn)
        conn.execute("SET LOCAL ROLE brain_schema_migrator")
        conn.execute("DROP TABLE brain_schema_versions")
        conn.commit()
        schema.ensure_schema_v1(conn)
        conn.execute("ALTER TABLE brain_schema_versions DISABLE TRIGGER brain_schema_versions_append_only")
        conn.execute(
            "INSERT INTO brain_schema_versions "
            "(version,migration_name,migration_checksum,applied_at,applied_by) "
            "VALUES (4,'future','aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',now(),current_user)"
        )
        conn.commit()
        with pytest.raises(RuntimeError, match="unsupported or non-contiguous"):
            schema.verify_schema_v1(conn)


def test_bootstrap_superuser_migrates_without_permanent_migrator_membership(brain_pg, monkeypatch):
    monkeypatch.setenv("APPLYPILOT_BRAIN_MIGRATION_ROLE", "postgres")
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        conn.execute("REVOKE brain_schema_migrator FROM postgres")
        conn.commit()
        schema.ensure_schema_v1(conn)
        assert conn.execute(
            "SELECT 1 FROM pg_auth_members membership "
            "JOIN pg_roles parent ON parent.oid=membership.roleid "
            "JOIN pg_roles member ON member.oid=membership.member "
            "WHERE parent.rolname='brain_schema_migrator' AND member.oid=10"
        ).fetchone() is None


def test_explicit_read_only_verifier_acl_and_schema_create_revocation(brain_pg, monkeypatch):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        conn.execute("CREATE ROLE brain_schema_reader LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE")
        conn.execute("CREATE ROLE brain_broad_role LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE")
        conn.execute("GRANT CREATE ON SCHEMA public TO brain_schema_reader, brain_broad_role")
        conn.commit()
    monkeypatch.setenv("APPLYPILOT_BRAIN_VERIFIER_ROLE", "brain_schema_reader")
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        schema.ensure_schema_v1(conn)
        assert conn.execute(
            "SELECT has_schema_privilege('brain_schema_verifier', 'public', 'USAGE') AS usage, "
            "has_schema_privilege('brain_schema_verifier', 'public', 'CREATE') AS create_privilege, "
            "has_table_privilege('brain_schema_verifier', 'brain_jobs', 'SELECT') AS read, "
            "has_table_privilege('brain_schema_verifier', 'brain_jobs', 'INSERT,UPDATE,DELETE,TRUNCATE') AS write"
        ).fetchone() == {"usage": True, "create_privilege": False, "read": True, "write": False}
        assert (
            conn.execute(
                "SELECT has_schema_privilege('brain_broad_role', 'public', 'CREATE') AS create_privilege"
            ).fetchone()["create_privilege"]
            is False
        )
        conn.rollback()
        schema.verify_schema_v1(conn)


def test_partition_creation_fails_closed_when_default_contains_policy_rows(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        _insert_policy(conn)
        assert (
            conn.execute("SELECT to_regclass('public.brain_job_decisions_default') AS relation").fetchone()["relation"]
            is None
        )
        with pytest.raises(psycopg.errors.CheckViolation, match="no partition"):
            _insert_decision(conn)
        conn.execute("CREATE TABLE brain_job_decisions_default PARTITION OF brain_job_decisions DEFAULT")
        conn.commit()
        _insert_decision(conn)
        with pytest.raises(RuntimeError, match="default partition contains"):
            schema.ensure_policy_partition(conn, "ats-v1")


def test_concurrent_installers_and_batch_claims_are_serialized_and_disjoint(brain_pg):
    install_barrier = Barrier(2)

    def install() -> None:
        with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
            install_barrier.wait()
            schema.ensure_schema_v1(conn)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: install(), range(2)))

    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        run_id, _ = _insert_migration_run_and_batches(conn)
        conn.commit()

    claim_barrier = Barrier(2)

    def claim(worker: str) -> int:
        with psycopg.connect(brain_pg, row_factory=dict_row) as conn, conn.transaction():
            conn.execute("SET LOCAL ROLE brain_schema_migrator")
            claimed = conn.execute(
                "WITH candidate AS ("
                " SELECT b.migration_run_id,b.source_namespace,b.source_table,b.migration_batch_id,b.batch_ordinal,"
                " head.migration_batch_event_id,head.attempt FROM brain_migration_batches b"
                " JOIN LATERAL (SELECT e.migration_batch_event_id,e.attempt,e.event_type,e.lease_expires_at "
                " FROM brain_migration_batch_events e WHERE e.migration_batch_id=b.migration_batch_id "
                " ORDER BY e.migration_batch_event_id DESC LIMIT 1) head ON TRUE"
                " WHERE b.migration_run_id=%s AND (head.event_type IN ('pending','failed','quarantined')"
                " OR (head.event_type='claimed' AND head.lease_expires_at<=now()))"
                " ORDER BY b.migration_batch_id FOR UPDATE OF b SKIP LOCKED LIMIT 1"
                ") INSERT INTO brain_migration_batch_events "
                "(migration_run_id,source_namespace,source_table,migration_batch_id,batch_ordinal,event_type,attempt,"
                "worker_id,lease_expires_at,supersedes_batch_event_id) "
                "SELECT migration_run_id,source_namespace,source_table,migration_batch_id,batch_ordinal,'claimed',"
                "attempt+1,%s,now()+interval '1 minute',migration_batch_event_id "
                "FROM candidate RETURNING migration_batch_id",
                (run_id, worker),
            ).fetchone()["migration_batch_id"]
            claim_barrier.wait()
            return claimed

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(claim, ("worker-1", "worker-2")))
    assert len(set(claims)) == 2


def test_email_outcome_job_lineage_and_unresolved_label_supersession_identity(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        email_id = conn.execute(
            "INSERT INTO brain_email_events "
            "(source_namespace, source_event_id, job_id, event_type, occurred_at) "
            "VALUES ('gmail', 'mail-job-1', 'job-1', 'reply', now()) RETURNING email_event_id"
        ).fetchone()["email_event_id"]
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            with conn.transaction():
                conn.execute(
                    "INSERT INTO brain_reviewed_outcomes "
                    "(source_namespace, source_event_id, job_id, email_event_id, review_status, normalized_stage, "
                    "reviewer, created_at, reviewed_at, updated_at) "
                    "VALUES ('review', 'wrong-job', 'job-2', %s, 'confirmed', 'interview', "
                    "'owner', now(), now(), now())",
                    (email_id,),
                )
        first = conn.execute(
            "INSERT INTO brain_label_events "
            "(source_namespace, source_event_id, source_item_id, source_item_url, project, method, "
            "label_name, label_value, occurred_at) "
            "VALUES ('review', 'unresolved-1', 'missing-1', 'https://missing/1', 'applypilot', "
            "'manual', 'fit', 'true', now()) RETURNING label_event_id"
        ).fetchone()["label_event_id"]
        with pytest.raises(psycopg.errors.CheckViolation, match="subject/source mismatch"):
            with conn.transaction():
                conn.execute(
                    "INSERT INTO brain_label_events "
                    "(source_namespace,source_event_id,source_item_id,source_item_url,project,method,"
                    "label_name,label_value,occurred_at,supersedes_label_event_id) "
                    "VALUES ('review','unresolved-url-change','missing-1','https://missing/changed','applypilot',"
                    "'manual','fit','false',now(),%s)",
                    (first,),
                )
        with pytest.raises(psycopg.errors.CheckViolation, match="subject/source mismatch"):
            with conn.transaction():
                conn.execute(
                    "INSERT INTO brain_label_events "
                    "(source_namespace, source_event_id, source_item_id, source_item_url, project, method, "
                    "label_name, label_value, occurred_at, supersedes_label_event_id) "
                    "VALUES ('review', 'unresolved-wrong', 'missing-2', 'https://missing/2', 'applypilot', "
                    "'manual', 'fit', 'false', now(), %s)",
                    (first,),
                )


def test_stable_source_namespace_survives_snapshot_changes_and_scopes_event_identity(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        conn.execute("SET LOCAL ROLE brain_schema_migrator")
        source_ids = []
        for fingerprint in (HASHES[0], HASHES[1]):
            source_ids.append(
                conn.execute(
                    "INSERT INTO brain_migration_sources "
                    "(source_namespace, source_fingerprint, byte_length, schema_metadata) "
                    "VALUES ('applypilot-sqlite', %s, 1, '{}') RETURNING migration_source_id",
                    (fingerprint,),
                ).fetchone()["migration_source_id"]
            )
        assert source_ids[0] != source_ids[1]
        conn.execute(
            "INSERT INTO brain_label_events "
            "(source_namespace, source_event_id, job_id, project, method, label_name, label_value, occurred_at) "
            "VALUES ('applypilot-sqlite', 'local-1', 'job-1', 'applypilot', 'import', 'fit', 'true', now()), "
            "('other-sqlite', 'local-1', 'job-1', 'applypilot', 'import', 'fit', 'true', now())"
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            with conn.transaction():
                conn.execute(
                    "INSERT INTO brain_label_events "
                    "(source_namespace, source_event_id, job_id, project, method, label_name, label_value, occurred_at) "
                    "VALUES ('applypilot-sqlite', 'local-1', 'job-1', 'applypilot', 'import', 'fit', 'false', now())"
                )


def test_artifact_request_identity_replica_metadata_and_protected_archive_manifest(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        schema.ensure_schema_v1(conn)
        conn.execute(
            "INSERT INTO brain_artifacts "
            "(request_id, artifact_hash, media_type, byte_length, schema_version, provenance, location) "
            "VALUES ('artifact-request-1', %s, 'application/json', 10, 1, '{\"source\":\"test\"}', 's3://staging/a')",
            (HASHES[0],),
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            with conn.transaction():
                conn.execute(
                    "INSERT INTO brain_artifacts "
                    "(request_id, artifact_hash, media_type, byte_length, schema_version, location) "
                    "VALUES ('artifact-request-1', %s, 'application/json', 10, 1, 's3://staging/b')",
                    (HASHES[1],),
                )
        conn.execute(
            "INSERT INTO brain_artifact_locations "
            "(artifact_hash, backend, bucket_or_container, object_key, provider_version_id, provider_checksum, "
            "storage_immutable, encryption_mode, encryption_key_id, durability, verified_at) "
            "VALUES (%s, 's3', 'brain-staging', 'objects/a', 'version-1', %s, "
            "FALSE, 'provider_managed', 'aws/s3', 'committed_unprotected', NULL)",
            (HASHES[0], HASHES[0]),
        )
        with pytest.raises(psycopg.errors.CheckViolation, match="protected verified immutable replica"):
            with conn.transaction():
                conn.execute(
                    "INSERT INTO brain_archive.brain_archive_manifests "
                    "(retry_identity, source_relation, artifact_hash, row_count) "
                    "VALUES ('archive-request-1', 'brain_job_decisions', %s, 1)",
                    (HASHES[0],),
                )
        conn.execute(
            "INSERT INTO brain_artifact_locations "
            "(artifact_hash, backend, bucket_or_container, object_key, provider_version_id, provider_checksum, "
            "storage_immutable, encryption_mode, encryption_key_id, durability, verified_at) "
            "VALUES (%s, 'azure', 'brain-archive', 'objects/a', 'etag-1', %s, "
            "TRUE, 'customer_managed', 'key-v1', 'verified', now())",
            (HASHES[0], HASHES[0]),
        )
        conn.execute(
            "INSERT INTO brain_archive.brain_archive_manifests "
            "(retry_identity, source_relation, artifact_hash, row_count) "
            "VALUES ('archive-request-1', 'brain_job_decisions', %s, 1)",
            (HASHES[0],),
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            with conn.transaction():
                conn.execute(
                    "INSERT INTO brain_archive.brain_archive_manifests "
                    "(retry_identity, source_relation, artifact_hash, row_count) "
                    "VALUES ('archive-request-1', 'brain_job_decisions', %s, 1)",
                    (HASHES[0],),
                )


def test_controller_policy_transition_requires_latest_gate_and_paused_bound_fleet_config(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        fleet_schema.ensure_schema_v3(conn)
        _grant_fleet_controller_contract(conn)
        conn.commit()
        _install_fixture_data(conn)
        _insert_policy(conn)
        _attach_policy_contract(conn, "ats-v1")
        conn.execute("SET LOCAL ROLE brain_schema_migrator")
        conn.execute("SELECT brain_transition_policy('ats-v1', 'validated')")
        conn.execute("SELECT brain_transition_policy('ats-v1', 'canary')")
        conn.commit()
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="controller"):
            with conn.transaction():
                conn.execute(
                    "UPDATE brain_decision_policies SET lifecycle='active', activated_at=now() "
                    "WHERE policy_version='ats-v1'"
                )
        conn.execute(
            "INSERT INTO fleet_decision_policies (policy_version, lane, status) "
            "VALUES ('ats-v1', 'ats', 'canary') "
            "ON CONFLICT (policy_version) DO UPDATE SET lane=EXCLUDED.lane, status=EXCLUDED.status"
        )
        conn.execute(
            "UPDATE fleet_decision_policies SET status='retired', retired_at=now() "
            "WHERE lane='ats' AND policy_version<>'ats-v1' AND status='active'"
        )
        conn.execute("UPDATE fleet_config SET paused=FALSE, ats_paused=FALSE, ats_policy_version=NULL WHERE id=1")
        conn.commit()
        with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState, match="lane must remain paused"):
            with conn.transaction():
                conn.execute("SET LOCAL ROLE brain_schema_migrator")
                conn.execute("SELECT brain_transition_policy('ats-v1', 'active')")
        conn.execute("UPDATE fleet_config SET paused=TRUE WHERE id=1")
        conn.execute(
            "INSERT INTO brain_policy_release_gate_events "
            "(policy_version,lane,lifecycle,gate_name,gate_state,checked_by,report_artifact_hash,mismatch_count) "
            "VALUES ('ats-v1','ats','active','parity','failed','owner',%s,1)",
            (HASHES[0],),
        )
        conn.commit()
        with pytest.raises(psycopg.errors.CheckViolation, match="latest mandatory gate"):
            with conn.transaction():
                conn.execute("SET LOCAL ROLE brain_schema_migrator")
                conn.execute("SELECT brain_transition_policy('ats-v1', 'active')")
        conn.execute(
            "INSERT INTO brain_policy_release_gate_events "
            "(policy_version,lane,lifecycle,gate_name,gate_state,checked_by,report_artifact_hash) "
            "VALUES ('ats-v1','ats','active','parity','passed','owner',%s)",
            (HASHES[0],),
        )
        conn.execute("SET LOCAL ROLE brain_schema_migrator")
        conn.execute("SELECT brain_transition_policy('ats-v1', 'active')")
        conn.execute("SELECT brain_transition_policy('ats-v1', 'retired')")
        assert conn.execute(
            "SELECT lifecycle, retired_at IS NOT NULL AS retired FROM brain_decision_policies "
            "WHERE policy_version='ats-v1'"
        ).fetchone() == {"lifecycle": "retired", "retired": True}


def test_atomic_activation_and_retirement_leave_leased_rows_untouched(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        fleet_schema.ensure_schema_v3(conn)
        _grant_fleet_controller_contract(conn)
        conn.execute(
            "UPDATE fleet_decision_policies SET status='retired',retired_at=now() WHERE lane='ats' AND status='active'"
        )
        conn.execute("UPDATE fleet_config SET paused=TRUE,ats_paused=FALSE,ats_policy_version=NULL WHERE id=1")
        conn.commit()
        _install_fixture_data(conn)
        conn.execute("UPDATE brain_jobs SET canonical_url='https://jobs/1' WHERE job_id='job-1'")
        conn.execute(
            "INSERT INTO brain_jobs (job_id,source_namespace,source_job_id,canonical_url,title) "
            "VALUES ('job-3','legacy','3','https://jobs/3','Leased')"
        )
        conn.commit()
        for policy in ("ats-v1", "ats-v2"):
            _insert_policy(conn, policy)
            schema.ensure_policy_partition(conn, policy)
            _attach_policy_contract(conn, policy)
            _insert_decision(
                conn,
                decision_id=f"decision-{policy}",
                source_id=f"source-{policy}",
                policy=policy,
                input_hash=HASHES[1] if policy == "ats-v1" else HASHES[2],
            )
            if policy == "ats-v1":
                with pytest.raises(psycopg.errors.UniqueViolation):
                    _insert_decision(
                        conn,
                        decision_id="decision-v1-ambiguous",
                        source_id="source-v1-ambiguous",
                        policy=policy,
                        input_hash=HASHES[3],
                    )
            with conn.transaction():
                conn.execute("SET LOCAL ROLE brain_schema_migrator")
                conn.execute("SELECT brain_transition_policy(%s,'validated')", (policy,))
                conn.execute("SELECT brain_transition_policy(%s,'canary')", (policy,))
                if policy == "ats-v1":
                    conn.execute("SELECT brain_transition_policy(%s,'active')", (policy,))
        conn.execute(
            "INSERT INTO apply_queue "
            "(url,application_url,score,status,lane,approved_batch,decision_id,policy_version,decision_action,"
            "qualification_verdict,qualification_score,qualification_floor,preference_score,outcome_score,"
            "final_score,decision_confidence,decision_created_at,decision_expires_at,input_hash,lease_owner,"
            "lease_expires_at) VALUES "
            "('https://jobs/3','https://apply/3',0.8,'leased','ats','v1:leased','decision-ats-v1','ats-v1','apply',"
            "'qualified',0.8,0.6,0.7,0.9,0.8,0.8,now(),now()+interval '1 day',%s,'worker-1',"
            "now()+interval '5 minutes'),"
            "('https://jobs/1','https://apply/1',0.8,'queued','ats','v1:queued','queued-v1','ats-v1','apply',"
            "'qualified',0.8,0.6,0.7,0.9,0.8,0.8,now(),now()+interval '1 day',%s,NULL,NULL)",
            (HASHES[1], HASHES[1]),
        )
        leased_before = conn.execute(
            "SELECT to_jsonb(q) AS row FROM apply_queue q WHERE url='https://jobs/3'"
        ).fetchone()["row"]
        conn.execute("SET LOCAL ROLE brain_schema_migrator")
        conn.execute("SELECT brain_transition_policy('ats-v2','active')")
        assert (
            conn.execute("SELECT to_jsonb(q) AS row FROM apply_queue q WHERE url='https://jobs/3'").fetchone()["row"]
            == leased_before
        )
        assert conn.execute(
            "SELECT policy_version,decision_id FROM apply_queue WHERE url='https://jobs/1'"
        ).fetchone() == {"policy_version": "ats-v2", "decision_id": "decision-ats-v2"}
        pause_before = conn.execute("SELECT paused,ats_paused,ats_apply_mode FROM fleet_config WHERE id=1").fetchone()
        conn.execute("SELECT brain_transition_policy('ats-v2','retired')")
        assert (
            conn.execute("SELECT lifecycle FROM brain_decision_policies WHERE policy_version='ats-v2'").fetchone()[
                "lifecycle"
            ]
            == "retired"
        )
        assert (
            conn.execute("SELECT status FROM fleet_decision_policies WHERE policy_version='ats-v2'").fetchone()[
                "status"
            ]
            == "retired"
        )
        assert (
            conn.execute("SELECT ats_policy_version FROM fleet_config WHERE id=1").fetchone()["ats_policy_version"]
            is None
        )
        assert conn.execute(
            "SELECT approved_batch,policy_version,decision_id FROM apply_queue WHERE url='https://jobs/1'"
        ).fetchone() == {"approved_batch": None, "policy_version": None, "decision_id": None}
        assert (
            conn.execute("SELECT to_jsonb(q) AS row FROM apply_queue q WHERE url='https://jobs/3'").fetchone()["row"]
            == leased_before
        )
        assert (
            conn.execute("SELECT paused,ats_paused,ats_apply_mode FROM fleet_config WHERE id=1").fetchone()
            == pause_before
        )


def test_parity_results_bind_exact_definition_and_protected_report(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        run_id, batch_ids = _insert_migration_run_and_batches(conn, 1)
        _claim_batch(conn, batch_ids[0])
        completed_batch = _complete_batch(conn, batch_ids[0])
        _checkpoint_batch(conn, batch_ids[0], completed_batch)
        completed_run = _complete_run(conn, run_id)
        _protect_artifact(conn, HASHES[0], "parity-report")
        parity_id = conn.execute(
            "INSERT INTO brain_parity_runs "
            "(migration_run_id,source_namespace,definition_version,completed_run_event_id,report_artifact_hash,"
            "final_delta_receipt_hash,writer_freeze_receipt_hash,started_at) "
            "VALUES (%s,'sqlite',1,%s,%s,%s,%s,now()) RETURNING parity_run_id",
            (run_id, completed_run, HASHES[0], HASHES[1], HASHES[2]),
        ).fetchone()["parity_run_id"]
        base_sql = (
            "INSERT INTO brain_parity_results "
            "(migration_run_id,source_namespace,parity_run_id,definition_version,check_key,table_name,check_type,"
            "source_count,target_count,source_hash,target_hash,report_artifact_hash) "
            "VALUES (%s,'sqlite',%s,1,'jobs',%s,%s,1,1,%s,%s,%s)"
        )
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            with conn.transaction():
                conn.execute(
                    base_sql,
                    (run_id, parity_id, "brain_label_events", "canonical", HASHES[1], HASHES[1], HASHES[0]),
                )
        with pytest.raises((psycopg.errors.CheckViolation, psycopg.errors.ForeignKeyViolation)):
            with conn.transaction():
                conn.execute(
                    base_sql,
                    (run_id, parity_id, "brain_jobs", "evil", HASHES[1], HASHES[1], HASHES[0]),
                )
        with pytest.raises(psycopg.errors.CheckViolation, match="protected"):
            with conn.transaction():
                conn.execute(
                    base_sql,
                    (run_id, parity_id, "brain_jobs", "canonical", HASHES[1], HASHES[1], HASHES[3]),
                )


def test_gate_definition_version_cannot_cross_satisfy_transition(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        _insert_policy(conn)
        _protect_artifact(conn, HASHES[0], "gate-v2")
        conn.execute(
            "INSERT INTO brain_policy_gate_definitions (definition_version,lane,lifecycle,gate_name) "
            "SELECT 2,lane,lifecycle,gate_name FROM brain_policy_gate_definitions WHERE definition_version=1"
        )
        definitions = conn.execute(
            "SELECT gate_name FROM brain_policy_gate_definitions "
            "WHERE definition_version=2 AND lane='ats' AND lifecycle='validated'"
        ).fetchall()
        for definition in definitions:
            conn.execute(
                "INSERT INTO brain_policy_release_gate_events "
                "(policy_version,lane,lifecycle,definition_version,gate_name,gate_state,checked_by,report_artifact_hash) "
                "VALUES ('ats-v1','ats','validated',2,%s,'passed','owner',%s)",
                (definition["gate_name"], HASHES[0]),
            )
        conn.execute("SET LOCAL ROLE brain_schema_migrator")
        with pytest.raises(psycopg.errors.CheckViolation, match="latest mandatory gate"):
            conn.execute("SELECT brain_transition_policy('ats-v1','validated')")


def test_partition_catalog_rejects_extra_index_and_constraint(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        _install_fixture_data(conn)
        _insert_policy(conn)
        partition = schema.ensure_policy_partition(conn, "ats-v1")
        conn.execute(
            pg_sql.SQL("CREATE INDEX {} ON {}.{} (created_at)").format(
                pg_sql.Identifier("brain_partition_evil_idx"),
                pg_sql.Identifier("public"),
                pg_sql.Identifier(partition),
            )
        )
        conn.commit()
        with pytest.raises(RuntimeError, match="partition.*index"):
            schema.verify_schema_v1(conn)
        conn.execute("DROP INDEX brain_partition_evil_idx")
        conn.execute(
            pg_sql.SQL("ALTER TABLE {}.{} ADD CONSTRAINT {} CHECK (true)").format(
                pg_sql.Identifier("public"),
                pg_sql.Identifier(partition),
                pg_sql.Identifier("brain_partition_evil_ck"),
            )
        )
        conn.commit()
        with pytest.raises(RuntimeError, match="partition.*constraint"):
            schema.verify_schema_v1(conn)


def test_exact_owner_rejects_membership_proxy_owner(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        schema.ensure_schema_v1(conn)
        conn.execute("CREATE ROLE brain_proxy_owner NOLOGIN")
        conn.execute("GRANT brain_proxy_owner TO brain_schema_migrator")
        conn.execute("ALTER TABLE brain_jobs OWNER TO brain_proxy_owner")
        conn.commit()
        with pytest.raises(RuntimeError, match="ownership mismatch.*brain_jobs.*brain_proxy_owner"):
            schema.verify_schema_v1(conn)


def test_bootstrap_v5_authority_creates_candidates_before_schema_and_reconciles_after(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        schema.ensure_brain_schema_v1(conn)
        require_disposable_postgres(conn)
        conn.execute("REVOKE ALL PRIVILEGES ON SCHEMA public FROM brain_candidate_reader, brain_candidate_writer")
        conn.execute("DROP ROLE brain_candidate_reader")
        conn.execute("DROP ROLE brain_candidate_writer")
        conn.execute(
            "CREATE ROLE brain_controller_test_login LOGIN NOINHERIT NOSUPERUSER "
            "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
        )
        conn.commit()

        topology = pg_roles.BootstrapTopology(
            database_owner_role="postgres",
            controller_role="brain_controller_test_login",
            verifier_role="brain_schema_verifier",
            migrator_role="brain_schema_migrator",
            retired_admin_roles=(),
            infrastructure_superuser_roles=("postgres",),
        )
        with conn.transaction():
            with conn.cursor() as cur:
                candidate_roles = pg_roles._install_brain_authority_in_transaction(cur, topology=topology)

        assert conn.execute(
            "SELECT current_user=session_user AS role_restored"
        ).fetchone()["role_restored"] is True
        assert candidate_roles.reader_role == "brain_candidate_reader"
        assert candidate_roles.writer_role == "brain_candidate_writer"
        assert conn.execute(
            "SELECT array_agg(version ORDER BY version) AS versions FROM public.brain_schema_versions"
        ).fetchone()["versions"] == [1, 2, 3, 4, 5, 6, 7]
        assert (
            conn.execute(
                "SELECT has_function_privilege('brain_candidate_writer',"
                "'public.brain_publish_v4_candidate(text,text,text,text,text,bigint,uuid,text,text,text,text,text,bigint)'"
                "::regprocedure,'EXECUTE') AS allowed"
            ).fetchone()["allowed"]
            is False
        )
        assert (
            conn.execute(
                "SELECT has_function_privilege('brain_candidate_writer',"
                "'public.brain_publish_v5_candidate(text,text,text,text,text,bigint,uuid,text,text,text,text,text,bigint)'"
                "::regprocedure,'EXECUTE') AS allowed"
            ).fetchone()["allowed"]
            is True
        )
        conn.commit()
        schema.verify_brain_schema_v7(conn)
        assert conn.execute(
            "SELECT membership.admin_option,membership.inherit_option,membership.set_option,"
            "grantor.rolname AS grantor_role "
            "FROM pg_auth_members membership "
            "JOIN pg_roles parent ON parent.oid=membership.roleid "
            "JOIN pg_roles member ON member.oid=membership.member "
            "JOIN pg_roles grantor ON grantor.oid=membership.grantor "
            "WHERE parent.rolname='brain_schema_migrator' AND member.rolname='postgres'"
        ).fetchone() is None


def test_v7_bootstrap_rejects_unauthorized_transitive_authority_descendant_and_rolls_back(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as conn:
        schema.ensure_brain_schema_v1(conn)
        require_disposable_postgres(conn)
        conn.execute(
            "CREATE ROLE brain_controller_test_login LOGIN NOINHERIT NOSUPERUSER "
            "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
        )
        conn.execute(
            "CREATE ROLE brain_rogue_writer NOLOGIN NOINHERIT NOSUPERUSER "
            "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
        )
        conn.execute("GRANT brain_schema_migrator TO brain_rogue_writer")
        conn.commit()
        versions_before = conn.execute(
            "SELECT array_agg(version ORDER BY version) AS versions FROM public.brain_schema_versions"
        ).fetchone()["versions"]
        topology = pg_roles.BootstrapTopology(
            database_owner_role="postgres",
            controller_role="brain_controller_test_login",
            verifier_role="brain_schema_verifier",
            migrator_role="brain_schema_migrator",
            retired_admin_roles=(),
            infrastructure_superuser_roles=("postgres",),
        )
        with pytest.raises(
            RuntimeError,
            match="unauthorized direct or transitive descendants|recursive authority membership contract mismatch",
        ):
            with conn.transaction():
                with conn.cursor() as cur:
                    pg_roles._install_brain_authority_in_transaction(cur, topology=topology)
        conn.rollback()
        assert conn.execute(
            "SELECT array_agg(version ORDER BY version) AS versions FROM public.brain_schema_versions"
        ).fetchone()["versions"] == versions_before
        assert conn.execute(
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_auth_members membership "
            "JOIN pg_roles parent ON parent.oid=membership.roleid "
            "JOIN pg_roles member ON member.oid=membership.member "
            "WHERE parent.rolname='brain_schema_migrator' "
            "AND member.rolname='brain_rogue_writer') AS present"
        ).fetchone()["present"] is True


def test_verifier_has_public_usage_and_only_enumerated_brain_reads(brain_pg):
    with psycopg.connect(brain_pg, row_factory=dict_row) as owner:
        fleet_schema.ensure_schema_v3(owner)
        _grant_fleet_controller_contract(owner)
        owner.commit()
        schema.ensure_schema_v1(owner)
        assert owner.execute(
            "SELECT has_schema_privilege('brain_schema_verifier','public','USAGE') AS public_usage,"
            "has_schema_privilege('brain_schema_verifier','public','CREATE') AS public_create,"
            "has_table_privilege('brain_schema_verifier','brain_jobs','SELECT') AS brain_read,"
            "has_table_privilege('brain_schema_verifier','fleet_config','SELECT') AS fleet_read"
        ).fetchone() == {
            "public_usage": True,
            "public_create": False,
            "brain_read": True,
            "fleet_read": False,
        }
    with psycopg.connect(_dsn_for(brain_pg, "brain_schema_verifier"), row_factory=dict_row) as verifier:
        assert verifier.execute("SELECT count(*) AS count FROM brain_jobs").fetchone()["count"] == 0
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            verifier.execute("SELECT count(*) FROM fleet_config")


# --- Runtime Packet S/H: V8 incarnation and retention authority ---------------

_SCHEMA_HISTORY_DIR = Path(__file__).parent / "fixtures" / "brain_schema_history"


def _load_schema_history() -> dict[int, dict]:
    history: dict[int, dict] = {}
    for path in sorted(_SCHEMA_HISTORY_DIR.glob("v*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        history[record["version"]] = record
    return history


def _append_v8(connection) -> None:
    """Append V8 over an installed V1-V7 authority, managing schema ownership."""
    with connection.transaction():
        connection.execute("ALTER SCHEMA public OWNER TO brain_schema_migrator")
        with connection.cursor() as cursor:
            schema.ensure_brain_schema_v8_in_transaction(
                cursor, expected_controller_role="brain_controller_test_login"
            )
        connection.execute("RESET ROLE")
        connection.execute("ALTER SCHEMA public OWNER TO pg_database_owner")


def _install_v8_authority(connection) -> None:
    _install_v7_authority(connection)
    _append_v8(connection)


def _append_v9(connection) -> None:
    with connection.transaction():
        connection.execute("ALTER SCHEMA public OWNER TO brain_schema_migrator")
        with connection.cursor() as cursor:
            schema.ensure_brain_schema_v9_in_transaction(
                cursor, expected_controller_role="brain_controller_test_login"
            )
        connection.execute("RESET ROLE")
        connection.execute("ALTER SCHEMA public OWNER TO pg_database_owner")


def _register_v8_artifact(connection, *, request_id: str, digest: str, artifact_hash: str):
    params = _v7_registration_params(
        connection, request_id=request_id, digest=digest, artifact_hash=artifact_hash
    )
    connection.execute("SET ROLE brain_artifact_authority_writer")
    connection.execute(_V7_REGISTER_SQL, params)
    connection.execute("RESET ROLE")
    connection.commit()
    return {
        "backend": "s3",
        "bucket_or_container": "immutable",
        "object_key": f"sha256/{artifact_hash}",
        "provider_version_id": f"version-{artifact_hash[:8]}",
        "provider_checksum": f"checksum-{artifact_hash[:8]}",
    }


def test_v8_database_incarnation_is_stable_singleton_resistant_to_guc_spoofing(brain_pg) -> None:
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v8_authority(connection)
        connection.commit()
        incarnation = connection.execute(
            "SELECT database_incarnation_id::text AS id "
            "FROM public.applypilot_database_identity WHERE singleton_id=1"
        ).fetchone()["id"]
        assert incarnation is not None
        connection.commit()

        # Stable across an independent connection.
        with psycopg.connect(brain_pg, row_factory=dict_row) as other:
            assert other.execute(
                "SELECT database_incarnation_id::text AS id "
                "FROM public.applypilot_database_identity WHERE singleton_id=1"
            ).fetchone()["id"] == incarnation

        # A spoofed custom GUC cannot change the authoritative incarnation.
        connection.execute(
            "SELECT set_config('applypilot.database_incarnation_id', %s, false)", ("deadbeef-spoof",)
        )
        assert connection.execute(
            "SELECT database_incarnation_id::text AS id "
            "FROM public.applypilot_database_identity WHERE singleton_id=1"
        ).fetchone()["id"] == incarnation
        connection.commit()

        # Ordinary update/delete/truncate are denied (append-only).
        for statement in (
            "UPDATE public.applypilot_database_identity SET database_incarnation_id=gen_random_uuid()",
            "DELETE FROM public.applypilot_database_identity",
            "TRUNCATE public.applypilot_database_identity",
        ):
            with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState, match="append-only"):
                with connection.transaction():
                    connection.execute(statement)

        # A second singleton row is rejected.
        with pytest.raises(psycopg.errors.UniqueViolation):
            with connection.transaction():
                connection.execute(
                    "INSERT INTO public.applypilot_database_identity (singleton_id) VALUES (1)"
                )
        assert connection.execute(
            "SELECT database_incarnation_id::text AS id "
            "FROM public.applypilot_database_identity WHERE singleton_id=1"
        ).fetchone()["id"] == incarnation
        connection.commit()
        schema.verify_brain_schema_v8(connection)


def test_v8_authority_requires_current_retention_attestation_lifecycle(brain_pg) -> None:
    artifact_hash = "2" * 64
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v8_authority(connection)
        location = _register_v8_artifact(
            connection,
            request_id="81111111-1111-1111-1111-111111111111",
            digest="1" * 64,
            artifact_hash=artifact_hash,
        )

        # An old registration with no V8 retention attestation is non-authoritative.
        assert connection.execute(
            "SELECT brain_artifact_is_authoritative(%s) AS authoritative", (artifact_hash,)
        ).fetchone()["authoritative"] is False
        connection.commit()

        attest_sql = (
            "SELECT brain_attest_artifact_retention("
            "%s,%s,%s,%s,%s,%s,%s::timestamptz,%s::timestamptz,%s,%s::timestamptz) AS id"
        )
        # verified_at is bound to the exact verified location observation; the live
        # lease must sit at or before the COMPLIANCE retention deadline.
        location_verified_at = connection.execute(
            "SELECT verified_at FROM public.brain_artifact_locations "
            "WHERE artifact_hash=%s AND object_key=%s",
            (artifact_hash, location["object_key"]),
        ).fetchone()["verified_at"]
        retain_until = connection.execute(
            "SELECT (clock_timestamp()+interval '1 second') AS retain_until"
        ).fetchone()["retain_until"]
        connection.commit()
        attest_args = (
            artifact_hash, location["backend"], location["bucket_or_container"],
            location["object_key"], location["provider_version_id"], location["provider_checksum"],
            retain_until, location_verified_at, "lease-token-1", retain_until,
        )
        connection.execute("SET ROLE brain_artifact_authority_writer")
        first_id = connection.execute(attest_sql, attest_args).fetchone()["id"]
        # Replay of an identical attestation cannot move the deadline.
        assert connection.execute(attest_sql, attest_args).fetchone()["id"] == first_id
        connection.execute("RESET ROLE")
        connection.commit()

        # A matching future COMPLIANCE attestation makes the artifact authoritative.
        assert connection.execute(
            "SELECT brain_artifact_is_authoritative(%s) AS authoritative", (artifact_hash,)
        ).fetchone()["authoritative"] is True
        connection.commit()

        # It becomes false once retain-until passes.
        connection.execute("SELECT pg_sleep(1.5)")
        connection.commit()
        assert connection.execute(
            "SELECT brain_artifact_is_authoritative(%s) AS authoritative", (artifact_hash,)
        ).fetchone()["authoritative"] is False
        connection.commit()

        # Extension requires a NEW append-only attestation (a new row, not a mutation).
        extend_retain = connection.execute(
            "SELECT (clock_timestamp()+interval '1 hour') AS retain_until"
        ).fetchone()["retain_until"]
        connection.commit()
        extend_args = (
            artifact_hash, location["backend"], location["bucket_or_container"],
            location["object_key"], location["provider_version_id"], location["provider_checksum"],
            extend_retain, location_verified_at, "lease-token-2", extend_retain,
        )
        connection.execute("SET ROLE brain_artifact_authority_writer")
        extended_id = connection.execute(attest_sql, extend_args).fetchone()["id"]
        connection.execute("RESET ROLE")
        connection.commit()
        assert extended_id != first_id
        assert connection.execute(
            "SELECT count(*) AS count FROM public.brain_artifact_retention_attestations "
            "WHERE artifact_hash=%s",
            (artifact_hash,),
        ).fetchone()["count"] == 2
        assert connection.execute(
            "SELECT brain_artifact_is_authoritative(%s) AS authoritative", (artifact_hash,)
        ).fetchone()["authoritative"] is True
        connection.commit()

        # No guessed backfill: an attestation for an unregistered artifact is rejected.
        connection.execute("SET ROLE brain_artifact_authority_writer")
        with pytest.raises(psycopg.errors.CheckViolation, match="verified registered artifact"):
            with connection.transaction():
                connection.execute(
                    attest_sql,
                    ("f" * 64, "s3", "immutable", "sha256/" + "f" * 64,
                     "version-ffffffff", "checksum-ffffffff",
                     extend_retain, location_verified_at, "lease-x", extend_retain),
                )
        connection.execute("RESET ROLE")
        connection.commit()

        # Retention rows are append-only.
        for statement in (
            "UPDATE public.brain_artifact_retention_attestations SET retain_until=now()+interval '10 years'",
            "DELETE FROM public.brain_artifact_retention_attestations",
        ):
            with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState, match="append-only"):
                with connection.transaction():
                    connection.execute(statement)
        connection.commit()
        schema.verify_brain_schema_v8(connection)


def test_v8_upgrade_appends_without_rewriting_historical_ledger(brain_pg) -> None:
    history = _load_schema_history()
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v7_authority(connection)
        before = {
            row["version"]: row
            for row in connection.execute(
                "SELECT version,migration_name,migration_checksum,applied_by "
                "FROM public.brain_schema_versions ORDER BY version"
            ).fetchall()
        }
        connection.commit()
        # The installed committed byte variants match the immutable history fixtures.
        for version in (1, 2, 5, 7):
            assert before[version]["migration_checksum"] == history[version]["current_committed"]["sha256"]
            assert before[version]["migration_name"] == history[version]["migration_name"]

        _append_v8(connection)

        after = {
            row["version"]: row
            for row in connection.execute(
                "SELECT version,migration_name,migration_checksum,applied_by "
                "FROM public.brain_schema_versions ORDER BY version"
            ).fetchall()
        }
        connection.commit()
        # Every historical V1-V7 ledger row is byte-for-byte unchanged.
        for version in (1, 2, 3, 4, 5, 6, 7):
            assert after[version] == before[version]
        assert after[8]["migration_checksum"] == schema._EXPECTED_V8_CHECKSUM
        assert after[8]["migration_name"] == schema._MIGRATION_V8_NAME
        assert after[8]["applied_by"] == "brain_schema_migrator"
        schema.verify_brain_schema_v8(connection)


def test_v8_rejects_known_historical_checksum_over_the_wrong_catalog(brain_pg) -> None:
    history = _load_schema_history()
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v8_authority(connection)
        connection.commit()
        schema.verify_brain_schema_v8(connection)

        # The published V7 prior variant substituted over the current catalog is rejected.
        v7_prior = next(
            variant["sha256"]
            for variant in history[7]["historical_variants"]
            if variant["label"] == "prior"
        )
        substitutions = (
            (7, v7_prior, "invalid version 7 contract"),
            (5, "9" * 64, "invalid version 5 contract"),
            (2, "8" * 64, "unrecognized version 2 checksum"),
            (1, "7" * 64, "unrecognized version 1 checksum"),
        )
        for version, wrong_checksum, message in substitutions:
            with pytest.raises(RuntimeError, match=message):
                with connection.transaction():
                    connection.execute("SET LOCAL session_replication_role=replica")
                    connection.execute(
                        "UPDATE public.brain_schema_versions SET migration_checksum=%s WHERE version=%s",
                        (wrong_checksum, version),
                    )
                    with connection.cursor() as cursor:
                        schema.verify_brain_schema_v8_in_transaction(cursor)
            connection.rollback()

        # The published legacy V1 variant, by contrast, remains an accepted ledger value.
        legacy_v1 = next(
            variant["sha256"]
            for variant in history[1]["historical_variants"]
            if variant.get("ledger_accepted")
        )
        with connection.transaction():
            connection.execute("SET LOCAL session_replication_role=replica")
            connection.execute(
                "UPDATE public.brain_schema_versions SET migration_checksum=%s WHERE version=1",
                (legacy_v1,),
            )
            with connection.cursor() as cursor:
                schema.verify_brain_schema_v8_in_transaction(cursor)
        connection.rollback()
        schema.verify_brain_schema_v8(connection)


# --- Runtime Packet S5 repair: V8 adversarial coverage ------------------------

_V8_ATTEST_SQL = (
    "SELECT brain_attest_artifact_retention("
    "%s,%s,%s,%s,%s,%s,%s::timestamptz,%s::timestamptz,%s,%s::timestamptz) AS id"
)
_V8_ATTESTATION_COLUMNS = (
    "database_incarnation_id,artifact_hash,backend,bucket_or_container,object_key,"
    "provider_version_id,provider_checksum,retention_mode,retain_until,verified_at,"
    "verification_lease_id,verification_lease_expires_at"
)


def _v8_current_incarnation(connection) -> str:
    return connection.execute(
        "SELECT database_incarnation_id::text AS id "
        "FROM public.applypilot_database_identity WHERE singleton_id=1"
    ).fetchone()["id"]


def _v8_location_verified_at(connection, artifact_hash: str, object_key: str):
    return connection.execute(
        "SELECT verified_at FROM public.brain_artifact_locations "
        "WHERE artifact_hash=%s AND object_key=%s",
        (artifact_hash, object_key),
    ).fetchone()["verified_at"]


def _v8_attest_args(location, *, artifact_hash, retain_until, verified_at, lease_id, lease_expires):
    return (
        artifact_hash, location["backend"], location["bucket_or_container"],
        location["object_key"], location["provider_version_id"], location["provider_checksum"],
        retain_until, verified_at, lease_id, lease_expires,
    )


def test_schema_history_fixtures_are_exact_raw_blobs() -> None:
    history = _load_schema_history()
    committed = {
        1: schema._SCHEMA_SQL,
        2: schema._SCHEMA_V2_SQL,
        5: schema._SCHEMA_V5_SQL,
        7: schema._SCHEMA_V7_SQL,
    }
    for version, record in history.items():
        committed_bytes = committed[version].read_bytes()
        assert record["current_committed"]["sha256"] == hashlib.sha256(committed_bytes).hexdigest()
        for variant in record["historical_variants"]:
            if "blob" not in variant:
                continue
            blob_path = _SCHEMA_HISTORY_DIR / variant["blob"]
            blob_bytes = blob_path.read_bytes()
            # The stored fixture is the exact raw historical migration bytes -- never a
            # reconstruction, never a truncated prefix used as authority.
            assert hashlib.sha256(blob_bytes).hexdigest() == variant["sha256"]
            if "sha256_prefix" in variant:
                assert variant["sha256"].startswith(variant["sha256_prefix"])
            # It is a genuinely earlier, behaviourally different variant, not the
            # current committed bytes wearing a historical label.
            assert blob_bytes != committed_bytes
            assert variant["sha256"] != record["current_committed"]["sha256"]


def test_v8_rejects_every_historical_variant_checksum_over_the_current_catalog(brain_pg) -> None:
    history = _load_schema_history()
    reject_message = {
        1: "unrecognized version 1 checksum",
        2: "unrecognized version 2 checksum",
        5: "invalid version 5 contract",
        7: "invalid version 7 contract",
    }
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v8_authority(connection)
        connection.commit()
        connection.commit()
        schema.verify_brain_schema_v8(connection)
        substitutions = [
            (version, variant["sha256"])
            for version, record in history.items()
            for variant in record["historical_variants"]
            if variant.get("blob") and not variant.get("ledger_accepted")
        ]
        # Every prefix-published historical variant now has a full, exact digest.
        assert len(substitutions) == 5
        for version, wrong_checksum in substitutions:
            with pytest.raises(RuntimeError, match=reject_message[version]):
                with connection.transaction():
                    connection.execute("SET LOCAL session_replication_role=replica")
                    connection.execute(
                        "UPDATE public.brain_schema_versions SET migration_checksum=%s WHERE version=%s",
                        (wrong_checksum, version),
                    )
                    with connection.cursor() as cursor:
                        schema.verify_brain_schema_v8_in_transaction(cursor)
            connection.rollback()
        connection.commit()
        schema.verify_brain_schema_v8(connection)


def test_v8_expired_lease_is_nonauthoritative_and_lease_beyond_retention_is_rejected(brain_pg) -> None:
    artifact_hash = "3" * 64
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v8_authority(connection)
        location = _register_v8_artifact(
            connection,
            request_id="83333333-3333-3333-3333-333333333333",
            digest="1" * 64,
            artifact_hash=artifact_hash,
        )
        loc_verified = _v8_location_verified_at(connection, artifact_hash, location["object_key"])
        window = connection.execute(
            "SELECT (clock_timestamp()+interval '1 hour') AS retain_until,"
            "(clock_timestamp()+interval '2 hours') AS beyond_retention"
        ).fetchone()
        connection.commit()

        connection.execute("SET ROLE brain_artifact_authority_writer")
        # (b) A lease that outlives its COMPLIANCE retention deadline is rejected outright.
        with pytest.raises(psycopg.errors.CheckViolation, match="lease_within_retention"):
            with connection.transaction():
                connection.execute(
                    _V8_ATTEST_SQL,
                    _v8_attest_args(
                        location, artifact_hash=artifact_hash,
                        retain_until=window["retain_until"], verified_at=loc_verified,
                        lease_id="lease-over", lease_expires=window["beyond_retention"],
                    ),
                )
        # (a) A lease already expired while retention is still in the future is
        #     insertable but NON-authoritative.
        connection.execute(
            _V8_ATTEST_SQL,
            _v8_attest_args(
                location, artifact_hash=artifact_hash,
                retain_until=window["retain_until"], verified_at=loc_verified,
                lease_id="lease-dead", lease_expires=loc_verified,
            ),
        )
        connection.execute("RESET ROLE")
        connection.commit()
        assert connection.execute(
            "SELECT brain_artifact_is_authoritative(%s) AS a", (artifact_hash,)
        ).fetchone()["a"] is False

        # A distinct attestation whose lease is live and retention is future is authoritative.
        live = connection.execute(
            "SELECT (clock_timestamp()+interval '3 hours') AS retain_until,"
            "(clock_timestamp()+interval '90 minutes') AS lease_expires"
        ).fetchone()
        connection.commit()
        connection.execute("SET ROLE brain_artifact_authority_writer")
        connection.execute(
            _V8_ATTEST_SQL,
            _v8_attest_args(
                location, artifact_hash=artifact_hash,
                retain_until=live["retain_until"], verified_at=loc_verified,
                lease_id="lease-live", lease_expires=live["lease_expires"],
            ),
        )
        connection.execute("RESET ROLE")
        connection.commit()
        assert connection.execute(
            "SELECT brain_artifact_is_authoritative(%s) AS a", (artifact_hash,)
        ).fetchone()["a"] is True
        connection.commit()
        schema.verify_brain_schema_v8(connection)


def test_v8_future_and_unbound_verified_at_are_rejected(brain_pg) -> None:
    artifact_hash = "4" * 64
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v8_authority(connection)
        location = _register_v8_artifact(
            connection,
            request_id="84444444-4444-4444-4444-444444444444",
            digest="1" * 64,
            artifact_hash=artifact_hash,
        )
        loc_verified = _v8_location_verified_at(connection, artifact_hash, location["object_key"])
        times = connection.execute(
            "SELECT (clock_timestamp()+interval '1 hour') AS retain_until,"
            "(clock_timestamp()+interval '1 hour') AS future_verified,"
            "(%s::timestamptz - interval '1 second') AS before_observation",
            (loc_verified,),
        ).fetchone()
        connection.commit()
        connection.execute("SET ROLE brain_artifact_authority_writer")
        # Future-dated verified_at is rejected.
        with pytest.raises(psycopg.errors.CheckViolation, match="verified_at cannot be in the future"):
            with connection.transaction():
                connection.execute(
                    _V8_ATTEST_SQL,
                    _v8_attest_args(
                        location, artifact_hash=artifact_hash,
                        retain_until=times["retain_until"], verified_at=times["future_verified"],
                        lease_id="lease-fut", lease_expires=times["retain_until"],
                    ),
                )
        # verified_at unbound from (predating) the location observation is rejected.
        with pytest.raises(psycopg.errors.CheckViolation, match="verified registered artifact location"):
            with connection.transaction():
                connection.execute(
                    _V8_ATTEST_SQL,
                    _v8_attest_args(
                        location, artifact_hash=artifact_hash,
                        retain_until=times["retain_until"], verified_at=times["before_observation"],
                        lease_id="lease-unbound", lease_expires=times["retain_until"],
                    ),
                )
        connection.execute("RESET ROLE")
        connection.commit()
        connection.commit()
        schema.verify_brain_schema_v8(connection)


def test_v8_retention_replay_is_exact_and_same_identity_conflicts_fail_closed(brain_pg) -> None:
    artifact_hash = "5" * 64
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v8_authority(connection)
        location = _register_v8_artifact(
            connection,
            request_id="85555555-5555-5555-5555-555555555555",
            digest="1" * 64,
            artifact_hash=artifact_hash,
        )
        loc_verified = _v8_location_verified_at(connection, artifact_hash, location["object_key"])
        retain_until = connection.execute(
            "SELECT (clock_timestamp()+interval '1 hour') AS r"
        ).fetchone()["r"]
        later_verified = connection.execute(
            "SELECT (clock_timestamp()-interval '1 millisecond') AS v"
        ).fetchone()["v"]
        connection.commit()

        connection.execute("SET ROLE brain_artifact_authority_writer")
        first_id = connection.execute(
            _V8_ATTEST_SQL,
            _v8_attest_args(
                location, artifact_hash=artifact_hash, retain_until=retain_until,
                verified_at=loc_verified, lease_id="lease-a", lease_expires=retain_until,
            ),
        ).fetchone()["id"]
        # Exact payload replay returns the same durable row.
        assert connection.execute(
            _V8_ATTEST_SQL,
            _v8_attest_args(
                location, artifact_hash=artifact_hash, retain_until=retain_until,
                verified_at=loc_verified, lease_id="lease-a", lease_expires=retain_until,
            ),
        ).fetchone()["id"] == first_id

        # Same retention identity, different observed payload -> fail closed.
        for verified_at, lease_id, lease_expires in (
            (loc_verified, "lease-b", retain_until),
            (later_verified, "lease-a", retain_until),
        ):
            with pytest.raises(psycopg.errors.UniqueViolation, match="conflicting retention attestation replay"):
                with connection.transaction():
                    connection.execute(
                        _V8_ATTEST_SQL,
                        _v8_attest_args(
                            location, artifact_hash=artifact_hash, retain_until=retain_until,
                            verified_at=verified_at, lease_id=lease_id, lease_expires=lease_expires,
                        ),
                    )
        connection.execute("RESET ROLE")
        connection.commit()
        assert connection.execute(
            "SELECT count(*) AS c FROM public.brain_artifact_retention_attestations WHERE artifact_hash=%s",
            (artifact_hash,),
        ).fetchone()["c"] == 1
        connection.commit()
        schema.verify_brain_schema_v8(connection)


def test_v8_old_incarnation_row_never_aliases_or_authorizes(brain_pg) -> None:
    artifact_hash = "6" * 64
    stale_incarnation = "00000000-0000-0000-0000-0000000000aa"
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v8_authority(connection)
        location = _register_v8_artifact(
            connection,
            request_id="86666666-6666-6666-6666-666666666666",
            digest="1" * 64,
            artifact_hash=artifact_hash,
        )
        loc_verified = _v8_location_verified_at(connection, artifact_hash, location["object_key"])
        current = _v8_current_incarnation(connection)
        assert stale_incarnation != current
        retain_until = connection.execute(
            "SELECT (clock_timestamp()+interval '1 hour') AS r"
        ).fetchone()["r"]
        connection.commit()

        # Forge an old-incarnation attestation (bypassing the incarnation-guard trigger)
        # with an otherwise-live lease and future retention.
        connection.execute("SET session_replication_role=replica")
        connection.execute(
            "INSERT INTO public.brain_artifact_retention_attestations (" + _V8_ATTESTATION_COLUMNS + ") "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,'COMPLIANCE',%s::timestamptz,%s::timestamptz,%s,%s::timestamptz)",
            (
                stale_incarnation, artifact_hash, location["backend"], location["bucket_or_container"],
                location["object_key"], location["provider_version_id"], location["provider_checksum"],
                retain_until, loc_verified, "stale-lease", retain_until,
            ),
        )
        connection.execute("SET session_replication_role=origin")
        connection.commit()
        stale_id = connection.execute(
            "SELECT retention_attestation_id AS id FROM public.brain_artifact_retention_attestations "
            "WHERE database_incarnation_id=%s",
            (stale_incarnation,),
        ).fetchone()["id"]

        # Old-incarnation data never confers authority.
        assert connection.execute(
            "SELECT brain_artifact_is_authoritative(%s) AS a", (artifact_hash,)
        ).fetchone()["a"] is False

        # Re-attesting the same logical identity under the current incarnation must NOT
        # alias the stale row -- it appends a fresh current-incarnation row.
        connection.execute("SET ROLE brain_artifact_authority_writer")
        current_id = connection.execute(
            _V8_ATTEST_SQL,
            _v8_attest_args(
                location, artifact_hash=artifact_hash, retain_until=retain_until,
                verified_at=loc_verified, lease_id="current-lease", lease_expires=retain_until,
            ),
        ).fetchone()["id"]
        connection.execute("RESET ROLE")
        connection.commit()
        assert current_id != stale_id
        assert connection.execute(
            "SELECT count(*) AS c FROM public.brain_artifact_retention_attestations WHERE artifact_hash=%s",
            (artifact_hash,),
        ).fetchone()["c"] == 2
        # Now the current-incarnation attestation authorizes.
        assert connection.execute(
            "SELECT brain_artifact_is_authoritative(%s) AS a", (artifact_hash,)
        ).fetchone()["a"] is True
        connection.commit()
        schema.verify_brain_schema_v8(connection)


def test_v8_authority_requires_the_exact_attested_location(brain_pg) -> None:
    artifact_hash = "7" * 64
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v8_authority(connection)
        location = _register_v8_artifact(
            connection,
            request_id="87777777-7777-7777-7777-777777777777",
            digest="1" * 64,
            artifact_hash=artifact_hash,
        )
        loc_verified = _v8_location_verified_at(connection, artifact_hash, location["object_key"])
        current = _v8_current_incarnation(connection)
        retain_until = connection.execute(
            "SELECT (clock_timestamp()+interval '1 hour') AS r"
        ).fetchone()["r"]
        connection.commit()

        # Forge an attestation that names a DIFFERENT object key than the sole verified
        # location for this hash holds. Pre-repair the predicate joined location by hash
        # alone and would have returned authoritative from that unrelated location.
        connection.execute("SET session_replication_role=replica")
        connection.execute(
            "INSERT INTO public.brain_artifact_retention_attestations (" + _V8_ATTESTATION_COLUMNS + ") "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,'COMPLIANCE',%s::timestamptz,%s::timestamptz,%s,%s::timestamptz)",
            (
                current, artifact_hash, location["backend"], location["bucket_or_container"],
                "sha256/" + artifact_hash + "-ghost", location["provider_version_id"],
                location["provider_checksum"], retain_until, loc_verified, "ghost-lease", retain_until,
            ),
        )
        connection.execute("SET session_replication_role=origin")
        connection.commit()
        # An unrelated verified location cannot satisfy authority for the named object.
        assert connection.execute(
            "SELECT brain_artifact_is_authoritative(%s) AS a", (artifact_hash,)
        ).fetchone()["a"] is False

        # An attestation bound to the exact verified location IS authoritative.
        connection.execute("SET ROLE brain_artifact_authority_writer")
        connection.execute(
            _V8_ATTEST_SQL,
            _v8_attest_args(
                location, artifact_hash=artifact_hash, retain_until=retain_until,
                verified_at=loc_verified, lease_id="exact-lease", lease_expires=retain_until,
            ),
        )
        connection.execute("RESET ROLE")
        connection.commit()
        assert connection.execute(
            "SELECT brain_artifact_is_authoritative(%s) AS a", (artifact_hash,)
        ).fetchone()["a"] is True
        connection.commit()
        schema.verify_brain_schema_v8(connection)


def test_v8_null_bucket_attestation_dedupes_and_authorizes(brain_pg) -> None:
    artifact_hash = "a" * 64
    with psycopg.connect(brain_pg, row_factory=dict_row) as connection:
        _install_v8_authority(connection)
        _register_v8_artifact(
            connection,
            request_id="8aaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            digest="1" * 64,
            artifact_hash=artifact_hash,
        )
        # Add a second verified location for the same hash with a NULL bucket_or_container.
        null_location = {
            "backend": "filesystem",
            "bucket_or_container": None,
            "object_key": "local/" + artifact_hash,
            "provider_version_id": "version-" + artifact_hash[:8],
            "provider_checksum": "checksum-" + artifact_hash[:8],
        }
        connection.execute(
            "INSERT INTO public.brain_artifact_locations "
            "(artifact_hash,backend,bucket_or_container,object_key,provider_version_id,"
            "provider_checksum,storage_immutable,encryption_mode,encryption_key_id,durability,verified_at) "
            "VALUES (%s,%s,NULL,%s,%s,%s,true,'customer_managed','kms-key-1','verified',clock_timestamp())",
            (
                artifact_hash, null_location["backend"], null_location["object_key"],
                null_location["provider_version_id"], null_location["provider_checksum"],
            ),
        )
        connection.commit()
        loc_verified = _v8_location_verified_at(connection, artifact_hash, null_location["object_key"])
        retain_until = connection.execute(
            "SELECT (clock_timestamp()+interval '1 hour') AS r"
        ).fetchone()["r"]
        connection.commit()

        connection.execute("SET ROLE brain_artifact_authority_writer")
        args = _v8_attest_args(
            null_location, artifact_hash=artifact_hash, retain_until=retain_until,
            verified_at=loc_verified, lease_id="null-lease", lease_expires=retain_until,
        )
        first_id = connection.execute(_V8_ATTEST_SQL, args).fetchone()["id"]
        # NULLS NOT DISTINCT + IS NOT DISTINCT FROM: a null-bucket replay dedupes.
        assert connection.execute(_V8_ATTEST_SQL, args).fetchone()["id"] == first_id
        connection.execute("RESET ROLE")
        connection.commit()
        assert connection.execute(
            "SELECT count(*) AS c FROM public.brain_artifact_retention_attestations "
            "WHERE artifact_hash=%s AND bucket_or_container IS NULL",
            (artifact_hash,),
        ).fetchone()["c"] == 1
        # The predicate resolves the null-bucket location join correctly.
        assert connection.execute(
            "SELECT brain_artifact_is_authoritative(%s) AS a", (artifact_hash,)
        ).fetchone()["a"] is True
        connection.commit()
        schema.verify_brain_schema_v8(connection)


def test_v8_concurrent_identical_attestation_is_idempotent(brain_pg) -> None:
    artifact_hash = "b" * 64
    with psycopg.connect(brain_pg, row_factory=dict_row) as setup:
        _install_v8_authority(setup)
        location = _register_v8_artifact(
            setup,
            request_id="8bbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            digest="1" * 64,
            artifact_hash=artifact_hash,
        )
        loc_verified = _v8_location_verified_at(setup, artifact_hash, location["object_key"])
        retain_until = setup.execute(
            "SELECT (clock_timestamp()+interval '1 hour') AS r"
        ).fetchone()["r"]
        setup.commit()
    args = _v8_attest_args(
        location, artifact_hash=artifact_hash, retain_until=retain_until,
        verified_at=loc_verified, lease_id="concurrent-lease", lease_expires=retain_until,
    )

    barrier = Barrier(2)

    def attest():
        with psycopg.connect(brain_pg, row_factory=dict_row) as contender:
            contender.execute("SET ROLE brain_artifact_authority_writer")
            barrier.wait()
            attest_id = contender.execute(_V8_ATTEST_SQL, args).fetchone()["id"]
            contender.commit()
            return attest_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = [future.result() for future in (pool.submit(attest), pool.submit(attest))]
    assert ids[0] == ids[1]
    with psycopg.connect(brain_pg, row_factory=dict_row) as verifier:
        assert verifier.execute(
            "SELECT count(*) AS c FROM public.brain_artifact_retention_attestations WHERE artifact_hash=%s",
            (artifact_hash,),
        ).fetchone()["c"] == 1
        verifier.commit()
        schema.verify_brain_schema_v8(verifier)


def test_v9_combined_operator_path_renews_and_persists_verifiable_evidence(brain_pg) -> None:
    key = b"v9-retention-signing-key-material" * 2
    content = b"content"
    artifact_hash = hashlib.sha256(content).hexdigest()
    provider_checksum = base64.b64encode(hashlib.sha256(content).digest()).decode()

    class S3:
        def __init__(self, retain_until: datetime) -> None:
            self.retain_until = retain_until
            self.head_count = 0
            self.get_count = 0

        def head_object(self, **_kwargs):
            self.head_count += 1
            return {
                "VersionId": "version-v9",
                "ContentLength": len(content),
                "ContentType": "application/json",
                "ChecksumSHA256": provider_checksum,
                "ETag": '"etag-v9"',
                "ObjectLockMode": "COMPLIANCE",
                "ObjectLockRetainUntilDate": self.retain_until,
                "ServerSideEncryption": "aws:kms",
                "SSEKMSKeyId": "kms-key-v9",
            }

        def get_object(self, **_kwargs):
            self.get_count += 1
            return {
                "Body": io.BytesIO(content),
                "ChecksumSHA256": provider_checksum,
                "ContentLength": len(content),
                "VersionId": "version-v9",
            }

    with psycopg.connect(brain_pg, row_factory=dict_row) as owner:
        _install_v8_authority(owner)
        _append_v9(owner)
        identity = owner.execute(
            "SELECT (pg_control_system()).system_identifier::text AS system_id,"
            "current_database() AS database_name"
        ).fetchone()
        assert owner.execute(
            "SELECT brain_artifact_is_authoritative(%s) AS authoritative", (artifact_hash,)
        ).fetchone()["authoritative"] is False
        owner.commit()

    issued = datetime.now(timezone.utc)
    manifest = {
        "schemaVersion": 1,
        "purpose": ARTIFACT_AUTHORITY_PURPOSE,
        "requestId": "99999999-9999-9999-9999-999999999999",
        "issuedAt": issued.isoformat().replace("+00:00", "Z"),
        "expiresAt": (issued + timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
        "destination": {
            "systemId": identity["system_id"],
            "databaseName": identity["database_name"],
        },
        "artifacts": [{
            "ordinal": 1,
            "artifactSha256": artifact_hash,
            "byteCount": len(content),
            "mediaType": "application/json",
            "backend": "s3",
            "bucket": "immutable",
            "objectKey": "sha256/v9-content",
            "versionId": "version-v9",
            "etag": "etag-v9",
            "awsChecksumSha256": provider_checksum,
            "storageImmutable": True,
            "encryptionMode": "customer_managed",
            "encryptionKeyId": "kms-key-v9",
            "policySourceId": "policy-v9",
        }],
    }
    canonical = canonical_manifest_bytes(manifest)
    envelope = {
        "algorithm": "hmac-sha256",
        "keyId": "key-v9",
        "manifestSha256": hashlib.sha256(canonical).hexdigest(),
        "signature": base64.urlsafe_b64encode(
            hmac.new(key, signing_message(canonical), hashlib.sha256).digest()
        ).rstrip(b"=").decode(),
    }
    s3 = S3(issued + timedelta(days=30))

    def connection_factory():
        connection = psycopg.connect(brain_pg, row_factory=dict_row)
        connection.execute("SET ROLE brain_artifact_authority_writer")
        connection.commit()
        return connection

    first = coordinate_artifact_authority(
        manifest,
        envelope,
        keys={"key-v9": key},
        expected_system_id=identity["system_id"],
        expected_database_name=identity["database_name"],
        s3_client=s3,
        connection_factory=connection_factory,
        now=issued,
        lease_duration=timedelta(microseconds=1),
    )
    assert first["authoritative"] is False

    renewed = coordinate_artifact_authority(
        manifest,
        envelope,
        keys={"key-v9": key},
        expected_system_id=identity["system_id"],
        expected_database_name=identity["database_name"],
        s3_client=s3,
        connection_factory=connection_factory,
        now=issued,
    )
    assert renewed["authoritative"] is True
    assert s3.head_count == 4 and s3.get_count == 4

    with psycopg.connect(brain_pg, row_factory=dict_row) as audit:
        rows = audit.execute(
            "SELECT attestation.*,location.verified_at AS location_verified_at "
            "FROM public.brain_artifact_retention_attestations attestation "
            "JOIN public.brain_artifact_locations location USING (artifact_hash) "
            "WHERE attestation.artifact_hash=%s ORDER BY retention_attestation_id",
            (artifact_hash,),
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["retain_until"] == rows[1]["retain_until"]
        assert rows[0]["payload_sha256"] != rows[1]["payload_sha256"]
        row = rows[-1]
        persisted = bytes(row["canonical_payload"])
        assert row["verified_at"] >= row["location_verified_at"]
        assert hashlib.sha256(persisted).hexdigest() == row["payload_sha256"]
        assert hmac.compare_digest(
            base64.urlsafe_b64encode(
                hmac.new(key, retention_signing_message(persisted), hashlib.sha256).digest()
            ).rstrip(b"=").decode(),
            row["signature"],
        )
        attestation = RetentionAttestation(
            artifact_hash=row["artifact_hash"],
            backend=row["backend"],
            bucket_or_container=row["bucket_or_container"],
            object_key=row["object_key"],
            provider_version_id=row["provider_version_id"],
            provider_checksum=row["provider_checksum"],
            retention_until=row["retain_until"],
            verified_at=row["verified_at"],
            verification_lease_id=row["verification_lease_id"],
            verification_expires_at=row["verification_lease_expires_at"],
            canonical_bytes=persisted,
            payload_sha256=row["payload_sha256"],
            key_id=row["signing_key_id"],
            signature=row["signature"],
        )
        validate_retention_attestation(
            attestation, now=datetime.now(timezone.utc), keys={"key-v9": key}
        )
        audit.commit()
        attest_sql = (
            "SELECT brain_attest_artifact_retention("
            "%s,%s,%s,%s,%s,%s,%s::timestamptz,%s::timestamptz,%s,%s::timestamptz,"
            "%s::bytea,%s,%s,%s) AS id"
        )
        persisted_args = (
            row["artifact_hash"], row["backend"], row["bucket_or_container"], row["object_key"],
            row["provider_version_id"], row["provider_checksum"], row["retain_until"],
            row["verified_at"], row["verification_lease_id"],
            row["verification_lease_expires_at"], persisted, row["payload_sha256"],
            row["signing_key_id"], row["signature"],
        )
        audit.execute("SET ROLE brain_artifact_authority_writer")
        assert audit.execute(attest_sql, persisted_args).fetchone()["id"] == row[
            "retention_attestation_id"
        ]
        with pytest.raises(psycopg.errors.UniqueViolation, match="conflicting signed"):
            with audit.transaction():
                audit.execute(attest_sql, (*persisted_args[:-1], "A" * 43))
        audit.execute("RESET ROLE")
        audit.commit()
        schema.verify_brain_schema_v9(audit)
