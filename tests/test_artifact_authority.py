from __future__ import annotations

import json
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import io

import pytest

from applypilot.brain.artifact_authority import (
    ArtifactAuthorityError,
    DEFAULT_VERIFICATION_LEASE,
    RETENTION_PURPOSE,
    VerifiedManifest,
    canonical_manifest_bytes,
    coordinate_artifact_authority,
    coordinate_artifact_registration,
    coordinate_retention_attestation,
    normalize_manifest,
    parse_json_strict,
    register_verified_manifest,
    retention_signing_message,
    seal_retention_attestations,
    signing_message,
    verify_manifest_signature,
    verify_s3_versions,
)


KEY = b"s" * 32
_NOW = datetime(2026, 7, 18, 12, 30, tzinfo=timezone.utc)


def _content_checksum() -> str:
    return base64.b64encode(hashlib.sha256(b"content").digest()).decode()


def _signed_manifest() -> tuple[dict, dict]:
    manifest = _manifest()
    manifest["artifacts"][0]["artifactSha256"] = hashlib.sha256(b"content").hexdigest()
    canonical = canonical_manifest_bytes(manifest)
    envelope = {
        "algorithm": "hmac-sha256",
        "keyId": "key-1",
        "manifestSha256": hashlib.sha256(canonical).hexdigest(),
        "signature": base64.urlsafe_b64encode(
            hmac.new(KEY, signing_message(canonical), hashlib.sha256).digest()
        ).rstrip(b"=").decode(),
    }
    return manifest, envelope


def _verified_with_s3(retain_until: datetime | None = None) -> tuple[VerifiedManifest, tuple[datetime, ...]]:
    manifest, envelope = _signed_manifest()
    verified = verify_manifest_signature(
        manifest,
        envelope,
        keys={"key-1": KEY},
        expected_system_id="42",
        expected_database_name="brain",
        now=_NOW,
    )
    s3 = _S3() if retain_until is None else _S3(retain_until=retain_until)
    verified, retentions = verify_s3_versions(verified, s3, clock=lambda: _NOW)
    return verified, retentions


def _manifest() -> dict:
    return {
        "schemaVersion": 1,
        "purpose": "brain-artifact-authority-registration-v1",
        "requestId": "11111111-1111-1111-1111-111111111111",
        "issuedAt": "2026-07-18T12:00:00Z",
        "expiresAt": "2026-07-18T13:00:00Z",
        "destination": {"systemId": "42", "databaseName": "brain"},
        "artifacts": [
            {
                "ordinal": 1,
                "artifactSha256": "a" * 64,
                "byteCount": 7,
                "mediaType": "application/json",
                "backend": "aws_s3",
                "bucket": "immutable",
                "objectKey": "sha256/aa",
                "versionId": "version-1",
                "etag": "etag-1",
                "awsChecksumSha256": base64.b64encode(hashlib.sha256(b"content").digest()).decode(),
                "storageImmutable": True,
                "encryptionMode": "customer_managed",
                "encryptionKeyId": "kms-key-1",
                "policySourceId": "policy-1",
            }
        ],
    }


class _Cursor:
    def __init__(self, receipt=None) -> None:
        self.params = None
        self.receipt = receipt or {
            "status": "registered",
            "request_id": "11111111-1111-1111-1111-111111111111",
            "manifest_sha256": "b" * 64,
            "destination_system_id": "42",
            "destination_database_name": "brain",
            "artifact_count": 1,
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, _sql, params) -> None:
        self.params = params

    def fetchone(self):
        return (self.receipt,)


class _Connection:
    def __init__(self) -> None:
        self.cursor_value = _Cursor()
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self.cursor_value

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


def test_registration_translates_signed_manifest_to_fixed_database_contract() -> None:
    manifest = normalize_manifest(_manifest())
    connection = _Connection()
    receipt = register_verified_manifest(connection, VerifiedManifest(manifest, "b" * 64, "key-1"))
    assert receipt["status"] == "registered"
    assert connection.committed is True
    payload = json.loads(connection.cursor_value.params[-1])
    assert payload == [
        {
            "artifact_hash": "a" * 64,
            "backend": "s3",
            "bucket": "immutable",
            "byte_length": 7,
            "encryption_key_id": "kms-key-1",
            "encryption_mode": "customer_managed",
            "media_type": "application/json",
            "object_key": "sha256/aa",
            "policy_source_id": "policy-1",
            "provider_checksum": base64.b64encode(hashlib.sha256(b"content").digest()).decode(),
            "provider_version_id": "version-1",
            "storage_immutable": True,
        }
    ]


@pytest.mark.parametrize("ordinals", ([0], [2], [1, 3]))
def test_manifest_rejects_non_positive_or_gapped_ordinals(ordinals: list[int]) -> None:
    manifest = _manifest()
    manifest["artifacts"] = [dict(manifest["artifacts"][0], ordinal=value, artifactSha256=f"{value + 1:064x}") for value in ordinals]
    with pytest.raises(ArtifactAuthorityError, match="ordinal"):
        normalize_manifest(manifest)


def test_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(ArtifactAuthorityError, match="duplicate JSON key"):
        parse_json_strict('{"schemaVersion":1,"schemaVersion":1}')


def test_signature_binds_canonical_manifest_time_and_destination() -> None:
    manifest = _manifest()
    canonical = canonical_manifest_bytes(manifest)
    digest = hashlib.sha256(canonical).hexdigest()
    key = b"s" * 32
    signature = base64.urlsafe_b64encode(
        hmac.new(key, signing_message(canonical), hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    verified = verify_manifest_signature(
        manifest,
        {"algorithm": "hmac-sha256", "keyId": "key-1", "manifestSha256": digest, "signature": signature},
        keys={"key-1": key},
        expected_system_id="42",
        expected_database_name="brain",
        now=datetime(2026, 7, 18, 12, 30, tzinfo=timezone.utc),
    )
    assert verified.manifest_sha256 == digest
    with pytest.raises(ArtifactAuthorityError, match="destination system mismatch"):
        verify_manifest_signature(
            manifest,
            {"algorithm": "hmac-sha256", "keyId": "key-1", "manifestSha256": digest, "signature": signature},
            keys={"key-1": key},
            expected_system_id="wrong",
            expected_database_name="brain",
            now=datetime(2026, 7, 18, 12, 30, tzinfo=timezone.utc),
        )


class _S3:
    def __init__(self, retain_until: datetime | None = None) -> None:
        self.retain_until = retain_until or datetime(2099, 1, 1, tzinfo=timezone.utc)

    def head_object(self, **_kwargs):
        return {
            "VersionId": "version-1",
            "ContentLength": 7,
            "ContentType": "application/json",
            "ChecksumSHA256": base64.b64encode(hashlib.sha256(b"content").digest()).decode(),
            "ETag": '"etag-1"',
            "ObjectLockMode": "COMPLIANCE",
            "ObjectLockRetainUntilDate": self.retain_until,
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": "kms-key-1",
        }

    def get_object(self, **_kwargs):
        return {
            "Body": io.BytesIO(b"content"),
            "ChecksumSHA256": base64.b64encode(hashlib.sha256(b"content").digest()).decode(),
            "ContentLength": 7,
            "VersionId": "version-1",
        }


def test_s3_verification_binds_exact_version_content_lock_and_encryption() -> None:
    manifest = normalize_manifest(_manifest())
    manifest["artifacts"][0]["artifactSha256"] = hashlib.sha256(b"content").hexdigest()
    verified = VerifiedManifest(manifest, "b" * 64, "key-1")
    result, retentions = verify_s3_versions(verified, _S3())
    assert result is verified
    assert retentions == (datetime(2099, 1, 1, tzinfo=timezone.utc),)


def test_s3_verification_rejects_missing_compliance_lock() -> None:
    class Unlocked(_S3):
        def head_object(self, **kwargs):
            result = super().head_object(**kwargs)
            result["ObjectLockMode"] = "GOVERNANCE"
            return result

    manifest = normalize_manifest(_manifest())
    manifest["artifacts"][0]["artifactSha256"] = hashlib.sha256(b"content").hexdigest()
    with pytest.raises(ArtifactAuthorityError, match="COMPLIANCE"):
        verify_s3_versions(VerifiedManifest(manifest, "b" * 64, "key-1"), Unlocked())


def test_invalid_signature_encoding_is_rejected_strictly() -> None:
    manifest = _manifest()
    canonical = canonical_manifest_bytes(manifest)
    with pytest.raises(ArtifactAuthorityError, match="signature encoding"):
        verify_manifest_signature(
            manifest,
            {
                "algorithm": "hmac-sha256",
                "keyId": "key-1",
                "manifestSha256": hashlib.sha256(canonical).hexdigest(),
                "signature": "not+url/safe!",
            },
            keys={"key-1": b"s" * 32},
            expected_system_id="42",
            expected_database_name="brain",
            now=datetime(2026, 7, 18, 12, 30, tzinfo=timezone.utc),
        )


def test_dry_run_verifies_s3_without_opening_database() -> None:
    manifest = _manifest()
    manifest["artifacts"][0]["artifactSha256"] = hashlib.sha256(b"content").hexdigest()
    canonical = canonical_manifest_bytes(manifest)
    key = b"s" * 32
    envelope = {
        "algorithm": "hmac-sha256",
        "keyId": "key-1",
        "manifestSha256": hashlib.sha256(canonical).hexdigest(),
        "signature": base64.urlsafe_b64encode(
            hmac.new(key, signing_message(canonical), hashlib.sha256).digest()
        ).rstrip(b"=").decode(),
    }
    result = coordinate_artifact_registration(
        manifest,
        envelope,
        keys={"key-1": key},
        expected_system_id="42",
        expected_database_name="brain",
        s3_client=_S3(),
        connection_factory=lambda: pytest.fail("dry-run opened a database connection"),
        dry_run=True,
        now=datetime(2026, 7, 18, 12, 30, tzinfo=timezone.utc),
    )
    assert result["dry_run"] is True
    assert result["authoritative"] is False
    assert result["promotable"] is False


# ---------------------------------------------------------------------------
# Packet A: durable retention lease binding through the V8 attestation contract
# ---------------------------------------------------------------------------


class _FakeAuthorityDB:
    """Idempotent append-only model of the V8 registration + retention contract."""

    def __init__(self) -> None:
        self.registrations: dict[str, dict] = {}
        self.registration_params: list[tuple] = []
        self.attestations: dict[tuple, int] = {}
        self.attest_params: list[tuple] = []
        self.executed: list[str] = []
        self._next_id = 5000
        self.attest_commit_failures = 0
        self.register_commit_failures = 0
        self.authority_override: bool | None = None
        self._recorded_payload: dict = {}

    def register(self, params: tuple) -> dict:
        self.registration_params.append(params)
        request_id = str(params[0])
        artifacts = json.loads(params[8])
        receipt = {
            "status": "registered",
            "request_id": request_id,
            "manifest_sha256": params[1],
            "destination_system_id": params[6],
            "destination_database_name": params[7],
            "artifact_count": len(artifacts),
        }
        return self.registrations.setdefault(request_id, receipt)

    def attest(self, params: tuple) -> int:
        self.attest_params.append(params)
        # V9 idempotent identity: exact signed evidence digest.
        identity = (params[0], params[11])
        if identity in self.attestations:
            existing = self.attestations[identity]
            # A same-identity replay with a different observed payload fails closed.
            if self._recorded_payload.get(identity) != params:
                raise RuntimeError("conflicting retention attestation replay")
            return existing
        self._next_id += 1
        self.attestations[identity] = self._next_id
        self._recorded_payload[identity] = params
        return self._next_id

    def connection(self):
        return _FakeAuthorityConnection(self)


class _FakeAuthorityCursor:
    def __init__(self, connection: "_FakeAuthorityConnection") -> None:
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, sql: str, params: tuple) -> None:
        flattened = " ".join(sql.split())
        self.connection.db.executed.append(flattened)
        if "brain_register_authoritative_artifact_manifest" in flattened:
            self.connection.last_op = "register"
            self._row = (self.connection.db.register(params),)
        elif "brain_attest_artifact_retention" in flattened:
            self.connection.last_op = "attest"
            self._row = (self.connection.db.attest(params),)
        elif "brain_artifact_is_authoritative" in flattened:
            self.connection.last_op = "query"
            authoritative = (
                self.connection.db.authority_override
                if self.connection.db.authority_override is not None
                else any(identity[0] == params[0] for identity in self.connection.db.attestations)
            )
            self._row = (authoritative,)
        else:
            raise AssertionError(f"unexpected SQL: {flattened}")

    def fetchone(self):
        return self._row


class _FakeAuthorityConnection:
    def __init__(self, db: _FakeAuthorityDB) -> None:
        self.db = db
        self.last_op: str | None = None
        self.committed = 0
        self.rolled_back = 0

    def cursor(self):
        return _FakeAuthorityCursor(self)

    def commit(self) -> None:
        if self.last_op == "register" and self.db.register_commit_failures > 0:
            self.db.register_commit_failures -= 1
            self.last_op = None
            raise RuntimeError("commit disposition lost")
        if self.last_op == "attest" and self.db.attest_commit_failures > 0:
            self.db.attest_commit_failures -= 1
            self.last_op = None
            raise RuntimeError("commit disposition lost")
        self.committed += 1

    def rollback(self) -> None:
        self.rolled_back += 1

    def close(self) -> None:
        return None


def _coordinate(db: _FakeAuthorityDB, retain_until: datetime | None = None, **overrides):
    manifest, envelope = _signed_manifest()
    observation_time = overrides.pop("observation_time", _NOW)
    kwargs = dict(
        keys={"key-1": KEY},
        expected_system_id="42",
        expected_database_name="brain",
        s3_client=_S3() if retain_until is None else _S3(retain_until=retain_until),
        connection_factory=db.connection,
        now=_NOW,
        clock=lambda: observation_time,
    )
    kwargs.update(overrides)
    return coordinate_retention_attestation(manifest, envelope, **kwargs)


def test_retention_evidence_normalizes_provider_instant_to_utc() -> None:
    eastern = timezone(timedelta(hours=-5))
    verified, retentions = _verified_with_s3(datetime(2099, 1, 1, 0, 0, tzinfo=eastern))
    (attestation,) = seal_retention_attestations(verified, retentions, keys={"key-1": KEY})
    assert attestation.retention_until == datetime(2099, 1, 1, 5, 0, tzinfo=timezone.utc)
    payload = json.loads(attestation.canonical_bytes)
    assert payload["retainUntil"] == "2099-01-01T05:00:00Z"
    assert payload["verifiedAt"] == "2026-07-18T12:30:00Z"
    assert payload["purpose"] == RETENTION_PURPOSE


def test_verification_lease_is_bounded_by_provider_retention() -> None:
    # A far-future retention leaves the lease bounded by the configured duration.
    verified, retentions = _verified_with_s3()
    (far,) = seal_retention_attestations(verified, retentions, keys={"key-1": KEY})
    assert far.verification_expires_at == _NOW + DEFAULT_VERIFICATION_LEASE
    assert far.verification_expires_at <= far.retention_until

    # A near retention deadline clamps the lease so it never outlives retention.
    near_deadline = _NOW + timedelta(hours=1)
    verified, retentions = _verified_with_s3(near_deadline)
    (near,) = seal_retention_attestations(verified, retentions, keys={"key-1": KEY})
    assert near.verification_expires_at == near_deadline
    assert near.verification_expires_at <= near.retention_until


def test_seal_rejects_retention_deadline_at_or_before_verification() -> None:
    verified, retentions = _verified_with_s3()
    with pytest.raises(ArtifactAuthorityError, match="retention"):
        seal_retention_attestations(
            verified, (verified.verified_at,), keys={"key-1": KEY}
        )


def test_coordinator_attests_retention_with_exact_v9_contract() -> None:
    db = _FakeAuthorityDB()
    result = _coordinate(db)
    assert result["authoritative"] is True
    assert result["retention_attestation_ids"] == [5001]
    # The coordinator calls exactly the V9 retention authority function, once.
    assert sum("brain_attest_artifact_retention" in sql for sql in db.executed) == 1
    assert sum("brain_register" in sql for sql in db.executed) == 0
    (params,) = db.attest_params
    assert params == (
        hashlib.sha256(b"content").hexdigest(),
        "s3",
        "immutable",
        "sha256/aa",
        "version-1",
        _content_checksum(),
        "2099-01-01T00:00:00Z",
        "2026-07-18T12:30:00Z",
        params[8],
        "2026-07-19T12:30:00Z",
        params[10],
        hashlib.sha256(params[10]).hexdigest(),
        "key-1",
        params[13],
    )
    # The lease id is a bound, non-empty token derived from the observation.
    assert isinstance(params[8], str) and params[8].strip()


def test_retention_payload_is_immutable_signed_and_deterministic() -> None:
    verified_a, retentions_a = _verified_with_s3()
    verified_b, retentions_b = _verified_with_s3()
    (first,) = seal_retention_attestations(verified_a, retentions_a, keys={"key-1": KEY})
    (second,) = seal_retention_attestations(verified_b, retentions_b, keys={"key-1": KEY})
    # Deterministic canonical payload and signature across independent runs.
    assert first.canonical_bytes == second.canonical_bytes
    assert first.signature == second.signature
    assert first.verification_lease_id == second.verification_lease_id
    # The signature verifies under the bound key and domain separation.
    expected = base64.urlsafe_b64encode(
        hmac.new(KEY, retention_signing_message(first.canonical_bytes), hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    assert first.signature == expected
    validate = __import__(
        "applypilot.brain.artifact_authority", fromlist=["validate_retention_attestation"]
    ).validate_retention_attestation
    validate(first, now=_NOW, keys={"key-1": KEY})


def test_canonical_signature_and_hash_drift_is_rejected() -> None:
    import dataclasses

    from applypilot.brain.artifact_authority import validate_retention_attestation

    verified, retentions = _verified_with_s3()
    (attestation,) = seal_retention_attestations(verified, retentions, keys={"key-1": KEY})
    # Rebinding a field without re-signing is caught as payload drift.
    drifted = dataclasses.replace(attestation, provider_version_id="version-2")
    with pytest.raises(ArtifactAuthorityError, match="drift"):
        validate_retention_attestation(drifted, now=_NOW, keys={"key-1": KEY})
    # A tampered signature is caught.
    forged = dataclasses.replace(attestation, signature="A" * 43)
    with pytest.raises(ArtifactAuthorityError, match="signature"):
        validate_retention_attestation(forged, now=_NOW, keys={"key-1": KEY})


def test_expired_verification_and_lease_beyond_retention_are_rejected() -> None:
    from applypilot.brain.artifact_authority import (
        build_retention_attestation,
        validate_retention_attestation,
    )

    common = dict(
        request_id="11111111-1111-1111-1111-111111111111",
        artifact_hash=hashlib.sha256(b"content").hexdigest(),
        backend="s3",
        bucket_or_container="immutable",
        object_key="sha256/aa",
        provider_version_id="version-1",
        provider_checksum=_content_checksum(),
        verified_at=_NOW,
        verification_lease_id="lease-1",
        keys={"key-1": KEY},
        key_id="key-1",
    )
    # A lease that outlives the provider retention deadline is rejected.
    beyond = build_retention_attestation(
        retention_until=_NOW + timedelta(hours=1),
        verification_expires_at=_NOW + timedelta(hours=2),
        **common,
    )
    with pytest.raises(ArtifactAuthorityError, match="retention"):
        validate_retention_attestation(beyond, now=_NOW, keys={"key-1": KEY})
    # A lease that has already expired at evaluation time is rejected.
    expired = build_retention_attestation(
        retention_until=_NOW + timedelta(days=30),
        verification_expires_at=_NOW + timedelta(hours=1),
        **common,
    )
    with pytest.raises(ArtifactAuthorityError, match="expired"):
        validate_retention_attestation(
            expired, now=_NOW + timedelta(hours=2), keys={"key-1": KEY}
        )


def test_missing_retention_evidence_never_reaches_the_attestation() -> None:
    class Unlocked(_S3):
        def head_object(self, **kwargs):
            result = super().head_object(**kwargs)
            result.pop("ObjectLockRetainUntilDate")
            return result

    db = _FakeAuthorityDB()
    manifest, envelope = _signed_manifest()
    with pytest.raises(ArtifactAuthorityError, match="retention"):
        coordinate_retention_attestation(
            manifest,
            envelope,
            keys={"key-1": KEY},
            expected_system_id="42",
            expected_database_name="brain",
            s3_client=Unlocked(),
            connection_factory=db.connection,
            now=_NOW,
        )
    assert db.attest_params == []
    assert db.attestations == {}


def test_retention_attestation_retry_is_idempotent() -> None:
    db = _FakeAuthorityDB()
    db.attest_commit_failures = 1
    result = _coordinate(db)
    # The ambiguous commit was retried and converged to the same durable id.
    assert result["retention_attestation_ids"] == [5001]
    assert len(db.attest_params) == 2
    assert db.attest_params[0] == db.attest_params[1]
    assert len(db.attestations) == 1


def test_provider_extension_appends_a_new_attestation() -> None:
    db = _FakeAuthorityDB()
    _coordinate(db)
    extended = _coordinate(db, retain_until=datetime(2100, 6, 1, tzinfo=timezone.utc))
    # A later provider retention creates a second append-only attestation, never an update.
    assert extended["retention_attestation_ids"] == [5002]
    assert len(db.attestations) == 2
    first_lease, second_lease = db.attest_params[0][8], db.attest_params[-1][8]
    assert first_lease != second_lease
    assert all("UPDATE" not in sql and "DELETE" not in sql for sql in db.executed)


def test_same_retention_new_observation_renews_expired_lease() -> None:
    db = _FakeAuthorityDB()
    first = _coordinate(db)
    later = _NOW + timedelta(days=2)
    renewed = _coordinate(db, observation_time=later)
    assert first["retention_attestation_ids"] == [5001]
    assert renewed["retention_attestation_ids"] == [5002]
    assert len(db.attestations) == 2
    assert db.attest_params[0][6] == db.attest_params[1][6]
    assert db.attest_params[0][11] != db.attest_params[1][11]


def test_combined_path_registers_then_reobserves_attests_and_queries() -> None:
    class CountingS3(_S3):
        def __init__(self) -> None:
            super().__init__()
            self.head_count = 0
            self.get_count = 0

        def head_object(self, **kwargs):
            self.head_count += 1
            return super().head_object(**kwargs)

        def get_object(self, **kwargs):
            self.get_count += 1
            return super().get_object(**kwargs)

    db = _FakeAuthorityDB()
    db.register_commit_failures = 1
    s3 = CountingS3()
    manifest, envelope = _signed_manifest()
    result = coordinate_artifact_authority(
        manifest,
        envelope,
        keys={"key-1": KEY},
        expected_system_id="42",
        expected_database_name="brain",
        s3_client=s3,
        connection_factory=db.connection,
        now=_NOW,
        clock=lambda: _NOW,
    )
    assert result["authoritative"] is True
    assert result["retention_attestation_ids"] == [5001]
    assert len(db.registrations) == 1
    assert s3.head_count == 2 and s3.get_count == 2
    register_params = [sql for sql in db.executed if "brain_register" in sql]
    assert len(register_params) == 2
    assert db.registration_params[0] == db.registration_params[1]
    assert any("brain_artifact_is_authoritative" in sql for sql in db.executed)


def test_combined_path_never_claims_authority_without_database_predicate() -> None:
    db = _FakeAuthorityDB()
    db.authority_override = False
    manifest, envelope = _signed_manifest()
    result = coordinate_artifact_authority(
        manifest,
        envelope,
        keys={"key-1": KEY},
        expected_system_id="42",
        expected_database_name="brain",
        s3_client=_S3(),
        connection_factory=db.connection,
        now=_NOW,
        clock=lambda: _NOW,
    )
    assert result["authoritative"] is False


def test_observation_expiry_during_transfer_fails_before_persistence() -> None:
    db = _FakeAuthorityDB()
    manifest, envelope = _signed_manifest()
    expiring = _NOW + timedelta(seconds=1)
    moments = iter((_NOW, _NOW + timedelta(seconds=2)))
    with pytest.raises(ArtifactAuthorityError, match="expired during verification"):
        coordinate_artifact_authority(
            manifest,
            envelope,
            keys={"key-1": KEY},
            expected_system_id="42",
            expected_database_name="brain",
            s3_client=_S3(retain_until=expiring),
            connection_factory=db.connection,
            now=_NOW,
            clock=lambda: next(moments),
        )
    assert db.attestations == {}


def test_retention_binding_never_leaks_the_signing_key() -> None:
    verified, retentions = _verified_with_s3()
    (attestation,) = seal_retention_attestations(verified, retentions, keys={"key-1": KEY})
    assert KEY not in attestation.canonical_bytes
    assert base64.b64encode(KEY).decode() not in attestation.signature
    assert KEY.hex() not in repr(attestation)
    db = _FakeAuthorityDB()
    result = _coordinate(db)
    assert KEY.hex() not in json.dumps(result, default=str)
