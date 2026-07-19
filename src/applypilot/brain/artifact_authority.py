"""Authenticated coordination for immutable brain-artifact authority manifests.

AWS verification deliberately completes before a database connection is opened. The database
function is the final authority boundary and owns exact-request replay semantics.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, BinaryIO, Callable, Mapping

import rfc8785


PURPOSE = "brain-artifact-authority-registration-v1"
RETENTION_PURPOSE = "brain-artifact-retention-attestation-v1"
SCHEMA_VERSION = 1
MAX_CLOCK_SKEW = timedelta(minutes=5)
# A verification lease is deliberately short-lived: authority lapses and must be
# re-observed long before the far-future COMPLIANCE retention deadline elapses.
DEFAULT_VERIFICATION_LEASE = timedelta(days=1)
# Stable namespace so a lease id is a pure function of the observed retention
# identity: an exact replay derives the same id; a provider extension derives a new one.
_RETENTION_LEASE_NAMESPACE = uuid.UUID("2f1d3c4b-5a6e-4f80-9b1c-0d2e3f4a5b6c")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactAuthorityError(ValueError):
    """A manifest cannot be trusted or registered."""


class AmbiguousCommitError(ArtifactAuthorityError):
    """The database connection failed while commit disposition was unknowable."""


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactAuthorityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise ArtifactAuthorityError(f"invalid JSON number: {value}")


def parse_json_strict(value: str | bytes, *, require_canonical: bool = True) -> dict[str, Any]:
    """Parse an object while rejecting duplicates and non-canonical byte encodings."""
    try:
        raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        parsed = json.loads(
            raw,
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_non_json_constant,
        )
    except ArtifactAuthorityError:
        raise
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactAuthorityError("invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ArtifactAuthorityError("JSON root must be an object")
    if require_canonical and raw != canonical_manifest_bytes(parsed):
        raise ArtifactAuthorityError("input must be exact canonical JSON")
    return parsed


def _utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ArtifactAuthorityError("timestamp must be RFC3339 text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactAuthorityError("timestamp must be RFC3339") from exc
    if parsed.tzinfo is None:
        raise ArtifactAuthorityError("timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _required_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        raise ArtifactAuthorityError(
            f"invalid {where} fields (missing={sorted(missing)}, extra={sorted(extra)})"
        )


def _nonempty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ArtifactAuthorityError(f"{field_name} must be non-empty text")
    return value


def _provider_checksum(value: object) -> str:
    checksum = _nonempty_text(value, "awsChecksumSha256")
    try:
        decoded = base64.b64decode(checksum, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ArtifactAuthorityError("awsChecksumSha256 must be canonical base64 SHA-256") from exc
    if len(decoded) != hashlib.sha256().digest_size or base64.b64encode(decoded).decode() != checksum:
        raise ArtifactAuthorityError("awsChecksumSha256 must be canonical base64 SHA-256")
    return checksum


def normalize_manifest(raw: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "purpose",
        "requestId",
        "issuedAt",
        "expiresAt",
        "destination",
        "artifacts",
    }
    _required_keys(raw, expected, "manifest")
    if raw["schemaVersion"] != SCHEMA_VERSION or raw["purpose"] != PURPOSE:
        raise ArtifactAuthorityError("unsupported schema version or purpose")
    request_id = _nonempty_text(raw["requestId"], "requestId")
    try:
        if str(uuid.UUID(request_id)) != request_id:
            raise ValueError
    except ValueError as exc:
        raise ArtifactAuthorityError("requestId must be a canonical UUID") from exc
    _utc(raw["issuedAt"])
    _utc(raw["expiresAt"])

    destination = raw["destination"]
    if not isinstance(destination, Mapping):
        raise ArtifactAuthorityError("destination must be an object")
    _required_keys(destination, {"systemId", "databaseName"}, "destination")
    _nonempty_text(destination["systemId"], "systemId")
    _nonempty_text(destination["databaseName"], "databaseName")

    artifacts = raw["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ArtifactAuthorityError("artifacts must be a non-empty array")

    normalized: list[dict[str, Any]] = []
    ordinals: set[int] = set()
    identities: set[tuple[str, str, str, str]] = set()
    required = {
        "ordinal",
        "artifactSha256",
        "byteCount",
        "mediaType",
        "backend",
        "bucket",
        "objectKey",
        "versionId",
        "etag",
        "awsChecksumSha256",
        "storageImmutable",
        "encryptionMode",
        "encryptionKeyId",
        "policySourceId",
    }
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise ArtifactAuthorityError("artifact must be an object")
        _required_keys(item, required, "artifact")
        entry = dict(item)
        backend = entry["backend"]
        if backend not in {"s3", "aws_s3"}:
            raise ArtifactAuthorityError("artifact backend must be s3 or aws_s3")
        entry["backend"] = "s3"
        ordinal = entry["ordinal"]
        digest = entry["artifactSha256"]
        size = entry["byteCount"]
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 1:
            raise ArtifactAuthorityError("ordinal must be a positive integer")
        if ordinal in ordinals:
            raise ArtifactAuthorityError("duplicate artifact ordinal")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ArtifactAuthorityError("artifactSha256 must be lowercase SHA-256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ArtifactAuthorityError("byteCount must be a non-negative integer")
        for name in ("mediaType", "bucket", "objectKey", "versionId", "etag"):
            _nonempty_text(entry[name], name)
        _provider_checksum(entry["awsChecksumSha256"])
        if entry["storageImmutable"] is not True:
            raise ArtifactAuthorityError("storageImmutable must be true")
        if entry["encryptionMode"] not in {"provider_managed", "customer_managed"}:
            raise ArtifactAuthorityError("unsupported encryptionMode")
        _nonempty_text(entry["encryptionKeyId"], "encryptionKeyId")
        if entry["policySourceId"] is not None:
            _nonempty_text(entry["policySourceId"], "policySourceId")
        identity = (digest, entry["bucket"], entry["objectKey"], entry["versionId"])
        if identity in identities:
            raise ArtifactAuthorityError("duplicate artifact identity")
        ordinals.add(ordinal)
        identities.add(identity)
        normalized.append(entry)
    normalized.sort(key=lambda item: item["ordinal"])
    if [item["ordinal"] for item in normalized] != list(range(1, len(normalized) + 1)):
        raise ArtifactAuthorityError("artifact ordinals must be contiguous starting at one")
    result = dict(raw)
    result["destination"] = dict(destination)
    result["artifacts"] = normalized
    return result


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    try:
        return rfc8785.dumps(manifest)
    except (rfc8785.CanonicalizationError, TypeError) as exc:
        raise ArtifactAuthorityError("value cannot be represented as canonical JSON") from exc


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def signing_message(canonical_bytes: bytes) -> bytes:
    """Domain-separate authority signatures from every other use of the HMAC key."""
    return PURPOSE.encode("ascii") + b"\x00" + canonical_bytes


@dataclass(frozen=True)
class VerifiedManifest:
    manifest: dict[str, Any]
    manifest_sha256: str
    key_id: str
    verified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class S3Observation:
    """One completed, live observation of every exact version in a manifest."""

    retentions: tuple[datetime, ...]
    completed_at: datetime


def verify_manifest_signature(
    manifest: Mapping[str, Any],
    envelope: Mapping[str, Any],
    *,
    keys: Mapping[str, bytes],
    expected_system_id: str,
    expected_database_name: str,
    now: datetime | None = None,
    clock_skew: timedelta = MAX_CLOCK_SKEW,
    allow_expired_replay: bool = False,
) -> VerifiedManifest:
    _required_keys(envelope, {"algorithm", "keyId", "manifestSha256", "signature"}, "signature")
    if envelope["algorithm"] != "hmac-sha256":
        raise ArtifactAuthorityError("unsupported signature algorithm")
    key_id = envelope["keyId"]
    if not isinstance(key_id, str) or key_id not in keys:
        raise ArtifactAuthorityError("unknown signing key")
    key = keys[key_id]
    if not isinstance(key, bytes) or len(key) < 32:
        raise ArtifactAuthorityError("HMAC key must contain at least 32 bytes")
    canonical = canonical_manifest_bytes(manifest)
    digest = hashlib.sha256(canonical).hexdigest()
    supplied_digest = envelope["manifestSha256"]
    if not isinstance(supplied_digest, str) or not hmac.compare_digest(supplied_digest, digest):
        raise ArtifactAuthorityError("manifest digest mismatch")
    expected = hmac.new(key, signing_message(canonical), hashlib.sha256).digest()
    signature = envelope["signature"]
    if not isinstance(signature, str):
        raise ArtifactAuthorityError("invalid signature encoding")
    try:
        supplied = base64.b64decode(
            signature + "=" * (-len(signature) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, base64.binascii.Error) as exc:
        raise ArtifactAuthorityError("invalid signature encoding") from exc
    canonical_signature = base64.urlsafe_b64encode(supplied).rstrip(b"=").decode()
    if len(supplied) != hashlib.sha256().digest_size or canonical_signature != signature:
        raise ArtifactAuthorityError("invalid signature encoding")
    if not hmac.compare_digest(supplied, expected):
        raise ArtifactAuthorityError("invalid manifest signature")
    normalized = normalize_manifest(manifest)
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    issued, expires = _utc(normalized["issuedAt"]), _utc(normalized["expiresAt"])
    if expires <= issued or current < issued - clock_skew or (current >= expires and not allow_expired_replay):
        raise ArtifactAuthorityError("manifest is outside its validity window")
    destination = normalized["destination"]
    system_id = _nonempty_text(expected_system_id, "expected_system_id")
    database_name = _nonempty_text(expected_database_name, "expected_database_name")
    if not hmac.compare_digest(destination["systemId"], system_id):
        raise ArtifactAuthorityError("destination system mismatch")
    if not hmac.compare_digest(destination["databaseName"], database_name):
        raise ArtifactAuthorityError("destination database mismatch")
    return VerifiedManifest(normalized, digest, key_id, current)


def _body_digests(body: BinaryIO, chunk_size: int = 1024 * 1024) -> tuple[str, str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    while True:
        chunk = body.read(chunk_size)
        if not chunk:
            break
        if not isinstance(chunk, bytes):
            raise ArtifactAuthorityError("S3 body did not return bytes")
        byte_count += len(chunk)
        digest.update(chunk)
    return digest.hexdigest(), base64.b64encode(digest.digest()).decode(), byte_count


def observe_s3_versions(
    verified: VerifiedManifest,
    s3_client: Any,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> S3Observation:
    """Verify every artifact's live S3 evidence and return each provider retention instant.

    The normalized ``ObjectLockRetainUntilDate`` per artifact is returned in verified-artifact
    order so the caller can bind it into the immutable retention attestation without a second HEAD.
    """
    retentions: list[datetime] = []
    for item in verified.manifest["artifacts"]:
        args = {"Bucket": item["bucket"], "Key": item["objectKey"], "VersionId": item["versionId"]}
        head = s3_client.head_object(**args, ChecksumMode="ENABLED")
        if head.get("DeleteMarker") is True or str(head.get("VersionId", "")) != item["versionId"]:
            raise ArtifactAuthorityError("S3 version mismatch")
        if head.get("ContentLength") != item["byteCount"]:
            raise ArtifactAuthorityError("S3 byte count mismatch")
        if head.get("ContentType") != item["mediaType"]:
            raise ArtifactAuthorityError("S3 media type mismatch")
        if head.get("ChecksumSHA256") != item["awsChecksumSha256"]:
            raise ArtifactAuthorityError("S3 provider checksum mismatch")
        if str(head.get("ETag", "")).strip('"') != item["etag"]:
            raise ArtifactAuthorityError("S3 ETag mismatch")
        if head.get("ObjectLockMode") != "COMPLIANCE":
            raise ArtifactAuthorityError("S3 object is not under COMPLIANCE Object Lock")
        retain_until = head.get("ObjectLockRetainUntilDate")
        if (
            not isinstance(retain_until, datetime)
            or retain_until.tzinfo is None
            or retain_until.astimezone(timezone.utc) <= verified.verified_at
        ):
            raise ArtifactAuthorityError("S3 Object Lock retention is not active")
        retentions.append(retain_until.astimezone(timezone.utc))
        server_encryption = head.get("ServerSideEncryption")
        if item["encryptionMode"] == "customer_managed":
            if server_encryption != "aws:kms" or head.get("SSEKMSKeyId") != item["encryptionKeyId"]:
                raise ArtifactAuthorityError("S3 customer-managed encryption mismatch")
        elif server_encryption != "AES256" or item["encryptionKeyId"] != "aws/s3":
            raise ArtifactAuthorityError("S3 provider-managed encryption mismatch")

        response = s3_client.get_object(**args, ChecksumMode="ENABLED")
        if str(response.get("VersionId", "")) != item["versionId"]:
            raise ArtifactAuthorityError("S3 downloaded version mismatch")
        if response.get("ChecksumSHA256") != item["awsChecksumSha256"]:
            raise ArtifactAuthorityError("S3 downloaded provider checksum mismatch")
        body = response.get("Body")
        if body is None or not hasattr(body, "read"):
            raise ArtifactAuthorityError("S3 response has no readable body")
        content_digest, provider_digest, byte_count = _body_digests(body)
        if content_digest != item["artifactSha256"]:
            raise ArtifactAuthorityError("S3 content digest mismatch")
        if provider_digest != item["awsChecksumSha256"]:
            raise ArtifactAuthorityError("S3 checksum does not match downloaded content")
        if byte_count != item["byteCount"] or response.get("ContentLength", byte_count) != byte_count:
            raise ArtifactAuthorityError("S3 downloaded byte count mismatch")
    completed_at = _normalize_instant(clock(), "S3 observation completion")
    if any(retention <= completed_at for retention in retentions):
        raise ArtifactAuthorityError("S3 Object Lock retention expired during verification")
    return S3Observation(tuple(retentions), completed_at)


def verify_s3_versions(
    verified: VerifiedManifest,
    s3_client: Any,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> tuple[VerifiedManifest, tuple[datetime, ...]]:
    """Compatibility wrapper returning the verified manifest and live retention instants."""
    observation = observe_s3_versions(verified, s3_client, clock=clock)
    return verified, observation.retentions


def _normalize_instant(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ArtifactAuthorityError(f"{field_name} must be a timezone-aware datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ArtifactAuthorityError(f"{field_name} must include a UTC offset")
    return value.astimezone(timezone.utc)


def _instant_text(value: datetime) -> str:
    """Render an instant as canonical RFC3339 UTC text (Z suffix, no offset)."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def retention_signing_message(canonical_bytes: bytes) -> bytes:
    """Domain-separate retention-attestation signatures from manifest signatures."""
    return RETENTION_PURPOSE.encode("ascii") + b"\x00" + canonical_bytes


@dataclass(frozen=True)
class RetentionAttestation:
    """An immutable, signed binding of one artifact to its live storage retention.

    The canonical payload and signature are computed once, at verification/seal time, so no
    instant (retention deadline, verification time, or lease expiry) can be silently regenerated
    on the path to persistence.
    """

    artifact_hash: str
    backend: str
    bucket_or_container: str | None
    object_key: str
    provider_version_id: str
    provider_checksum: str
    retention_until: datetime
    verified_at: datetime
    verification_lease_id: str
    verification_expires_at: datetime
    canonical_bytes: bytes
    payload_sha256: str
    key_id: str
    signature: str

    def db_params(self) -> tuple[Any, ...]:
        """Exact positional arguments for the V9 ``brain_attest_artifact_retention`` contract."""
        return (
            self.artifact_hash,
            self.backend,
            self.bucket_or_container,
            self.object_key,
            self.provider_version_id,
            self.provider_checksum,
            _instant_text(self.retention_until),
            _instant_text(self.verified_at),
            self.verification_lease_id,
            _instant_text(self.verification_expires_at),
            self.canonical_bytes,
            self.payload_sha256,
            self.key_id,
            self.signature,
        )


def _retention_payload(
    *,
    request_id: str,
    artifact_hash: str,
    backend: str,
    bucket_or_container: str | None,
    object_key: str,
    provider_version_id: str,
    provider_checksum: str,
    retention_until: datetime,
    verified_at: datetime,
    verification_lease_id: str,
    verification_expires_at: datetime,
) -> dict[str, Any]:
    return {
        "purpose": RETENTION_PURPOSE,
        "requestId": request_id,
        "artifactSha256": artifact_hash,
        "backend": backend,
        "bucketOrContainer": bucket_or_container,
        "objectKey": object_key,
        "providerVersionId": provider_version_id,
        "providerChecksum": provider_checksum,
        "retentionMode": "COMPLIANCE",
        "retainUntil": _instant_text(retention_until),
        "verifiedAt": _instant_text(verified_at),
        "verificationLeaseId": verification_lease_id,
        "verificationExpiresAt": _instant_text(verification_expires_at),
    }


def _retention_signing_key(keys: Mapping[str, bytes], key_id: str) -> bytes:
    key = keys.get(key_id) if isinstance(keys, Mapping) else None
    if not isinstance(key, bytes) or len(key) < 32:
        raise ArtifactAuthorityError("retention signing key must contain at least 32 bytes")
    return key


def build_retention_attestation(
    *,
    request_id: str,
    artifact_hash: str,
    backend: str,
    bucket_or_container: str | None,
    object_key: str,
    provider_version_id: str,
    provider_checksum: str,
    retention_until: datetime,
    verified_at: datetime,
    verification_lease_id: str,
    verification_expires_at: datetime,
    keys: Mapping[str, bytes],
    key_id: str,
) -> RetentionAttestation:
    """Construct a self-consistent, signed retention attestation from exact bound fields."""
    retention = _normalize_instant(retention_until, "retention_until")
    verified = _normalize_instant(verified_at, "verified_at")
    expires = _normalize_instant(verification_expires_at, "verification_expires_at")
    lease_id = _nonempty_text(verification_lease_id, "verification_lease_id")
    payload = _retention_payload(
        request_id=request_id,
        artifact_hash=artifact_hash,
        backend=backend,
        bucket_or_container=bucket_or_container,
        object_key=object_key,
        provider_version_id=provider_version_id,
        provider_checksum=provider_checksum,
        retention_until=retention,
        verified_at=verified,
        verification_lease_id=lease_id,
        verification_expires_at=expires,
    )
    canonical = canonical_manifest_bytes(payload)
    key = _retention_signing_key(keys, key_id)
    signature = base64.urlsafe_b64encode(
        hmac.new(key, retention_signing_message(canonical), hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    return RetentionAttestation(
        artifact_hash=artifact_hash,
        backend=backend,
        bucket_or_container=bucket_or_container,
        object_key=object_key,
        provider_version_id=provider_version_id,
        provider_checksum=provider_checksum,
        retention_until=retention,
        verified_at=verified,
        verification_lease_id=lease_id,
        verification_expires_at=expires,
        canonical_bytes=canonical,
        payload_sha256=hashlib.sha256(canonical).hexdigest(),
        key_id=key_id,
        signature=signature,
    )


def validate_retention_attestation(
    attestation: RetentionAttestation, *, now: datetime, keys: Mapping[str, bytes]
) -> None:
    """Re-derive and verify every immutable invariant of a retention attestation.

    Rejects canonical/signature/hash drift, a verification instant in the future, an expired
    verification lease, and a lease whose expiry outlives the provider retention deadline.
    """
    evaluation = _normalize_instant(now, "now")
    payload = _retention_payload(
        request_id=json.loads(attestation.canonical_bytes)["requestId"],
        artifact_hash=attestation.artifact_hash,
        backend=attestation.backend,
        bucket_or_container=attestation.bucket_or_container,
        object_key=attestation.object_key,
        provider_version_id=attestation.provider_version_id,
        provider_checksum=attestation.provider_checksum,
        retention_until=attestation.retention_until,
        verified_at=attestation.verified_at,
        verification_lease_id=attestation.verification_lease_id,
        verification_expires_at=attestation.verification_expires_at,
    )
    canonical = canonical_manifest_bytes(payload)
    if canonical != attestation.canonical_bytes:
        raise ArtifactAuthorityError("retention attestation payload drift")
    if hashlib.sha256(canonical).hexdigest() != attestation.payload_sha256:
        raise ArtifactAuthorityError("retention attestation digest drift")
    key = _retention_signing_key(keys, attestation.key_id)
    expected = base64.urlsafe_b64encode(
        hmac.new(key, retention_signing_message(canonical), hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    if not hmac.compare_digest(expected, attestation.signature):
        raise ArtifactAuthorityError("retention attestation signature mismatch")
    if attestation.retention_until <= attestation.verified_at:
        raise ArtifactAuthorityError("retention deadline is not in the future of verification")
    if attestation.verified_at > evaluation:
        raise ArtifactAuthorityError("retention verified_at is in the future")
    if attestation.verification_expires_at < attestation.verified_at:
        raise ArtifactAuthorityError("verification lease predates its verification")
    if attestation.verification_expires_at > attestation.retention_until:
        raise ArtifactAuthorityError("verification lease outlives provider retention")
    if attestation.verification_expires_at <= evaluation:
        raise ArtifactAuthorityError("verification lease has expired")


def _retention_lease_id(
    item: Mapping[str, Any], retention_until: datetime, verified_at: datetime
) -> str:
    seed = "\x00".join(
        (
            item["artifactSha256"],
            item["backend"],
            item["bucket"],
            item["objectKey"],
            item["versionId"],
            item["awsChecksumSha256"],
            _instant_text(retention_until),
            _instant_text(verified_at),
        )
    )
    return str(uuid.uuid5(_RETENTION_LEASE_NAMESPACE, seed))


def seal_retention_attestations(
    verified: VerifiedManifest,
    retentions: tuple[datetime, ...],
    *,
    keys: Mapping[str, bytes],
    lease_duration: timedelta = DEFAULT_VERIFICATION_LEASE,
    now: datetime | None = None,
    observation_completed_at: datetime | None = None,
) -> tuple[RetentionAttestation, ...]:
    """Bind each verified artifact to a bounded, signed retention attestation.

    ``verified_at`` is the exact instant the live storage observation completed; the lease expiry
    is clamped so it never exceeds the provider retention deadline.
    """
    artifacts = verified.manifest["artifacts"]
    if len(retentions) != len(artifacts):
        raise ArtifactAuthorityError("retention evidence count does not match verified artifacts")
    if lease_duration <= timedelta(0):
        raise ArtifactAuthorityError("verification lease duration must be positive")
    verified_at = _normalize_instant(
        observation_completed_at or verified.verified_at, "observation_completed_at"
    )
    evaluation = (now or verified_at).astimezone(timezone.utc)
    request_id = verified.manifest["requestId"]
    key_id = verified.key_id
    attestations: list[RetentionAttestation] = []
    for item, retain_until in zip(artifacts, retentions):
        retention = _normalize_instant(retain_until, "ObjectLockRetainUntilDate")
        if retention <= verified_at:
            raise ArtifactAuthorityError("retention deadline is not in the future of verification")
        expires = min(verified_at + lease_duration, retention)
        attestation = build_retention_attestation(
            request_id=request_id,
            artifact_hash=item["artifactSha256"],
            backend=item["backend"],
            bucket_or_container=item["bucket"],
            object_key=item["objectKey"],
            provider_version_id=item["versionId"],
            provider_checksum=item["awsChecksumSha256"],
            retention_until=retention,
            verified_at=verified_at,
            verification_lease_id=_retention_lease_id(item, retention, verified_at),
            verification_expires_at=expires,
            keys=keys,
            key_id=key_id,
        )
        validate_retention_attestation(attestation, now=evaluation, keys=keys)
        attestations.append(attestation)
    return tuple(attestations)


_RETENTION_SQL = (
    "SELECT brain_attest_artifact_retention("
    "%s, %s, %s, %s, %s, %s, %s::timestamptz, %s::timestamptz, %s, %s::timestamptz, "
    "%s::bytea, %s, %s, %s)"
)


def attest_retention(connection: Any, attestations: tuple[RetentionAttestation, ...]) -> list[int]:
    """Append each retention attestation through the exact V9 authority function.

    A commit exception is classified as ambiguous so the caller replays the identical, idempotent
    request rather than guessing a later deadline or minting a second row.
    """
    ids: list[int] = []
    try:
        with connection.cursor() as cursor:
            for attestation in attestations:
                cursor.execute(_RETENTION_SQL, attestation.db_params())
                row = cursor.fetchone()
                value = next(iter(row.values())) if isinstance(row, Mapping) else (row[0] if row else None)
                if value is None:
                    raise ArtifactAuthorityError("database returned no retention attestation id")
                ids.append(int(value))
    except Exception:
        try:
            connection.rollback()
        except Exception:
            pass
        raise
    try:
        connection.commit()
    except Exception as exc:
        raise AmbiguousCommitError("retention attestation commit disposition is ambiguous") from exc
    return ids


_PERSISTED_RETENTION_COLUMNS = (
    "retention_attestation_id",
    "database_incarnation_id",
    "artifact_hash",
    "backend",
    "bucket_or_container",
    "object_key",
    "provider_version_id",
    "provider_checksum",
    "retain_until",
    "verified_at",
    "verification_lease_id",
    "verification_lease_expires_at",
    "canonical_payload",
    "payload_sha256",
    "signing_key_id",
    "signature",
)


def _persisted_retention_row(row: object) -> dict[str, Any]:
    if isinstance(row, Mapping):
        values = {name: row.get(name) for name in _PERSISTED_RETENTION_COLUMNS}
    elif isinstance(row, (tuple, list)) and len(row) == len(_PERSISTED_RETENTION_COLUMNS):
        values = dict(zip(_PERSISTED_RETENTION_COLUMNS, row, strict=True))
    else:
        raise ArtifactAuthorityError("database returned an invalid retention verification row")
    required_text = (
        "artifact_hash",
        "backend",
        "object_key",
        "provider_version_id",
        "provider_checksum",
        "verification_lease_id",
        "payload_sha256",
        "signing_key_id",
        "signature",
    )
    if not isinstance(values["retention_attestation_id"], int) or any(
        not isinstance(values[name], str) for name in required_text
    ):
        raise ArtifactAuthorityError("database returned malformed retention verification fields")
    if values["bucket_or_container"] is not None and not isinstance(
        values["bucket_or_container"], str
    ):
        raise ArtifactAuthorityError("database returned malformed retention bucket")
    if not all(
        isinstance(values[name], datetime)
        for name in ("retain_until", "verified_at", "verification_lease_expires_at")
    ):
        raise ArtifactAuthorityError("database returned malformed retention instants")
    try:
        values["canonical_payload"] = bytes(values["canonical_payload"])
        values["database_incarnation_id"] = str(uuid.UUID(str(values["database_incarnation_id"])))
    except (TypeError, ValueError) as exc:
        raise ArtifactAuthorityError("database returned malformed bound retention evidence") from exc
    return values


def verify_persisted_retention(
    connection: Any,
    retention_attestation_ids: tuple[int, ...],
    *,
    keys: Mapping[str, bytes],
    now: datetime,
) -> list[int]:
    """Re-read, cryptographically verify, and verifier-mark exact committed V9 rows."""
    verification_ids: list[int] = []
    try:
        with connection.cursor() as cursor:
            for attestation_id in retention_attestation_ids:
                cursor.execute(
                    "SELECT * FROM brain_get_artifact_retention_for_verification(%s)",
                    (attestation_id,),
                )
                raw = cursor.fetchone()
                if raw is None:
                    raise ArtifactAuthorityError("committed retention attestation is unavailable")
                row = _persisted_retention_row(raw)
                if row["retention_attestation_id"] != attestation_id:
                    raise ArtifactAuthorityError("database returned a different retention attestation")
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
                    canonical_bytes=row["canonical_payload"],
                    payload_sha256=row["payload_sha256"],
                    key_id=row["signing_key_id"],
                    signature=row["signature"],
                )
                validate_retention_attestation(attestation, now=now, keys=keys)
                cursor.execute(
                    "SELECT brain_verify_artifact_retention(%s,%s::uuid,%s,%s,%s)",
                    (
                        attestation_id,
                        row["database_incarnation_id"],
                        attestation.payload_sha256,
                        attestation.key_id,
                        attestation.signature,
                    ),
                )
                verified_row = cursor.fetchone()
                verification_id = (
                    next(iter(verified_row.values()))
                    if isinstance(verified_row, Mapping)
                    else (verified_row[0] if verified_row else None)
                )
                if not isinstance(verification_id, int):
                    raise ArtifactAuthorityError("database returned no retention verification id")
                verification_ids.append(verification_id)
    except Exception:
        try:
            connection.rollback()
        except Exception:
            pass
        raise
    try:
        connection.commit()
    except Exception as exc:
        raise AmbiguousCommitError("retention verification commit disposition is ambiguous") from exc
    return verification_ids


def _database_artifacts(verified: VerifiedManifest) -> list[dict[str, Any]]:
    return [
        {
            "artifact_hash": item["artifactSha256"],
            "byte_length": item["byteCount"],
            "media_type": item["mediaType"],
            "backend": item["backend"],
            "bucket": item["bucket"],
            "object_key": item["objectKey"],
            "provider_version_id": item["versionId"],
            "provider_checksum": item["awsChecksumSha256"],
            "storage_immutable": item["storageImmutable"],
            "encryption_mode": item["encryptionMode"],
            "encryption_key_id": item["encryptionKeyId"],
            "policy_source_id": item["policySourceId"],
        }
        for item in verified.manifest["artifacts"]
    ]


def _registration_params(verified: VerifiedManifest) -> tuple[Any, ...]:
    manifest = verified.manifest
    destination = manifest["destination"]
    return (
        manifest["requestId"],
        verified.manifest_sha256,
        manifest["purpose"],
        verified.key_id,
        manifest["issuedAt"],
        manifest["expiresAt"],
        destination["systemId"],
        destination["databaseName"],
        json.dumps(_database_artifacts(verified), sort_keys=True, separators=(",", ":")),
    )


def validate_registration_receipt(receipt: object, verified: VerifiedManifest) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise ArtifactAuthorityError("database returned an invalid authority receipt")
    manifest = verified.manifest
    expected = {
        "request_id": manifest["requestId"],
        "manifest_sha256": verified.manifest_sha256,
        "destination_system_id": manifest["destination"]["systemId"],
        "destination_database_name": manifest["destination"]["databaseName"],
        "artifact_count": len(manifest["artifacts"]),
    }
    labels = {
        "request_id": "request ID",
        "manifest_sha256": "manifest digest",
        "destination_system_id": "destination system",
        "destination_database_name": "destination database",
        "artifact_count": "artifact count",
    }
    if receipt.get("status") != "registered":
        raise ArtifactAuthorityError("database receipt is not registered")
    for key, value in expected.items():
        actual = receipt.get(key)
        if key == "request_id":
            actual = str(actual) if actual is not None else None
        if actual != value:
            raise ArtifactAuthorityError(f"database receipt {labels[key]} mismatch")
    return receipt


def register_verified_manifest(connection: Any, verified: VerifiedManifest) -> dict[str, Any]:
    """Execute one request. A commit exception is deliberately classified as ambiguous."""
    sql = """SELECT brain_register_authoritative_artifact_manifest(
        %s::uuid, %s, %s, %s, %s::timestamptz, %s::timestamptz,
        %s, %s, %s::jsonb
    )"""
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, _registration_params(verified))
            row = cursor.fetchone()
        if not row:
            raise ArtifactAuthorityError("database returned no authority receipt")
        receipt = next(iter(row.values())) if isinstance(row, Mapping) else row[0]
        if isinstance(receipt, str):
            receipt = parse_json_strict(receipt, require_canonical=False)
        validated = validate_registration_receipt(receipt, verified)
    except Exception:
        try:
            connection.rollback()
        except Exception:
            pass
        raise
    try:
        connection.commit()
    except Exception as exc:
        raise AmbiguousCommitError("database commit disposition is ambiguous") from exc
    return validated


def _register_once(connection_factory: Callable[[], Any], verified: VerifiedManifest) -> dict[str, Any]:
    connection = connection_factory()
    try:
        return register_verified_manifest(connection, verified)
    finally:
        try:
            connection.close()
        except Exception:
            pass


def coordinate_artifact_registration(
    manifest: Mapping[str, Any],
    envelope: Mapping[str, Any],
    *,
    keys: Mapping[str, bytes],
    expected_system_id: str,
    expected_database_name: str,
    s3_client: Any,
    connection_factory: Callable[[], Any] | None,
    dry_run: bool = False,
    now: datetime | None = None,
    allow_expired_replay: bool = False,
) -> dict[str, Any]:
    if dry_run and allow_expired_replay:
        raise ArtifactAuthorityError("dry-run cannot perform expired replay recovery")
    verified = verify_manifest_signature(
        manifest,
        envelope,
        keys=keys,
        expected_system_id=expected_system_id,
        expected_database_name=expected_database_name,
        now=now,
        allow_expired_replay=allow_expired_replay,
    )
    verified, _retentions = verify_s3_versions(verified, s3_client)
    if dry_run:
        return {
            "artifact_count": len(verified.manifest["artifacts"]),
            "authoritative": False,
            "dry_run": True,
            "manifest_sha256": verified.manifest_sha256,
            "promotable": False,
            "request_id": verified.manifest["requestId"],
        }
    if connection_factory is None:
        raise ArtifactAuthorityError("database connection is required")
    try:
        return _register_once(connection_factory, verified)
    except AmbiguousCommitError:
        # The only safe recovery operation is the exact same database request. The v6 database
        # API must return the original receipt before applying its normal expiry check on replay.
        return _register_once(connection_factory, verified)


def _attest_once(
    connection_factory: Callable[[], Any],
    attestations: tuple[RetentionAttestation, ...],
) -> list[int]:
    connection = connection_factory()
    try:
        return attest_retention(connection, attestations)
    finally:
        try:
            connection.close()
        except Exception:
            pass


def _verify_persisted_once(
    connection_factory: Callable[[], Any],
    retention_attestation_ids: tuple[int, ...],
    *,
    keys: Mapping[str, bytes],
    now: datetime,
) -> list[int]:
    connection = connection_factory()
    try:
        return verify_persisted_retention(
            connection, retention_attestation_ids, keys=keys, now=now
        )
    finally:
        try:
            connection.close()
        except Exception:
            pass


def query_artifact_authority(
    connection_factory: Callable[[], Any], artifact_hashes: tuple[str, ...]
) -> dict[str, bool]:
    """Read the database authority predicate after the attestation commit."""
    connection = connection_factory()
    statuses: dict[str, bool] = {}
    try:
        with connection.cursor() as cursor:
            for artifact_hash in artifact_hashes:
                cursor.execute(
                    "SELECT brain_artifact_is_authoritative(%s) AS authoritative",
                    (artifact_hash,),
                )
                row = cursor.fetchone()
                if isinstance(row, Mapping):
                    authoritative = row.get("authoritative")
                else:
                    authoritative = row[0] if row else None
                if not isinstance(authoritative, bool):
                    raise ArtifactAuthorityError("database returned no artifact authority decision")
                statuses[artifact_hash] = authoritative
    finally:
        try:
            connection.rollback()
        except Exception:
            pass
        try:
            connection.close()
        except Exception:
            pass
    return statuses


def coordinate_retention_attestation(
    manifest: Mapping[str, Any],
    envelope: Mapping[str, Any],
    *,
    keys: Mapping[str, bytes],
    expected_system_id: str,
    expected_database_name: str,
    s3_client: Any,
    connection_factory: Callable[[], Any] | None,
    verifier_connection_factory: Callable[[], Any] | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
    lease_duration: timedelta = DEFAULT_VERIFICATION_LEASE,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Re-verify a signed manifest's live S3 retention and append a durable V9 attestation.

    Authority is conferred only by the database's ``brain_artifact_is_authoritative`` predicate,
    which requires a matching COMPLIANCE retention attestation whose lease is still live. A
    historical (pre-V8, or merely registered) artifact stays non-authoritative until this genuine
    re-verification appends a fresh attestation; no structurally forgeable manifest can shortcut it.
    Retention evidence is sealed into an immutable signed attestation before any database round-trip
    so no observed instant is regenerated between verify, sign, and persist.
    """
    verified = verify_manifest_signature(
        manifest,
        envelope,
        keys=keys,
        expected_system_id=expected_system_id,
        expected_database_name=expected_database_name,
        now=now,
    )
    observation = observe_s3_versions(verified, s3_client, clock=clock)
    attestations = seal_retention_attestations(
        verified,
        observation.retentions,
        keys=keys,
        lease_duration=lease_duration,
        now=observation.completed_at,
        observation_completed_at=observation.completed_at,
    )
    if dry_run:
        return {
            "artifact_count": len(attestations),
            "authoritative": False,
            "dry_run": True,
            "manifest_sha256": verified.manifest_sha256,
            "request_id": verified.manifest["requestId"],
            "retention_attestation_count": len(attestations),
            "storage_observation_completed_at": _instant_text(observation.completed_at),
        }
    if connection_factory is None or verifier_connection_factory is None:
        raise ArtifactAuthorityError("separate writer and verifier database connections are required")
    try:
        retention_ids = _attest_once(connection_factory, attestations)
    except AmbiguousCommitError:
        # Each attestation is idempotent on exact replay (the V9 function returns the durable row
        # for an identical retention identity and fails closed on a conflicting payload), so a lost
        # commit disposition is resolved by re-issuing the identical request.
        retention_ids = _attest_once(connection_factory, attestations)
    retention_id_tuple = tuple(retention_ids)
    verification_now = observation.completed_at
    try:
        verification_ids = _verify_persisted_once(
            verifier_connection_factory,
            retention_id_tuple,
            keys=keys,
            now=verification_now,
        )
    except AmbiguousCommitError:
        verification_ids = _verify_persisted_once(
            verifier_connection_factory,
            retention_id_tuple,
            keys=keys,
            now=verification_now,
        )
    artifact_hashes = tuple(item.artifact_hash for item in attestations)
    authority = query_artifact_authority(verifier_connection_factory, artifact_hashes)
    return {
        "authoritative": bool(authority) and all(authority.values()),
        "artifact_authority": authority,
        "manifest_sha256": verified.manifest_sha256,
        "request_id": verified.manifest["requestId"],
        "retention_attestation_ids": retention_ids,
        "retention_verification_ids": verification_ids,
        "storage_observation_completed_at": _instant_text(observation.completed_at),
    }


def coordinate_artifact_authority(
    manifest: Mapping[str, Any],
    envelope: Mapping[str, Any],
    *,
    keys: Mapping[str, bytes],
    expected_system_id: str,
    expected_database_name: str,
    s3_client: Any,
    connection_factory: Callable[[], Any] | None,
    verifier_connection_factory: Callable[[], Any] | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
    allow_expired_replay: bool = False,
    lease_duration: timedelta = DEFAULT_VERIFICATION_LEASE,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Execute the complete register, re-observe, attest, and predicate-gated path."""
    if dry_run and allow_expired_replay:
        raise ArtifactAuthorityError("dry-run cannot perform expired replay recovery")
    verified = verify_manifest_signature(
        manifest,
        envelope,
        keys=keys,
        expected_system_id=expected_system_id,
        expected_database_name=expected_database_name,
        now=now,
        allow_expired_replay=allow_expired_replay,
    )
    initial_observation = observe_s3_versions(verified, s3_client, clock=clock)
    if dry_run:
        return {
            "artifact_count": len(verified.manifest["artifacts"]),
            "authoritative": False,
            "dry_run": True,
            "manifest_sha256": verified.manifest_sha256,
            "promotable": False,
            "request_id": verified.manifest["requestId"],
            "storage_observation_completed_at": _instant_text(initial_observation.completed_at),
        }
    if connection_factory is None or verifier_connection_factory is None:
        raise ArtifactAuthorityError("separate writer and verifier database connections are required")
    try:
        registration = _register_once(connection_factory, verified)
    except AmbiguousCommitError:
        registration = _register_once(connection_factory, verified)

    # This must be a new provider round-trip after the durable registration commit.
    final_observation = observe_s3_versions(verified, s3_client, clock=clock)
    attestations = seal_retention_attestations(
        verified,
        final_observation.retentions,
        keys=keys,
        lease_duration=lease_duration,
        now=final_observation.completed_at,
        observation_completed_at=final_observation.completed_at,
    )
    try:
        retention_ids = _attest_once(connection_factory, attestations)
    except AmbiguousCommitError:
        retention_ids = _attest_once(connection_factory, attestations)
    retention_id_tuple = tuple(retention_ids)
    verification_now = final_observation.completed_at
    try:
        verification_ids = _verify_persisted_once(
            verifier_connection_factory,
            retention_id_tuple,
            keys=keys,
            now=verification_now,
        )
    except AmbiguousCommitError:
        verification_ids = _verify_persisted_once(
            verifier_connection_factory,
            retention_id_tuple,
            keys=keys,
            now=verification_now,
        )
    artifact_hashes = tuple(attestation.artifact_hash for attestation in attestations)
    authority = query_artifact_authority(verifier_connection_factory, artifact_hashes)
    return {
        **registration,
        "authoritative": bool(authority) and all(authority.values()),
        "artifact_authority": authority,
        "retention_attestation_ids": retention_ids,
        "retention_verification_ids": verification_ids,
        "storage_observation_completed_at": _instant_text(final_observation.completed_at),
    }
