# Immutable brain schema history fixtures

Each `v<N>.json` file records the published committed byte variants of
`src/applypilot/brain/schema_v<N>.sql` as established across ApplyPilot history.

These fixtures are **immutable** reference data. They exist so the V8 forward
upgrade path (`applypilot.brain.schema.ensure_brain_schema_v8`) and its deep
verifier can be exercised against every historical schema variant:

- `current_committed.sha256` is the SHA-256 of the current committed migration
  bytes. It is what a freshly installed ledger records for that version, and V8
  appends over it without rewriting it (byte-for-byte unchanged ledger rows).
- `historical_variants[]` enumerates earlier committed byte variants. Where the
  full 64-hex digest was published it is pinned in `sha256`; where only the
  short published identifier is known it is recorded in `sha256_prefix`. A
  historical checksum substituted over a catalog it does not describe is rejected
  by the deep verifier, except the V1 `supported_live_legacy` variant which the
  ledger is explicitly allowed to retain.
