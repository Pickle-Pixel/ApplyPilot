# Immutable brain schema history fixtures

Each `v<N>.json` file records the published committed byte variants of
`src/applypilot/brain/schema_v<N>.sql` as established across ApplyPilot history.

These fixtures are **immutable** reference data. They exist so the V8 forward
upgrade path (`applypilot.brain.schema.ensure_brain_schema_v8`) and its deep
verifier can be exercised against every historical schema variant:

- `current_committed.sha256` is the SHA-256 of the current committed migration
  bytes. It is what a freshly installed ledger records for that version, and V8
  appends over it without rewriting it (byte-for-byte unchanged ledger rows).
- `historical_variants[]` enumerates earlier committed byte variants. Each entry
  pins the full 64-hex `sha256` of the exact historical migration bytes. A
  historical checksum substituted over a catalog it does not describe is rejected
  by the deep verifier, except the V1 `supported_live_legacy` variant which the
  ledger is explicitly allowed to retain.

## Exact raw blobs (`blobs/`)

Every prefix-published variant is now backed by the **exact raw blob bytes**
extracted from Git history, stored verbatim under `blobs/` and marked `-text` in
`.gitattributes` so they are never normalized. Each variant records:

- `sha256` — full digest; a test asserts `sha256(blob bytes) == sha256` so the
  fixture can never silently drift to a reconstructed or prefix-truncated value.
- `sha256_prefix` — the historically published short identifier (informational
  only; the full `sha256` is the authority).
- `blob` — path to the raw bytes under this directory.
- `source_commit` / `git_blob` — provenance: the commit and Git blob object id
  the bytes were extracted from.

The blobs are genuinely earlier, behaviourally different migration bodies (not
whitespace variants). They exist so `test_v8_rejects_known_historical_checksum_*`
can prove — with exact, non-prefix digests — that substituting any of them over
the current V8 catalog is rejected by the deep verifier. They are never applied
as migrations; the supported forward path installs the current committed bytes.
