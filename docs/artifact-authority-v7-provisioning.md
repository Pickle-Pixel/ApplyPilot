# Artifact authority v7 role provisioning

Artifact authority topology is a privileged provisioning boundary. The
permanent owner membership is exactly
`brain_artifact_authority_owner -> brain_schema_migrator`, with `ADMIN FALSE`,
`INHERIT FALSE`, and `SET TRUE`. Its grantor must be PostgreSQL bootstrap role
OID 10, and that role must still have `rolsuper = true`. The contract does not
depend on the bootstrap role's cluster-specific name.

When OID 10 owns the database, no provider membership is allowed. When a
dedicated control-plane provider owns the database, exactly two additional
permanent memberships are required:
`brain_schema_migrator -> <current database owner>` for writes and
`brain_schema_verifier -> <current database owner>` for the durable identity
read, both with the same three options and grantor OID 10. The provider is
exactly LOGIN, NOSUPERUSER, NOCREATEROLE, and NOINHERIT, with no CREATEDB,
REPLICATION, or BYPASSRLS authority and no other role memberships. Bootstrap
and existing-topology upgrade accept its independent authentication secret only
through `APPLYPILOT_DATABASE_OWNER_PG_PASSWORD` and rotate a client-generated
SCRAM-SHA-256 verifier in the same atomic transaction. Before V10, the writer role
has no descendants; V10 replaces that legacy closure with the fixed,
single-purpose login edges described below.

V10 adds two fixed, separated database identities:

- `brain_artifact_authority_writer_login` may `SET ROLE brain_artifact_authority_writer` only.
- `brain_artifact_authority_verifier_login` may `SET ROLE brain_artifact_authority_verifier` only.

Both login roles are `NOINHERIT`, unprivileged, and have no administrative
membership. Provision their authentication secrets out of band; never reuse a
credential or DSN between them. The supported coordinator requires
`--writer-database-dsn-file` and `--verifier-database-dsn-file`, authenticates
each DSN as its exact fixed login role, and then activates only the matching
capability role. The legacy single `--database-dsn-file` path fails closed.

The writer can register a manifest and append signed retention evidence, but
that row is never authoritative by itself. After the writer commits, the
verifier connection independently re-reads the durable row, validates its
exact canonical payload, digest, key ID, and HMAC signature against the pinned
operator keyring, and writes an exact-row verification through a verifier-only
function. The cutover-facing `brain_artifact_is_authoritative` predicate
requires that immutable verifier result. Writer sessions have neither table
access nor function execution authority over the verification ledger.

The permanent provider SET edge avoids a grant/revoke crash window during
schema migration. A non-superuser provider only migrates a graph that was
already provisioned exactly; it never creates or repairs topology. A genuine
CREATEROLE first-time attempt fails closed because PostgreSQL adds a creator
membership that does not match this contract.

`pg_database_owner` cannot have explicit members. A dedicated provider
therefore preserves that role's ownership of `public`, grants the fixed
migrator the schema grant options required by immutable historical migrations,
and performs migration through the permanent SET edge.
