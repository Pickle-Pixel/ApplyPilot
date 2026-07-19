# Authority Rollback State Machine Design

## Objective

Make forward V7 rollback safe against hostile PostgreSQL name resolution, already-connected authority sessions, database and schema owners, procedure and archive authority, same-inode HBA edits, ambiguous commit acknowledgements, and retries by a different authenticated break-glass superuser.

## Architecture

Rollback becomes an explicit state machine with authenticated evidence at its irreversible boundaries:

1. Validate the authenticated target using only `pg_catalog` objects under a pinned `search_path`.
2. Commit first-pass database containment and a target-stable marker.
3. Discover and terminate affected non-break-glass, non-controller sessions after `NOLOGIN` is durable.
4. Re-run the idempotent containment SQL to remove grants or objects raced by those sessions, then assert ownership, session, login, ACL, membership, and reconnect closure.
5. Revalidate the authenticated live HBA bytes immediately before descriptor-relative replacement.
6. Restore HBA only after final database containment evidence is visible.

The marker identity binds database name, database OID, cluster system identifier, rollback SQL digest, and source receipt digest. Executor identity is recorded as an observation, not included in the stable marker ID, allowing another approved break-glass superuser to authenticate and continue the same rollback.

## Catalog containment

The generated containment SQL uses `pg_catalog` qualification throughout and pins `search_path` for its transaction. Its role closure uses recursive set semantics over role membership and default-ACL edges, avoiding simple-path enumeration. It inventories public `brain_*` objects plus every object and schema owner or ACL in `brain_archive`.

Functions, procedures, and other executable `pg_proc` kinds are dispatched by `prokind`; procedures use `REVOKE ... ON PROCEDURE`, while functions, aggregates, and window functions use `REVOKE ... ON FUNCTION`. Database and relevant schema ownership are transferred away from affected roles before sessions are terminated. The controller and authenticated break-glass roles are protected from termination while their authority edges and non-owner ACL routes are still contained.

## Failure evidence

A commit call can fail after PostgreSQL has committed. On commit acknowledgement failure, rollback attempts a catalog-qualified marker observation using a fresh transaction. It emits an authenticated `commit_outcome_unknown` receipt containing the target, stable marker binding, source evidence, original commit exception, marker observation, and any rollback exception. Rollback failure is grouped with, and never replaces, the original commit exception.

If a later approved break-glass superuser retries, it reuses the stable marker and predecessor evidence while recording its own executor identity.

## HBA mutation

HBA validation returns authenticated expected live bytes and their digest. `_replace_atomic` opens the target immediately before replacement, verifies inode/device, reads the opened descriptor, and compares exact bytes and SHA-256 to the authenticated expectation. A same-inode edit therefore aborts before mutation.

## Verification

Real PostgreSQL 18 regressions cover hostile `public.pg_roles` and `public.pg_database` shadows, persistent database-owner regrant races, controller and break-glass preservation, procedures, archive ownership, bounded cyclic closure, commit acknowledgement loss, rollback-error preservation, and alternate break-glass retry. A real filesystem test covers an in-place HBA edit between preflight and replacement. Mock/source-search checks are replaced where a real boundary is available.
