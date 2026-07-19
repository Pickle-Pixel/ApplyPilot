# Authority Rollback State Machine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all quality-review gaps in the authenticated PostgreSQL authority rollback before any HBA admission restoration.

**Architecture:** Refactor rollback into explicit target validation, first-pass containment commit, affected-session fencing, second-pass containment, final assertions, and HBA restoration. Every PostgreSQL query is catalog-qualified; every irreversible or ambiguous boundary emits target-bound authenticated evidence.

**Tech Stack:** Python 3.12, psycopg 3, PostgreSQL 18, pytest, Ruff

---

### Task 1: Hostile catalog resolution

**Files:**
- Modify: `scripts/rollback-fleet-pg-role.py`
- Test: `tests/test_pg_role_bootstrap.py`

- [ ] Add a real PG18 test that creates hostile `public.pg_roles` and `public.pg_database` shadows returning OID `424242`, sets attacker `search_path`, and proves authenticated target validation still reads `pg_catalog` and rejects a recreated database OID.
- [ ] Run the test against commit `3968416d` and record the expected red result.
- [ ] Qualify all target, marker, and postcondition catalog objects/functions and pin the connection transaction to `pg_catalog`.
- [ ] Re-run the hostile test and the rollback unit subset.

### Task 2: Persistent-session and ownership fence

**Files:**
- Modify: `src/applypilot/fleet/pg_roles.py`
- Modify: `scripts/rollback-fleet-pg-role.py`
- Test: `tests/test_pg_role_bootstrap.py`

- [ ] Add a real PG18 database-owner session that survives first containment, attempts to regrant PUBLIC schema/function authority, and creates a `brain_*` procedure; assert the controller and break-glass sessions are not terminated.
- [ ] Run the test and record the persistent-session escape.
- [ ] Extend rollback SQL to transfer affected database/schema ownership, protect controller/break-glass principals, and persist the affected-role set needed by the runtime fence.
- [ ] Add helpers to terminate affected sessions only after durable `NOLOGIN`, then execute containment a second time and assert zero affected sessions, logins, reconnect paths, ownership, ACL, membership, or executable authority.
- [ ] Re-run the persistent-session test twice to prove idempotence.

### Task 3: Complete executable and archive containment

**Files:**
- Modify: `src/applypilot/fleet/pg_roles.py`
- Test: `tests/test_pg_role_bootstrap.py`

- [ ] Add real PG18 procedure grants, procedure ownership, `brain_archive` schema ownership, relation ownership, and ACL grants to an undeclared provider role.
- [ ] Run the test and record the procedure/archive escape.
- [ ] Inventory all relevant `pg_proc.prokind` values and emit object-kind-correct revokes for functions and procedures across public `brain_*` and `brain_archive`.
- [ ] Assert no non-owner executable, archive relation/schema ownership, or ACL routes remain.
- [ ] Replace recursive simple-path enumeration with bounded recursive set semantics and test a cyclic authority/default-ACL graph.

### Task 4: Mutation-time HBA binding

**Files:**
- Modify: `scripts/rollback-fleet-pg-role.py`
- Test: `tests/test_pg_role_bootstrap.py`

- [ ] Add a real filesystem test that edits the live HBA in place after authenticated preflight while preserving inode/device.
- [ ] Run it and record that the old implementation replaces the edited file.
- [ ] Pass authenticated expected bytes and SHA-256 into `_replace_atomic`; read and compare the target through an opened descriptor immediately before `os.replace`.
- [ ] Re-run the test and verify the edited live HBA is preserved and replacement aborts.

### Task 5: Ambiguous commit and alternate executor evidence

**Files:**
- Modify: `scripts/rollback-fleet-pg-role.py`
- Test: `tests/test_pg_role_bootstrap.py`

- [ ] Add commit-acknowledgement-loss tests where the marker is visible after `commit()` raises, including a simultaneous `rollback()` failure.
- [ ] Add a retry test using a second authenticated break-glass superuser.
- [ ] Make marker IDs target-stable by removing executor identity from the ID binding while retaining per-attempt executor evidence.
- [ ] Emit an authenticated commit-unknown receipt preserving the original commit and rollback errors and the marker observation.
- [ ] Verify the alternate executor reuses the stable marker without weakening target or predecessor checks.

### Task 6: Cumulative verification and checkpoint

**Files:**
- Verify: `scripts/rollback-fleet-pg-role.py`
- Verify: `scripts/upgrade-brain-authority-existing-topology.py`
- Verify: `src/applypilot/fleet/pg_roles.py`
- Verify: `tests/test_pg_role_bootstrap.py`
- Verify: `tests/test_upgrade_brain_authority_existing_topology_cli.py`

- [ ] Run `python -m pytest -q -rs tests/test_pg_role_bootstrap.py tests/test_upgrade_brain_authority_existing_topology_cli.py` and require all PG18 tests green with only the fixed PG16 fixture skip.
- [ ] Run `python -m py_compile` over all five owned Python files.
- [ ] Run Ruff over the four configured files and `git diff --check`.
- [ ] Review the full diff against every quality finding, commit the cumulative repair, and repeat the exact verification after commit.

### Task 7: Preserve ACL grantee identity

**Files:**
- Modify: `src/applypilot/fleet/pg_roles.py`
- Test: `tests/test_pg_role_bootstrap.py`

- [ ] Add a real PG18 regression containing both pseudo-role `PUBLIC` grants and grants to a quoted real role named `"PUBLIC"` across practical ACL classes.
- [ ] Prove the string-sentinel implementation misroutes the quoted role.
- [ ] Carry `grantee_oid`, `grantee_name`, and `is_public` through inventory and branch only on `is_public` in local and cross-database revocation.
- [ ] Re-run the regression twice and assert neither authority route remains.

### Task 8: Upgrade legacy markers and neutralize default ACLs

**Files:**
- Modify: `scripts/rollback-fleet-pg-role.py`
- Test: `tests/test_pg_role_bootstrap.py`

- [ ] Add real PG18 tests for an interrupted exact ten-column legacy marker and for marker creation under inherited default table ACLs.
- [ ] Add wrong-shape tests and require failure before schema mutation.
- [ ] Accept only the exact legacy or current contract; upgrade legacy markers transactionally with conservative `completed_phases = ARRAY['phase1']`.
- [ ] Revoke inherited marker ACLs using explicit OID/public identity, then validate the exact current contract and zero nonowner ACLs.
- [ ] Resume phase two from the upgraded authenticated marker without evidence loss.

### Task 9: Emit durable phase-specific failure evidence

**Files:**
- Modify: `scripts/rollback-fleet-pg-role.py`
- Test: `tests/test_pg_role_bootstrap.py`

- [ ] Add regressions for ordinary termination, phase-two, and postcondition failures after durable phase one.
- [ ] Emit an authenticated partial receipt containing the attempted phase and latest authenticated durable marker for each failure.
- [ ] Make commit-unknown evidence consume the known durable marker so phase one is never represented by an empty completed-phase set.
- [ ] Verify receipt signatures and target/source bindings in tests.

### Task 10: Close the same-inode HBA replacement race

**Files:**
- Modify: `scripts/rollback-fleet-pg-role.py`
- Test: `tests/test_pg_role_bootstrap.py`

- [ ] Add a Linux/WSL regression that writes to the authenticated inode immediately before replacement.
- [ ] Implement same-directory `renameat2(RENAME_EXCHANGE)`, authenticate the displaced live inode after exchange, and atomically exchange back on drift.
- [ ] Fail closed without mutation when atomic exchange is unavailable or cannot be authenticated.
- [ ] Verify concurrent bytes survive the rejected operation and the successful path preserves mode, ownership, and requested payload.

### Task 11: Round-two cumulative gate

- [ ] Run the complete PG18 suite, including interrupted legacy resume and quoted-role containment.
- [ ] Run the real POSIX HBA regression under WSL/Linux.
- [ ] Run Ruff, `py_compile`, and `git diff --check`.
- [ ] Inspect the final diff for unrelated changes and create one consolidated clean commit only after every P0-P2 gate is green.
