# PostgreSQL 18 Exact-Head CI Repair Design

**Date:** 2026-07-20
**Candidate under diagnosis:** `3f9ec16db91acd0cc4293b5a4fc4d6d7c3a9ae2a`
**Failed fork run:** `29736923848`
**Scope:** CI and test-harness repair only; no runtime behavior or live database mutation

## Objective

Restore trustworthy GitHub exact-head verification for the PostgreSQL 18 migration candidate without weakening its TLS, principal-isolation, filesystem-integrity, or disposable-database requirements.

Success means a new exact commit passes both supported Linux Python versions, the bounded Windows-native shards, lint, packaging, and the non-publishing container build. The repaired head must then receive fresh independent reviews and signed checkpoint evidence before it can replace `3f9ec16d` as the promotion candidate.

## Verified Failure Model

Fork run `29736923848` used the exact candidate commit. Python 3.11 completed with 4,251 passes, 139 skips, and 18 failures. Python 3.12 reached the same test summary before the workflow marked it cancelled after the 3.11 matrix failure. The Windows shards and container build passed.

The 18 failures have three causes:

1. Fifteen tests use hardened runtime-principal fixtures that require TLS. The workflow starts the stock `postgres:18` service without TLS, producing `server does not support SSL, but SSL was required`.
2. Two Windows-native contract nodes invoke `cmd.exe` or `Get-Acl` but remain in the broad Ubuntu test collection.
3. One POSIX descriptor-integrity test monkeypatches `os.open`. Production capability detection correctly checks membership in `os.supports_dir_fd`, but the test's wrapper is not represented in that set, so the test disables the capability it intends to exercise.

These are CI and test-harness defects. The run does not justify changing migration runtime code or weakening TLS.

## Considered Approaches

### A. Repair the existing workflow and harness (selected)

Enable TLS inside the existing PostgreSQL 18 service container, route only the two Windows-native nodes to the existing Windows matrix, preserve the POSIX capability during monkeypatching, and disable matrix fail-fast.

This is the smallest change, retains the current official PostgreSQL image, and keeps the security contract aligned with the authoritative GGGTower environment.

### B. Build a custom TLS-enabled PostgreSQL CI image

This could make service startup more declarative, but it introduces a new image build, provenance surface, cache dependency, and maintenance path. It is unnecessary for a one-workflow configuration gap.

### C. Relax TLS or skip the failing tests

Rejected. Setting `sslmode=disable` or `prefer`, broadly skipping database tests, or suppressing platform failures would make CI green by removing the controls the migration is required to prove.

## Selected Design

### PostgreSQL service boundary

Add a Linux step before the disposable-cluster marker that uses the GitHub-provided PostgreSQL service container ID to:

- generate a one-day self-signed test certificate inside the disposable container;
- assign the certificate and private key to the `postgres` account with restrictive key permissions;
- enable PostgreSQL SSL and restart the service;
- poll readiness with a bounded retry loop;
- connect with `sslmode=require` and assert `SHOW ssl` reports `on` before marking the database disposable.

The certificate is ephemeral and test-only. The workflow must not add repository secrets, registry authentication, image publication, or live database addresses.

Both the marker step and pytest receive a loopback DSN with `sslmode=require`. Existing database and cluster markers remain mandatory, so tests cannot mistake an unmarked database for a disposable target.

### Platform partition

Add one bounded Windows matrix packet containing exactly:

- `tests/test_load_canary_home_spend_cap.py::test_native_exit_guard_stops_later_steps_after_a_failed_command`
- `tests/test_mapped_role_deployment_contract.py::test_hba_post_write_failure_restores_hba_and_outputs_byte_exact`

The Linux pytest command deselects those exact nodes. Entire files are not excluded, preserving portable coverage from the rest of each file.

The POSIX descriptor test remains in Linux. Its test harness will patch a copied capability set so the tracking wrapper for `os.open` retains the original function's `dir_fd` capability. Production rollback code remains unchanged.

### Evidence completeness

Set `strategy.fail-fast: false` on the Linux Python matrix. Python 3.11 and 3.12 must both finish, ensuring a failure on one interpreter cannot erase evidence for the other.

### Regression guards

Extend the static workflow contract tests to assert:

- the TLS configuration/probe step exists;
- CI DSNs use `sslmode=require`;
- the two Windows-native nodes are present in the Windows packet and deselected from Linux;
- the Linux matrix has `fail-fast: false`;
- disposable database and cluster markers remain present.

The descriptor test itself is the regression proof for the capability-preserving harness adjustment.

## Verification Sequence

1. Run the static workflow contract tests and YAML parsing checks locally.
2. Run the two Windows-native nodes on Windows.
3. Run all focused tests affected by the repair; record skips explicitly where the local OS cannot exercise POSIX behavior.
4. Commit the focused repair and verify commit, tree, parent, cleanliness, and diff scope.
5. Push the new exact branch head and dispatch the fork CI workflow.
6. Require all Linux 3.11/3.12, Windows, lint, package-build, and container-build jobs to complete successfully.
7. Run independent spec-compliance and code-quality reviews concurrently against the immutable commit.
8. Generate a new bundle and signed exact-head checkpoint containing commit, tree, parent, bundle hash, test outputs, review verdicts, and remaining blockers.

Prior exact-head evidence for `3f9ec16d` remains historical evidence but is not final proof for the new commit.

## Safety and Rollback

- Do not connect to or mutate the live database.
- Do not begin cutover.
- Do not weaken TLS or disposable-database checks.
- Do not modify runtime application behavior for these CI failures.
- If TLS service configuration fails, preserve the failed logs and revert only the focused CI/test commit; the current `3f9ec16d` bundle and checkpoint remain intact.
- The upstream PR stays draft until the exact repaired head has green CI and completed reviews. Upstream workflow approval may still require a repository administrator.

## Acceptance Gates

The repair is acceptable only when:

- the diff is limited to workflow/test infrastructure described above;
- static regression tests pass;
- both supported Linux Python jobs finish green against TLS-enabled PostgreSQL 18;
- all bounded Windows jobs, lint, package build, and container build pass;
- independent reviewers return GO on the same immutable head;
- the new signed checkpoint verifies; and
- live cutover remains explicitly blocked pending its separate operational gates.
