# Runtime Matrix Semantic Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind Runtime launcher and scheduled-task semantics to committed source and fail closed on unsupported PowerShell service generators.

**Architecture:** Extend committed-tree discovery in `scripts/brain_command_authority.py`; keep all bindings deterministic and repository-relative. Regenerate the checked-in matrix only after focused hostile tests pass.

**Tech Stack:** Python 3.12, Git committed-tree plumbing, pytest, canonical JSON.

---

### Task 1: Add failing semantic-binding tests

**Files:**
- Modify: `tests/test_brain_command_authority_matrix.py`

- [ ] Add tests that mutate a committed launcher blob and assert coverage reports a stale and unclassified row.
- [ ] Add static and dynamic scheduled-task fixtures and assert executable, arguments, principal, and explicit dynamic markers are bound.
- [ ] Add duplicate/ambiguous field fixtures and assert `AuthorityError`.
- [ ] Add PowerShell and alternative shell service-generator fixtures and assert unsupported-generator failure.
- [ ] Add comment-suppression and exact activation-token regressions.
- [ ] Add a relative `--matrix` CLI test.
- [ ] Run the new tests and confirm they fail for the intended missing behavior.

### Task 2: Implement committed semantic binding

**Files:**
- Modify: `scripts/brain_command_authority.py`
- Modify: `scripts/verify-brain-command-authority.py`

- [ ] Add committed-blob SHA-256 to generic file canonical targets.
- [ ] Parse unique static scheduled-task action and registration fields; emit deterministic dynamic markers for current legitimate dynamic expressions.
- [ ] Reject duplicate/ambiguous fields and unsupported service-generation forms.
- [ ] Resolve and contain repository and matrix paths before receipt generation.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Regenerate and verify the exact matrix

**Files:**
- Modify: `config/brain-command-authority-matrix.runtime.json`

- [ ] Regenerate discovered canonical targets while preserving reviewed classification fields by exact surface ID.
- [ ] Run matrix tests, inventory check, exact verifier, Ruff, compileall, `git diff --check`, and cleanliness checks.
- [ ] Commit the bounded repair, create an exact-head Git bundle, verify it, and record SHA-256/head/tree/scope.
