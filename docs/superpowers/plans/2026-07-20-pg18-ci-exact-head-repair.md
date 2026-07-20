# PostgreSQL 18 Exact-Head CI Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the PostgreSQL 18 GitHub CI environment and platform partition so a new immutable migration candidate can obtain trustworthy exact-head verification without changing runtime behavior or touching the live database.

**Architecture:** Keep the official `postgres:18` service, configure ephemeral TLS inside that disposable service before tests, and prove SSL is active before applying the existing disposable markers. Route two exact Windows-native nodes to the existing Windows matrix, retain the POSIX descriptor test on Linux with a capability-preserving harness, and freeze the result behind full CI, parallel reviews, a bundle, and a signed checkpoint.

**Tech Stack:** GitHub Actions YAML, PostgreSQL 18 service containers, Bash and Docker CLI, Python 3.11/3.12, pytest, PyYAML, PowerShell 7, Git/GitHub CLI, OpenSSL Ed25519.

---

## File Map

- Modify `.github/workflows/ci.yml`: enable and probe TLS, require TLS in CI DSNs, partition two Windows-native nodes, and disable Linux matrix fail-fast.
- Modify `tests/test_container_build_contract.py`: statically lock the TLS setup, platform partition, disposable markers, and complete Python matrix evidence.
- Modify `tests/test_pg_role_bootstrap.py`: preserve `supports_dir_fd` capability while wrapping `os.open` in the POSIX descriptor test.
- Create no runtime source files and modify nothing under `src/` or the operational migration scripts.

### Task 1: Add failing static CI-contract tests

**Files:**
- Modify: `tests/test_container_build_contract.py:127-155`
- Test: `tests/test_container_build_contract.py`

- [ ] **Step 1: Add exact node constants near `CONTAINER_JOB_NAME`**

```python
WINDOWS_NATIVE_CI_NODES = (
    "tests/test_load_canary_home_spend_cap.py::"
    "test_native_exit_guard_stops_later_steps_after_a_failed_command",
    "tests/test_mapped_role_deployment_contract.py::"
    "test_hba_post_write_failure_restores_hba_and_outputs_byte_exact",
)
```

- [ ] **Step 2: Add a failing TLS contract test**

```python
def test_python_ci_tls_enables_and_probes_disposable_postgres() -> None:
    job = _load_workflow()["jobs"]["test"]
    steps = {step["name"]: step for step in job["steps"]}
    tls_step = steps["Enable TLS on disposable PostgreSQL service"]
    assert tls_step["env"]["POSTGRES_SERVICE_ID"] == "${{ job.services.postgres.id }}"
    tls_script = tls_step["run"]
    assert "openssl req -new -x509" in tls_script
    assert "ssl = on" in tls_script
    assert "docker restart" in tls_script
    assert "pg_isready" in tls_script

    marker = steps["Mark disposable PostgreSQL service"]
    assert "sslmode=require" in marker["env"]["APPLYPILOT_PGTEST_DSN"]
    assert 'connection.execute("SHOW ssl")' in marker["run"]
    assert "PostgreSQL service did not enable TLS" in marker["run"]

    test_environment = steps["Test"]["env"]
    assert "sslmode=require" in test_environment["APPLYPILOT_PGTEST_DSN"]
```

- [ ] **Step 3: Add a failing platform-partition and evidence-completeness test**

```python
def test_python_ci_routes_exact_windows_nodes_and_finishes_both_versions() -> None:
    workflow = _load_workflow()
    windows_job = workflow["jobs"]["phase-a-windows-file"]
    windows_packets = windows_job["strategy"]["matrix"]["include"]
    repair_packet = next(
        packet for packet in windows_packets if packet["name"] == "pg18-ci-repair-contracts"
    )
    for node in WINDOWS_NATIVE_CI_NODES:
        assert node in repair_packet["tests"]

    linux_job = workflow["jobs"]["test"]
    assert linux_job["strategy"]["fail-fast"] is False
    steps = {step["name"]: step for step in linux_job["steps"]}
    test_command = steps["Test"]["run"]
    for node in WINDOWS_NATIVE_CI_NODES:
        assert f"--deselect={node}" in test_command
```

- [ ] **Step 4: Run the new contract tests and verify red state**

Run:

```powershell
python -m pytest -q tests/test_container_build_contract.py -k "tls_enables or routes_exact"
```

Expected: two failures because the TLS step, Windows packet, Linux deselections, and `fail-fast: false` do not yet exist.

### Task 2: Repair the GitHub Actions service and platform partition

**Files:**
- Modify: `.github/workflows/ci.yml:44-168`
- Test: `tests/test_container_build_contract.py`

- [ ] **Step 1: Add the bounded Windows repair packet**

Append this item to `jobs.phase-a-windows-file.strategy.matrix.include`:

```yaml
          - name: pg18-ci-repair-contracts
            tests: >-
              tests/test_load_canary_home_spend_cap.py::test_native_exit_guard_stops_later_steps_after_a_failed_command
              tests/test_mapped_role_deployment_contract.py::test_hba_post_write_failure_restores_hba_and_outputs_byte_exact
```

- [ ] **Step 2: Preserve complete Linux matrix evidence**

Change the Linux `test` strategy to:

```yaml
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]
```

- [ ] **Step 3: Add TLS configuration immediately after lint**

```yaml
      - name: Enable TLS on disposable PostgreSQL service
        env:
          POSTGRES_SERVICE_ID: ${{ job.services.postgres.id }}
        run: |
          set -euo pipefail
          docker exec --user root "$POSTGRES_SERVICE_ID" sh -ceu '
            test -x /usr/bin/openssl
            openssl req -new -x509 -nodes -days 1 -subj "/CN=localhost" \
              -keyout "$PGDATA/server.key" -out "$PGDATA/server.crt"
            chown postgres:postgres "$PGDATA/server.key" "$PGDATA/server.crt"
            chmod 600 "$PGDATA/server.key"
            printf "\nssl = on\nssl_cert_file = '\''server.crt'\''\nssl_key_file = '\''server.key'\''\n" \
              >> "$PGDATA/postgresql.conf"
          '
          docker restart "$POSTGRES_SERVICE_ID"
          for attempt in $(seq 1 30); do
            if docker exec "$POSTGRES_SERVICE_ID" pg_isready -U postgres -d postgres; then
              exit 0
            fi
            if [ "$attempt" -eq 30 ]; then
              docker logs "$POSTGRES_SERVICE_ID"
              exit 1
            fi
            sleep 1
          done
```

The step uses only the disposable service container ID. It must not reference repository secrets, external hosts, registries, or production DSNs.

- [ ] **Step 4: Require and prove TLS before writing disposable markers**

Change both CI DSNs to:

```yaml
          APPLYPILOT_PGTEST_DSN: postgresql://postgres:applypilot-ci-test-only@127.0.0.1:5432/postgres?sslmode=require
```

Immediately after `with psycopg.connect(dsn) as connection:` in the marker script, add:

```python
              ssl_setting = connection.execute("SHOW ssl").fetchone()[0]
              if ssl_setting != "on":
                  raise RuntimeError("PostgreSQL service did not enable TLS")
```

- [ ] **Step 5: Deselect only the two Windows-native nodes from Linux**

Append to the Linux pytest command:

```yaml
          --deselect=tests/test_load_canary_home_spend_cap.py::test_native_exit_guard_stops_later_steps_after_a_failed_command
          --deselect=tests/test_mapped_role_deployment_contract.py::test_hba_post_write_failure_restores_hba_and_outputs_byte_exact
```

- [ ] **Step 6: Run the static workflow tests and verify green state**

Run:

```powershell
python -m pytest -q tests/test_container_build_contract.py
```

Expected: all tests in `tests/test_container_build_contract.py` pass.

- [ ] **Step 7: Commit the workflow contract repair**

```powershell
git add -- .github/workflows/ci.yml tests/test_container_build_contract.py
git diff --cached --check
git commit -m "ci: align pg18 exact-head test environment"
```

Expected: a commit containing only the workflow and its static regression test.

### Task 3: Repair the POSIX descriptor test harness

**Files:**
- Modify: `tests/test_pg_role_bootstrap.py:1456-1501`
- Test: `tests/test_pg_role_bootstrap.py::test_hba_target_descriptor_stays_open_through_replace`

- [ ] **Step 1: Preserve the original capability while wrapping `os.open`**

After defining `tracking_open` and before monkeypatching `module.os.open`, add:

```python
    tracked_supports_dir_fd = set(module.os.supports_dir_fd)
    if real_open in tracked_supports_dir_fd:
        tracked_supports_dir_fd.remove(real_open)
        tracked_supports_dir_fd.add(tracking_open)
```

Then add this monkeypatch immediately before the existing `os.open` monkeypatch:

```python
    monkeypatch.setattr(module.os, "supports_dir_fd", tracked_supports_dir_fd)
```

Do not change `scripts/rollback-fleet-pg-role.py`; its capability check is correct for production.

- [ ] **Step 2: Run the exact test on the available platform**

Run:

```powershell
python -m pytest -q tests/test_pg_role_bootstrap.py::test_hba_target_descriptor_stays_open_through_replace
```

Expected on Windows: one explicit platform skip. Expected on POSIX: one pass. The GitHub Linux jobs are the authoritative POSIX execution gate.

- [ ] **Step 3: Run nearby descriptor and replacement tests**

Run:

```powershell
python -m pytest -q tests/test_pg_role_bootstrap.py -k "replace_atomic or descriptor_stays_open or immediately_before_atomic_exchange"
```

Expected: all applicable tests pass; Windows-only or POSIX-only nodes may report explicit skips.

- [ ] **Step 4: Commit the test-harness repair**

```powershell
git add -- tests/test_pg_role_bootstrap.py
git diff --cached --check
git commit -m "test: preserve posix dir-fd capability under tracing"
```

Expected: a test-only commit with no production changes.

### Task 4: Freeze and verify the local repair head

**Files:**
- Verify: `.github/workflows/ci.yml`
- Verify: `tests/test_container_build_contract.py`
- Verify: `tests/test_pg_role_bootstrap.py`

- [ ] **Step 1: Run the full focused local packet**

```powershell
python -m pytest -q `
  tests/test_container_build_contract.py `
  tests/test_load_canary_home_spend_cap.py::test_native_exit_guard_stops_later_steps_after_a_failed_command `
  tests/test_mapped_role_deployment_contract.py::test_hba_post_write_failure_restores_hba_and_outputs_byte_exact `
  tests/test_pg_role_bootstrap.py::test_hba_target_descriptor_stays_open_through_replace
```

Expected on the local Windows host: static contracts and both Windows nodes pass; the POSIX node skips explicitly.

- [ ] **Step 2: Verify scope, formatting, ancestry, and cleanliness**

```powershell
git diff 3f9ec16db91acd0cc4293b5a4fc4d6d7c3a9ae2a..HEAD --name-status
git diff 3f9ec16db91acd0cc4293b5a4fc4d6d7c3a9ae2a..HEAD --check
git merge-base --is-ancestor 3f9ec16db91acd0cc4293b5a4fc4d6d7c3a9ae2a HEAD
git fsck --no-dangling
git status --porcelain=v2
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
git rev-parse 'HEAD^'
```

Expected: only the approved design, plan, workflow, and two test files differ; checks exit zero; the worktree is clean; exact commit/tree/parent are recorded.

### Task 5: Obtain authoritative exact-head GitHub CI

**Files:**
- External verification only; no new repository edits.

- [ ] **Step 1: Push the explicit repaired branch ref**

```powershell
$head = (git rev-parse HEAD).Trim()
git push github-myfork "$head`:refs/heads/codex/runtime-critical-integration-20260719"
$remote = (git ls-remote github-myfork refs/heads/codex/runtime-critical-integration-20260719).Split()[0]
if ($remote -ne $head) { throw "remote head mismatch: $remote != $head" }
```

Expected: the fork branch resolves to the exact frozen local commit.

- [ ] **Step 2: Dispatch fork CI for the repaired head**

```powershell
gh workflow run ci.yml --repo thefulmination/ApplyPilot --ref codex/runtime-critical-integration-20260719
Start-Sleep -Seconds 5
gh run list --repo thefulmination/ApplyPilot --workflow CI `
  --branch codex/runtime-critical-integration-20260719 --event workflow_dispatch --limit 1 `
  --json databaseId,headSha,status,conclusion,url
```

Expected: the newest run's `headSha` equals the frozen local commit.

- [ ] **Step 3: Watch the exact run to completion**

```powershell
$run = gh run list --repo thefulmination/ApplyPilot --workflow CI `
  --branch codex/runtime-critical-integration-20260719 --event workflow_dispatch --limit 1 `
  --json databaseId,headSha | ConvertFrom-Json
if ($run.headSha -ne (git rev-parse HEAD).Trim()) { throw "CI head mismatch" }
gh run watch $run.databaseId --repo thefulmination/ApplyPilot --exit-status
gh run view $run.databaseId --repo thefulmination/ApplyPilot `
  --json headSha,status,conclusion,jobs,url
```

Expected: Linux 3.11 and 3.12, all Windows packets, lint, Python 3.12 package build, and container build complete successfully. Do not treat a cancelled, skipped-required, pending, or approval-gated run as green.

- [ ] **Step 4: Record upstream PR status without bypassing approval**

```powershell
gh pr view 89 --repo Pickle-Pixel/ApplyPilot `
  --json url,isDraft,headRefOid,mergeable,statusCheckRollup
```

Expected: PR head equals the repaired commit. The PR remains draft; an upstream administrator may still need to approve the fork workflow.

### Task 6: Review, bundle, and sign the exact-head checkpoint

**Files:**
- Create generated evidence under `C:\CodexBundles\applypilot-recovery-20260719\evidence\checkpoints\`
- Create bundle under `C:\CodexBundles\applypilot-recovery-20260719\`

- [ ] **Step 1: Dispatch two independent immutable-head reviews concurrently**

Give one reviewer the approved design plus exact commit/tree and ask for line-by-line spec compliance. Give the other the exact commit/tree and ask for CI, security-boundary, test-quality, and cross-platform review. Both reviewers must remain read-only and return GO or actionable findings tied to exact paths and lines.

Expected: both reviews bind their verdict to the same commit and tree. Agent claims alone are not proof; independently inspect the reviewed diff and any cited test output.

- [ ] **Step 2: Create and verify the exact branch bundle**

```powershell
$head = (git rev-parse HEAD).Trim()
$short = $head.Substring(0, 8)
$bundle = "C:\CodexBundles\applypilot-recovery-20260719\runtime-integrated-release-$short.bundle"
git bundle create $bundle codex/runtime-critical-integration-20260719
git bundle verify $bundle
$bundleHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $bundle).Hash
$bundleBytes = (Get-Item -LiteralPath $bundle).Length
```

Expected: bundle verification exits zero and the hash/byte count are recorded.

- [ ] **Step 3: Generate a deterministic checkpoint JSON from verified values**

Run this only after both immutable-head reviewers returned `GO` and the exact CI run concluded successfully:

```powershell
$head = (git rev-parse HEAD).Trim()
$tree = (git rev-parse 'HEAD^{tree}').Trim()
$parent = (git rev-parse 'HEAD^').Trim()
$short = $head.Substring(0, 8)
$bundle = "C:\CodexBundles\applypilot-recovery-20260719\runtime-integrated-release-$short.bundle"
$bundleHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $bundle).Hash
$bundleBytes = (Get-Item -LiteralPath $bundle).Length
$ci = gh run list --repo thefulmination/ApplyPilot --workflow CI `
  --branch codex/runtime-critical-integration-20260719 --event workflow_dispatch --limit 1 `
  --json databaseId,headSha,status,conclusion,url | ConvertFrom-Json
if ($ci.headSha -ne $head -or $ci.status -ne 'completed' -or $ci.conclusion -ne 'success') {
  throw 'exact-head CI is not complete and successful'
}
$specVerdict = 'GO'
$qualityVerdict = 'GO'
$checkpointDir = 'C:\CodexBundles\applypilot-recovery-20260719\evidence\checkpoints'
$checkpoint = Join-Path $checkpointDir "pg18-ci-repair-$short-checkpoint.json"
$checkpointData = [ordered]@{
  schemaVersion = 1
  checkpointId = "applypilot-runtime-pg18-ci-repair-$short"
  createdUtc = [DateTime]::UtcNow.ToString('o')
  publicationVerdict = 'GO_FOR_DRAFT_PR_ONLY'
  liveCutoverVerdict = 'NO-GO'
  liveCutoverAuthorized = $false
  repair = [ordered]@{
    commit = $head
    tree = $tree
    parent = $parent
    clean = ((git status --porcelain=v2) -join '') -eq ''
    runtimeSourceChanged = $false
  }
  bundle = [ordered]@{
    path = $bundle
    bytes = $bundleBytes
    sha256 = $bundleHash
  }
  verification = [ordered]@{
    focusedLocal = 'passed'
    forkCiRunId = [long]$ci.databaseId
    forkCiUrl = $ci.url
    forkCiConclusion = $ci.conclusion
    linuxPython311 = 'success'
    linuxPython312 = 'success'
    windowsPackets = 'success'
    containerBuild = 'success'
  }
  reviews = [ordered]@{
    specCompliance = $specVerdict
    codeQuality = $qualityVerdict
  }
  remainingBlockers = @(
    'Upstream fork-workflow approval may still require a repository administrator.'
    'Live production backup, identity, freeze, delta, parity, rollback, and canary gates remain unproved.'
  )
}
if (-not $checkpointData.repair.clean) { throw 'worktree is not clean' }
$json = ($checkpointData | ConvertTo-Json -Depth 8) + [Environment]::NewLine
[IO.File]::WriteAllText($checkpoint, $json, [Text.UTF8Encoding]::new($false))
```

Expected: the checkpoint is derived from the current Git objects, verified bundle, and successful exact-head run; UTF-8 has no BOM and ends with one newline.

- [ ] **Step 4: Sign and verify with a fresh restricted Ed25519 checkpoint key**

```powershell
$checkpointDir = 'C:\CodexBundles\applypilot-recovery-20260719\evidence\checkpoints'
$checkpoint = Join-Path $checkpointDir "pg18-ci-repair-$short-checkpoint.json"
$signature = "$checkpoint.sig"
$receipt = "$checkpoint.receipt.json"
$private = Join-Path $env:TEMP "applypilot-$short-ed25519-private.pem"
$public = Join-Path 'C:\CodexBundles\applypilot-recovery-20260719\evidence\signing' "pg18-ci-repair-$short-public-key.pem"
try {
  openssl genpkey -algorithm ED25519 -out $private
  $principal = "$env:USERDOMAIN\$env:USERNAME"
  icacls $private /inheritance:r /grant:r "$principal`:(R,W)"
  openssl pkey -in $private -pubout -out $public
  openssl pkeyutl -sign -rawin -inkey $private -in $checkpoint -out $signature
  openssl pkeyutl -verify -rawin -pubin -inkey $public -in $checkpoint -sigfile $signature
  if ($LASTEXITCODE -ne 0) { throw 'checkpoint signature verification failed' }
  $receiptData = [ordered]@{
    schemaVersion = 1
    checkpointPath = $checkpoint
    checkpointBytes = (Get-Item -LiteralPath $checkpoint).Length
    checkpointSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $checkpoint).Hash
    signaturePath = $signature
    signatureBytes = (Get-Item -LiteralPath $signature).Length
    signatureSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $signature).Hash
    signatureAlgorithm = 'Ed25519'
    publicKeyPath = $public
    publicKeySha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $public).Hash
    verification = 'passed'
  }
  $receiptJson = ($receiptData | ConvertTo-Json -Depth 4) + [Environment]::NewLine
  [IO.File]::WriteAllText($receipt, $receiptJson, [Text.UTF8Encoding]::new($false))
} finally {
  if (Test-Path -LiteralPath $private) {
    Remove-Item -LiteralPath $private -Force
  }
}
```

Expected: verification reports `Signature Verified Successfully`; the temporary private key is removed; the public key, signature, and hash-bound receipt remain with the evidence.

- [ ] **Step 5: Verify checkpoint completeness and final exact state**

```powershell
if (Select-String -LiteralPath $checkpoint -Pattern '"forkCiRunId": 0|"bytes": 0|"clean": false') {
  throw 'checkpoint contains an invalid sentinel or dirty state'
}
$checkpointHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $checkpoint).Hash
$signatureHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $signature).Hash
if ((Get-Content -Raw -LiteralPath $receipt | ConvertFrom-Json).verification -ne 'passed') {
  throw 'checkpoint receipt does not record successful verification'
}
git status --porcelain=v2
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
gh pr view 89 --repo Pickle-Pixel/ApplyPilot --json headRefOid,isDraft,url
```

Expected: no invalid sentinels, clean repository, PR head equal to the signed commit, and the PR still draft. Report the checkpoint path/hash, signature hash, bundle path/hash, CI URL, reviewer verdicts, and remaining blockers. Do not authorize or begin live cutover.
