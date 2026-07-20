# Mac Offsite-Apply Worker — Owner Runbook

Adds a macOS machine (different LAN, intermittently on) as an offsite-ATS apply worker
(`mac-0`). Design: `docs/superpowers/specs/2026-07-03-mac-fleet-worker-design.md`.
LinkedIn NEVER runs on this machine (lane B stays on the home IP).

## A. Home box (one-time, ~15 min)

1. **Tailscale**: install + sign in on the home box (skip if `tailscale ip -4` already
   prints a 100.x address).
2. **Decide the branch the Mac runs.** The Mac pulls `applypilot-private` — push the
   branch you want live (currently `applypilot-hardening-and-brainstorm-integration`;
   use `main` only if the fleet code you run is merged there):
   ```powershell
   git push private applypilot-hardening-and-brainstorm-integration
   ```
3. **Harden PG for the tailnet** (elevated PowerShell, repo root). Follow the
   authenticated mapped-role onboarding command in
   `docs/railway-canonical-worker-deployment.md`: use the offline break-glass DSN,
   operator-held rollback HMAC key, reviewed regrant manifest, enrolled node ID,
   and a unique mapped login role. The setup prompts securely for that role's NEW
   SCRAM password — save it for step C. Never use the shared `fleet_worker` role:
   ```powershell
   .\setup-fleet-pg-tailscale.ps1 -NodeId mac-0 -Contract apply `
     -Role mac_0_apply -RegrantManifest .\evidence\reviewed-regrant-manifest.json
   ```
   It prints the Home Tailscale IP + DSN + pgpass line the Mac setup will ask for.
4. **Push profile + resume into the fleet_assets table** (so the Mac hydrates them from
   PG — nothing PII transits OneDrive/AirDrop):
   ```powershell
   .\.conda-env\python.exe -c "from applypilot.apply import pgqueue; import pathlib; h=pathlib.Path.home()/'.applypilot'; c=pgqueue.connect('host=localhost port=5432 dbname=applypilot_fleet user=postgres connect_timeout=5'); pgqueue.put_asset(c,'profile.json',(h/'profile.json').read_bytes()); pgqueue.put_asset(c,'resume.pdf',(h/'resume.pdf').read_bytes()); c.close(); print('assets pushed')"
   ```
   (Adjust the two paths if your profile/resume live elsewhere.)

## B. GitHub (one-time, ~2 min)

The Mac authenticates with a READ-ONLY deploy key it generates during setup. When the
setup script pauses and shows a public key: GitHub → `thefulmination/applypilot-private`
→ Settings → Deploy keys → *Add deploy key* → paste → leave **Allow write access
UNCHECKED** → Add.

## C. The Mac (one-time, ~20 min)

1. Install Tailscale (App Store or https://tailscale.com/download) and sign in to YOUR
   tailnet.
2. Copy the single file `setup-mac-worker.sh` to the Mac (AirDrop/USB is fine — it
   contains no secrets) and run:
   ```bash
   bash setup-mac-worker.sh
   ```
   Prompts, in order: deploy-key pause (step B) → branch → home Tailscale IP →
   `fleet_worker` PG password → `ANTHROPIC_API_KEY` → `DEEPSEEK_API_KEY`.
   Keys land ONLY in `~/applypilot-fleet/.applypilot/fleet-worker.env` and
   `~/.pgpass`, both `chmod 600`.
3. Sanity: `launchctl list | grep applypilot` shows the agent;
   `tail -f ~/applypilot-fleet/logs/wrapper.log ~/applypilot-fleet/.applypilot/logs/worker-0.log`
   shows `worker started` (wrapper.log) and the live apply-agent transcript (worker-0.log).
   The wrapper sets a macOS `caffeinate` assertion by default so idle sleep does not
   silently remove the worker from Tailscale; set `APPLYPILOT_MAC_CAFFEINATE=0` in
   `~/applypilot-fleet/.applypilot/fleet-worker.env` only if you intentionally want
   normal sleep behavior.

## D. Canary go-live (home box)

1. Fleet must be alive with approved work queued (if dormant: `.\register-fleet-tasks.ps1`
   and confirm `apply_queue` has `status='queued'` rows with `approved_batch` set).
2. Arm a small ATS canary so the fleet auto-pauses after 5 applies (PowerShell
   SINGLE-quoted so the inner python double quotes pass through verbatim):
   ```powershell
   .\.conda-env\python.exe -c 'from applypilot.apply import pgqueue; c=pgqueue.connect("host=localhost port=5432 dbname=applypilot_fleet user=postgres connect_timeout=5"); cur=c.cursor(); cur.execute("UPDATE fleet_config SET canary_enabled=TRUE, canary_remaining=5 WHERE id=1"); c.commit(); c.close(); print("canary armed: 5")'
   ```
3. Watch live — console `http://<home-tailscale-ip>:8787` (reachable tailnet-wide), or
   (the doubled `''` is a literal single quote inside a PS single-quoted string):
   ```powershell
   .\.conda-env\python.exe -c 'from applypilot.apply import pgqueue; c=pgqueue.connect("host=localhost port=5432 dbname=applypilot_fleet user=postgres connect_timeout=5"); cur=c.cursor(); cur.execute("SELECT worker_id, state, current_job, last_beat FROM worker_heartbeat WHERE worker_id LIKE ''mac-%''"); [print(r) for r in cur.fetchall()]; c.close()'
   ```
4. Verify the canary applies landed (`apply_queue.status='applied'` with
   `worker_id='mac-0'`), then disarm the canary / raise caps as usual.

## E. How updates reach the Mac

Push to the pinned branch on `private`; within `UPDATE_CHECK_SECONDS` (6 h, or next Mac
boot) the wrapper fetches, SIGTERMs the worker (it FINISHES the current job — Task-1
handler — then exits), hard-resets to origin, and restarts. `pip install -e .` re-runs
only when `pyproject.toml` changed. Force it immediately: on the Mac,
`launchctl kickstart -k gui/$(id -u)/com.applypilot.fleetworker`.

## F. Kill switch (all owner-side, no access to the Mac needed)

1. Rotate the API keys (Anthropic + DeepSeek consoles).
2. Rotate the PG credential: re-run `.\setup-fleet-pg-tailscale.ps1` with a new password.
3. Remove the Mac from the tailnet (Tailscale admin console).

## G. Troubleshooting

| Symptom | Check |
|---|---|
| No heartbeat row for `mac-0` | Mac on? `launchctl list \| grep applypilot`; `logs/wrapper.log`; also check `logs/wrapper.err.log` (launchd load failures surface ONLY there) |
| `FATAL: no Playwright chromium` in wrapper.log | Re-run `PLAYWRIGHT_BROWSERS_PATH=~/applypilot-fleet/.playwright-browsers ~/applypilot-fleet/.venv/bin/python -m playwright install chromium` |
| PG connect fails | `tailscale ping <home-ts-ip>` from the Mac; pg_hba rule + firewall rule on home box (`setup-fleet-pg-tailscale.ps1` re-run is safe) |
| Mac disappears from Tailscale while otherwise healthy | Confirm `APPLYPILOT_MAC_CAFFEINATE=1` in `~/applypilot-fleet/.applypilot/fleet-worker.env`; the wrapper should log `caffeinate active`. This prevents idle sleep but cannot keep a powered-off or closed-lid Mac online |
| Applies fail with missing profile/resume | Re-run runbook A.4 (asset push), then on the Mac delete `~/applypilot-fleet/.applypilot/profile.json` and re-run `setup-mac-worker.sh` step or copy manually |
| Worker never updates | `git -C ~/applypilot-fleet fetch origin <branch>` by hand — deploy key revoked? branch deleted? |
| Email-verification jobs fail/park | The Mac now uses the fleet OTP relay (`APPLYPILOT_INBOX_AUTH_MODE=relay`). Ensure `applypilot-fleet-otp-home` is running on the home box — see docs/fleet-otp-relay-runbook.md. |
| Worker throttled oddly / shares limits | `wrapper.log` "egress=" line shows `0.0.0.0` → the boot-time `curl api.ipify.org` failed and the Mac fell into the shared 0.0.0.0 governor bucket; restart the agent (`launchctl kickstart -k gui/$(id -u)/com.applypilot.fleetworker`) once the network is up |
