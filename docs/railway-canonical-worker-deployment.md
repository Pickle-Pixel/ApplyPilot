# Railway Canonical Worker Deployment

The Railway worker runs the canonical v3 `applypilot-fleet-apply` executable. The
retired `applypilot.apply.container_worker` must never be used.

## Required service configuration

Set these Railway variables before deploying:

- `FLEET_PG_DSN`: a sealed DSN for this replica's unique mapped login role. It
  must not be the Railway-generated admin reference, an admin URL, a `postgres` login, or
  the shared `fleet_worker` login. Generate one mapped role per node and build
  the DSN with the Railway private host plus that role's rotated SCRAM secret.
- `DEEPSEEK_API_KEY`: sealed service variable used by the local LiteLLM proxy.
- `APPLYPILOT_WORKER_ID`: unique stable worker identity. Do not reuse one value
  across replicas.
- `APPLYPILOT_WORKER_CONTRACT=apply`: contract stored beside the login role in
  `fleet_worker_principals`.
- `APPLYPILOT_RELEASE_VERSION`: exact value pinned in
  `fleet_config.pinned_worker_version`, formatted as
  `0.3.0+git.tree.<7-lowercase-hex>`.

Optional variables:

- `FLEET_MACHINE_OWNER` and `APPLYPILOT_FLEET_LABEL`, both defaulting to
  `railway`. When either is set, set both to the same value.
- `FLEET_HOME_IP`: egress identity recorded in telemetry; defaults to `railway`.
- `APPLYPILOT_APPLY_AGENT`: defaults to `claude`.
- `APPLYPILOT_APPLY_MODEL`: defaults to `deepseek-chat`.
- `APPLYPILOT_FALLBACK_AGENT`: optional comma-separated fallback chain.
- `APPLYPILOT_CHROME_SLOT`: required when multiple workers share a filesystem or
  browser namespace.

Attach a persistent volume at `/data/applypilot` containing non-empty
`profile.json` and `resume.pdf`. The entrypoint refuses to start without them.

Every fleet DSN must contain an explicit password. ApplyPilot does not consult
`.pgpass`, password files, service files, or ambient libpq connection variables.
`FLEET_PG_DSN` is the sole accepted DSN environment variable.

At startup the entrypoint parses the DSN and rejects missing usernames,
`postgres`, and `fleet_worker`. It then connects and requires `session_user` and
`current_user` to be identical to the unique mapped login role, with the exact
`APPLYPILOT_WORKER_ID` and `APPLYPILOT_WORKER_CONTRACT` mapping. A mismatch exits
before LiteLLM or the worker starts.
The service also refuses and unsets generic, admin, controller, and ambient
libpq database variables before any child process starts. A worker environment
containing one of those variables is a deployment error, even when
`FLEET_PG_DSN` is otherwise valid.

## Controller migration and database ACLs

Use the Railway admin reference `APPLYPILOT_ADMIN_PG_DSN=${{Postgres.DATABASE_URL}}`
only for the one-time bootstrap. The bootstrap session user, normally
`postgres`, must be named in `infrastructure_superuser_roles`. It remains
`SUPERUSER LOGIN` as the provider's offline break-glass authority and is an
explicit member of the effective database CONNECT allowlist. This design does
not claim to eliminate PostgreSQL's superuser bypass; it isolates the exception.
Store that credential offline, do not set its DSN in controller or worker
services, and use it only for bootstrap or execution of the prewritten rollback.

Bootstrap creates the dedicated `LOGIN NOINHERIT NOSUPERUSER` database owner
(the canonical import login), LOGIN/NOSUPERUSER controller, NOLOGIN verifier,
NOLOGIN migrator, and the fixed `fleet_backup` LOGIN topology. The database
owner can activate only the exact migrator and verifier roles through OID-10-
granted `SET`-only memberships; it receives no standing inherited authority.
Its independent password is accepted only through
`APPLYPILOT_DATABASE_OWNER_PG_PASSWORD`, converted client-side to a SCRAM-SHA-256
verifier, and rotated in the same atomic transaction. The separately generated
fleet-migration LOGIN password is supplied only as
`APPLYPILOT_FLEET_MIGRATOR_PG_PASSWORD` and receives the same client-side SCRAM
handling. `fleet_backup` is
`NOINHERIT NOSUPERUSER NOCREATEDB
NOCREATEROLE NOREPLICATION NOBYPASSRLS CONNECTION LIMIT 1`; it has no inbound
or outbound memberships and owns no object. It receives only database CONNECT,
schema USAGE, table/view/materialized-view SELECT, sequence SELECT, and explicit
`pg_catalog.pg_control_system()` EXECUTE for target-identity receipts. Per-database
settings pin `search_path=pg_catalog, pg_temp` and
`default_transaction_read_only=on`. Bootstrap also installs global migrator
default ACLs for future schema USAGE and table/sequence SELECT. It strips LOGIN,
SUPERUSER, BYPASSRLS, CREATEROLE, CREATEDB, and REPLICATION from separately
named legacy roles in `retired_admin_roles`; the infrastructure break-glass role
must never appear in that list. After bootstrap, delete
`APPLYPILOT_ADMIN_PG_DSN`, `APPLYPILOT_CONTROLLER_PG_PASSWORD`,
`APPLYPILOT_BACKUP_PG_PASSWORD`, and
`APPLYPILOT_DATABASE_OWNER_PG_PASSWORD`, and
`APPLYPILOT_FLEET_MIGRATOR_PG_PASSWORD` from the shell. Use
`APPLYPILOT_CONTROLLER_PG_DSN` only for bounded controller operations. The
one-shot authority migration has a separate authenticated boundary: its target
DSN must log in as `applypilot_fleet_migrator_login`, which has one SET-only edge
to the NOLOGIN `applypilot_fleet_migrator` capability. Never substitute the
controller, database owner, provider admin, or another role in that DSN, and
never expose any migration or controller credential to a worker service.

Example one-time bootstrap:

```powershell
$env:APPLYPILOT_ADMIN_PG_DSN = '<provider-admin-url>'
$env:APPLYPILOT_CONTROLLER_PG_PASSWORD = '<new-scram-secret>'
$env:APPLYPILOT_BACKUP_PG_PASSWORD = '<independent-backup-scram-secret>'
$env:APPLYPILOT_DATABASE_OWNER_PG_PASSWORD = '<independent-canonical-import-scram-secret>'
$env:APPLYPILOT_FLEET_MIGRATOR_PG_PASSWORD = '<independent-one-shot-migrator-scram-secret>'
$env:APPLYPILOT_ROLLBACK_HMAC_KEY_HEX = '<operator-held-key-as-at-least-64-hex-characters>'
$env:APPLYPILOT_ROLLBACK_HMAC_KEY_ID = '<operator-key-id>'
python scripts/bootstrap-fleet-pg-roles.py `
  --database-owner-role applypilot_database_owner `
  --controller-role applypilot_deployment_controller `
  --verifier-role brain_schema_verifier `
  --migrator-role brain_schema_migrator `
  --retired-admin-role legacy_fleet_admin `
  --infrastructure-superuser-role postgres `
  --receipt-path .\evidence\bootstrap-receipt.json `
  --rollback-sql .\evidence\bootstrap-rollback.sql
Remove-Item Env:APPLYPILOT_ADMIN_PG_DSN, Env:APPLYPILOT_CONTROLLER_PG_PASSWORD, Env:APPLYPILOT_BACKUP_PG_PASSWORD, Env:APPLYPILOT_DATABASE_OWNER_PG_PASSWORD, Env:APPLYPILOT_FLEET_MIGRATOR_PG_PASSWORD, Env:APPLYPILOT_ROLLBACK_HMAC_KEY_HEX, Env:APPLYPILOT_ROLLBACK_HMAC_KEY_ID
```

After the authenticated bootstrap or existing-topology upgrade receipt passes
all gates, configure the one-shot private authority migrator with these four
service variables:

```text
AUTHORITY_MIGRATOR_TARGET_DSN=postgresql://applypilot_fleet_migrator_login:<separate-secret>@<exact-private-host>/<database>?sslmode=require
AUTHORITY_MIGRATOR_EXPECTED_PRIVATE_HOST=<exact-private-host>
AUTHORITY_MIGRATOR_EXPECTED_DB=<database>
AUTHORITY_MIGRATOR_EXPECTED_SYSID=<pg_control-system-identifier>
```

`AUTHORITY_MIGRATOR_TARGET_DSN` is a secret and must stay in the service-variable
channel; never pass it on argv or write it to evidence. The host must exactly
match `AUTHORITY_MIGRATOR_EXPECTED_PRIVATE_HOST` and must be either a full
`*.railway.internal` name or a full Tailscale MagicDNS `*.*.ts.net` name. The DSN
must contain the fixed `applypilot_fleet_migrator_login` user, its explicit
password, and an explicit TLS mode of `require`, `verify-ca`, or `verify-full`.
It must not use `hostaddr`, libpq service/passfile indirection, connection
`options`, multiple hosts, a public hostname, or ambient `PG*` connection
variables.

Railway executes the pinned `railway.toml` start command. For an equivalent
operator-controlled one-shot invocation, keep the DSN in the environment and
pass only the environment-variable name and non-secret identity pins:

```powershell
python scripts/run-private-authority-migrator.py `
  --target-dsn-env AUTHORITY_MIGRATOR_TARGET_DSN `
  --expected-private-host $env:AUTHORITY_MIGRATOR_EXPECTED_PRIVATE_HOST `
  --expected-database-name $env:AUTHORITY_MIGRATOR_EXPECTED_DB `
  --expected-system-identifier $env:AUTHORITY_MIGRATOR_EXPECTED_SYSID
```

The migrator validates `session_user` and `current_user` as the dedicated login
before reading target identity, acquiring a migration lock, using `SET ROLE`, or
executing DDL. Do not run it until the reviewed migration, rollback, and exact
target gates are complete.

Worker reconciliation has a strict rollout prerequisite. First complete either
the new-topology role bootstrap above or the existing-topology authority upgrade.
Then authenticate its committed receipt and confirm every role/authority gate,
including the complete `fleet_backup` boundary. Only after those gates pass may
an operator onboard, reconcile, or rotate a worker role. Worker reconciliation
full-validates the backup boundary before it generates rollback SQL, writes
durable evidence, or changes a role; a missing or drifted backup role fails with
`complete role bootstrap or existing-topology authority upgrade before worker
reconciliation`.

Before role hardening, prepare a reviewed JSON regrant manifest:

```json
{
  "database_owner_role": "applypilot_database_owner",
  "controller_roles": ["applypilot_deployment_controller"],
  "verifier_roles": ["brain_schema_verifier"],
  "retired_admin_roles": ["legacy_fleet_admin"],
  "backup_role": "fleet_backup",
  "migrator_role": "brain_schema_migrator",
  "infrastructure_superuser_roles": ["postgres"],
  "expected_service_roles": [],
  "regrants": [
    {
      "object_kind": "table",
      "qualified_name": "public.service_health",
      "privileges": ["SELECT"],
      "grantee": "applypilot_deployment_controller"
    }
  ]
}
```

`expected_service_roles` must exactly acknowledge other active database service
principals during the maintenance window. Every `regrants` item is parsed as a
structured object kind, qualified name, privilege allowlist, and approved
grantee. Raw SQL, comments, control characters, unknown fields, unsupported
privileges, secrets, and PUBLIC grantees are rejected before mutation.

After the prerequisite sequence above is authenticated, run
`scripts/setup-fleet-pg-tailscale.ps1` once per enrolled node identity, contract,
and unique role. Reconciliation executes `REVOKE CONNECT ON DATABASE`
from PUBLIC, transfers ownership to the dedicated NOLOGIN/NOSUPERUSER owner,
retires the named legacy/shared admin logins while preserving the isolated
infrastructure break-glass role, revokes broad
PUBLIC/object/default ACLs, then grants CONNECT only to the owner, controller,
verifier, fixed backup principal, mapped per-node contract roles, and named infrastructure break-glass
roles. Final validation enumerates every direct and effective grantee, including
database-owner and superuser behavior, proves that reconnect-capable roles are
exactly the explicitly allowlisted LOGIN roles, and checks the complete
function-only worker privilege boundary after SCRAM rotation. The normal
controller remains NOSUPERUSER and operational, but it cannot perform this
offline role/cluster reconciliation. For each setup or rotation, inject the
explicitly allowlisted break-glass `APPLYPILOT_ADMIN_PG_DSN` and the operator-held
receipt HMAC key into the operator shell only; remove them immediately afterward.
Neither value may be passed in argv, written to a receipt, or installed in a
worker service:

```powershell
$env:APPLYPILOT_ADMIN_PG_DSN = '<offline-break-glass-url>'
$env:APPLYPILOT_ROLLBACK_HMAC_KEY_HEX = '<operator-held-key-as-at-least-64-hex-characters>'
$env:APPLYPILOT_ROLLBACK_HMAC_KEY_ID = '<operator-key-id>'
& scripts/setup-fleet-pg-tailscale.ps1 `
  -NodeId '<enrolled-node-id>' `
  -Contract apply `
  -Role '<unique-mapped-login-role>' `
  -RegrantManifest .\evidence\reviewed-regrant-manifest.json `
  -ReceiptPath .\evidence\mapped-role-receipt.json `
  -RollbackSql .\evidence\mapped-role-rollback.sql
Remove-Item Env:APPLYPILOT_ADMIN_PG_DSN, Env:APPLYPILOT_ROLLBACK_HMAC_KEY_HEX, Env:APPLYPILOT_ROLLBACK_HMAC_KEY_ID
```

The command writes an operator-HMAC-authenticated, secret-free JSON receipt and
rollback SQL before changing `pg_hba.conf`. Prepared, database-reconciled, and
deployment-committed receipt states are authenticated separately. Every state is
bound to the database name, PostgreSQL cluster system identifier, database OID,
rollback mode, rollback path, and rollback SHA-256. The receipt records the principal/service/default-ACL inventory,
allowlist, effective grantees, mapping, target-role attributes and memberships,
direct database/schema/table/sequence/function/type ACLs, relevant default ACLs,
PUBLIC database `CONNECT`, `CREATE`, and `TEMPORARY`, every structured regrant
target ACL, the principal mapping's original `created_at`, PostgreSQL 18 inbound
and outbound membership grantor plus `ADMIN`/`INHERIT`/`SET` options, HBA
inventory, and backup path. It stores
only structured fields and the rollback SQL hash, never raw regrant SQL. Keep both artifacts with the
deployment evidence. The HBA edit inserts localhost-only SCRAM rules for
`fleet_backup` (`127.0.0.1/32` and `::1/128`) followed by IPv4/IPv6 rejects,
then the mapped-role tailnet allow and rejects, all before every pre-existing host rule. It rejects all include directives and input
control characters, uses an atomic replacement, validates both parsed rules and
effective first-match order, reloads, and restores the exact original plus
pre-write receipt outputs on any post-write failure. Receipt, rollback, HBA,
candidate, and backup path collisions abort before database or HBA mutation.
Every durable prepared or database-reconciled receipt has
`escalation_required=true` and `in_doubt=true`. This remains authoritative if
the process crashes after the database commit but before final receipt
replacement. Only a durably replaced `status=deployment_committed` or
`status=bootstrap_committed` receipt clears both flags. Deployment finalization
serializes the complete committed receipt to a same-directory temporary file,
flushes its contents and preserved destination ACL/owner metadata to disk, then
atomically replaces the in-doubt receipt. It never clears `in_doubt` in place;
abrupt termination therefore leaves either the intact database-reconciled
in-doubt receipt or the complete committed receipt, never partial JSON.

Logical backup provisioning is fail-closed when any application table has RLS
enabled/forced or when `pg_largeobject_metadata` contains a row. The backup role
does not receive `BYPASSRLS` and the launcher must never use
`pg_dump --enable-row-security`; either condition requires an explicit reviewed
policy change before backup or cutover. Receipts record the RLS/large-object
policy, exact role settings, default ACL closure, membership/ownership counts,
database-identity readability, and `pg_control_system()` capability without a
password, DSN, verifier, or other secret.

Rollback order is: pause all lanes, retrieve the offline break-glass credential,
verify the rollback file's SHA-256 against the durable receipt, and establish and
validate the named break-glass database session. The executor authenticates the
live HBA bytes without restoring them, commits authenticated phase-one database
containment, terminates affected non-break-glass, non-controller sessions,
repeats the idempotent rollback SQL in phase two, and asserts the final database
containment postconditions. Only then may it restore the recorded HBA backup and
reload through the already-established session. HBA admission remains closed
until both containment phases and the postconditions are durably evidenced.
Each containment phase is a separate fail-stop transaction equivalent to
`psql --single-transaction --set=ON_ERROR_STOP=on`; never pipe or execute the
generated statements independently. Use the executable workflow:

On Windows, the rollback executor deliberately refuses `--restore-hba` because
the required descriptor-relative atomic exchange is unavailable. Run the
database rollback first, verify both containment phases and postconditions, then
restore the authenticated HBA backup through the protected host procedure and
reload PostgreSQL. On a supported Linux host only, append `--restore-hba` to the
command after preflight confirms the receipt's signed `hba_restore` record.

```powershell
$env:APPLYPILOT_ADMIN_PG_DSN = '<offline-break-glass-url>'
$env:APPLYPILOT_ROLLBACK_HMAC_KEY_HEX = '<same-operator-held-key-as-setup>'
$env:APPLYPILOT_ROLLBACK_HMAC_KEY_ID = '<same-operator-key-id-as-setup>'
python scripts/rollback-fleet-pg-role.py `
  --receipt .\deployment-receipts\fleet-role-receipt.json `
  --rollback-sql .\deployment-receipts\fleet-role-rollback.sql
Remove-Item Env:APPLYPILOT_ADMIN_PG_DSN, Env:APPLYPILOT_ROLLBACK_HMAC_KEY_HEX, Env:APPLYPILOT_ROLLBACK_HMAC_KEY_ID
```

The executor refuses a SHA-256 mismatch, a non-superuser or unreceipted session,
an HBA reload before break-glass authentication, or any SQL error. After it
commits, revoke or disable the new
mapped login, restore the prior service credentials, and validate connectivity
for every regranted dependency. The rollback restores the locked pre-bootstrap
role attributes, ownership, memberships, CONNECT ACL, object ACL, and default
ACL snapshot. It intentionally contains no role password or DSN. Re-isolate the
break-glass credential after validation. A provider that cannot preserve this
authority, create the topology, strip legacy attributes, or execute the
prewritten rollback is unsupported and the bootstrap fails closed.

Mapped-role rollback is deliberately a quarantine operation, not a password
restore. For a role that existed before reconciliation, rollback removes the new
contract, schema, type, function, and structured regrant grants; restores its
prior direct/default ACLs, exact PostgreSQL 18 membership rows, non-secret
attributes, and principal mapping including its historical `created_at`; and
forces the role to `NOLOGIN`. The receipt sets
`credential_forward_reconcile_required=true` because the prior password or SCRAM
verifier cannot appear in secret-free evidence. The role must remain quarantined
until the controller forward-reconciles a newly generated password and all final
privilege and identity checks pass; only then may LOGIN be restored. Never reuse
the pre-rollback password. If reconciliation created the role, rollback first
removes its mapping and applied structured regrants, then runs `DROP OWNED BY`
and `DROP ROLE`. Execute mapped-role rollback with the isolated infrastructure
break-glass authority, then remove that DSN from the operator environment.

## External worker release connectivity

The Railway-hosted worker uses the database service's valid
`*.railway.internal` hostname over Railway's authenticated private mesh. Tarpon,
GGGTower, and any other worker outside that project must not resolve or route to
this name directly. A released external worker must use one of these
authenticated paths:

- an authenticated Tailscale/private gateway into the Railway private network,
  addressed by its full Tailscale MagicDNS hostname such as
  `pg-gateway.example-tailnet.ts.net` and restricted by tailnet ACLs;
- a database provider endpoint whose certificate validates the public hostname,
  configured with `sslmode=verify-full` and an explicit `sslrootcert` (including
`sslrootcert=system` when the issuer is in the system trust store).

A raw Tailscale IP, arbitrary RFC1918/link-local address, test-network address,
Unix socket, or shortened `*.ts.net` name is not an authenticated private
topology contract. Only exact loopback (`localhost`, `127.0.0.1`, or `::1`) is
accepted for local testing. All other targets must satisfy the public
`verify-full` certificate contract unless they match the Railway or full
MagicDNS suffixes above; Railway and Tailscale provide the encrypted private
transport for those two suffixes.

For a provider endpoint with publicly valid hostname identity, the contract is:

```text
postgresql://USER:PASSWORD@db.example.com:5432/fleet?sslmode=verify-full&sslrootcert=system
```

Keep credentials in the machine's secret store. Do not put a real DSN in task
arguments, checked-in configuration, or logs. A generic database variable is
never accepted as an ApplyPilot fleet fallback, including on external workers.

### Railway public proxy limitation

The observed Railway public proxy certificate has `subject CN=localhost`,
`issuer CN=root-ca`, and only `SAN DNS:localhost`. Connecting through the public
proxy hostname with `sslmode=verify-full` and `sslrootcert=system` therefore
fails hostname/certificate verification. The current Railway public proxy is
migration/admin-only and is not a fleet release path. It does not satisfy the
external-worker cutover gate, and the runbook must not treat a successful
`sslmode=require` connection as release evidence.

Migration and administrative tooling that uses the proxy remains separate from
the fleet worker release contract and requires an explicit operator-approved
procedure. Do not deploy Tarpon or GGGTower against `DATABASE_PUBLIC_URL` until
Railway presents a certificate valid for the proxy hostname or one of the
authenticated private paths above is in place.

## Connection ownership

This API remains connection-per-process: a long-lived worker opens one
connection and reuses it for its transaction loop. Do not reconnect for every
lease or result write, and do not add a pool inside `pgqueue.connect()`.

A controller that fans out concurrent work must own a separate, bounded
connection pool sized to its concurrency and Railway connection budget. Each
concurrent transaction checks out one connection from that controller-owned
pool; workers continue to receive and reuse a direct connection.

## Release sequence

1. Keep global and ATS lane pauses enabled, canary capacity at zero, and spend
   caps at zero.
2. Build the image from the reviewed release commit.
3. Calculate that commit's tree identity with
   `python -c "from applypilot.fleet.software_version import current_sw_version; print(current_sw_version())"`.
4. Set `APPLYPILOT_RELEASE_VERSION` to that exact value and pin the same value in
   Postgres.
5. Apply schema migrations from the owner/migration service, including
   `20260717_002_lane_specific_canary_pins`, and verify the fleet ledger. Then
   install and verify canonical brain schema v3. The worker has no
   schema-authority role and exits if required fleet objects are absent.
6. Inventory principals and services, review the regrant manifest, create the
   unique mapped login role, and archive its rollback SQL/receipt.
7. Set the replica's unique mapped `FLEET_PG_DSN`, worker ID, and contract.
8. Deploy one replica with a unique worker ID.
9. Require a fresh heartbeat whose `sw_version`, machine owner, role, and worker
   ID match the release contract.
10. Verify the worker remains unable to lease while the lane is paused.
11. Configure the exact lane worker/version pin and arm a separately approved
    ATS or LinkedIn canary only after that lane's deployment and policy gates
    pass.

Do not scale by cloning a service with the same `APPLYPILOT_WORKER_ID`. Each
replica needs an independently stable identity or a launcher that derives and
persists one before the process starts.
