# Cross-Machine Sync

`ai-usage` can optionally sync normalized usage records between machines through a self-hosted server. The feature is disabled unless a valid client secrets file exists. The open-source binary does not include a default server URL or token.

## Model

- Each client uploads its own local records.
- Each client pulls records from other configured machines.
- Remote records are cached below the normal cache root under `remote/<machine_id>.bin`.
- Display commands merge local and remote records, so existing charts and breakdowns keep the same shape.
- `--host <machine_id>` filters the merged view to one machine.

The server is intended for one user operating their own endpoint. It stores records in SQLite and exposes `/v1/upload`, `/v1/pull`, `/v1/machines`, `/v1/integrity/report`, `/v1/integrity/reports`, and `/v1/health`.

## Server Setup

Create a service user and directories:

```bash
sudo useradd --system --home /var/lib/ai-usage --shell /usr/sbin/nologin ai-usage
sudo install -d -o ai-usage -g ai-usage -m 0750 /var/lib/ai-usage
sudo install -d -o ai-usage -g ai-usage -m 0755 /var/lib/ai-usage/bin
sudo install -d -m 0755 /etc/ai-usage-server
```

Download and install the matching server release asset:

```bash
curl -L -o /tmp/ai-usage-server \
  https://github.com/SihaoLiu/ai-usage/releases/latest/download/ai-usage-server-x86_64-linux-gnu
sudo install -o ai-usage -g ai-usage -m 0755 /tmp/ai-usage-server /var/lib/ai-usage/bin/ai-usage-server
```

Or build the server binary from source:

```bash
cargo build --release -p ai-usage-server
sudo install -o ai-usage -g ai-usage -m 0755 target/release/ai-usage-server /var/lib/ai-usage/bin/ai-usage-server
```

Generate one shared token and keep it private:

```bash
openssl rand -base64 32
```

Copy `examples/server-config-template.yaml` to `/etc/ai-usage-server/config.yaml`, then replace the placeholder token and host ids:

```yaml
listen: "127.0.0.1:8787"
db_path: "/var/lib/ai-usage/data.db"
shared_token: "replace-me"
allowed_hosts:
  - "workstation-home"
  - "laptop"
max_body_bytes: 1048576
max_batch_records: 1000
log_level: "info"
auto_update:
  enabled: false
  interval_seconds: 3600
```

Install the systemd example:

```bash
sudo install -m 0644 crates/ai-usage-server/deploy/ai-usage-server.service.example /etc/systemd/system/ai-usage-server.service
sudo systemctl daemon-reload
sudo systemctl enable --now ai-usage-server
```

Expose the local server through Caddy or another HTTPS reverse proxy. The client requires an `https://` URL.

```caddyfile
usage.your-domain.example {
    reverse_proxy 127.0.0.1:8787
}
```

Check the public health endpoint:

```bash
curl https://usage.your-domain.example/v1/health
```

The health response includes the running server `version`, `schema_version`, `uptime_seconds`, stable `instance_id`, and an optional request pacing policy. Current clients use that policy automatically; older clients ignore the additional fields.

## Client Setup

Create a disabled template:

```bash
ai-usage sync init
```

Edit `~/.secrets/ai-usage.yaml`, replace the placeholders, and set `enabled: true`:

```yaml
sync:
  enabled: true
  server_url: "https://usage.your-domain.example"
  token: "replace-me"
  machine_id: "workstation-home"
  upload_project_hash: true
  request_timeout_seconds: 15
```

Keep the file private:

```bash
chmod 600 ~/.secrets/ai-usage.yaml
```

Run a manual upload and pull:

```bash
ai-usage sync push
ai-usage sync pull
ai-usage sync status
```

Normal monitor mode starts a background sync worker after local cache refreshes when sync is enabled. If the secrets file is missing, invalid, or too widely readable, sync is disabled and the rest of the CLI continues to work.

## Load Balancing and Recovery

Clients identify themselves to current servers with their configured `machine_id`. The server maintains a separate token bucket for each machine, so a busy client cannot consume another client's request allowance. A bounded aggregate bucket protects the whole service from request floods, and idle per-machine buckets are discarded so authenticated clients cannot grow the limiter indefinitely by rotating identifiers. Clients from older releases that do not send the identity header remain supported through a legacy bucket.

The health response derives a sustained request interval from the per-machine limit, the aggregate limit, and the configured `allowed_hosts`. It also assigns each configured machine a stable phase within that interval. This spreads simultaneous client starts across the server's available capacity instead of relying on `429` responses after a burst. Clients increase the negotiated interval after timeouts, transient server errors, or rate limits, then reduce it toward the server baseline after consecutive successful requests. A malformed policy cannot make a client sleep for more than five minutes between requests.

Normal uploads use snapshot reconciliation. An append-only cache change sends only new or changed fingerprints and records; a full manifest and finalization are reserved for deletions or missing local state. Request bodies are kept below approximately 700 KiB, and pull pages are capped at 5,000 records by both current clients and servers. The server cap also protects small deployments when an older client requests a 20,000-record page.

Monitor mode applies a stable per-machine delay of up to 60 seconds after cache refreshes, preventing machines with similar schedules from synchronizing at once. A local file lock also prevents multiple `ai-usage` processes on one machine from running overlapping sync cycles.

The client retries temporary transport failures and HTTP `408`, `425`, `429`, `500`, `502`, `503`, and `504` responses with exponential, per-machine jitter. A server `Retry-After` header is honored. Snapshot finalization has a two-minute minimum request budget because it may remove stale rows on a small server, while other requests retain the configured timeout. If a whole background cycle still fails, the worker retries it automatically with exponential backoff up to 30 minutes instead of waiting for the next monitor refresh.

When the local vendor caches have not changed, a small upload receipt lets the upload path skip loading the full fingerprint state and rescanning every record. The receipt is scoped to the server instance, URL, machine ID, and project-hash policy. The client checks both the server's record count and per-host content revision before using it, so a replaced or restored database, missing host, or wire-affecting configuration change triggers reconciliation. A v3.0.0 server exposes neither instance IDs nor content revisions, so current clients force both a full upload reconciliation and a pull backfill at least every six hours while connected to one. The large fingerprint file is rebuildable state: corruption cannot strand the server, because the next local cache change falls back to a full manifest.

Full reconciliation persists one attempt ID before sending its manifest. Retries and process restarts reuse that ID until the client confirms completion, while a later reconciliation receives a new ID. The client atomically stores the ready-to-finalize attempt ID, scope, and manifest separately from the last confirmed manifest, so a cache change during recovery cannot pair one attempt with another manifest or erase the deletion baseline. The server assigns a persistent arrival order to every snapshot attempt and permits only the latest attempt for a machine to write or finalize. Unknown and superseded attempts receive HTTP 409 across every snapshot endpoint, causing current clients to discard and rebuild them while remaining visibly unsuccessful to older clients. Retrying the latest completed finalize remains a no-op. Repeated manifests do not rewrite rows already marked with the same attempt ID, and machine record counts are maintained from transactional insert and delete deltas instead of rescanning every record after each batch.

## Integrity Checks

Integrity checks compute a SHA-256 digest for the machine's own cached records, submit that report to the server, download the other machine reports, and recompute the same digest over the pulled remote cache for each reported host. The monitor prompt shows `Integrity Checked` when the local remote-cache digest matches the host's own server report, and `Integrity Failed` when any reported host differs.

The checked range is independent of local time zones. Clients include records with RFC3339 timestamps earlier than the current UTC day's `00:00:00Z` boundary, so every machine checks the same complete UTC-day prefix and avoids the still-changing current UTC day. The digest uses normalized usage fields that both local and remote caches persist, including vendor, dedup key, timestamps, model, effort, fast tier, token counts, and embedded costs. It does not include local source file paths or project hashes.

Successful background checks are reused for up to six hours within the same server, machine, and upload-policy scope. A new UTC-day boundary, a sync-scope change, or any uploaded or pulled change inside the stable historical range invalidates the saved result immediately. `ai-usage sync push` and `ai-usage sync pull` always force a new check. Older servers that do not expose the integrity endpoints still allow record push and pull, but they cannot produce a checked integrity result until upgraded.

Each integrity run writes compact JSONL summaries below the cache root under `integrity/`. The local machine report is written as `local-<machine_id>.jsonl`; each remote verification view is written as `remote-<machine_id>.jsonl`. Each file contains one summary line with the range, digest, record count, and expected-versus-actual server comparison for remote views. This replaces the previous per-record transcripts, which could grow to hundreds of megabytes.

To debug a mismatch, compare the owner machine's `local-<machine_id>.jsonl` summary with the viewing machine's `remote-<machine_id>.jsonl` summary. The expected and actual counts and digests identify which host and stable range need to be pulled again.

## Viewing Data

Aggregate all local and remote records:

```bash
ai-usage
```

Show one machine:

```bash
ai-usage --host laptop
```

Combine host and tool filters:

```bash
ai-usage --tool claude --host laptop
```

## Operations

### Upgrade ordering when a release adds a vendor

When a release adds a new tracked vendor (as v2.5.0 adds `kimi`), upgrade the server binary first, then upgrade every client promptly.

A newer client against an older server degrades gracefully: it holds the new vendor's uploads back (reported as `holding back ... record(s)`), pulls the previous vendor set (reported as `server does not serve vendor(s) ... yet`), and skips integrity checking for that cycle. Everything self-heals on the first sync after the server upgrade.

The reverse skew is the one to keep short: a client still on the previous release cannot pull the new vendor's records, so its integrity verification of hosts that already upload them fails on every cycle and repeatedly clears and refetches its remote cache. This resolves as soon as that machine upgrades (monitor mode's `update` command or `--auto-update` handles this).

The server uses SQLite WAL mode and stores its database at the configured `db_path`. Write transactions are serialized through an asynchronous gate, avoiding competing `BEGIN IMMEDIATE` transactions and `database is locked` failures while reads remain concurrent. A simple backup can be taken with SQLite's online backup command:

```bash
sqlite3 /var/lib/ai-usage/data.db ".backup '/var/lib/ai-usage/backup-YYYY-MM-DD.db'"
```

Useful checks:

```bash
systemctl status ai-usage-server
journalctl -u ai-usage-server
ai-usage sync status
```

Client-side sync errors are recorded in the local cache root as `sync_state.json` and `sync.log`.

Current clients and servers remain wire-compatible with v3.0.0 during a staggered rollout. Upgrade the server first when practical so current clients receive per-machine rate limits and smaller server-enforced pull pages immediately.

## Auto Update

Automatic server updates are disabled by default. To enable them, run the server from the writable service binary path shown above and set:

```yaml
auto_update:
  enabled: true
  interval_seconds: 3600
```

The server checks GitHub Releases for a newer `ai-usage-server-<target>` asset. When a newer binary is installed successfully, the process exits and systemd restarts it. The service unit must use `Restart=always` and start `/var/lib/ai-usage/bin/ai-usage-server` so the `ai-usage` service user can replace the executable.

### Rename compatibility

A running binary resolves its update asset by name: it asks for `ai-usage-<target>` (client) or `ai-usage-server-<target>` (server) and ignores any other asset in the release. The asset name is only a lookup label; the file it points at carries the current binary regardless of what that file is named.

Binaries published before the project was renamed from `vibe-usage` look for the old `vibe-usage-<target>` and `vibe-usage-server-<target>` names instead. To let those binaries self-update one last time onto the renamed binary, the release workflow uploads each artifact twice: once under the current `ai-usage-*` name and once under the legacy `vibe-usage-*` name (identical bytes, second filename). An old binary finds its legacy name, installs the renamed binary, and from the next check on looks for the `ai-usage-*` name.

Keep publishing the legacy names until no deployment older than the rename remains; removing them earlier leaves any pre-rename binary unable to find its asset, so it stays on its installed version until it is replaced by hand. Once every machine has updated past the rename release, drop the `legacy_*` entries from the release workflow.

## Troubleshooting

- `sync: disabled`: create `~/.secrets/ai-usage.yaml` or set `sync.enabled: true`.
- Permission warning: run `chmod 600 ~/.secrets/ai-usage.yaml`.
- `401` from the server: client and server tokens do not match.
- `403` from upload: `machine_id` is not listed in `allowed_hosts`.
- Repeated `429`: confirm each current client has a distinct valid `machine_id`. Current clients retry the server's `Retry-After` automatically; older clients share the legacy rate-limit bucket until upgraded.
- Repeated timeout or `502`/`503`/`504`: inspect the reverse-proxy and service logs. Current clients retry transient failures automatically, but a persistent error usually indicates an unavailable service or proxy limit.
- `another sync is already running`: wait for the existing local process to finish; overlapping manual and monitor syncs are intentionally suppressed.
- No remote records: run `ai-usage sync push` on the source machine, then `ai-usage sync pull` on the viewing machine.
