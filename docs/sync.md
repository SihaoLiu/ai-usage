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

The health response includes the running server `version`, `schema_version`, and `uptime_seconds`.

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

## Integrity Checks

Each sync cycle computes a SHA-256 digest for the machine's own cached records, submits that report to the server, downloads the other machine reports, and recomputes the same digest over the pulled remote cache for each reported host. The monitor prompt shows `Integrity Checked` when the local remote-cache digest matches the host's own server report, and `Integrity Failed` when any reported host differs.

The checked range is independent of local time zones. Clients include records with RFC3339 timestamps earlier than the current UTC day's `00:00:00Z` boundary, so every machine checks the same complete UTC-day prefix and avoids the still-changing current UTC day. The digest uses normalized usage fields that both local and remote caches persist, including vendor, dedup key, timestamps, model, effort, fast tier, token counts, and embedded costs. It does not include local source file paths or project hashes.

`ai-usage sync push` refreshes local caches, uploads records, and submits the local integrity report. `ai-usage sync pull` downloads remote records and verifies them against the reports stored on the server. Older servers that do not expose the integrity endpoints still allow record push and pull, but they cannot produce a checked integrity result until upgraded.

Each integrity run writes compact JSONL transcripts below the cache root under `integrity/`. The local machine report is written as `local-<machine_id>.jsonl`; each remote verification view is written as `remote-<machine_id>.jsonl`. The first line is a summary with the range, digest, record count, and expected-versus-actual server comparison for remote views. Later lines are sorted per-record fingerprints with the hashed dedup key, canonical record hash, timestamps, model, token counts, and costs.

To debug a mismatch between two machines, copy the owner machine's `local-<machine_id>.jsonl` to the other machine and compare it with that machine's `remote-<machine_id>.jsonl`. For example, if `kilia` is checking records from `icarus3`, compare `kilia`'s `~/.cache/ai-usage/integrity/remote-icarus3.jsonl` with `icarus3`'s copied `~/.cache/ai-usage/integrity/local-icarus3.jsonl`.

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

The server uses SQLite WAL mode and stores its database at the configured `db_path`. A simple backup can be taken with SQLite's online backup command:

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
- No remote records: run `ai-usage sync push` on the source machine, then `ai-usage sync pull` on the viewing machine.
