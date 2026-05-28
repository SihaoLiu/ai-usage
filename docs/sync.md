# Cross-Machine Sync

`vibe-usage` can optionally sync normalized usage records between machines through a self-hosted server. The feature is disabled unless a valid client secrets file exists. The open-source binary does not include a default server URL or token.

## Model

- Each client uploads its own local records.
- Each client pulls records from other configured machines.
- Remote records are cached below the normal cache root under `remote/<machine_id>.bin`.
- Display commands merge local and remote records, so existing charts and breakdowns keep the same shape.
- `--host <machine_id>` filters the merged view to one machine.

The server is intended for one user operating their own endpoint. It stores records in SQLite and exposes `/v1/upload`, `/v1/pull`, `/v1/machines`, and `/v1/health`.

## Server Setup

Create a service user and directories:

```bash
sudo useradd --system --home /var/lib/vibe-usage --shell /usr/sbin/nologin vibe-usage
sudo install -d -o vibe-usage -g vibe-usage -m 0750 /var/lib/vibe-usage
sudo install -d -o vibe-usage -g vibe-usage -m 0755 /var/lib/vibe-usage/bin
sudo install -d -m 0755 /etc/vibe-usage-server
```

Download and install the matching server release asset:

```bash
curl -L -o /tmp/vibe-usage-server \
  https://github.com/SihaoLiu/ai-usage/releases/latest/download/vibe-usage-server-x86_64-linux-gnu
sudo install -o vibe-usage -g vibe-usage -m 0755 /tmp/vibe-usage-server /var/lib/vibe-usage/bin/vibe-usage-server
```

Or build the server binary from source:

```bash
cargo build --release -p vibe-usage-server
sudo install -o vibe-usage -g vibe-usage -m 0755 target/release/vibe-usage-server /var/lib/vibe-usage/bin/vibe-usage-server
```

Generate one shared token and keep it private:

```bash
openssl rand -base64 32
```

Copy `examples/server-config-template.yaml` to `/etc/vibe-usage-server/config.yaml`, then replace the placeholder token and host ids:

```yaml
listen: "127.0.0.1:8787"
db_path: "/var/lib/vibe-usage/data.db"
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
sudo install -m 0644 crates/vibe-usage-server/deploy/vibe-usage-server.service.example /etc/systemd/system/vibe-usage-server.service
sudo systemctl daemon-reload
sudo systemctl enable --now vibe-usage-server
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
vibe-usage sync init
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
vibe-usage sync push
vibe-usage sync pull
vibe-usage sync status
```

Normal monitor mode starts a background sync worker after local cache refreshes when sync is enabled. If the secrets file is missing, invalid, or too widely readable, sync is disabled and the rest of the CLI continues to work.

## Viewing Data

Aggregate all local and remote records:

```bash
vibe-usage
```

Show one machine:

```bash
vibe-usage --host laptop
```

Combine host and vendor filters:

```bash
vibe-usage --vendor claude --host laptop
```

## Operations

The server uses SQLite WAL mode and stores its database at the configured `db_path`. A simple backup can be taken with SQLite's online backup command:

```bash
sqlite3 /var/lib/vibe-usage/data.db ".backup '/var/lib/vibe-usage/backup-YYYY-MM-DD.db'"
```

Useful checks:

```bash
systemctl status vibe-usage-server
journalctl -u vibe-usage-server
vibe-usage sync status
```

Client-side sync errors are recorded in the local cache root as `sync_state.json` and `sync.log`.

## Auto Update

Automatic server updates are disabled by default. To enable them, run the server from the writable service binary path shown above and set:

```yaml
auto_update:
  enabled: true
  interval_seconds: 3600
```

The server checks GitHub Releases for a newer `vibe-usage-server-<target>` asset. When a newer binary is installed successfully, the process exits and systemd restarts it. The service unit must use `Restart=always` and start `/var/lib/vibe-usage/bin/vibe-usage-server` so the `vibe-usage` service user can replace the executable.

## Troubleshooting

- `sync: disabled`: create `~/.secrets/ai-usage.yaml` or set `sync.enabled: true`.
- Permission warning: run `chmod 600 ~/.secrets/ai-usage.yaml`.
- `401` from the server: client and server tokens do not match.
- `403` from upload: `machine_id` is not listed in `allowed_hosts`.
- No remote records: run `vibe-usage sync push` on the source machine, then `vibe-usage sync pull` on the viewing machine.
