CREATE TABLE IF NOT EXISTS records (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    host_id        TEXT NOT NULL,
    vendor         TEXT NOT NULL,
    dedup_key      TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    timestamp_utc  TEXT NOT NULL,
    session_start  TEXT NOT NULL,
    session_end    TEXT NOT NULL,
    model          TEXT NOT NULL,
    effort         TEXT,
    fast_tier      INTEGER NOT NULL DEFAULT -1,
    input_tokens   INTEGER NOT NULL,
    output_tokens  INTEGER NOT NULL,
    cache_read     INTEGER NOT NULL,
    cache_creation INTEGER NOT NULL,
    reasoning_out  INTEGER NOT NULL,
    project_hash   TEXT,
    uploaded_at    TEXT NOT NULL,
    UNIQUE(host_id, vendor, dedup_key)
);

CREATE INDEX IF NOT EXISTS idx_records_seq ON records(seq);
CREATE INDEX IF NOT EXISTS idx_records_host_seq ON records(host_id, seq);
CREATE INDEX IF NOT EXISTS idx_records_timestamp ON records(timestamp_utc);

CREATE TABLE IF NOT EXISTS machines (
    host_id      TEXT PRIMARY KEY,
    last_seen    TEXT NOT NULL,
    record_count INTEGER NOT NULL DEFAULT 0
);
