use rusqlite::{OptionalExtension, Transaction, params};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotAttemptState {
    Active,
    Completed,
    Superseded,
}

pub fn register(
    tx: &Transaction<'_>,
    host_id: &str,
    snapshot_id: &str,
) -> rusqlite::Result<SnapshotAttemptState> {
    if let Some(state) = lookup(tx, host_id, snapshot_id)? {
        return Ok(state);
    }
    tx.execute(
        "INSERT INTO snapshot_attempts (host_id, snapshot_id)
         VALUES (?1, ?2)",
        params![host_id, snapshot_id],
    )?;
    lookup(tx, host_id, snapshot_id)?.ok_or(rusqlite::Error::QueryReturnedNoRows)
}

pub fn lookup(
    tx: &Transaction<'_>,
    host_id: &str,
    snapshot_id: &str,
) -> rusqlite::Result<Option<SnapshotAttemptState>> {
    let Some((attempt_order, completed)) = tx
        .query_row(
            "SELECT attempt_order, completed_at IS NOT NULL
             FROM snapshot_attempts
             WHERE host_id = ?1 AND snapshot_id = ?2",
            params![host_id, snapshot_id],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, bool>(1)?)),
        )
        .optional()?
    else {
        return Ok(None);
    };
    let latest_order = tx.query_row(
        "SELECT MAX(attempt_order) FROM snapshot_attempts WHERE host_id = ?1",
        [host_id],
        |row| row.get::<_, i64>(0),
    )?;
    if attempt_order != latest_order {
        Ok(Some(SnapshotAttemptState::Superseded))
    } else if completed {
        Ok(Some(SnapshotAttemptState::Completed))
    } else {
        Ok(Some(SnapshotAttemptState::Active))
    }
}

pub fn complete(
    tx: &Transaction<'_>,
    host_id: &str,
    snapshot_id: &str,
    completed_at: &str,
) -> rusqlite::Result<()> {
    tx.execute(
        "UPDATE snapshot_attempts
         SET completed_at = ?3
         WHERE host_id = ?1 AND snapshot_id = ?2 AND completed_at IS NULL",
        params![host_id, snapshot_id, completed_at],
    )?;
    Ok(())
}
