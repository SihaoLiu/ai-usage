//! Sync-worker status polling and status-line formatting.

use chrono::{DateTime, Duration};

use crate::sync;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntegrityStatus {
    Unavailable,
    Pending,
    Checking { percent: u8 },
    Checked { duration: std::time::Duration },
    Failed,
}

pub(crate) fn initial_integrity_status(sync_enabled: bool) -> IntegrityStatus {
    if sync_enabled {
        IntegrityStatus::Pending
    } else {
        IntegrityStatus::Unavailable
    }
}

pub(crate) fn poll_sync_worker_status(
    worker: Option<&sync::worker::SyncWorker>,
    observed_revision: &mut u64,
) -> Option<sync::worker::SyncStats> {
    let worker = worker?;
    let stats = worker.stats();
    if stats.revision == *observed_revision {
        return None;
    }
    *observed_revision = stats.revision;
    Some(stats)
}

pub(crate) fn current_sync_status(worker: Option<&sync::worker::SyncWorker>) -> Option<String> {
    worker.and_then(|worker| format_monitor_sync_status(&worker.stats()))
}

pub(crate) fn integrity_status_from_verification(
    verification: &sync::integrity::IntegrityVerification,
    duration: std::time::Duration,
) -> IntegrityStatus {
    match verification {
        sync::integrity::IntegrityVerification::Checked { .. } => {
            IntegrityStatus::Checked { duration }
        }
        sync::integrity::IntegrityVerification::Failed { .. } => IntegrityStatus::Failed,
    }
}

pub(crate) fn monitor_sync_interval(monitor_interval: std::time::Duration) -> std::time::Duration {
    (monitor_interval / 3).max(std::time::Duration::from_secs(60))
}

pub(crate) fn monitor_sync_stagger(
    sync_interval: std::time::Duration,
    machine_id: &str,
) -> std::time::Duration {
    let window = (sync_interval / 4).min(std::time::Duration::from_secs(60));
    let window_millis = window.as_millis().min(u64::MAX as u128) as u64;
    let millis =
        sync::timing::stable_hash(machine_id, 0x7379_6e63) % window_millis.saturating_add(1);
    std::time::Duration::from_millis(millis)
}

pub(crate) fn monitor_sync_delay(
    monitor_interval: std::time::Duration,
    machine_id: &str,
) -> std::time::Duration {
    let sync_interval = monitor_sync_interval(monitor_interval);
    sync_interval.saturating_add(monitor_sync_stagger(sync_interval, machine_id))
}

pub(crate) fn monitor_sync_deadline_after_refresh(
    now: std::time::Instant,
    current_deadline: std::time::Instant,
    monitor_interval: std::time::Duration,
    machine_id: &str,
) -> std::time::Instant {
    let triggered_deadline =
        now + monitor_sync_stagger(monitor_sync_interval(monitor_interval), machine_id);
    current_deadline.min(triggered_deadline)
}

pub(crate) fn monitor_deadlines_after_interval_change(
    now: std::time::Instant,
    monitor_interval_seconds: u64,
    machine_id: &str,
) -> (std::time::Instant, std::time::Instant) {
    let monitor_interval = std::time::Duration::from_secs(monitor_interval_seconds);
    (
        now + monitor_interval,
        now + monitor_sync_delay(monitor_interval, machine_id),
    )
}

pub(crate) fn auto_update_deadline_after(
    now: std::time::Instant,
    auto_update_interval_seconds: u64,
) -> std::time::Instant {
    now + ai_usage_updater::normalize_auto_update_interval(auto_update_interval_seconds)
}

pub(crate) fn format_manual_sync_progress(event: &sync::engine::SyncProgress) -> Option<String> {
    match event {
        sync::engine::SyncProgress::UploadPlanned {
            total_records,
            total_batches,
            skipped_records,
        } => {
            if *skipped_records == 0 {
                Some(format!(
                    "sync push: {total_records} records to upload in {total_batches} batches"
                ))
            } else {
                Some(format!(
                    "sync push: {total_records} records to upload in {total_batches} batches, {skipped_records} already synced"
                ))
            }
        }
        sync::engine::SyncProgress::UploadBatchFinished {
            batch_index,
            total_batches,
            uploaded_records,
            total_records,
            accepted,
            ignored,
        } => Some(format!(
            "sync push: [{}] {batch_index}/{total_batches} batches, {uploaded_records}/{total_records} records, accepted {accepted}, ignored {ignored}",
            progress_bar(*batch_index, *total_batches)
        )),
        sync::engine::SyncProgress::UploadFinished {
            uploaded_records,
            total_records,
            accepted,
            ignored,
        } => Some(format!(
            "sync push: complete, {uploaded_records}/{total_records} records, accepted {accepted}, ignored {ignored}"
        )),
        sync::engine::SyncProgress::PullPageFinished {
            page_index,
            pulled_records,
            max_seq,
            ..
        } => Some(format!(
            "sync pull: page {page_index}, {pulled_records} records pulled, latest seq {max_seq}"
        )),
        sync::engine::SyncProgress::PullFinished {
            pages,
            pulled_records,
            max_seq,
        } => Some(format!(
            "sync pull: complete, {pages} pages, {pulled_records} records pulled, latest seq {max_seq}"
        )),
        sync::engine::SyncProgress::UploadVendorHeldBack { vendor, records } => Some(format!(
            "sync push: server does not accept vendor {vendor} yet, holding back {records} record(s)"
        )),
        sync::engine::SyncProgress::PullVendorsUnavailable { vendors } => Some(format!(
            "sync pull: server does not serve vendor(s) {} yet, their remote records are unavailable",
            vendors.join(", ")
        )),
        sync::engine::SyncProgress::IntegrityUnsupported => {
            Some("sync integrity: server does not support integrity reports".to_string())
        }
        sync::engine::SyncProgress::IntegrityReportSubmitted {
            record_count,
            range_end_utc,
        } => Some(format!(
            "sync integrity: submitted {record_count} records through {range_end_utc}"
        )),
        sync::engine::SyncProgress::IntegrityCheckProgress { percent } => {
            Some(format!("sync integrity: checking {percent}%"))
        }
        sync::engine::SyncProgress::IntegrityCheckFinished { verification } => Some(format!(
            "sync integrity: {}",
            format_integrity_verification(verification)
        )),
        sync::engine::SyncProgress::IntegrityCheckReused { checked_hosts } => Some(format!(
            "sync integrity: reused recent check for {checked_hosts} hosts"
        )),
    }
}

pub(crate) fn format_http_progress(event: &sync::client::HttpProgress) -> String {
    match event {
        sync::client::HttpProgress::RateLimited {
            attempt,
            retry_after,
        } => format!(
            "sync: rate limited, retrying in {} (attempt {attempt})",
            format_retry_duration(*retry_after)
        ),
    }
}

pub(crate) fn format_monitor_sync_status(stats: &sync::worker::SyncStats) -> Option<String> {
    format_monitor_sync_status_at(stats, chrono::Utc::now())
}

pub(crate) fn format_monitor_sync_status_at(
    stats: &sync::worker::SyncStats,
    now: DateTime<chrono::Utc>,
) -> Option<String> {
    if stats.running {
        return Some(match stats.progress.as_ref() {
            Some(progress) => format!("Sync: {}", format_monitor_worker_progress(progress)),
            None => "Sync: running".to_string(),
        });
    }

    if let Some(error) = stats.last_error.as_ref() {
        return Some(format!("Sync: error: {error}"));
    }

    if stats.success_count > 0 {
        let finished = stats
            .last_finished_at
            .as_deref()
            .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
            .map(|value| value.with_timezone(&chrono::Utc));
        if let Some(finished) = finished {
            return Some(format!(
                "Sync: checked {}",
                format_sync_age(now.signed_duration_since(finished))
            ));
        }
        return Some("Sync: checked".to_string());
    }

    None
}

pub(crate) fn format_monitor_worker_progress(
    progress: &sync::worker::SyncWorkerProgress,
) -> String {
    match progress {
        sync::worker::SyncWorkerProgress::Sync(event) => match event {
            sync::engine::SyncProgress::UploadPlanned {
                total_records,
                total_batches,
                ..
            } => format!("push queued {total_records} records in {total_batches} batches"),
            sync::engine::SyncProgress::UploadBatchFinished {
                batch_index,
                total_batches,
                uploaded_records,
                total_records,
                ..
            } => format!(
                "push {batch_index}/{total_batches} batches, {uploaded_records}/{total_records} records"
            ),
            sync::engine::SyncProgress::UploadFinished {
                uploaded_records,
                total_records,
                ..
            } => format!("push complete, {uploaded_records}/{total_records} records"),
            sync::engine::SyncProgress::UploadVendorHeldBack { vendor, records } => {
                format!("push holding back {records} {vendor} record(s), server too old")
            }
            sync::engine::SyncProgress::PullVendorsUnavailable { vendors } => {
                format!(
                    "pull missing {} from server, server too old",
                    vendors.join(", ")
                )
            }
            sync::engine::SyncProgress::PullPageFinished {
                page_index,
                pulled_records,
                max_seq,
                ..
            } => format!(
                "pull page {page_index}, {pulled_records} records pulled, latest seq {max_seq}"
            ),
            sync::engine::SyncProgress::PullFinished {
                pages,
                pulled_records,
                max_seq,
            } => format!(
                "pull complete, {pages} pages, {pulled_records} records pulled, latest seq {max_seq}"
            ),
            sync::engine::SyncProgress::IntegrityUnsupported => "integrity unsupported".to_string(),
            sync::engine::SyncProgress::IntegrityReportSubmitted {
                record_count,
                range_end_utc,
            } => format!("integrity submitted {record_count} records through {range_end_utc}"),
            sync::engine::SyncProgress::IntegrityCheckProgress { percent } => {
                format!("integrity checking {percent}%")
            }
            sync::engine::SyncProgress::IntegrityCheckFinished { verification } => {
                format!("integrity {}", format_integrity_verification(verification))
            }
            sync::engine::SyncProgress::IntegrityCheckReused { checked_hosts } => {
                format!("integrity reused recent check for {checked_hosts} hosts")
            }
        },
        sync::worker::SyncWorkerProgress::Http(event) => match event {
            sync::client::HttpProgress::RateLimited {
                attempt,
                retry_after,
            } => format!(
                "rate limited, retrying in {} (attempt {attempt})",
                format_retry_duration(*retry_after)
            ),
        },
    }
}

pub(crate) fn format_integrity_verification(
    verification: &sync::integrity::IntegrityVerification,
) -> String {
    match verification {
        sync::integrity::IntegrityVerification::Checked { checked_hosts } => {
            format!("checked {checked_hosts} hosts")
        }
        sync::integrity::IntegrityVerification::Failed { failures } => {
            format!("failed {} hosts", failures.len())
        }
    }
}

pub(crate) fn format_sync_age(age: Duration) -> String {
    let seconds = age.num_seconds().max(0);
    if seconds < 60 {
        "just now".to_string()
    } else if seconds < 3600 {
        format!("{} min ago", seconds / 60)
    } else if seconds < 86_400 {
        format!("{} h ago", seconds / 3600)
    } else {
        format!("{} d ago", seconds / 86_400)
    }
}

pub(crate) fn progress_bar(done: usize, total: usize) -> String {
    const WIDTH: usize = 20;
    let filled = if total == 0 {
        WIDTH
    } else {
        done.saturating_mul(WIDTH).min(total.saturating_mul(WIDTH)) / total
    };
    format!("{}{}", "#".repeat(filled), "-".repeat(WIDTH - filled))
}

pub(crate) fn format_retry_duration(duration: std::time::Duration) -> String {
    if duration.as_millis() == 0 {
        "0s".to_string()
    } else if duration.as_secs() < 10 {
        format!("{:.1}s", duration.as_secs_f64())
    } else {
        format!("{}s", duration.as_secs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_integrity_status_never_fabricates_a_completed_check() {
        assert_eq!(initial_integrity_status(true), IntegrityStatus::Pending);
        assert_eq!(
            initial_integrity_status(false),
            IntegrityStatus::Unavailable
        );
    }

    #[test]
    fn sync_integrity_verification_maps_to_prompt_status() {
        let duration = std::time::Duration::from_millis(12);
        assert_eq!(
            integrity_status_from_verification(
                &sync::integrity::IntegrityVerification::Checked { checked_hosts: 2 },
                duration,
            ),
            IntegrityStatus::Checked { duration }
        );

        assert_eq!(
            integrity_status_from_verification(
                &sync::integrity::IntegrityVerification::Failed {
                    failures: vec![sync::integrity::IntegrityFailure {
                        host_id: "laptop".to_string(),
                        range_end_utc: "2026-06-01T00:00:00Z".to_string(),
                        expected_record_count: 1,
                        actual_record_count: 0,
                        expected_digest_sha256: "a".repeat(64),
                        actual_digest_sha256: "b".repeat(64),
                    }],
                },
                duration,
            ),
            IntegrityStatus::Failed
        );
    }

    #[test]
    fn sync_interval_is_one_third_of_monitor_interval_with_minimum() {
        assert_eq!(
            monitor_sync_interval(std::time::Duration::from_secs(3600)),
            std::time::Duration::from_secs(1200)
        );
        assert_eq!(
            monitor_sync_interval(std::time::Duration::from_secs(180)),
            std::time::Duration::from_secs(60)
        );
        assert_eq!(
            monitor_sync_interval(std::time::Duration::from_secs(30)),
            std::time::Duration::from_secs(60)
        );
    }

    #[test]
    fn sync_stagger_is_stable_and_bounded() {
        let interval = std::time::Duration::from_secs(1200);
        let first = monitor_sync_stagger(interval, "workstation");
        let second = monitor_sync_stagger(interval, "workstation");

        assert_eq!(first, second);
        assert!(first <= std::time::Duration::from_secs(60));
        assert!(first <= interval / 4);
    }

    #[test]
    fn interval_change_reschedules_refresh_and_sync_deadlines() {
        let now = std::time::Instant::now();

        let (next_refresh, next_sync) =
            monitor_deadlines_after_interval_change(now, 3600, "workstation");

        assert_eq!(next_refresh, now + std::time::Duration::from_secs(3600));
        assert_eq!(
            next_sync,
            now + monitor_sync_delay(std::time::Duration::from_secs(3600), "workstation")
        );
    }

    #[test]
    fn refresh_completion_does_not_postpone_a_pending_sync() {
        let now = std::time::Instant::now();

        assert_eq!(
            monitor_sync_deadline_after_refresh(
                now,
                now,
                std::time::Duration::from_secs(30),
                "workstation",
            ),
            now
        );
    }

    #[test]
    fn manual_sync_progress_formats_push_pull_and_retry_events() {
        assert_eq!(
            format_manual_sync_progress(&sync::engine::SyncProgress::UploadPlanned {
                total_records: 1001,
                total_batches: 2,
                skipped_records: 7,
            }),
            Some("sync push: 1001 records to upload in 2 batches, 7 already synced".to_string())
        );
        assert_eq!(
                format_manual_sync_progress(&sync::engine::SyncProgress::UploadBatchFinished {
                    batch_index: 1,
                    total_batches: 2,
                    uploaded_records: 1000,
                    total_records: 1001,
                    accepted: 997,
                    ignored: 3,
                }),
                Some(
                    "sync push: [##########----------] 1/2 batches, 1000/1001 records, accepted 997, ignored 3"
                        .to_string()
                )
            );
        assert_eq!(
            format_manual_sync_progress(&sync::engine::SyncProgress::PullPageFinished {
                page_index: 2,
                page_records: 5000,
                pulled_records: 10000,
                max_seq: 123,
                truncated: true,
            }),
            Some("sync pull: page 2, 10000 records pulled, latest seq 123".to_string())
        );
        assert_eq!(
            format_http_progress(&sync::client::HttpProgress::RateLimited {
                attempt: 2,
                retry_after: std::time::Duration::from_millis(1100),
            }),
            "sync: rate limited, retrying in 1.1s (attempt 2)"
        );
    }

    #[test]
    fn monitor_sync_status_formats_running_progress_and_completion() {
        let running = sync::worker::SyncStats {
            running: true,
            progress: Some(sync::worker::SyncWorkerProgress::Sync(
                sync::engine::SyncProgress::PullPageFinished {
                    page_index: 2,
                    page_records: 5000,
                    pulled_records: 10000,
                    max_seq: 123,
                    truncated: true,
                },
            )),
            ..sync::worker::SyncStats::default()
        };
        assert_eq!(
            format_monitor_sync_status_at(
                &running,
                chrono::DateTime::parse_from_rfc3339("2026-05-19T12:00:00Z")
                    .expect("timestamp")
                    .with_timezone(&chrono::Utc),
            ),
            Some("Sync: pull page 2, 10000 records pulled, latest seq 123".to_string())
        );

        let completed = sync::worker::SyncStats {
            success_count: 1,
            last_finished_at: Some("2026-05-19T11:42:00Z".to_string()),
            ..sync::worker::SyncStats::default()
        };
        assert_eq!(
            format_monitor_sync_status_at(
                &completed,
                chrono::DateTime::parse_from_rfc3339("2026-05-19T12:00:00Z")
                    .expect("timestamp")
                    .with_timezone(&chrono::Utc),
            ),
            Some("Sync: checked 18 min ago".to_string())
        );
    }
}
