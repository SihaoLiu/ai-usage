use crate::sync::client::{HttpProgress, SyncHttpClient};
use crate::sync::config::EnabledSyncConfig;
use crate::sync::engine::{self, SyncError, SyncProgress};
use crate::sync::state;
use chrono::Utc;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const MAX_BACKOFF: Duration = Duration::from_secs(60);
const JOIN_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_LOG_BYTES: u64 = 1024 * 1024;
const MAX_PANICS: u32 = 3;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SyncStats {
    pub success_count: u64,
    pub error_count: u64,
    pub revision: u64,
    pub running: bool,
    pub last_started_at: Option<String>,
    pub last_finished_at: Option<String>,
    pub last_error: Option<String>,
    pub progress: Option<SyncWorkerProgress>,
    pub integrity_verification: Option<crate::sync::integrity::IntegrityVerification>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncWorkerProgress {
    Sync(SyncProgress),
    Http(HttpProgress),
}

pub struct SyncWorker {
    shared: Arc<WorkerShared>,
    handle: Option<JoinHandle<()>>,
    join_timeout: Duration,
}

struct WorkerShared {
    inner: Mutex<WorkerInner>,
    condvar: Condvar,
}

#[derive(Default)]
struct WorkerInner {
    requested: bool,
    shutdown: bool,
    stats: SyncStats,
}

#[derive(Clone, Copy)]
struct WorkerSettings {
    initial_backoff: Duration,
    max_backoff: Duration,
    join_timeout: Duration,
    max_panics: u32,
}

impl Default for WorkerSettings {
    fn default() -> Self {
        Self {
            initial_backoff: INITIAL_BACKOFF,
            max_backoff: MAX_BACKOFF,
            join_timeout: JOIN_TIMEOUT,
            max_panics: MAX_PANICS,
        }
    }
}

impl SyncWorker {
    pub fn spawn(cache_root: PathBuf, config: EnabledSyncConfig) -> Self {
        let runner_root = cache_root.clone();
        let runner_config = config.clone();
        Self::spawn_with_shared_settings(cache_root, WorkerSettings::default(), move |shared| {
            let http_shared = Arc::clone(&shared);
            let client = SyncHttpClient::new_with_progress(runner_config.clone(), move |event| {
                http_shared.mark_progress(SyncWorkerProgress::Http(event.clone()));
            });
            engine::run_sync_cycle_with_progress(&runner_root, &runner_config, &client, |event| {
                shared.mark_progress(SyncWorkerProgress::Sync(event.clone()))
            })
        })
    }

    pub fn request_sync(&self) {
        self.shared.request();
    }

    pub fn stats(&self) -> SyncStats {
        self.shared.stats()
    }

    pub fn shutdown(&mut self) {
        self.shared.shutdown();
        if let Some(handle) = self.handle.take() {
            let deadline = Instant::now() + self.join_timeout;
            while !handle.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(10));
            }
            if handle.is_finished() {
                let _ = handle.join();
            }
        }
    }

    #[cfg(test)]
    fn spawn_with_settings<F>(cache_root: PathBuf, settings: WorkerSettings, run_cycle: F) -> Self
    where
        F: FnMut() -> Result<(), SyncError> + Send + 'static,
    {
        let mut run_cycle = run_cycle;
        Self::spawn_with_shared_settings(cache_root, settings, move |_| run_cycle())
    }

    fn spawn_with_shared_settings<F>(
        cache_root: PathBuf,
        settings: WorkerSettings,
        mut run_cycle: F,
    ) -> Self
    where
        F: FnMut(Arc<WorkerShared>) -> Result<(), SyncError> + Send + 'static,
    {
        let shared = Arc::new(WorkerShared::new());
        let worker_shared = Arc::clone(&shared);
        let runner_shared = Arc::clone(&shared);
        let handle = thread::spawn(move || {
            run_worker_loop(worker_shared, cache_root, settings, move || {
                run_cycle(Arc::clone(&runner_shared))
            });
        });
        Self {
            shared,
            handle: Some(handle),
            join_timeout: settings.join_timeout,
        }
    }
}

impl Drop for SyncWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl WorkerShared {
    fn new() -> Self {
        Self {
            inner: Mutex::new(WorkerInner::default()),
            condvar: Condvar::new(),
        }
    }

    fn request(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|err| err.into_inner());
        inner.requested = true;
        self.condvar.notify_one();
    }

    fn shutdown(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|err| err.into_inner());
        inner.shutdown = true;
        self.condvar.notify_one();
    }

    fn stats(&self) -> SyncStats {
        self.inner
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .stats
            .clone()
    }

    fn wait_for_request(&self) -> bool {
        let mut inner = self.inner.lock().unwrap_or_else(|err| err.into_inner());
        while !inner.requested && !inner.shutdown {
            inner = self
                .condvar
                .wait(inner)
                .unwrap_or_else(|err| err.into_inner());
        }
        if inner.shutdown {
            return false;
        }
        inner.requested = false;
        true
    }

    fn wait_until_or_shutdown(&self, deadline: Instant) -> bool {
        let mut inner = self.inner.lock().unwrap_or_else(|err| err.into_inner());
        while !inner.shutdown {
            let now = Instant::now();
            if now >= deadline {
                inner.requested = false;
                return true;
            }
            let timeout = deadline.saturating_duration_since(now);
            let (next_inner, _) = self
                .condvar
                .wait_timeout(inner, timeout)
                .unwrap_or_else(|err| err.into_inner());
            inner = next_inner;
        }
        false
    }

    fn mark_running(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|err| err.into_inner());
        inner.stats.running = true;
        inner.stats.revision += 1;
        inner.stats.last_started_at = Some(Utc::now().to_rfc3339());
        inner.stats.progress = None;
    }

    fn mark_success(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|err| err.into_inner());
        inner.stats.running = false;
        inner.stats.success_count += 1;
        inner.stats.revision += 1;
        inner.stats.last_finished_at = Some(Utc::now().to_rfc3339());
        inner.stats.last_error = None;
        inner.stats.progress = None;
    }

    fn mark_error(&self, message: String) {
        let mut inner = self.inner.lock().unwrap_or_else(|err| err.into_inner());
        inner.stats.running = false;
        inner.stats.error_count += 1;
        inner.stats.revision += 1;
        inner.stats.last_finished_at = Some(Utc::now().to_rfc3339());
        inner.stats.last_error = Some(message);
        inner.stats.progress = None;
    }

    fn mark_progress(&self, progress: SyncWorkerProgress) {
        let mut inner = self.inner.lock().unwrap_or_else(|err| err.into_inner());
        if let SyncWorkerProgress::Sync(SyncProgress::IntegrityCheckFinished { verification }) =
            &progress
        {
            inner.stats.integrity_verification = Some(verification.clone());
        }
        inner.stats.progress = Some(progress);
        inner.stats.revision += 1;
    }
}

fn run_worker_loop<F>(
    shared: Arc<WorkerShared>,
    cache_root: PathBuf,
    settings: WorkerSettings,
    mut run_cycle: F,
) where
    F: FnMut() -> Result<(), SyncError>,
{
    let mut backoff = settings.initial_backoff;
    let mut retry_after: Option<Instant> = None;
    let mut panic_count = 0;

    loop {
        if !shared.wait_for_request() {
            break;
        }

        if let Some(deadline) = retry_after
            && !shared.wait_until_or_shutdown(deadline)
        {
            break;
        }

        shared.mark_running();
        let result = panic::catch_unwind(AssertUnwindSafe(&mut run_cycle));
        match result {
            Ok(Ok(())) => {
                record_success(&cache_root);
                shared.mark_success();
                backoff = settings.initial_backoff;
                retry_after = None;
            }
            Ok(Err(err)) => {
                let message = err.to_string();
                record_failure(&cache_root, &message);
                shared.mark_error(message);
                retry_after = Some(Instant::now() + backoff);
                backoff = next_backoff(backoff, settings.max_backoff);
            }
            Err(_) => {
                panic_count += 1;
                let message = "sync worker panicked".to_string();
                record_failure(&cache_root, &message);
                shared.mark_error(message);
                if panic_count > settings.max_panics {
                    break;
                }
                retry_after = Some(Instant::now() + backoff);
                backoff = next_backoff(backoff, settings.max_backoff);
            }
        }
    }
}

fn next_backoff(current: Duration, max: Duration) -> Duration {
    current.saturating_mul(2).min(max)
}

fn record_failure(cache_root: &Path, message: &str) {
    let mut sync_state = state::load_sync_state(cache_root);
    sync_state.last_error = Some(message.to_string());
    if let Err(err) = state::save_sync_state(cache_root, &sync_state) {
        let _ = append_log(cache_root, &format!("failed to persist sync error: {err}"));
    }
    let _ = append_log(cache_root, message);
}

fn record_success(cache_root: &Path) {
    let mut sync_state = state::load_sync_state(cache_root);
    sync_state.last_error = None;
    if let Err(err) = state::save_sync_state(cache_root, &sync_state) {
        let _ = append_log(
            cache_root,
            &format!("failed to persist sync success: {err}"),
        );
    }
}

fn append_log(cache_root: &Path, message: &str) -> std::io::Result<()> {
    fs::create_dir_all(cache_root)?;
    let log_path = cache_root.join("sync.log");
    if fs::metadata(&log_path)
        .map(|metadata| metadata.len() >= MAX_LOG_BYTES)
        .unwrap_or(false)
    {
        let rotated_path = cache_root.join("sync.log.1");
        let _ = fs::remove_file(&rotated_path);
        fs::rename(&log_path, rotated_path)?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    writeln!(file, "[{}] {}", Utc::now().to_rfc3339(), message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ai-usage-worker-test-{name}-{stamp}"));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn test_settings() -> WorkerSettings {
        WorkerSettings {
            initial_backoff: Duration::from_millis(5),
            max_backoff: Duration::from_millis(20),
            join_timeout: Duration::from_secs(1),
            max_panics: 3,
        }
    }

    fn wait_until(mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if done() {
                return;
            }
            thread::sleep(Duration::from_millis(5));
        }
        panic!("condition was not reached");
    }

    #[test]
    fn request_signal_coalesces_pending_requests() {
        let shared = WorkerShared::new();

        shared.request();
        shared.request();
        shared.request();

        assert!(shared.wait_for_request());
        let inner = shared.inner.lock().unwrap();
        assert!(!inner.requested);
    }

    #[test]
    fn next_backoff_caps_at_maximum() {
        assert_eq!(
            next_backoff(Duration::from_secs(1), Duration::from_secs(60)),
            Duration::from_secs(2)
        );
        assert_eq!(
            next_backoff(Duration::from_secs(40), Duration::from_secs(60)),
            Duration::from_secs(60)
        );
    }

    #[test]
    fn worker_records_failures_and_retries_after_backoff() {
        let cache_root = unique_temp_dir("retry");
        let runs = Arc::new(AtomicUsize::new(0));
        let run_counter = Arc::clone(&runs);
        let mut worker =
            SyncWorker::spawn_with_settings(cache_root.clone(), test_settings(), move || {
                let run = run_counter.fetch_add(1, Ordering::SeqCst);
                if run == 0 {
                    return Err(SyncError::new("temporary failure"));
                }
                Ok(())
            });

        worker.request_sync();
        wait_until(|| worker.stats().error_count == 1);
        worker.request_sync();
        wait_until(|| worker.stats().success_count == 1);

        let stats = worker.stats();
        assert_eq!(stats.error_count, 1);
        assert_eq!(stats.success_count, 1);
        assert_eq!(state::load_sync_state(&cache_root).last_error, None);
        worker.shutdown();
    }

    #[test]
    fn worker_survives_caught_panics() {
        let cache_root = unique_temp_dir("panic");
        let runs = Arc::new(AtomicUsize::new(0));
        let run_counter = Arc::clone(&runs);
        let mut worker =
            SyncWorker::spawn_with_settings(cache_root.clone(), test_settings(), move || {
                let run = run_counter.fetch_add(1, Ordering::SeqCst);
                if run == 0 {
                    panic!("worker test panic");
                }
                Ok(())
            });

        worker.request_sync();
        wait_until(|| worker.stats().error_count == 1);
        worker.request_sync();
        wait_until(|| worker.stats().success_count == 1);

        let stats = worker.stats();
        assert_eq!(stats.error_count, 1);
        assert_eq!(stats.success_count, 1);
        worker.shutdown();
    }

    #[test]
    fn worker_preserves_integrity_verification_after_success() {
        let shared = WorkerShared::new();
        let verification =
            crate::sync::integrity::IntegrityVerification::Checked { checked_hosts: 1 };

        shared.mark_running();
        shared.mark_progress(SyncWorkerProgress::Sync(
            SyncProgress::IntegrityCheckFinished {
                verification: verification.clone(),
            },
        ));
        shared.mark_success();

        assert_eq!(shared.stats().integrity_verification, Some(verification));
    }
}
