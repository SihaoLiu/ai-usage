use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const SAMPLE_INTERVAL: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProcessUsage {
    pub cpu_percent: f32,
    pub memory_bytes: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ProcessUsageDisplay {
    full: String,
    compact: String,
}

impl ProcessUsageDisplay {
    pub(crate) fn new(usage: ProcessUsage) -> Self {
        Self {
            full: format!(
                "CPU: {:.1}%  |  Mem: {}",
                usage.cpu_percent,
                format_memory(usage.memory_bytes, false)
            ),
            compact: format!(
                "CPU:{:.1}% | Mem:{}",
                usage.cpu_percent,
                format_memory(usage.memory_bytes, true)
            ),
        }
    }

    pub fn full(&self) -> &str {
        &self.full
    }

    pub fn compact(&self) -> &str {
        &self.compact
    }
}

pub type ProcessUsageSnapshot = Arc<ProcessUsageDisplay>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProcessCounters {
    cpu_time: Duration,
    memory_bytes: u64,
}

pub struct ProcessUsageMonitor {
    snapshot: Arc<Mutex<Option<ProcessUsageSnapshot>>>,
    stop: Option<mpsc::Sender<()>>,
    worker: Option<JoinHandle<()>>,
}

impl ProcessUsageMonitor {
    pub fn start() -> Self {
        let snapshot = Arc::new(Mutex::new(None));
        let worker_snapshot = Arc::clone(&snapshot);
        let (stop, stop_receiver) = mpsc::channel();
        let worker = std::thread::Builder::new()
            .name("ai-usage-resources".to_string())
            .spawn(move || sample_current_process(worker_snapshot, stop_receiver))
            .ok();

        Self {
            snapshot,
            stop: worker.as_ref().map(|_| stop),
            worker,
        }
    }

    pub fn snapshot(&self) -> Option<ProcessUsageSnapshot> {
        self.snapshot
            .lock()
            .ok()
            .and_then(|snapshot| snapshot.clone())
    }
}

impl Drop for ProcessUsageMonitor {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn sample_current_process(
    snapshot: Arc<Mutex<Option<ProcessUsageSnapshot>>>,
    stop: mpsc::Receiver<()>,
) {
    let Some(mut previous) = platform::read_process_counters() else {
        return;
    };
    let mut previous_at = Instant::now();
    publish_usage(
        &snapshot,
        ProcessUsage {
            cpu_percent: 0.0,
            memory_bytes: previous.memory_bytes,
        },
    );

    loop {
        match stop.recv_timeout(SAMPLE_INTERVAL) {
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }

        let sampled_at = Instant::now();
        if let Some(current) = platform::read_process_counters() {
            publish_usage(
                &snapshot,
                usage_between(previous, current, sampled_at.duration_since(previous_at)),
            );
            previous = current;
            previous_at = sampled_at;
        }
    }
}

fn publish_usage(snapshot: &Mutex<Option<ProcessUsageSnapshot>>, usage: ProcessUsage) {
    if let Ok(mut current) = snapshot.lock() {
        *current = Some(Arc::new(ProcessUsageDisplay::new(usage)));
    }
}

fn usage_between(
    previous: ProcessCounters,
    current: ProcessCounters,
    elapsed: Duration,
) -> ProcessUsage {
    let elapsed_seconds = elapsed.as_secs_f64();
    let cpu_percent = if elapsed_seconds > 0.0 {
        current
            .cpu_time
            .saturating_sub(previous.cpu_time)
            .as_secs_f64()
            / elapsed_seconds
            * 100.0
    } else {
        0.0
    };
    ProcessUsage {
        cpu_percent: cpu_percent as f32,
        memory_bytes: current.memory_bytes,
    }
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
fn current_process_cpu_time() -> Option<Duration> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    let usage = unsafe { usage.assume_init() };
    timeval_duration(usage.ru_utime)?.checked_add(timeval_duration(usage.ru_stime)?)
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
fn timeval_duration(value: libc::timeval) -> Option<Duration> {
    let seconds = u64::try_from(value.tv_sec).ok()?;
    let microseconds = u32::try_from(value.tv_usec).ok()?;
    (microseconds < 1_000_000).then(|| Duration::new(seconds, microseconds * 1_000))
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod platform {
    use super::{ProcessCounters, current_process_cpu_time};

    pub(super) fn read_process_counters() -> Option<ProcessCounters> {
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let resident_pages = statm.split_whitespace().nth(1)?.parse::<u64>().ok()?;
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size <= 0 {
            return None;
        }
        Some(ProcessCounters {
            cpu_time: current_process_cpu_time()?,
            memory_bytes: resident_pages.checked_mul(page_size as u64)?,
        })
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use std::ffi::c_void;

    use super::{ProcessCounters, current_process_cpu_time};

    pub(super) fn read_process_counters() -> Option<ProcessCounters> {
        let mut task_info = std::mem::MaybeUninit::<libc::proc_taskinfo>::zeroed();
        let expected = std::mem::size_of::<libc::proc_taskinfo>();
        let read = unsafe {
            libc::proc_pidinfo(
                libc::getpid(),
                libc::PROC_PIDTASKINFO,
                0,
                task_info.as_mut_ptr().cast::<c_void>(),
                expected as i32,
            )
        };
        if read != expected as i32 {
            return None;
        }
        let task_info = unsafe { task_info.assume_init() };
        Some(ProcessCounters {
            cpu_time: current_process_cpu_time()?,
            memory_bytes: task_info.pti_resident_size,
        })
    }
}

#[cfg(windows)]
mod platform {
    use std::mem::{size_of, zeroed};
    use std::time::Duration;

    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    use super::ProcessCounters;

    pub(super) fn read_process_counters() -> Option<ProcessCounters> {
        let process = unsafe { GetCurrentProcess() };
        let mut creation: FILETIME = unsafe { zeroed() };
        let mut exit: FILETIME = unsafe { zeroed() };
        let mut kernel: FILETIME = unsafe { zeroed() };
        let mut user: FILETIME = unsafe { zeroed() };
        if unsafe { GetProcessTimes(process, &mut creation, &mut exit, &mut kernel, &mut user) }
            == 0
        {
            return None;
        }

        let mut memory: PROCESS_MEMORY_COUNTERS = unsafe { zeroed() };
        memory.cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        if unsafe { K32GetProcessMemoryInfo(process, &mut memory, memory.cb) } == 0 {
            return None;
        }

        let cpu_ticks = filetime_ticks(kernel).checked_add(filetime_ticks(user))?;
        Some(ProcessCounters {
            cpu_time: Duration::from_nanos(cpu_ticks.checked_mul(100)?),
            memory_bytes: memory.WorkingSetSize as u64,
        })
    }

    fn filetime_ticks(value: FILETIME) -> u64 {
        (u64::from(value.dwHighDateTime) << 32) | u64::from(value.dwLowDateTime)
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    windows
)))]
mod platform {
    use super::ProcessCounters;

    pub(super) fn read_process_counters() -> Option<ProcessCounters> {
        None
    }
}

pub fn process_usage_text(usage: Option<&ProcessUsageDisplay>, compact: bool) -> &str {
    match (usage, compact) {
        (Some(usage), true) => usage.compact(),
        (Some(usage), false) => usage.full(),
        (None, true) => "CPU:-- | Mem:--",
        (None, false) => "CPU: --  |  Mem: --",
    }
}

fn format_memory(bytes: u64, compact: bool) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;

    if bytes >= GIB {
        format!(
            "{:.2}{}",
            bytes as f64 / GIB as f64,
            if compact { "G" } else { " GiB" }
        )
    } else if bytes >= MIB {
        format!(
            "{:.2}{}",
            bytes as f64 / MIB as f64,
            if compact { "M" } else { " MiB" }
        )
    } else if bytes >= KIB {
        format!(
            "{:.2}{}",
            bytes as f64 / KIB as f64,
            if compact { "K" } else { " KiB" }
        )
    } else {
        format!("{bytes}{}", if compact { "B" } else { " B" })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_formats_cpu_and_resident_memory() {
        let usage = ProcessUsage {
            cpu_percent: 12.34,
            memory_bytes: 1_331_438_182,
        };

        let display = ProcessUsageDisplay::new(usage);
        assert_eq!(display.full(), "CPU: 12.3%  |  Mem: 1.24 GiB");
        assert_eq!(display.compact(), "CPU:12.3% | Mem:1.24G");
    }

    #[test]
    fn status_has_an_explicit_unavailable_state() {
        assert_eq!(process_usage_text(None, false), "CPU: --  |  Mem: --");
        assert_eq!(process_usage_text(None, true), "CPU:-- | Mem:--");
    }

    #[test]
    fn resident_memory_uses_binary_units() {
        assert_eq!(format_memory(1023, false), "1023 B");
        assert_eq!(format_memory(1024, false), "1.00 KiB");
        assert_eq!(format_memory(1024 * 1024, false), "1.00 MiB");
        assert_eq!(format_memory(1024 * 1024 * 1024, false), "1.00 GiB");
    }

    #[test]
    fn cpu_percentage_uses_current_process_time_delta() {
        let before = ProcessCounters {
            cpu_time: Duration::from_millis(100),
            memory_bytes: 1,
        };
        let after = ProcessCounters {
            cpu_time: Duration::from_millis(850),
            memory_bytes: 42,
        };

        let usage = usage_between(before, after, Duration::from_millis(250));

        assert_eq!(usage.cpu_percent, 300.0);
        assert_eq!(usage.memory_bytes, 42);
    }

    #[test]
    fn monitor_publishes_a_current_process_snapshot() {
        let monitor = ProcessUsageMonitor::start();
        let deadline = Instant::now() + Duration::from_secs(2);

        let usage = loop {
            if let Some(usage) = monitor.snapshot() {
                break usage;
            }
            assert!(
                Instant::now() < deadline,
                "process usage snapshot timed out"
            );
            std::thread::sleep(Duration::from_millis(10));
        };

        assert!(usage.full().starts_with("CPU: "));
        assert!(usage.full().contains("  |  Mem: "));
        assert!(!usage.full().contains("Mem: 0 B"));
    }
}
