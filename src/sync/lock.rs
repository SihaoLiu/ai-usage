use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::Path;

const SYNC_LOCK_FILE: &str = "sync.lock";

pub struct SyncLock {
    file: File,
    #[cfg(not(unix))]
    path: std::path::PathBuf,
}

impl SyncLock {
    pub fn try_acquire(cache_root: &Path) -> io::Result<Option<Self>> {
        fs::create_dir_all(cache_root)?;
        let path = cache_root.join(SYNC_LOCK_FILE);
        Self::try_acquire_path(&path)
    }

    #[cfg(unix)]
    fn try_acquire_path(path: &Path) -> io::Result<Option<Self>> {
        use std::os::fd::AsRawFd;

        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if result == 0 {
            return Ok(Some(Self { file }));
        }
        let err = io::Error::last_os_error();
        if matches!(err.raw_os_error(), Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN)
        {
            Ok(None)
        } else {
            Err(err)
        }
    }

    #[cfg(not(unix))]
    fn try_acquire_path(path: &Path) -> io::Result<Option<Self>> {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(file) => Ok(Some(Self {
                file,
                path: path.to_path_buf(),
            })),
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => Ok(None),
            Err(err) => Err(err),
        }
    }
}

#[cfg(unix)]
impl Drop for SyncLock {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;

        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

#[cfg(not(unix))]
impl Drop for SyncLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ai-usage-lock-test-{name}-{stamp}"));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn lock_excludes_other_processes_until_released() {
        let cache_root = unique_temp_dir("exclusive");
        let first = SyncLock::try_acquire(&cache_root)
            .expect("first acquisition")
            .expect("first lock");

        assert!(
            SyncLock::try_acquire(&cache_root)
                .expect("second acquisition")
                .is_none()
        );

        drop(first);
        assert!(
            SyncLock::try_acquire(&cache_root)
                .expect("acquisition after release")
                .is_some()
        );
    }
}
