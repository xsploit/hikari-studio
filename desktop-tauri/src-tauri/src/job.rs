//! Windows Job Object that kills every process in it when the app exits.
//!
//! KILL_ON_JOB_CLOSE means the kernel kills the backend and engine when our
//! last handle closes, which also happens if this app crashes or is killed.
//! Child processes (server.py -> hikari-live-engine.exe) join the job
//! automatically.
//!
//! Every Win32 call is checked and failures come back as `io::Error` (with
//! the OS error code), so the caller can log them and fall back instead of
//! silently losing the cleanup guarantee.

#[cfg(windows)]
mod imp {
    use std::{io, mem::size_of, ptr::null};
    use windows_sys::Win32::{
        Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE},
        System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
            SetInformationJobObject, TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        },
    };

    /// A valid job handle, stored as an integer so the struct is Send + Sync.
    pub struct Job(usize);

    impl Job {
        pub fn new() -> io::Result<Self> {
            unsafe {
                let h = CreateJobObjectW(null(), null());
                if h.is_null() || h == INVALID_HANDLE_VALUE {
                    return Err(with_context("CreateJobObjectW"));
                }
                // From here on, Drop closes the handle on any early return.
                let job = Job(h as usize);
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let ok = SetInformationJobObject(
                    job.handle(),
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const _,
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );
                if ok == 0 {
                    return Err(with_context("SetInformationJobObject(KILL_ON_JOB_CLOSE)"));
                }
                Ok(job)
            }
        }

        /// Put a freshly spawned child (and so all its future children) in the job.
        pub fn assign(&self, child: &tokio::process::Child) -> io::Result<()> {
            let Some(ph) = child.raw_handle() else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "AssignProcessToJobObject: child has already exited",
                ));
            };
            if unsafe { AssignProcessToJobObject(self.handle(), ph as HANDLE) } == 0 {
                return Err(with_context("AssignProcessToJobObject"));
            }
            Ok(())
        }

        /// Kill every process in the job now. The job stays usable.
        pub fn kill_all(&self) -> io::Result<()> {
            if unsafe { TerminateJobObject(self.handle(), 1) } == 0 {
                return Err(with_context("TerminateJobObject"));
            }
            Ok(())
        }

        fn handle(&self) -> HANDLE {
            self.0 as HANDLE
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            unsafe { CloseHandle(self.handle()) };
        }
    }

    /// The last OS error, prefixed with the call that failed.
    fn with_context(call: &str) -> io::Error {
        let e = io::Error::last_os_error();
        io::Error::new(e.kind(), format!("{call} failed: {e}"))
    }

    #[cfg(test)]
    impl Job {
        pub fn active_processes(&self) -> u32 {
            use windows_sys::Win32::System::JobObjects::{
                JobObjectBasicAccountingInformation, QueryInformationJobObject,
                JOBOBJECT_BASIC_ACCOUNTING_INFORMATION,
            };
            unsafe {
                let mut info: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = std::mem::zeroed();
                let ok = QueryInformationJobObject(
                    self.handle(),
                    JobObjectBasicAccountingInformation,
                    &mut info as *mut _ as *mut _,
                    size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
                    std::ptr::null_mut(),
                );
                assert_ne!(ok, 0, "QueryInformationJobObject failed");
                info.ActiveProcesses
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::Job;
        use std::time::Duration;
        use windows_sys::Win32::System::Threading::CREATE_NO_WINDOW;

        /// cmd.exe -> ping.exe mirrors python -> hikari-live-engine.exe:
        /// the grandchild must land in the job and die with kill_all().
        #[tokio::test]
        async fn kills_child_and_grandchild() {
            let job = Job::new().expect("job creation");
            let mut child = tokio::process::Command::new("cmd")
                .args(["/c", "ping -n 60 127.0.0.1 >nul"])
                .creation_flags(CREATE_NO_WINDOW)
                .spawn()
                .unwrap();
            job.assign(&child).expect("assign");
            tokio::time::sleep(Duration::from_millis(700)).await;
            assert!(job.active_processes() >= 2, "cmd + ping (+ conhost) should all be in the job");
            job.kill_all().expect("kill_all");
            child.wait().await.unwrap();
            tokio::time::sleep(Duration::from_millis(200)).await;
            assert_eq!(job.active_processes(), 0, "nothing may survive kill_all");
        }

        /// Dropping the job (what happens if the shell crashes) kills the tree too.
        #[tokio::test]
        async fn closing_the_handle_kills_the_tree() {
            let job = Job::new().expect("job creation");
            let mut child = tokio::process::Command::new("cmd")
                .args(["/c", "ping -n 60 127.0.0.1 >nul"])
                .creation_flags(CREATE_NO_WINDOW)
                .spawn()
                .unwrap();
            job.assign(&child).expect("assign");
            drop(job);
            // ping alone would run ~59 s. The kernel's kill-on-close exit code
            // is 0, so timing is what proves the kill.
            tokio::time::timeout(Duration::from_secs(5), child.wait())
                .await
                .expect("child must die when the job handle closes")
                .unwrap();
        }

        /// A failed assignment is reported, not swallowed.
        #[tokio::test]
        async fn assign_after_exit_is_an_error() {
            let job = Job::new().expect("job creation");
            let mut child = tokio::process::Command::new("cmd")
                .args(["/c", "exit 0"])
                .creation_flags(CREATE_NO_WINDOW)
                .spawn()
                .unwrap();
            child.wait().await.unwrap();
            assert!(job.assign(&child).is_err());
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use std::io;

    /// Non-Windows: no job objects; children are still killed via kill_on_drop.
    pub struct Job;
    impl Job {
        pub fn new() -> io::Result<Self> {
            Err(io::Error::new(io::ErrorKind::Unsupported, "job objects are Windows-only"))
        }
        pub fn assign(&self, _child: &tokio::process::Child) -> io::Result<()> {
            Ok(())
        }
        pub fn kill_all(&self) -> io::Result<()> {
            Ok(())
        }
    }
}

pub use imp::Job;
