//! Operating-system seam for the rank daemon: byte-stream links, the
//! monotonic clock, and the wake pipe.
//!
//! The trait pattern follows the crate's other seams (`KvPrefetchBackend`,
//! `SwitchTier`, `SwapDevice`): the daemon state machine depends on [`Link`]
//! / clock functions, tests substitute scripted fakes, and the production
//! wiring uses the `std`/`libc` implementations below. `unsafe` is confined
//! to the documented libc calls at the bottom (the crate rule: no unsafe
//! except fd/signal/process libc calls).

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::os::unix::io::RawFd;
use std::os::unix::net::UnixStream;

use super::status::{Result, SparkStatus};

/// A bidirectional byte stream (work-control socket, final-event socket, or
/// the CUDA-resident Unix socket). `read`/`write` report
/// [`io::ErrorKind::WouldBlock`] when the link is nonblocking and not ready.
pub trait Link {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize>;
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize>;
    fn set_nonblocking(&mut self, nonblocking: bool) -> io::Result<()>;

    /// C `SparkRingDaemonWriteAll`: loop on partial writes; `WouldBlock`
    /// maps to [`SparkStatus::Busy`], other errors to [`SparkStatus::IoError`].
    fn write_all_status(&mut self, mut buffer: &[u8]) -> Result<()> {
        while !buffer.is_empty() {
            match self.write(buffer) {
                Ok(0) => return Err(SparkStatus::IoError),
                Ok(written) => buffer = &buffer[written..],
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    return Err(SparkStatus::Busy)
                }
                Err(_) => return Err(SparkStatus::IoError),
            }
        }
        Ok(())
    }
}

/// Maps an `io::Error` to the C read-path statuses (`WouldBlock` → `BUSY`,
/// everything else → `IO_ERROR`).
pub fn io_status(error: &io::Error) -> SparkStatus {
    match error.kind() {
        io::ErrorKind::WouldBlock => SparkStatus::Busy,
        _ => SparkStatus::IoError,
    }
}

impl Link for UnixStream {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        Read::read(self, buffer)
    }

    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        Write::write(self, buffer)
    }

    fn set_nonblocking(&mut self, nonblocking: bool) -> io::Result<()> {
        UnixStream::set_nonblocking(self, nonblocking)
    }
}

impl Link for TcpStream {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        Read::read(self, buffer)
    }

    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        Write::write(self, buffer)
    }

    fn set_nonblocking(&mut self, nonblocking: bool) -> io::Result<()> {
        TcpStream::set_nonblocking(self, nonblocking)
    }
}

/// C: `SparkNetMonotonicNs` (`clock_gettime(CLOCK_MONOTONIC)`; the C returns
/// 0 when the clock is unavailable and so does this).
pub fn monotonic_ns() -> u64 {
    // SAFETY: `clock_gettime` with a valid clock id writes the pointed-to
    // `timespec`; `timespec` is zero-initialized and plain-old-data.
    unsafe {
        let mut time: libc::timespec = std::mem::zeroed();
        if libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) != 0 {
            return 0;
        }
        time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64
    }
}

/// C: `SparkNetConfigureLowLatencyTcp` (TCP_NODELAY + SO_KEEPALIVE).
/// Best-effort like the C (which reports `-232`/`-233` on failure; here an
/// `Err(IoError)`).
pub fn configure_low_latency_tcp(stream: &TcpStream) -> Result<()> {
    use std::os::unix::io::AsRawFd;
    let fd = stream.as_raw_fd();
    // SAFETY: `setsockopt` on a live socket fd with a valid option value
    // pointer; `on` outlives the call.
    unsafe {
        let on: libc::c_int = 1;
        let no_delay = libc::setsockopt(
            fd,
            libc::IPPROTO_TCP,
            libc::TCP_NODELAY,
            &on as *const libc::c_int as *const libc::c_void,
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        );
        if no_delay != 0 {
            return Err(SparkStatus::IoError);
        }
        let keepalive = libc::setsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_KEEPALIVE,
            &on as *const libc::c_int as *const libc::c_void,
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        );
        if keepalive != 0 {
            return Err(SparkStatus::IoError);
        }
    }
    Ok(())
}

/// C: the daemon's wake pipe (`SparkRingDaemonOpenWakePipe`): both ends
/// nonblocking; [`WakePipe::wake`] writes one byte (EAGAIN is fine — the
/// pipe is already readable), [`WakePipe::drain`] consumes pending bytes.
pub struct WakePipe {
    read_fd: RawFd,
    write_fd: RawFd,
}

impl WakePipe {
    pub fn new() -> Result<Self> {
        let mut fds = [-1 as RawFd; 2];
        // SAFETY: `pipe` writes two fds into `fds`; both are then set
        // nonblocking via `fcntl` before use, and owned by the returned
        // struct (closed in `Drop`).
        unsafe {
            if libc::pipe(fds.as_mut_ptr()) != 0 {
                return Err(SparkStatus::IoError);
            }
            for &fd in &fds {
                if libc::fcntl(fd, libc::F_SETFL, libc::O_NONBLOCK) != 0 {
                    libc::close(fds[0]);
                    libc::close(fds[1]);
                    return Err(SparkStatus::IoError);
                }
            }
        }
        Ok(Self { read_fd: fds[0], write_fd: fds[1] })
    }

    /// Raw fd of the read end (for the poll loop).
    pub fn read_fd(&self) -> RawFd {
        self.read_fd
    }

    pub fn wake(&self) {
        let byte = [1u8; 1];
        // SAFETY: one-byte write to the owned pipe write end; failure
        // (including EAGAIN when the pipe is full) is benign.
        unsafe {
            libc::write(self.write_fd, byte.as_ptr() as *const libc::c_void, 1);
        }
    }

    /// Returns the number of pending wake bytes consumed.
    pub fn drain(&self) -> usize {
        let mut buffer = [0u8; 64];
        let mut drained = 0usize;
        loop {
            // SAFETY: read into the stack buffer from the owned pipe read
            // end; the fd is valid for the struct's lifetime.
            let count = unsafe {
                libc::read(self.read_fd, buffer.as_mut_ptr() as *mut libc::c_void, buffer.len())
            };
            if count <= 0 {
                break;
            }
            drained += count as usize;
            if (count as usize) < buffer.len() {
                break;
            }
        }
        drained
    }
}

impl Drop for WakePipe {
    fn drop(&mut self) {
        // SAFETY: both fds are owned by this struct and closed exactly once.
        unsafe {
            libc::close(self.read_fd);
            libc::close(self.write_fd);
        }
    }
}

/// C: `SparkRingDaemonMinNonzeroNs`.
pub fn min_nonzero_ns(left: u64, right: u64) -> u64 {
    if left == 0 {
        return right;
    }
    if right == 0 {
        return left;
    }
    left.min(right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_clock_advances() {
        let first = monotonic_ns();
        let second = monotonic_ns();
        assert!(first > 0);
        assert!(second >= first);
    }

    #[test]
    fn wake_pipe_round_trip() {
        let pipe = WakePipe::new().unwrap();
        assert_eq!(pipe.drain(), 0);
        pipe.wake();
        pipe.wake();
        assert!(pipe.drain() >= 1);
        assert_eq!(pipe.drain(), 0);
    }

    #[test]
    fn unix_stream_link_nonblocking() {
        let (mut left, mut right) = UnixStream::pair().unwrap();
        left.set_nonblocking(true).unwrap();
        right.set_nonblocking(true).unwrap();
        // WouldBlock maps to BUSY on an empty read.
        let mut buffer = [0u8; 8];
        let error = Link::read(&mut left, &mut buffer).unwrap_err();
        assert_eq!(io_status(&error), SparkStatus::Busy);
        Link::write_all_status(&mut right, b"abc").unwrap();
        let count = Link::read(&mut left, &mut buffer).unwrap();
        assert_eq!(&buffer[..count], b"abc");
    }
}
