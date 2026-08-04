//! Nonblocking socket primitives — port of `runtime/net.h` plus the
//! socket/poll idioms `node/backend.c` uses directly.
//!
//! This is the ONLY module in the backend with `unsafe` code; every wrapper
//! is a thin, individually documented libc call. Everything is nonblocking;
//! there is deliberately no async runtime in `spark-node`.

use std::ffi::CString;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::io::{AsRawFd, FromRawFd, OwnedFd, RawFd};

use spark_serve::serving_engine::ServingStatus;

/// `SPARK_NET_LISTEN_BACKLOG`.
pub const LISTEN_BACKLOG: i32 = 64;

/// `SparkNetSetNonblocking`.
pub fn set_nonblocking(fd: RawFd) -> Result<(), ServingStatus> {
    // SAFETY: `fcntl` on any valid fd; flags are only read and OR-ed back.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL, 0) };
    if flags < 0 {
        return Err(ServingStatus::IoError);
    }
    // SAFETY: same fd, setting O_NONBLOCK in addition to existing flags.
    if unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(ServingStatus::IoError);
    }
    Ok(())
}

/// `SparkNetConfigureLowLatencyTcp` (TCP_NODELAY + SO_KEEPALIVE).
pub fn configure_low_latency_tcp(fd: RawFd) -> Result<(), ServingStatus> {
    let enabled: libc::c_int = 1;
    // SAFETY: setsockopt with a valid pointer to `enabled` for both calls.
    let nodelay = unsafe {
        libc::setsockopt(
            fd,
            libc::IPPROTO_TCP,
            libc::TCP_NODELAY,
            (&enabled as *const libc::c_int).cast(),
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        )
    };
    if nodelay < 0 {
        return Err(ServingStatus::IoError);
    }
    // SAFETY: same as above.
    let keepalive = unsafe {
        libc::setsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_KEEPALIVE,
            (&enabled as *const libc::c_int).cast(),
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        )
    };
    if keepalive < 0 {
        return Err(ServingStatus::IoError);
    }
    Ok(())
}

/// `SparkNetMonotonicNs` (0 when the clock is unavailable, as in C).
pub fn monotonic_ns() -> u64 {
    let mut timestamp = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: `timestamp` is a valid out-pointer; CLOCK_MONOTONIC always exists.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut timestamp) } != 0 {
        return 0;
    }
    (timestamp.tv_sec as u64) * 1_000_000_000 + (timestamp.tv_nsec as u64)
}

/// `SparkNetCreateListenSocket` (AF_INET, SO_REUSEADDR, bind, listen).
pub fn create_listen_socket(bind_address: &str, port: u32) -> Result<OwnedFd, ServingStatus> {
    let address_text = CString::new(bind_address).map_err(|_| ServingStatus::InvalidArgument)?;
    // SAFETY: socket creation; no preconditions.
    let fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
    if fd < 0 {
        return Err(ServingStatus::RouteNotFound);
    }
    let result = configure_listen_socket(fd, &address_text, port);
    if let Err(status) = result {
        // SAFETY: fd is valid and owned here on the error path.
        unsafe { libc::close(fd) };
        return Err(status);
    }
    // SAFETY: fd is valid, open, and uniquely owned here.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

fn configure_listen_socket(
    fd: RawFd,
    address_text: &CString,
    port: u32,
) -> Result<(), ServingStatus> {
    let option: libc::c_int = 1;
    // SAFETY: valid pointer to `option`.
    if unsafe {
        libc::setsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_REUSEADDR,
            (&option as *const libc::c_int).cast(),
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        )
    } < 0
    {
        return Err(ServingStatus::InternalError);
    }
    let mut address: libc::sockaddr_in = unsafe { std::mem::zeroed() };
    address.sin_family = libc::AF_INET as libc::sa_family_t;
    address.sin_port = (port as u16).to_be();
    let ipv4: std::net::Ipv4Addr = address_text
        .to_str()
        .map_err(|_| ServingStatus::InvalidArgument)?
        .parse()
        .map_err(|_| ServingStatus::InvalidArgument)?;
    address.sin_addr.s_addr = u32::from(ipv4).to_be();
    // SAFETY: `address` is a fully initialized sockaddr_in.
    if unsafe {
        libc::bind(
            fd,
            (&address as *const libc::sockaddr_in).cast(),
            std::mem::size_of::<libc::sockaddr_in>() as libc::socklen_t,
        )
    } < 0
    {
        return Err(ServingStatus::RouteNotFound);
    }
    // SAFETY: fd is a bound stream socket.
    if unsafe { libc::listen(fd, LISTEN_BACKLOG) } < 0 {
        return Err(ServingStatus::RouteNotFound);
    }
    Ok(())
}

/// `SparkRingServiceBackendStartConnectToAddress` over `getaddrinfo`
/// (`SparkRingServiceBackendConnectSocket`): nonblocking connect of the first
/// AF_INET/SOCK_STREAM address that starts. Returns `(fd, connecting)` —
/// `connecting` is true when the connect is in flight (EINPROGRESS/EALREADY).
pub fn connect_socket(host: &str, port: u32) -> Result<(OwnedFd, bool), ServingStatus> {
    let host_text = CString::new(host).map_err(|_| ServingStatus::InvalidArgument)?;
    let service_text =
        CString::new(port.to_string()).map_err(|_| ServingStatus::InvalidArgument)?;
    let mut hints: libc::addrinfo = unsafe { std::mem::zeroed() };
    hints.ai_family = libc::AF_INET;
    hints.ai_socktype = libc::SOCK_STREAM;
    let mut results: *mut libc::addrinfo = std::ptr::null_mut();
    // SAFETY: valid C strings and out-pointer; `results` freed below on all paths.
    let gai_status = unsafe {
        libc::getaddrinfo(host_text.as_ptr(), service_text.as_ptr(), &hints, &mut results)
    };
    if gai_status != 0 {
        return Err(ServingStatus::Busy);
    }
    let mut found: Option<(OwnedFd, bool)> = None;
    let mut entry = results;
    while !entry.is_null() {
        // SAFETY: `entry` is a live node of the getaddrinfo list.
        let info = unsafe { &*entry };
        if let Some(connected) = start_connect_to_address(info) {
            found = Some(connected);
            break;
        }
        entry = info.ai_next;
    }
    // SAFETY: `results` was returned by getaddrinfo above and not yet freed.
    unsafe { libc::freeaddrinfo(results) };
    found.ok_or(ServingStatus::Busy)
}

fn start_connect_to_address(entry: &libc::addrinfo) -> Option<(OwnedFd, bool)> {
    // SAFETY: socket creation with the address family's parameters.
    let fd = unsafe { libc::socket(entry.ai_family, entry.ai_socktype, entry.ai_protocol) };
    if fd < 0 {
        return None;
    }
    if set_nonblocking(fd).is_err() || configure_low_latency_tcp(fd).is_err() {
        // SAFETY: fd valid and owned here.
        unsafe { libc::close(fd) };
        return None;
    }
    // SAFETY: ai_addr/ai_addrlen describe a valid peer address for fd.
    let status = unsafe { libc::connect(fd, entry.ai_addr, entry.ai_addrlen) };
    if status == 0 {
        // SAFETY: fd valid and uniquely owned.
        return Some((unsafe { OwnedFd::from_raw_fd(fd) }, false));
    }
    let error = io::Error::last_os_error().raw_os_error().unwrap_or(0);
    if error == libc::EINPROGRESS || error == libc::EALREADY {
        // SAFETY: fd valid and uniquely owned.
        return Some((unsafe { OwnedFd::from_raw_fd(fd) }, true));
    }
    // SAFETY: fd valid and owned here.
    unsafe { libc::close(fd) };
    None
}

/// `SparkRingServiceBackendCheckWorkOutputConnect`: completes a nonblocking
/// connect. Returns `Ok(true)` when connected, `Ok(false)` when still in
/// flight, `Err(RouteNotFound)` when the connect failed (the caller drops
/// and restarts the socket, as in C).
pub fn check_connect(fd: RawFd) -> Result<bool, ServingStatus> {
    let revents = poll_one(fd, libc::POLLOUT, 0)?;
    if revents & (libc::POLLOUT | libc::POLLERR | libc::POLLHUP) == 0 {
        return Ok(false);
    }
    let mut error: libc::c_int = 0;
    let mut error_bytes = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
    // SAFETY: valid out-pointers for SO_ERROR.
    if unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_ERROR,
            (&mut error as *mut libc::c_int).cast(),
            &mut error_bytes,
        )
    } < 0
    {
        return Err(ServingStatus::RouteNotFound);
    }
    if error == 0 {
        return Ok(true);
    }
    if error == libc::EINPROGRESS || error == libc::EALREADY {
        return Ok(false);
    }
    Err(ServingStatus::RouteNotFound)
}

/// Nonblocking `accept`; `Ok(None)` on EAGAIN/EWOULDBLOCK/EINTR (as in C).
pub fn accept_nonblocking(listener: RawFd) -> Result<Option<OwnedFd>, ServingStatus> {
    // SAFETY: accept on a listening fd with no address out-params.
    let fd = unsafe { libc::accept(listener, std::ptr::null_mut(), std::ptr::null_mut()) };
    if fd < 0 {
        let error = io::Error::last_os_error().raw_os_error().unwrap_or(0);
        if error == libc::EAGAIN || error == libc::EWOULDBLOCK || error == libc::EINTR {
            return Ok(None);
        }
        return Err(ServingStatus::IoError);
    }
    // SAFETY: fd valid and uniquely owned.
    Ok(Some(unsafe { OwnedFd::from_raw_fd(fd) }))
}

/// AF_UNIX/SOCK_STREAM connect to `socket_path` (blocking fd, as the C
/// resident handshake uses bounded `poll` + `read` on a blocking socket).
pub fn unix_connect(socket_path: &str) -> Result<OwnedFd, ServingStatus> {
    // SAFETY: socket creation; no preconditions.
    let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM, 0) };
    if fd < 0 {
        return Err(ServingStatus::IoError);
    }
    let path_bytes = socket_path.as_bytes();
    let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    address.sun_family = libc::AF_UNIX as libc::sa_family_t;
    if path_bytes.len() >= address.sun_path.len() {
        // SAFETY: fd valid and owned here.
        unsafe { libc::close(fd) };
        return Err(ServingStatus::CapacityExceeded);
    }
    address.sun_path[..path_bytes.len()].copy_from_slice(unsafe {
        // SAFETY: reinterpreting &[u8] as &[i8] of the same length.
        &*(path_bytes as *const [u8] as *const [i8])
    });
    // SAFETY: `address` is a fully initialized sockaddr_un.
    if unsafe {
        libc::connect(
            fd,
            (&address as *const libc::sockaddr_un).cast(),
            std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t,
        )
    } != 0
    {
        // SAFETY: fd valid and owned here.
        unsafe { libc::close(fd) };
        return Err(ServingStatus::RouteNotFound);
    }
    // SAFETY: fd valid and uniquely owned.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// One `read(2)` call. `Err` with `WouldBlock` preserves EAGAIN/EWOULDBLOCK;
/// EINTR surfaces as `Interrupted` (call sites decide whether to retry,
/// matching the C loops).
pub fn read_once(fd: RawFd, buffer: &mut [u8]) -> io::Result<usize> {
    // SAFETY: read into a valid mutable slice.
    let got = unsafe { libc::read(fd, buffer.as_mut_ptr().cast(), buffer.len()) };
    if got < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(got as usize)
}

/// One `write(2)` call (same error conventions as [`read_once`]).
pub fn write_once(fd: RawFd, buffer: &[u8]) -> io::Result<usize> {
    // SAFETY: write from a valid slice.
    let written = unsafe { libc::write(fd, buffer.as_ptr().cast(), buffer.len()) };
    if written < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(written as usize)
}

/// `poll(2)` event masks re-exported for the backend's poll loops.
pub use libc::{POLLERR, POLLHUP, POLLIN, POLLNVAL, POLLOUT};

/// One `poll(2)` call on a single fd; returns the revents mask.
pub fn poll_one(fd: RawFd, events: i16, timeout_ms: i32) -> Result<i16, ServingStatus> {
    let mut descriptor = libc::pollfd { fd, events, revents: 0 };
    // SAFETY: `descriptor` is a valid pollfd.
    let result = unsafe { libc::poll(&mut descriptor, 1, timeout_ms) };
    if result < 0 {
        let error = io::Error::last_os_error().raw_os_error().unwrap_or(0);
        if error == libc::EINTR {
            return Ok(0);
        }
        return Err(ServingStatus::IoError);
    }
    if result == 0 {
        return Ok(0);
    }
    Ok(descriptor.revents)
}

/// `SparkRingServiceBackendResidentWriteFull`: loop until every byte is out
/// or a write fails (blocking fd).
pub fn write_full(fd: RawFd, buffer: &[u8]) -> Result<(), ServingStatus> {
    let mut offset = 0usize;
    while offset < buffer.len() {
        match write_once(fd, &buffer[offset..]) {
            Ok(0) => return Err(ServingStatus::IoError),
            Ok(written) => offset += written,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(ServingStatus::IoError),
        }
    }
    Ok(())
}

/// `SparkRingServiceBackendResidentReadBounded`: read exactly
/// `buffer.len()` bytes, waiting up to `timeout_ms` per poll round.
pub fn read_bounded(fd: RawFd, buffer: &mut [u8], timeout_ms: i32) -> Result<(), ServingStatus> {
    let mut offset = 0usize;
    while offset < buffer.len() {
        if poll_one(fd, libc::POLLIN, timeout_ms)? <= 0 {
            return Err(ServingStatus::Busy);
        }
        match read_once(fd, &mut buffer[offset..]) {
            Ok(0) => return Err(ServingStatus::RouteNotFound),
            Ok(got) => offset += got,
            Err(error)
                if matches!(
                    error.raw_os_error(),
                    Some(code) if code == libc::EINTR || code == libc::EAGAIN || code == libc::EWOULDBLOCK
                ) => {}
            Err(_) => return Err(ServingStatus::RouteNotFound),
        }
    }
    Ok(())
}

/// Test helper: a connected pair of unix stream sockets.
#[cfg(test)]
pub fn socketpair() -> io::Result<(OwnedFd, OwnedFd)> {
    let mut pair = [0 as RawFd; 2];
    // SAFETY: valid out-array.
    if unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, pair.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: both fds valid and uniquely owned.
    Ok((unsafe { OwnedFd::from_raw_fd(pair[0]) }, unsafe { OwnedFd::from_raw_fd(pair[1]) }))
}

/// RawFd accessor kept in one place so call sites stay on `OwnedFd`.
pub fn raw_fd(fd: &OwnedFd) -> RawFd {
    fd.as_raw_fd()
}

/// Bytes of an `OsStr` path (unix sockets take raw bytes).
pub fn path_bytes(path: &std::ffi::OsStr) -> &[u8] {
    path.as_bytes()
}
