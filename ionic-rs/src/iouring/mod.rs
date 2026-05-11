//! io_uring read engine with registered files, registered buffers, and a
//! slot pool.
//!
//! ## Lifetime invariants
//!
//! - Registered fds must outlive the engine (we own `Vec<File>`).
//! - Registered buffers must not move and must outlive any in-flight read.
//!   `Drop` calls `unregister_buffers` before the boxes are freed.
//! - The constructor finishes file + buffer registration before returning;
//!   submitting before that is a kernel-side error.

mod coalesce;
pub use coalesce::{coalesce_chunks, ChunkRead, LogicalSegment, ScatterEntry};

/// `bytes_read` may be less than requested on short reads (e.g. EOF).
#[derive(Debug, Clone, Copy)]
pub struct Completion {
    pub slot: u32,
    pub bytes_read: usize,
}

use std::collections::VecDeque;
use std::fs::File;
use std::os::fd::{AsRawFd, RawFd};
use std::path::Path;

use io_uring::{opcode, types, IoUring};

use crate::error::{Error, Result};

pub const DEFAULT_QUEUE_DEPTH: u32 = 64;
/// 512 KiB: empirically the best amortization of per-chunk overhead vs.
/// throughput between 128 KiB and 4 MiB.
pub const DEFAULT_CHUNK_BYTES: usize = 512 * 1024;

/// Backing storage for io_uring's registered staging buffers. The CUDA
/// pipeline uses [`crate::cuda::PinnedBuf`] for direct-DMA H2D; tests use
/// `Box<[u8]>`, which bounces through the driver's pinned buffer.
pub trait StagingBuffer: Send {
    fn alloc(bytes: usize) -> Result<Self>
    where
        Self: Sized;
    fn as_slice(&self) -> &[u8];
    fn as_mut_slice(&mut self) -> &mut [u8];
}

impl StagingBuffer for Box<[u8]> {
    fn alloc(bytes: usize) -> Result<Self> {
        Ok(vec![0u8; bytes].into_boxed_slice())
    }
    fn as_slice(&self) -> &[u8] {
        self
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        self
    }
}

impl StagingBuffer for crate::cuda::PinnedBuf {
    fn alloc(bytes: usize) -> Result<Self> {
        Self::alloc(bytes)
    }
    fn as_slice(&self) -> &[u8] {
        self.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

pub struct ReadEngine<B: StagingBuffer = Box<[u8]>> {
    ring: IoUring,
    /// Owned files. Index in this Vec equals the index into the kernel's
    /// registered-files table (`Fixed(idx)` in opcodes).
    files: Vec<File>,
    /// One staging buffer per slot, length == `qd`. Index equals
    /// `buf_index` in the registered-buffers table.
    bufs: Vec<B>,
    /// Free slot indices. `pop_front()` to claim, `push_back()` to release.
    free: VecDeque<u32>,
    chunk_bytes: usize,
    qd: u32,
}

impl<B: StagingBuffer> ReadEngine<B> {
    /// Pre-allocates `qd` chunk-sized staging buffers and registers them.
    /// Files are added later via `register_file`.
    pub fn new(qd: u32, chunk_bytes: usize) -> Result<Self> {
        if qd == 0 || chunk_bytes == 0 {
            return Err(Error::Other(format!(
                "invalid ReadEngine config: qd={qd} chunk_bytes={chunk_bytes}"
            )));
        }
        let ring = IoUring::new(qd).map_err(|e| io_err("IoUring::new", &e))?;

        let mut bufs: Vec<B> = Vec::with_capacity(qd as usize);
        for _ in 0..qd {
            bufs.push(B::alloc(chunk_bytes)?);
        }

        let iovecs: Vec<libc::iovec> = bufs
            .iter()
            .map(|b| {
                let s = b.as_slice();
                libc::iovec {
                    iov_base: s.as_ptr() as *mut _,
                    iov_len: s.len(),
                }
            })
            .collect();
        // SAFETY: each iovec points into a buffer we own; `Drop` unregisters
        // before the buffers are freed (field drop order: ring before bufs).
        unsafe {
            ring.submitter()
                .register_buffers(&iovecs)
                .map_err(|e| io_err("register_buffers", &e))?;
        }

        let free = (0..qd).collect();
        Ok(Self {
            ring,
            files: Vec::new(),
            bufs,
            free,
            chunk_bytes,
            qd,
        })
    }

    pub fn chunk_bytes(&self) -> usize {
        self.chunk_bytes
    }

    pub fn queue_depth(&self) -> u32 {
        self.qd
    }

    /// Register a file and return its `fd_idx`. Re-registers the entire fd
    /// table; cost is negligible since n is tiny.
    pub fn register_file(&mut self, path: &Path) -> Result<u32> {
        let file =
            File::open(path).map_err(|e| Error::Other(format!("open {}: {e}", path.display())))?;
        let mut fds: Vec<RawFd> = self.files.iter().map(|f| f.as_raw_fd()).collect();
        fds.push(file.as_raw_fd());
        // First registration has nothing to unregister (ENXIO is fine).
        let _ = self.ring.submitter().unregister_files();
        self.ring
            .submitter()
            .register_files(&fds)
            .map_err(|e| io_err("register_files", &e))?;
        let idx = self.files.len() as u32;
        self.files.push(file);
        Ok(idx)
    }

    pub fn acquire_slot(&mut self) -> Option<u32> {
        self.free.pop_front()
    }

    /// Queue an `IORING_OP_READ_FIXED` SQE; doesn't submit. `user_data` is
    /// the slot so completions carry it back.
    pub fn submit_read(&mut self, slot: u32, chunk: &ChunkRead) -> Result<()> {
        let buf_ptr = self.bufs[slot as usize].as_mut_slice().as_mut_ptr();
        let entry = opcode::ReadFixed::new(
            types::Fixed(chunk.fd_idx),
            buf_ptr,
            chunk.len as u32,
            slot as u16,
        )
        .offset(chunk.offset)
        .build()
        .user_data(slot as u64);
        // SAFETY: pointers refer to memory we own (`bufs[slot]`) and a
        // registered fd (`files`); both outlive the in-flight op (caller
        // tracks completion before recycling slot or dropping engine).
        unsafe {
            self.ring
                .submission()
                .push(&entry)
                .map_err(|_| Error::Other("submission queue push failed (queue full?)".into()))?;
        }
        Ok(())
    }

    pub fn flush_submissions(&mut self) -> Result<usize> {
        self.ring.submit().map_err(|e| io_err("submit", &e))
    }

    /// Block until at least one CQE is available.
    pub fn submit_and_wait_one(&mut self) -> Result<()> {
        self.ring
            .submit_and_wait(1)
            .map_err(|e| io_err("submit_and_wait", &e))?;
        Ok(())
    }

    /// Drain ready CQEs without blocking.
    pub fn peek_completions(&mut self) -> Result<Vec<Completion>> {
        let cqes: Vec<io_uring::cqueue::Entry> = self.ring.completion().collect();
        let mut out = Vec::with_capacity(cqes.len());
        for cqe in cqes {
            let slot = cqe.user_data() as u32;
            let result = cqe.result();
            if result < 0 {
                return Err(Error::IoUring {
                    op: "read_fixed",
                    errno: -result,
                });
            }
            out.push(Completion {
                slot,
                bytes_read: result as usize,
            });
        }
        Ok(out)
    }

    /// `bytes` is the count from the matching `Completion`.
    pub fn slot_buffer(&self, slot: u32, bytes: usize) -> &[u8] {
        &self.bufs[slot as usize].as_slice()[..bytes]
    }

    /// Caller must ensure no consumer is still reading from this slot's
    /// buffer (e.g. the H2D copy has completed).
    pub fn release_slot(&mut self, slot: u32) {
        self.free.push_back(slot);
    }

    /// Synchronous submit + drain loop. Used by tests; the CUDA pipeline
    /// drives the lower-level primitives directly.
    pub fn fetch<F>(&mut self, chunks: &[ChunkRead], mut on_chunk: F) -> Result<()>
    where
        F: FnMut(&[u8], &[ScatterEntry]) -> Result<()>,
    {
        if chunks.is_empty() {
            return Ok(());
        }

        let total = chunks.len();
        let mut next_chunk = 0usize;
        let mut completed = 0usize;
        let mut slot_to_chunk: Vec<Option<usize>> = vec![None; self.qd as usize];

        while completed < total {
            let mut pushed_any = false;
            while next_chunk < total {
                let slot = match self.free.pop_front() {
                    Some(s) => s,
                    None => break,
                };
                let chunk = &chunks[next_chunk];
                let buf_ptr = self.bufs[slot as usize].as_mut_slice().as_mut_ptr();

                let entry = opcode::ReadFixed::new(
                    types::Fixed(chunk.fd_idx),
                    buf_ptr,
                    chunk.len as u32,
                    slot as u16,
                )
                .offset(chunk.offset)
                .build()
                .user_data(slot as u64);

                // SAFETY: pointers refer to memory we own and registered fds;
                // both outlive the in-flight op (synchronous wait below).
                unsafe {
                    self.ring.submission().push(&entry).map_err(|_| {
                        Error::Other("submission queue push failed (queue full?)".into())
                    })?;
                }
                slot_to_chunk[slot as usize] = Some(next_chunk);
                next_chunk += 1;
                pushed_any = true;
            }

            let outstanding = next_chunk - completed;
            if pushed_any || outstanding > 0 {
                let want = if outstanding > 0 { 1 } else { 0 };
                self.ring
                    .submit_and_wait(want)
                    .map_err(|e| io_err("submit_and_wait", &e))?;
            }

            // Collect into a Vec so the completion() borrow doesn't conflict
            // with the bufs borrow below.
            let cqes: Vec<io_uring::cqueue::Entry> = self.ring.completion().collect();
            for cqe in cqes {
                let slot = cqe.user_data() as u32;
                let chunk_idx = slot_to_chunk[slot as usize].take().ok_or_else(|| {
                    Error::Other(format!(
                        "spurious CQE for slot {slot} with no in-flight chunk"
                    ))
                })?;
                let result = cqe.result();
                if result < 0 {
                    return Err(Error::IoUring {
                        op: "read_fixed",
                        errno: -result,
                    });
                }
                let bytes_read = result as usize;
                let chunk = &chunks[chunk_idx];
                let buf = &self.bufs[slot as usize].as_slice()[..bytes_read];

                // Short-read fixup: trim/clamp scatter entries so callers see
                // only what's inside `buf`. Fast path when the read was full.
                if chunk
                    .scatter
                    .iter()
                    .all(|s| s.staging_offset + s.len <= bytes_read)
                {
                    on_chunk(buf, &chunk.scatter)?;
                } else {
                    let trimmed: Vec<ScatterEntry> = chunk
                        .scatter
                        .iter()
                        .filter_map(|s| {
                            if s.staging_offset >= bytes_read {
                                None
                            } else {
                                let avail = bytes_read - s.staging_offset;
                                Some(ScatterEntry {
                                    staging_offset: s.staging_offset,
                                    len: s.len.min(avail),
                                    dst_offset: s.dst_offset,
                                    user_data: s.user_data,
                                })
                            }
                        })
                        .collect();
                    on_chunk(buf, &trimmed)?;
                }
                self.free.push_back(slot);
                completed += 1;
            }
        }
        Ok(())
    }
}

impl<B: StagingBuffer> Drop for ReadEngine<B> {
    fn drop(&mut self) {
        // Release the kernel's refs before the field-drop sequence frees
        // buffers and files.
        let _ = self.ring.submitter().unregister_buffers();
        let _ = self.ring.submitter().unregister_files();
    }
}

fn io_err(op: &'static str, e: &std::io::Error) -> Error {
    Error::IoUring {
        op,
        errno: e.raw_os_error().unwrap_or(0),
    }
}
