//! CUDA pipeline orchestration. Two threads:
//!
//! - **Scheduler** (caller's thread): drives the io_uring engine. Submits
//!   reads, drains CQEs, forwards completed slots to the worker, recycles
//!   slots when the worker signals H2D done.
//! - **DMA worker** (`std::thread::scope`-spawned per [`run`] call): keeps
//!   up to two H2D copies in flight (one per CUDA stream, matching NVIDIA's
//!   two-async-copy-engine hardware), polls `cuEventQuery` non-blocking,
//!   signals back to the scheduler.
//!
//! `std::thread::scope` joins the worker before [`run`] returns. A slot's
//! staging buffer can't be reused until the worker has signalled done, so
//! the worker's read can't race with a kernel write into the same buffer.

use std::path::Path;
use std::sync::mpsc;
use std::sync::Mutex;

use crate::cuda::{self, CuContext, CuDevice, CuEvent, CuStream, DeviceBuf, PinnedBuf};
use crate::error::{Error, Result};
use crate::iouring::{coalesce_chunks, ChunkRead, LogicalSegment, ReadEngine};

/// Unit of work passed from scheduler to DMA worker. `buf_ptr` is valid
/// until the worker signals the slot's done.
#[derive(Clone, Copy)]
struct DmaJob {
    slot: u32,
    chunk_idx: usize,
    buf_ptr: *const u8,
    bytes_read: usize,
}

// SAFETY: `buf_ptr` is a snapshot into a `PinnedBuf` owned by the engine;
// the slot isn't recycled until the worker signals done.
unsafe impl Send for DmaJob {}

pub struct CudaPipeline {
    /// Declaration order is drop order: engine/streams/events hold pointers
    /// into the primary context and must drop before `ctx`.
    engine: ReadEngine<PinnedBuf>,
    streams: [CuStream; 2],
    events: [CuEvent; 2],
    ctx: CuContext,
}

impl CudaPipeline {
    pub fn new(device_ordinal: i32, qd: u32, chunk_bytes: usize) -> Result<Self> {
        let dev = CuDevice::get(device_ordinal)?;
        let ctx = CuContext::primary_retain(dev)?;
        // PinnedBuf::alloc, cuStreamCreate, cuEventCreate all need a current
        // context.
        let (engine, streams, events) = ctx.with_current(|| {
            let engine = ReadEngine::<PinnedBuf>::new(qd, chunk_bytes)?;
            let streams = [CuStream::new()?, CuStream::new()?];
            let events = [CuEvent::new()?, CuEvent::new()?];
            Ok((engine, streams, events))
        })?;
        Ok(Self {
            engine,
            streams,
            events,
            ctx,
        })
    }

    pub fn register_file(&mut self, path: &Path) -> Result<u32> {
        self.engine.register_file(path)
    }

    /// Allocate on the pipeline's primary context.
    pub fn alloc_device_buf(&self, bytes: usize) -> Result<DeviceBuf> {
        self.ctx.with_current(|| DeviceBuf::alloc(bytes))
    }

    /// Run the given logical reads. Each segment's `dst_offset` is
    /// reinterpreted as a `CUdeviceptr`. Returns when every requested byte
    /// is on the device; worker errors surface here.
    pub fn run(&mut self, segments: &[LogicalSegment]) -> Result<()> {
        let chunks = coalesce_chunks(segments, self.engine.chunk_bytes());
        let total = chunks.len();
        if total == 0 {
            return Ok(());
        }

        let qd = self.engine.queue_depth() as usize;
        let mut slot_to_chunk: Vec<usize> = vec![0; qd];

        // Scheduler ↔ worker channels. mpsc::channel is unbounded but
        // bounded de facto by QD: at most QD slots can be in flight, and
        // each occupies at most one DmaJob in transit.
        let (job_tx, job_rx) = mpsc::channel::<DmaJob>();
        let (done_tx, done_rx) = mpsc::channel::<u32>();

        // Worker error surfaces here on join.
        let worker_err: Mutex<Option<Error>> = Mutex::new(None);

        // Destructure for disjoint field borrows: scheduler gets `&mut engine`,
        // worker gets `&` to the rest.
        let Self {
            engine,
            streams,
            events,
            ctx,
        } = self;
        let chunks_ref = &chunks;
        let worker_err_ref = &worker_err;

        std::thread::scope(|scope| -> Result<()> {
            scope.spawn(move || {
                if let Err(e) = run_dma_worker(job_rx, done_tx, chunks_ref, streams, events, ctx) {
                    *worker_err_ref.lock().unwrap() = Some(e);
                }
            });

            let reader_result = run_scheduler_loop(
                engine,
                &chunks,
                &mut slot_to_chunk,
                &job_tx,
                &done_rx,
                total,
            );

            // Closing the sender lets the worker observe Disconnected and exit.
            drop(job_tx);
            reader_result
        })?;

        // Worker error trumps a clean reader return: if the worker failed
        // mid-stream the GPU buffer is incomplete.
        if let Some(e) = worker_err.into_inner().unwrap() {
            return Err(e);
        }

        // Per-stream sync to establish a happens-before edge for consumers
        // on other streams (e.g. torch's default stream reading the DLPack
        // tensor); cuEvent ordering only applies along the same stream.
        self.ctx.with_current(|| {
            self.streams[0].synchronize()?;
            self.streams[1].synchronize()
        })?;
        Ok(())
    }
}

fn run_scheduler_loop(
    engine: &mut ReadEngine<PinnedBuf>,
    chunks: &[ChunkRead],
    slot_to_chunk: &mut [usize],
    job_tx: &mpsc::Sender<DmaJob>,
    done_rx: &mpsc::Receiver<u32>,
    total: usize,
) -> Result<()> {
    let mut next_chunk: usize = 0; // index of next chunk to submit
    let mut io_outstanding: usize = 0; // submitted, awaiting CQE
    let mut dma_outstanding: usize = 0; // forwarded to worker, awaiting done signal
    let mut completed: usize = 0; // slots fully recycled

    while completed < total {
        // 1. Drain worker done-signals, recycle slots.
        while let Ok(slot) = done_rx.try_recv() {
            engine.release_slot(slot);
            dma_outstanding -= 1;
            completed += 1;
        }
        if completed == total {
            break;
        }

        // 2. Submit reads while slots and chunks are available.
        while next_chunk < total {
            let slot = match engine.acquire_slot() {
                Some(s) => s,
                None => break,
            };
            if let Err(e) = engine.submit_read(slot, &chunks[next_chunk]) {
                engine.release_slot(slot);
                return Err(e);
            }
            slot_to_chunk[slot as usize] = next_chunk;
            next_chunk += 1;
            io_outstanding += 1;
        }
        engine.flush_submissions()?;

        // 3. Block for progress on whichever side is outstanding.
        if io_outstanding > 0 {
            engine.submit_and_wait_one()?;
        } else if dma_outstanding > 0 {
            let slot = done_rx.recv().map_err(|_| {
                Error::Other("DMA worker disconnected with chunks still pending".into())
            })?;
            engine.release_slot(slot);
            dma_outstanding -= 1;
            completed += 1;
            continue;
        } else {
            return Err(Error::Other(format!(
                "pipeline state inconsistent: completed={completed} total={total} io=0 dma=0"
            )));
        }

        // 4. Forward newly-ready CQEs to the worker.
        for c in engine.peek_completions()? {
            let chunk_idx = slot_to_chunk[c.slot as usize];
            let buf_ptr = engine.slot_buffer(c.slot, c.bytes_read).as_ptr();
            let job = DmaJob {
                slot: c.slot,
                chunk_idx,
                buf_ptr,
                bytes_read: c.bytes_read,
            };
            io_outstanding -= 1;
            if job_tx.send(job).is_err() {
                return Err(Error::Other("DMA worker disconnected mid-stream".into()));
            }
            dma_outstanding += 1;
        }
    }
    Ok(())
}

/// `cuEventQuery` keeps slot recycling non-blocking, so io_uring submission
/// and CUDA dispatch progress independently.
fn run_dma_worker(
    job_rx: mpsc::Receiver<DmaJob>,
    done_tx: mpsc::Sender<u32>,
    chunks: &[ChunkRead],
    streams: &[CuStream; 2],
    events: &[CuEvent; 2],
    ctx: &CuContext,
) -> Result<()> {
    ctx.with_current(|| -> Result<()> {
        let mut in_flight: [Option<DmaJob>; 2] = [None, None];

        loop {
            // 1. Poll events for completed H2Ds, signal slot back.
            for stream_idx in 0..2 {
                if let Some(job) = in_flight[stream_idx] {
                    if events[stream_idx].query()? {
                        if done_tx.send(job.slot).is_err() {
                            return Ok(());
                        }
                        in_flight[stream_idx] = None;
                    }
                }
            }

            // 2. Fill any empty stream with a new job.
            let mut got_any = false;
            for stream_idx in 0..2 {
                if in_flight[stream_idx].is_some() {
                    continue;
                }
                let job = match job_rx.try_recv() {
                    Ok(j) => Some(j),
                    Err(mpsc::TryRecvError::Empty) => {
                        if in_flight[0].is_none() && in_flight[1].is_none() && stream_idx == 0 {
                            // Both streams idle, no queued work: block on recv.
                            match job_rx.recv() {
                                Ok(j) => Some(j),
                                Err(_) => return Ok(()),
                            }
                        } else {
                            None
                        }
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        if in_flight[0].is_none() && in_flight[1].is_none() {
                            return Ok(());
                        }
                        None
                    }
                };
                if let Some(job) = job {
                    dispatch_h2d(&job, chunks, &streams[stream_idx])?;
                    events[stream_idx].record(&streams[stream_idx])?;
                    in_flight[stream_idx] = Some(job);
                    got_any = true;
                }
            }

            if !got_any {
                std::hint::spin_loop();
            }
        }
    })
}

/// One `cuMemcpyHtoDAsync` per scatter entry, skipping anything past the
/// kernel's short-read boundary.
fn dispatch_h2d(job: &DmaJob, chunks: &[ChunkRead], stream: &CuStream) -> Result<()> {
    let chunk = &chunks[job.chunk_idx];
    for s in &chunk.scatter {
        if s.staging_offset >= job.bytes_read {
            continue;
        }
        let avail = job.bytes_read - s.staging_offset;
        let len = s.len.min(avail);
        // SAFETY: buf_ptr is valid for `bytes_read` bytes until this slot is
        // signalled done.
        let src = unsafe { job.buf_ptr.add(s.staging_offset) };
        cuda::memcpy_h2d_async(s.dst_offset as u64, src, len, stream)?;
    }
    Ok(())
}
