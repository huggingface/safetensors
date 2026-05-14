//! Minimal Metal allocator for host-shared MTLBuffers (Apple silicon UMA).
//!
//! Shared-mode buffers are CPU- and GPU-visible: writing through
//! [`MtlBuffer::contents_ptr`] is observable from the GPU after a normal
//! command-buffer commit (no `didModifyRange:` needed on Apple silicon).

use std::ffi::c_void;
use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};

static DEVICE: OnceLock<DeviceHandle> = OnceLock::new();

struct DeviceHandle(Retained<ProtocolObject<dyn MTLDevice>>);

// MTLDevice is documented thread-safe.
unsafe impl Send for DeviceHandle {}
unsafe impl Sync for DeviceHandle {}

fn device() -> Result<&'static ProtocolObject<dyn MTLDevice>, &'static str> {
    if let Some(h) = DEVICE.get() {
        return Ok(&h.0);
    }
    let dev = MTLCreateSystemDefaultDevice().ok_or("MTLCreateSystemDefaultDevice returned nil")?;
    let _ = DEVICE.set(DeviceHandle(dev));
    Ok(&DEVICE.get().unwrap().0)
}

pub struct MtlBuffer {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    nbytes: usize,
    contents: *mut c_void,
}

// MTLBuffer is documented thread-safe; disjoint writes through the host
// alias are sound from multiple threads.
unsafe impl Send for MtlBuffer {}
unsafe impl Sync for MtlBuffer {}

impl MtlBuffer {
    pub fn alloc_shared(nbytes: usize) -> Result<Self, String> {
        let dev = device().map_err(str::to_owned)?;
        let len = nbytes.max(1);
        let buf = dev
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| format!("MTLDevice::newBufferWithLength failed for {nbytes} bytes"))?;
        let contents = buf.contents().as_ptr();
        Ok(Self {
            buf,
            nbytes,
            contents,
        })
    }

    pub fn contents_ptr(&self) -> *mut c_void {
        self.contents
    }

    /// The Objective-C buffer pointer (`id<MTLBuffer>`). PyTorch's MPS
    /// `from_dlpack` expects this - not the `contents` pointer - in
    /// `DLTensor.data`, since the MPS allocator tracks buffers by ID.
    pub fn as_metal_id_ptr(&self) -> *mut c_void {
        &*self.buf as *const ProtocolObject<dyn MTLBuffer> as *mut c_void
    }

    #[allow(dead_code)]
    pub fn nbytes(&self) -> usize {
        self.nbytes
    }
}
