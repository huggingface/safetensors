//! cuFile driver lifecycle management
//!
//! The driver must be initialized once and closed on shutdown.
//! This module provides a thread-safe singleton pattern.

use super::bindings::{check_cufile_error, cuFileDriverClose, cuFileDriverOpen};
use super::error::GdsError;
use std::sync::{Arc, Mutex, Once};

static DRIVER_INIT: Once = Once::new();
static mut DRIVER_INSTANCE: Option<Arc<Mutex<GdsDriverInner>>> = None;

/// Inner driver state
struct GdsDriverInner {
    initialized: bool,
}

/// GPU Direct Storage driver singleton
///
/// Ensures cuFileDriverOpen is called exactly once and cuFileDriverClose
/// is called on drop.
#[derive(Clone)]
pub struct GdsDriver {
    inner: Arc<Mutex<GdsDriverInner>>,
}

impl GdsDriver {
    /// Get or initialize the GDS driver
    ///
    /// This function is thread-safe and will only initialize the driver once.
    pub fn get() -> Result<Self, GdsError> {
        unsafe {
            DRIVER_INIT.call_once(|| {
                match Self::initialize() {
                    Ok(driver) => {
                        DRIVER_INSTANCE = Some(driver.inner.clone());
                    }
                    Err(e) => {
                        eprintln!("Failed to initialize GDS driver: {}", e);
                    }
                }
            });

            DRIVER_INSTANCE
                .clone()
                .ok_or(GdsError::DriverInitFailed)
                .map(|inner| GdsDriver { inner })
        }
    }

    /// Initialize the cuFile driver
    fn initialize() -> Result<Self, GdsError> {
        unsafe {
            let result = cuFileDriverOpen();
            check_cufile_error(result, "cuFileDriverOpen")?;

            Ok(GdsDriver {
                inner: Arc::new(Mutex::new(GdsDriverInner { initialized: true })),
            })
        }
    }

    /// Check if driver is initialized
    pub fn is_initialized(&self) -> bool {
        self.inner.lock().unwrap().initialized
    }
}

impl Drop for GdsDriverInner {
    fn drop(&mut self) {
        if self.initialized {
            unsafe {
                let result = cuFileDriverClose();
                if let Err(e) = check_cufile_error(result, "cuFileDriverClose") {
                    eprintln!("Warning: Failed to close cuFile driver: {}", e);
                }
            }
            self.initialized = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Only run if GDS is available
    fn test_driver_singleton() {
        let driver1 = GdsDriver::get();
        let driver2 = GdsDriver::get();

        assert!(driver1.is_ok());
        assert!(driver2.is_ok());

        // Both should reference the same driver
        let d1 = driver1.unwrap();
        let d2 = driver2.unwrap();
        
        assert!(d1.is_initialized());
        assert!(d2.is_initialized());
    }
}
