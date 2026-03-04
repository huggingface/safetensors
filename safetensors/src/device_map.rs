//! `device_map` parsing logic

// TODO: handle "" empty key for default matching & defaulting for cpu when unspecified & prefix
// matching

use core::{fmt::Display, num::ParseIntError, str::FromStr};

use hashbrown::HashMap;

#[derive(Debug)]
/// Errors that can occur during device parsing.
pub enum DeviceParsingError {
    /// Error when parsing a CUDA index from a string.
    ParseInt(ParseIntError),
    /// Provided device is unsupported
    UnsupportedDevice(String),
}

impl std::fmt::Display for DeviceParsingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceParsingError::ParseInt(e) => write!(f, "Invalid CUDA index: {}", e),
            DeviceParsingError::UnsupportedDevice(s) => write!(f, "Unsupported device: {}", s),
        }
    }
}

impl From<ParseIntError> for DeviceParsingError {
    fn from(e: ParseIntError) -> Self {
        DeviceParsingError::ParseInt(e)
    }
}

#[derive(Debug)]
/// Different target devices supported for loading tensors.
pub enum Device {
    /// CPU
    Cpu,
    /// Explicit CUDA target
    Cuda(usize),
    /// Anonymous target, will be resolved to a specific device by the loading function, usually
    /// CUDA
    Index(usize),
}

impl FromStr for Device {
    type Err = DeviceParsingError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cpu" => Ok(Device::Cpu),
            "cuda" => Ok(Device::Cuda(0)),
            s if s.starts_with("cuda:") => {
                let index_str = &s[5..];
                let index = index_str.parse::<usize>()?;
                Ok(Device::Cuda(index))
            }
            s => s
                .parse::<usize>()
                .map(Device::Index)
                .map_err(|_| DeviceParsingError::UnsupportedDevice(s.to_string())),
        }
    }
}

impl Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(idx) => {
                write!(f, "cuda:{}", idx)
            }
            Device::Index(idx) => write!(f, "{}", idx),
        }
    }
}

/// Represents a mapping from tensor names to a target device for loading tensors.
pub type DeviceMap = HashMap<String, Device>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cpu() {
        let device = Device::from_str("cpu").unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn parse_cuda_default() {
        let device = Device::from_str("cuda").unwrap();
        assert!(matches!(device, Device::Cuda(0)));
    }

    #[test]
    fn parse_cuda_with_index() {
        let device = Device::from_str("cuda:0").unwrap();
        assert!(matches!(device, Device::Cuda(0)));

        let device = Device::from_str("cuda:1").unwrap();
        assert!(matches!(device, Device::Cuda(1)));

        let device = Device::from_str("cuda:7").unwrap();
        assert!(matches!(device, Device::Cuda(7)));
    }

    #[test]
    fn parse_bare_index() {
        let device = Device::from_str("0").unwrap();
        assert!(matches!(device, Device::Index(0)));

        let device = Device::from_str("3").unwrap();
        assert!(matches!(device, Device::Index(3)));
    }

    #[test]
    fn parse_cuda_invalid_index() {
        let result = Device::from_str("cuda:abc");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, DeviceParsingError::ParseInt(_)));
        assert!(err.to_string().contains("Invalid CUDA index"));
    }

    #[test]
    fn parse_unsupported_device() {
        let result = Device::from_str("tpu");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, DeviceParsingError::UnsupportedDevice(_)));
        assert!(err.to_string().contains("Unsupported device: tpu"));
    }

    #[test]
    fn parse_int_error_conversion() {
        let parse_err = "not_a_number".parse::<usize>().unwrap_err();
        let device_err: DeviceParsingError = parse_err.into();
        assert!(matches!(device_err, DeviceParsingError::ParseInt(_)));
    }

    #[test]
    fn device_map_insert_and_lookup() {
        let mut map: DeviceMap = DeviceMap::new();
        map.insert(
            "layer.0.weight".to_string(),
            Device::from_str("cuda:0").unwrap(),
        );
        map.insert(
            "layer.1.weight".to_string(),
            Device::from_str("cuda:1").unwrap(),
        );
        map.insert("embed.weight".to_string(), Device::from_str("cpu").unwrap());

        assert_eq!(map.len(), 3);
        assert!(matches!(map.get("layer.0.weight"), Some(Device::Cuda(0))));
        assert!(matches!(map.get("layer.1.weight"), Some(Device::Cuda(1))));
        assert!(matches!(map.get("embed.weight"), Some(Device::Cpu)));
        assert!(map.get("nonexistent").is_none());
    }
}
