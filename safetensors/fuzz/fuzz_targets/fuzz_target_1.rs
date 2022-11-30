#![no_main]

use libfuzzer_sys::fuzz_target;
use safetensors::tensor::SafeTensors;

fuzz_target!(|data: &[u8]| {
    let _ = SafeTensors::deserialize(data);
});
