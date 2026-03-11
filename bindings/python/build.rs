use std::env;

fn main() {
    // Check for hmll feature request via environment variable
    // This allows `pip install safetensors[hmll]` to work by setting
    // SAFETENSORS_ENABLE_HMLL=1 in the environment
    if env::var("SAFETENSORS_ENABLE_HMLL").is_ok() {
        println!("cargo:rustc-cfg=feature=\"hmll\"");
    }
}
