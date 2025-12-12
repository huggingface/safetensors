fn main() {
    // Try linking necessary CUDA libraries on Linux for GDS
    #[cfg(all(target_os = "linux", feature = "cuda-gds"))]
    {
        // Add CUDA library paths for linking
        println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=/usr/local/cuda-12/lib64");
        println!("cargo:rustc-link-search=/usr/local/cuda-13.0/lib64");
        println!("cargo:rustc-link-lib=cufile");
    }
}
