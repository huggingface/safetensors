use std::env;

fn main() {
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        link_cuda();
    }
    if env::var("CARGO_FEATURE_CUFILE").is_ok() {
        link_cufile();
    }
}

fn link_cuda() {
    // Try CUDA_PATH/CUDA_HOME, then common defaults
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-search=native={}/lib", cuda_path);
    } else if let Ok(cuda_home) = env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
        println!("cargo:rustc-link-search=native={}/lib", cuda_home);
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib");
        println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
        println!("cargo:rustc-link-search=native=/opt/cuda/lib");
    }

    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
}

fn link_cufile() {
    if let Some(dir) = find_cufile_lib_dir() {
        println!("cargo:rustc-link-search=native={}", dir);

        // pip packages only ship libcufile.so.0 (no unversioned symlink).
        // Create a temporary symlink in OUT_DIR so the linker can find -lcufile.
        let out_dir = env::var("OUT_DIR").unwrap();
        let versioned = std::path::PathBuf::from(&dir).join("libcufile.so.0");
        let symlink_path = std::path::PathBuf::from(&out_dir).join("libcufile.so");
        if versioned.exists() && !symlink_path.exists() {
            let _ = std::os::unix::fs::symlink(&versioned, &symlink_path);
            println!("cargo:rustc-link-search=native={}", out_dir);
        }

        // Set rpath for runtime discovery
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir);
    }
    println!("cargo:rustc-link-lib=dylib=cufile");
}

fn find_cufile_lib_dir() -> Option<String> {
    // Try CUFILE_ROOT environment variable
    if let Ok(cufile_root) = env::var("CUFILE_ROOT") {
        return Some(format!("{}/lib", cufile_root));
    }

    // Try to find via pip nvidia.cufile package
    let python_cmd =
        "import nvidia.cufile; import os; print(os.path.dirname(nvidia.cufile.__file__))";

    // Try PYO3_PYTHON first (set by maturin), then common python interpreters
    let pythons: Vec<String> = env::var("PYO3_PYTHON")
        .into_iter()
        .chain(env::var("PYTHON_SYS_EXECUTABLE"))
        .chain(["python3".to_string(), "python".to_string()])
        .collect();

    for python in &pythons {
        if let Ok(output) = std::process::Command::new(python)
            .args(["-c", python_cmd])
            .output()
        {
            if output.status.success() {
                let pkg_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let lib_dir = format!("{}/lib", pkg_dir);
                if std::path::Path::new(&lib_dir).exists() {
                    return Some(lib_dir);
                }
            }
        }
    }

    // Try standard CUDA toolkit location
    let cuda_lib = "/usr/local/cuda/lib64";
    if std::path::Path::new(cuda_lib).join("libcufile.so").exists()
        || std::path::Path::new(cuda_lib)
            .join("libcufile.so.0")
            .exists()
    {
        return Some(cuda_lib.to_string());
    }

    None
}
