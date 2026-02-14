use std::env;

fn main() {
    // Set rpath for cuFile so libcufile.so.0 can be found at runtime
    #[cfg(feature = "cufile")]
    {
        if let Some(dir) = find_cufile_lib_dir() {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{}", dir);
        }
    }
}

#[cfg(feature = "cufile")]
fn find_cufile_lib_dir() -> Option<String> {
    if let Ok(cufile_root) = env::var("CUFILE_ROOT") {
        return Some(format!("{}/lib", cufile_root));
    }

    let python_cmd =
        "import nvidia.cufile; import os; print(os.path.dirname(nvidia.cufile.__file__))";

    // Check PYO3_PYTHON first (set by maturin), then system python
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(p) = env::var("PYO3_PYTHON") {
        candidates.push(p);
    }
    if let Ok(p) = env::var("PYTHON_SYS_EXECUTABLE") {
        candidates.push(p);
    }
    candidates.push("python3".to_string());

    for python in &candidates {
        if let Ok(output) = std::process::Command::new(python)
            .args(["-c", python_cmd])
            .output()
        {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    return Some(format!("{}/lib", path));
                }
            }
        }
    }
    None
}
