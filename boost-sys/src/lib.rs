use std::path::{Path, PathBuf};

pub fn headers() -> PathBuf {
    let out_dir = env!("OUT_DIR");
    let out_dir = Path::new(out_dir);
    out_dir.join("boost-headers")
}
