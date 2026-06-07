use anyhow::Context;
use std::{env, path::Path};

const CPP_FILES: &[&str] = &[
    "cpp/curve.cpp",
    "cpp/num.cpp",
    "cpp/point.cpp",
    "cpp/polygon_set.cpp",
    "cpp/polygon.cpp",
    "cpp/polygon_with_holes.cpp",
    "cpp/triangulation.cpp",
];

fn main() -> anyhow::Result<()> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let gmp_mpfr_headers =
        env::var("DEP_GMP_INCLUDE_DIR").expect("gmp-mpfr-sys crate should set this metadataa");

    cxx_build::bridges(["src/lib.rs", "src/triangulation.rs"])
        .compiler("g++")
        .cpp(true)
        .files(CPP_FILES)
        .includes([
            &manifest_dir.join("include"),
            &boost_sys::headers(),
            Path::new(&gmp_mpfr_headers),
        ])
        .std("c++20")
        .try_compile("cgal")
        .context("CGAL wrapper compilation failed")?;

    println!("cargo:rerun-if-changed=cpp");

    println!("cargo:rustc-link-lib=static=gmp");
    println!("cargo:rustc-link-lib=static=mpfr");

    Ok(())
}
