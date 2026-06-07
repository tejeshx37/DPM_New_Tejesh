use anyhow::{bail, Context};
use std::{env, fs, io, path::Path, process::ExitStatus};

fn main() -> anyhow::Result<()> {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR should be set");
    let out_dir = Path::new(&out_dir);
    build_boost(out_dir)
}

fn build_boost(out: &Path) -> anyhow::Result<()> {
    let working_dir = &out.join("boost");
    let src_dir = Path::new("boost");
    // Copy is necessary since building process adds a lot of files to source tree
    // which causes rerun-if-changed to trigger rebuild despite having no significant changes
    build_utils::copy_dir(src_dir, working_dir)
        .context(String::from("Failed to copy boost source"))?;

    // run `bootstrap.{sh|bat}`
    let exit_status =
        boost_bootstrap(working_dir).context("boost: failed to run bootstrap script")?;
    if !exit_status.success() {
        bail!("boost: bootstrapping failed with exit status {exit_status}");
    }

    // run `b2 headers`
    let exit_status =
        boost_header_install(working_dir).context("boost: failed to run b2 engine")?;
    if !exit_status.success() {
        bail!("boost: header install failed with exit status {exit_status}");
    }

    let include_dir = out.join("boost-headers");

    build_utils::copy_dir(&working_dir.join("boost"), &include_dir.join("boost"))
        .context("Failed to copy boost headers")?;

    fs::remove_dir_all(working_dir).context("boost: failed to remove working directory")?;

    println!("cargo:rerun-if-changed={}", src_dir.display());

    Ok(())
}

fn boost_bootstrap(dir: &Path) -> io::Result<ExitStatus> {
    let toolset_arg = if cfg!(windows) {
        "mingw"
    } else {
        "--with-toolset=gcc"
    };
    build_utils::run_script("bootstrap", dir, [toolset_arg])
}

fn boost_header_install(dir: &Path) -> io::Result<ExitStatus> {
    let b2 = if cfg!(windows) {
        format!("{}", dir.join("b2.exe").display())
    } else {
        String::from("./b2")
    };
    build_utils::run_with_args(b2, dir, ["headers"])
}
