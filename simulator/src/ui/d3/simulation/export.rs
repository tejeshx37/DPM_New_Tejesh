//! CSV export for the 3D simulation page (B10).
//!
//! Writes a snapshot of the current solver state plus all captured time
//! series into a user-chosen directory. The format mirrors the 2D
//! `simulator::model::engine::export` module (separate `Points` /
//! `Elements` / `Stress` files) so downstream tooling that already
//! consumes the 2D output only needs to adapt to the extra columns.
//!
//! All writers use the `csv` crate, already a workspace dependency.

use std::{
    fs,
    io::Result as IoResult,
    path::{Path, PathBuf},
};

use cpd::d3::Computer3D;
use mesh::d3::Mesh3D;

use super::{History, State};

/// Result of an export attempt — number of files written and any error
/// surfaced to the UI.
#[derive(Debug, Default, Clone)]
pub struct ExportReport {
    pub files: Vec<PathBuf>,
    pub error: Option<String>,
}

pub fn export(state: &State, mesh_opt: Option<&Mesh3D>, dir: &Path) -> ExportReport {
    let mut report = ExportReport::default();
    if let Err(e) = fs::create_dir_all(dir) {
        report.error = Some(format!("create_dir_all failed: {e}"));
        return report;
    }

    // Snapshot of current solver state, if any.
    if let Some(c) = state.computer.as_ref() {
        if let Err(e) = write_points(c, dir, &mut report.files) {
            report.error = Some(format!("points.csv: {e}"));
            return report;
        }
        if let Err(e) = write_stress(c, dir, &mut report.files) {
            report.error = Some(format!("stress.csv: {e}"));
            return report;
        }
    }

    // Mesh connectivity if available (independent of solver state).
    if let Some(mesh) = mesh_opt {
        if let Err(e) = write_tets(mesh, dir, &mut report.files) {
            report.error = Some(format!("tetrahedra.csv: {e}"));
            return report;
        }
    }

    // Time-series histories.
    if let Err(e) = write_history(&state.history, dir, &mut report.files) {
        report.error = Some(format!("history: {e}"));
    }
    report
}

fn write_points(c: &Computer3D, dir: &Path, files: &mut Vec<PathBuf>) -> IoResult<()> {
    let path = dir.join("points.csv");
    let mut w = csv::Writer::from_path(&path).map_err(io_err)?;
    w.write_record(["index", "x", "y", "z", "ux", "uy", "uz", "vx", "vy", "vz"])
        .map_err(io_err)?;
    for (i, n) in c.nodes.iter().enumerate() {
        let d = n.position - n.initial_position;
        w.write_record([
            i.to_string(),
            f(n.position.x),
            f(n.position.y),
            f(n.position.z),
            f(d.x),
            f(d.y),
            f(d.z),
            f(n.velocity.x),
            f(n.velocity.y),
            f(n.velocity.z),
        ])
        .map_err(io_err)?;
    }
    w.flush()?;
    files.push(path);
    Ok(())
}

fn write_stress(c: &Computer3D, dir: &Path, files: &mut Vec<PathBuf>) -> IoResult<()> {
    let path = dir.join("stress.csv");
    let mut w = csv::Writer::from_path(&path).map_err(io_err)?;
    w.write_record([
        "index",
        "sigma_xx",
        "sigma_yy",
        "sigma_zz",
        "sigma_yz",
        "sigma_xz",
        "sigma_xy",
        "von_mises",
        "strain_energy",
        "is_broken",
    ])
    .map_err(io_err)?;
    for (i, e) in c.elements.iter().enumerate() {
        let s = &e.stress;
        let vm = von_mises(s);
        w.write_record([
            i.to_string(),
            f(s.m11),
            f(s.m22),
            f(s.m33),
            f(s.m23),
            f(s.m13),
            f(s.m12),
            f(vm),
            f(e.strain_energy),
            e.is_broken.to_string(),
        ])
        .map_err(io_err)?;
    }
    w.flush()?;
    files.push(path);
    Ok(())
}

fn write_tets(mesh: &Mesh3D, dir: &Path, files: &mut Vec<PathBuf>) -> IoResult<()> {
    let path = dir.join("tetrahedra.csv");
    let mut w = csv::Writer::from_path(&path).map_err(io_err)?;
    w.write_record(["index", "n0", "n1", "n2", "n3"])
        .map_err(io_err)?;
    for (i, t) in mesh.tetrahedra.iter().enumerate() {
        w.write_record([
            i.to_string(),
            t[0].to_string(),
            t[1].to_string(),
            t[2].to_string(),
            t[3].to_string(),
        ])
        .map_err(io_err)?;
    }
    w.flush()?;
    files.push(path);
    Ok(())
}

fn write_history(history: &History, dir: &Path, files: &mut Vec<PathBuf>) -> IoResult<()> {
    // VM stats over time.
    if !history.stress.is_empty() {
        let path = dir.join("history_stress.csv");
        let mut w = csv::Writer::from_path(&path).map_err(io_err)?;
        w.write_record(["t", "vm_min", "vm_mean", "vm_max"])
            .map_err(io_err)?;
        for (t, s) in &history.stress {
            w.write_record([
                f(*t),
                f(s.min_von_mises),
                f(s.mean_von_mises),
                f(s.max_von_mises),
            ])
            .map_err(io_err)?;
        }
        w.flush()?;
        files.push(path);
    }
    // Per-region averaged displacement + force.
    if !history.regions.is_empty() {
        let path = dir.join("history_regions.csv");
        let mut w = csv::Writer::from_path(&path).map_err(io_err)?;
        w.write_record([
            "t", "region", "ux", "uy", "uz", "fx", "fy", "fz",
        ])
        .map_err(io_err)?;
        // Sorted for deterministic output.
        let mut names: Vec<&String> = history.regions.keys().collect();
        names.sort();
        for name in names {
            for (t, a) in &history.regions[name] {
                w.write_record([
                    f(*t),
                    name.clone(),
                    f(a.mean_displacement.x),
                    f(a.mean_displacement.y),
                    f(a.mean_displacement.z),
                    f(a.mean_force.x),
                    f(a.mean_force.y),
                    f(a.mean_force.z),
                ])
                .map_err(io_err)?;
            }
        }
        w.flush()?;
        files.push(path);
    }
    // Inspect element VM history.
    if !history.element.is_empty() {
        let path = dir.join("history_inspect_element.csv");
        let mut w = csv::Writer::from_path(&path).map_err(io_err)?;
        w.write_record(["t", "von_mises"]).map_err(io_err)?;
        for (t, v) in &history.element {
            w.write_record([f(*t), f(*v)]).map_err(io_err)?;
        }
        w.flush()?;
        files.push(path);
    }
    // Inspect vertex |u| and |F| history.
    if !history.vertex.is_empty() {
        let path = dir.join("history_inspect_vertex.csv");
        let mut w = csv::Writer::from_path(&path).map_err(io_err)?;
        w.write_record(["t", "disp_mag", "force_mag"]).map_err(io_err)?;
        for (t, d, fmag) in &history.vertex {
            w.write_record([f(*t), f(*d), f(*fmag)]).map_err(io_err)?;
        }
        w.flush()?;
        files.push(path);
    }
    Ok(())
}

fn f(v: f32) -> String {
    format!("{v}")
}

fn von_mises(s: &nalgebra::Matrix3<f32>) -> f32 {
    let d12 = s.m11 - s.m22;
    let d23 = s.m22 - s.m33;
    let d31 = s.m33 - s.m11;
    let sh = s.m12 * s.m12 + s.m23 * s.m23 + s.m13 * s.m13;
    (0.5 * (d12 * d12 + d23 * d23 + d31 * d31 + 6.0 * sh)).sqrt()
}

fn io_err(e: csv::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
}
