//! Dedicated 3D Boundary Conditions phase.
//!
//! Mirrors the 2D pipeline's separate BC step. The user picks a BC kind
//! per boundary region here; the Simulation page consumes those choices
//! when it builds the solver. The page also renders the mesh with the
//! current BC arrows / pinned markers overlaid, so users can see what
//! their settings will do without leaving the page.

use egui::{Color32, SidePanel, Ui};
use mesh::d3::Mesh3D;

use super::simulation;

pub fn show(sim_state: &mut simulation::State, meshes: &[Option<Mesh3D>], ui: &mut Ui) {
    let combined = simulation::combine_active(meshes);
    SidePanel::right("d3_bc_side_panel")
        .resizable(true)
        .default_width(320.0)
        .show_inside(ui, |ui| {
            ui.heading("Boundary Conditions");
            ui.separator();
            match combined.as_ref() {
                None => {
                    ui.colored_label(
                        Color32::YELLOW,
                        "No mesh available. Generate one on the Meshing page first.",
                    );
                }
                Some(mesh) => {
                    ui.label(format!(
                        "{} boundary region{}",
                        mesh.boundary_faces.regions.len(),
                        if mesh.boundary_faces.regions.len() == 1 {
                            ""
                        } else {
                            "s"
                        }
                    ));
                    ui.separator();
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        simulation::add_bc_controls(sim_state, mesh, ui);
                    });
                }
            }
        });
    simulation::show_viewport(sim_state, combined.as_ref(), ui);
}
