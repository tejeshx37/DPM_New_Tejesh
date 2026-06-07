//! 3D drawing page: parametric shape dialogs + preview viewport.
//!
//! For this milestone the preview viewport renders shapes as wireframes
//! using egui's 2D `Painter` with a manual perspective projection. The
//! intended successor is an embedded wgpu surface via `egui-wgpu`; the
//! `viewport` module is structured so the renderer can be swapped without
//! touching the page or dialog code.

pub mod dialogs;
pub mod shape;
pub mod viewport;

use egui::{Align, Layout, ScrollArea, SidePanel, Ui};
use serde::{Deserialize, Serialize};

use shape::{CsgOp3D, Geometry3D, Shape3D, ShapeKind};
use viewport::ViewportState;

/// Persisted drawing-page state for 3D projects.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct State {
    #[serde(default)]
    pub geometry: Geometry3D,
    #[serde(default)]
    pub current_op: CsgOp3D,
    /// Dialog currently open, if any.
    #[serde(default)]
    pub open_dialog: Option<ShapeKind>,
    /// In-progress parameters for the open dialog.
    #[serde(default)]
    pub dialog_state: dialogs::DialogState,
    #[serde(default)]
    pub viewport: ViewportState,
}

/// Drives the 3D drawing page UI. Called from `App::add_contents` when the
/// active project's dimension is `D3`.
pub fn show(state: &mut State, ui: &mut Ui) {
    SidePanel::right("d3_drawing_side_panel")
        .resizable(true)
        .default_width(220.0)
        .show_inside(ui, |ui| {
            add_toolbar(state, ui);
            ui.separator();
            add_shape_list(state, ui);
        });

    ui.with_layout(Layout::top_down(Align::Center), |ui| {
        viewport::show(&mut state.viewport, &state.geometry, ui);
    });

    if let Some(kind) = state.open_dialog {
        if let Some(shape) = dialogs::show(kind, &mut state.dialog_state, ui.ctx()) {
            state.geometry.shapes.push((shape, state.current_op));
            state.open_dialog = None;
        }
    }
}

fn add_toolbar(state: &mut State, ui: &mut Ui) {
    ui.heading("3D Shapes");
    ui.label("Add a primitive:");

    let mut open = |kind: ShapeKind| {
        state.open_dialog = Some(kind);
        state.dialog_state = dialogs::DialogState::for_kind(kind);
    };

    if ui.button("Cube").clicked() {
        open(ShapeKind::Cube);
    }
    if ui.button("Cuboid").clicked() {
        open(ShapeKind::Cuboid);
    }
    if ui.button("Sphere").clicked() {
        open(ShapeKind::Sphere);
    }
    if ui.button("Cylinder").clicked() {
        open(ShapeKind::Cylinder);
    }

    ui.add_space(6.0);
    ui.label("CSG operation for next shape:");
    ui.horizontal(|ui| {
        ui.selectable_value(&mut state.current_op, CsgOp3D::Union, "Union");
        ui.selectable_value(&mut state.current_op, CsgOp3D::Difference, "Difference");
    });
}

fn add_shape_list(state: &mut State, ui: &mut Ui) {
    ui.label(format!("Shapes ({})", state.geometry.shapes.len()));
    let mut remove_idx: Option<usize> = None;
    ScrollArea::vertical().show(ui, |ui| {
        for (idx, entry) in state.geometry.shapes.iter().enumerate() {
            let shape: &Shape3D = &entry.0;
            let op: &CsgOp3D = &entry.1;
            ui.horizontal(|ui| {
                ui.label(format!(
                    "{}. {} ({})",
                    idx + 1,
                    shape.kind().label(),
                    op.label()
                ));
                if ui.small_button("x").clicked() {
                    remove_idx = Some(idx);
                }
            });
        }
    });
    if let Some(idx) = remove_idx {
        state.geometry.shapes.remove(idx);
    }
}
