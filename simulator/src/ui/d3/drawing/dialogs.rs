//! Parametric dialogs for the four 3D primitives. Each dialog is an
//! `egui::Window`; the active dialog is tracked via `ShapeKind` in the page
//! state and the in-progress numeric fields live in `DialogState`.

use egui::{Context, DragValue, Window};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use super::shape::{Shape3D, ShapeKind};

/// Mutable scratch state for whichever shape dialog is open. Carries enough
/// numeric fields for every shape kind; the dialog reads/writes only the
/// subset it needs.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct DialogState {
    pub center: [f64; 3],
    pub size: f64,
    pub extents: [f64; 3],
    pub radius: f64,
    pub axis: [f64; 3],
    pub height: f64,
}

impl DialogState {
    /// Sensible defaults per shape kind so the user lands on a usable shape.
    pub fn for_kind(kind: ShapeKind) -> Self {
        match kind {
            ShapeKind::Cube => Self {
                center: [0.0, 0.0, 0.0],
                size: 1.0,
                ..Default::default()
            },
            ShapeKind::Cuboid => Self {
                center: [0.0, 0.0, 0.0],
                extents: [2.0, 1.0, 1.0],
                ..Default::default()
            },
            ShapeKind::Sphere => Self {
                center: [0.0, 0.0, 0.0],
                radius: 0.5,
                ..Default::default()
            },
            ShapeKind::Cylinder => Self {
                center: [0.0, 0.0, 0.0],
                radius: 0.5,
                axis: [0.0, 0.0, 1.0],
                height: 1.0,
                ..Default::default()
            },
        }
    }
}

/// Shows the dialog for `kind`. Returns the constructed shape on the frame
/// the user clicks "Add". `None` otherwise.
pub fn show(kind: ShapeKind, state: &mut DialogState, ctx: &Context) -> Option<Shape3D> {
    let mut result: Option<Shape3D> = None;
    let mut open = true;
    Window::new(format!("New {}", kind.label()))
        .open(&mut open)
        .resizable(false)
        .show(ctx, |ui| {
            match kind {
                ShapeKind::Cube => {
                    center_row(ui, &mut state.center);
                    ui.horizontal(|ui| {
                        ui.label("Size");
                        ui.add(DragValue::new(&mut state.size).speed(0.05).clamp_range(1e-6..=f64::INFINITY));
                    });
                    if ui.button("Add to scene").clicked() && state.size > 0.0 {
                        result = Some(Shape3D::Cube {
                            center: Vector3::from(state.center),
                            size: state.size,
                        });
                    }
                }
                ShapeKind::Cuboid => {
                    center_row(ui, &mut state.center);
                    ui.horizontal(|ui| {
                        ui.label("Extents (x, y, z)");
                        ui.add(DragValue::new(&mut state.extents[0]).speed(0.05).clamp_range(1e-6..=f64::INFINITY));
                        ui.add(DragValue::new(&mut state.extents[1]).speed(0.05).clamp_range(1e-6..=f64::INFINITY));
                        ui.add(DragValue::new(&mut state.extents[2]).speed(0.05).clamp_range(1e-6..=f64::INFINITY));
                    });
                    if ui.button("Add to scene").clicked()
                        && state.extents.iter().all(|e| *e > 0.0)
                    {
                        result = Some(Shape3D::Cuboid {
                            center: Vector3::from(state.center),
                            extents: Vector3::from(state.extents),
                        });
                    }
                }
                ShapeKind::Sphere => {
                    center_row(ui, &mut state.center);
                    ui.horizontal(|ui| {
                        ui.label("Radius");
                        ui.add(DragValue::new(&mut state.radius).speed(0.05).clamp_range(1e-6..=f64::INFINITY));
                    });
                    if ui.button("Add to scene").clicked() && state.radius > 0.0 {
                        result = Some(Shape3D::Sphere {
                            center: Vector3::from(state.center),
                            radius: state.radius,
                        });
                    }
                }
                ShapeKind::Cylinder => {
                    ui.horizontal(|ui| {
                        ui.label("Base center");
                        ui.add(DragValue::new(&mut state.center[0]).speed(0.05));
                        ui.add(DragValue::new(&mut state.center[1]).speed(0.05));
                        ui.add(DragValue::new(&mut state.center[2]).speed(0.05));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Axis direction");
                        ui.add(DragValue::new(&mut state.axis[0]).speed(0.05));
                        ui.add(DragValue::new(&mut state.axis[1]).speed(0.05));
                        ui.add(DragValue::new(&mut state.axis[2]).speed(0.05));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Radius");
                        ui.add(DragValue::new(&mut state.radius).speed(0.05).clamp_range(1e-6..=f64::INFINITY));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Height");
                        ui.add(DragValue::new(&mut state.height).speed(0.05).clamp_range(1e-6..=f64::INFINITY));
                    });
                    let axis_vec = Vector3::from(state.axis);
                    let axis_ok = axis_vec.norm() > 1e-9;
                    let geom_ok = state.radius > 0.0 && state.height > 0.0;
                    if ui
                        .add_enabled(axis_ok && geom_ok, egui::Button::new("Add to scene"))
                        .clicked()
                    {
                        result = Some(Shape3D::Cylinder {
                            base_center: Vector3::from(state.center),
                            axis: axis_vec,
                            radius: state.radius,
                            height: state.height,
                        });
                    }
                    if !axis_ok {
                        ui.colored_label(egui::Color32::YELLOW, "Axis direction must be non-zero");
                    }
                }
            }
        });
    result
}

fn center_row(ui: &mut egui::Ui, center: &mut [f64; 3]) {
    ui.horizontal(|ui| {
        ui.label("Center (x, y, z)");
        ui.add(DragValue::new(&mut center[0]).speed(0.05));
        ui.add(DragValue::new(&mut center[1]).speed(0.05));
        ui.add(DragValue::new(&mut center[2]).speed(0.05));
    });
}
