//! Area-selection tool for the 3D Meshing page: drag a circle or rectangle
//! on the viewport, project it onto the shape's surface via raycasting, and
//! turn it into a [`super::RefinementRegion`].
//!
//! Lifecycle (all driven off the same `egui::Response` the camera uses,
//! primary-button drag only):
//! - drag start: raycast the press position against every shape, anchor to
//!   the nearest hit (shape index, surface point, normal, tangent axes).
//! - drag continue: raycast the *current* cursor against the tangent plane
//!   at the anchor (not the curved surface again) to measure the extent —
//!   more stable near silhouette edges than re-hitting the surface.
//! - drag release: commit a region and ask the caller to regenerate.
//! - Escape: abort the pending drag without creating a region.

use egui::{Color32, Key, Pos2, Stroke};
use nalgebra::{Matrix4, Vector3};

use super::super::drawing::{raycast, shape::Geometry3D, viewport::ray_from_screen};
use super::{MeshToolMode, RefinementRegion, RefinementShape, State};

#[derive(Debug, Clone)]
pub struct PendingSelection {
    shape_index: usize,
    kind: MeshToolMode,
    anchor_world: Vector3<f64>,
    normal: Vector3<f64>,
    tangent_u: Vector3<f64>,
    tangent_v: Vector3<f64>,
    start_screen: Pos2,
    cur_screen: Pos2,
    /// Last valid extent, reused when the tangent-plane raycast momentarily
    /// goes unstable (near-grazing view angle) so the drag doesn't jitter.
    last_extent: [f64; 2],
}

/// Drive the drag lifecycle for one frame. Returns `true` if a region was
/// just committed (caller should regenerate the mesh).
pub fn handle_selection_input(
    state: &mut State,
    geometry: &Geometry3D,
    response: &egui::Response,
    rect: egui::Rect,
    view_proj: &Matrix4<f64>,
    ui: &mut egui::Ui,
) -> bool {
    if state.pending_selection.is_some() && ui.input(|i| i.key_pressed(Key::Escape)) {
        state.pending_selection = None;
        return false;
    }

    let dragging_primary = response.dragged_by(egui::PointerButton::Primary);

    if dragging_primary && state.pending_selection.is_none() {
        // Drag start: raycast against every shape, anchor to the nearest.
        let Some(press_pos) = response.interact_pointer_pos() else {
            return false;
        };
        let Some((origin, dir)) = ray_from_screen(view_proj, rect, press_pos) else {
            return false;
        };
        let shapes = geometry.shapes.iter().map(|(s, _)| s).enumerate();
        if let Some((shape_index, hit)) = raycast::intersect_nearest(shapes, origin, dir) {
            let default_extent = shape_extent_estimate(geometry, shape_index);
            state.pending_selection = Some(PendingSelection {
                shape_index,
                kind: state.tool_mode,
                anchor_world: hit.point,
                normal: hit.normal,
                tangent_u: hit.tangent_u,
                tangent_v: hit.tangent_v,
                start_screen: press_pos,
                cur_screen: press_pos,
                last_extent: [default_extent * 0.15, default_extent * 0.15],
            });
        }
        return false;
    }

    if dragging_primary {
        // Drag continue: update the tracked cursor and re-derive the
        // extent from a tangent-plane raycast (see module docs).
        if let Some(cur_pos) = response.interact_pointer_pos() {
            if let Some(pending) = state.pending_selection.as_mut() {
                pending.cur_screen = cur_pos;
                if let Some((origin, dir)) = ray_from_screen(view_proj, rect, cur_pos) {
                    update_extent(pending, origin, dir);
                }
            }
        }
        return false;
    }

    // Not dragging: if a selection was pending, this is the release frame.
    if let Some(pending) = state.pending_selection.take() {
        let region = build_region(state, &pending);
        state.refinement_regions.push(region);
        return true;
    }

    false
}

/// Ray-vs-tangent-plane intersection at the anchor, used to measure how far
/// the cursor has been dragged in world units without re-raycasting the
/// (possibly curved) shape surface directly.
fn update_extent(pending: &mut PendingSelection, origin: Vector3<f64>, dir: Vector3<f64>) {
    let denom = dir.dot(&pending.normal);
    if denom.abs() < 1e-6 {
        // Near-parallel to the tangent plane: freeze at the last valid
        // extent rather than producing an unstable/huge value.
        return;
    }
    let t = (pending.anchor_world - origin).dot(&pending.normal) / denom;
    if t < 0.0 {
        return;
    }
    let point = origin + dir * t;
    let rel = point - pending.anchor_world;
    let u = rel.dot(&pending.tangent_u);
    let v = rel.dot(&pending.tangent_v);

    match pending.kind {
        MeshToolMode::SelectCircle => {
            let r = (u * u + v * v).sqrt();
            pending.last_extent = [r, r];
        }
        MeshToolMode::SelectRectangle => {
            pending.last_extent = [u.abs(), v.abs()];
        }
        MeshToolMode::Camera => {}
    }
}

fn build_region(state: &mut State, pending: &PendingSelection) -> RefinementRegion {
    let id = state.next_region_id;
    state.next_region_id += 1;

    let shape = match pending.kind {
        MeshToolMode::SelectRectangle => RefinementShape::Rectangle {
            center_world: pending.anchor_world,
            half_extents_world: [
                pending.last_extent[0].max(1e-6),
                pending.last_extent[1].max(1e-6),
            ],
            u_axis: pending.tangent_u,
            v_axis: pending.tangent_v,
        },
        _ => RefinementShape::Circle {
            center_world: pending.anchor_world,
            radius_world: pending.last_extent[0].max(1e-6),
            normal_world: pending.normal,
        },
    };

    RefinementRegion {
        id,
        shape_index: pending.shape_index,
        shape,
        density_multiplier: 1.0,
        falloff: 0.5,
    }
}

/// Rough size estimate for a shape (its AABB diagonal), used to size the
/// default drag extent before the user has moved the cursor at all.
fn shape_extent_estimate(geometry: &Geometry3D, shape_index: usize) -> f64 {
    geometry
        .shapes
        .get(shape_index)
        .map(|(shape, _)| {
            let (lo, hi) = shape.aabb();
            (hi - lo).norm()
        })
        .unwrap_or(1.0)
}

/// Paint the in-progress drag (screen-space, translucent) and every
/// committed region (projected world-space outline + label).
pub fn paint_overlays(
    state: &State,
    painter: &egui::Painter,
    rect: egui::Rect,
    view_proj: &Matrix4<f64>,
) {
    if let Some(pending) = &state.pending_selection {
        let color = Color32::from_rgba_unmultiplied(255, 220, 80, 90);
        let stroke = Stroke::new(1.5, Color32::from_rgb(255, 220, 80));
        match pending.kind {
            MeshToolMode::SelectRectangle => {
                let r = egui::Rect::from_two_pos(pending.start_screen, pending.cur_screen);
                painter.rect_filled(r, 0.0, color);
                painter.rect_stroke(r, 0.0, stroke);
            }
            _ => {
                let radius = pending.start_screen.distance(pending.cur_screen);
                painter.circle_filled(pending.start_screen, radius, color);
                painter.circle_stroke(pending.start_screen, radius, stroke);
            }
        }
    }

    for region in &state.refinement_regions {
        paint_committed_region(region, painter, rect, view_proj);
    }
}

fn paint_committed_region(
    region: &RefinementRegion,
    painter: &egui::Painter,
    rect: egui::Rect,
    view_proj: &Matrix4<f64>,
) {
    let refine = region.density_multiplier >= 1.0;
    let color = if refine {
        Color32::from_rgb(90, 220, 140)
    } else {
        Color32::from_rgb(230, 140, 90)
    };
    let stroke = Stroke::new(1.5, color);

    let (centroid, boundary_points) = match &region.shape {
        RefinementShape::Circle {
            center_world,
            radius_world,
            normal_world,
        } => {
            let (u, v) = tangent_axes(*normal_world);
            let n = 24;
            let pts: Vec<Vector3<f64>> = (0..=n)
                .map(|i| {
                    let theta = (i as f64) / (n as f64) * std::f64::consts::TAU;
                    center_world + (u * theta.cos() + v * theta.sin()) * *radius_world
                })
                .collect();
            (*center_world, pts)
        }
        RefinementShape::Rectangle {
            center_world,
            half_extents_world,
            u_axis,
            v_axis,
        } => {
            let hu = half_extents_world[0];
            let hv = half_extents_world[1];
            let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0)];
            let pts = corners
                .iter()
                .map(|(su, sv)| center_world + u_axis * (su * hu) + v_axis * (sv * hv))
                .collect();
            (*center_world, pts)
        }
    };

    let screen_pts: Vec<Pos2> = boundary_points
        .iter()
        .filter_map(|p| super::super::drawing::viewport::project(view_proj, rect, *p))
        .collect();
    if screen_pts.len() >= 2 {
        painter.add(egui::Shape::line(screen_pts, stroke));
    }
    if let Some(label_pos) = super::super::drawing::viewport::project(view_proj, rect, centroid) {
        painter.text(
            label_pos,
            egui::Align2::CENTER_CENTER,
            format!("R{}: x{:.1}", region.id, region.density_multiplier),
            egui::FontId::monospace(10.0),
            color,
        );
    }
}

fn tangent_axes(normal: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let helper = if normal.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let u = normal.cross(&helper).normalize();
    let v = normal.cross(&u).normalize();
    (u, v)
}
