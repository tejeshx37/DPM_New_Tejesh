//! 3D shape data types — the four primitives the simulator supports
//! parametrically: Cube, Cuboid, Sphere, Cylinder. Also the CSG operation
//! enum mirroring the 2D Join/Difference flow.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Tag for the four primitive shape kinds. Used to drive dialog selection
/// and shape-list labels without matching on the full enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShapeKind {
    Cube,
    Cuboid,
    Sphere,
    Cylinder,
}

impl ShapeKind {
    pub fn label(self) -> &'static str {
        match self {
            ShapeKind::Cube => "Cube",
            ShapeKind::Cuboid => "Cuboid",
            ShapeKind::Sphere => "Sphere",
            ShapeKind::Cylinder => "Cylinder",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Shape3D {
    Cube {
        center: Vector3<f64>,
        size: f64,
    },
    Cuboid {
        center: Vector3<f64>,
        extents: Vector3<f64>,
    },
    Sphere {
        center: Vector3<f64>,
        radius: f64,
    },
    Cylinder {
        base_center: Vector3<f64>,
        axis: Vector3<f64>,
        radius: f64,
        height: f64,
    },
}

impl Shape3D {
    pub fn kind(&self) -> ShapeKind {
        match self {
            Shape3D::Cube { .. } => ShapeKind::Cube,
            Shape3D::Cuboid { .. } => ShapeKind::Cuboid,
            Shape3D::Sphere { .. } => ShapeKind::Sphere,
            Shape3D::Cylinder { .. } => ShapeKind::Cylinder,
        }
    }

    /// Canonical boundary-region names that the mesher will produce for
    /// this shape. Used by the Boundary Conditions phase to enumerate
    /// regions *before* a mesh has been generated, so the user can
    /// assign BCs against shapes directly. Stays in sync with
    /// `mesh::d3::cuboid::generate`, `sphere::generate`, and
    /// `cylinder::generate` — if those add or rename regions, update
    /// this list too.
    pub fn region_names(&self) -> &'static [&'static str] {
        match self {
            Shape3D::Cube { .. } | Shape3D::Cuboid { .. } => {
                &["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
            }
            Shape3D::Sphere { .. } => &["surface"],
            Shape3D::Cylinder { .. } => &["side", "bottom", "top"],
        }
    }

    /// Axis-aligned bounding box for camera framing. Returns (min, max).
    pub fn aabb(&self) -> (Vector3<f64>, Vector3<f64>) {
        match self {
            Shape3D::Cube { center, size } => {
                let h = Vector3::new(*size, *size, *size) * 0.5;
                (center - h, center + h)
            }
            Shape3D::Cuboid { center, extents } => {
                let h = extents * 0.5;
                (center - h, center + h)
            }
            Shape3D::Sphere { center, radius } => {
                let h = Vector3::new(*radius, *radius, *radius);
                (center - h, center + h)
            }
            Shape3D::Cylinder {
                base_center,
                axis,
                radius,
                height,
            } => {
                // Conservative AABB: take radius in all directions and the
                // full axis extent. Fine for camera framing.
                let axis_n = axis.normalize();
                let top = base_center + axis_n * *height;
                let r = Vector3::new(*radius, *radius, *radius);
                let lo = base_center.inf(&top) - r;
                let hi = base_center.sup(&top) + r;
                (lo, hi)
            }
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CsgOp3D {
    #[default]
    Union,
    Difference,
}

impl CsgOp3D {
    pub fn label(self) -> &'static str {
        match self {
            CsgOp3D::Union => "Union",
            CsgOp3D::Difference => "Difference",
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Geometry3D {
    /// Ordered list of shape + operation pairs. Applied in order, mirroring
    /// the 2D Join/Difference pipeline. CSG boolean evaluation itself is
    /// Step 3 work; for now the list is interpreted only by the preview
    /// renderer (which shows Difference shapes as wireframes to signal
    /// subtractive intent).
    #[serde(default)]
    pub shapes: Vec<(Shape3D, CsgOp3D)>,
}

impl Geometry3D {
    /// Combined bounding box of all shapes, or a unit cube at the origin if
    /// empty. Used by the viewport camera to auto-frame the scene.
    pub fn aabb(&self) -> (Vector3<f64>, Vector3<f64>) {
        let mut iter = self.shapes.iter().map(|(s, _)| s.aabb());
        let Some((mut lo, mut hi)) = iter.next() else {
            return (Vector3::new(-0.5, -0.5, -0.5), Vector3::new(0.5, 0.5, 0.5));
        };
        for (l, h) in iter {
            lo = lo.inf(&l);
            hi = hi.sup(&h);
        }
        (lo, hi)
    }
}
