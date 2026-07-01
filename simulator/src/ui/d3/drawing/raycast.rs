//! Closed-form ray-vs-primitive intersection for `Shape3D`.
//!
//! Because the four primitives (Cube, Cuboid, Sphere, Cylinder) are all
//! analytic shapes, this is plain ray-primitive intersection math — no BVH
//! or mesh-triangle raycasting is needed. Used by the Meshing page's
//! area-selection tool to figure out which shape/surface point a screen
//! click lands on.

use nalgebra::Vector3;

use super::shape::Shape3D;

/// Result of a successful ray-shape intersection.
pub struct RayHit {
    /// Ray parameter at the hit point (`origin + dir * t`).
    pub t: f64,
    pub point: Vector3<f64>,
    pub normal: Vector3<f64>,
    pub tangent_u: Vector3<f64>,
    pub tangent_v: Vector3<f64>,
}

/// Intersect a ray (`origin`, normalized `dir`) against a single shape.
/// Returns the nearest hit with `t >= 0`, or `None` if the ray misses.
pub fn intersect_shape(shape: &Shape3D, origin: Vector3<f64>, dir: Vector3<f64>) -> Option<RayHit> {
    match shape {
        Shape3D::Cube { center, size } => {
            let extents = Vector3::new(*size, *size, *size);
            intersect_aabb(*center, extents, origin, dir)
        }
        Shape3D::Cuboid { center, extents } => intersect_aabb(*center, *extents, origin, dir),
        Shape3D::Sphere { center, radius } => intersect_sphere(*center, *radius, origin, dir),
        Shape3D::Cylinder {
            base_center,
            axis,
            radius,
            height,
        } => intersect_cylinder(*base_center, *axis, *radius, *height, origin, dir),
    }
}

/// Find the nearest hit across every shape in a slice, tagged with its
/// index in the slice. Used at drag-start to pick which shape the user
/// clicked on.
pub fn intersect_nearest<'a>(
    shapes: impl Iterator<Item = (usize, &'a Shape3D)>,
    origin: Vector3<f64>,
    dir: Vector3<f64>,
) -> Option<(usize, RayHit)> {
    shapes
        .filter_map(|(idx, shape)| intersect_shape(shape, origin, dir).map(|hit| (idx, hit)))
        .min_by(|(_, a), (_, b)| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal))
}

fn orthonormal_basis(normal: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let helper = if normal.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let u = normal.cross(&helper).normalize();
    let v = normal.cross(&u).normalize();
    (u, v)
}

/// Ray-vs-axis-aligned-box slab test. Recovers the hit face's normal from
/// whichever axis produced the tightest (entry) slab bound.
fn intersect_aabb(
    center: Vector3<f64>,
    extents: Vector3<f64>,
    origin: Vector3<f64>,
    dir: Vector3<f64>,
) -> Option<RayHit> {
    let half = extents * 0.5;
    let lo = center - half;
    let hi = center + half;

    let mut t_min = f64::NEG_INFINITY;
    let mut t_max = f64::INFINITY;
    let mut hit_axis = 0usize;
    let mut hit_sign = -1.0_f64;

    for axis in 0..3 {
        let d = dir[axis];
        let o = origin[axis];
        if d.abs() < 1e-12 {
            if o < lo[axis] || o > hi[axis] {
                return None;
            }
            continue;
        }
        let inv_d = 1.0 / d;
        let mut t0 = (lo[axis] - o) * inv_d;
        let mut t1 = (hi[axis] - o) * inv_d;
        let mut sign = -1.0;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
            sign = 1.0;
        }
        if t0 > t_min {
            t_min = t0;
            hit_axis = axis;
            hit_sign = sign;
        }
        t_max = t_max.min(t1);
        if t_min > t_max {
            return None;
        }
    }

    let t = if t_min >= 0.0 { t_min } else { t_max };
    if t < 0.0 {
        return None;
    }

    let point = origin + dir * t;
    let mut normal = Vector3::zeros();
    normal[hit_axis] = hit_sign;
    let (tangent_u, tangent_v) = orthonormal_basis(normal);

    Some(RayHit {
        t,
        point,
        normal,
        tangent_u,
        tangent_v,
    })
}

fn intersect_sphere(
    center: Vector3<f64>,
    radius: f64,
    origin: Vector3<f64>,
    dir: Vector3<f64>,
) -> Option<RayHit> {
    let oc = origin - center;
    let b = oc.dot(&dir);
    let c = oc.dot(&oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let t0 = -b - sqrt_disc;
    let t1 = -b + sqrt_disc;
    let t = if t0 >= 0.0 {
        t0
    } else if t1 >= 0.0 {
        t1
    } else {
        return None;
    };

    let point = origin + dir * t;
    let normal = (point - center).normalize();
    let (tangent_u, tangent_v) = orthonormal_basis(normal);

    Some(RayHit {
        t,
        point,
        normal,
        tangent_u,
        tangent_v,
    })
}

/// Ray vs. a finite cylinder: infinite-cylinder quadratic clipped to the
/// height range, plus the two end-cap disks. The three sub-cases (side,
/// bottom cap, top cap) are evaluated independently and the nearest valid
/// hit wins.
fn intersect_cylinder(
    base_center: Vector3<f64>,
    axis: Vector3<f64>,
    radius: f64,
    height: f64,
    origin: Vector3<f64>,
    dir: Vector3<f64>,
) -> Option<RayHit> {
    let axis_n = if axis.norm() < 1e-9 {
        Vector3::new(0.0, 0.0, 1.0)
    } else {
        axis.normalize()
    };

    let mut best: Option<RayHit> = None;
    let mut consider = |hit: RayHit| {
        if hit.t >= 0.0 && best.as_ref().map_or(true, |b| hit.t < b.t) {
            best = Some(hit);
        }
    };

    // Side (lateral) surface: project the ray onto the plane perpendicular
    // to the axis and solve the 2D circle intersection.
    let rel_o = origin - base_center;
    let o_axial = rel_o.dot(&axis_n);
    let o_perp = rel_o - axis_n * o_axial;
    let d_axial = dir.dot(&axis_n);
    let d_perp = dir - axis_n * d_axial;

    let a = d_perp.dot(&d_perp);
    if a > 1e-12 {
        let b = 2.0 * o_perp.dot(&d_perp);
        let c = o_perp.dot(&o_perp) - radius * radius;
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let sqrt_disc = disc.sqrt();
            for t in [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)] {
                if t < 0.0 {
                    continue;
                }
                let z = o_axial + d_axial * t;
                if z < 0.0 || z > height {
                    continue;
                }
                let point = origin + dir * t;
                let radial = point - (base_center + axis_n * z);
                let normal = radial.normalize();
                let (tangent_u, tangent_v) = orthonormal_basis(normal);
                consider(RayHit {
                    t,
                    point,
                    normal,
                    tangent_u,
                    tangent_v,
                });
                break; // nearest valid root only
            }
        }
    }

    // Cap disks (bottom at z=0, top at z=height): ray-vs-plane, then
    // radius check.
    for (z_plane, normal_sign) in [(0.0, -1.0), (height, 1.0)] {
        if d_axial.abs() < 1e-12 {
            continue;
        }
        let t = (z_plane - o_axial) / d_axial;
        if t < 0.0 {
            continue;
        }
        let point = origin + dir * t;
        let radial = point - (base_center + axis_n * z_plane);
        if radial.norm() > radius {
            continue;
        }
        let normal = axis_n * normal_sign;
        let (tangent_u, tangent_v) = orthonormal_basis(normal);
        consider(RayHit {
            t,
            point,
            normal,
            tangent_u,
            tangent_v,
        });
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_hits_cube_face() {
        let shape = Shape3D::Cube {
            center: Vector3::zeros(),
            size: 2.0,
        };
        let hit = intersect_shape(&shape, Vector3::new(-5.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0))
            .expect("should hit");
        assert!((hit.point - Vector3::new(-1.0, 0.0, 0.0)).norm() < 1e-9);
        assert!((hit.normal - Vector3::new(-1.0, 0.0, 0.0)).norm() < 1e-9);
    }

    #[test]
    fn ray_misses_cube() {
        let shape = Shape3D::Cube {
            center: Vector3::zeros(),
            size: 2.0,
        };
        assert!(intersect_shape(&shape, Vector3::new(-5.0, 5.0, 5.0), Vector3::new(1.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn ray_hits_sphere_surface() {
        let shape = Shape3D::Sphere {
            center: Vector3::new(1.0, 0.0, 0.0),
            radius: 2.0,
        };
        let hit = intersect_shape(&shape, Vector3::new(1.0, 0.0, -10.0), Vector3::new(0.0, 0.0, 1.0))
            .expect("should hit");
        assert!((hit.point - Vector3::new(1.0, 0.0, -2.0)).norm() < 1e-9);
    }

    #[test]
    fn ray_tangent_to_sphere_grazes() {
        let shape = Shape3D::Sphere {
            center: Vector3::zeros(),
            radius: 1.0,
        };
        // Ray exactly along the silhouette (offset == radius) should still
        // produce a single (double) root, not panic or NaN.
        let hit = intersect_shape(&shape, Vector3::new(1.0, 0.0, -10.0), Vector3::new(0.0, 0.0, 1.0));
        assert!(hit.is_some());
    }

    #[test]
    fn ray_hits_cylinder_side() {
        let shape = Shape3D::Cylinder {
            base_center: Vector3::zeros(),
            axis: Vector3::new(0.0, 0.0, 1.0),
            radius: 1.0,
            height: 2.0,
        };
        let hit = intersect_shape(&shape, Vector3::new(-5.0, 0.0, 1.0), Vector3::new(1.0, 0.0, 0.0))
            .expect("should hit side");
        assert!((hit.point - Vector3::new(-1.0, 0.0, 1.0)).norm() < 1e-9);
    }

    #[test]
    fn ray_hits_cylinder_cap() {
        let shape = Shape3D::Cylinder {
            base_center: Vector3::zeros(),
            axis: Vector3::new(0.0, 0.0, 1.0),
            radius: 1.0,
            height: 2.0,
        };
        let hit = intersect_shape(&shape, Vector3::new(0.0, 0.0, 10.0), Vector3::new(0.0, 0.0, -1.0))
            .expect("should hit top cap");
        assert!((hit.point - Vector3::new(0.0, 0.0, 2.0)).norm() < 1e-9);
    }

    #[test]
    fn ray_along_cylinder_axis_hits_bottom_cap() {
        let shape = Shape3D::Cylinder {
            base_center: Vector3::zeros(),
            axis: Vector3::new(0.0, 0.0, 1.0),
            radius: 1.0,
            height: 2.0,
        };
        let hit = intersect_shape(&shape, Vector3::new(0.0, 0.0, -10.0), Vector3::new(0.0, 0.0, 1.0))
            .expect("should hit bottom cap along axis");
        assert!((hit.point - Vector3::new(0.0, 0.0, 0.0)).norm() < 1e-9);
    }

    #[test]
    fn ray_missing_cylinder_entirely() {
        let shape = Shape3D::Cylinder {
            base_center: Vector3::zeros(),
            axis: Vector3::new(0.0, 0.0, 1.0),
            radius: 1.0,
            height: 2.0,
        };
        assert!(intersect_shape(&shape, Vector3::new(-5.0, 5.0, 1.0), Vector3::new(1.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn intersect_nearest_picks_closest_shape() {
        let shapes = vec![
            Shape3D::Sphere { center: Vector3::new(5.0, 0.0, 0.0), radius: 1.0 },
            Shape3D::Sphere { center: Vector3::new(2.0, 0.0, 0.0), radius: 1.0 },
        ];
        let (idx, _hit) = intersect_nearest(
            shapes.iter().enumerate(),
            Vector3::new(-10.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        )
        .expect("should hit one of the spheres");
        assert_eq!(idx, 1);
    }
}
