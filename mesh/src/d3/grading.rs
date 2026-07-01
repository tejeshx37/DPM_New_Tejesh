//! Local mesh density hints for the structured 3D generators.
//!
//! A [`DensityHint`] describes a user-drawn refinement region (circle or
//! rectangle collapsed to an isotropic radius, see the meshing UI) that
//! should locally bias tetrahedra/vertex spacing without changing the
//! lattice topology of the structured generators in `cuboid`/`sphere`/
//! `cylinder`. This is a deliberate v1 approximation ("graded structured
//! grid") rather than true local adaptive refinement: grid lines along an
//! axis bunch up near a hint's coordinate on that axis, which densifies the
//! whole grid slab through the shape rather than an isolated 3D
//! neighborhood. Good enough to demonstrate coarsen/refine control; true
//! adaptive tetrahedral insertion is out of scope (see plan doc).

use nalgebra::Vector3;

/// A local density request in world space.
#[derive(Debug, Clone, Copy)]
pub struct DensityHint {
    pub center_world: Vector3<f64>,
    pub radius_world: f64,
    /// >1.0 refines (denser), <1.0 coarsens (sparser). Neutral at 1.0.
    pub multiplier: f32,
    /// Fraction of `radius_world` used as the Gaussian bump's blend margin;
    /// larger falloff spreads the density change over a wider region.
    pub falloff: f32,
}

/// Compute `n + 1` graded 1D positions in `[min, max]` such that spacing is
/// finer near each bump's `center` and blends smoothly back to uniform
/// spacing elsewhere.
///
/// `bumps` is `(center, sigma, multiplier)`: `center` and `sigma` are in the
/// same coordinate space as `min`/`max`, and `multiplier` follows the same
/// >1 = denser / <1 = sparser convention as [`DensityHint::multiplier`].
///
/// Uses the standard "equal-integral" graded-node-placement technique:
/// build a strictly positive density field `rho(t)`, integrate it, then
/// place node `i` where the cumulative integral equals `i / n` of the
/// total. This guarantees monotonically increasing (non-inverting)
/// positions as long as `rho > 0` everywhere, regardless of how extreme the
/// requested multipliers are.
pub fn graded_axis_positions(
    min: f64,
    max: f64,
    n: usize,
    bumps: &[(f64, f64, f32)],
) -> Vec<f64> {
    if bumps.is_empty() || max <= min {
        return (0..=n).map(|i| min + (max - min) * (i as f64 / n.max(1) as f64)).collect();
    }

    const SAMPLES: usize = 256;
    const MIN_RHO: f64 = 0.05;

    let rho = |t: f64| -> f64 {
        let mut r = 1.0_f64;
        for &(center, sigma, multiplier) in bumps {
            let sigma = sigma.max(1e-9);
            let d = (t - center) / sigma;
            let bump = (-0.5 * d * d).exp();
            r += (multiplier as f64 - 1.0) * bump;
        }
        r.max(MIN_RHO)
    };

    // Cumulative trapezoidal integral of rho over [min, max].
    let dt = (max - min) / SAMPLES as f64;
    let mut cum = vec![0.0_f64; SAMPLES + 1];
    for i in 1..=SAMPLES {
        let t0 = min + (i - 1) as f64 * dt;
        let t1 = min + i as f64 * dt;
        cum[i] = cum[i - 1] + 0.5 * (rho(t0) + rho(t1)) * dt;
    }
    let total = cum[SAMPLES];

    let mut positions = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let target = total * (i as f64 / n.max(1) as f64);
        let idx = cum.partition_point(|&c| c < target).min(SAMPLES);
        let t = if idx == 0 {
            min
        } else {
            let c0 = cum[idx - 1];
            let c1 = cum[idx];
            let t0 = min + (idx - 1) as f64 * dt;
            let t1 = min + idx as f64 * dt;
            if (c1 - c0).abs() < 1e-15 {
                t1
            } else {
                t0 + (target - c0) / (c1 - c0) * (t1 - t0)
            }
        };
        positions.push(t);
    }
    // Guard against float round-off producing a non-monotonic tail.
    positions[0] = min;
    positions[n] = max;
    for i in 1..=n {
        if positions[i] < positions[i - 1] {
            positions[i] = positions[i - 1];
        }
    }
    positions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_bumps_is_uniform() {
        let p = graded_axis_positions(0.0, 10.0, 5, &[]);
        assert_eq!(p, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn neutral_bump_is_effectively_uniform() {
        let p = graded_axis_positions(0.0, 10.0, 10, &[(5.0, 1.0, 1.0)]);
        for w in p.windows(2) {
            assert!((w[1] - w[0] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn refine_bump_densifies_near_center() {
        let p = graded_axis_positions(0.0, 10.0, 20, &[(5.0, 1.0, 4.0)]);
        // Spacing near the center should be tighter than spacing at the ends.
        let mid = p.len() / 2;
        let center_spacing = p[mid + 1] - p[mid];
        let end_spacing = p[1] - p[0];
        assert!(center_spacing < end_spacing);
    }

    #[test]
    fn coarsen_bump_widens_near_center() {
        let p = graded_axis_positions(0.0, 10.0, 20, &[(5.0, 1.0, 0.2)]);
        let mid = p.len() / 2;
        let center_spacing = p[mid + 1] - p[mid];
        let end_spacing = p[1] - p[0];
        assert!(center_spacing > end_spacing);
    }

    #[test]
    fn always_monotonic_even_with_extreme_multipliers() {
        let p = graded_axis_positions(
            0.0,
            10.0,
            30,
            &[(2.0, 0.3, 5.0), (8.0, 0.3, 0.1), (5.0, 0.5, 5.0)],
        );
        for w in p.windows(2) {
            assert!(w[1] >= w[0]);
        }
        assert_eq!(*p.first().unwrap(), 0.0);
        assert_eq!(*p.last().unwrap(), 10.0);
    }
}
