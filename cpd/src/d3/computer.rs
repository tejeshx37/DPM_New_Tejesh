//! 3D simulation driver. Holds nodes, elements, and configuration; the
//! `step` method advances one Verlet integration time step. Element
//! stress and nodal force assembly run under rayon for parallelism on
//! large meshes.

use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{BoundaryCondition3D, Config3D, Element3D, Node3D};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Computer3D {
    pub nodes: Vec<Node3D>,
    pub elements: Vec<Element3D>,
    pub config: Config3D,
    pub iterations: u64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct StressStats {
    pub min_von_mises: f32,
    pub max_von_mises: f32,
    pub mean_von_mises: f32,
}

impl Computer3D {
    /// Build a solver from a mesh. `vertices` are world-space reference
    /// positions; `tets` index into `vertices`. Each node's mass is the
    /// density times the sum of incident tetrahedron volumes divided by
    /// four (each tet contributes equally to its four corner nodes).
    /// Returns `None` if any tetrahedron is degenerate.
    pub fn from_mesh(
        vertices: &[Vector3<f32>],
        tets: &[[usize; 4]],
        config: Config3D,
    ) -> Option<Self> {
        let mut elements = Vec::with_capacity(tets.len());
        let mut node_volumes = vec![0.0_f32; vertices.len()];

        for &t in tets {
            let positions = [vertices[t[0]], vertices[t[1]], vertices[t[2]], vertices[t[3]]];
            let element = Element3D::from_reference(t, positions)?;
            for &idx in &t {
                node_volumes[idx] += element.volume;
            }
            elements.push(element);
        }

        let nodes: Vec<Node3D> = vertices
            .iter()
            .zip(node_volumes.iter())
            .map(|(pos, vol)| {
                // Each tet shares its volume across 4 nodes.
                let mass = config.material.density * vol * 0.25;
                // Avoid zero-mass nodes that aren't part of any tet.
                Node3D::new(*pos, mass.max(1e-12))
            })
            .collect();

        Some(Self {
            nodes,
            elements,
            config,
            iterations: 0,
        })
    }

    /// Apply a boundary condition to a set of node indices in bulk.
    pub fn set_bc(&mut self, indices: &[usize], bc: BoundaryCondition3D) {
        for &i in indices {
            if let Some(n) = self.nodes.get_mut(i) {
                n.bc = bc.clone();
            }
        }
    }

    pub fn time(&self) -> f32 {
        self.iterations as f32 * self.config.time_delta_seconds
    }

    /// Advance one explicit Verlet step.
    pub fn step(&mut self) {
        let material = self.config.material;
        let dt = self.config.time_delta_seconds;
        let damping = material.damping;

        // 1. Element strain/stress refresh.
        let node_positions: Vec<Vector3<f32>> = self.nodes.iter().map(|n| n.position).collect();
        self.elements.par_iter_mut().for_each(|element| {
            let p = [
                node_positions[element.indices[0]],
                node_positions[element.indices[1]],
                node_positions[element.indices[2]],
                node_positions[element.indices[3]],
            ];
            element.update_strain_stress(p, &material);
        });

        // 2. Assemble internal forces onto nodes. Compute per-element
        //    contributions in parallel, then scatter sequentially (the
        //    scatter is cheap relative to stress eval and avoids the need
        //    for atomic adds).
        let per_element_forces: Vec<[Vector3<f32>; 4]> = self
            .elements
            .par_iter()
            .map(|e| e.nodal_forces())
            .collect();
        for n in self.nodes.iter_mut() {
            n.force = Vector3::zeros();
        }
        for (element, forces) in self.elements.iter().zip(per_element_forces.iter()) {
            for (idx, f) in element.indices.iter().zip(forces.iter()) {
                self.nodes[*idx].force += f;
            }
        }

        // 3. Integrate. Pinned axes overwrite velocity to zero on those
        //    components; ConstantDisplacement ramps the position
        //    directly; ConstantForce adds to nodal force first.
        let t = self.time();
        self.nodes.par_iter_mut().for_each(|n| {
            let bc = n.bc.clone();
            match bc {
                BoundaryCondition3D::Free => {
                    integrate_free(n, damping, dt);
                }
                BoundaryCondition3D::Pinned { axes } => {
                    integrate_free(n, damping, dt);
                    if axes.x {
                        n.position.x = n.initial_position.x;
                        n.velocity.x = 0.0;
                    }
                    if axes.y {
                        n.position.y = n.initial_position.y;
                        n.velocity.y = 0.0;
                    }
                    if axes.z {
                        n.position.z = n.initial_position.z;
                        n.velocity.z = 0.0;
                    }
                }
                BoundaryCondition3D::ConstantForce { force } => {
                    n.force += Vector3::new(force[0], force[1], force[2]);
                    integrate_free(n, damping, dt);
                }
                BoundaryCondition3D::ConstantDisplacement {
                    axes,
                    displacement,
                    ramp_seconds,
                } => {
                    integrate_free(n, damping, dt);
                    let target = BoundaryCondition3D::ramped_displacement(
                        Vector3::from(displacement),
                        ramp_seconds,
                        t,
                    );
                    if axes.x {
                        n.position.x = n.initial_position.x + target.x;
                        n.velocity.x = 0.0;
                    }
                    if axes.y {
                        n.position.y = n.initial_position.y + target.y;
                        n.velocity.y = 0.0;
                    }
                    if axes.z {
                        n.position.z = n.initial_position.z + target.z;
                        n.velocity.z = 0.0;
                    }
                }
            }
        });

        self.iterations += 1;
    }

    pub fn stress_stats(&self) -> StressStats {
        if self.elements.is_empty() {
            return StressStats::default();
        }
        let (mut min_vm, mut max_vm, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0_f32);
        for e in &self.elements {
            let vm = von_mises(&e.stress);
            min_vm = min_vm.min(vm);
            max_vm = max_vm.max(vm);
            sum += vm;
        }
        StressStats {
            min_von_mises: min_vm,
            max_von_mises: max_vm,
            mean_von_mises: sum / self.elements.len() as f32,
        }
    }

    /// Reset state to t=0 with current configuration; useful for re-runs
    /// after changing material parameters or BCs.
    pub fn reset(&mut self) {
        self.iterations = 0;
        for n in &mut self.nodes {
            n.position = n.initial_position;
            n.velocity = Vector3::zeros();
            n.force = Vector3::zeros();
        }
        for e in &mut self.elements {
            e.stress = Matrix3::zeros();
            e.strain = Matrix3::zeros();
        }
    }
}

fn integrate_free(node: &mut Node3D, damping: f32, dt: f32) {
    // Verlet with viscous damping: dv = ((F - c v) / m) dt.
    let dv = (node.force - node.velocity * damping) * (dt / node.mass);
    node.velocity += dv;
    node.position += node.velocity * dt;
}

fn von_mises(s: &Matrix3<f32>) -> f32 {
    // sqrt(0.5 * ((s11-s22)^2 + (s22-s33)^2 + (s33-s11)^2 + 6*(s12^2+s23^2+s13^2)))
    let d12 = s.m11 - s.m22;
    let d23 = s.m22 - s.m33;
    let d31 = s.m33 - s.m11;
    let sh = s.m12 * s.m12 + s.m23 * s.m23 + s.m13 * s.m13;
    (0.5 * (d12 * d12 + d23 * d23 + d31 * d31 + 6.0 * sh)).sqrt()
}

/// Run `n` steps. Convenience wrapper for the simulator UI.
pub fn run_steps(computer: &mut Computer3D, n: u64) {
    for _ in 0..n {
        computer.step();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_tet_mesh() -> (Vec<Vector3<f32>>, Vec<[usize; 4]>) {
        let v = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        let t = vec![[0, 1, 2, 3]];
        (v, t)
    }

    #[test]
    fn computer_from_single_tet_builds() {
        let (v, t) = single_tet_mesh();
        let c = Computer3D::from_mesh(&v, &t, Config3D::default()).unwrap();
        assert_eq!(c.elements.len(), 1);
        assert_eq!(c.nodes.len(), 4);
        assert!((c.elements[0].volume - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn rest_configuration_has_zero_stress() {
        let (v, t) = single_tet_mesh();
        let mut c = Computer3D::from_mesh(&v, &t, Config3D::default()).unwrap();
        c.step();
        assert!(c.elements[0].stress.norm() < 1e-3);
    }

    #[test]
    fn uniform_stretch_produces_positive_normal_stress() {
        let (mut v, t) = single_tet_mesh();
        // Stretch along x by 1%.
        for p in &mut v {
            p.x *= 1.01;
        }
        let mut c = Computer3D::from_mesh(&v, &t, Config3D::default()).unwrap();
        // Override initial positions back to rest to record the stretch
        // as a deformation.
        let rest = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        for (n, r) in c.nodes.iter_mut().zip(rest.iter()) {
            n.initial_position = *r;
        }
        // Rebuild element with stretched current positions but rest ref.
        c.elements[0] = Element3D::from_reference([0, 1, 2, 3], rest).unwrap();
        c.step();
        assert!(c.elements[0].stress.m11 > 0.0);
    }
}
