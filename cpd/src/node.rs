use crate::{
    boundary_condition::{BoundaryCondition, Displacement},
    time_series_value::TimeStampedValue,
    TimeSeriesValue,
};
use cgal::triangulation::{Data as TriangulationData, Vertex};
use fxhash::FxHashMap;
use nalgebra::Vector2;
use rand::prelude::*;
use rand_distr::UnitDisc;
use rayon::prelude::*;
use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
};

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub enum Node {
    Interior(NodeData),
    OnBoundary(NodeData, BoundaryCondition),
}

impl Deref for Node {
    type Target = NodeData;

    fn deref(&self) -> &Self::Target {
        match self {
            Node::Interior(data) | Node::OnBoundary(data, _) => data,
        }
    }
}

impl DerefMut for Node {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Node::Interior(data) | Node::OnBoundary(data, _) => data,
        }
    }
}

impl Node {
    pub fn position_time_series(&self) -> &TimeSeriesValue<Vector2<f32>> {
        &self.position_time_series
    }

    pub fn force(&self) -> &Vector2<f32> {
        &self.force
    }

    pub fn velocity(&self) -> &Vector2<f32> {
        &self.velocity
    }

    pub(crate) fn reset(&mut self) {
        self.deref_mut().reset();
    }

    pub(crate) fn apply_force_and_bc(
        &mut self,
        force: Vector2<f32>,
        iterations: u128,
        damping_constant: f32,
        time_delta: f32,
    ) {
        self.force = force;
        macro_rules! velocity_delta {
            ($mass:expr, $force:expr, $velocity:expr) => {
                (($force - $velocity * damping_constant) * time_delta) / $mass
            };
        }
        let time = iterations as f32 * time_delta;
        macro_rules! update_pos_and_velocity {
            ($node:expr) => {{
                $node.velocity =
                    $node.velocity + velocity_delta!($node.mass(), $node.force, $node.velocity);
                let position: Vector2<f32> = $node.position() + $node.velocity * time_delta;
                $node.position_time_series.set_or_push(time, position);
            }};
        }
        match self {
            Node::OnBoundary(node, BoundaryCondition::Displacement(displacement)) => {
                let mut position: Vector2<f32> = *node.position();
                macro_rules! update_pos_and_velocity_comp {
                    ( $comp:ident ) => {{
                        node.velocity.$comp = node.velocity.$comp
                            + velocity_delta!(node.mass(), node.force.$comp, node.velocity.$comp);
                        position.$comp += node.velocity.$comp * time_delta;
                    }};
                }
                match &displacement {
                    Displacement::X(f) => {
                        if let Some(x) = f.of(time) {
                            position.x = node.initial_position.x + x;
                        }
                        update_pos_and_velocity_comp!(y);
                    }
                    Displacement::Y(f) => {
                        if let Some(y) = f.of(time) {
                            position.y = node.initial_position.y + y;
                        }
                        update_pos_and_velocity_comp!(x);
                    }
                    Displacement::XY(vf) => {
                        if let Some(x) = vf.x.of(time) {
                            position.x = node.initial_position.x + x;
                        }
                        if let Some(y) = vf.y.of(time) {
                            position.y = node.initial_position.y + y;
                        }
                    }
                }
                node.position_time_series.set_or_push(time, position);
            }
            Node::OnBoundary(node, BoundaryCondition::Force(external_force)) => {
                node.force.zip_apply(external_force, |fv, f| {
                    if let Some(v) = f.of(time) {
                        *fv += v;
                    }
                });
                update_pos_and_velocity!(self);
            }
            Node::OnBoundary(_, BoundaryCondition::Free) | Node::Interior(_) => {
                update_pos_and_velocity!(self);
            }
        };
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct NodeData {
    pub(crate) position_time_series: TimeSeriesValue<Vector2<f32>>,
    pub(crate) force: Vector2<f32>,
    pub(crate) velocity: Vector2<f32>,
    initial_velocity: Vector2<f32>,
    initial_position: Vector2<f32>,
    incident_faces: Box<[usize]>,
    mass: f32,
}

impl NodeData {
    fn new(
        position: Vector2<f32>,
        incident_faces: Box<[usize]>,
        mass: f32,
        initial_velocity: Vector2<f32>,
    ) -> Self {
        Self {
            position_time_series: TimeSeriesValue::Single(position),
            force: Vector2::zeros(),
            velocity: initial_velocity,
            initial_velocity,
            initial_position: position,
            incident_faces,
            mass,
        }
    }

    pub fn position(&self) -> &Vector2<f32> {
        self.position_time_series.latest()
    }

    pub(crate) fn initial_position(&self) -> &Vector2<f32> {
        &self.initial_position
    }

    pub(crate) fn mass(&self) -> f32 {
        self.mass
    }

    pub(crate) fn scale_mass(&mut self, scale: f32) {
        self.mass *= scale;
    }

    pub(crate) fn incident_faces(&self) -> &[usize] {
        &self.incident_faces
    }

    pub(crate) fn reset(&mut self) {
        self.force.x = 0.0;
        self.force.y = 0.0;
        self.velocity = self.initial_velocity;
        match &mut self.position_time_series {
            TimeSeriesValue::Single(v) => *v = self.initial_position,
            TimeSeriesValue::Series(series) => {
                series.clear();
                series.push(TimeStampedValue {
                    time_stamp: 0.0,
                    value: self.initial_position,
                });
            }
        }
    }
}

fn face_area(index: usize, triangulation_data: &TriangulationData) -> f32 {
    let indices = triangulation_data.faces()[index].0;
    let vertices: &[Vertex] = triangulation_data.vertices();
    let point = |index: usize| vertices[indices[index]].point();
    let pq: Vector2<f32> = point(1) - point(0);
    let pr: Vector2<f32> = point(2) - point(0);
    (pq.x * pr.y - pq.y * pr.x).abs()
}

fn voronoi_tile_area(vertex: &Vertex, triangulation_data: &TriangulationData) -> f32 {
    vertex
        .incident_faces()
        .iter()
        .copied()
        .map(|index| face_area(index, triangulation_data))
        .sum()
}

fn random_velocity() -> Vector2<f32> {
    let mut rng = rand::thread_rng();
    UnitDisc.map(Vector2::from).sample(&mut rng) * 1e-4
}

pub fn nodes(
    triangulation_data: &TriangulationData,
    boundary_conditions: &FxHashMap<usize, BoundaryCondition>,
    density: f32,
) -> Box<[Node]> {
    triangulation_data
        .vertices()
        .par_iter()
        .enumerate()
        .map(|(i, vertex)| {
            let node_data = NodeData::new(
                *vertex.point(),
                vertex.incident_faces().clone(),
                density * voronoi_tile_area(vertex, triangulation_data) / 3.0,
                random_velocity(),
            );
            match boundary_conditions.get(&i).cloned() {
                Some(bc) => Node::OnBoundary(node_data, bc),
                None => Node::Interior(node_data),
            }
        })
        .collect()
}
