use super::{
    boundary_average::ForceAndDisplacement, boundary_condition::BoundaryCondition, config::Config,
    element::Element, node, BoundaryAverage, BoundaryInfo, ExportData, Matrix2, Node,
    TimeSeriesValue, TimeStampedValue,
};
use cgal::{triangulation, BoundaryId};
use fxhash::{FxHashMap, FxHashSet};
use nalgebra::Vector2;
use rayon::prelude::*;
use std::{
    cmp::Ordering,
    fmt::Debug,
    time::{Duration, Instant},
};

#[derive(Debug)]
pub struct Unconfigured;

#[derive(Debug)]
pub struct InProgress {
    steps: u128,
    iterations: u128,
    runtime: Option<Duration>,
    config: Box<Config>,
}

#[derive(Debug)]
pub struct Done {
    steps: u128,
    iterations: u128,
    runtime: Option<Duration>,
    config: Box<Config>,
}

#[sealed::sealed]
pub trait State {
    fn iterations(&self) -> u128;
    fn runtime(&self) -> Option<Duration>;
    fn time_elapsed(&self) -> f32;
    fn config(&self) -> Option<&Config>;
}

#[sealed::sealed]
impl State for Unconfigured {
    fn iterations(&self) -> u128 {
        0
    }

    fn runtime(&self) -> Option<Duration> {
        Some(Duration::ZERO)
    }

    fn time_elapsed(&self) -> f32 {
        0.0
    }

    fn config(&self) -> Option<&Config> {
        None
    }
}

#[sealed::sealed]
impl State for InProgress {
    fn iterations(&self) -> u128 {
        self.iterations
    }

    fn runtime(&self) -> Option<Duration> {
        self.runtime
    }

    fn time_elapsed(&self) -> f32 {
        self.config.time_delta().as_secs_f32() * self.iterations as f32
    }

    fn config(&self) -> Option<&Config> {
        Some(&self.config)
    }
}

#[sealed::sealed]
impl State for Done {
    fn iterations(&self) -> u128 {
        self.iterations
    }

    fn runtime(&self) -> Option<Duration> {
        self.runtime
    }

    fn time_elapsed(&self) -> f32 {
        self.config.duration().as_secs_f32()
    }

    fn config(&self) -> Option<&Config> {
        Some(&self.config)
    }
}

#[derive(Debug)]
pub struct Computer<S: State> {
    nodes: Box<[Node]>,
    elements: Box<[Element]>,
    state: S,
    boundary_infos: FxHashMap<BoundaryId, BoundaryInfo>,
    data_recorded_boundary: Option<(FxHashSet<usize>, BoundaryAverage)>,
}

impl<S: State> Computer<S> {
    fn min_max_stress(&self) -> (Matrix2<f32>, Matrix2<f32>) {
        let iter = || self.elements.iter().map(Element::stress).copied();
        iter().zip(iter()).par_bridge().reduce(
            || (Matrix2::from_element(f32::MAX), Matrix2::zeros()),
            |a, b| (a.0.zip_map(&b.0, f32::min), a.1.zip_map(&b.1, f32::max)),
        )
    }

    pub fn iterations(&self) -> u128 {
        self.state.iterations()
    }

    pub fn runtime(&self) -> Option<Duration> {
        self.state.runtime()
    }

    pub fn record_stress_data(&mut self, index: usize) {
        let time_stamp = self.state.time_elapsed();
        let element = &mut self.elements[index];
        element.stress_time_series = TimeSeriesValue::Series(vec![TimeStampedValue {
            time_stamp,
            value: *element.stress(),
        }]);
    }

    pub fn stop_recording_stress_data(&mut self, index: usize) {
        let element = &mut self.elements[index];
        element.stress_time_series = TimeSeriesValue::Single(*element.stress());
    }

    pub fn record_vertex_position(&mut self, index: usize) {
        let time_stamp = self.state.time_elapsed();
        let node = &mut self.nodes[index];
        node.position_time_series = TimeSeriesValue::Series(vec![TimeStampedValue {
            time_stamp,
            value: *node.position(),
        }]);
    }

    pub fn stop_recording_vertex_position(&mut self, index: usize) {
        self.nodes[index].position_time_series =
            TimeSeriesValue::Single(*self.nodes[index].position());
    }

    pub fn record_boundary_data(&mut self, id: BoundaryId) {
        let info = &self.boundary_infos[&id];
        let data = match info.boundary_condition {
            BoundaryCondition::Free => BoundaryAverage::ForceAndDisplacement(vec![]),
            BoundaryCondition::Force(_) => BoundaryAverage::Displacement(vec![]),
            BoundaryCondition::Displacement(_) => BoundaryAverage::Force(vec![]),
        };
        self.data_recorded_boundary = Some((info.node_indices.clone(), data));
    }

    pub fn stop_recording_boundary_data(&mut self) {
        self.data_recorded_boundary = None;
    }

    fn reset_boundary_data(&mut self) {
        if let Some((_, data)) = &mut self.data_recorded_boundary {
            data.reset();
        }
    }

    pub fn export_data(&self) -> ExportData {
        let (min_stress, max_stress) = self.min_max_stress();
        ExportData {
            nodes: self.nodes.clone(),
            elements: self.elements.clone(),
            boundary_infos: self.boundary_infos.clone(),
            boundary_average_data: self
                .data_recorded_boundary
                .as_ref()
                .map(|(_, data)| data.clone()),
            config: self.state.config().copied(),
            iterations: self.iterations(),
            min_stress,
            max_stress,
        }
    }
}

fn apply_config(
    mut nodes: Box<[Node]>,
    initial_density: f32,
    config: Config,
    reset: bool,
) -> (Box<[Node]>, InProgress) {
    let scale = config.material_props().bulk_props().density() / initial_density;
    nodes.par_iter_mut().for_each(|node| {
        node.scale_mass(scale);
        if reset {
            node.reset();
        }
    });
    (
        nodes,
        InProgress {
            steps: config.duration().as_nanos() / config.time_delta().as_nanos(),
            iterations: 0,
            runtime: Some(Duration::ZERO),
            config: Box::new(config),
        },
    )
}

impl Computer<Unconfigured> {
    pub fn configure(self, config: Config) -> Computer<InProgress> {
        let (nodes, state) = apply_config(self.nodes, 1.0, config, false);
        Computer {
            nodes,
            state,
            elements: self.elements,
            boundary_infos: self.boundary_infos,
            data_recorded_boundary: self.data_recorded_boundary,
        }
    }
}

fn outer_product_sum(y_ba: &Vector2<f32>, y_ca: &Vector2<f32>) -> Matrix2<f32> {
    Matrix2::from(y_ba * Vector2::x().transpose() + y_ca * Vector2::y().transpose())
}

fn delaunay_deformation_tensor(
    d_ba: &Vector2<f32>,
    d_ca: &Vector2<f32>,
    r_ba: &Vector2<f32>,
    r_ca: &Vector2<f32>,
) -> Matrix2<f32> {
    outer_product_sum(d_ba, d_ca)
        * outer_product_sum(r_ba, r_ca)
            .try_inverse()
            .expect("Malformed delaunay configuration")
}

fn green_lagrange_strain_tensor(f: &Matrix2<f32>) -> Matrix2<f32> {
    let c: Matrix2<f32> = f.transpose() * f;
    (c - Matrix2::identity()) / 2.0
}

fn strain_energy(stress: &Matrix2<f32>, strain: &Matrix2<f32>) -> f32 {
    (stress.m11 * strain.m11 + stress.m22 * strain.m22 + 4.0 * stress.m12 * strain.m21) / 2.0
}

fn update_element(time_stamp: f32, element: &mut Element, config: &Config, nodes: [&Node; 3]) {
    element.strain = {
        let r_ba: Vector2<f32> = nodes[1].initial_position() - nodes[0].initial_position();
        let r_ca: Vector2<f32> = nodes[2].initial_position() - nodes[0].initial_position();

        let d_ba: Vector2<f32> = nodes[1].position() - nodes[0].position();
        let d_ca: Vector2<f32> = nodes[2].position() - nodes[0].position();

        let f: Matrix2<f32> = delaunay_deformation_tensor(&d_ba, &d_ca, &r_ba, &r_ca);

        green_lagrange_strain_tensor(&f)
    };
    element.stress_time_series.set_or_push(
        time_stamp,
        config.material_props().eval_stress(&element.strain),
    );
    element.strain_energy = strain_energy(element.stress(), &element.strain);
    element.is_broken = config
        .material_props()
        .bulk_props()
        .failure_criteria()
        .satisfies(element.strain_energy, element.stress());
}

fn force(config: &Config, nodes: [&Node; 3], element: &Element) -> Vector2<f32> {
    let (de, area) = {
        let r_ba: Vector2<f32> = nodes[1].initial_position() - nodes[0].initial_position();
        let r_ca: Vector2<f32> = nodes[2].initial_position() - nodes[0].initial_position();
        let r_bc: Vector2<f32> = r_ba - r_ca;

        let area_vector_component = r_ba.x * r_ca.y - r_ca.x * r_ba.y;

        let d_ba: Vector2<f32> = nodes[1].position() - nodes[0].position();
        let d_ca: Vector2<f32> = nodes[2].position() - nodes[0].position();

        let p = d_ca.x * r_ba.y - r_ca.y * d_ba.x;
        let q = r_ca.x * d_ba.y - d_ca.y * r_ba.x;
        let r = r_ca.x * d_ba.x - d_ca.x * r_ba.x;
        let s = d_ca.y * r_ba.y - r_ca.y * d_ba.y;

        let a = area_vector_component.powi(2);
        macro_rules! de {
            ($f1:expr, $f2:expr) => {{
                let xy = ($f1 * r_bc.x - $f2 * r_bc.y) / 2.0;
                Matrix2::new($f1 * -r_bc.y, xy, xy, $f2 * r_bc.x).map(|v| v / a)
            }};
        }
        let de: Vector2<Matrix2<f32>> = Vector2::new(de!(p, r), de!(s, q));
        (de, area_vector_component.abs() / 2.0)
    };

    let dz: Vector2<Matrix2<f32>> = de.map(|strain| config.material_props().eval_stress(&strain));

    let force = |dz: Matrix2<f32>, de: Matrix2<f32>| {
        -(dz.m11 * element.strain.m11
            + de.m11 * element.stress().m11
            + dz.m22 * element.strain.m22
            + de.m22 * element.stress().m22
            + (dz.m12 * element.strain.m12 + de.m12 * element.stress().m12) * 4.0)
    };

    dz.zip_map(&de, force) * area / 2.0
}

pub enum AdvanceResult {
    InProgress(Computer<InProgress>),
    Done(Computer<Done>),
}

impl Computer<InProgress> {
    pub fn reconfigure(mut self, config: Config) -> Computer<InProgress> {
        let (nodes, state) = apply_config(
            self.nodes,
            *self.state.config.material_props().bulk_props().density(),
            config,
            self.state.iterations > 0,
        );
        self.elements.par_iter_mut().for_each(Element::reset);
        Computer {
            nodes,
            state,
            elements: self.elements,
            boundary_infos: self.boundary_infos,
            data_recorded_boundary: self.data_recorded_boundary,
        }
    }

    pub fn progress(&self) -> f32 {
        (self.iterations() as f32) / (self.total_iterations() as f32)
    }

    pub fn total_iterations(&self) -> u128 {
        self.state.steps
    }

    fn update_boundary_data(&mut self, time_stamp: f32) {
        let Some((node_indices, data)) = &mut self.data_recorded_boundary else {
            return;
        };
        let node_indices = &*node_indices;
        macro_rules! sum {
            ($map_node:expr) => {
                node_indices
                    .par_iter()
                    .map(|index| &self.nodes[*index])
                    .map($map_node)
                    .sum()
            };
        }
        let boundary_nodes = node_indices.len() as f32;
        match data {
            BoundaryAverage::Force(series) => {
                let sum: Vector2<f32> = sum!(|node| node.force());
                series.push(TimeStampedValue {
                    time_stamp,
                    value: sum / boundary_nodes,
                });
            }
            BoundaryAverage::Displacement(series) => {
                let sum: Vector2<f32> = sum!(|node| node.position());
                series.push(TimeStampedValue {
                    time_stamp,
                    value: sum / boundary_nodes,
                });
            }
            BoundaryAverage::ForceAndDisplacement(series) => {
                let mut sum: ForceAndDisplacement = sum!(|node| ForceAndDisplacement {
                    force: node.force,
                    displacement: *node.position()
                });
                sum.force /= boundary_nodes;
                sum.displacement /= boundary_nodes;
                series.push(TimeStampedValue {
                    time_stamp,
                    value: sum,
                });
            }
        }
    }

    pub fn advance(mut self) -> AdvanceResult {
        let now = Instant::now();
        let config = &self.state.config;
        let time_stamp = self.state.time_elapsed();
        self.elements
            .par_iter_mut()
            .filter(|element| !element.is_broken)
            .for_each(|element| {
                let node = |i| &self.nodes[element.indices[i]];
                update_element(time_stamp, element, config, [node(0), node(1), node(2)])
            });
        let forces: Vec<Vector2<f32>> = self
            .nodes
            .par_iter()
            .enumerate()
            .map(|(index, node)| {
                node.incident_faces()
                    .iter()
                    .copied()
                    .filter(|index| !self.elements[*index].is_broken)
                    .map(|element_index| {
                        let element = &self.elements[element_index];
                        let indices = element.indices();
                        let pos_at_indices = indices
                            .iter()
                            .enumerate()
                            .find_map(|(idx, i)| (*i == index).then_some(idx))
                            .expect("Node index should match one of the indices");
                        let node_at = |i| &self.nodes[element.indices[i]];
                        let nodes = [
                            node,
                            node_at((pos_at_indices + 1) % 3),
                            node_at((pos_at_indices + 2) % 3),
                        ];
                        force(config, nodes, element)
                    })
                    .sum()
            })
            .collect();
        // Position update has to be done at the end otherwise it
        // will interfere with force calculation
        forces
            .into_par_iter()
            .zip(self.nodes.par_iter_mut())
            .for_each(|(force, node)| {
                node.apply_force_and_bc(
                    force,
                    self.state.iterations,
                    *config.material_props().bulk_props().damping(),
                    config.time_delta().as_secs_f32(),
                )
            });
        self.update_boundary_data(time_stamp);
        self.state.iterations += 1;
        if let Some(runtime) = &mut self.state.runtime {
            *runtime += now.elapsed();
        }
        if self.state.iterations >= self.state.steps {
            AdvanceResult::Done(Computer::<Done> {
                nodes: self.nodes,
                elements: self.elements,
                state: Done {
                    steps: self.state.steps,
                    iterations: self.state.iterations,
                    runtime: self.state.runtime,
                    config: self.state.config,
                },
                boundary_infos: self.boundary_infos,
                data_recorded_boundary: self.data_recorded_boundary,
            })
        } else {
            AdvanceResult::InProgress(self)
        }
    }

    pub fn set_duration(&mut self, duration: Duration) -> Result<(), String> {
        let config = &mut self.state.config;
        let completed_duration = (self.state.iterations as f32) * config.time_delta().as_secs_f32();
        let completed_duration = Duration::from_secs_f32(completed_duration);
        if duration < completed_duration {
            Err(String::from(
                "Cannot reduce duration to a value which is less than duration elapsed",
            ))
        } else {
            config.set_duration(duration);
            self.state.steps = config.duration().as_nanos() / config.time_delta().as_nanos();
            Ok(())
        }
    }

    pub fn reset(&mut self) {
        self.nodes.par_iter_mut().for_each(Node::reset);
        self.elements.par_iter_mut().for_each(Element::reset);
        self.state.iterations = 0;
        self.state.runtime = Some(Duration::ZERO);
        self.reset_boundary_data();
    }
}

pub type SetDurationResult = Result<Computer<InProgress>, (Computer<Done>, String)>;

impl Computer<Done> {
    pub fn reconfigure(mut self, config: Config) -> Computer<InProgress> {
        let (nodes, state) = apply_config(
            self.nodes,
            *self.state.config.material_props().bulk_props().density(),
            config,
            true,
        );
        self.elements.par_iter_mut().for_each(Element::reset);
        Computer {
            nodes,
            state,
            elements: self.elements,
            boundary_infos: self.boundary_infos,
            data_recorded_boundary: self.data_recorded_boundary,
        }
    }

    pub fn total_iterations(&self) -> u128 {
        self.state.steps
    }

    pub fn set_duration(mut self, duration: Duration) -> SetDurationResult {
        let config = &mut self.state.config;
        let completed_duration = (self.state.iterations as f32) * config.time_delta().as_secs_f32();
        let completed_duration = Duration::from_secs_f32(completed_duration);
        if duration < completed_duration {
            Err((
                self,
                String::from(
                    "Cannot reduce duration to a value which is less than duration elapsed",
                ),
            ))
        } else {
            config.set_duration(duration);
            Ok(Computer {
                nodes: self.nodes,
                elements: self.elements,
                state: InProgress {
                    steps: config.duration().as_nanos() / config.time_delta().as_nanos(),
                    iterations: self.state.iterations,
                    runtime: self.state.runtime,
                    config: self.state.config,
                },
                boundary_infos: self.boundary_infos,
                data_recorded_boundary: self.data_recorded_boundary,
            })
        }
    }

    pub fn reset(mut self) -> Computer<InProgress> {
        self.nodes.par_iter_mut().for_each(Node::reset);
        self.elements.par_iter_mut().for_each(Element::reset);
        self.reset_boundary_data();
        Computer {
            nodes: self.nodes,
            elements: self.elements,
            state: InProgress {
                steps: self.state.steps,
                iterations: 0,
                runtime: Some(Duration::ZERO),
                config: self.state.config,
            },
            boundary_infos: self.boundary_infos,
            data_recorded_boundary: self.data_recorded_boundary,
        }
    }
}

pub fn unconfigured(
    triangulation_data: &triangulation::Data,
    boundary_point_map: &FxHashMap<BoundaryId, FxHashSet<usize>>,
    boundary_conditions: &FxHashMap<BoundaryId, BoundaryCondition>,
    point_boundary_conditions: &FxHashMap<usize, BoundaryCondition>,
) -> Computer<Unconfigured> {
    Computer::<Unconfigured> {
        nodes: node::nodes(triangulation_data, point_boundary_conditions, 1.0),
        elements: triangulation_data
            .faces()
            .par_iter()
            .map(|face| face.0)
            .map(Element::new)
            .collect(),
        state: Unconfigured,
        boundary_infos: boundary_point_map
            .iter()
            .map(|(id, indices)| {
                let info = BoundaryInfo {
                    boundary_condition: boundary_conditions[id].clone(),
                    node_indices: indices.clone(),
                };
                (*id, info)
            })
            .collect(),
        data_recorded_boundary: None,
    }
}

pub enum ImportResult {
    Unconfigured(Computer<Unconfigured>),
    InProgress((Computer<InProgress>, Config)),
    Done((Computer<Done>, Config)),
    Err(String),
}

pub fn from_export_data(export_data: ExportData) -> ImportResult {
    let Some(config) = export_data.config else {
        return ImportResult::Unconfigured(Computer {
            nodes: export_data.nodes,
            elements: export_data.elements,
            state: Unconfigured,
            boundary_infos: export_data.boundary_infos,
            data_recorded_boundary: None,
        });
    };
    let steps = config.duration().as_nanos() / config.time_delta().as_nanos();
    match export_data.iterations.cmp(&steps) {
        Ordering::Less => ImportResult::InProgress((
            Computer {
                nodes: export_data.nodes,
                elements: export_data.elements,
                state: InProgress {
                    steps,
                    iterations: export_data.iterations,
                    runtime: (export_data.iterations == 0).then_some(Duration::ZERO),
                    config: Box::new(config),
                },
                boundary_infos: export_data.boundary_infos,
                data_recorded_boundary: None,
            },
            config,
        )),
        Ordering::Equal => ImportResult::Done((
            Computer {
                nodes: export_data.nodes,
                elements: export_data.elements,
                state: Done {
                    steps,
                    iterations: steps,
                    runtime: None,
                    config: Box::new(config),
                },
                boundary_infos: export_data.boundary_infos,
                data_recorded_boundary: None,
            },
            config,
        )),
        Ordering::Greater => ImportResult::Err(String::from(
            "Exported data has more iterations than allowed",
        )),
    }
}
