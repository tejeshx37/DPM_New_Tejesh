mod export;

use super::{
    boundary_conditions::BoundaryConditions,
    project::data::{Data, WithCpdExportData, WithMesh},
    state_channel, PolygonData, RefreshToken,
};
use cgal::BoundaryId;
use cpd::{
    boundary_condition::BoundaryCondition, computer, config::Config as CpdConfig, ExportData,
};
use derive_getters::Getters;
use egui::ahash::HashSetExt;
use fxhash::{FxHashMap, FxHashSet};
use mesh::Mesh;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    ops::Deref,
    path::PathBuf,
    sync::{
        mpsc::{self, TryRecvError},
        Arc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};
use typed_builder::TypedBuilder;

#[derive(Debug, Clone, Copy, Deserialize)]
struct EngineEnvConfig {
    time_step_factor: f64,
}

impl Default for EngineEnvConfig {
    fn default() -> Self {
        Self {
            time_step_factor: 0.96,
        }
    }
}

lazy_static::lazy_static! {
    static ref ENGINE_ENV_CONFIG: EngineEnvConfig = envy::prefixed("DPM_SIM_")
        .from_env::<EngineEnvConfig>()
        .unwrap_or_default();
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum State {
    #[default]
    Unconfigured,
    Running,
    Paused,
    Finished,
}

#[derive(Debug, Clone, Getters)]
pub struct Frame {
    data: ExportData,
    progress: f32,
    runtime: Option<Duration>,
    iterations: u128,
    total_iterations: Option<u128>,
}

#[derive(Debug, Clone, Getters, TypedBuilder)]
pub struct Config {
    cpd_config: CpdConfig,
    refresh_period: u128,
    export_config: Option<ExportConfig>,
}

pub use export::ExportConfig;

#[derive(Debug, Clone, Getters)]
pub struct PlotItems {
    free: FxHashSet<BoundaryId>,
    force: FxHashSet<BoundaryId>,
    displacement: FxHashSet<BoundaryId>,
}

impl PlotItems {
    fn from(boundary_conditions: &BoundaryConditions) -> Self {
        boundary_conditions.iter().fold(
            PlotItems {
                free: FxHashSet::with_capacity(boundary_conditions.len()),
                force: FxHashSet::with_capacity(boundary_conditions.len()),
                displacement: FxHashSet::with_capacity(boundary_conditions.len()),
            },
            |mut items, (id, condition)| {
                match condition {
                    BoundaryCondition::Free => {
                        items.free.insert(*id);
                    }
                    BoundaryCondition::Force(_) => {
                        items.force.insert(*id);
                    }
                    BoundaryCondition::Displacement(_) => {
                        items.displacement.insert(*id);
                    }
                }
                items
            },
        )
    }

    pub fn contains_id(&self, id: &BoundaryId) -> bool {
        self.free.contains(id) || self.force.contains(id) || self.displacement.contains(id)
    }
}

#[derive(Debug)]
pub struct Engine<T: RefreshToken> {
    project_data: Data<WithMesh>,
    plot_items: PlotItems,
    worker: Worker<T>,
    refresh_token_set: bool,
}

pub fn optimal_time_delta(density: f64, elasticity_modulus: f64, mesh: &Mesh) -> Option<f64> {
    if !density.is_finite() || density.abs() == 0.0 {
        return None;
    }
    if !elasticity_modulus.is_finite() || elasticity_modulus.abs() == 0.0 {
        return None;
    }
    let delta = ENGINE_ENV_CONFIG.time_step_factor * mesh.smallest_side_length()
        / (elasticity_modulus / density).sqrt();
    Some(delta)
}

macro_rules! frame {
    ( $computer:expr ) => {
        match &$computer {
            Computer::Unconfigured(c) => Frame {
                data: c.export_data(),
                progress: 0.0,
                runtime: c.runtime(),
                iterations: 0,
                total_iterations: None,
            },
            Computer::InProgress(c) => Frame {
                data: c.export_data(),
                progress: c.progress(),
                runtime: c.runtime(),
                iterations: c.iterations(),
                total_iterations: Some(c.total_iterations()),
            },
            Computer::Done(c) => Frame {
                data: c.export_data(),
                progress: 100.0,
                runtime: c.runtime(),
                iterations: c.iterations(),
                total_iterations: Some(c.total_iterations()),
            },
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CpdConfigDiffResult {
    NoDiff,
    NeedsReconfiguration,
    UpdateDuration,
}

#[derive(Debug, TypedBuilder)]
pub struct Senders {
    config_sender: state_channel::Sender<Box<Config>>,
    state_sender: state_channel::Sender<State>,
    frame_sender: state_channel::Sender<Frame>,
    error_sender: state_channel::Sender<String>,
}

impl<T: RefreshToken> Engine<T> {
    pub fn new(project_data: Data<WithMesh>, senders: Senders) -> Self {
        let computer = Self::new_computer(
            &project_data.state().mesh,
            &project_data.state().boundary_conditions,
        );
        let state = project_data.state();
        Self {
            plot_items: PlotItems::from(&state.boundary_conditions),
            worker: Worker::new(
                computer,
                None,
                state.mesh.clone(),
                State::Unconfigured,
                senders,
            ),
            project_data,
            refresh_token_set: false,
        }
    }

    fn new_computer(mesh: &Mesh, boundary_conditions: &BoundaryConditions) -> Computer {
        let point_boundary_conditions: FxHashMap<usize, BoundaryCondition> = mesh
            .point_id_map()
            .par_iter()
            .map(|(vertex_index, boundary_ids)| {
                let condition = boundary_ids
                    .iter()
                    .map(|id| match &boundary_conditions[id] {
                        BoundaryCondition::Free => BoundaryCondition::Free,
                        BoundaryCondition::Force(force) => {
                            // Distribute force between all particles on the boundary
                            BoundaryCondition::Force(force.map(|f| {
                                f.scale_amplitude(1.0 / mesh.boundary_point_map()[id].len() as f32)
                            }))
                        }
                        BoundaryCondition::Displacement(displacement) => {
                            BoundaryCondition::Displacement(displacement.clone())
                        }
                    })
                    .sum();
                (*vertex_index, condition)
            })
            .collect();
        Computer::Unconfigured(computer::unconfigured(
            mesh.triangulation_data(),
            mesh.boundary_point_map(),
            boundary_conditions,
            &point_boundary_conditions,
        ))
    }

    pub fn new_with_cpd_data(
        project_data: Data<WithCpdExportData>,
        senders: Senders,
    ) -> Result<Self, String> {
        let (project_data, export_data) = project_data.without_export_data();
        use computer::ImportResult;
        let config_with_cpd = |cpd_config| {
            Some(Box::new(
                Config::builder()
                    .cpd_config(cpd_config)
                    .refresh_period(1)
                    .export_config(None)
                    .build(),
            ))
        };
        let (computer, config) = match computer::from_export_data(export_data) {
            ImportResult::Unconfigured(c) => (Computer::Unconfigured(c), None),
            ImportResult::InProgress((c, config)) => {
                (Computer::InProgress(c), config_with_cpd(config))
            }
            ImportResult::Done((c, config)) => (Computer::Done(c), config_with_cpd(config)),
            ImportResult::Err(err) => return Err(err),
        };
        let state = match &computer {
            Computer::Unconfigured(_) => State::Unconfigured,
            Computer::InProgress(_) => State::Paused,
            Computer::Done(_) => State::Finished,
        };
        if senders.state_sender.send(state).is_err() {
            return Err(String::from("State sender already dropped"));
        }
        if senders.frame_sender.send(frame!(computer)).is_err() {
            return Err(String::from("Frame sender already dropped"));
        }
        if let Some(config) = &config {
            if senders.config_sender.send(config.clone()).is_err() {
                return Err(String::from("Config sender already dropped"));
            }
        }
        let pstate = project_data.state();
        Ok(Self {
            plot_items: PlotItems::from(&pstate.boundary_conditions),
            worker: Worker::new(computer, config, pstate.mesh.clone(), state, senders),
            project_data,
            refresh_token_set: false,
        })
    }

    pub fn set_refresh_token(&mut self, refresh_token: impl Into<T>) {
        if self.refresh_token_set {
            return;
        }
        self.worker
            .send_command(Command::SetRefreshToken(refresh_token.into()));
        self.refresh_token_set = true;
    }

    pub fn project_data(&self) -> &Data<WithMesh> {
        &self.project_data
    }

    pub fn plot_items(&self) -> &PlotItems {
        &self.plot_items
    }

    pub fn take_project_data(self) -> Data<WithMesh> {
        self.project_data
    }

    pub fn polygon_data(&self) -> &PolygonData {
        &self.project_data.state().polygon_data
    }

    pub fn configure(&mut self, config: Box<Config>) {
        self.worker.send_command(Command::Configure(config));
    }

    pub fn play(&mut self) {
        self.worker.send_command(Command::Play);
    }

    pub fn pause(&mut self) {
        self.worker.send_command(Command::Pause);
    }

    pub fn rewind(&mut self) {
        self.worker.send_command(Command::Reset);
    }

    pub fn record_stress_data_of_element(&mut self, index: usize) {
        self.worker.send_command(Command::RecordStressData(index));
    }

    pub fn stop_recording_stress_data(&mut self, index: usize) {
        self.worker
            .send_command(Command::StopRecordingStressData(index));
    }

    pub fn record_vertex_position(&mut self, index: usize) {
        self.worker
            .send_command(Command::RecordVertexPosition(index));
    }

    pub fn stop_recording_vertex_position(&mut self, index: usize) {
        self.worker
            .send_command(Command::StopRecordingVertexPosition(index));
    }

    pub fn record_boundary_data(&mut self, id: BoundaryId) {
        self.worker.send_command(Command::RecordBoundaryData(id));
    }

    pub fn stop_recording_boundary_data(&mut self) {
        self.worker.send_command(Command::StopRecordingBoundaryData);
    }
}

#[derive(Debug, Clone)]
enum Command<T> {
    SetRefreshToken(T),
    Configure(Box<Config>),
    Play,
    Pause,
    Reset,
    RecordStressData(usize),
    StopRecordingStressData(usize),
    RecordVertexPosition(usize),
    StopRecordingVertexPosition(usize),
    RecordBoundaryData(BoundaryId),
    StopRecordingBoundaryData,
}

#[derive(Debug)]
struct Worker<T: RefreshToken> {
    command_sender: mpsc::Sender<Command<T>>,
    handle: Option<JoinHandle<()>>,
}

#[derive(Debug)]
enum Computer {
    Unconfigured(computer::Computer<computer::Unconfigured>),
    InProgress(computer::Computer<computer::InProgress>),
    Done(computer::Computer<computer::Done>),
}

struct RunArgs<T> {
    state: State,
    config: Option<Box<Config>>,
    computer: Computer,
    refresh_token: T,
    mesh: Arc<Mesh>,
    command_receiver: mpsc::Receiver<Command<T>>,
    senders: Senders,
}

type ConfigureResult = Result<Computer, (Computer, String)>;

impl<T: RefreshToken> Worker<T> {
    fn new(
        computer: Computer,
        config: Option<Box<Config>>,
        mesh: Arc<Mesh>,
        state: State,
        senders: Senders,
    ) -> Self {
        let (command_sender, command_receiver) = mpsc::channel();
        Self {
            command_sender,
            handle: {
                let run_args = RunArgs {
                    state,
                    config,
                    computer,
                    refresh_token: T::default(),
                    mesh,
                    command_receiver,
                    senders,
                };
                Some(thread::spawn(move || Self::run(run_args)))
            },
        }
    }

    fn run(mut run_args: RunArgs<T>) {
        let senders = &run_args.senders;
        loop {
            match run_args.command_receiver.try_recv() {
                Ok(command) => {
                    let result = Self::process_command(
                        command,
                        &mut run_args.refresh_token,
                        &mut run_args.state,
                        run_args.computer,
                        &mut run_args.config,
                        &run_args.mesh,
                    );
                    run_args.computer = match result {
                        Ok(computer) => computer,
                        Err((computer, err)) => {
                            if senders.error_sender.send(err).is_err() {
                                break;
                            }
                            computer
                        }
                    };
                    if senders.state_sender.send(run_args.state).is_err()
                        || senders
                            .frame_sender
                            .send(frame!(run_args.computer))
                            .is_err()
                    {
                        break;
                    }
                    if let Some(config) = &run_args.config {
                        if senders.config_sender.send(config.clone()).is_err() {
                            break;
                        }
                    }
                    run_args.refresh_token.refresh();
                }
                Err(TryRecvError::Disconnected) => {
                    break;
                }
                Err(TryRecvError::Empty) => {}
            }
            if run_args.state != State::Running {
                continue;
            };
            run_args.computer = match run_args.computer {
                Computer::InProgress(computer) => {
                    let (frame, computer, err) = Self::advance_simulation(
                        computer,
                        run_args
                            .config
                            .as_ref()
                            .expect("Config should be present at InProgress state"),
                    );
                    if frame
                        .and_then(|frame| senders.frame_sender.send(frame).err())
                        .is_some()
                        || err
                            .and_then(|err| senders.error_sender.send(err).err())
                            .is_some()
                    {
                        break;
                    }
                    run_args.refresh_token.refresh();
                    computer
                }
                Computer::Done(computer) => {
                    if senders.state_sender.send(State::Finished).is_err() {
                        break;
                    }
                    run_args.state = State::Finished;
                    Computer::Done(computer)
                }
                _ => unreachable!(),
            }
        }
    }

    fn diff_cpd_config(old: &CpdConfig, new: &CpdConfig) -> CpdConfigDiffResult {
        if old.material_props() != new.material_props() || old.time_delta() != new.time_delta() {
            CpdConfigDiffResult::NeedsReconfiguration
        } else if old.duration() == new.duration() {
            CpdConfigDiffResult::NoDiff
        } else {
            CpdConfigDiffResult::UpdateDuration
        }
    }

    fn diff_export_path(old: Option<&Config>, new: &Config) -> Option<PathBuf> {
        let new_export_path = new.export_config().as_ref().map(ExportConfig::export_path);
        let old_export_path = old
            .and_then(|config| config.export_config().as_ref())
            .map(ExportConfig::export_path);
        match (old_export_path, new_export_path) {
            (None, None) | (Some(_), None) => None,
            (None, Some(path)) => Some(path.clone()),
            (Some(old), Some(new)) => (old != new).then_some(new.clone()),
        }
    }

    fn process_command(
        command: Command<T>,
        refresh_token: &mut T,
        state: &mut State,
        computer: Computer,
        config: &mut Option<Box<Config>>,
        mesh: &Mesh,
    ) -> ConfigureResult {
        let computer = match command {
            Command::SetRefreshToken(token) => {
                *refresh_token = token;
                computer
            }
            Command::Configure(in_config) => {
                let result = Self::configure(config, &in_config, computer, mesh);
                if result.is_ok() {
                    *state = State::Paused;
                    *config = Some(in_config);
                }
                return result;
            }
            Command::Play => {
                *state = State::Running;
                computer
            }
            Command::Pause => {
                *state = State::Paused;
                computer
            }
            Command::Reset => match computer {
                Computer::Unconfigured(_) => panic!("Cannot reset an unconfigured computer"),
                Computer::InProgress(mut c) => {
                    c.reset();
                    let computer = Computer::InProgress(c);
                    *state = State::Paused;
                    computer
                }
                Computer::Done(c) => {
                    let computer = Computer::InProgress(c.reset());
                    *state = State::Paused;
                    computer
                }
            },
            Command::RecordStressData(handle) => {
                let mut computer = computer;
                match &mut computer {
                    Computer::Unconfigured(c) => c.record_stress_data(handle),
                    Computer::InProgress(c) => c.record_stress_data(handle),
                    Computer::Done(c) => c.record_stress_data(handle),
                }
                computer
            }
            Command::StopRecordingStressData(handle) => {
                let mut computer = computer;
                match &mut computer {
                    Computer::Unconfigured(c) => c.stop_recording_stress_data(handle),
                    Computer::InProgress(c) => c.stop_recording_stress_data(handle),
                    Computer::Done(c) => c.stop_recording_stress_data(handle),
                }
                computer
            }
            Command::RecordVertexPosition(index) => {
                let mut computer = computer;
                match &mut computer {
                    Computer::Unconfigured(c) => c.record_vertex_position(index),
                    Computer::InProgress(c) => c.record_vertex_position(index),
                    Computer::Done(c) => c.record_vertex_position(index),
                }
                computer
            }
            Command::StopRecordingVertexPosition(index) => {
                let mut computer = computer;
                match &mut computer {
                    Computer::Unconfigured(c) => c.stop_recording_vertex_position(index),
                    Computer::InProgress(c) => c.stop_recording_vertex_position(index),
                    Computer::Done(c) => c.stop_recording_vertex_position(index),
                }
                computer
            }
            Command::RecordBoundaryData(id) => {
                let mut computer = computer;
                match &mut computer {
                    Computer::Unconfigured(c) => c.record_boundary_data(id),
                    Computer::InProgress(c) => c.record_boundary_data(id),
                    Computer::Done(c) => c.record_boundary_data(id),
                }
                computer
            }
            Command::StopRecordingBoundaryData => {
                let mut computer = computer;
                match &mut computer {
                    Computer::Unconfigured(c) => c.stop_recording_boundary_data(),
                    Computer::InProgress(c) => c.stop_recording_boundary_data(),
                    Computer::Done(c) => c.stop_recording_boundary_data(),
                }
                computer
            }
        };
        Ok(computer)
    }

    fn configure(
        config: &mut Option<Box<Config>>,
        in_config: &Config,
        computer: Computer,
        mesh: &Mesh,
    ) -> ConfigureResult {
        let changed_export_path =
            Self::diff_export_path(config.as_ref().map(|b| b.deref()), in_config);
        let err_opt = changed_export_path.and_then(|path| export::mesh(mesh, &path).err());
        if let Some(err) = err_opt {
            return Err((computer, format!("Failed to export mesh {err}")));
        }
        let diff_opt = config
            .as_ref()
            .map(|c| c.cpd_config())
            .map(|old| Self::diff_cpd_config(old, in_config.cpd_config()));
        let computer = match diff_opt {
            Some(diff) => match diff {
                CpdConfigDiffResult::NoDiff => computer,
                CpdConfigDiffResult::NeedsReconfiguration => Computer::InProgress(match computer {
                    Computer::Unconfigured(_) => unreachable!(),
                    Computer::InProgress(c) => c.reconfigure(*in_config.cpd_config()),
                    Computer::Done(c) => c.reconfigure(*in_config.cpd_config()),
                }),
                CpdConfigDiffResult::UpdateDuration => {
                    let duration = *in_config.cpd_config().duration();
                    return match computer {
                        Computer::Unconfigured(_) => unreachable!(),
                        Computer::InProgress(mut c) => match c.set_duration(duration) {
                            Ok(()) => Ok(Computer::InProgress(c)),
                            Err(err) => Err((Computer::InProgress(c), err)),
                        },
                        Computer::Done(c) => c
                            .set_duration(duration)
                            .map(Computer::InProgress)
                            .map_err(|(c, e)| (Computer::Done(c), e)),
                    };
                }
            },
            None => match computer {
                Computer::Unconfigured(c) => {
                    Computer::InProgress(c.configure(*in_config.cpd_config()))
                }
                Computer::InProgress(_) | Computer::Done(_) => unreachable!(),
            },
        };
        Ok(computer)
    }

    fn advance_simulation(
        computer: computer::Computer<computer::InProgress>,
        config: &Config,
    ) -> (Option<Frame>, Computer, Option<String>) {
        use computer::AdvanceResult;
        match computer.advance() {
            AdvanceResult::InProgress(c) => {
                let refresh_period = *config.refresh_period();
                let time_step = c.iterations();
                let refresh = refresh_period <= 1 || time_step % refresh_period == 0;
                let export_config = config.export_config().as_ref();
                let export = export_config
                    .map(ExportConfig::export_period)
                    .copied()
                    .is_some_and(|p| p <= 1 || time_step % p == 0);
                let computer = Computer::InProgress(c);
                if refresh || export {
                    let frame = frame!(computer);
                    let error = export
                        .then(|| {
                            export::data(frame.data(), export_config.unwrap(), time_step).err()
                        })
                        .flatten()
                        .map(|err| format!("Failed to export data {err}"));
                    (Some(frame), computer, error)
                } else {
                    (None, computer, None)
                }
            }
            AdvanceResult::Done(c) => {
                let time_step = c.iterations();
                let computer = Computer::Done(c);
                let frame = frame!(computer);
                let error = config
                    .export_config()
                    .as_ref()
                    .and_then(|config| export::data(frame.data(), config, time_step).err())
                    .map(|err| format!("Failed to export data {err}"));
                (Some(frame), computer, error)
            }
        }
    }

    fn send_command(&mut self, command: Command<T>) {
        self.command_sender
            .send(command)
            .map_err(|_| {
                self.handle
                    .take()
                    .expect("Handle should be present")
                    .join()
                    .expect_err("Sender should be dropped only if worker thread has panicked")
            })
            .unwrap()
    }
}
