use super::{
    project::data::{Data, WithBoundaryConditions, WithMesh},
    state_channel, PolygonData, RefreshToken,
};
use cgal::{PolygonSet, PolygonSetInput};
use mesh::{Callback, Mesh};
use std::{
    sync::{mpsc, Arc},
    thread::{self, JoinHandle},
};

#[derive(Debug, Default)]
pub enum State {
    #[default]
    Idle,
    GeneratingMesh(mesh::State),
    Mesh(Arc<Mesh>),
}

#[derive(Debug)]
pub struct MeshGenerator<T: RefreshToken> {
    project_data: Data<WithBoundaryConditions>,
    worker: Worker<T>,
    refresh_token_set: bool,
}

impl<T: RefreshToken> MeshGenerator<T> {
    pub fn new(
        project_data: Data<WithBoundaryConditions>,
        state_sender: state_channel::Sender<State>,
        error_sender: state_channel::Sender<String>,
    ) -> Self {
        Self {
            project_data,
            worker: Worker::new(state_sender, error_sender),
            refresh_token_set: false,
        }
    }

    pub fn new_with_mesh(
        project_data: Data<WithMesh>,
        state_sender: state_channel::Sender<State>,
        error_sender: state_channel::Sender<String>,
    ) -> Result<Self, String> {
        let (project_data, mesh) = project_data.without_mesh();
        if state_sender.send(State::Mesh(mesh)).is_err() {
            Err(String::from("State channel is already dropped"))
        } else {
            Ok(Self {
                project_data,
                worker: Worker::new(state_sender, error_sender),
                refresh_token_set: false,
            })
        }
    }

    pub fn set_refresh_token(&mut self, refresh_token: impl Into<T>) {
        if self.refresh_token_set {
            return;
        }
        self.worker
            .send(Command::SetRefreshToken(refresh_token.into()));
        self.refresh_token_set = true;
    }

    pub fn polygon_data(&self) -> &PolygonData {
        &self.project_data.state().polygon_data
    }

    pub fn project_data(&self) -> &Data<WithBoundaryConditions> {
        &self.project_data
    }

    pub fn project_data_with_bc(self) -> Data<WithBoundaryConditions> {
        self.project_data
    }

    pub fn generate(
        &mut self,
        num_points: u32,
        size_bound_override: Option<f64>,
    ) -> Result<(), String> {
        if num_points == 0 {
            return Err(String::from("Number of points must be greater than 0"));
        }
        self.worker.send(Command::Input(Input {
            polygon_set_inputs: self.polygon_data().inputs.clone(),
            num_points,
            size_bound_override,
        }));
        Ok(())
    }
}

#[derive(Debug)]
enum Command<T: RefreshToken> {
    SetRefreshToken(T),
    Input(Input),
}

#[derive(Debug)]
struct Input {
    polygon_set_inputs: Vec<PolygonSetInput>,
    num_points: u32,
    size_bound_override: Option<f64>,
}

#[derive(Debug)]
struct Worker<T: RefreshToken> {
    command_sender: mpsc::Sender<Command<T>>,
    handle: Option<JoinHandle<()>>,
}

impl<T: RefreshToken> Worker<T> {
    fn new(
        state_sender: state_channel::Sender<State>,
        error_sender: state_channel::Sender<String>,
    ) -> Self {
        let (command_sender, command_receiver) = mpsc::channel();
        Self {
            command_sender,
            handle: {
                Some(thread::spawn(move || {
                    Self::command_queue(command_receiver, T::default(), state_sender, error_sender)
                }))
            },
        }
    }

    fn send(&mut self, command: Command<T>) {
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

    fn command_queue(
        command_receiver: mpsc::Receiver<Command<T>>,
        mut refresh_token: T,
        state_sender: state_channel::Sender<State>,
        error_sender: state_channel::Sender<String>,
    ) {
        while let Ok(command) = command_receiver.recv() {
            let input = match command {
                Command::SetRefreshToken(token) => {
                    refresh_token = token;
                    continue;
                }
                Command::Input(input) => input,
            };
            macro_rules! send_state_discard_err {
                ( $state:expr ) => {
                    if state_sender.send($state).is_ok() {
                        refresh_token.refresh();
                    }
                };
            }
            macro_rules! send_state {
                ( $state:expr ) => {
                    if state_sender.send($state).is_err() {
                        break;
                    }
                    refresh_token.refresh();
                };
            }
            macro_rules! send_err {
                ( $err:expr ) => {{
                    if error_sender.send($err).is_err() {
                        break;
                    }
                    refresh_token.refresh();
                }};
            }
            let polygon_set = PolygonSet::from_inputs(&input.polygon_set_inputs)
                .expect("Polygon set inputs are valid");
            let result = Mesh::generate(
                &polygon_set.polygon_with_holes()[0],
                input.num_points,
                input.size_bound_override,
                Callback::from(|state| send_state_discard_err!(State::GeneratingMesh(state))),
            );
            match result {
                Ok(mesh) => {
                    send_state!(State::Mesh(Arc::new(mesh)));
                }
                Err(err) => {
                    send_err!(err);
                }
            }
        }
    }
}
