mod dialog;

use super::{bottom_panel, error_dialog, plot_utils, unicode_symbols, ContextWrapper};
use crate::model::{
    mesh_generator::{MeshGenerator, State},
    project::data::{Data, WithBoundaryConditions, WithMesh, WithShape},
    state_channel::{self, Receiver, STReceiver, Sender},
};
use cgal::triangulation;
use egui::{Button, CentralPanel, Color32, Frame, Ui};
use egui_plot::{Line, PlotUi, Points, Polygon};
use mesh::{Constraint, Mesh};
use nalgebra::Vector2;
use rayon::prelude::*;
use std::{iter, mem};

#[derive(Debug)]
pub struct Page {
    mesh_generator: MeshGenerator<ContextWrapper>,
    dialog_state: Option<dialog::State>,
    input_error: Option<String>,
    show_wireframe_only: bool,
    hide_mesh: bool,
    show_constraints: bool,
    show_interior_points: bool,
    state_receiver: STReceiver<State>,
    error_receiver: Receiver<String, Option<String>>,
}

#[derive(Debug)]
pub enum MenuResponse {
    Noop(Page),
    EditBoundaryConditions(Data<WithBoundaryConditions>),
    EditShape(Data<WithShape>),
}

#[derive(Debug)]
pub enum Response {
    Noop(Page),
    RunSimulation(Data<WithMesh>),
}

impl From<Data<WithBoundaryConditions>> for Page {
    fn from(project_data: Data<WithBoundaryConditions>) -> Self {
        Self::with_mesh_generator(|state_sender, error_sender| {
            MeshGenerator::new(project_data, state_sender, error_sender)
        })
    }
}

impl From<Data<WithMesh>> for Page {
    fn from(project_data: Data<WithMesh>) -> Self {
        Self::with_mesh_generator(|state_sender, error_sender| {
            MeshGenerator::new_with_mesh(project_data, state_sender, error_sender)
                .expect("State channel is active")
        })
    }
}

impl Page {
    fn with_mesh_generator(
        mesh_generator: impl FnOnce(Sender<State>, Sender<String>) -> MeshGenerator<ContextWrapper>,
    ) -> Self {
        let (state_sender, state_receiver) = state_channel::same_type_with_default(5);
        let (error_sender, error_receiver) = state_channel::with_default(1);
        Self {
            mesh_generator: mesh_generator(state_sender, error_sender),
            dialog_state: None,
            input_error: None,
            show_wireframe_only: true,
            hide_mesh: false,
            show_constraints: true,
            show_interior_points: true,
            state_receiver,
            error_receiver,
        }
    }

    #[must_use]
    pub fn add_menu_items(self, ui: &mut Ui) -> MenuResponse {
        puffin::profile_function!();
        #[derive(Debug, Default)]
        struct Response {
            edit_bc: bool,
            edit_shape: bool,
        }
        let opt = ui
            .menu_button("Edit", |ui| {
                let mut response = Response::default();
                if ui.button("Edit conditions").clicked() {
                    response.edit_bc = true;
                    ui.close_menu();
                }
                if ui.button("Edit shape").clicked() {
                    response.edit_shape = true;
                    ui.close_menu();
                }
                response
            })
            .inner;
        match opt {
            Some(response) => {
                if response.edit_bc {
                    MenuResponse::EditBoundaryConditions(self.mesh_generator.project_data_with_bc())
                } else if response.edit_shape {
                    MenuResponse::EditShape(
                        self.mesh_generator
                            .project_data_with_bc()
                            .without_boundary_conditions()
                            .0,
                    )
                } else {
                    MenuResponse::Noop(self)
                }
            }
            None => MenuResponse::Noop(self),
        }
    }

    #[must_use]
    pub fn add_contents(mut self, ui: &mut Ui) -> Response {
        puffin::profile_function!();
        self.mesh_generator.set_refresh_token(ui.ctx());
        ui.heading("Meshing");

        self.state_receiver
            .update()
            .expect("Sender should not be dropped");

        let response =
            bottom_panel::show("meshing_bottom_panel", ui, |ui| self.add_bottom_panel(ui)).inner;

        match response {
            Response::Noop(page) => {
                self = page;
            }
            Response::RunSimulation(pd) => {
                return Response::RunSimulation(pd);
            }
        }

        CentralPanel::default()
            .frame(Frame::default())
            .show_inside(ui, |ui| self.add_preview(ui));

        let response =
            Self::input_dialog_and_error_ui(ui, &mut self.dialog_state, &mut self.input_error);

        if let Some(err) = &self
            .error_receiver
            .update_and_get()
            .expect("Sender should not be dropped")
        {
            if error_dialog::show(err, ui.ctx()).closed() {
                self.error_receiver.data = None;
            }
        }

        if let Some(data) = response {
            self.mesh_generator
                .generate(data.num_points, data.size_bound_override)
                .expect("Input should valildate num points");
        }
        Response::Noop(self)
    }

    fn add_bottom_panel(mut self, ui: &mut Ui) -> Response {
        puffin::profile_function!();
        let state = mem::take(&mut self.state_receiver.data);
        self.mesh_controls(ui, &state);
        if let State::Mesh(mesh) = &state {
            Self::mesh_info(ui, mesh.triangulation_data());
        }
        ui.horizontal(|ui| match state {
            State::Mesh(mesh) => {
                if ui.button("Run").clicked() {
                    Response::RunSimulation(
                        self.mesh_generator.project_data_with_bc().with_mesh(mesh),
                    )
                } else {
                    self.state_receiver.data = State::Mesh(mesh);
                    Response::Noop(self)
                }
            }
            other => {
                ui.add_enabled(false, Button::new("Run"))
                    .on_disabled_hover_text("Cannot run simulation until meshing is done");
                self.state_receiver.data = other;
                Response::Noop(self)
            }
        })
        .inner
    }

    fn add_preview(&self, ui: &mut Ui) {
        puffin::profile_function!();
        match &self.state_receiver.data {
            State::Idle | State::GeneratingMesh(_) => self.plot_polygon_set_and_mesh(ui, |_| ()),
            State::Mesh(mesh) => {
                self.plot_polygon_set_and_mesh(ui, |ui: &mut PlotUi| self.plot_mesh(ui, mesh));
            }
        }
    }

    fn plot_polygon_set_and_mesh<F>(&self, ui: &mut Ui, add_mesh_contents: F)
    where
        F: Fn(&mut PlotUi),
    {
        puffin::profile_function!();
        plot_utils::plot_without_clutter("meshing_plot").show(ui, |ui| {
            let polygon_set = self.mesh_generator.polygon_data().polygon_set();
            plot_utils::plot_polygon_set(ui, polygon_set, plot_utils::default_transform);
            add_mesh_contents(ui);
        });
    }

    fn plot_mesh(&self, ui: &mut PlotUi, mesh: &Mesh) {
        puffin::profile_function!();
        if !self.hide_mesh {
            if self.show_wireframe_only {
                Self::show_wireframe_mesh(ui, mesh.triangulation_data());
            } else {
                Self::show_colored_mesh(ui, mesh.triangulation_data());
            }
            return;
        }

        let collect_points = |xor| {
            mesh.triangulation_data()
                .vertices()
                .par_iter()
                .enumerate()
                .filter(|(index, _)| mesh.point_id_map().contains_key(index) ^ xor)
                .map(|(_, vertex)| vertex.point().map(|v| v as f64).into())
                .collect::<Vec<[f64; 2]>>()
        };

        if self.show_interior_points {
            ui.points(
                Points::new(collect_points(true))
                    .radius(1.0)
                    .color(super::on_primary_color(ui.ctx())),
            );
        }

        if self.show_constraints {
            ui.points(
                Points::new(collect_points(false))
                    .radius(1.0)
                    .color(Color32::GREEN),
            );
            let vec_to_arr = |v: &Vector2<f64>| [v.x, v.y];
            mesh.constraints()
                .iter()
                .flat_map(|(_, constraint)| {
                    let iter: Box<dyn Iterator<Item = &[Vector2<f64>; 2]>> = match constraint {
                        Constraint::Line(arr) => Box::new(iter::once(arr)),
                        Constraint::PolyLine(boxed) => Box::new(boxed.iter()),
                    };
                    iter
                })
                .map(|[a, b]| Line::new(vec![vec_to_arr(a), vec_to_arr(b)]))
                .for_each(|line| ui.line(line));
        }
    }

    fn show_wireframe_mesh(ui: &mut PlotUi, data: &triangulation::Data) {
        puffin::profile_function!();
        let line_color = super::on_primary_color(ui.ctx());
        data.edges()
            .iter()
            .map(|pair| {
                let convert_vertex = |vertex: &Vector2<f32>| vertex.map(|v| v as f64).into();
                vec![
                    convert_vertex(data.vertices()[pair.0].point()),
                    convert_vertex(data.vertices()[pair.1].point()),
                ]
            })
            .map(|series| Line::new(series).color(line_color))
            .for_each(|line| ui.line(line));
    }

    fn show_colored_mesh(ui: &mut PlotUi, data: &triangulation::Data) {
        puffin::profile_function!();
        data.faces()
            .iter()
            .map(|face| {
                face.0
                    .map(|index| data.vertices()[index].point().data.0[0].map(|v| v as f64))
                    .to_vec()
            })
            .map(Polygon::new)
            .for_each(|polygon| ui.polygon(polygon));
    }

    fn mesh_controls(&mut self, ui: &mut Ui, state: &State) {
        ui.horizontal(|ui| match state {
            State::Idle => {
                let response = ui
                    .button("Generate mesh")
                    .on_hover_text("Click to generate mesh");
                if response.clicked() {
                    self.dialog_state = Some(dialog::State::default());
                }
            }
            State::GeneratingMesh(state) => {
                ui.spinner();
                let label = match state {
                    mesh::State::Init => {
                        const_format::formatcp!("Initializing{}", unicode_symbols::ELLIPSIS)
                    }
                    mesh::State::GeneratingConstraints => const_format::formatcp!(
                        "Generating constraints{}",
                        unicode_symbols::ELLIPSIS
                    ),
                    mesh::State::Triangulating => const_format::formatcp!(
                        "Triangulating, this may take a while{}",
                        unicode_symbols::ELLIPSIS
                    ),
                    mesh::State::GeneratingAssociativeData => const_format::formatcp!(
                        "Generating associative data{}",
                        unicode_symbols::ELLIPSIS
                    ),
                    mesh::State::FindingSmallestEdge => const_format::formatcp!(
                        "Finding smallest edge{}",
                        unicode_symbols::ELLIPSIS
                    ),
                    mesh::State::Done => "Done",
                };
                ui.label(label);
            }
            State::Mesh(_) => self.add_regen_button_and_viewmode_toggles(ui),
        });
    }

    fn add_regen_button_and_viewmode_toggles(&mut self, ui: &mut Ui) {
        let response = ui
            .button("Regenerate mesh")
            .on_hover_text("Click to regenerate mesh");
        if response.clicked() {
            self.dialog_state = Some(dialog::State::default());
        }
        if self.hide_mesh {
            ui.checkbox(&mut self.show_constraints, "Show constraints");
            ui.checkbox(&mut self.show_interior_points, "Show interior points");
        } else {
            ui.checkbox(&mut self.show_wireframe_only, "Show wireframe only");
        }
    }

    fn mesh_info(ui: &mut Ui, triangulation_data: &triangulation::Data) {
        ui.horizontal(|ui| {
            ui.label(format!("Elements: {}", triangulation_data.faces().len()));
            ui.label(format!("Points: {}", triangulation_data.vertices().len()));
        });
    }

    #[must_use]
    fn input_dialog_and_error_ui(
        ui: &mut Ui,
        dialog_state: &mut Option<dialog::State>,
        input_error: &mut Option<String>,
    ) -> Option<dialog::Data> {
        if let Some(err) = input_error.as_ref() {
            if error_dialog::show(err, ui.ctx()).closed() {
                *input_error = None;
            }
            return None;
        }

        let mut state = dialog_state.take()?;
        use dialog::Response;
        match dialog::show(&mut state, ui.ctx()) {
            Response::Noop => {
                *dialog_state = Some(state);
                None
            }
            Response::DataResult(result) => match result {
                Ok(data) => Some(data),
                Err(err) => {
                    *dialog_state = Some(state);
                    *input_error = err.into();
                    None
                }
            },
            Response::Cancel => None,
        }
    }
}

mod serde_impl {
    use super::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Debug, Serialize, Deserialize)]
    enum WrappedProjectData {
        WithBoundaryConditions(Data<WithBoundaryConditions>),
        WithMesh(Data<WithMesh>),
    }

    impl Serialize for Page {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let project_data = self.mesh_generator.project_data().clone();
            let project_data = match &self.state_receiver.data {
                State::Mesh(mesh) => {
                    WrappedProjectData::WithMesh(project_data.with_mesh(mesh.clone()))
                }
                _ => WrappedProjectData::WithBoundaryConditions(project_data),
            };
            project_data.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for Page {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            WrappedProjectData::deserialize(deserializer).map(|wpd| match wpd {
                WrappedProjectData::WithBoundaryConditions(project_data) => {
                    Page::from(project_data)
                }
                WrappedProjectData::WithMesh(project_data) => Page::from(project_data),
            })
        }
    }
}
