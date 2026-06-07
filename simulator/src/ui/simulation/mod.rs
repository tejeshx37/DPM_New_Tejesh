mod config_dialog;
mod plot_dialog;

use super::{bottom_panel, error_dialog, plot_utils, unicode_symbols, ContextWrapper};
use crate::model::{
    engine::{Config, Engine, Frame, Senders, State},
    project::data::{Data, WithBoundaryConditions, WithCpdExportData, WithMesh, WithShape},
    state_channel::{self, Receiver, STReceiver},
};
use cgal::BoundaryId;
use cpd::{BoundaryAverage, ExportData, TimeStampedValue};
use ecolor::Hsva;
use egui::{
    Button, CentralPanel, CollapsingHeader, Color32, Key, Modifiers, ProgressBar, SidePanel,
    Stroke, Ui, Vec2, Vec2b, WidgetText,
};
use egui_plot::{AxisHints, Line, Plot, PlotPoint, PlotUi, Points, Polygon};
use nalgebra::{Matrix2, Vector2};
use nalgebra_ext::matrix2::Component;
use rayon::prelude::*;
use std::hash::Hash;
use strum::IntoEnumIterator;
use typed_builder::TypedBuilder;

const ORANGE: Color32 = Color32::from_rgb(0xFF, 0xA5, 0x00);

#[derive(Debug, TypedBuilder)]
struct Receivers {
    config_receiver: Receiver<Box<Config>, Option<Box<Config>>>,
    state_receiver: STReceiver<State>,
    frame_receiver: Receiver<Frame, Option<Frame>>,
    error_receiver: Receiver<String, Option<String>>,
}

#[derive(Debug)]
pub struct Page {
    engine: Engine<ContextWrapper>,
    config_dialog_state: Option<config_dialog::State>,
    configure_error: Option<String>,
    show_stress_gradients: bool,
    stress_tensor_component: Component,
    receivers: Receivers,
    selected_element_index: Option<usize>,
    selected_vertex_index: Option<usize>,
    plot_dialog_state: Option<plot_dialog::State>,
    selected_boundary_id: Option<BoundaryId>,
}

#[derive(Debug)]
pub enum MenuResponse {
    Noop(Page),
    EditMesh(Data<WithMesh>),
    EditBoundaryConditions(Data<WithBoundaryConditions>),
    EditShape(Data<WithShape>),
}

#[derive(Debug)]
pub enum Response {
    Noop(Page),
}

fn senders_and_receivers() -> (Senders, Receivers) {
    let (config_sender, config_receiver) = state_channel::with_default(1);
    let (state_sender, state_receiver) = state_channel::same_type_with_default(1);
    let (frame_sender, frame_receiver) = state_channel::with_default(1);
    let (error_sender, error_receiver) = state_channel::with_default(1);
    let senders = Senders::builder()
        .config_sender(config_sender)
        .state_sender(state_sender)
        .frame_sender(frame_sender)
        .error_sender(error_sender)
        .build();
    let receivers = Receivers::builder()
        .config_receiver(config_receiver)
        .state_receiver(state_receiver)
        .frame_receiver(frame_receiver)
        .error_receiver(error_receiver)
        .build();
    (senders, receivers)
}

impl From<Data<WithMesh>> for Page {
    fn from(project_data: Data<WithMesh>) -> Self {
        let (senders, receivers) = senders_and_receivers();
        Self::with_engine(Engine::new(project_data, senders), receivers)
    }
}

impl TryFrom<Data<WithCpdExportData>> for Page {
    type Error = String;

    fn try_from(project_data: Data<WithCpdExportData>) -> Result<Self, Self::Error> {
        let (senders, receivers) = senders_and_receivers();
        Engine::new_with_cpd_data(project_data, senders)
            .map(|engine| Self::with_engine(engine, receivers))
    }
}

#[derive(Debug, Default, Clone, Copy)]
enum FramePlotHoverResponse {
    #[default]
    Noop,
    ElementIndex(usize),
    VertexIndex(usize),
}

#[derive(Debug, Default, Clone, Copy)]
enum FramePreviewResponse {
    #[default]
    Noop,
    ElementSelected(usize),
    VertexSelected(usize),
}

impl Page {
    fn with_engine(engine: Engine<ContextWrapper>, receivers: Receivers) -> Self {
        Self {
            engine,
            config_dialog_state: None,
            configure_error: None,
            show_stress_gradients: true,
            stress_tensor_component: Component::default(),
            receivers,
            selected_element_index: None,
            selected_vertex_index: None,
            plot_dialog_state: None,
            selected_boundary_id: None,
        }
    }

    fn add_edit_menu(self, ui: &mut Ui) -> MenuResponse {
        puffin::profile_function!();
        #[derive(Debug, Default)]
        struct Response {
            edit_mesh: bool,
            edit_bc: bool,
            edit_shape: bool,
        }
        let opt = ui
            .menu_button("Edit", |ui| {
                let mut response = Response::default();
                if ui.button("Edit mesh").clicked() {
                    response.edit_mesh = true;
                    ui.close_menu();
                }
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
        let Some(response) = opt else {
            return MenuResponse::Noop(self);
        };
        if response.edit_mesh {
            MenuResponse::EditMesh(self.engine.take_project_data())
        } else if response.edit_bc {
            MenuResponse::EditBoundaryConditions(self.engine.take_project_data().without_mesh().0)
        } else if response.edit_shape {
            MenuResponse::EditShape(
                self.engine
                    .take_project_data()
                    .without_mesh()
                    .0
                    .without_boundary_conditions()
                    .0,
            )
        } else {
            MenuResponse::Noop(self)
        }
    }

    fn add_plot_menu(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        ui.menu_button("Plot", |ui| {
            let discard_plot =
                self.selected_element_index.is_some() && ui.button("Discard stress plot").clicked();
            if discard_plot {
                ui.close_menu();
                self.stop_recording_stress_data();
            }

            let discard_plot = self.selected_vertex_index.is_some()
                && ui.button("Discard displacement plot").clicked();
            if discard_plot {
                ui.close_menu();
                self.stop_recording_vertex_position();
            }

            let response = ui.button("Boundary average").on_hover_text(
                "Plot average displacement or force on a boundary. \n\
            Click to open a dialog from where you can choose the boundary.",
            );
            if response.clicked() {
                ui.close_menu();
                self.plot_dialog_state = Some(plot_dialog::State::new(
                    self.engine.plot_items().clone(),
                    self.engine.polygon_data().clone(),
                ));
            }

            let discard_plot = self.selected_boundary_id.is_some()
                && ui.button("Discard boundary average plot").clicked();
            if discard_plot {
                ui.close_menu();
                self.selected_boundary_id = None;
                self.engine.stop_recording_boundary_data();
            }
        });
    }

    #[must_use]
    pub fn add_menu_items(self, ui: &mut Ui) -> MenuResponse {
        puffin::profile_function!();
        let response = self.add_edit_menu(ui);
        let MenuResponse::Noop(mut page) = response else {
            return response;
        };
        page.add_plot_menu(ui);
        MenuResponse::Noop(page)
    }

    fn color_for_component(component: Component) -> Color32 {
        match component {
            Component::XX => Color32::RED,
            Component::XY => ORANGE,
            Component::YX => Color32::YELLOW,
            Component::YY => Color32::LIGHT_GREEN,
        }
    }

    fn instructions(ui: &mut Ui, state: &State) {
        puffin::profile_function!();
        ui.collapsing("Instructions", |ui| {
            if state == &State::Unconfigured {
                ui.label("Configure engine by clicking the button below.");
            }
            ui.label("You can plot stress in an element by hovering over an element and clicking it.\n\
            Similarly, you can plot displacement of a node by hovering over it (it will be highlighted \
            if it's within the vicinity of mouse pointer) and right-clicking the mouse.\n\
            The 'Plot' menu item in the menu bar has options to discard existing plots, and also has an option to \
            select a boundary in order to plot average force / displacement on it. Do note that if it is a force \
            boundary, only average displacement will be plotted and vice-versa, since force and displacement are \
            conjugate pairs. Whereas for a free boundary, both will be plotted. \n\
            You can choose to not display stress by coloring elements, which will result in a faster simulation.\n\
            You can also select which component of the stress should be used to color the triangles by \
            selecting one of the radio buttons in the control panel.\n\
            Hovering over an element will display it's stress component and strain energy at the moment.\n\
            You can zoom in / out by holding down Ctrl and scrolling mouse wheel up / down.\n\
            You can drag the image by holding down primary mouse button and dragging the mouse.");
            if state == &State::Unconfigured {
                return;
            }
            ui.label("You can pause / play and rewind the simulation.\nProgress of simulation will be \
            displayed in the progress bar.\n\
            You can reconfigure the engine by clicking the button in the bottom panel.\n\
            Changing values such as refresh period or duration of the simulation will not \
            cause the simulation to restart.");
        });
    }

    #[must_use]
    pub fn add_contents(mut self, ui: &mut Ui) -> Response {
        puffin::profile_function!();
        self.engine.set_refresh_token(ui.ctx());
        ui.heading("Simulation");

        macro_rules! update_receivers {
            ( $( $receiver:ident ),*) => {
                $(
                    self.receivers
            .$receiver
            .update()
            .expect("Sender should not be dropped");
                )*
            };
        }
        update_receivers!(
            frame_receiver,
            state_receiver,
            config_receiver,
            error_receiver
        );

        Self::instructions(ui, &self.receivers.state_receiver.data);

        bottom_panel::show("simulation_bottom_panel", ui, |ui| self.add_controls(ui));

        macro_rules! frame {
            () => {
                self.receivers.frame_receiver.data.as_ref()
            };
        }

        let element_opt = self
            .selected_element_index
            .and_then(|index| frame!().map(|frame| &frame.data().elements()[index]))
            .and_then(|element| element.stress_time_series().as_series());

        if let Some(series) = element_opt {
            SidePanel::left("simulation_left_plot_panel")
                .show_inside(ui, |ui| self.show_time_series_stress_plot(ui, series));
        }

        macro_rules! right_plot_panel {
            ( $add_contents:expr ) => {
                SidePanel::right("simulation_right_plot_panel").show_inside(ui, $add_contents);
            };
        }
        let node_opt = self
            .selected_vertex_index
            .and_then(|index| frame!().map(|frame| &frame.data().nodes()[index]))
            .and_then(|node| node.position_time_series().as_series());
        let boundary_avg_opt =
            frame!().and_then(|frame| frame.data().boundary_average_data().as_ref());
        match (node_opt, boundary_avg_opt) {
            (None, None) => {}
            (None, Some(data)) => {
                right_plot_panel!(|ui| {
                    self.show_time_series_boundary_average_data_plot(ui, data);
                });
            }
            (Some(series), None) => {
                right_plot_panel!(|ui| { self.show_time_series_displacement_plot(ui, series) });
            }
            (Some(series), Some(data)) => {
                right_plot_panel!(|ui| {
                    let size = ui.available_size();
                    let avg_plot_size = Vec2::new(size.x, size.y * 2.0 / 3.0);
                    ui.allocate_ui(avg_plot_size, |ui| {
                        self.show_time_series_boundary_average_data_plot(ui, data)
                    });
                    ui.separator();
                    self.show_time_series_displacement_plot(ui, series);
                });
            }
        }

        match self.show_preview(ui) {
            FramePreviewResponse::Noop => {}
            FramePreviewResponse::ElementSelected(handle) => self.record_stress_data(handle),
            FramePreviewResponse::VertexSelected(index) => self.record_vertex_position(index),
        }

        if let Some(err) = &self.receivers.error_receiver.data {
            if error_dialog::show(err, ui.ctx()).closed() {
                self.receivers.error_receiver.data = None;
            }
        }

        let config = Self::input_dialog_and_error_ui(
            ui,
            &mut self.config_dialog_state,
            &mut self.configure_error,
        );
        if let Some(config) = config {
            self.engine.configure(config);
        }

        if let Some(state) = &mut self.plot_dialog_state {
            use plot_dialog::Response;
            match plot_dialog::show(ui.ctx(), state) {
                Response::Noop => {}
                Response::Cancel => {
                    self.plot_dialog_state = None;
                }
                Response::BoundaryId(id) => {
                    self.plot_dialog_state = None;
                    if self.selected_boundary_id != Some(id) {
                        self.selected_boundary_id = Some(id);
                        self.engine.record_boundary_data(id);
                    }
                }
            }
        }

        Response::Noop(self)
    }

    fn record_stress_data(&mut self, index: usize) {
        puffin::profile_function!();
        if self.selected_element_index == Some(index) {
            return;
        }
        self.stop_recording_stress_data();
        self.engine.record_stress_data_of_element(index);
        self.selected_element_index = Some(index);
    }

    fn stop_recording_stress_data(&mut self) {
        puffin::profile_function!();
        if let Some(handle) = self.selected_element_index.take() {
            self.engine.stop_recording_stress_data(handle);
        }
    }

    fn record_vertex_position(&mut self, index: usize) {
        puffin::profile_function!();
        if self.selected_vertex_index == Some(index) {
            return;
        }
        self.stop_recording_vertex_position();
        self.engine.record_vertex_position(index);
        self.selected_vertex_index = Some(index);
    }

    fn stop_recording_vertex_position(&mut self) {
        puffin::profile_function!();
        if let Some(index) = self.selected_vertex_index.take() {
            self.engine.stop_recording_vertex_position(index);
        }
    }

    fn add_controls(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        let state = self.receivers.state_receiver.data;

        ui.horizontal(|ui| {
            let label = if state == State::Unconfigured {
                "Configure Engine"
            } else {
                "Reconfigure Engine"
            };
            let response = ui.add_enabled(state != State::Running, Button::new(label));
            if response.clicked() {
                let mesh = self.engine.project_data().state().mesh.clone();
                let current_config = self.receivers.config_receiver.data.as_ref();
                self.config_dialog_state.replace(match current_config {
                    Some(config) => config_dialog::State::new(config, mesh),
                    None => config_dialog::State::default(mesh),
                });
            }
            ui.checkbox(&mut self.show_stress_gradients, "Show stress gradients");
        });

        if self.show_stress_gradients {
            ui.horizontal(|ui| {
                ui.label("Stress component: ");
                Component::iter().for_each(|component| {
                    ui.radio_value(
                        &mut self.stress_tensor_component,
                        component,
                        component.to_string(),
                    )
                    .on_hover_text("Use this component of stress to color elements.");
                });
            });
        }

        if state == State::Unconfigured {
            return;
        }

        let frame = self.receivers.frame_receiver.data.as_ref();

        let opt = frame.and_then(|f| {
            f.total_iterations()
                .map(|total_iterations| (f.iterations(), total_iterations))
        });
        if let Some((iterations, total_iterations)) = opt {
            ui.label(format!("Progress: {iterations}/{total_iterations}"));
        };

        if let Some(runtime) = frame.and_then(|f| *f.runtime()) {
            ui.label(format!("Runtime: {runtime:#?}"));
        }

        let progress = frame.map(|frame| *frame.progress()).unwrap_or_default();
        ui.horizontal(|ui| {
            self.playback_toggle(ui, state);
            if progress > 0.0 {
                let response = ui
                    .button(unicode_symbols::REFRESH)
                    .on_hover_text("Reset simulation.");
                if response.clicked() {
                    self.engine.rewind();
                }
            }
            ui.add(ProgressBar::new(progress).show_percentage());
        });
    }

    fn playback_toggle(&mut self, ui: &mut Ui, state: State) {
        puffin::profile_function!();
        let symbol = if state == State::Running {
            unicode_symbols::PAUSE
        } else if state == State::Paused {
            unicode_symbols::PLAY
        } else {
            return;
        };
        shortcut!(PLAYBACK, Modifiers::NONE, Key::Space);
        let response = ui
            .add(Button::new(symbol).shortcut_text(ui.ctx().format_shortcut(&PLAYBACK_SHORTCUT)))
            .on_hover_text(if state == State::Running {
                "Pause simulation."
            } else {
                "Resume simulation."
            });
        let toggle_playback = response.clicked() || super::consume_shortcut(ui, &PLAYBACK_SHORTCUT);
        if !toggle_playback {
            return;
        }
        if state == State::Running {
            self.engine.pause();
        } else {
            self.engine.play();
        }
    }

    fn plot_series_for_vector(
        series: &[TimeStampedValue<Vector2<f32>>],
        index: usize,
    ) -> Vec<[f64; 2]> {
        puffin::profile_function!();
        series
            .par_iter()
            .map(|value| [*value.time_stamp() as f64, value.value()[index] as f64])
            .collect()
    }

    fn default_open_collapsing_plot<T, I, S>(
        heading: T,
        ui: &mut Ui,
        plot_id: I,
        plot_cursor_group_id: &'static str,
        plot_y_label: &'static str,
        plot_series: S,
        plot_color: Color32,
    ) where
        T: Into<WidgetText>,
        I: Hash,
        S: Fn() -> Vec<[f64; 2]>,
    {
        CollapsingHeader::new(heading)
            .default_open(true)
            .show(ui, |ui| {
                Plot::new(plot_id)
                    .link_cursor(plot_cursor_group_id, true, true)
                    .custom_x_axes(vec![AxisHints::new_x().label("Duration")])
                    .custom_y_axes(vec![AxisHints::new_y().label(plot_y_label)])
                    .show(ui, |ui| {
                        ui.line(Line::new(plot_series()).color(plot_color));
                    })
            });
    }

    fn show_time_series_displacement_plot(
        &self,
        ui: &mut Ui,
        series: &[TimeStampedValue<Vector2<f32>>],
    ) {
        puffin::profile_function!();
        ui.label("Displacement plot");
        macro_rules! id_source {
            ( $comp:literal ) => {
                const_format::formatcp!("simulation_displacement_{}_plot", $comp)
            };
        }
        macro_rules! plot {
            ( $desired_size:expr, $comp:literal, $index:expr, $color:expr ) => {
                ui.allocate_ui($desired_size, |ui| {
                    Self::default_open_collapsing_plot(
                        $comp,
                        ui,
                        id_source!($comp),
                        "simulation_displacement_plot_group",
                        "Displacement",
                        || Self::plot_series_for_vector(series, $index),
                        $color,
                    )
                });
            };
        }
        let size = ui.available_size();
        let desired_size = Vec2::new(size.x, size.y / 2.0 - ui.spacing().item_spacing.y);
        plot!(desired_size, "Dx", 0, Color32::RED);
        plot!(desired_size, "Dy", 1, Color32::YELLOW);
    }

    fn plot_series_for_stress_component(
        series: &[TimeStampedValue<Matrix2<f32>>],
        component: Component,
    ) -> Vec<[f64; 2]> {
        puffin::profile_function!();
        series
            .par_iter()
            .map(|value| {
                [
                    *value.time_stamp() as f64,
                    *value.value().index(component) as f64,
                ]
            })
            .collect()
    }

    fn show_time_series_stress_plot(&self, ui: &mut Ui, series: &[TimeStampedValue<Matrix2<f32>>]) {
        puffin::profile_function!();
        ui.label("Stress plot");
        let size = ui.available_size();
        let desired_size = Vec2::new(size.x, size.y / 4.0 - ui.spacing().item_spacing.y);
        Component::iter().for_each(|component| {
            ui.allocate_ui(desired_size, |ui| {
                Self::default_open_collapsing_plot(
                    format!("E{component}"),
                    ui,
                    format!("simulation_stress_{component}_plot"),
                    "simulation_stress_plot_group",
                    "Stress",
                    || Self::plot_series_for_stress_component(series, component),
                    Self::color_for_component(component),
                )
            });
        });
    }

    fn show_time_series_boundary_average_data_plot(&self, ui: &mut Ui, data: &BoundaryAverage) {
        puffin::profile_function!();
        ui.label("Boundary average plot");
        let size = ui.available_size();
        macro_rules! id_source {
            ( $comp:literal ) => {
                const_format::formatcp!("simulation_boundary_average_{}_plot", $comp)
            };
        }
        macro_rules! vector_series {
            ( $series:expr, $index:expr ) => {
                || {
                    $series
                        .par_iter()
                        .map(|tsv| [*tsv.time_stamp() as f64, tsv.value()[$index] as f64])
                        .collect()
                }
            };
            ($series:expr, $kind:ident, $index:expr) => {
                || {
                    $series
                        .par_iter()
                        .map(|tsv| [*tsv.time_stamp() as f64, tsv.value().$kind()[$index] as f64])
                        .collect()
                }
            };
        }
        macro_rules! vector_plot {
            ($desired_size:expr, $vector_name:literal, $comp:literal, $series:expr, $color:expr) => {
                ui.allocate_ui($desired_size, |ui| {
                    Self::default_open_collapsing_plot(
                        $comp,
                        ui,
                        id_source!($comp),
                        "simulation_boundary_average_plot_group",
                        $vector_name,
                        $series,
                        $color,
                    );
                });
            };
        }
        match data {
            BoundaryAverage::Force(series) => {
                let desired_size = Vec2::new(size.x, size.y / 2.0 - ui.spacing().item_spacing.y);
                vector_plot!(
                    desired_size,
                    "Force",
                    "Fx",
                    vector_series!(series, 0),
                    Color32::RED
                );
                vector_plot!(
                    desired_size,
                    "Force",
                    "Fy",
                    vector_series!(series, 1),
                    Color32::YELLOW
                );
            }
            BoundaryAverage::Displacement(series) => {
                let desired_size = Vec2::new(size.x, size.y / 2.0 - ui.spacing().item_spacing.y);
                vector_plot!(
                    desired_size,
                    "Displacement",
                    "Dx",
                    vector_series!(series, 0),
                    Color32::RED
                );
                vector_plot!(
                    desired_size,
                    "Displacement",
                    "Dy",
                    vector_series!(series, 1),
                    Color32::YELLOW
                );
            }
            BoundaryAverage::ForceAndDisplacement(series) => {
                let desired_size = Vec2::new(size.x, size.y / 4.0 - ui.spacing().item_spacing.y);
                vector_plot!(
                    desired_size,
                    "Force",
                    "Fx",
                    vector_series!(series, force, 0),
                    Color32::RED
                );
                vector_plot!(
                    desired_size,
                    "Force",
                    "Fy",
                    vector_series!(series, force, 1),
                    ORANGE
                );
                vector_plot!(
                    desired_size,
                    "Displacement",
                    "Dx",
                    vector_series!(series, displacement, 0),
                    Color32::YELLOW
                );
                vector_plot!(
                    desired_size,
                    "Displacement",
                    "Dy",
                    vector_series!(series, displacement, 1),
                    Color32::LIGHT_GREEN
                );
            }
        }
    }

    fn show_preview(&self, ui: &mut Ui) -> FramePreviewResponse {
        puffin::profile_function!();
        CentralPanel::default()
            .frame(egui::Frame::default())
            .show_inside(ui, |ui| self.preview_contents(ui))
            .inner
    }

    fn preview_contents(&self, ui: &mut Ui) -> FramePreviewResponse {
        puffin::profile_function!();
        let plot = || {
            plot_utils::plot_without_clutter("simulation_preview_plot")
                .auto_bounds(Vec2b::FALSE)
                .allow_double_click_reset(false)
        };
        macro_rules! plot {
            ( $ui:expr ) => {{
                let polygon_set = self.engine.polygon_data().polygon_set();
                plot_utils::plot_polygon_set($ui, polygon_set, plot_utils::default_transform)
            }};
            ( $ui:expr, $frame:expr ) => {{
                plot!($ui);
                self.plot_frame($ui, $frame)
            }};
        }

        let frame_opt = self.receivers.frame_receiver.data.as_ref();
        let Some(frame) = frame_opt else {
            plot().show(ui, |ui| plot!(ui));
            return FramePreviewResponse::Noop;
        };

        let plot_response = plot().show(ui, |ui| plot!(ui, frame));
        let response = plot_response.response;

        let response = match plot_response.inner {
            FramePlotHoverResponse::Noop => response,
            FramePlotHoverResponse::ElementIndex(index) => response.on_hover_ui_at_pointer(|ui| {
                let element = &frame.data().elements()[index];
                ui.label(Self::format_stress(
                    *element.stress().index(self.stress_tensor_component),
                ));
                ui.label(format!("Strain energy: {:.2}", element.strain_energy()));
                ui.label("Click to plot stress");
            }),
            FramePlotHoverResponse::VertexIndex(index) => response.on_hover_ui_at_pointer(|ui| {
                let node = &frame.data().nodes()[index];
                ui.label(format!(
                    "Position: {:.2}i + {:.2}j",
                    node.position().x,
                    node.position().y
                ));
                ui.label(format!(
                    "Velocity: {:.2}i + {:.2}j",
                    node.velocity().x,
                    node.velocity().y
                ));
                ui.label(format!(
                    "Force: {:.2}i + {:.2}j",
                    node.force().x,
                    node.force().y
                ));
                ui.label("Click to plot displacement");
            }),
        };

        if !response.clicked() {
            return FramePreviewResponse::Noop;
        };

        match plot_response.inner {
            FramePlotHoverResponse::Noop => FramePreviewResponse::Noop,
            FramePlotHoverResponse::ElementIndex(handle) => {
                FramePreviewResponse::ElementSelected(handle)
            }
            FramePlotHoverResponse::VertexIndex(index) => {
                FramePreviewResponse::VertexSelected(index)
            }
        }
    }

    fn p_is_inside_abc(a: &[f32; 2], b: &[f32; 2], c: &[f32; 2], p: [f32; 2]) -> bool {
        puffin::profile_function!();
        let d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
        let ba = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / d;
        let bb = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / d;
        let bc = 1.0 - ba - bb;
        ba >= 0.0 && bb >= 0.0 && bc >= 0.0
    }

    fn format_stress(stress: f32) -> String {
        puffin::profile_function!();
        let stress_abs = stress.abs();
        let sign = if stress_abs == 0.0 || stress.is_sign_positive() {
            char::default()
        } else {
            '-'
        };
        let stress = stress_abs;
        macro_rules! fmt {
            ( $stress:expr, $unit:expr ) => {
                format!("Stress: {sign}{:.2} {}", $stress, $unit)
            };
        }
        if stress >= 9e8 {
            fmt!(stress / 1e9, "GPa")
        } else if stress >= 9e5 {
            fmt!(stress / 1e6, "MPa")
        } else if stress >= 9e2 {
            fmt!(stress / 1e3, "kPa")
        } else {
            fmt!(stress, "Pa")
        }
    }

    fn plot_element(
        ui: &mut PlotUi,
        data: &ExportData,
        index: usize,
        stress: f32,
        min_stress: f32,
        max_stress: f32,
    ) -> FramePlotHoverResponse {
        puffin::profile_function!();
        let color = {
            puffin::profile_scope!("lerp_color");
            let stress_range = max_stress - min_stress;
            let t = if stress_range == 0.0 {
                1.0
            } else {
                (max_stress - stress) / stress_range
            };
            Hsva {
                // lerp h b/w red and blue
                h: egui::lerp(0.0..=(2.0 / 3.0), t),
                s: 1.0,
                v: 1.0,
                a: 1.0,
            }
        };

        let vertices = data.element_vertices(index);

        let hover_response = {
            puffin::profile_scope!("element_hover_response");
            ui.pointer_coordinate()
                .and_then(|coords| {
                    let transform = ui.transform();
                    let pos = transform.position_from_point(&coords);
                    data.elements()[index]
                        .indices()
                        .iter()
                        .copied()
                        .find_map(|index| {
                            let node_plot_point =
                                PlotPoint::from(data.node_position(index).map(|v| v as f64));
                            let plot_point_vec = node_plot_point.to_pos2() - coords.to_pos2();
                            let node_pos = transform.position_from_point(&node_plot_point);
                            let pos_vec = node_pos - pos;
                            (pos_vec.length_sq() < 30.0 && plot_point_vec.length_sq() < 1e-7)
                                .then_some(FramePlotHoverResponse::VertexIndex(index))
                        })
                        .or_else(|| {
                            Self::p_is_inside_abc(
                                &vertices[0],
                                &vertices[1],
                                &vertices[2],
                                [coords.x as f32, coords.y as f32],
                            )
                            .then_some(FramePlotHoverResponse::ElementIndex(index))
                        })
                })
                .unwrap_or_default()
        };

        {
            puffin::profile_scope!("ui_add_polygon");
            ui.polygon(
                Polygon::new(vertices.map(|v| v.map(|f| f as f64)).to_vec())
                    .stroke(Stroke::new(0.35, super::on_primary_color(ui.ctx())))
                    .fill_color(color),
            );
        }

        if let FramePlotHoverResponse::VertexIndex(index) = hover_response {
            puffin::profile_scope!("ui_add_hovered_point");
            ui.points(
                Points::new(vec![data.node_position(index).map(|v| v as f64)])
                    .color(super::on_primary_color(ui.ctx()))
                    .radius(2.0),
            );
        }

        hover_response
    }

    fn plot_frame(&self, ui: &mut PlotUi, frame: &Frame) -> FramePlotHoverResponse {
        puffin::profile_function!();
        let data = frame.data();
        if self.show_stress_gradients {
            self.plot_stress_gradients(ui, data)
        } else {
            self.plot_points(ui, data);
            FramePlotHoverResponse::Noop
        }
    }

    fn plot_stress_gradients(&self, ui: &mut PlotUi, data: &ExportData) -> FramePlotHoverResponse {
        puffin::profile_function!();
        let min_stress = *data.min_stress().index(self.stress_tensor_component);
        let max_stress = *data.max_stress().index(self.stress_tensor_component);
        data.elements()
            .iter()
            .enumerate()
            .filter(|(_, element)| !element.is_broken())
            .map(|(index, element)| {
                Self::plot_element(
                    ui,
                    data,
                    index,
                    *element.stress().index(self.stress_tensor_component),
                    min_stress,
                    max_stress,
                )
            })
            .filter(|response| !matches!(response, FramePlotHoverResponse::Noop))
            .last()
            .unwrap_or_default()
    }

    fn plot_points(&self, ui: &mut PlotUi, data: &ExportData) {
        puffin::profile_function!();
        ui.points(
            Points::new(
                data.nodes()
                    .par_iter()
                    .map(|node| node.position())
                    .copied()
                    .map(|vec| vec.map(|v| v as f64))
                    .map(Into::into)
                    .collect::<Vec<_>>(),
            )
            .radius(1.0)
            .color(super::on_primary_color(ui.ctx())),
        );
    }

    #[must_use]
    fn input_dialog_and_error_ui(
        ui: &mut Ui,
        dialog_state: &mut Option<config_dialog::State>,
        configure_error: &mut Option<String>,
    ) -> Option<Box<Config>> {
        if let Some(err) = configure_error {
            if error_dialog::show(err, ui.ctx()).closed() {
                *configure_error = None;
            }
            return None;
        }

        let mut state = dialog_state.take()?;
        use config_dialog::Response;
        match config_dialog::show(&mut state, ui.ctx()) {
            Response::Noop => {
                *dialog_state = Some(state);
                None
            }
            Response::ConfigResult(result) => match result {
                Ok(config) => Some(config),
                Err(err) => {
                    *dialog_state = Some(state);
                    *configure_error = err.into();
                    None
                }
            },
            Response::Cancel => None,
        }
    }
}

mod serde_impl {
    use super::*;
    use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Debug, Serialize, Deserialize)]
    enum WrappedProjectData {
        WithCpdData(Box<Data<WithCpdExportData>>),
        WithMesh(Data<WithMesh>),
    }

    impl Serialize for Page {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let project_data = self.engine.project_data().clone();
            let project_data = match &self.receivers.frame_receiver.data {
                Some(frame) => WrappedProjectData::WithCpdData(Box::new(
                    project_data.with_export_data(frame.data().clone()),
                )),
                None => WrappedProjectData::WithMesh(project_data),
            };
            project_data.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for Page {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            WrappedProjectData::deserialize(deserializer).and_then(|wpd| {
                match wpd {
                    WrappedProjectData::WithCpdData(project_data) => Page::try_from(*project_data),
                    WrappedProjectData::WithMesh(project_data) => Ok(Page::from(project_data)),
                }
                .map_err(de::Error::custom)
            })
        }
    }
}
