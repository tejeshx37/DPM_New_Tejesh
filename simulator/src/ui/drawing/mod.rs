mod dialog;
mod shape;

use super::{bottom_panel, error_dialog, plot_utils, ContextWrapper};
use crate::model::{
    project::data::{Data, WithShape},
    shape_configurator::{Configurator, Snapshot, State},
    state_channel::{self, Receiver, STReceiver},
};
use cgal::{PolygonSet, PolygonSetInput};
use egui::{Align, Button, CentralPanel, Frame, Key, Modifiers, RadioButton, SidePanel, Ui};
use enum_map::EnumMap;
use shape::Shape;
use std::mem;
use strum::IntoEnumIterator;
use uuid::Uuid;

shortcut!(UNDO, Modifiers::COMMAND, Key::Z);
shortcut!(REDO, Modifiers::COMMAND, Key::Y);

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Mode {
    #[default]
    Join,
    Difference,
}

#[derive(Debug)]
pub enum Response {
    Noop(Page),
    SetBoundaryConditions(Data<WithShape>),
}

#[derive(Debug)]
pub struct Page {
    configurator: Configurator<ContextWrapper>,
    selected_shape: Option<Shape>,
    dialog_states: EnumMap<Shape, Option<dialog::State>>,
    input_error: Option<String>,
    mode: Mode,
    state_receiver: STReceiver<State>,
    error_receiver: Receiver<String, Option<String>>,
    polygon_set: Option<(Uuid, PolygonSet)>,
}

impl Default for Page {
    fn default() -> Self {
        Page::from(Data::default())
    }
}

impl From<Data<WithShape>> for Page {
    fn from(project_data: Data<WithShape>) -> Self {
        let (state_sender, state_receiver) = state_channel::same_type_with_default(1);
        let (error_sender, error_receiver) = state_channel::with_default(1);
        Self {
            configurator: Configurator::new(project_data, state_sender, error_sender)
                .expect("State channel is active right now"),
            selected_shape: None,
            dialog_states: EnumMap::default(),
            input_error: None,
            mode: Mode::default(),
            state_receiver,
            error_receiver,
            polygon_set: None,
        }
    }
}

impl Page {
    fn update_polygon_set(polygon_set: &mut Option<(Uuid, PolygonSet)>, snapshot: &Snapshot) {
        let id_matches = polygon_set
            .as_ref()
            .is_some_and(|(id, _)| snapshot.id().eq(id));
        if id_matches {
            return;
        }
        polygon_set.replace((snapshot.id(), snapshot.polygon_set()));
    }

    fn polygon_set(&self) -> Option<&PolygonSet> {
        self.polygon_set
            .as_ref()
            .map(|(_, polygon_set)| polygon_set)
    }

    pub fn add_menu_items(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        let State::Generated(snapshot) = self
            .state_receiver
            .update_and_get()
            .expect("Sender should not be dropped")
        else {
            return;
        };
        if !snapshot.can_undo() && !snapshot.can_redo() {
            return;
        }
        ui.menu_button("Edit", |ui| {
            let button_with_shortcut_clicked = |ui: &mut Ui, text: &str, shortcut| {
                ui.add(Button::new(text).shortcut_text(ui.ctx().format_shortcut(&shortcut)))
                    .clicked()
            };
            if snapshot.can_undo() && button_with_shortcut_clicked(ui, "Undo", UNDO_SHORTCUT) {
                self.configurator.undo();
                ui.close_menu();
            }
            if snapshot.can_redo() && button_with_shortcut_clicked(ui, "Redo", REDO_SHORTCUT) {
                self.configurator.redo();
                ui.close_menu();
            }
        });
    }

    fn instructions(ui: &mut Ui) {
        ui.collapsing("Instructions", |ui| {
            ui.label(
                "Draw shapes by performing boolean set operations with basic shapes.\n\
            Click on one of the shape buttons to draw a particular shape.\n\
            Union / Difference will merge / cut out new shape from existing one.",
            );
        });
    }

    #[must_use]
    pub fn add_contents(mut self, ui: &mut Ui) -> Response {
        puffin::profile_function!();
        self.configurator.set_refresh_token(ui.ctx());
        ui.heading("Drawing");
        Self::instructions(ui);

        self.state_receiver
            .update()
            .expect("Sender should not be dropped");

        match &self.state_receiver.data {
            State::Processing => {}
            State::Generated(snapshot) => Self::update_polygon_set(&mut self.polygon_set, snapshot),
        }

        enum BottomPanelResponse {
            SetBoundaryConditions(Data<WithShape>),
            Noop(State),
        }

        let response = bottom_panel::show("drawing_bottom_panel", ui, |ui| {
            match mem::take(&mut self.state_receiver.data) {
                State::Processing => BottomPanelResponse::Noop(State::Processing),
                State::Generated(snapshot) => {
                    let num_polygons = snapshot.num_polygons();
                    let response = ui
                        .add_enabled(num_polygons == 1, Button::new("Finish"))
                        .on_hover_text("Click to move on to boundary conditions page.")
                        .on_disabled_hover_text(if num_polygons == 0 {
                            "Cannot finish without any shapes."
                        } else {
                            "There are disjoint polygons, remove or merge them to finish drawing."
                        });
                    if response.clicked() {
                        BottomPanelResponse::SetBoundaryConditions(Data::from(snapshot))
                    } else {
                        BottomPanelResponse::Noop(State::Generated(snapshot))
                    }
                }
            }
        });

        match response.inner {
            BottomPanelResponse::SetBoundaryConditions(project_data) => {
                return Response::SetBoundaryConditions(project_data)
            }
            BottomPanelResponse::Noop(state) => {
                self.state_receiver.data = state;
            }
        }

        SidePanel::left("drawing_controls_panel").show_inside(ui, |ui| self.add_controls(ui));
        CentralPanel::default()
            .frame(Frame::default())
            .show_inside(ui, |ui| Self::add_preview(ui, self.polygon_set()));

        if let Some(err) = &self
            .error_receiver
            .update_and_get()
            .expect("Sender should not be dropped")
        {
            if error_dialog::show(err, ui.ctx()).closed() {
                self.error_receiver.data = None;
            }
        }

        if let Some(err) = self.input_error.as_ref() {
            if error_dialog::show(err, ui.ctx()).closed() {
                self.input_error = None;
            }
            return Response::Noop(self);
        }

        let Some(shape) = self.selected_shape else {
            return Response::Noop(self);
        };
        let state = self.dialog_states[shape].get_or_insert_with(|| dialog::State::from(shape));
        match dialog::show(state, ui.ctx()) {
            dialog::Response::Noop => {}
            dialog::Response::Input(result) => match result {
                Ok(kind) => {
                    let input = match self.mode {
                        Mode::Join => PolygonSetInput::Join(kind),
                        Mode::Difference => PolygonSetInput::Difference(kind),
                    };
                    self.configurator.join_or_diff(input);
                    self.selected_shape = None;
                }
                Err(err) => self.input_error = err.into(),
            },
            dialog::Response::Cancel => self.selected_shape = None,
        }
        Response::Noop(self)
    }

    fn add_controls(&mut self, ui: &mut Ui) {
        ui.vertical_centered_justified(|ui| {
            let opt = ui
                .group(|shape_buttons_ui| {
                    Shape::iter()
                        .filter(|shape| shape_buttons_ui.button(shape.to_string()).clicked())
                        .last()
                })
                .inner;
            if let Some(shape) = opt {
                self.selected_shape = Some(shape);
            }
            let opt = ui
                .group(|ui| {
                    ui.with_layout(ui.layout().with_cross_align(Align::Min), |mode_ui| {
                        Self::add_mode_radio_buttons(mode_ui, self.mode, &self.state_receiver.data)
                    })
                    .inner
                })
                .inner;
            if let Some(mode) = opt {
                self.mode = mode;
            }
            let is_polygon_set_empty = match &self.state_receiver.data {
                State::Processing => return,
                State::Generated(snapshot) => snapshot.num_polygons() == 0,
            };
            if super::consume_shortcut(ui, &UNDO_SHORTCUT) {
                self.configurator.undo();
            }
            if super::consume_shortcut(ui, &REDO_SHORTCUT) {
                self.configurator.redo();
            }
            let reset = !is_polygon_set_empty && ui.button("Clear").clicked();
            if reset {
                self.configurator.reset();
                self.mode = Mode::default();
            }
        });
    }

    #[must_use]
    fn add_mode_radio_buttons(ui: &mut Ui, mode: Mode, state: &State) -> Option<Mode> {
        let mut new_mode = None;
        if ui
            .add(RadioButton::new(mode == Mode::Join, "Union"))
            .clicked()
        {
            new_mode = Some(Mode::Join);
        }
        let can_remove = if let State::Generated(snapshot) = state {
            snapshot.num_polygons() > 0
        } else {
            false
        };
        if ui
            .add_enabled(
                can_remove,
                RadioButton::new(mode == Mode::Difference, "Difference"),
            )
            .clicked()
        {
            new_mode = Some(Mode::Difference);
        }
        new_mode
    }

    fn add_preview(ui: &mut Ui, polygon_set: Option<&PolygonSet>) {
        ui.centered_and_justified(|ui| {
            let mut show_empty_canvas = || {
                egui::Frame::canvas(ui.style()).show(ui, |ui| {
                    ui.label("There is nothing to show here. Try adding some shapes!")
                });
            };
            match polygon_set {
                Some(polygon_set) => {
                    if polygon_set.is_empty() {
                        show_empty_canvas()
                    } else {
                        plot_utils::plot("drawing_plot").show(ui, |ui| {
                            plot_utils::plot_polygon_set(
                                ui,
                                polygon_set,
                                plot_utils::default_transform,
                            );
                        });
                    }
                }
                None => show_empty_canvas(),
            }
        });
    }
}

mod serde_impl {
    use super::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    impl Serialize for Page {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            match &self.state_receiver.data {
                State::Processing => Data::<WithShape>::default().serialize(serializer),
                State::Generated(snapshot) => {
                    Data::<WithShape>::from(snapshot.clone()).serialize(serializer)
                }
            }
        }
    }

    impl<'de> Deserialize<'de> for Page {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            Data::<WithShape>::deserialize(deserializer).map(Page::from)
        }
    }
}
