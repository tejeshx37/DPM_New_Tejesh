use super::{
    close_project_dialog, delete_project_dialog, page::Page, unicode_symbols, ProjectHandle,
};
use crate::{
    model::project::{ClosedHandle, Manager, OpenHandle, UntitledHandle, PROJECT_FILE_EXT},
    ui::{d3, error_dialog},
};
use cpd::Dimension;
use eframe::{CreationContext, Frame};
use egui::{
    menu, Button, CentralPanel, Context, Image, Key, KeyboardShortcut, Modifiers, SidePanel,
    TopBottomPanel, Ui, ViewportCommand, Visuals, WidgetText,
};
use rfd::FileDialog;
use serde::{Deserialize, Serialize};
use std::{mem, ops::DerefMut};
use strum::AsRefStr;

macro_rules! shortcut {
    ( $name:ident, $modifiers:expr, $key:expr) => {
        paste::paste! {
            const [<$name _SHORTCUT>]: egui::KeyboardShortcut = egui::KeyboardShortcut::new($modifiers, $key);
        }
    };
}

shortcut!(NEW_PROJECT, Modifiers::COMMAND, Key::N);
shortcut!(OPEN_PROJECT, Modifiers::COMMAND, Key::O);
shortcut!(OPEN_WORKSPACE, Modifiers::COMMAND, Key::W);
shortcut!(SAVE_PROJECT, Modifiers::COMMAND, Key::S);
shortcut!(
    SAVE_AS_PROJECT,
    Modifiers::COMMAND.plus(Modifiers::SHIFT),
    Key::S
);
shortcut!(TOGGLE_PROJECT_PANEL, Modifiers::COMMAND, Key::P);

#[derive(Debug, Serialize, Deserialize)]
struct PageData {
    page: Page,
    show_disjoint_dialog: bool,
    /// Project dimension. Defaults to 2D so projects saved before this field
    /// existed deserialize correctly. Immutable once a project is created.
    #[serde(default)]
    dimension: Dimension,
    /// State for the 3D pipeline. Empty for 2D projects.
    #[serde(default)]
    d3_state: d3::State,
}

impl Default for PageData {
    fn default() -> Self {
        Self {
            page: Page::drawing(),
            show_disjoint_dialog: false,
            dimension: Dimension::D2,
            d3_state: d3::State::default(),
        }
    }
}

impl PageData {
    fn new_with_dimension(dimension: Dimension) -> Self {
        Self {
            dimension,
            ..Self::default()
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Theme {
    Light,
    #[default]
    Dark,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct App {
    project_manager: Manager<PageData>,
    project_handle: ProjectHandle,
    close_project_dialog: Option<close_project_dialog::State>,
    delete_project_dialog: Option<delete_project_dialog::State>,
    error: Option<String>,
    theme: Theme,
    show_project_panel: bool,
}

impl Default for App {
    fn default() -> Self {
        let manager = Manager::default().expect("Unable to create project manager");
        let project_handle = Self::next_untitled_or_open_project(&manager);
        Self {
            project_handle,
            project_manager: manager,
            close_project_dialog: None,
            delete_project_dialog: None,
            error: None,
            theme: Theme::default(),
            show_project_panel: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, AsRefStr)]
enum LabelButton {
    Save,
    Close,
    Delete,
}

impl LabelButton {
    const fn symbol(&self) -> &'static str {
        match self {
            LabelButton::Save => unicode_symbols::FILE,
            LabelButton::Close => unicode_symbols::CROSS,
            LabelButton::Delete => unicode_symbols::TRASH_CAN,
        }
    }
}

struct LabelResponse<H, const N: usize> {
    project_handle: H,
    selected: bool,
    buttons: [(LabelButton, bool); N],
}

impl<H, const N: usize> LabelResponse<H, N> {
    fn has_some_value(&self) -> bool {
        self.selected || self.buttons.iter().any(|(_, clicked)| *clicked)
    }

    fn is_button_clicked(&self, button: LabelButton) -> bool {
        self.buttons
            .iter()
            .find_map(|(b, c)| (button == *b).then_some(*c))
            .unwrap_or_default()
    }
}

fn project_label<H, const N: usize>(
    ui: &mut Ui,
    project_handle: H,
    is_selected: bool,
    text: impl Into<WidgetText>,
    buttons: [LabelButton; N],
) -> LabelResponse<H, N> {
    ui.horizontal(|ui| LabelResponse {
        project_handle,
        selected: ui.selectable_label(is_selected, text).clicked(),
        buttons: buttons.map(|button| {
            (
                button,
                ui.button(button.symbol())
                    .on_hover_text(button.as_ref())
                    .clicked(),
            )
        }),
    })
    .inner
}

macro_rules! delete_project {
    ($($type:ident),*) => {
        paste::paste! {
            $(
                fn [<delete_ $type:lower _project>](&mut self, handle: &[<$type Handle>]) {
                    let needs_handle_update = self.project_handle == ProjectHandle::$type(*handle);
                    let result = self.project_manager.[<delete_ $type:lower _project>](handle);
                    match result {
                        Ok(()) => {
                            if needs_handle_update {
                                self.project_handle =
                                    Self::next_untitled_or_open_project(&self.project_manager);
                            }
                        }
                        Err(err) => self.error = Some(err.to_string()),
                    }
                }
            )*
        }
    };
}

impl App {
    pub fn new(cc: &CreationContext<'_>) -> Self {
        puffin::profile_function!();
        // Register the shared wgpu scene renderer so 3D viewport paint
        // callbacks can locate it via `egui_wgpu::CallbackResources`.
        // Safe no-op if the wgpu backend isn't active (e.g. test contexts).
        if let Some(render_state) = cc.wgpu_render_state.as_ref() {
            let scene = d3::drawing::viewport::wgpu_scene::Scene::new(
                &render_state.device,
                render_state.target_format,
            );
            render_state
                .renderer
                .write()
                .callback_resources
                .insert(scene);
        }
        cc.storage
            .and_then(|storage| eframe::get_value(storage, eframe::APP_KEY))
            .unwrap_or_default()
    }

    fn add_menu(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| self.add_file_menu_items(ui));
            ui.menu_button("View", |ui| self.add_view_menu_items(ui));
            if let Some(page_data) = self.page_data() {
                let dimension = page_data.dimension;
                // 2D pipeline gets the existing per-page menu items; 3D has
                // no page-specific menu yet (drawing-only milestone).
                if matches!(dimension, Dimension::D2) {
                    page_data.page = mem::take(&mut page_data.page).add_menu_items(ui);
                }
                ui.separator();
                ui.label(format!("Mode: {}", dimension.label()));
            }
            ui.centered_and_justified(|ui| {
                ui.label(format!("Workspace - {}", self.project_manager.workspace()));
            });
        });
    }

    fn update_theme(&mut self, ctx: &Context) {
        ctx.set_visuals(match self.theme {
            Theme::Light => Visuals::light(),
            Theme::Dark => Visuals::dark(),
        })
    }

    fn add_file_menu_items(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        self.add_new_project_menu_button(ui);
        self.add_open_project_menu_button(ui);
        self.add_open_workspace_menu_button(ui);
        self.add_open_recents_menu_button(ui);
        self.add_save_menu_buttons(ui);
        self.add_clear_data_menu_button(ui);
        if ui.button("Quit").clicked() {
            ui.ctx().send_viewport_cmd(ViewportCommand::Close);
        }
    }

    fn button_with_shortcut_clicked(
        ui: &mut Ui,
        text: impl Into<WidgetText>,
        shortcut: &KeyboardShortcut,
    ) -> bool {
        ui.add(Button::new(text).shortcut_text(ui.ctx().format_shortcut(shortcut)))
            .clicked()
    }

    fn add_new_project_menu_button(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        if Self::button_with_shortcut_clicked(ui, "New 2D Project", &NEW_PROJECT_SHORTCUT) {
            ui.close_menu();
            self.create_untitled_project_with_dimension(Dimension::D2);
        }
        if ui.button("New 3D Project").clicked() {
            ui.close_menu();
            self.create_untitled_project_with_dimension(Dimension::D3);
        }
    }

    fn create_untitled_project(&mut self) {
        self.create_untitled_project_with_dimension(Dimension::D2);
    }

    fn create_untitled_project_with_dimension(&mut self, dimension: Dimension) {
        puffin::profile_function!();
        self.project_manager.create_untitled_project();
        let handle = self
            .project_manager
            .untitled_project_handles()
            .last()
            .expect("At least one untitled project handle should exist");
        // Override the freshly-defaulted PageData to record the chosen
        // dimension. PageData::default() gives D2; for D3 projects we swap
        // in a fresh PageData with the dimension flag set.
        if matches!(dimension, Dimension::D3) {
            use std::ops::DerefMut;
            let project = &mut self.project_manager[handle];
            *project.state_mut().deref_mut() = PageData::new_with_dimension(Dimension::D3);
        }
        self.project_handle = ProjectHandle::Untitled(handle);
    }

    fn add_open_project_menu_button(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        if !Self::button_with_shortcut_clicked(ui, "Open Project", &OPEN_PROJECT_SHORTCUT) {
            return;
        }
        ui.close_menu();
        self.open_project();
    }

    fn open_project(&mut self) {
        puffin::profile_function!();
        let opt = FileDialog::new()
            .add_filter("Project", &[PROJECT_FILE_EXT.to_string_lossy()])
            .set_directory(self.project_manager.workspace().path())
            .pick_file();
        let Some(path) = opt else {
            return;
        };
        match self.project_manager.open_project(path) {
            Ok(handle) => self.project_handle = ProjectHandle::Open(handle),
            Err(err) => self.error = Some(err.to_string()),
        }
    }

    fn add_open_workspace_menu_button(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        if !Self::button_with_shortcut_clicked(ui, "Open Workspace", &OPEN_WORKSPACE_SHORTCUT) {
            return;
        }
        ui.close_menu();
        self.open_workspace();
    }

    fn open_workspace(&mut self) {
        puffin::profile_function!();
        let opt = FileDialog::new()
            .set_directory(self.project_manager.workspace().path())
            .pick_folder();
        let Some(path) = opt else {
            return;
        };
        if let Err(err) = self.project_manager.set_workspace(path) {
            self.error = Some(err.to_string());
        }
    }

    fn add_open_recents_menu_button(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        if !self.project_manager.has_recent_projects()
            && !self.project_manager.has_recent_workspaces()
        {
            return;
        }
        ui.menu_button("Open Recent", |ui| {
            self.add_recent_projects_in_menu(ui);
            self.add_recent_workspaces_in_menu(ui);
            if !ui.button("Clear Recents").clicked() {
                return;
            }
            ui.close_menu();
            self.project_manager.clear_recents();
        });
    }

    fn add_recent_projects_in_menu(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        if !self.project_manager.has_recent_projects() {
            return;
        }
        let opt = self
            .project_manager
            .recent_project_handles()
            .filter_map(|handle| {
                ui.button(self.project_manager[handle].name().to_string_lossy())
                    .clicked()
                    .then_some(*handle)
            })
            .next();
        if let Some(handle) = opt {
            ui.close_menu();
            match self.project_manager.open_recent_project(&handle) {
                Ok(handle) => self.project_handle = ProjectHandle::Open(handle),
                Err(err) => self.error = Some(err.to_string()),
            }
            return;
        };
        ui.separator();
    }

    fn add_recent_workspaces_in_menu(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        if !self.project_manager.has_recent_workspaces() {
            return;
        }
        let opt = self
            .project_manager
            .recent_workspaces()
            .filter_map(|handle| {
                ui.button(self.project_manager[handle].to_string())
                    .clicked()
                    .then_some(*handle)
            })
            .next();
        if let Some(handle) = opt {
            ui.close_menu();
            match self.project_manager.open_recent_workspace(&handle) {
                Ok(()) => {}
                Err(err) => self.error = Some(err.to_string()),
            }
            return;
        };
        ui.separator();
    }

    fn add_save_menu_buttons(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        match self.project_handle {
            ProjectHandle::Open(handle) => {
                if Self::button_with_shortcut_clicked(ui, "Save", &SAVE_PROJECT_SHORTCUT) {
                    ui.close_menu();
                    self.save_open_project(&handle)
                }

                if Self::button_with_shortcut_clicked(
                    ui,
                    const_format::formatcp!("Save As{}", unicode_symbols::ELLIPSIS),
                    &SAVE_AS_PROJECT_SHORTCUT,
                ) {
                    ui.close_menu();
                    self.save_open_project_at_path(&handle);
                }
            }
            ProjectHandle::Untitled(handle) => {
                if Self::button_with_shortcut_clicked(ui, "Save", &SAVE_PROJECT_SHORTCUT) {
                    ui.close_menu();
                    self.save_untitled_project(&handle);
                }
            }
            _ => {}
        }
    }

    fn add_view_menu_items(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        self.add_theme_menu_button(ui);
        self.add_project_panel_toggle(ui);
    }

    fn add_theme_menu_button(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        ui.menu_button("Theme", |ui| {
            let theme = self.theme;
            ui.selectable_value(
                &mut self.theme,
                Theme::Light,
                const_format::formatcp!("{} Light mode", unicode_symbols::SUN),
            );
            ui.selectable_value(
                &mut self.theme,
                Theme::Dark,
                const_format::formatcp!("{} Dark mode", unicode_symbols::MOON),
            );
            if self.theme != theme {
                self.update_theme(ui.ctx());
            }
        });
    }

    fn add_project_panel_toggle(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        if !self.has_projects_to_show() {
            return;
        }
        let response = ui.add(
            Button::new(if self.show_project_panel {
                "Hide project panel"
            } else {
                "Show project pane"
            })
            .shortcut_text(ui.ctx().format_shortcut(&TOGGLE_PROJECT_PANEL_SHORTCUT)),
        );
        if response.clicked() {
            self.toggle_project_panel();
        }
    }

    fn toggle_project_panel(&mut self) {
        self.show_project_panel = !self.show_project_panel;
    }

    fn save_open_project(&mut self, handle: &OpenHandle) {
        puffin::profile_function!();
        match self
            .project_manager
            .get_open_project_mut(handle)
            .expect("Handle is valid")
            .save()
        {
            Ok(()) => {}
            Err(err) => {
                self.error = Some(err.to_string());
            }
        }
    }

    fn save_open_project_at_path(&mut self, handle: &OpenHandle) {
        puffin::profile_function!();
        let opt = FileDialog::new()
            .add_filter("Project", &[PROJECT_FILE_EXT.to_string_lossy()])
            .set_directory(self.project_manager.workspace().path())
            .set_file_name(self.project_manager[handle].name().to_string_lossy())
            .save_file();
        let Some(path) = opt else {
            return;
        };
        let needs_handle_update = self.project_handle == ProjectHandle::Open(*handle);
        match self.project_manager.save_open_project_in_path(handle, path) {
            Ok(handle) => {
                if needs_handle_update {
                    self.project_handle = ProjectHandle::Open(handle);
                }
            }
            Err(err) => {
                self.error = Some(err.to_string());
            }
        }
    }

    fn save_untitled_project(&mut self, handle: &UntitledHandle) -> Option<OpenHandle> {
        puffin::profile_function!();
        let opt = FileDialog::new()
            .add_filter("Project", &[PROJECT_FILE_EXT.to_string_lossy()])
            .set_directory(self.project_manager.workspace().path())
            .add_filter("Project", &[PROJECT_FILE_EXT.to_string_lossy()])
            .save_file();
        let path = opt?;
        let needs_handle_update = self.project_handle == ProjectHandle::Untitled(*handle);
        match self.project_manager.save_untitled_project(handle, path) {
            Ok(handle) => {
                if needs_handle_update {
                    self.project_handle = ProjectHandle::Open(handle);
                }
                Some(handle)
            }
            Err(err) => {
                self.error = Some(err.to_string());
                None
            }
        }
    }

    fn add_clear_data_menu_button(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        if !ui.button("Clear Data").clicked() {
            return;
        }
        ui.close_menu();
        *self = App::default();
    }

    fn open_closed_project(&mut self, handle: &ClosedHandle) {
        puffin::profile_function!();
        match self.project_manager.open_closed_project(handle) {
            Ok(handle) => self.project_handle = ProjectHandle::Open(handle),
            Err(err) => self.error = Some(err.to_string()),
        }
    }

    fn refresh_workspace(&mut self) {
        puffin::profile_function!();
        if let Err(err) = self.project_manager.refresh_workspace() {
            self.error = Some(err.to_string());
        }
    }

    fn add_projects(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        ui.horizontal(|ui| {
            ui.label("Projects");
            if ui
                .button(unicode_symbols::REFRESH)
                .on_hover_text("Reload Workspace")
                .clicked()
            {
                self.refresh_workspace();
            }
        });
        ui.separator();
        self.add_untitled_projects(ui);
        self.add_open_projects(ui);
        self.add_closed_projects(ui);
    }

    fn add_untitled_projects(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        let opt = self
            .project_manager
            .untitled_project_handles()
            .map(|handle| {
                project_label(
                    ui,
                    handle,
                    self.project_handle == ProjectHandle::Untitled(handle),
                    self.project_manager[handle].name().to_string_lossy(),
                    [LabelButton::Save, LabelButton::Close],
                )
            })
            .find(LabelResponse::has_some_value);
        let Some(response) = opt else {
            return;
        };
        if response.selected {
            self.project_handle = ProjectHandle::Untitled(response.project_handle);
        }
        if response.is_button_clicked(LabelButton::Close) {
            self.close_project_dialog = Some(close_project_dialog::State::new(
                self.project_manager[response.project_handle].name().clone(),
                ProjectHandle::Untitled(response.project_handle),
            ));
        } else if response.is_button_clicked(LabelButton::Save) {
            self.save_untitled_project(&response.project_handle);
        }
    }

    fn add_open_projects(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        let opt = self
            .project_manager
            .open_project_handles()
            .map(|handle| {
                project_label(
                    ui,
                    *handle,
                    self.project_handle == ProjectHandle::Open(*handle),
                    self.project_manager[handle].name().to_string_lossy(),
                    [LabelButton::Save, LabelButton::Close, LabelButton::Delete],
                )
            })
            .find(LabelResponse::has_some_value);
        let Some(response) = opt else {
            return;
        };
        if response.selected {
            self.project_handle = ProjectHandle::Open(response.project_handle);
        }
        if response.is_button_clicked(LabelButton::Delete) {
            self.delete_project_dialog = Some(delete_project_dialog::State::new(
                self.project_manager[&response.project_handle]
                    .name()
                    .clone(),
                ProjectHandle::Open(response.project_handle),
            ));
        } else if response.is_button_clicked(LabelButton::Close) {
            self.close_project_dialog = Some(close_project_dialog::State::new(
                self.project_manager[&response.project_handle]
                    .name()
                    .clone(),
                ProjectHandle::Open(response.project_handle),
            ));
        } else if response.is_button_clicked(LabelButton::Save) {
            self.save_open_project(&response.project_handle);
        }
    }

    fn add_closed_projects(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        let opt = self
            .project_manager
            .closed_project_handles()
            .map(|handle| {
                project_label(
                    ui,
                    *handle,
                    self.project_handle == ProjectHandle::Closed(*handle),
                    self.project_manager[handle].name().to_string_lossy(),
                    [LabelButton::Delete],
                )
            })
            .find(LabelResponse::has_some_value);
        let Some(response) = opt else {
            return;
        };
        if response.is_button_clicked(LabelButton::Delete) {
            self.delete_project_dialog = Some(delete_project_dialog::State::new(
                self.project_manager[&response.project_handle]
                    .name()
                    .clone(),
                ProjectHandle::Closed(response.project_handle),
            ));
        } else if response.selected {
            self.open_closed_project(&response.project_handle);
        }
    }

    fn next_untitled_or_open_project<D>(project_manager: &Manager<D>) -> ProjectHandle
    where
        D: Default + Serialize + for<'de> Deserialize<'de>,
    {
        project_manager
            .untitled_project_handles()
            .map(ProjectHandle::Untitled)
            .chain(
                project_manager
                    .open_project_handles()
                    .copied()
                    .map(ProjectHandle::Open),
            )
            .next()
            .unwrap_or_default()
    }

    fn add_welcome(&self, ui: &mut Ui) {
        puffin::profile_function!();
        ui.vertical_centered_justified(|ui| {
            ui.add_space(3.0 * ui.max_rect().height() / 10.0);
            ui.add(
                Image::new(egui::include_image!("../assets/simulator-icon.svg"))
                    .tint(ui.visuals().text_color())
                    .maintain_aspect_ratio(true)
                    .max_width(160.0),
            );
            let mut shortcut_info = |info: &str, shortcut| {
                ui.heading(format!("{info} - {}", ui.ctx().format_shortcut(&shortcut)));
            };
            shortcut_info("Create a new project", NEW_PROJECT_SHORTCUT);
            shortcut_info("Open an existing project", OPEN_PROJECT_SHORTCUT);
            shortcut_info("Choose a workspace", OPEN_WORKSPACE_SHORTCUT);
        });
    }

    fn close_open_project(&mut self, handle: &OpenHandle) {
        puffin::profile_function!();
        self.project_manager.close_open_project(handle);
        self.project_handle = Self::next_untitled_or_open_project(&self.project_manager);
    }

    delete_project!(Open, Closed);

    fn show_close_project_dialog(&mut self, ctx: &Context) {
        puffin::profile_function!();
        let Some(state) = &mut self.close_project_dialog else {
            return;
        };
        use close_project_dialog::Response;
        match close_project_dialog::show(state, ctx) {
            Response::Noop => {}
            Response::Save(handle) => match handle {
                ProjectHandle::Invalid | ProjectHandle::Recent(_) | ProjectHandle::Closed(_) => {
                    unreachable!()
                }
                ProjectHandle::Open(handle) => {
                    self.close_project_dialog = None;
                    self.close_open_project(&handle);
                }
                ProjectHandle::Untitled(handle) => {
                    self.close_project_dialog = None;
                    if let Some(handle) = self.save_untitled_project(&handle) {
                        self.close_open_project(&handle);
                    }
                }
            },
            Response::Discard(handle) => match handle {
                ProjectHandle::Invalid | ProjectHandle::Recent(_) | ProjectHandle::Closed(_) => {
                    unreachable!()
                }
                ProjectHandle::Open(handle) => {
                    self.close_project_dialog = None;
                    self.close_open_project(&handle);
                }
                ProjectHandle::Untitled(handle) => {
                    self.close_project_dialog = None;
                    self.project_manager.discard_untitled_project(&handle);
                    self.project_handle =
                        Self::next_untitled_or_open_project(&self.project_manager);
                }
            },
            Response::Cancel => {
                self.close_project_dialog = None;
            }
        }
    }

    fn show_delete_project_dialog(&mut self, ctx: &Context) {
        puffin::profile_function!();
        let Some(state) = &mut self.delete_project_dialog else {
            return;
        };
        use delete_project_dialog::Response;
        match delete_project_dialog::show(state, ctx) {
            Response::Noop => {}
            Response::Delete(handle) => match handle {
                ProjectHandle::Invalid | ProjectHandle::Recent(_) | ProjectHandle::Untitled(_) => {
                    unreachable!()
                }
                ProjectHandle::Open(handle) => {
                    self.delete_project_dialog = None;
                    self.delete_open_project(&handle);
                }
                ProjectHandle::Closed(handle) => {
                    self.delete_project_dialog = None;
                    self.delete_closed_project(&handle);
                }
            },
            Response::Cancel => {
                self.delete_project_dialog = None;
            }
        }
    }

    fn add_shortcuts(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        macro_rules! shortcut {
            ( $shortcut:expr, $command:expr ) => {
                if super::consume_shortcut(ui, &$shortcut) {
                    $command;
                }
            };
        }
        shortcut!(NEW_PROJECT_SHORTCUT, self.create_untitled_project());
        shortcut!(OPEN_PROJECT_SHORTCUT, self.open_project());
        shortcut!(OPEN_WORKSPACE_SHORTCUT, self.open_workspace());

        match self.project_handle {
            ProjectHandle::Open(handle) => {
                shortcut!(SAVE_PROJECT_SHORTCUT, self.save_open_project(&handle));
                shortcut!(
                    SAVE_AS_PROJECT_SHORTCUT,
                    self.save_open_project_at_path(&handle)
                );
            }
            ProjectHandle::Untitled(handle) => {
                shortcut!(SAVE_PROJECT_SHORTCUT, {
                    self.save_untitled_project(&handle);
                });
            }
            _ => {}
        }

        shortcut!(TOGGLE_PROJECT_PANEL_SHORTCUT, self.toggle_project_panel());
    }

    fn page_data(&mut self) -> Option<&mut PageData> {
        match self.project_handle {
            ProjectHandle::Open(handle) => Some(
                self.project_manager
                    .get_open_project_mut(&handle)
                    .expect("Handle is valid")
                    .state_mut()
                    .deref_mut(),
            ),
            ProjectHandle::Untitled(handle) => {
                Some(self.project_manager[handle].state_mut().deref_mut())
            }
            _ => None,
        }
    }

    fn add_contents(&mut self, ui: &mut Ui) {
        puffin::profile_function!();
        let Some(page_data) = self.page_data() else {
            ui.centered_and_justified(|ui| {
                ui.heading("Select a project from the side panel!");
            });
            return;
        };

        match page_data.dimension {
            Dimension::D2 => {
                if page_data.show_disjoint_dialog
                    && error_dialog::show(
                        "There are disjoint shapes, please merge them or remove",
                        ui.ctx(),
                    )
                    .closed()
                {
                    page_data.show_disjoint_dialog = false;
                }

                ui.vertical_centered_justified(|ui| {
                    page_data.page = mem::take(&mut page_data.page).add_contents(ui);
                });
            }
            Dimension::D3 => {
                add_d3_stage_bar(&mut page_data.d3_state.stage, ui);
                match page_data.d3_state.stage {
                    d3::Stage::Drawing => {
                        d3::drawing::show(&mut page_data.d3_state.drawing, ui);
                    }
                    d3::Stage::BoundaryConditions => {
                        let geometry = page_data.d3_state.drawing.geometry.clone();
                        d3::boundary_conditions::show(
                            &mut page_data.d3_state.simulation,
                            &geometry,
                            ui,
                        );
                    }
                    d3::Stage::Meshing => {
                        let geometry = page_data.d3_state.drawing.geometry.clone();
                        let region_bcs = page_data.d3_state.simulation.region_bcs.clone();
                        d3::meshing::show(
                            &mut page_data.d3_state.meshing,
                            &geometry,
                            &region_bcs,
                            ui,
                        );
                    }
                    d3::Stage::Simulation => {
                        let meshes = page_data.d3_state.meshing.meshes.clone();
                        d3::simulation::show(
                            &mut page_data.d3_state.simulation,
                            &meshes,
                            ui,
                        );
                    }
                }
            }
        }
    }

}

fn add_d3_stage_bar(stage: &mut d3::Stage, ui: &mut Ui) {
    ui.horizontal(|ui| {
        ui.selectable_value(stage, d3::Stage::Drawing, "Drawing");
        ui.selectable_value(
            stage,
            d3::Stage::BoundaryConditions,
            "Boundary Conditions",
        );
        ui.selectable_value(stage, d3::Stage::Meshing, "Meshing");
        ui.selectable_value(stage, d3::Stage::Simulation, "Simulation");
        ui.separator();
        if let Some(next) = stage.next() {
            if ui
                .button(format!("Next → {}", next.label()))
                .on_hover_text("Advance to the next pipeline phase")
                .clicked()
            {
                *stage = next;
            }
        } else {
            ui.label("(end of pipeline)");
        }
    });
    ui.separator();
}

impl App {
    fn has_projects_to_show(&self) -> bool {
        self.project_manager.has_untitled_projects()
            || self.project_manager.has_open_projects()
            || self.project_manager.has_closed_projects()
    }
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        puffin::profile_function!();
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        puffin::profile_function!();
        egui_extras::install_image_loaders(ctx);
        let change_theme = ctx.style().visuals.dark_mode && self.theme != Theme::Dark;
        if change_theme {
            self.update_theme(ctx);
        }
        self.show_close_project_dialog(ctx);
        self.show_delete_project_dialog(ctx);
        TopBottomPanel::top("top_panel").show(ctx, |ui| self.add_menu(ui));
        if self.has_projects_to_show() && self.show_project_panel {
            SidePanel::left("projects_panel").show(ctx, |ui| self.add_projects(ui));
        }
        CentralPanel::default().show(ctx, |ui| {
            self.add_shortcuts(ui);

            if self.has_projects_to_show() {
                self.add_contents(ui);
            } else {
                self.add_welcome(ui);
            }

            self.error = self
                .error
                .take()
                .filter(|err| !error_dialog::show(err, ui.ctx()).closed());
        });
    }
}
