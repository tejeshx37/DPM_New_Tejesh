#[macro_use]
pub mod app;
mod always_open_window;
pub mod boundary_conditions;
mod close_project_dialog;
pub mod d3;
mod delete_project_dialog;
mod dialog_utils;
pub mod drawing;
pub mod meshing;
pub mod simulation;
pub mod unicode_symbols;

/// Logical namespace alias for the 2D pipeline pages, matching the
/// `cpd::d2` / `mesh::d2` / `simulator::ui::d3` pattern (F20). Pages
/// remain at their existing paths so the dozens of internal `super::`
/// references in the 2D modules don't need to change; this module just
/// makes `crate::ui::d2::drawing`, `crate::ui::d2::meshing`, etc.
/// resolve so downstream code can write dimension-symmetric paths.
pub mod d2 {
    #[allow(unused_imports)]
    pub use super::{boundary_conditions, drawing, meshing, simulation};
}

mod project_handle;
use ecolor::Color32;
use egui::{Context, KeyboardShortcut, Ui};
pub use project_handle::ProjectHandle;

pub mod page;

pub mod error_dialog;

mod context_wrapper;
pub use context_wrapper::ContextWrapper;

mod plot_utils;

mod bottom_panel;

pub fn is_dark_mode(ctx: &Context) -> bool {
    ctx.style().visuals.dark_mode
}

pub fn on_primary_color(ctx: &Context) -> Color32 {
    if is_dark_mode(ctx) {
        Color32::WHITE
    } else {
        Color32::BLACK
    }
}

pub fn consume_shortcut(ui: &mut Ui, shortcut: &KeyboardShortcut) -> bool {
    ui.input_mut(|w| w.consume_shortcut(shortcut))
}
