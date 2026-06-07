//! 3D pipeline UI — drawing, meshing, and simulation pages for projects
//! created in 3D mode. Meshing and simulation are stubs in this milestone;
//! drawing is wired up to the parametric Shape3D dialogs and the preview
//! viewport.

pub mod drawing;

use serde::{Deserialize, Serialize};

/// Persisted state for the 3D pipeline of a project. Mirrors the structure
/// of the 2D `PageData` but lives behind the dimension toggle. Currently
/// only the drawing-page state is populated; meshing/simulation will join
/// in later steps.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct State {
    #[serde(default)]
    pub drawing: drawing::State,
}
