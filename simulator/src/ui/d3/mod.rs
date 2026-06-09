//! 3D pipeline UI — drawing, meshing, and simulation pages for projects
//! created in 3D mode. Drawing and meshing are wired up; simulation lands
//! in the next milestone alongside the 3D DPM solver.

pub mod boundary_conditions;
pub mod drawing;
pub mod meshing;
pub mod simulation;

use serde::{Deserialize, Serialize};

/// Active page within the 3D pipeline. Stored in the project so the
/// selection persists across sessions. Phase order — Drawing → Meshing
/// → BoundaryConditions → Simulation — matches the 2D pipeline so users
/// see the same flow in both dimensions.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stage {
    #[default]
    Drawing,
    BoundaryConditions,
    Meshing,
    Simulation,
}

impl Stage {
    /// Stage to advance to via the "Next →" button. Returns `None` at
    /// the end of the pipeline.
    pub fn next(self) -> Option<Stage> {
        match self {
            Stage::Drawing => Some(Stage::BoundaryConditions),
            Stage::BoundaryConditions => Some(Stage::Meshing),
            Stage::Meshing => Some(Stage::Simulation),
            Stage::Simulation => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Stage::Drawing => "Drawing",
            Stage::BoundaryConditions => "Boundary Conditions",
            Stage::Meshing => "Meshing",
            Stage::Simulation => "Simulation",
        }
    }
}

/// Persisted state for the 3D pipeline of a project. Mirrors the structure
/// of the 2D `PageData` but lives behind the dimension toggle.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct State {
    #[serde(default)]
    pub stage: Stage,
    #[serde(default)]
    pub drawing: drawing::State,
    #[serde(default)]
    pub meshing: meshing::State,
    #[serde(default)]
    pub simulation: simulation::State,
}
