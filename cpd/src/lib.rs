mod boundary_average;
pub mod boundary_condition;
mod boundary_info;
pub mod computer;
pub mod config;
mod element;
mod material;
mod node;
mod time_series_value;

pub use boundary_average::{BoundaryAverage, ForceAndDisplacement};
pub(crate) use boundary_info::BoundaryInfo;
use cgal::BoundaryId;
pub use element::Element;
pub use material::{
    isotropic::Props as IsotropicMaterialProps, orthotropic::Props as OrthotropicMaterialProps,
    BulkProps as BulkMaterialProps, ElasticityCondition, FailureCriteria, Props as MaterialProps,
};
pub use node::{Node, NodeData};
pub use time_series_value::{TimeSeriesValue, TimeStampedValue};

use derive_getters::Getters;
use fxhash::FxHashMap;
use nalgebra::Matrix2;

/// Spatial dimension of a DPM project. Selected at project creation time and
/// immutable for the lifetime of the project — shapes, meshes, and physics are
/// dimension-specific so switching mid-project is meaningless.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dimension {
    #[default]
    D2,
    D3,
}

impl Dimension {
    pub fn label(self) -> &'static str {
        match self {
            Dimension::D2 => "2D",
            Dimension::D3 => "3D",
        }
    }
}

/// 2D pipeline — the current production implementation. All existing nodes,
/// elements, materials, boundary conditions, and the time-integration loop
/// live here. New 2D code should reach for these via `cpd::d2::*` rather than
/// the crate root re-exports, which are kept for backward compatibility.
pub mod d2 {
    pub use crate::boundary_condition::{self, BoundaryCondition};
    pub use crate::computer::{self, Computer};
    pub use crate::element::Element;
    pub use crate::node::{Node, NodeData};
    pub use crate::material::{
        isotropic::Props as IsotropicMaterialProps,
        orthotropic::Props as OrthotropicMaterialProps,
        BulkProps as BulkMaterialProps,
        ElasticityCondition,
        FailureCriteria,
        Props as MaterialProps,
    };
    pub use crate::ExportData;
}

/// 3D pipeline — DPM solver for tetrahedral particle stencils. Mirrors the
/// 2D pipeline (`cpd::d2`) but operates on 4-node tets with `Matrix3`
/// deformation gradients and `Vector3` BCs.
pub mod d3;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Getters)]
pub struct ExportData {
    nodes: Box<[Node]>,
    elements: Box<[Element]>,
    boundary_infos: FxHashMap<BoundaryId, BoundaryInfo>,
    #[cfg_attr(feature = "serde", serde(skip))]
    boundary_average_data: Option<BoundaryAverage>,
    config: Option<config::Config>,
    iterations: u128,
    min_stress: Matrix2<f32>,
    max_stress: Matrix2<f32>,
}

impl ExportData {
    pub fn element_vertices(&self, index: usize) -> [[f32; 2]; 3] {
        self.elements[index].indices.map(|i| self.node_position(i))
    }

    pub fn node_position(&self, index: usize) -> [f32; 2] {
        self.nodes[index].position().data.0[0]
    }
}
