//! 3D DPM solver — tetrahedral particle stencils, linear isotropic
//! elasticity, explicit velocity-Verlet integration with optional damping.
//!
//! Designed as a direct mirror of the 2D solver so the simulator can drive
//! both pipelines through analogous types: `Node3D` ↔ `Node`, `Element3D`
//! (tet) ↔ `Element` (triangle), `Computer3D` ↔ `Computer`. Feature parity
//! with the 2D solver (orthotropic materials, failure criteria, boundary
//! averages, time-series recording) is intentionally deferred — this
//! module ships the minimum viable physics so users can validate a 3D
//! simulation end-to-end. Those extensions plug in alongside without
//! reshaping the core types.

mod boundary_condition;
mod computer;
mod config;
mod element;
mod material;
mod node;

pub use boundary_condition::{Axis, AxisTimeSeries, BoundaryCondition3D, TimeSeries};
pub use computer::{run_steps, Computer3D, RegionAverages, StressStats};
pub use config::Config3D;
pub use element::Element3D;
pub use material::{
    BulkProps3D, FailureCriteria3D, IsotropicProps3D, MaterialProps3D, OrthotropicProps3D,
};
pub use node::Node3D;
