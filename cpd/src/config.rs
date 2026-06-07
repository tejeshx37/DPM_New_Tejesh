use super::material;
use derive_getters::Getters;
use std::time::Duration;
use typed_builder::TypedBuilder;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, TypedBuilder, Getters)]
pub struct Config {
    material_props: material::Props,
    duration: Duration,
    time_delta: Duration,
}

impl Config {
    pub(crate) fn set_duration(&mut self, duration: Duration) {
        self.duration = duration;
    }
}
