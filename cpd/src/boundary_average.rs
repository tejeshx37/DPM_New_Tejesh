use super::TimeStampedValue;
use derive_getters::Getters;
use nalgebra::Vector2;
use std::iter::Sum;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Default, Clone, Copy, Getters)]
pub struct ForceAndDisplacement {
    pub(crate) force: Vector2<f32>,
    pub(crate) displacement: Vector2<f32>,
}

impl Sum for ForceAndDisplacement {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(ForceAndDisplacement::default(), |mut sum, value| {
            sum.force += value.force;
            sum.displacement += value.displacement;
            sum
        })
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub enum BoundaryAverage {
    Force(Vec<TimeStampedValue<Vector2<f32>>>),
    Displacement(Vec<TimeStampedValue<Vector2<f32>>>),
    ForceAndDisplacement(Vec<TimeStampedValue<ForceAndDisplacement>>),
}

impl BoundaryAverage {
    pub(crate) fn reset(&mut self) {
        match self {
            BoundaryAverage::Force(series) | BoundaryAverage::Displacement(series) => {
                series.clear();
            }
            BoundaryAverage::ForceAndDisplacement(series) => {
                series.clear();
            }
        }
    }
}
