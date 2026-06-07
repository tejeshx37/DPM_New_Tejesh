use function::Function;
use nalgebra::Vector2;
use std::{iter::Sum, ops::Add};

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub enum Displacement {
    X(Function),
    Y(Function),
    XY(Vector2<Function>),
}

impl Add for Displacement {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::X(x1), Self::X(x2)) => Self::X(x1 + x2),
            (Self::X(x), Self::Y(y)) | (Self::Y(y), Self::X(x)) => Self::XY(Vector2::new(x, y)),
            (Self::X(x), Self::XY(v)) | (Self::XY(v), Self::X(x)) => {
                let [[vx, vy]] = v.data.0;
                Self::XY(Vector2::new(vx + x, vy))
            }
            (Self::Y(y1), Self::Y(y2)) => Self::Y(y1 + y2),
            (Self::Y(y), Self::XY(v)) | (Self::XY(v), Self::Y(y)) => {
                let [[vx, vy]] = v.data.0;
                Self::XY(Vector2::new(vx, vy + y))
            }
            (Self::XY(v1), Self::XY(v2)) => {
                let [[v1x, v1y]] = v1.data.0;
                let [[v2x, v2y]] = v2.data.0;
                Self::XY(Vector2::new(v1x + v2x, v1y + v2y))
            }
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Default)]
pub enum BoundaryCondition {
    #[default]
    Free,
    Force(Vector2<Function>),
    Displacement(Displacement),
}

impl Add for BoundaryCondition {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Free, other) => other,
            (Self::Force(f), Self::Free) => Self::Force(f),
            (Self::Force(f1), Self::Force(f2)) => {
                let [[x1, y1]] = f1.data.0;
                let [[x2, y2]] = f2.data.0;
                Self::Force(Vector2::new(x1 + x2, y1 + y2))
            }
            (Self::Force(_), Self::Displacement(d)) => Self::Displacement(d),
            (Self::Displacement(d1), Self::Displacement(d2)) => Self::Displacement(d1 + d2),
            (Self::Displacement(d), _) => Self::Displacement(d),
        }
    }
}

impl Sum for BoundaryCondition {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(BoundaryCondition::default(), Add::add)
    }
}
