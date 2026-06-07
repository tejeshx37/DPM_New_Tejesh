use cgal::{curve::Curve, triangulation, BoundaryId, PolygonWithHoles};
use derive_getters::Getters;
use fxhash::{FxHashMap, FxHashSet};
use nalgebra::Vector2;
use rayon::prelude::*;
use std::iter;

const ASPECT_BOUND: f64 = 0.125;

pub type PointIdxToIdsMap = FxHashMap<usize, FxHashSet<BoundaryId>>;
pub type BoundaryIdToCountMap = FxHashMap<BoundaryId, FxHashSet<usize>>;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub enum Constraint {
    Line([Vector2<f64>; 2]),
    PolyLine(Box<[[Vector2<f64>; 2]]>),
}

impl Constraint {
    fn create(curve: &Curve, num_points: f64, total_perimeter: f64) -> Self {
        match curve {
            Curve::Line(line) => Constraint::Line([
                line.end_points().start().into(),
                line.end_points().end().into(),
            ]),
            Curve::Ellipse(_) => {
                let split_count = (curve.length() * num_points) / total_perimeter;
                let generated_points = curve.split(split_count as u32);
                Constraint::PolyLine(
                    generated_points
                        .iter()
                        .map(Into::into)
                        .zip(generated_points.iter().skip(1).map(Into::into))
                        .map(|(a, b)| [a, b])
                        .collect(),
                )
            }
        }
    }

    fn contains_point(&self, q: &Vector2<f64>) -> bool {
        match self {
            Constraint::Line(arr) => is_on_same_segment(&arr[0], q, &arr[1]),
            Constraint::PolyLine(boxed) => {
                boxed.par_iter().any(|[p, r]| is_on_same_segment(p, q, r))
            }
        }
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &[Vector2<f64>; 2]> + '_> {
        match self {
            Constraint::Line(arr) => Box::new(iter::once(arr)),
            Constraint::PolyLine(boxed) => Box::new(boxed.iter()),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Getters)]
pub struct Mesh {
    triangulation_data: triangulation::Data,
    constraints: Box<[(BoundaryId, Constraint)]>,
    point_id_map: PointIdxToIdsMap,
    boundary_point_map: BoundaryIdToCountMap,
    smallest_side_length: f64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum State {
    Init,
    GeneratingConstraints,
    Triangulating,
    GeneratingAssociativeData,
    FindingSmallestEdge,
    Done,
}

#[derive(Default)]
pub enum Callback<'a> {
    Some(Box<dyn FnMut(State) + 'a>),
    #[default]
    None,
}

impl<'a> Callback<'a> {
    fn invoke(&mut self, state: State) {
        match self {
            Callback::Some(f) => f(state),
            Callback::None => {}
        }
    }
}

impl<'a, F> From<F> for Callback<'a>
where
    F: FnMut(State) + 'a,
{
    fn from(value: F) -> Self {
        Self::Some(Box::new(value))
    }
}

impl Mesh {
    pub fn generate(
        polygon: &PolygonWithHoles,
        num_points: u32,
        size_bound_override: Option<f64>,
        mut state_callback: Callback,
    ) -> Result<Self, String> {
        state_callback.invoke(State::Init);
        let num_points = num_points as f64;
        let total_perimeter: f64 = polygon
            .boundaries_iter()
            .map(|(_, curve)| curve.length())
            .sum();
        let size_bound = size_bound_override.unwrap_or(total_perimeter / num_points);

        state_callback.invoke(State::GeneratingConstraints);

        let constraints: Vec<(BoundaryId, Constraint)> = polygon
            .boundaries_iter()
            .map(|(boundary_id, curve)| {
                (
                    boundary_id,
                    Constraint::create(curve, num_points, total_perimeter),
                )
            })
            .collect();

        let flattened_constraints: Box<[[Vector2<f64>; 2]]> = constraints
            .iter()
            .flat_map(|(_, constraint)| constraint.iter())
            .copied()
            .collect();

        state_callback.invoke(State::Triangulating);

        let triangulation_data =
            triangulation::triangulate(&flattened_constraints, ASPECT_BOUND, size_bound)?;

        state_callback.invoke(State::GeneratingAssociativeData);

        let point_id_map =
            generate_point_index_to_boundary_ids_map(&triangulation_data, &constraints);

        let boundary_point_map =
            generate_boundary_id_to_point_count_map(&triangulation_data, &constraints);

        state_callback.invoke(State::FindingSmallestEdge);

        let smallest_side_length = triangulation_data
            .faces()
            .par_iter()
            .map(|face| {
                let ith_point = |i: usize| triangulation_data.vertices()[i].point();
                face.0
                    .into_iter()
                    .cycle()
                    .map(ith_point)
                    .zip(face.0.into_iter().skip(1).cycle().map(ith_point))
                    .take(face.0.len())
                    .map(|(p, q)| (p - q).magnitude_squared())
                    .reduce(f32::min)
                    .unwrap_or_else(|| panic!("Mesh should not have any invalid vertex points"))
            })
            .reduce(|| f32::MAX, f32::min)
            .sqrt() as f64;

        let constraints = constraints.into_boxed_slice();

        state_callback.invoke(State::Done);

        Ok(Self {
            point_id_map,
            boundary_point_map,
            constraints,
            smallest_side_length,
            triangulation_data,
        })
    }
}

// q = p + l(r - p). l belongs to [0, 1]
fn is_on_same_segment(p: &Vector2<f64>, q: &Vector2<f64>, r: &Vector2<f64>) -> bool {
    let qp: Vector2<f64> = q - p;
    let rp: Vector2<f64> = r - p;

    let area = (qp.x * rp.y - qp.y * rp.x) / 2.0;
    let tolerance = 1e-14;
    if area.abs() > tolerance {
        return false;
    }

    if qp.magnitude() <= tolerance {
        return true;
    }

    let dot = rp.dot(&qp);
    if dot.abs() == 0.0 {
        return false;
    }

    let l = qp.dot(&qp) / dot;
    if !l.is_finite() {
        return false;
    }

    let min_l = -tolerance;
    let max_l = 1.0 + tolerance;
    l >= min_l && l <= max_l
}

fn generate_point_index_to_boundary_ids_map(
    data: &triangulation::Data,
    constraints: &[(BoundaryId, Constraint)],
) -> PointIdxToIdsMap {
    data.vertices()
        .par_iter()
        .enumerate()
        .filter_map(|(index, vertex)| {
            let boundary_ids: FxHashSet<BoundaryId> = constraints
                .par_iter()
                .filter_map(|(id, constraint)| {
                    constraint
                        .contains_point(&vertex.point().map(|v| v as f64))
                        .then_some(*id)
                })
                .collect();
            if boundary_ids.is_empty() {
                None
            } else {
                Some((index, boundary_ids))
            }
        })
        .collect()
}

fn generate_boundary_id_to_point_count_map(
    data: &triangulation::Data,
    constraints: &[(BoundaryId, Constraint)],
) -> BoundaryIdToCountMap {
    data.vertices().iter().enumerate().fold(
        {
            let mut map = BoundaryIdToCountMap::default();
            map.reserve(constraints.len());
            map
        },
        |mut map, (index, vertex)| {
            constraints.iter().for_each(|(boundary_id, constraint)| {
                if constraint.contains_point(&vertex.point().map(|v| v as f64)) {
                    map.entry(*boundary_id)
                        .and_modify(|indices| {
                            indices.insert(index);
                        })
                        .or_default();
                }
            });
            map
        },
    )
}

/// 2D pipeline — the current production mesh implementation. New 2D code
/// should reach for `mesh::d2::*` rather than the crate root; the crate root
/// re-exports are kept for backward compatibility.
pub mod d2 {
    pub use crate::{
        BoundaryIdToCountMap, Callback, Constraint, Mesh, PointIdxToIdsMap, State,
    };
}

/// 3D pipeline — placeholder stubs. Tetrahedral meshing for the four
/// primitive shapes (Cube, Cuboid, Sphere, Cylinder) lands in Step 3 of the
/// refactor; this module exists so downstream code can already reference
/// `mesh::d3::*` paths.
pub mod d3 {
    use nalgebra::Vector3;

    /// 3D tetrahedral mesh — vertices plus 4-index tets. Step 3 fills in the
    /// generation logic; for now this is a passive container.
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[derive(Debug, Clone, Default)]
    pub struct Mesh3D {
        pub vertices: Vec<Vector3<f64>>,
        pub tetrahedra: Vec<[usize; 4]>,
    }
}

#[cfg(test)]
mod tests {
    use super::{Callback, Mesh, State};
    use cgal::{num::Rational, PolygonSet, PolygonSetInput, PolygonSetInputKind, RationalPoint};
    use nalgebra::Vector2;
    use test_case::test_case;

    #[test_case(Vector2::new(0.0, 0.0), Vector2::new(1.0, 1.0), Vector2::new(2.0, 2.0) => true)]
    #[test_case(Vector2::new(2.0, 2.0), Vector2::new(1.0, 1.0), Vector2::new(0.0, 0.0) => true)]
    #[test_case(Vector2::new(0.0, 0.0), Vector2::new(2.0, 2.0), Vector2::new(2.0, 2.0) => true)]
    #[test_case(Vector2::new(0.0, 0.0), Vector2::new(0.0, 0.0), Vector2::new(2.0, 2.0) => true)]
    #[test_case(Vector2::new(0.0, 0.0), Vector2::new(2.0, 2.0), Vector2::new(1.0, 1.0) => false)]
    #[test_case(Vector2::new(1.0, 1.0), Vector2::new(2.0, 2.0), Vector2::new(0.0, 0.0) => false)]
    #[test_case(Vector2::new(0.0, 0.0), Vector2::new(1.0, 1.1), Vector2::new(2.0, 2.0) => false)]
    #[test_case(Vector2::new(0.0, 0.0), Vector2::new(1.0, 0.9), Vector2::new(2.0, 2.0) => false)]
    #[test_case(Vector2::new(0.0, 0.0), Vector2::new(0.0, 0.1), Vector2::new(2.0, 2.0) => false)]
    #[test_case(Vector2::new(0.0, 0.0), Vector2::new(1.0, 1.0), Vector2::new(2.0, 1.9) => false)]
    fn is_on_same_segment_works_within_err(
        p: Vector2<f64>,
        q: Vector2<f64>,
        r: Vector2<f64>,
    ) -> bool {
        super::is_on_same_segment(&p, &q, &r)
    }

    #[test_case(100)]
    #[test_case(300)]
    #[test_case(500)]
    #[test_case(1000)]
    fn generate_works(num_points: u32) {
        let mut polygon_set = PolygonSet::default();

        let fraction = |num, den| Rational::new_fraction_i32(num, den).unwrap();

        let input = PolygonSetInput::Join(PolygonSetInputKind::LinearPolygon(vec![
            RationalPoint::default(),
            RationalPoint::new(1, 0),
            RationalPoint::new(1, 1),
            RationalPoint::new(0, 1),
        ]));
        polygon_set.process_input(&input).unwrap();

        let input = PolygonSetInput::Difference(PolygonSetInputKind::Circle {
            center: RationalPoint::new(fraction(1, 2), fraction(1, 2)),
            diameter: fraction(3, 5),
        });
        polygon_set.process_input(&input).unwrap();

        let input = PolygonSetInput::Difference(PolygonSetInputKind::Circle {
            center: RationalPoint::new(fraction(3, 20), fraction(1, 2)),
            diameter: fraction(1, 5),
        });
        polygon_set.process_input(&input).unwrap();

        let input = PolygonSetInput::Difference(PolygonSetInputKind::Circle {
            center: RationalPoint::new(fraction(17, 20), fraction(1, 2)),
            diameter: fraction(1, 5),
        });
        polygon_set.process_input(&input).unwrap();

        let mut states = Vec::with_capacity(6);

        assert!(Mesh::generate(
            &polygon_set.polygon_with_holes()[0],
            num_points,
            None,
            Callback::from(|state| states.push(state))
        )
        .is_ok());

        let expected_states = vec![
            State::Init,
            State::GeneratingConstraints,
            State::Triangulating,
            State::GeneratingAssociativeData,
            State::FindingSmallestEdge,
            State::Done,
        ];

        assert_eq!(states, expected_states);
    }
}
