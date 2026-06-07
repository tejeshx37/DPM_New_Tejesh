use cgal_sys::triangulation::EpickPoint;
use cxx::UniquePtr;
use derive_getters::Getters;
use nalgebra::Vector2;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy)]
pub struct Face(pub [usize; 3]);

impl From<&cgal_sys::triangulation::Face> for Face {
    fn from(value: &cgal_sys::triangulation::Face) -> Self {
        Self([*value.at(0), *value.at(1), *value.at(2)])
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy)]
pub struct IndexPair(pub usize, pub usize);

impl From<&cgal_sys::triangulation::IndexPair> for IndexPair {
    fn from(value: &cgal_sys::triangulation::IndexPair) -> Self {
        Self(
            cgal_sys::triangulation::get_first_index(value),
            cgal_sys::triangulation::get_second_index(value),
        )
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Getters)]
pub struct Vertex {
    point: Vector2<f32>,
    incident_faces: Box<[usize]>,
}

impl From<&cgal_sys::triangulation::Vertex> for Vertex {
    fn from(value: &cgal_sys::triangulation::Vertex) -> Self {
        let point = cgal_sys::triangulation::get_point(value);
        Self {
            point: Vector2::new(*point.x() as f32, *point.y() as f32),
            incident_faces: cgal_sys::triangulation::get_incident_faces(value)
                .iter()
                .copied()
                .collect(),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Getters)]
pub struct Data {
    faces: Box<[Face]>,
    edges: Box<[IndexPair]>,
    vertices: Box<[Vertex]>,
}

impl From<UniquePtr<cgal_sys::triangulation::Data>> for Data {
    fn from(value: UniquePtr<cgal_sys::triangulation::Data>) -> Self {
        Self {
            faces: value.faces().iter().map(Into::into).collect(),
            edges: value.edges().iter().map(Into::into).collect(),
            vertices: value.vertices().iter().map(Into::into).collect(),
        }
    }
}

pub fn triangulate(
    constraints: &[[Vector2<f64>; 2]],
    aspect_bound: f64,
    size_bound: f64,
) -> Result<Data, String> {
    let constraints = constraints
        .iter()
        .map(|arr| arr.map(|point| cgal_sys::triangulation::create_epick_point(point.x, point.y)))
        .map(|[first, second]: [UniquePtr<EpickPoint>; 2]| {
            cgal_sys::triangulation::create_point_pair(&first, &second)
        })
        .fold(
            cgal_sys::triangulation::create_constraints(constraints.len()),
            |mut vec, pair| {
                cgal_sys::triangulation::push_back(vec.pin_mut(), pair);
                vec
            },
        );
    cgal_sys::triangulation::triangulate(&constraints, aspect_bound, size_bound)
        .map(Into::into)
        .map_err(|err| err.to_string())
}
