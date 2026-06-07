#[cxx::bridge(namespace = "Triangulation")]
mod ffi {
    unsafe extern "C++" {
        include!("cgal-sys/cpp/pair_utils.h");
        include!("cgal-sys/cpp/triangulation.h");
        include!("cgal-sys/cpp/vector_utils.h");

        #[namespace = ""]
        type Point = crate::Point;

        type EpickPoint;
        fn create_epick_point(x: f64, y: f64) -> UniquePtr<EpickPoint>;

        fn x(self: &EpickPoint) -> &f64;
        fn y(self: &EpickPoint) -> &f64;

        type Face;
        fn at(self: &Face, index: usize) -> &usize;

        type IndexPair;
        #[rust_name = "get_first_index"]
        #[namespace = ""]
        fn first(pair: &IndexPair) -> usize;
        #[rust_name = "get_second_index"]
        #[namespace = ""]
        fn second(pair: &IndexPair) -> usize;

        type Vertex;
        #[rust_name = "get_point"]
        #[namespace = ""]
        fn first(vertex: &Vertex) -> &EpickPoint;
        #[rust_name = "get_incident_faces"]
        #[namespace = ""]
        fn second(vertex: &Vertex) -> &CxxVector<usize>;

        type Data;
        fn faces(self: &Data) -> &CxxVector<Face>;
        fn edges(self: &Data) -> &CxxVector<IndexPair>;
        fn vertices(self: &Data) -> &CxxVector<Vertex>;

        type PointPair;
        fn create_point_pair(first: &EpickPoint, second: &EpickPoint) -> UniquePtr<PointPair>;

        type Constraints;
        #[namespace = ""]
        #[rust_name = "create_constraints"]
        fn create_vector(capacity: usize) -> UniquePtr<Constraints>;
        fn reserve(self: Pin<&mut Constraints>, capacity: usize);
        #[namespace = ""]
        fn push_back(vec: Pin<&mut Constraints>, constraint: UniquePtr<PointPair>);

        fn triangulate(
            constraints: &Constraints,
            aspect_bound: f64,
            size_bound: f64,
        ) -> Result<UniquePtr<Data>>;
    }
}

pub use ffi::*;
