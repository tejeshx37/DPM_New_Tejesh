use cgal::{PolygonSet, PolygonSetInput};
use serde::{Deserialize, Serialize};
use std::cell::OnceCell;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PolygonData {
    pub inputs: Vec<PolygonSetInput>,
    #[serde(skip)]
    polygon_set: OnceCell<PolygonSet>,
}

impl PolygonData {
    pub(super) fn new(inputs: Vec<PolygonSetInput>) -> Self {
        Self {
            inputs,
            polygon_set: OnceCell::new(),
        }
    }

    pub fn polygon_set(&self) -> &PolygonSet {
        self.polygon_set
            .get_or_init(|| PolygonSet::from_inputs(&self.inputs).expect("Inputs are valid"))
    }
}
