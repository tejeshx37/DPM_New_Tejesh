use csv::WriterBuilder;
use derive_getters::Getters;
use enum_map::EnumMap;
use mesh::Mesh;
use nalgebra_ext::matrix2::Component;
use std::{
    fs, iter,
    path::{Path, PathBuf},
};
use typed_builder::TypedBuilder;

#[derive(Debug, Clone, Getters, TypedBuilder)]
pub struct ExportConfig {
    export_points: bool,
    export_stress_components: EnumMap<Component, bool>,
    export_period: u128,
    export_path: PathBuf,
}

pub fn mesh(mesh: &Mesh, path: &Path) -> csv::Result<()> {
    fs::create_dir_all(path)?;
    let faces = mesh.triangulation_data().faces();
    let mut writer = WriterBuilder::default()
        .buffer_capacity(faces.len())
        .from_path(path.join("Elements").with_extension("csv"))?;
    faces.iter().try_for_each(|face| writer.serialize(face.0))
}

pub fn data(data: &cpd::ExportData, config: &ExportConfig, time_step: u128) -> csv::Result<()> {
    fs::create_dir_all(&config.export_path)?;
    if config.export_points {
        let mut writer = WriterBuilder::default()
            .buffer_capacity(data.nodes().len())
            .has_headers(true)
            .from_path(
                config
                    .export_path
                    .join(format!("Points_{time_step}"))
                    .with_extension("csv"),
            )?;
        writer.write_record(["X", "Y"])?;
        data.nodes()
            .iter()
            .try_for_each(|node| writer.serialize(node.position()))?;
    }
    let header = config
        .export_stress_components
        .iter()
        .filter_map(|(component, export)| export.then_some(component))
        .map(|component| format!("E{}", component.as_ref()))
        .chain(iter::once(String::from("Broken")))
        .collect::<Vec<_>>();
    if header.len() <= 1 {
        return Ok(());
    }
    let mut writer = WriterBuilder::default()
        .buffer_capacity(data.elements().len())
        .has_headers(true)
        .from_path(
            config
                .export_path
                .join(format!("Stress_{time_step}"))
                .with_extension("csv"),
        )?;
    writer.write_record(&header)?;
    data.elements()
        .iter()
        .map(|element| {
            config
                .export_stress_components
                .iter()
                .filter_map(|(component, export)| {
                    export.then_some(*element.stress().index(component))
                })
                .map(|value| value.to_string())
                .chain(iter::once(element.is_broken().to_string()))
                .collect::<Vec<_>>()
        })
        .try_for_each(|record| writer.write_record(record))
}
