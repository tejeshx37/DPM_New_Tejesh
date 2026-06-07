use crate::{
    model::engine::{self, Config, ExportConfig},
    ui::{
        always_open_window::AlwaysOpenWindow,
        dialog_utils::{self, ok_cancel},
    },
};
use cpd::{
    BulkMaterialProps, ElasticityCondition, FailureCriteria, IsotropicMaterialProps, MaterialProps,
    OrthotropicMaterialProps,
};
use egui::{Align, Checkbox, Context, Layout, Ui};
use enum_map::{Enum, EnumMap};
use mesh::Mesh;
use nalgebra_ext::matrix2::Component;
use rfd::FileDialog;
use std::{path::PathBuf, sync::Arc, time::Duration};
use strum::{AsRefStr, Display, EnumIter, IntoEnumIterator};

const INPUT_SECTION_MARGIN: f32 = 8.0;

#[derive(Debug, Clone, Copy, Enum, Display, AsRefStr, EnumIter)]
#[strum(serialize_all = "title_case")]
enum SimulationConfigInput {
    Duration,
    RefreshPeriod,
}

impl SimulationConfigInput {
    fn info_text(&self) -> &str {
        match self {
            SimulationConfigInput::Duration => "Total duration of the simulation.",
            SimulationConfigInput::RefreshPeriod => "Rate at which new frames should be displayed. For example \
            a refresh rate of 100 will display new frame at every 100 timesteps. Higher the period, slower the update, \
            but faster the simulation since rendering a frame takes up a lot of time.",
        }
    }
}

#[derive(Debug, Clone, Copy, Enum, Display, AsRefStr, EnumIter)]
#[strum(serialize_all = "title_case")]
enum BulkMaterialPropsInput {
    Density,
    Damping,
    FailureStrainEnergy,
    FailureTensionalStress,
    FailureCompressionalStress,
}

impl BulkMaterialPropsInput {
    fn info_text(&self) -> &str {
        match self {
            BulkMaterialPropsInput::Density => "Density of the material",
            BulkMaterialPropsInput::Damping => "Damping constant of the material",
            BulkMaterialPropsInput::FailureStrainEnergy => {
                "Strain energy at which cracks start to form.\nOptional parameter, leave blank if not necessary."
            }
            BulkMaterialPropsInput::FailureTensionalStress => {
                "Tensional stress at which cracks start to form.\nOptional parameter, leave blank if not necessary."
            }
            BulkMaterialPropsInput::FailureCompressionalStress => {
                "Compressional stress at which cracks starts to form.\nOptional parameter, leave blank if not necessary."
            }
        }
    }
}

#[derive(Debug, Default)]
struct IsotropicMaterialPropsInput {
    elasticity_modulus: String,
    poissons_ratio: String,
    elasticity_condition: ElasticityCondition,
}

impl From<&IsotropicMaterialProps> for IsotropicMaterialPropsInput {
    fn from(value: &IsotropicMaterialProps) -> Self {
        Self {
            elasticity_modulus: value.elasticity_modulus().to_string(),
            poissons_ratio: value.poissons_ratio().to_string(),
            elasticity_condition: *value.elasticity_condition(),
        }
    }
}

#[derive(Debug, Default)]
struct OrthotropicMaterialPropsInput {
    elasticity_modulus_x: String,
    elasticity_modulus_y: String,
    poissons_ratio_xy: String,
    poissons_ratio_yx: String,
    shear_modulus_xy: String,
}

impl From<&OrthotropicMaterialProps> for OrthotropicMaterialPropsInput {
    fn from(value: &OrthotropicMaterialProps) -> Self {
        Self {
            elasticity_modulus_x: value.elasticity_modulus_x().to_string(),
            elasticity_modulus_y: value.elasticity_modulus_y().to_string(),
            poissons_ratio_xy: value.poissons_ratio_xy().to_string(),
            poissons_ratio_yx: value.poissons_ratio_yx().to_string(),
            shear_modulus_xy: value.shear_modulus_xy().to_string(),
        }
    }
}

#[derive(Debug)]
enum MaterialPropsInput {
    Isotropic(IsotropicMaterialPropsInput),
    Orthotropic(OrthotropicMaterialPropsInput),
}

impl Default for MaterialPropsInput {
    fn default() -> Self {
        Self::Isotropic(IsotropicMaterialPropsInput::default())
    }
}

impl From<&MaterialProps> for MaterialPropsInput {
    fn from(value: &MaterialProps) -> Self {
        match value {
            MaterialProps::Isotropic(value) => {
                MaterialPropsInput::Isotropic(IsotropicMaterialPropsInput::from(value))
            }
            MaterialProps::Orthotropic(value) => {
                MaterialPropsInput::Orthotropic(OrthotropicMaterialPropsInput::from(value))
            }
        }
    }
}

#[derive(Debug, Default)]
struct ExportConfigInput {
    export_points: bool,
    exported_stress_components: EnumMap<Component, bool>,
    export_period: String,
    export_path: PathBuf,
}

#[derive(Debug)]
pub struct State {
    simulation_config_input: EnumMap<SimulationConfigInput, String>,
    time_step_input: String,
    bulk_material_props_input: EnumMap<BulkMaterialPropsInput, String>,
    material_props_input: MaterialPropsInput,
    export: bool,
    export_config_input: ExportConfigInput,
    mesh: Arc<Mesh>,
}

impl State {
    pub fn default(mesh: Arc<Mesh>) -> Self {
        Self {
            simulation_config_input: EnumMap::default(),
            time_step_input: String::default(),
            bulk_material_props_input: EnumMap::default(),
            material_props_input: MaterialPropsInput::default(),
            export: false,
            export_config_input: ExportConfigInput::default(),
            mesh,
        }
    }

    pub fn new(config: &Config, mesh: Arc<Mesh>) -> Self {
        let mp = config.cpd_config().material_props();
        let bp = mp.bulk_props();
        let fc = bp.failure_criteria();
        let ec = config.export_config();
        let opt_to_string =
            |opt: &Option<f32>| opt.map(|value| value.to_string()).unwrap_or_default();
        Self {
            simulation_config_input: enum_map::enum_map! {
                SimulationConfigInput::Duration => config.cpd_config().duration().as_secs_f32().to_string(),
                SimulationConfigInput::RefreshPeriod => config.refresh_period().to_string(),
            },
            time_step_input: format!("{:e}", config.cpd_config().time_delta().as_secs_f64()),
            bulk_material_props_input: enum_map::enum_map! {
                BulkMaterialPropsInput::Density => bp.density().to_string(),
                BulkMaterialPropsInput::Damping => bp.damping().to_string(),
                BulkMaterialPropsInput::FailureStrainEnergy => opt_to_string(fc.strain_energy()) ,
                BulkMaterialPropsInput::FailureTensionalStress => opt_to_string(fc.tensional_stress()),
                BulkMaterialPropsInput::FailureCompressionalStress => opt_to_string(fc.compressional_stress()),
            },
            material_props_input: MaterialPropsInput::from(mp),
            export: ec.is_some(),
            export_config_input: ec
                .as_ref()
                .map(|ec| ExportConfigInput {
                    export_points: *ec.export_points(),
                    exported_stress_components: *ec.export_stress_components(),
                    export_period: ec.export_period().to_string(),
                    export_path: ec.export_path().to_owned(),
                })
                .unwrap_or_default(),
            mesh,
        }
    }
}

impl TryFrom<&State> for Config {
    type Error = String;

    fn try_from(state: &State) -> Result<Self, Self::Error> {
        macro_rules! parse_f32 {
            ($name:expr, $value:expr) => {
                $value
                    .parse()
                    .map_err(|_| format!("Invalid {} {}", $name, $value))
                    .and_then(|value: f32| {
                        if !value.is_finite() {
                            Err(format!("{} should be finite, i.e. no NaN or Inf", $name))
                        } else {
                            Ok(value)
                        }
                    })
            };
        }
        macro_rules! parse_input {
            ($inputs_field:ident, $variant:ident) => {{
                let input = paste::paste! { [< $inputs_field:camel >]::$variant };
                let value = &state.$inputs_field[input];
                parse_f32!(input, value)
            }};
        }
        macro_rules! failure_criteria {
            ( $( $criteria:ident ),* ) => {{
                let builder = FailureCriteria::builder();
                $(
                    let builder = {
                        let variant = paste::paste! { BulkMaterialPropsInput::[< Failure $criteria >] };
                        let input = &state.bulk_material_props_input[variant];
                        let value = (!input.is_empty()).then(|| input
                            .parse()
                            .map_err(|_| format!("Invalid {variant} {input}")))
                            .transpose()?;
                        paste::paste! { builder.[< $criteria:snake >](value) }
                    };
                )*
                builder.build()
            }};
        }
        let failure_critera = failure_criteria!(StrainEnergy, TensionalStress, CompressionalStress);
        let ec = &state.export_config_input;
        let bulk_props = BulkMaterialProps::builder()
            .density(parse_input!(bulk_material_props_input, Density)?)
            .damping(parse_input!(bulk_material_props_input, Damping)?)
            .failure_criteria(failure_critera)
            .build();
        let material_props = match &state.material_props_input {
            MaterialPropsInput::Isotropic(input) => MaterialProps::Isotropic(
                IsotropicMaterialProps::builder()
                    .bulk_props(bulk_props)
                    .elasticity_condition(input.elasticity_condition)
                    .elasticity_modulus(parse_f32!("elasticity modulus", input.elasticity_modulus)?)
                    .poissons_ratio(parse_f32!("poisson's ratio", input.poissons_ratio)?)
                    .build(),
            ),
            MaterialPropsInput::Orthotropic(input) => MaterialProps::Orthotropic(
                OrthotropicMaterialProps::builder()
                    .bulk_props(bulk_props)
                    .elasticity_modulus_x(parse_f32!("Ex", input.elasticity_modulus_x)?)
                    .elasticity_modulus_y(parse_f32!("Ey", input.elasticity_modulus_y)?)
                    .poissons_ratio_xy(parse_f32!("Vxy", input.poissons_ratio_xy)?)
                    .poissons_ratio_yx(parse_f32!("Vyx", input.poissons_ratio_yx)?)
                    .shear_modulus_xy(parse_f32!("Gxy", input.shear_modulus_xy)?)
                    .build(),
            ),
        };
        let cpd_config = cpd::config::Config::builder()
            .material_props(material_props)
            .duration(parse_input!(simulation_config_input, Duration).map(Duration::from_secs_f32)?)
            .time_delta(
                state
                    .time_step_input
                    .parse()
                    .map(Duration::from_secs_f32)
                    .map_err(|_| format!("Invalid time step {}", state.time_step_input))?,
            )
            .build();
        let export_config =
            state
                .export
                .then(|| {
                    if !ec.export_path.is_dir() {
                        Err(format!(
                            "Export path '{}' does not exist",
                            ec.export_path.display()
                        ))
                    } else {
                        Ok(ExportConfig::builder()
                            .export_points(ec.export_points)
                            .export_stress_components(ec.exported_stress_components)
                            .export_period(ec.export_period.parse().map_err(|_| {
                                format!("Invalid export period {}", ec.export_period)
                            })?)
                            .export_path(ec.export_path.to_owned())
                            .build())
                    }
                })
                .transpose()?;
        let config = Self::builder()
            .cpd_config(cpd_config)
            .refresh_period({
                let value = &state.simulation_config_input[SimulationConfigInput::RefreshPeriod];
                value
                    .parse()
                    .map_err(|_| format!("Invalid refresh period {value}"))?
            })
            .export_config(export_config)
            .build();
        Ok(config)
    }
}

pub enum Response {
    Noop,
    ConfigResult(Result<Box<Config>, String>),
    Cancel,
}

pub fn show(state: &mut State, ctx: &Context) -> Response {
    AlwaysOpenWindow::new("Engine config")
        .resizable(false)
        .show(ctx, |ui| window_ui(state, ui))
}

fn window_ui(state: &mut State, ui: &mut Ui) -> Response {
    ui.add_space(INPUT_SECTION_MARGIN);
    ui.group(|ui| simulation_config_table_layout(state, ui));
    ui.add_space(INPUT_SECTION_MARGIN);
    ui.group(|ui| material_props_table_layout(state, ui));
    ui.add_space(INPUT_SECTION_MARGIN);
    ui.group(|ui| time_step_input(state, ui));
    ui.add_space(INPUT_SECTION_MARGIN);
    ui.checkbox(&mut state.export, "Export data")
        .on_hover_text("Check to configure which all data should be exported.");
    if state.export {
        ui.add_space(INPUT_SECTION_MARGIN);
        ui.group(|ui| export_config_layout(&mut state.export_config_input, ui));
    }
    ui.add_space(INPUT_SECTION_MARGIN);
    ui.with_layout(
        Layout::right_to_left(Align::Min),
        |ui| match ok_cancel::buttons(ui) {
            ok_cancel::Response::Ok => {
                Response::ConfigResult(Config::try_from(&*state).map(Box::new))
            }
            ok_cancel::Response::Cancel => Response::Cancel,
            ok_cancel::Response::Noop => Response::Noop,
        },
    )
    .inner
}

fn time_step_input(state: &mut State, ui: &mut Ui) {
    if let Some(optimal_time_step) = optimal_time_step(state) {
        ui.horizontal(|ui| {
            ui.label(format!("Optimal time step is {optimal_time_step:e}"));
            if ui.button("Use this value").clicked() {
                state.time_step_input = format!("{:e}", optimal_time_step);
            }
        });
    }
    ui.horizontal(|ui| {
        ui.label("Time step");
        ui.text_edit_singleline(&mut state.time_step_input)
            .on_hover_text(
                "Time increment (dt) at each iteration of the simulation.\n\
        Smaller the time step, slower and better the simulation.\n\
        Higher the time step, faster but very inaccurate simulation.",
            );
    });
}

fn optimal_time_step(state: &State) -> Option<f64> {
    state.bulk_material_props_input[BulkMaterialPropsInput::Density]
        .parse::<f64>()
        .ok()
        .and_then(|density| {
            let e_opt = match &state.material_props_input {
                MaterialPropsInput::Isotropic(p) => p.elasticity_modulus.parse::<f64>().ok(),
                MaterialPropsInput::Orthotropic(p) => p
                    .elasticity_modulus_x
                    .parse::<f64>()
                    .ok()
                    .zip(p.elasticity_modulus_y.parse().ok())
                    .map(|(ex, ey)| ex.max(ey)),
            };
            e_opt.map(|e| (density, e))
        })
        .and_then(|(density, elasticity_modulus)| {
            engine::optimal_time_delta(density, elasticity_modulus, &state.mesh)
        })
}

fn simulation_config_table_layout(state: &mut State, ui: &mut Ui) {
    SimulationConfigInput::iter().for_each(|input| {
        num_input_row(
            ui,
            input.as_ref(),
            &mut state.simulation_config_input[input],
            input.info_text(),
        );
    });
}

fn bulk_material_props_input_layout(state: &mut State, ui: &mut Ui) {
    BulkMaterialPropsInput::iter().for_each(|input| {
        num_input_row(
            ui,
            input.as_ref(),
            &mut state.bulk_material_props_input[input],
            input.info_text(),
        );
    });
}

fn isotropic_material_props_input_layout(input: &mut IsotropicMaterialPropsInput, ui: &mut Ui) {
    num_input_row(
        ui,
        "Elasticity modulus",
        &mut input.elasticity_modulus,
        "Elasticity modulus of the material",
    );
    num_input_row(
        ui,
        "Poisson's ratio",
        &mut input.poissons_ratio,
        "Poisson's ratio of the material",
    );
    ui.horizontal(|ui| {
        ui.label("Elasticity condition:");
        ui.selectable_value(
            &mut input.elasticity_condition,
            ElasticityCondition::PlaneStress,
            "Plain stress",
        );
        ui.selectable_value(
            &mut input.elasticity_condition,
            ElasticityCondition::PlaneStrain,
            "Plain strain",
        );
    });
}

fn orthotropic_material_props_input_layout(input: &mut OrthotropicMaterialPropsInput, ui: &mut Ui) {
    use dialog_utils::Field;
    ui.horizontal(|ui| {
        ui.label("Elasticity modulus:");
        dialog_utils::single_line_double_input_field(
            ui,
            Field {
                name: "Ex",
                value: &mut input.elasticity_modulus_x,
            },
            Field {
                name: "Ey",
                value: &mut input.elasticity_modulus_y,
            },
        );
    });
    ui.horizontal(|ui| {
        ui.label("Poisson's ratio:");
        dialog_utils::single_line_double_input_field(
            ui,
            Field {
                name: "Vxy",
                value: &mut input.poissons_ratio_xy,
            },
            Field {
                name: "Vyx",
                value: &mut input.poissons_ratio_yx,
            },
        );
    });
    ui.horizontal(|ui| {
        ui.label("Shear modulus (Gxy)");
        ui.text_edit_singleline(&mut input.shear_modulus_xy);
    });
}

fn material_props_table_layout(state: &mut State, ui: &mut Ui) {
    ui.horizontal(|ui| {
        ui.label("Material kind:");
        let response = ui.selectable_label(
            matches!(state.material_props_input, MaterialPropsInput::Isotropic(_)),
            "Isotropic",
        );
        if response.clicked() {
            state.material_props_input =
                MaterialPropsInput::Isotropic(IsotropicMaterialPropsInput::default());
        }
        let response = ui.selectable_label(
            matches!(
                state.material_props_input,
                MaterialPropsInput::Orthotropic(_)
            ),
            "Orthotropic",
        );
        if response.clicked() {
            state.material_props_input =
                MaterialPropsInput::Orthotropic(OrthotropicMaterialPropsInput::default());
        }
    });
    match &mut state.material_props_input {
        MaterialPropsInput::Isotropic(input) => {
            isotropic_material_props_input_layout(input, ui);
        }
        MaterialPropsInput::Orthotropic(input) => {
            orthotropic_material_props_input_layout(input, ui);
        }
    }
    bulk_material_props_input_layout(state, ui);
}

fn export_config_layout(state: &mut ExportConfigInput, ui: &mut Ui) {
    ui.with_layout(ui.layout().with_cross_align(Align::Min), |ui| {
        ui.horizontal(|ui| {
            ui.label("Export points");
            ui.add(Checkbox::without_text(&mut state.export_points))
                .on_hover_text("Check to export points to a file named Points_<timestep>.csv");
        });
        ui.horizontal(|ui| {
            ui.label("Export stress components");
            Component::iter().for_each(|comp| {
                ui.checkbox(&mut state.exported_stress_components[comp], comp.as_ref())
                    .on_hover_text("Check to export this component of stress to a file name Stress_<timestep>.csv");
            });
        });
        ui.horizontal(|ui| {
            ui.label("Export period");
            ui.text_edit_singleline(&mut state.export_period).on_hover_text("Time step interval at \
            which data should be exported.\nFor example, for an export period of 100, at every 100th \
            time step, data will be exported.");
        });
        ui.horizontal(|ui| {
            let opt = ui
                .button("Select export path")
                .clicked()
                .then(|| {
                    FileDialog::new()
                        .set_directory(&state.export_path)
                        .pick_folder()
                })
                .flatten();
            if let Some(path) = opt {
                state.export_path = path;
            }
            ui.label(state.export_path.display().to_string()).on_hover_text("The path to which \
            files should be written to.\nCannot be empty");
        });
    });
}

fn num_input_row(ui: &mut Ui, label: &str, text: &mut String, info_text: &str) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.text_edit_singleline(text).on_hover_text(info_text);
    });
}
