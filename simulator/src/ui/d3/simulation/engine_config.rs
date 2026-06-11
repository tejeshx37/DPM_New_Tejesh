//! Engine config modal — every simulation parameter visible at once.
//!
//! Replaces the click-through-each-collapsible flow on the side panel
//! with a single dialog the user fills in one pass (matches the reference
//! UI the user provided). The side panel still has the collapsibles for
//! quick adjustments; opening this dialog gives a consolidated view.
//!
//! Quick Material Preset dropdown sets typical engineering values for
//! Steel, Aluminum, Concrete, Glass, and Rubber so users don't have to
//! look up moduli and densities every time they spin up a new sim.

use cpd::d3::{BulkProps3D, FailureCriteria3D, IsotropicProps3D, MaterialProps3D, OrthotropicProps3D};
use egui::{Color32, ComboBox, Context, DragValue, Slider, Ui};

use super::State;

const MATERIAL_PRESETS: &[(&str, IsotropicProps3D)] = &[];

fn make_preset(name: &'static str) -> Option<(IsotropicProps3D, &'static str)> {
    // E in Pa, density kg/m^3, damping arbitrary scaled, stresses Pa.
    // Failure values are rough engineering yield numbers for the metal
    // / ceramic / polymer they're modelling.
    let bulk = |density, damping, w, sigma_t, sigma_c| BulkProps3D {
        density,
        damping,
        failure_criteria: FailureCriteria3D {
            strain_energy: Some(w),
            tensional_stress: Some(sigma_t),
            compressional_stress: Some(sigma_c),
        },
    };
    let preset = match name {
        "Steel (mild)" => IsotropicProps3D {
            elasticity_modulus: 2.0e11,
            poissons_ratio: 0.30,
            bulk: bulk(7850.0, 200.0, 1.0e6, 4.0e8, 4.0e8),
        },
        "Aluminum (6061)" => IsotropicProps3D {
            elasticity_modulus: 6.9e10,
            poissons_ratio: 0.33,
            bulk: bulk(2700.0, 150.0, 5.0e5, 2.7e8, 2.7e8),
        },
        "Concrete" => IsotropicProps3D {
            elasticity_modulus: 3.0e10,
            poissons_ratio: 0.20,
            bulk: bulk(2400.0, 250.0, 1.0e4, 3.0e6, 3.0e7),
        },
        "Glass" => IsotropicProps3D {
            elasticity_modulus: 7.0e10,
            poissons_ratio: 0.22,
            bulk: bulk(2500.0, 50.0, 5.0e4, 5.0e7, 1.0e9),
        },
        "Rubber" => IsotropicProps3D {
            elasticity_modulus: 5.0e7,
            poissons_ratio: 0.49,
            bulk: bulk(1100.0, 800.0, 1.0e6, 5.0e7, 5.0e7),
        },
        _ => return None,
    };
    let suggested_dt = match name {
        "Steel (mild)" | "Aluminum (6061)" | "Concrete" | "Glass" => "1e-7",
        "Rubber" => "1e-6",
        _ => "1e-7",
    };
    Some((preset, suggested_dt))
}

const PRESET_NAMES: &[&str] = &[
    "Steel (mild)",
    "Aluminum (6061)",
    "Concrete",
    "Glass",
    "Rubber",
];

pub fn show_modal(state: &mut State, ctx: &Context) {
    let _ = MATERIAL_PRESETS;
    if !state.engine_config_open {
        return;
    }
    let mut open = true;
    egui::Window::new("Engine config")
        .open(&mut open)
        .resizable(true)
        .default_width(480.0)
        .default_height(620.0)
        .show(ctx, |ui| add_body(state, ui));
    if !open {
        state.engine_config_open = false;
    }
}

fn add_body(state: &mut State, ui: &mut Ui) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        // ── Top-of-dialog fracture status banner ──────────────────────
        let fc = state.material.failure_criteria();
        let fracture_enabled = fc.strain_energy.is_some()
            || fc.tensional_stress.is_some()
            || fc.compressional_stress.is_some();
        if fracture_enabled {
            ui.colored_label(
                Color32::from_rgb(120, 220, 120),
                "✓ Failure criteria active — fracture can occur",
            );
        } else {
            ui.colored_label(
                Color32::from_rgb(255, 160, 100),
                "⚠ No failure criteria — fracture is disabled. Pick a preset or tick a threshold below.",
            );
        }

        // ── Preset-just-applied confirmation ──────────────────────────
        if let Some(name) = state.preset_just_applied.take() {
            ui.colored_label(
                Color32::from_rgb(120, 220, 120),
                format!("Applied {name} — failure criteria, density, damping, E, ν and Δt all set."),
            );
        }

        ui.add_space(4.0);

        ui.group(|ui| {
            ui.label("Time / forcing");
            two_col(ui, "Duration (s)", |ui| {
                ui.add(
                    DragValue::new(&mut state.duration)
                        .speed(0.001)
                        .clamp_range(1e-6..=f32::MAX)
                        .min_decimals(3)
                        .max_decimals(9),
                );
            });
            two_col(ui, "Refresh / sample stride (steps)", |ui| {
                ui.add(
                    DragValue::new(&mut state.plots.sample_stride)
                        .speed(1.0)
                        .clamp_range(1..=1000u32),
                );
            });
            two_col(ui, "Steps per frame", |ui| {
                ui.add(Slider::new(&mut state.steps_per_frame, 1..=1000));
            });
            two_col(ui, "Body force X (m/s²)", |ui| {
                ui.add(DragValue::new(&mut state.body_force[0]).speed(0.1));
            });
            two_col(ui, "Body force Y", |ui| {
                ui.add(DragValue::new(&mut state.body_force[1]).speed(0.1));
            });
            two_col(ui, "Body force Z", |ui| {
                ui.add(DragValue::new(&mut state.body_force[2]).speed(0.1));
            });
        });

        ui.add_space(6.0);

        ui.group(|ui| {
            ui.label("Quick Material Preset");
            ui.horizontal(|ui| {
                ui.label("Select a preset:");
                let mut chosen: Option<&str> = None;
                ComboBox::from_id_source("engine_cfg_preset")
                    .selected_text("Select a preset…")
                    .show_ui(ui, |ui| {
                        for name in PRESET_NAMES {
                            if ui.selectable_label(false, *name).clicked() {
                                chosen = Some(*name);
                            }
                        }
                    });
                if let Some(name) = chosen {
                    if let Some((preset, dt)) = make_preset(name) {
                        state.material = MaterialProps3D::Isotropic(preset);
                        if let Ok(parsed) = dt.parse::<f32>() {
                            state.time_delta = parsed;
                        }
                        state.preset_just_applied = Some(name);
                        ui.ctx().request_repaint();
                    }
                }
            });
            ui.label("Sets E, ν, ρ, damping, and failure thresholds to typical engineering values. Time step is adjusted to a CFL-safe estimate.");
        });

        ui.add_space(6.0);

        ui.group(|ui| {
            ui.label("Material");
            let is_iso = matches!(state.material, MaterialProps3D::Isotropic(_));
            let is_ortho = matches!(state.material, MaterialProps3D::Orthotropic(_));
            ui.horizontal(|ui| {
                ui.label("Material kind:");
                if ui.selectable_label(is_iso, "Isotropic").clicked() && !is_iso {
                    let bulk = state.material.bulk().clone();
                    state.material = MaterialProps3D::Isotropic(IsotropicProps3D {
                        bulk,
                        ..IsotropicProps3D::default()
                    });
                }
                if ui.selectable_label(is_ortho, "Orthotropic").clicked() && !is_ortho {
                    let bulk = state.material.bulk().clone();
                    state.material = MaterialProps3D::Orthotropic(OrthotropicProps3D {
                        bulk,
                        ..OrthotropicProps3D::default()
                    });
                }
            });

            // ── Material-section fracture status ─────────────────────
            let fc = state.material.failure_criteria();
            let fracture_on = fc.strain_energy.is_some()
                || fc.tensional_stress.is_some()
                || fc.compressional_stress.is_some();
            if fracture_on {
                ui.colored_label(
                    Color32::from_rgb(120, 220, 120),
                    "✓ Fracture thresholds set",
                );
            } else {
                ui.colored_label(
                    Color32::from_rgb(255, 160, 100),
                    "⚠ No thresholds — tick at least one below to enable fracture",
                );
            }
            ui.add_space(2.0);

            match &mut state.material {
                MaterialProps3D::Isotropic(p) => {
                    two_col(ui, "Elasticity modulus E (Pa)", |ui| {
                        ui.add(
                            DragValue::new(&mut p.elasticity_modulus)
                                .speed(1e6)
                                .clamp_range(1.0..=f32::MAX),
                        );
                    });
                    two_col(ui, "Poisson's ratio ν", |ui| {
                        ui.add(
                            DragValue::new(&mut p.poissons_ratio)
                                .speed(0.01)
                                .clamp_range(0.0..=0.49),
                        );
                    });
                }
                MaterialProps3D::Orthotropic(p) => {
                    two_col(ui, "E_x / E_y / E_z (Pa)", |ui| {
                        ui.add(DragValue::new(&mut p.elasticity_modulus_x).speed(1e6));
                        ui.add(DragValue::new(&mut p.elasticity_modulus_y).speed(1e6));
                        ui.add(DragValue::new(&mut p.elasticity_modulus_z).speed(1e6));
                    });
                    two_col(ui, "ν_xy / ν_xz / ν_yz", |ui| {
                        ui.add(DragValue::new(&mut p.poissons_ratio_xy).speed(0.01).clamp_range(0.0..=0.49));
                        ui.add(DragValue::new(&mut p.poissons_ratio_xz).speed(0.01).clamp_range(0.0..=0.49));
                        ui.add(DragValue::new(&mut p.poissons_ratio_yz).speed(0.01).clamp_range(0.0..=0.49));
                    });
                    two_col(ui, "G_xy / G_xz / G_yz (Pa)", |ui| {
                        ui.add(DragValue::new(&mut p.shear_modulus_xy).speed(1e6));
                        ui.add(DragValue::new(&mut p.shear_modulus_xz).speed(1e6));
                        ui.add(DragValue::new(&mut p.shear_modulus_yz).speed(1e6));
                    });
                }
            }
            ui.label("Elasticity condition: Three Dimensional");
            let bulk = match &mut state.material {
                MaterialProps3D::Isotropic(p) => &mut p.bulk,
                MaterialProps3D::Orthotropic(p) => &mut p.bulk,
            };
            two_col(ui, "Density ρ (kg/m³)", |ui| {
                ui.add(
                    DragValue::new(&mut bulk.density)
                        .speed(10.0)
                        .clamp_range(1e-6..=f32::MAX),
                );
            });
            two_col(ui, "Damping c", |ui| {
                ui.add(
                    DragValue::new(&mut bulk.damping)
                        .speed(0.1)
                        .clamp_range(0.0..=f32::MAX),
                );
            });
            optional_field(ui, "Failure strain energy (J/m³)", &mut bulk.failure_criteria.strain_energy, 1e3, 0.0);
            optional_field(ui, "Failure tensional stress (Pa)", &mut bulk.failure_criteria.tensional_stress, 1e6, 0.0);
            optional_field(ui, "Failure compressional stress (Pa)", &mut bulk.failure_criteria.compressional_stress, 1e6, 0.0);
        });

        ui.add_space(6.0);

        ui.group(|ui| {
            ui.label("Time step");
            two_col(ui, "Initial time step Δt (s)", |ui| {
                ui.add(
                    DragValue::new(&mut state.time_delta)
                        .speed(1e-7)
                        .clamp_range(1e-9..=1.0)
                        .min_decimals(6)
                        .max_decimals(9),
                );
            });
            // Adaptive time-stepping deferred — explicit Verlet doesn't
            // adapt without bigger changes; surfacing as a placeholder
            // would mislead the user. Note the future plan instead.
            ui.label("Adaptive time stepping: planned. Pick Δt below the material's CFL limit (≈ h/c).");
        });

        ui.add_space(6.0);

        ui.group(|ui| {
            ui.label("Runtime");
            ui.checkbox(&mut state.use_gpu_stresses, "Enable GPU acceleration (strain/stress)");
            ui.checkbox(&mut state.auto_export, "Auto-export CSV when duration is reached");
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button("Close").clicked() {
                state.engine_config_open = false;
            }
        });
    });
}

fn two_col(ui: &mut Ui, label: &str, add: impl FnOnce(&mut Ui)) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            add(ui);
        });
    });
}

fn optional_field(ui: &mut Ui, label: &str, slot: &mut Option<f32>, speed: f32, min: f32) {
    let mut enabled = slot.is_some();
    ui.horizontal(|ui| {
        ui.checkbox(&mut enabled, label);
        if enabled {
            let mut v = slot.unwrap_or(min.max(1.0));
            ui.add(DragValue::new(&mut v).speed(speed).clamp_range(min..=f32::MAX));
            *slot = Some(v);
        } else {
            *slot = None;
        }
    });
}
