use nalgebra::Vector3;

use super::{
    Axis, BoundaryCondition3D, BulkProps3D, Computer3D, Config3D, FailureCriteria3D,
    IsotropicProps3D, MaterialProps3D, run_steps,
};

#[test]
fn nan_explosion_detected() {
    let v = vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 0.0, 1.0),
    ];
    let t = vec![[0, 1, 2, 3]];
    let mut cfg = Config3D::default();
    let bulk = BulkProps3D {
        density: 7850.0,
        damping: 200.0,
        failure_criteria: FailureCriteria3D {
            strain_energy: Some(1.0e6),
            tensional_stress: Some(4.0e8),
            compressional_stress: Some(4.0e8),
        },
    };
    cfg.material = MaterialProps3D::Isotropic(IsotropicProps3D {
        elasticity_modulus: 2.0e11,
        poissons_ratio: 0.30,
        bulk,
    });
    cfg.time_delta_seconds = 1e-7;
    cfg.duration_seconds = 1.0;

    let mut c = Computer3D::from_mesh(&v, &t, cfg).unwrap();
    c.set_bc(
        &[3],
        BoundaryCondition3D::ConstantDisplacement {
            axes: Axis::ALL,
            displacement: [0.005, 0.0, 0.0],
            ramp_seconds: 5e-5,
        },
    );

    run_steps(&mut c, 1000);

    assert!(
        !c.has_nan_positions(),
        "Simulation diverged: NaN positions detected after 1000 steps",
    );
}
