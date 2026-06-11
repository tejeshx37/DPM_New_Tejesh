#[test]
fn test_nan_explosion() {
    let v = vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 0.0, 1.0),
    ];
    let t = vec![[0, 1, 2, 3]];
    let mut cfg = Config3D::default();
    
    // Steel (mild)
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
    // Constant displacement on node 3
    c.set_bc(
        &[3],
        BoundaryCondition3D::ConstantDisplacement {
            axes: Axis::ALL,
            displacement: [0.005, 0.0, 0.0],
            ramp_seconds: 5e-5,
        },
    );
    
    run_steps(&mut c, 1000);
    
    for n in &c.nodes {
        assert!(!n.position.x.is_nan(), "Position X is NaN!");
        assert!(!n.position.y.is_nan(), "Position Y is NaN!");
        assert!(!n.position.z.is_nan(), "Position Z is NaN!");
    }
}
