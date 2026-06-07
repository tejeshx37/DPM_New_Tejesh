# DPM Simulator

**DPM Simulator** is a high-performance simulation application for Continuum-interaction Particle Dynamics (CPD) and Discrete Particle Modeling. It is built in Rust, utilizing a custom-built computational engine, and provides a modern graphical user interface using `eframe` and `egui`. The software relies on CGAL (Computational Geometry Algorithms Library) for robust 2D constrained Delaunay triangulation and meshing.

## 🚀 Overview

The simulator enables users to model complex geometric constraints, generate highly accurate meshes using exact algebraic numeric traits, apply boundary conditions and material properties, and run discrete-continuum hybrid simulations. It uses safe multi-threading via `rayon` to ensure maximal performance.

## 🏗️ Project Architecture (Workspace Crates)

The project is organized as a Cargo Workspace encompassing several dedicated crates:

### 1. `simulator` (Main Application)
The graphical frontend and orchestration layer.
- **`src/main.rs`**: Entry point that configures logging, initializes the window, and launches the `eframe` application.
- **`src/ui/`**: Contains the `egui`-based user interface code. It handles rendering menus, simulation viewport, parameter input panels, and plotting using `egui_plot`.
- **`src/model/`**: Contains the application state, managing the active mesh, the `cpd` configuration, and the simulation loop interactions.

### 2. `cpd` (Continuum interaction Particle Dynamics)
The core physical simulation engine.
- Contains modules for:
  - **`element.rs` & `node.rs`**: Defines the building blocks of the simulation mesh.
  - **`material/`**: Implements material behaviors, including `Isotropic`, `Orthotropic`, generic `BulkProps`, `ElasticityCondition`, and `FailureCriteria`.
  - **`boundary_condition.rs` & `boundary_average.rs`**: Handles kinematic boundary conditions and records resultant forces/displacements.
  - **`computer.rs`**: The main numerical solver running iterative computations to evolve the system over time.
- Uses `nalgebra` for heavy matrix math and `rayon` for parallelization.

### 3. `mesh`
The meshing subsystem.
- Responsible for taking a `PolygonWithHoles` (from the `cgal` crate) and triangulating it into a discrete `Mesh`.
- Converts boundaries and shapes into constrained Delaunay structures.
- Traces constraints (Lines, PolyLines from Ellipses) to precisely split boundaries according to a requested point count (`num_points`) and aspect ratio limits.

### 4. `cgal` (Safe Rust Wrapper)
A high-level, safe Rust wrapper around CGAL primitives.
- Provides ergonomic Rust interfaces for geometric structures: `Curve`, `RationalPoint`, `PolygonSet`, and `triangulation`.
- Translates CGAL's exact numeric types (`Rational`, `Algebraic`) into safe Rust semantics.

### 5. `cgal-sys` (C++ to Rust FFI Bridge)
The unsafe/raw FFI bindings to CGAL.
- Uses `cxx` to build a seamless bridge between Rust and C++.
- Contains `cpp/` header implementations (like `num.h`, `curve.h`, `traits.h`) which wrap native CGAL classes.
- A `build.rs` script compiles the C++ code and statically links the CGAL libraries into the Rust binary.

### 6. `boost-sys`
A bundled version of the Boost C++ library headers.
- CGAL fundamentally requires Boost (especially `boost::multiprecision` for algebraic calculations). This crate builds the necessary Boost headers and exports the include paths to `cgal-sys` during the build process, ensuring no system-level dependencies are strictly required.

### 7. `nalgebra-ext`
Extensions to the `nalgebra` linear algebra library.
- Provides additional utilities, custom serialization wrappers, or helper traits that the `cpd` simulator needs beyond the standard `nalgebra` offerings.

### 8. `function`
Shared mathematical functions and expression parsing.
- Likely provides runtime function evaluation or generalized curve functions used to define loading curves, boundary conditions, or constraints over time.

### 9. `build-utils`
Shared build-script utilities.
- Contains helper functions utilized by the various `build.rs` scripts within the workspace (such as managing C++ include paths and environment variables).

## 🛠️ Build Instructions

### Prerequisites
- **Rust Toolchain**: 1.70+ recommended (`cargo`).
- **CMake**: Required by `cxx` and some C++ build steps.
- **C++ Compiler**: A modern C++20 compliant compiler (Clang/GCC/MSVC).
- *(macOS users)*: Xcode Command Line Tools.

### Compiling
Run the following in the root directory to compile the entire workspace:

```bash
cargo build --release
```

*Note: The first build will take significantly longer as `cgal-sys` compiles the Boost headers and CGAL C++ code via `cxx`.*

### Running the Simulator
To launch the graphical application:

```bash
cargo run --release -p simulator
```

## ⚙️ How It Works (Simulation Pipeline)
1. **Geometry Definition**: The user defines geometric boundaries (linear polygons, circles) in the UI.
2. **Meshing**: The `mesh` crate sends these boundary constraints to `cgal` which invokes the C++ CGAL constrained Delaunay triangulator.
3. **Property Assignment**: The resulting points (nodes) and faces (elements) are fed into the `cpd` engine. Users assign materials (e.g., Isotropic elasticity) and kinematic boundary conditions (fixed displacements, forces).
4. **Time Integration**: The `computer` module iteratively advances the system in time, applying internal stiffness and external loads, solving the resulting kinematic equations.
5. **Visualization**: The `simulator` crate reads the updated node positions and stresses at each frame, rendering the deflecting mesh and outputting analytical plots.

## 📄 License
*Specify license here.*
