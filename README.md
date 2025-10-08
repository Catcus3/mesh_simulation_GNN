# Mesh Simulation GNN (Work in Progress)  
*A graph neural network surrogate for mesh-based physical simulation*

---

## Overview  
This project investigates whether **Graph Neural Networks (GNNs)** can serve as surrogate models for **mesh-based physical simulations** such as Finite Element Methods (FEM) or Computational Fluid Dynamics (CFD).  

Traditional solvers are accurate but computationally expensive, especially for large meshes or long time simulations. This project aims to **learn the physics from simulation data**, replacing parts of classical solvers with learned message-passing networks that approximate outputs much faster, while maintaining acceptable error bounds.

The goal is to generalise across mesh geometries, resolutions, and boundary conditions — laying the groundwork for neural PDE solvers, differentiable physics engines, and simulation acceleration in scientific and engineering domains.

---

## Core Idea  
Each mesh (triangular, tetrahedral, etc.) is converted into a **graph representation**:  
- **Nodes** represent vertices or elements.  
- **Edges** represent mesh connectivity or physical adjacency.  
- **Node & edge features** encode geometry, material properties, boundary flags, velocities, etc.  

The GNN applies **message passing** over this graph to predict physical quantities (displacement, stress, fluid velocity) either in one or multiple time steps.

---

## Technical Objectives  
1. **Mesh-to-Graph Conversion**  
   - Parse mesh file formats and build node/element connectivity.  
   - Construct `torch_geometric.data.Data` graphs with edge indices and features.  
   - Handle variable graph sizes and dynamic attributes.  

2. **GNN Model Architecture**  
   - Implement encoding, message passing (processing), and decoding blocks (inspired by MeshGraphNets) :contentReference[oaicite:0]{index=0}  
   - Use MLPs with relational input (node + edge features), skip connections, and normalization.  
   - Explore attention or hierarchical aggregation to capture long-range dependencies.

3. **Training & Loss Design**  
   - Train from ground-truth simulation trajectories (e.g. velocity, pressure, acceleration) :contentReference[oaicite:1]{index=1}  
   - Apply masking so loss is computed only over relevant node types (e.g. excluding walls) :contentReference[oaicite:2]{index=2}  
   - Normalize training targets using dataset statistics to stabilise learning.  

4. **Evaluation & Diagnostics**  
   - Compute node-level MSE / RMSE / relative error.  
   - Visualise error heatmaps and residual distributions on mesh geometry.  
   - Benchmark inference runtime vs baseline solvers as mesh scale grows.  

---

## Current Progress  
- Developed mesh → graph conversion modules and data loader scaffolding.  
- Built a prototype GNN model for single-step prediction.  
- Completed basic experiments on synthetic and canonical fluid meshes.  
- Set up modular training, validation, checkpointing, and loss infrastructure.

---

## Planned Next Steps  
- Implement **multi-step rollout prediction**, recursively feeding predictions as next-step inputs (following the MeshGraphNets temporal modelling approach).  
- Expand to **flow simulation domains**, e.g. using datasets like CylinderFlow (Navier-Stokes) to test generalisation across trajectories and mesh topologies :contentReference[oaicite:3]{index=3}  
- Integrate **masking strategies** to exclude wall or boundary nodes from loss, following practices in MeshGraphNets tutorial :contentReference[oaicite:4]{index=4}  
- Perform **hyperparameter scaling experiments**, varying number of message-passing layers, hidden dimensions, training length, and dataset size to study scaling behavior :contentReference[oaicite:5]{index=5}  
- Support **mesh resolution generalisation**: test trained models on unseen geometries or finer/coarser meshes.  
- Add attention or hierarchical GNN modules to improve long-range information flow.  
- Incorporate richer **visualisation tools** (e.g. VTK, ParaView, geometry-based error overlays).  
- Systematically compare **surrogate accuracy vs classical simulation cost** across domains and scales.  

---

## Broader Vision  
This project is positioned at the interface of machine learning and computational physics. Its ambition is to enable **fast, differentiable, and generalizable surrogate simulators**, with applications in:

- Engineering design and optimisation  
- Fluid dynamics modelling  
- Structural mechanics  
- Real-time physics for animation, robotics, and virtual environments  
