# 3D-SpatialViewBench: Evaluation of Egocentric 3D Reasoning in LLMs

[![Live Demo](https://img.shields.io/badge/Demo-Live%20Platform-blue)](https://3d-spatialviewbench-9zz43nkaiop8du9zzbjzau.streamlit.app/)
[![GitHub](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/SAVRUNLU/3D-SpatialViewBench)

3D-SpatialViewBench evaluates the capacity of text-only large language models to maintain and update a mental spatial coordinate system through rotation and translation. Unlike vision-language benchmarks, this task isolates symbolic geometric reasoning by providing precise coordinates and transformations in text.

## Core Methodology

The benchmark consists of 60 deterministic multiple-choice items divided into three complexity tiers:

- **Level 1 (Simple Rotations):** Single-axis yaw transitions mapping world vectors to camera-relative directions.
- **Level 2 (Translated Origins):** Combined translational shifts and rotations to probe egocentric origin handling.
- **Level 3 (Composite Transforms):** Sequential intrinsic compositions (Yaw $\rightarrow$ Pitch) requiring the application of sequential rotation matrices $\mathbf{R}_{final} = \mathbf{R}_y(\theta) \cdot \mathbf{R}_x(\phi)$.

Ground truth is derived via integer rotation matrices in homogeneous form. By restricting movements to $90^\circ$ increments, we ensure zero floating-point ambiguity and perfectly verifiable results.

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Data Generation
Produce the seeded 60-item dataset:
```bash
python generate_dataset.py
```

### 3. Model Evaluation
Configuration requires an API key for the evaluation harness:
```bash
export EVREN_API_KEY="your_key_here"
python evaluate.py
```
Outputs are written to `results/summary_metrics.csv` and `results/raw_predictions.json`.

### 4. Result Inspection
Launch the diagnostic dashboard:
```bash
streamlit run app.py
```

## Technical Specifications

| Metric | Value |
| :--- | :--- |
| **Items** | 60 MCQs |
| **Option Count** | 4 per item |
| **Key Balance** | A:15, B:13, C:13, D:19 |
| **Success Baseline** | $25\%$ (Random Guess) |
| **Coordinate Frame** | Right-Handed (+X: Right, +Y: Up, -Z: Forward) |
