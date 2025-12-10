# Behavioral Profiling & Structural Inelasticity of Urban Parks

This repository contains the code and data for the study **"Beyond Administrative Boundaries: Real-time Behavioral Profiling and Structural Inelasticity of Urban Park Usage"**.

## Study Overview
We analyze real-time population data from 18 parks in Seoul to identify distinct behavioral usage patterns. Using XGBoost and SHAP analysis, we demonstrate that a park's usage type is primarily determined by its macro-urban context (transit, density) rather than internal facilities—a phenomenon we term **"Structural Inelasticity"**.

## Repository Structure

```
park-behavioral-clustering/
├── data/
│   ├── processed/
│   │   ├── final_dataset_18parks.csv          # Final input for modeling
│   │   └── enhanced_behavioral_clusters.csv   # Clustering results
│   ├── interim/
│   │   └── final_dataset_FULL.csv             # Raw feature set
│
├── notebooks/
│   ├── 01_Behavioral_Clustering.ipynb         # Phase 1: Identifying the 4 Clusters
│   ├── 02_Variable_Engineering.ipynb          # Phase 2: Feature Selection (20 variables)
│   ├── 03_Predictive_Modeling.ipynb           # Phase 3: XGBoost vs Baseline Models
│   └── 04_Interpretation_and_Simulation.ipynb # Phase 4: SHAP Analysis & Inelasticity
│
├── results/
│   ├── figures/                               # Key manuscript figures
│   └── tables/                                # Performance metrics & variable lists
│
├── writing/
│   ├── manuscript.tex                         # LaTeX Manuscript
│   └── image_prompts.md                       # Prompts for graphical abstract generation
│
└── requirements.txt                           # Python dependencies
```

## How to Replicate
1.  **Environment**: Install dependencies via `pip install -r requirements.txt`.
2.  **Run Order**: Execute notebooks `01` through `04` in sequence.
    *   *Note*: The data in `data/processed` is already the output of notebooks 01-02, so you can jump straight to `03_Predictive_Modeling.ipynb` to reproduce the XGBoost results.

## Key Findings
*   **Behavioral Clusters**: Identified 4 distinct types (Evening Urban, Afternoon Local, Mega Destination, Niche).
*   **Predictive Power**: XGBoost achieved **47.1% accuracy** (vs 25% random), proving physical environment predicts behavior.
*   **Structural Inelasticity**: Transit Accessibility (`transit_accessibility_index`) is the #1 predictor, outweighing internal facility counts.

## Contact
For questions, please open an issue in this repository.
