import nbformat as nbf
import os

# Paths
BASE_DIR = r"c:\Users\jour\Documents\GitHub\accessibility_park-1\park-behavioral-clustering"
SRC_DIR = os.path.join(BASE_DIR, "src")
OUTPUT_NB = os.path.join(SRC_DIR, "analysis_park_behavioral_profiling.ipynb")

# Read scripts
def read_script(filename):
    with open(os.path.join(SRC_DIR, filename), 'r', encoding='utf-8') as f:
        return f.read()

script_1 = read_script("run_clustering.py")
script_2 = read_script("run_predictive_modeling.py")
script_3 = read_script("run_interpretation.py")

# Create Notebook
nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("# Park Behavioral Profiling & Structural Diagnosis\n\nThis notebook reproduces the analysis for:\n1. Behavioral Clustering (K-Means, k=5)\n2. Predictive Modeling (LOOCV)\n3. Model Interpretation (SHAP, PDP)\n4. Structural Diagnosis (Efficiency Index)"),
    
    nbf.v4.new_markdown_cell("## 1. Behavioral Clustering"),
    nbf.v4.new_code_cell(script_1),
    
    nbf.v4.new_markdown_cell("## 2. Predictive Modeling (LOOCV)"),
    nbf.v4.new_code_cell(script_2),
    
    nbf.v4.new_markdown_cell("## 3. Interpretation & Diagnosis"),
    nbf.v4.new_code_cell(script_3)
]

# Write
with open(OUTPUT_NB, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook created at {OUTPUT_NB}")
