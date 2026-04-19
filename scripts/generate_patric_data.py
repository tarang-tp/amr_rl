import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resistance.data_loaders.patric_loader import ECOLI_CIPRO_GENES

def generate_patric_data(n_isolates=2500):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 1. Write card_genes.txt
    card_genes_path = data_dir / "card_genes.txt"
    with open(card_genes_path, "w") as f:
        for gene in ECOLI_CIPRO_GENES:
            f.write(gene + "\n")
    print(f"Generated {card_genes_path}")
    
    # 2. Generate TSV
    rng = np.random.default_rng(42)
    
    genes_matrix = rng.integers(0, 2, size=(n_isolates, len(ECOLI_CIPRO_GENES)))
    
    # Simulate MIC using the same logic as MockPATRICLoader
    gene_effects = rng.uniform(0.5, 2.0, size=len(ECOLI_CIPRO_GENES))
    log_mic = genes_matrix @ gene_effects + rng.normal(0, 0.3, size=n_isolates)
    log_mic = np.clip(log_mic, 0, 12)
    mic_values = np.expm1(log_mic) # inverse of log1p used in load()
    
    df = pd.DataFrame(genes_matrix, columns=ECOLI_CIPRO_GENES)
    df["genome_id"] = [f"1000.{i}" for i in range(1, n_isolates + 1)]
    df["antibiotic"] = "ciprofloxacin"
    df["measurement_value"] = mic_values
    df["measurement_unit"] = "mg/L"
    df["species"] = "Escherichia coli"
    
    tsv_path = data_dir / "patric_amr.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"Generated {tsv_path} with {len(df)} isolates")

if __name__ == "__main__":
    generate_patric_data()
