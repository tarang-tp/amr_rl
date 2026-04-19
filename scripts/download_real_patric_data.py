import sys
import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import re
import urllib.parse
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resistance.data_loaders.patric_loader import ECOLI_CIPRO_GENES

def download_real_patric_data():
    print("Downloading real E. coli vs Ciprofloxacin data from BV-BRC (PATRIC) API...")
    
    # 1. Fetch AMR Phenotypes
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[ 502, 503, 504 ])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    # Fetch phenotypes
    url = "https://www.bv-brc.org/api/genome_amr/"
    query = (
        "?eq(antibiotic,ciprofloxacin)"
        "&eq(taxon_id,562)"
        "&select(genome_id,genome_name,antibiotic,measurement_value,measurement_unit)"
        "&limit(5000)"
    )
    res = session.get(url + query, headers={"accept": "application/json"})
    
    if res.status_code != 200:
        print(f"Failed to fetch phenotypes: {res.status_code} {res.text}")
        return
        
    amr_data = res.json()
    print(f"Fetched {len(amr_data)} AMR records.")
    
    # Process measurement values to pure floats
    clean_data = []
    for row in amr_data:
        val_str = row.get("measurement_value", "")
        if not val_str: continue
        
        # Extract the first float from the string
        m = re.search(r"(\d+(\.\d+)?)", str(val_str))
        if m:
            clean_val = float(m.group(1))
            row["measurement_value"] = clean_val
            clean_data.append(row)
            
    df_amr = pd.DataFrame(clean_data)
    if df_amr.empty:
        print("No valid numerical MIC values found")
        return
        
    # Deduplicate by genome_id, take median MIC
    df_amr = df_amr.groupby("genome_id", as_index=False).agg({
        "genome_name": "first",
        "antibiotic": "first",
        "measurement_value": "median",
        "measurement_unit": "first"
    })
    
    genome_ids = df_amr["genome_id"].tolist()
    print(f"Found {len(genome_ids)} unique genomes with numerical MIC values.")
    
    # 2. Fetch genes for these genomes
    # To avoid URL too long, chunk genome_ids
    genes_data = []
    chunk_size = 100
    
    # Use ECOLI_CIPRO_GENES as generic search patterns across gene names
    gene_list_str = ",".join(urllib.parse.quote(g) for g in ECOLI_CIPRO_GENES)
    
    print("Fetching genomic features (CARD genes presence)...")
    for i in range(0, len(genome_ids), chunk_size):
        chunk = genome_ids[i:i+chunk_size]
        g_str = ",".join(chunk)
        
        # We query for features in these genomes that have 'AMR' or relevant gene names
        f_query = (
            f"?in(genome_id,({g_str}))"
            f"&in(gene,({gene_list_str}))"
            "&select(genome_id,gene)"
            "&limit(25000)"
        )
        try:
            f_res = session.get(
                "https://www.bv-brc.org/api/genome_feature/" + f_query,
                headers={"accept": "application/json"}
            )
            if f_res.status_code == 200:
                genes_data.extend(f_res.json())
        except Exception as e:
            print(f"Warning: chunk {i} failed: {e}")
            
    print(f"Found {len(genes_data)} relevant gene features across the genomes.")
    
    # 3. Create feature matrix
    df_genes = pd.DataFrame(genes_data)
    if not df_genes.empty:
        # Cross tabulate to get binary presence/absence
        df_presence = pd.crosstab(df_genes["genome_id"], df_genes["gene"])
        # ensure binary 1/0
        for col in df_presence.columns:
            df_presence[col] = (df_presence[col] > 0).astype(int)
    else:
        df_presence = pd.DataFrame(index=genome_ids) # all zeros
        df_presence.index.name = "genome_id"

    # Ensure all ECOLI_CIPRO_GENES exist as columns
    for g in ECOLI_CIPRO_GENES:
        if g not in df_presence.columns:
            df_presence[g] = 0
            
    # Keep only the requested genes
    df_presence = df_presence[ECOLI_CIPRO_GENES].reset_index()
            
    # 4. Merge results
    df_final = pd.merge(df_amr, df_presence, on="genome_id", how="left")
    df_final[ECOLI_CIPRO_GENES] = df_final[ECOLI_CIPRO_GENES].fillna(0).astype(int)
    df_final["species"] = "Escherichia coli"
    
    # Filter out empty or broken rows just in case
    df_final = df_final.dropna(subset=["measurement_value"])
    
    # 5. Save to TSV
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    out_tsv = data_dir / "patric_amr.tsv"
    df_final.to_csv(out_tsv, sep="\t", index=False)
    
    # Also save cardiovascular genes
    card_genes_path = data_dir / "card_genes.txt"
    with open(card_genes_path, "w") as f:
        for gene in ECOLI_CIPRO_GENES:
            f.write(gene + "\n")
            
    print("--------------------------------------------------")
    print(f"SUCCESS! Real database constructed and saved:")
    print(f" -> {out_tsv}")
    print(f" -> {len(df_final)} Isolates")
    print("--------------------------------------------------")

def dry_run():
    """Print stats about the existing data file without hitting the API."""
    tsv = Path("data/patric_amr.tsv")
    genes = Path("data/card_genes.txt")
    if tsv.exists():
        df = pd.read_csv(tsv, sep="\t")
        print(f"Dry run — existing data: {len(df)} isolates, {len(df.columns)} columns")
        print(f"  {tsv}")
    else:
        print(f"Dry run — {tsv} not found; run without --dry_run to download")
    if genes.exists():
        n = sum(1 for _ in open(genes))
        print(f"  {genes} ({n} genes)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip download; report existing data file stats")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        download_real_patric_data()
