"""
scripts/download_data.py

Fetch E. coli ciprofloxacin AMR phenotype and resistance gene data
from the BV-BRC REST API (https://www.bv-brc.org/api/).

Outputs
-------
  data/processed/patric_ecoli_cipro.tsv  -- AMR phenotype records
  data/processed/card_cipro_genes.txt    -- resistance gene annotations per genome (TSV)

API endpoints
-------------
  AMR phenotype:   /api/genome_amr/
  Specialty genes: /api/sp_gene/

Pagination: cursor(*) initially, then cursor(TOKEN) from X-Cursor-Mark header.

Usage
-----
  python scripts/download_data.py              # hit the real API
  python scripts/download_data.py --dry_run    # synthetic files, no internet
  python scripts/download_data.py --limit 500  # small pull for testing
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BVBRC_BASE = "https://www.bv-brc.org/api"
AMR_ENDPOINT = f"{BVBRC_BASE}/genome_amr/"
GENE_ENDPOINT = f"{BVBRC_BASE}/sp_gene/"

AMR_FIELDS = [
    "genome_id", "genome_name", "antibiotic", "resistant_phenotype",
    "measurement_value", "measurement_unit", "measurement_sign",
]
GENE_FIELDS = [
    "genome_id", "gene", "product", "classification", "antibiotic",
]

PAGE_SIZE = 10_000

# EUCAST breakpoints for ciprofloxacin (mg/L)
EUCAST_SUSCEPTIBLE = 0.25
EUCAST_RESISTANT   = 0.5

_CIPRO_MIC_LADDER = [0.015, 0.03, 0.06, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


# ── API helpers ───────────────────────────────────────────────────────────────

def _rql(filters: str, fields: list[str], page_size: int, cursor: str) -> str:
    select = "select(" + ",".join(fields) + ")"
    limit  = f"limit({page_size},0)"
    # Base64 cursor tokens contain /, +, = — encode them so they survive in the URL
    safe_cursor = quote(cursor, safe="") if cursor != "*" else "*"
    cur    = f"cursor({safe_cursor})"
    return f"{filters}&{select}&{limit}&{cur}"


def _fetch_page(
    endpoint: str,
    rql: str,
    retries: int = 3,
    backoff: float = 2.0,
) -> tuple[pd.DataFrame, str | None]:
    """GET one page from BV-BRC; return (dataframe, next_cursor)."""
    url = f"{endpoint}?{rql}"
    headers = {"Accept": "text/csv"}

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=90)
            resp.raise_for_status()
            text = resp.text.strip()
            if not text:
                return pd.DataFrame(columns=["__empty__"]), None
            df = pd.read_csv(io.StringIO(text), low_memory=False)
            next_cursor = resp.headers.get("X-Cursor-Mark")
            return df, next_cursor
        except requests.RequestException as exc:
            if attempt < retries - 1:
                wait = backoff ** attempt
                log.warning(f"Request failed ({exc}), retrying in {wait:.0f}s…")
                time.sleep(wait)
            else:
                raise


def fetch_all(
    endpoint: str,
    base_filters: str,
    fields: list[str],
    page_size: int = PAGE_SIZE,
    max_records: int = 0,           # 0 = unlimited
) -> pd.DataFrame:
    """Paginate through BV-BRC cursor API and return full result as DataFrame."""
    cursor = "*"
    pages: list[pd.DataFrame] = []
    total = 0

    while True:
        rql = _rql(base_filters, fields, page_size, cursor)
        log.info(f"  cursor={cursor[:30]!r}…")
        df, next_cursor = _fetch_page(endpoint, rql)

        # Drop placeholder columns that sneak in on empty responses
        df = df[[c for c in df.columns if not c.startswith("__")]]
        if df.empty:
            break

        pages.append(df)
        total += len(df)
        log.info(f"  +{len(df):,} records (running total: {total:,})")

        if not next_cursor or next_cursor == cursor or len(df) < page_size:
            break
        if max_records and total >= max_records:
            log.info(f"  Reached max_records={max_records:,}, stopping.")
            break
        cursor = next_cursor

    return pd.concat(pages, ignore_index=True) if pages else pd.DataFrame(columns=fields)


def fetch_amr_data(limit: int = PAGE_SIZE) -> pd.DataFrame:
    log.info("Fetching AMR phenotype data (E. coli / ciprofloxacin)…")
    filters = "eq(antibiotic,ciprofloxacin)&eq(taxon_id,562)"
    return fetch_all(AMR_ENDPOINT, filters, AMR_FIELDS, page_size=limit)


def fetch_gene_data(limit: int = PAGE_SIZE, max_records: int = 250_000) -> pd.DataFrame:
    log.info("Fetching specialty gene data (E. coli antibiotic resistance)…")
    # eq(property,...) is unsupported by BV-BRC for this field — keyword() is required.
    # eq(taxon_id,562) scopes to E. coli; keyword filters to AR entries.
    # The full dataset is 3M+ rows; max_records caps the pull since coverage saturates
    # well before then (~70k unique genomes appear by 200k records).
    filters = "eq(taxon_id,562)&keyword(Antibiotic%20Resistance)"
    return fetch_all(GENE_ENDPOINT, filters, GENE_FIELDS,
                     page_size=limit, max_records=max_records)


# ── Dry-run synthetic data ────────────────────────────────────────────────────

def _make_synthetic_amr(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    from resistance.data_loaders.patric_loader import ECOLI_CIPRO_GENES  # noqa: F401
    rng = np.random.default_rng(seed)
    genome_ids = [f"1234567.{i}" for i in range(n)]

    # Roughly bell-shaped MIC distribution around 0.25 mg/L
    weights = [1, 2, 4, 6, 8, 8, 6, 4, 2, 1, 1, 1]
    probs = np.array(weights, dtype=float) / sum(weights)
    mic_values = rng.choice(_CIPRO_MIC_LADDER, size=n, p=probs)

    phenotypes = []
    for m in mic_values:
        if m <= EUCAST_SUSCEPTIBLE:
            phenotypes.append("Susceptible")
        elif m <= EUCAST_RESISTANT:
            phenotypes.append("Intermediate")
        else:
            phenotypes.append("Resistant")

    return pd.DataFrame({
        "genome_id":          genome_ids,
        "genome_name":        [f"Escherichia coli strain {i}" for i in range(n)],
        "antibiotic":         ["ciprofloxacin"] * n,
        "resistant_phenotype": phenotypes,
        "measurement_value":  mic_values,
        "measurement_unit":   ["mg/L"] * n,
        "measurement_sign":   ["="] * n,
    })


def _make_synthetic_genes(genome_ids: list[str], seed: int = 42) -> pd.DataFrame:
    from resistance.data_loaders.patric_loader import ECOLI_CIPRO_GENES
    rng = np.random.default_rng(seed)
    rows = []
    for gid in genome_ids:
        # Each isolate carries 0–5 resistance genes (Poisson-ish)
        n_genes = int(np.clip(rng.poisson(2), 0, len(ECOLI_CIPRO_GENES)))
        if n_genes == 0:
            continue
        chosen = rng.choice(ECOLI_CIPRO_GENES, size=n_genes, replace=False)
        for gene in chosen:
            rows.append({
                "genome_id":      gid,
                "gene":           gene,
                "product":        f"{gene} protein",
                "classification": "fluoroquinolone resistance",
                "antibiotic":     "ciprofloxacin",
            })
    if not rows:
        return pd.DataFrame(columns=GENE_FIELDS)
    return pd.DataFrame(rows)


def dry_run(amr_path: Path, gene_path: Path, n: int = 2000) -> None:
    log.info(f"Dry run: generating {n} synthetic isolates (no API calls)…")
    amr_df  = _make_synthetic_amr(n)
    gene_df = _make_synthetic_genes(amr_df["genome_id"].tolist())
    amr_path.parent.mkdir(parents=True, exist_ok=True)
    amr_df.to_csv(amr_path,  sep="\t", index=False)
    gene_df.to_csv(gene_path, sep="\t", index=False)
    log.info(f"Wrote {amr_path}  ({len(amr_df):,} rows)")
    log.info(f"Wrote {gene_path} ({len(gene_df):,} rows)")
    _print_summary(amr_df, gene_df)


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(amr_df: pd.DataFrame, gene_df: pd.DataFrame) -> None:
    n = len(amr_df)
    if n == 0:
        log.info("No AMR records.")
        return

    mics = amr_df["measurement_value"].dropna().astype(float)
    med  = mics.median()
    q1   = mics.quantile(0.25)
    q3   = mics.quantile(0.75)

    n_s  = int((mics <= EUCAST_SUSCEPTIBLE).sum())
    n_r  = int((mics >  EUCAST_RESISTANT).sum())
    n_i  = n - n_s - n_r

    n_genes = gene_df["gene"].nunique() if not gene_df.empty else 0

    log.info("=" * 55)
    log.info(f"  Isolates:      {n:,}")
    log.info(f"  MIC median:    {med:.4f} mg/L  (IQR {q1:.4f}–{q3:.4f})")
    log.info(f"  EUCAST S<=0.25: {n_s:,}  I: {n_i:,}  R>0.5: {n_r:,}")
    log.info(f"  Unique genes:  {n_genes}")
    log.info("=" * 55)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download BV-BRC AMR + specialty gene data for E. coli ciprofloxacin"
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="Generate synthetic files without API calls")
    parser.add_argument("--limit", type=int, default=PAGE_SIZE,
                        help=f"Records per API page (default {PAGE_SIZE})")
    parser.add_argument("--amr_out",  type=str,
                        default="data/processed/patric_ecoli_cipro.tsv")
    parser.add_argument("--gene_out", type=str,
                        default="data/processed/card_cipro_genes.txt")
    parser.add_argument("--n_synthetic", type=int, default=2000,
                        help="Synthetic isolate count for --dry_run")
    parser.add_argument("--gene_max", type=int, default=250_000,
                        help="Max gene records to fetch (0=unlimited, default 250k)")
    args = parser.parse_args()

    amr_path  = Path(args.amr_out)
    gene_path = Path(args.gene_out)
    amr_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        dry_run(amr_path, gene_path, n=args.n_synthetic)
        return

    amr_df  = fetch_amr_data(limit=args.limit)
    gene_df = fetch_gene_data(limit=args.limit, max_records=args.gene_max)

    if amr_df.empty:
        log.error("No AMR records returned — check API connectivity.")
        sys.exit(1)

    amr_df.to_csv(amr_path,  sep="\t", index=False)
    gene_df.to_csv(gene_path, sep="\t", index=False)
    log.info(f"Saved {amr_path}  ({len(amr_df):,} records)")
    log.info(f"Saved {gene_path} ({len(gene_df):,} records)")
    _print_summary(amr_df, gene_df)


if __name__ == "__main__":
    main()
