# src/mimic_paths.py (From MIMIC-III)
import os

def find_file(root_dir: str, filename: str) -> str:
    """
    Find a file anywhere under root_dir (handles nested folders).
    Returns the first match.
    """
    for root, _, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"Could not find {filename} under: {root_dir}")

def get_mimic_paths(mimic_dir: str) -> dict:
    """
    Returns paths to key MIMIC CSVs.
    """
    return {
        "ADMISSIONS": find_file(mimic_dir, "ADMISSIONS.csv"),
        "TRANSFERS": find_file(mimic_dir, "TRANSFERS.csv"),
        "ICUSTAYS": find_file(mimic_dir, "ICUSTAYS.csv"),
        "CALLOUT": find_file(mimic_dir, "CALLOUT.csv"),
        "DIAGNOSES_ICD": find_file(mimic_dir, "DIAGNOSES_ICD.csv"),
        "EDSTAYS": find_file(mimic_dir,"edstays.csv"),
    }