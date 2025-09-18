#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd, unicodedata
from pathlib import Path

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def norm_text(s: str) -> str:
    return strip_accents(str(s)).lower().strip()

def main(xlsx):
    df = pd.read_excel(xlsx)
    # primeira linha como cabeçalho bruto
    header_row = df.iloc[0].tolist()
    raw = [str(x) if pd.notna(x) else "" for x in header_row]
    norm = [norm_text(x) for x in raw]
    print(f"[arquivo] {Path(xlsx).name}")
    print("=== Cabeçalho bruto (linha 1) ===")
    for i, v in enumerate(raw):
        print(f"{i:02d}: {v}")
    print("=== Cabeçalho normalizado ===")
    for i, v in enumerate(norm):
        print(f"{i:02d}: {v}")
    # também lista os nomes de colunas do pandas depois de drop de colunas vazias
    df2 = df.dropna(axis=1, how="all").reset_index(drop=True).copy()
    df2.columns = [f"col_{i}" for i in range(len(df2.columns))]
    print(f"colunas pandas fallback: {list(df2.columns)}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    args = ap.parse_args()
    main(args.xlsx)
