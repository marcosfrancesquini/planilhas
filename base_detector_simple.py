#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_detector_simple.py
-----------------------
Versão simplificada do preditor sem viés.

Uso:
    python base_detector_simple.py saiu2190.xlsx
"""

import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

# ======= CONFIG =======
BASE_WEIGHTS = np.array([1.0, -0.40, 0.10, 0.35, 0.30,
                         0.30, 0.10, 0.15, 0.60, -0.25])

SPREAD_60, SPREAD_75, SPREAD_90 = 0.20, 0.40, 0.60
OUTDIR = Path("out_simple")
OUTDIR.mkdir(exist_ok=True)

# ======= HELPERS =======
def normalize_name(s): 
    return str(s).strip().lower()

def find_block_starts(df):
    return df.index[df.astype(str).apply(
        lambda row: row.str.contains("numero base", case=False, na=False)
    ).any(axis=1)].tolist()

def build_block(df, start_idx, end_idx):
    header_vals = [str(x).strip() if pd.notna(x) else "" for x in df.iloc[start_idx].tolist()]
    data = df.iloc[start_idx+1:end_idx].copy()
    data = data.dropna(axis=1, how="all")
    headers = header_vals[:len(data.columns)]
    if len(headers) < len(data.columns):
        headers += [f"col_{i}" for i in range(len(headers), len(data.columns))]
    uniq, counts = [], {}
    for h in headers:
        key = h if h else "col"
        if key in counts:
            counts[key] += 1; uniq.append(f"{key}_{counts[key]}")
        else:
            counts[key] = 0; uniq.append(key)
    data.columns = uniq
    data = data.dropna(how="all").reset_index(drop=True)
    return data

def extract_blocks(df):
    starts = find_block_starts(df)
    blocks = []
    for i, s in enumerate(starts):
        e = starts[i+1] if i+1 < len(starts) else len(df)
        block_df = build_block(df, s, e)
        blocks.append(block_df)
    return blocks

def decil_of_row(ridx, n):
    if n <= 0: return 1
    return int(np.ceil((ridx+1)/n*10))

# ======= PREDICT =======
def predict_scores(path, alpha=0.6):
    df = pd.read_excel(path)
    blocks = extract_blocks(df)
    value_scores = defaultdict(float)

    for b in blocks:
        n = len(b)
        if n == 0: continue
        w = (BASE_WEIGHTS - BASE_WEIGHTS.min()) / (BASE_WEIGHTS.max() - BASE_WEIGHTS.min() + 1e-9)
        for ridx in range(n):
            dec = decil_of_row(ridx, n) - 1
            row_score = float(w[dec])
            for col in b.columns:
                val = b.iloc[ridx][col]
                if pd.isna(val): continue
                try: v = int(float(str(val).strip()))
                except: continue
                if 0 <= v <= 99:
                    spread = 0.0
                    if v >= 90: spread = SPREAD_90
                    elif v >= 75: spread = SPREAD_75
                    elif v >= 60: spread = SPREAD_60
                    value_scores[v] += row_score + spread
    items = [{"valor": v, "score": s} for v, s in value_scores.items()]
    df_rank = pd.DataFrame(items).sort_values(["score","valor"], ascending=[False,True]).reset_index(drop=True)
    return df_rank

def rerank_with_diversity(df_rank, topk=50, low_cap=10, mid_cap=14, high_min=18,
                          neighbor_penalty=0.35, neighbor_radius=2):
    selected = []
    low_cnt=mid_cnt=high_cnt=0
    remaining = df_rank.copy()
    def bucket(v):
        if v<=29: return "low"
        elif v<=59: return "mid"
        else: return "high"
    while len(selected) < topk and not remaining.empty:
        rem = remaining.copy()
        if selected:
            sel_set = set(selected)
            penal = []
            for v in rem["valor"]:
                penalty = sum(neighbor_penalty for s in sel_set if abs(v-s)<=neighbor_radius)
                penal.append(penalty)
            rem["score_adj"] = rem["score"] - np.array(penal)
        else:
            rem["score_adj"] = rem["score"]
        def quota_ok(v):
            b = bucket(v)
            if b=="low" and low_cnt>=low_cap: return False
            if b=="mid" and mid_cnt>=mid_cap: return False
            return True
        rem = rem[rem["valor"].apply(quota_ok)]
        if rem.empty: break
        choice = rem.sort_values(["score_adj","valor"], ascending=[False,True]).iloc[0]
        v = int(choice["valor"])
        selected.append(v)
        if v<=29: low_cnt+=1
        elif v<=59: mid_cnt+=1
        else: high_cnt+=1
        remaining = remaining[remaining["valor"]!=v]
    return selected

# ======= MAIN =======
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python base_detector_simple.py arquivo.xlsx")
        sys.exit(1)
    path = sys.argv[1]
    df_rank = predict_scores(path)
    top_vals = rerank_with_diversity(df_rank, topk=50)
    print(">>> Top-50 sugeridos:", ", ".join(map(str, top_vals)))
    out_path = OUTDIR / (Path(path).stem + "_top50.csv")
    pd.DataFrame({"valor": top_vals}).to_csv(out_path, index=False)
    print(f"[ok] Resultado salvo em {out_path}")
