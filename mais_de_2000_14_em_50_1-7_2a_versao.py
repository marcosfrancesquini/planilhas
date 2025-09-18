#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_detector_noleak_diverse.py
-------------------------------
Preditor de números "promissores" sem usar a coluna `numero base` (NO-LEAK),
com:
  - Priors de decil aprendidos no treino.
  - Espalhamento reforçado (60/75/90+).
  - Cotas por faixa (0–29, 30–59, 60–99).
  - Penalização de vizinhos para evitar valores consecutivos.

Uso:

1) Treinar com um arquivo que contenha `numero base`:
   python base_detector_noleak_diverse.py --train saiu.xlsx --model ./out/model.pkl

2) Predizer em arquivo sem `numero base`:
   python base_detector_noleak_diverse.py --predict saiu_sem_base.xlsx --model ./out/model.pkl --topk 50
"""

import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ===================== Utils =====================

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
            counts[key] += 1
            uniq.append(f"{key}_{counts[key]}")
        else:
            counts[key] = 0
            uniq.append(key)
    data.columns = uniq
    data = data.dropna(how="all").reset_index(drop=True)
    return data

def extract_blocks(df):
    starts = find_block_starts(df)
    blocks = []
    for i, s in enumerate(starts):
        e = starts[i+1] if i+1 < len(starts) else len(df)
        block_df = build_block(df, s, e)
        blocks.append({"start": s, "end": e, "data": block_df})
    return blocks

def decil_of_row(ridx, n):
    if n <= 0: return 1
    return int(np.ceil((ridx+1)/n*10))

# ===================== Priors =====================

BASE_WEIGHTS = np.array([1.0, -0.40, 0.10, 0.35, 0.30,
                         0.30, 0.10, 0.15, 0.60, -0.25])

SPREAD_60 = 0.20   # bônus 60–74
SPREAD_75 = 0.40   # bônus 75–89
SPREAD_90 = 0.60   # bônus 90–99

# ===================== Treino =====================

def train_model(train_paths):
    dec_counts = np.zeros(10, dtype=float)
    dec_total  = np.zeros(10, dtype=float)
    col_hits = defaultdict(float)
    col_tot  = defaultdict(int)

    for path in train_paths:
        df = pd.read_excel(path)
        blocks = extract_blocks(df)
        for b in blocks:
            dfb = b["data"]; n = len(dfb)
            if n == 0: continue
            nb_cols = [c for c in dfb.columns if normalize_name(c) == "numero base"]
            if not nb_cols: continue
            nb_mask = dfb[nb_cols[0]].notna().to_numpy()

            for ridx, is_nb in enumerate(nb_mask):
                dec = decil_of_row(ridx, n) - 1
                dec_total[dec] += 1
                if is_nb:
                    dec_counts[dec] += 1

            idx_nb = np.where(nb_mask)[0]
            if len(idx_nb) == 0: continue
            for col in dfb.columns:
                if col == nb_cols[0]: continue
                present = dfb[col].notna().to_numpy()[idx_nb].mean()
                col_hits[normalize_name(col)] += float(present)
                col_tot[normalize_name(col)]  += 1

    dec_prior = np.where(dec_total > 0, dec_counts / dec_total, 0.0)
    if dec_prior.sum() > 0:
        dec_prior = dec_prior / (np.abs(dec_prior).max() or 1.0)
    else:
        m = BASE_WEIGHTS - BASE_WEIGHTS.min()
        dec_prior = m / (m.max() or 1.0)

    col_reliability = {c: (col_hits[c] / max(1, col_tot[c])) for c in col_tot.keys()}

    return {"dec_prior": dec_prior, "col_reliability": col_reliability}

# ===================== Predição =====================

def predict_scores(xlsx_path, model, alpha=0.6):
    col_rel = model["col_reliability"]
    dec_prior = model["dec_prior"]
    df = pd.read_excel(xlsx_path)
    blocks = extract_blocks(df)

    top_cols = [c for c,_ in sorted(col_rel.items(), key=lambda x: x[1], reverse=True)[:6]]

    value_scores = defaultdict(float)
    for b in blocks:
        dfb = b["data"]; n = len(dfb)
        if n == 0: continue

        w = BASE_WEIGHTS.copy()
        w_norm = (w-w.min())/(w.max()-w.min()+1e-9)

        for ridx in range(n):
            dec = decil_of_row(ridx, n) - 1
            line_score = float(dec_prior[dec] * w_norm[dec])

            present_boost = 0.0
            for col in dfb.columns:
                cn = normalize_name(col)
                if cn in top_cols and pd.notna(dfb.iloc[ridx][col]):
                    present_boost += col_rel[cn]

            row_score = line_score * (1.0 + alpha * present_boost)

            for col in dfb.columns:
                cn = normalize_name(col)
                if cn not in top_cols: continue
                val = dfb.iloc[ridx][col]
                if pd.isna(val): continue
                try: v = int(float(str(val).strip()))
                except: continue
                if not (0 <= v <= 99): continue

                spread = 0.0
                if v >= 90: spread = SPREAD_90
                elif v >= 75: spread = SPREAD_75
                elif v >= 60: spread = SPREAD_60

                value_scores[v] += (row_score + spread)

    items = [{"valor": v, "score": s} for v, s in value_scores.items()]
    df_rank = pd.DataFrame(items).sort_values(["score","valor"], ascending=[False,True]).reset_index(drop=True)
    return df_rank

# ===================== Re-rank com diversidade =====================

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
                penalty = 0.0
                for s in sel_set:
                    if abs(v - s) <= neighbor_radius:
                        penalty += neighbor_penalty
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

        if high_cnt < high_min:
            rem_high = rem[rem["valor"]>=60]
            if not rem_high.empty:
                choice = rem_high.sort_values(["score_adj","valor"], ascending=[False,True]).iloc[0]
            else:
                choice = rem.sort_values(["score_adj","valor"], ascending=[False,True]).iloc[0]
        else:
            choice = rem.sort_values(["score_adj","valor"], ascending=[False,True]).iloc[0]

        v = int(choice["valor"])
        selected.append(v)
        if v<=29: low_cnt+=1
        elif v<=59: mid_cnt+=1
        else: high_cnt+=1

        remaining = remaining[remaining["valor"]!=v]

    return selected

# ===================== CLI =====================

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    ap = argparse.ArgumentParser(description="Preditor NO-LEAK com espalhamento, cotas e diversidade")
    ap.add_argument("--train", nargs="*", help="Arquivos .xlsx com `numero base` (treino)")
    ap.add_argument("--predict", nargs="*", help="Arquivos .xlsx sem `numero base` (predição)")
    ap.add_argument("--model", type=str, required=True, help="Caminho do modelo .pkl")
    ap.add_argument("--topk", type=int, default=50, help="Top-K candidatos")
    ap.add_argument("--outdir", type=str, default="out_noleak_diverse", help="Pasta de saída")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.train:
        model = train_model(args.train)
        save_model(model, args.model)
        print(f"[ok] Modelo salvo em {args.model}")

    if args.predict:
        model = load_model(args.model)
        for inp in args.predict:
            df_rank = predict_scores(inp, model)
            top_vals = rerank_with_diversity(df_rank, topk=args.topk)
            base = Path(inp).stem
            pd.DataFrame({"valor": top_vals}).to_csv(outdir / f"{base}_top{args.topk}.csv", index=False)
            print(f">>> {base}: {', '.join(map(str, top_vals))}")


if __name__ == "__main__":
    main()
