#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_with_model.py v1.2
--------------------------
Predição NO-LEAK usando modelo pré-treinado (model.pkl), com:
- Priors de decil + confiabilidade de colunas do modelo
- Espalhamento 60/75/90+
- Cotas por faixa + penalização de vizinhos (determinístico)
- Detecção de blocos robusta (aceita "numero base" e "número base")
- Fallbacks e mensagens de diagnóstico

Uso:
  python predict_with_model.py --predict planilha.xlsx --model model.pkl --topk 50
  python predict_with_model.py --predict a.xlsx b.xlsx --model model.pkl --topk 50

Requisitos:
  pip install pandas numpy openpyxl
"""

import argparse, pickle, sys, unicodedata
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# ---------------- utils ----------------
def strip_accents(s:str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def norm_text(s) -> str:
    return strip_accents(str(s)).lower().strip()

def find_block_starts(df: pd.DataFrame):
    # considera "numero base" com ou sem acento, em qualquer coluna da linha
    mask = []
    for _, row in df.astype(str).iterrows():
        row_norm = [norm_text(x) for x in row.values]
        has_nb = any(("numero base" in x) or ("numero" in x and "base" in x) for x in row_norm)
        mask.append(has_nb)
    return list(df.index[np.array(mask, dtype=bool)])

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
        blocks.append(build_block(df, s, e))
    if not blocks:
        # fallback: trata arquivo inteiro como 1 bloco sem cabeçalho estruturado
        data = df.dropna(axis=1, how="all").reset_index(drop=True).copy()
        data.columns = [f"col_{i}" for i in range(len(data.columns))]
        blocks = [data]
    return blocks

def decil_of_row(ridx, n):
    if n<=0: return 1
    return int(np.ceil((ridx+1)/n*10))

BASE_DECIL_WEIGHTS = np.array([1.00, -0.40, 0.10, 0.35, 0.30,
                               0.30, 0.10, 0.15, 0.60, -0.25], dtype=float)
SPREAD_60, SPREAD_75, SPREAD_90 = 0.20, 0.40, 0.60

def predict_scores_with_model(xlsx_path: str, model: dict, alpha: float = 0.6, verbose=True):
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        raise RuntimeError(f"Falha ao abrir {xlsx_path}: {e}")

    blocks = extract_blocks(df)
    if verbose:
        print(f"[diag] {Path(xlsx_path).name}: blocos detectados = {len(blocks)}")

    dec_prior = np.array(model.get("dec_prior", np.zeros(10)))
    col_rel   = model.get("col_reliability", {})
    if dec_prior.shape[0] != 10:
        m = BASE_DECIL_WEIGHTS - BASE_DECIL_WEIGHTS.min()
        dec_prior = m / (m.max() or 1.0)

    # top colunas mais confiáveis
    top_cols = [c for c,_ in sorted(col_rel.items(), key=lambda x: x[1], reverse=True)[:6]]
    if verbose:
        print(f"[diag] top_cols (modelo): {top_cols[:6]}")

    value_scores = defaultdict(float)
    cells_0_99 = 0
    for bi, b in enumerate(blocks, 1):
        n = len(b)
        if n == 0: 
            continue
        w = BASE_DECIL_WEIGHTS
        w_norm = (w - w.min())/(w.max() - w.min() + 1e-9)

        cols_for_presence = [c for c in b.columns if norm_text(c) in [norm_text(x) for x in top_cols]]
        use_all_cols_for_values = False
        if not cols_for_presence:
            use_all_cols_for_values = True
            if verbose:
                print(f"[diag] bloco {bi}: nenhuma top_col encontrada; usando TODAS as colunas para extrair 0..99")

        for ridx in range(n):
            dec = decil_of_row(ridx, n) - 1
            line_score = float(dec_prior[dec] * w_norm[dec])

            present_boost = 0.0
            if cols_for_presence:
                for col in cols_for_presence:
                    if pd.notna(b.iloc[ridx][col]):
                        present_boost += col_rel.get(norm_text(col), 0.0)

            row_score = line_score * (1.0 + alpha * present_boost)

            cols_for_values = b.columns if use_all_cols_for_values else cols_for_presence
            if len(cols_for_values)==0:
                cols_for_values = b.columns

            for col in cols_for_values:
                val = b.iloc[ridx][col]
                if pd.isna(val): 
                    continue
                try:
                    v = int(float(str(val).strip()))
                except:
                    continue
                if 0 <= v <= 99:
                    cells_0_99 += 1
                    spread = SPREAD_90 if v>=90 else SPREAD_75 if v>=75 else SPREAD_60 if v>=60 else 0.0
                    value_scores[v] += row_score + spread

    if verbose:
        print(f"[diag] células 0..99 consideradas: {cells_0_99}")

    items = [{"valor": v, "score": s} for v, s in value_scores.items()]
    if not items:
        raise RuntimeError("Nenhum valor 0..99 extraído. Verifique o formato da planilha.")
    df_rank = pd.DataFrame(items).sort_values(["score","valor"], ascending=[False,True]).reset_index(drop=True)
    return df_rank

def rerank_with_diversity(df_rank: pd.DataFrame, topk=50,
                          low_cap=10, mid_cap=14, high_min=18,
                          neighbor_penalty=0.35, neighbor_radius=2):
    selected = []
    low_cnt=mid_cnt=high_cnt=0
    remaining = df_rank.copy()

    def bucket(v):
        return "low" if v<=29 else "mid" if v<=59 else "high"

    while len(selected) < topk and not remaining.empty:
        rem = remaining.copy()
        if selected:
            sel_set = set(selected)
            rem["score_adj"] = rem.apply(
                lambda r: r["score"] - sum(neighbor_penalty for s in sel_set if abs(int(r["valor"])-int(s))<=neighbor_radius),
                axis=1
            )
        else:
            rem["score_adj"] = rem["score"]

        def quota_ok(v):
            b = bucket(int(v))
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

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--outdir", default="out_predict")
    args = ap.parse_args()

    # carregar modelo
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    for xlsx in args.predict:
        df_rank = predict_scores_with_model(xlsx, model, verbose=True)
        top_vals = rerank_with_diversity(df_rank, topk=args.topk)
        base = Path(xlsx).stem
        pd.DataFrame({"valor": top_vals}).to_csv(outdir / f"{base}_top{args.topk}.csv", index=False)
        print(f">>> {base}: {', '.join(map(str, top_vals))}")
        print(f"[ok] salvo em {outdir / f'{base}_top{args.topk}.csv'}")

if __name__ == "__main__":
    main()
