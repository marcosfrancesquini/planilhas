#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_with_model_v3.py
Calibragem com presets + must-have para 1–10 e 20–30, usando model.pkl.
Uso:
  python predict_with_model_v3.py --predict planilha_2823.xlsx --model model.pkl --topk 50 --preset balanced
  # OU ajustar manual:
  python predict_with_model_v3.py --predict planilha_2823.xlsx --model model.pkl --topk 50 \
    --low-cap 8 --mid-cap 12 --high-min 22 --neighbor-penalty 0.45 --neighbor-radius 2 \
    --must10 2 --must20 3
"""

import argparse, pickle, sys, unicodedata
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# ---------- utils ----------
def strip_accents(s:str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
def norm_text(s) -> str:
    return strip_accents(str(s)).lower().strip()

def find_block_starts(df: pd.DataFrame):
    mask = []
    for _, row in df.astype(str).iterrows():
        row_norm = [norm_text(x) for x in row.values]
        has_nb = any(("numero base" in x) or ("numero" in x and "base" in x) or ("número base" in x) for x in row_norm)
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

        # tenta casar nomes normalizados
        norm_top = [norm_text(x) for x in top_cols]
        cols_for_presence = [c for c in b.columns if norm_text(c) in norm_top]
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

# seleção com cotas + must-have
def select_with_constraints(df_rank: pd.DataFrame, topk=50,
                            low_cap=8, mid_cap=12, high_min=22,
                            must10=2, must20=3,
                            neighbor_penalty=0.45, neighbor_radius=2):
    """
    - cotas: máx para 0–29 e 30–59; mínimo para 60–99
    - must-have: garantir pelo menos 'must10' em 1–10 e 'must20' em 20–30
    - penalização de vizinhos
    """
    selected = []
    low_cnt = mid_cnt = high_cnt = 0
    remaining = df_rank.copy()

    def bucket(v):
        return "low" if v<=29 else "mid" if v<=59 else "high"

    def in_1_10(v):  return 1 <= v <= 10
    def in_20_30(v): return 20 <= v <= 30

    # 1) pré-seleção obrigatória (must-have) se existir score
    def greedy_pick(filter_fn, needed):
        nonlocal remaining, selected, low_cnt, mid_cnt, high_cnt
        if needed <= 0: return
        pool = remaining[remaining["valor"].apply(filter_fn)].copy()
        while needed > 0 and not pool.empty and len(selected) < topk:
            # sem vizinhança na fase must-have (garante presença)
            choice = pool.sort_values(["score","valor"], ascending=[False,True]).iloc[0]
            v = int(choice["valor"])
            # respeita cotas (não bloquear totalmente low/mid)
            b = bucket(v)
            if (b=="low" and low_cnt>=low_cap) or (b=="mid" and mid_cnt>=mid_cap):
                pool = pool[pool["valor"]!=v]
                continue
            selected.append(v)
            if b=="low": low_cnt+=1
            elif b=="mid": mid_cnt+=1
            else: high_cnt+=1
            remaining = remaining[remaining["valor"]!=v]
            pool = pool[pool["valor"]!=v]
            needed -= 1

    greedy_pick(in_1_10, must10)
    greedy_pick(in_20_30, must20)

    # 2) seleção principal com vizinhança + cotas
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
            # não forço high_min aqui; controlo no final com preenchimento
            return True

        rem = rem[rem["valor"].apply(quota_ok)]
        if rem.empty:
            break

        choice = rem.sort_values(["score_adj","valor"], ascending=[False,True]).iloc[0]
        v = int(choice["valor"])
        selected.append(v)
        b = bucket(v)
        if b=="low": low_cnt+=1
        elif b=="mid": mid_cnt+=1
        else: high_cnt+=1
        remaining = remaining[remaining["valor"]!=v]

    # 3) se ainda não bateu high_min (60–99), completa com os melhores altos
    if high_cnt < high_min and len(selected) < topk:
        need = min(high_min - high_cnt, topk - len(selected))
        pool = remaining[remaining["valor"]>=60].copy()
        pool = pool.sort_values(["score","valor"], ascending=[False,True]).head(need)
        for _, row in pool.iterrows():
            v = int(row["valor"])
            selected.append(v); high_cnt += 1
            remaining = remaining[remaining["valor"]!=v]
            if len(selected) >= topk: break

    return selected

# ---------- presets ----------
PRESETS = {
    # meio-termo bom para suas observações
    "balanced": dict(low_cap=8, mid_cap=12, high_min=22, must10=2, must20=3,
                     neighbor_penalty=0.45, neighbor_radius=2),
    # puxa mais altos
    "high":     dict(low_cap=6, mid_cap=10, high_min=26, must10=1, must20=2,
                     neighbor_penalty=0.40, neighbor_radius=2),
    # espalha mais no baixo/médio
    "spread":   dict(low_cap=10, mid_cap=14, high_min=18, must10=3, must20=4,
                     neighbor_penalty=0.50, neighbor_radius=2),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--preset", choices=list(PRESETS.keys()))
    # overrides manuais
    ap.add_argument("--low-cap", type=int)
    ap.add_argument("--mid-cap", type=int)
    ap.add_argument("--high-min", type=int)
    ap.add_argument("--must10", type=int)
    ap.add_argument("--must20", type=int)
    ap.add_argument("--neighbor-penalty", type=float)
    ap.add_argument("--neighbor-radius", type=int)
    ap.add_argument("--outdir", default="out_predict")
    args = ap.parse_args()

    # carregar modelo
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # preset base
    cfg = PRESETS.get(args.preset, PRESETS["balanced"]).copy() if args.preset else PRESETS["balanced"].copy()
    # overrides
    for k in ["low_cap","mid_cap","high_min","must10","must20","neighbor_penalty","neighbor_radius"]:
        cli = getattr(args, k if k not in ("low_cap","mid_cap","high_min") else k.replace("_","-"), None)
    # ler overrides corretamente
    if args.low_cap is not None: cfg["low_cap"] = args.low_cap
    if args.mid_cap is not None: cfg["mid_cap"] = args.mid_cap
    if args.high_min is not None: cfg["high_min"] = args.high_min
    if args.must10 is not None: cfg["must10"] = args.must10
    if args.must20 is not None: cfg["must20"] = args.must20
    if args.neighbor_penalty is not None: cfg["neighbor_penalty"] = args.neighbor_penalty
    if args.neighbor_radius is not None: cfg["neighbor_radius"] = args.neighbor_radius

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print(f"[cfg] preset={args.preset or 'balanced'} | {cfg}")

    for xlsx in args.predict:
        df_rank = predict_scores_with_model(xlsx, model, verbose=True)
        top_vals = select_with_constraints(df_rank, topk=args.topk, **cfg)
        base = Path(xlsx).stem
        pd.DataFrame({"valor": top_vals}).to_csv(outdir / f"{base}_top{args.topk}.csv", index=False)
        print(f">>> {base}: {', '.join(map(str, top_vals))}")
        print(f"[ok] salvo em {outdir / f'{base}_top{args.topk}.csv'}")

if __name__ == "__main__":
    main()
