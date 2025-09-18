#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_with_model_v4_super.py
- Matching robusto por aliases/normalização
- Suporte a --use-cols "colA;colB;colC" para forçar colunas (match por substring normalizada)
- Suporte a --force-smart: seleciona automaticamente colunas "numéricas 0..99" (>=30% das células são inteiros 0..99)
- Presets: balanced, high, high15, combo, combo_vintage
"""
import argparse, pickle, unicodedata
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def norm_text(s: str) -> str:
    return strip_accents(str(s)).lower().strip()

def find_block_starts(df: pd.DataFrame):
    starts = []
    for idx, row in df.astype(str).iterrows():
        row_norm = [norm_text(x) for x in row.values]
        has_nb = any(("numero base" in x) or ("número base" in x) or ("numero" in x and "base" in x) for x in row_norm)
        if has_nb:
            starts.append(idx)
    return starts

def build_block(df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    header_vals = [str(x).strip() if pd.notna(x) else "" for x in df.iloc[start_idx].tolist()]
    data = df.iloc[start_idx + 1:end_idx].copy()
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

def extract_blocks(df: pd.DataFrame):
    starts = find_block_starts(df)
    blocks = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(df)
        blocks.append(build_block(df, s, e))
    if not blocks:
        data = df.dropna(axis=1, how="all").reset_index(drop=True).copy()
        data.columns = [f"col_{i}" for i in range(len(data.columns))]
        blocks = [data]
    return blocks

BASE_DECIL_WEIGHTS = np.array([1.00, -0.40, 0.10, 0.35, 0.30, 0.30, 0.10, 0.15, 0.60, -0.25], dtype=float)
SPREAD_60, SPREAD_75, SPREAD_90 = 0.20, 0.40, 0.60
def decil_of_row(ridx: int, nrows: int) -> int:
    if nrows <= 0: return 1
    return int(np.ceil((ridx + 1) / nrows * 10))

ALIASES = {
    "máxima freq. nos blocos completos": ["maxima", "freq", "blocos", "completos"],
    "mínima freq. nos blocos completos": ["minima", "freq", "blocos", "completos"],
    "frequência no último bloco incompleto": ["frequencia", "ultimo", "bloco", "incompleto"],
    "número": ["numero", "número", "num"]
}

def looks_numeric_0_99(series: pd.Series, threshold=0.30) -> bool:
    vals = 0; ok = 0
    for x in series.dropna().astype(str).tolist():
        vals += 1
        try:
            v = int(float(x.strip()))
            if 0 <= v <= 99: ok += 1
        except:
            pass
    return vals>0 and (ok/vals)>=threshold

def match_model_columns(block_df: pd.DataFrame, col_reliability: dict, force_cols=None, force_smart=False):
    cols = block_df.columns.tolist()
    norm_cols = {c: norm_text(c) for c in cols}

    # 0) --use-cols (prioridade)
    if force_cols:
        targets = [norm_text(x) for x in force_cols]
        matched = [c for c in cols if any(t in norm_cols[c] for t in targets)]
        if matched:
            return matched

    # 1) match direto por igualdade com chaves do modelo
    model_keys_norm = {norm_text(k): v for k, v in col_reliability.items()}
    direct = [c for c in cols if norm_cols[c] in model_keys_norm]
    if direct:
        return direct

    # 2) match por aliases (todas as palavras-chave presentes)
    alias_match = []
    for c in cols:
        nc = norm_cols[c]
        for canon, words in ALIASES.items():
            if all(w in nc for w in words):
                alias_match.append(c); break
    if alias_match:
        return alias_match

    # 3) heurística: colunas com muitos números 0..99
    if force_smart:
        smart = [c for c in cols if looks_numeric_0_99(block_df[c], threshold=0.30)]
        if smart:
            return smart

    # 4) fallback: vazio (caller decide usar todas)
    return []

def predict_scores_with_model(xlsx_path: str, model_dict: dict, alpha: float = 0.6, verbose: bool = True,
                              use_cols_cli=None, force_smart=False) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    blocks = extract_blocks(df)
    if verbose:
        print(f"[diag] {Path(xlsx_path).name}: blocos = {len(blocks)}")

    dec_prior = np.array(model_dict.get("dec_prior", np.zeros(10)))
    col_rel = model_dict.get("col_reliability", {})
    if dec_prior.shape[0] != 10:
        m = BASE_DECIL_WEIGHTS - BASE_DECIL_WEIGHTS.min()
        dec_prior = m / (m.max() or 1.0)

    value_scores = defaultdict(float)

    for bi, b in enumerate(blocks, 1):
        n = len(b)
        if n == 0: 
            continue
        w = BASE_DECIL_WEIGHTS
        w_norm = (w - w.min())/(w.max() - w.min() + 1e-9)

        cols_for_presence = match_model_columns(b, col_rel, force_cols=use_cols_cli, force_smart=force_smart)

        use_all_cols_for_values = False
        if not cols_for_presence:
            use_all_cols_for_values = True
            if verbose:
                print(f"[diag] bloco {bi}: sem match -> usando TODAS as colunas")
        else:
            if verbose:
                print(f"[diag] bloco {bi}: colunas usadas: {cols_for_presence[:8]}{'...' if len(cols_for_presence)>8 else ''}")

        for ridx in range(n):
            dec = decil_of_row(ridx, n) - 1
            line_score = float(dec_prior[dec] * w_norm[dec])

            present_boost = 0.0
            if cols_for_presence:
                for col in cols_for_presence:
                    if col in b.columns and pd.notna(b.iloc[ridx][col]):
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
                    spread = SPREAD_90 if v>=90 else SPREAD_75 if v>=75 else SPREAD_60 if v>=60 else 0.0
                    value_scores[v] += row_score + spread

    items = [{"valor": v, "score": s} for v, s in value_scores.items()]
    if not items:
        raise RuntimeError("Nenhum valor 0..99 extraído. Verifique a planilha.")
    df_rank = pd.DataFrame(items).sort_values(["score","valor"], ascending=[False,True]).reset_index(drop=True)
    return df_rank

# Seletores iguais ao patched anterior
def select_balanced(df_rank: pd.DataFrame, topk: int) -> list[int]:
    return df_rank.sort_values(["score","valor"], ascending=[False,True]).head(topk)["valor"].astype(int).tolist()

def select_high(df_rank: pd.DataFrame, topk: int, min_ge40: int = 28) -> list[int]:
    ordered = df_rank.sort_values(["score","valor"], ascending=[False,True])
    high = ordered[ordered["valor"]>=40]["valor"].astype(int).tolist()
    res = high[:topk]
    if len(res)<topk:
        fill = [v for v in ordered["valor"].astype(int).tolist() if v not in res]
        res += fill[:(topk-len(res))]
    ge40 = sum(1 for v in res if v>=40)
    if ge40 < min_ge40:
        pool = [v for v in high if v not in res]
        need = min(min_ge40 - ge40, topk - len(res))
        res += pool[:need]
        res = res[:topk]
    return res

def select_with_constraints(df_rank: pd.DataFrame, topk=50,
                            low_cap=6, mid_cap=10, high_min=28,
                            must10=1, must20=2, must30=2,
                            neighbor_penalty=0.50, neighbor_radius=2,
                            min_ge40=14) -> list[int]:
    selected = []
    low_cnt = mid_cnt = high_cnt = 0
    remaining = df_rank.copy()

    def bucket(v):
        return "low" if v<=29 else "mid" if v<=59 else "high"
    def in_1_10(v):  return 1 <= v <= 10
    def in_10_19(v): return 10 <= v <= 19
    def in_20_29(v): return 20 <= v <= 29

    def greedy_pick(filter_fn, needed):
        nonlocal remaining, selected, low_cnt, mid_cnt, high_cnt
        if needed <= 0: return
        pool = remaining[remaining["valor"].apply(filter_fn)].copy().sort_values(["score","valor"], ascending=[False,True])
        while needed>0 and not pool.empty and len(selected)<topk:
            v = int(pool.iloc[0]["valor"]); pool = pool.iloc[1:]
            b = bucket(v)
            if b=="low" and low_cnt>=low_cap: continue
            if b=="mid" and mid_cnt>=mid_cap: continue
            selected.append(v)
            if b=="low": low_cnt+=1
            elif b=="mid": mid_cnt+=1
            else: high_cnt+=1
            remaining = remaining[remaining["valor"]!=v]
            needed -= 1

    greedy_pick(in_1_10, must10)
    greedy_pick(in_10_19, must20)
    greedy_pick(in_20_29, must30)

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
        b = bucket(v)
        if b=="low": low_cnt+=1
        elif b=="mid": mid_cnt+=1
        else: high_cnt+=1
        remaining = remaining[remaining["valor"]!=v]

    if high_cnt < high_min and len(selected) < topk:
        need = min(high_min - high_cnt, topk - len(selected))
        pool = remaining[remaining["valor"]>=60].copy().sort_values(["score","valor"], ascending=[False,True]).head(need)
        for _, row in pool.iterrows():
            v = int(row["valor"])
            selected.append(v); high_cnt += 1
            remaining = remaining[remaining["valor"]!=v]
            if len(selected) >= topk: break

    ge40_have = sum(1 for v in selected if v >= 40)
    if ge40_have < min_ge40 and len(selected) < topk:
        need = min(min_ge40 - ge40_have, topk - len(selected))
        pool = remaining[remaining["valor"]>=40].copy().sort_values(["score","valor"], ascending=[False,True]).head(need)
        for _, row in pool.iterrows():
            v = int(row["valor"])
            selected.append(v)
            remaining = remaining[remaining["valor"]!=v]
            if len(selected) >= topk: break

    return selected[:topk]

def select_combo_vintage(df_rank: pd.DataFrame, topk=50,
                         min_ge40=30, high_min=32, low_cap=5, mid_cap=9) -> list[int]:
    ordered = df_rank.sort_values(["score","valor"], ascending=[False,True]).reset_index(drop=True)

    def bucket(v):
        return "low" if v<=29 else "mid" if v<=59 else "high"
    selected, low_cnt, mid_cnt, high_cnt = [], 0, 0, 0

    for _, row in ordered.iterrows():
        v = int(row["valor"])
        b = bucket(v)
        if b=="low" and low_cnt>=low_cap:  continue
        if b=="mid" and mid_cnt>=mid_cap:  continue
        selected.append(v)
        if b=="low": low_cnt+=1
        elif b=="mid": mid_cnt+=1
        else: high_cnt+=1
        if len(selected) >= topk: break

    if high_cnt < high_min and len(selected) < topk:
        pool = ordered[ordered["valor"]>=60]
        for _, row in pool.iterrows():
            v = int(row["valor"])
            if v not in selected:
                selected.append(v); high_cnt += 1
                if len(selected) >= topk or high_cnt >= high_min: break

    ge40_have = sum(1 for v in selected if v >= 40)
    if ge40_have < min_ge40 and len(selected) < topk:
        pool = ordered[ordered["valor"]>=40]
        for _, row in pool.iterrows():
            v = int(row["valor"])
            if v not in selected:
                selected.append(v)
                if len(selected) >= topk: break

    return selected[:topk]

def run_preset(df_rank: pd.DataFrame, preset: str, topk: int,
               low_cap=6, mid_cap=10, high_min=28, must10=1, must20=2, must30=2,
               neighbor_penalty=0.50, neighbor_radius=2, min_ge40=14):
    preset = preset.lower()
    if preset == "balanced":
        return select_balanced(df_rank, topk)
    if preset == "high":
        return select_high(df_rank, topk, min_ge40=max(min_ge40, 28))
    if preset == "high15":
        return select_with_constraints(df_rank, topk=topk,
                                       low_cap=max(low_cap, 6), mid_cap=max(mid_cap, 10), high_min=max(high_min, 30),
                                       must10=max(must10, 1), must20=max(must20, 2), must30=max(must30, 2),
                                       neighbor_penalty=max(neighbor_penalty, 0.65), neighbor_radius=2,
                                       min_ge40=max(min_ge40, 14))
    if preset == "combo":
        return select_with_constraints(df_rank, topk=topk,
                                       low_cap=low_cap, mid_cap=mid_cap, high_min=high_min,
                                       must10=must10, must20=must20, must30=must30,
                                       neighbor_penalty=neighbor_penalty, neighbor_radius=neighbor_radius,
                                       min_ge40=min_ge40)
    if preset == "combo_vintage":
        return select_combo_vintage(df_rank, topk=topk,
                                    min_ge40=max(min_ge40, 30), high_min=max(high_min, 32),
                                    low_cap=max(low_cap, 5), mid_cap=max(mid_cap, 9))
    raise ValueError(f"Preset desconhecido: {preset}")

def main():
    ap = argparse.ArgumentParser(description="Predict 0..99 com model.pkl (dict).")
    ap.add_argument("--predict", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--preset", required=True, choices=["balanced","high","high15","combo","combo_vintage"])
    ap.add_argument("--low-cap", type=int, default=6)
    ap.add_argument("--mid-cap", type=int, default=10)
    ap.add_argument("--high-min", type=int, default=28)
    ap.add_argument("--must10", type=int, default=1)
    ap.add_argument("--must20", type=int, default=2)
    ap.add_argument("--must30", type=int, default=2)
    ap.add_argument("--neighbor-penalty", type=float, default=0.50)
    ap.add_argument("--neighbor-radius", type=int, default=2)
    ap.add_argument("--min-ge40", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--use-cols", type=str, default="", help="Lista separada por ';' com nomes/trechos de coluna a usar (case/acento-insensitive)")
    ap.add_argument("--force-smart", action="store_true", help="Detecta automaticamente colunas numéricas 0..99 (>=30%)")

    args = ap.parse_args()
    with open(args.model, "rb") as f:
        model_dict = pickle.load(f)
    if not isinstance(model_dict, dict):
        raise ValueError("model.pkl deve ser dict com 'dec_prior' e 'col_reliability'.")

    use_cols_cli = [s for s in args.use_cols.split(";") if s.strip()] if args.use_cols else None

    df_rank = predict_scores_with_model(args.predict, model_dict, alpha=args.alpha, verbose=True,
                                        use_cols_cli=use_cols_cli, force_smart=args.force_smart)

    nums = run_preset(df_rank, args.preset, args.topk,
                      low_cap=args.low_cap, mid_cap=args.mid_cap, high_min=args.high_min,
                      must10=args.must10, must20=args.must20, must30=args.must30,
                      neighbor_penalty=args.neighbor_penalty, neighbor_radius=args.neighbor_radius,
                      min_ge40=args.min_ge40)

    base = Path(args.predict).stem
    print(f">>> {base} [{args.preset}|top{args.topk}]: {', '.join(map(str, nums))}")
    outdir = Path("out_predict"); outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{base}_top{args.topk}_{args.preset}.csv"
    pd.DataFrame({"valor": nums}).to_csv(outpath, index=False, encoding="utf-8")
    print(f"[ok] salvo em: {outpath}")

if __name__ == "__main__":
    main()
