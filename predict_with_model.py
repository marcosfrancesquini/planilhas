#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_with_model.py
---------------------
Predição NO-LEAK usando um modelo pré-treinado (model.pkl).

Uso:
  python predict_with_model.py --predict planilha.xlsx --model model.pkl --topk 50

Também aceita múltiplas planilhas:
  python predict_with_model.py --predict a.xlsx b.xlsx c.xlsx --model model.pkl --topk 50

Requisitos:
  pip install pandas numpy openpyxl
"""

import argparse
import pickle
from pathlib import Path
from collections import defaultdict
import sys

import numpy as np
import pandas as pd


# ===================== Helpers de parsing =====================

def normalize_name(s):
    return str(s).strip().lower()

def find_block_starts(df: pd.DataFrame):
    # Linha de cabeçalho é aquela que contém "numero base" (case-insensitive) em alguma coluna
    return df.index[df.astype(str).apply(
        lambda row: row.str.contains("numero base", case=False, na=False)
    ).any(axis=1)].tolist()

def build_block(df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    # Cabeçalho na linha start_idx
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

def extract_blocks(df: pd.DataFrame):
    starts = find_block_starts(df)
    blocks = []
    for i, s in enumerate(starts):
        e = starts[i+1] if i+1 < len(starts) else len(df)
        block_df = build_block(df, s, e)
        blocks.append(block_df)
    return blocks

def decil_of_row(ridx: int, n_rows: int) -> int:
    if n_rows <= 0:
        return 1
    return int(np.ceil((ridx+1)/n_rows*10))


# ===================== Priors e parâmetros =====================

BASE_DECIL_WEIGHTS = np.array([1.00, -0.40, 0.10, 0.35, 0.30,
                               0.30, 0.10, 0.15, 0.60, -0.25], dtype=float)

SPREAD_60 = 0.20  # 60–74
SPREAD_75 = 0.40  # 75–89
SPREAD_90 = 0.60  # 90–99


# ===================== Predição com modelo =====================

def predict_scores_with_model(xlsx_path: str, model: dict, alpha: float = 0.6) -> pd.DataFrame:
    """
    Gera um ranking (valor 0..99, score) usando:
      - priors de decil aprendidos (model["dec_prior"])
      - confiabilidade de colunas (model["col_reliability"]) para selecionar top colunas
      - espalhamento reforçado 60/75/90+
    Sem olhar 'numero base' na fase de predição.
    """
    try:
        df = pd.read_excel(xlsx_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo não encontrado: {xlsx_path}")
    except Exception as e:
        raise RuntimeError(f"Falha ao abrir {xlsx_path}: {e}")

    blocks = extract_blocks(df)
    if not blocks:
        # fallback: tenta usar o arquivo inteiro como um "bloco" sem cabeçalho
        # (isso evita lista vazia caso a planilha tenha sido exportada sem aquela linha de 'numero base')
        blocks = [df.dropna(how="all", axis=1).reset_index(drop=True)]

    dec_prior = np.array(model.get("dec_prior", np.zeros(10, dtype=float)))
    col_rel   = model.get("col_reliability", {})
    if dec_prior.shape[0] != 10:
        # fallback seguro
        m = BASE_DECIL_WEIGHTS - BASE_DECIL_WEIGHTS.min()
        dec_prior = m / (m.max() or 1.0)

    # Seleciona as top colunas mais confiáveis
    top_cols = [c for c,_ in sorted(col_rel.items(), key=lambda x: x[1], reverse=True)[:6]]

    value_scores = defaultdict(float)
    any_cell = False  # para detectar se achamos números 0..99

    for b in blocks:
        n = len(b)
        if n == 0:
            continue

        w = BASE_DECIL_WEIGHTS.copy()
        w_norm = (w - w.min()) / (w.max() - w.min() + 1e-9)

        # Se as top_cols não existirem no bloco, vamos tentar usar TODAS as colunas como fallback ao extrair
        cols_for_presence = [c for c in b.columns if normalize_name(c) in top_cols]
        use_all_cols_for_values = False
        if not cols_for_presence:
            # se nenhuma top-coluna aparecer, ainda calculamos presence como 0 e,
            # na hora de extrair valores, liberamos para todas as colunas
            use_all_cols_for_values = True

        for ridx in range(n):
            dec = decil_of_row(ridx, n) - 1
            line_score = float(dec_prior[dec] * w_norm[dec])

            # reforço de presença das colunas confiáveis
            present_boost = 0.0
            if cols_for_presence:
                for col in cols_for_presence:
                    if pd.notna(b.iloc[ridx][col]):
                        present_boost += col_rel.get(normalize_name(col), 0.0)

            row_score = line_score * (1.0 + alpha * present_boost)

            # extrair valores 0..99
            cols_for_values = b.columns if use_all_cols_for_values else cols_for_presence
            if len(cols_for_values) == 0:
                cols_for_values = b.columns  # fallback final

            for col in cols_for_values:
                val = b.iloc[ridx][col]
                if pd.isna(val):
                    continue
                try:
                    v = int(float(str(val).strip()))
                except Exception:
                    continue
                if 0 <= v <= 99:
                    any_cell = True
                    spread = 0.0
                    if v >= 90:
                        spread = SPREAD_90
                    elif v >= 75:
                        spread = SPREAD_75
                    elif v >= 60:
                        spread = SPREAD_60
                    value_scores[v] += (row_score + spread)

    items = [{"valor": v, "score": s} for v, s in value_scores.items()]
    if not items and not any_cell:
        raise RuntimeError(
            "Nenhum valor 0..99 foi localizado nas células da planilha. "
            "Verifique se o arquivo está no formato esperado (números inteiros de 0 a 99 nas colunas)."
        )

    df_rank = pd.DataFrame(items)
    if df_rank.empty or "score" not in df_rank.columns:
        # Protege contra KeyError: 'score'
        raise RuntimeError(
            "Não foi possível construir o ranking (lista vazia). "
            "Cheque se as colunas numéricas 0..99 existem na planilha."
        )

    df_rank = df_rank.sort_values(["score", "valor"], ascending=[False, True]).reset_index(drop=True)
    return df_rank


def rerank_with_diversity(df_rank: pd.DataFrame, topk: int = 50,
                          low_cap: int = 10, mid_cap: int = 14, high_min: int = 18,
                          neighbor_penalty: float = 0.35, neighbor_radius: int = 2) -> list:
    """
    Seleção greedy com:
      - cotas: máx 0–29 (low_cap), máx 30–59 (mid_cap), mínimo 60–99 (high_min)
      - penalidade para vizinhos próximos (|Δ| <= neighbor_radius)
      - determinístico (sem random)
    """
    selected = []
    low_cnt = mid_cnt = high_cnt = 0
    remaining = df_rank.copy()

    def bucket(v):
        if v <= 29: return "low"
        elif v <= 59: return "mid"
        else: return "high"

    while len(selected) < topk and not remaining.empty:
        rem = remaining.copy()
        # penalização de vizinhos
        if selected:
            sel_set = set(selected)
            penal = []
            for v in rem["valor"]:
                penalty = 0.0
                for s in sel_set:
                    if abs(int(v) - int(s)) <= neighbor_radius:
                        penalty += neighbor_penalty
                penal.append(penalty)
            rem["score_adj"] = rem["score"] - np.array(penal)
        else:
            rem["score_adj"] = rem["score"]

        # aplica cotas
        def quota_ok(v):
            b = bucket(int(v))
            if b == "low" and low_cnt >= low_cap: return False
            if b == "mid" and mid_cnt >= mid_cap: return False
            return True

        rem = rem[rem["valor"].apply(quota_ok)]
        if rem.empty:
            break

        # escolhe melhor ajustado
        choice = rem.sort_values(["score_adj", "valor"], ascending=[False, True]).iloc[0]
        v = int(choice["valor"])
        selected.append(v)
        if v <= 29: low_cnt += 1
        elif v <= 59: mid_cnt += 1
        else: high_cnt += 1

        remaining = remaining[remaining["valor"] != v]

    return selected


# ===================== CLI =====================

def main():
    ap = argparse.ArgumentParser(description="Predição com modelo NO-LEAK já treinado (model.pkl).")
    ap.add_argument("--predict", nargs="+", required=True, help="Arquivos .xlsx (um ou mais) para predição.")
    ap.add_argument("--model", required=True, help="Caminho para model.pkl (treinado previamente).")
    ap.add_argument("--topk", type=int, default=50, help="Tamanho da lista final (default 50).")
    ap.add_argument("--outdir", default="out_predict", help="Pasta de saída (default: out_predict)")
    args = ap.parse_args()

    # carregar modelo
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[erro] model.pkl não encontrado em: {model_path}", file=sys.stderr)
        sys.exit(2)

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"[erro] Falha ao carregar model.pkl: {e}", file=sys.stderr)
        sys.exit(2)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for inp in args.predict:
        try:
            df_rank = predict_scores_with_model(inp, model)
            top_vals = rerank_with_diversity(df_rank, topk=args.topk)
        except Exception as e:
            print(f"[erro] {inp}: {e}", file=sys.stderr)
            continue

        base = Path(inp).stem
        out_csv = outdir / f"{base}_top{args.topk}.csv"
        pd.DataFrame({"valor": top_vals}).to_csv(out_csv, index=False)
        print(f">>> {base}: {', '.join(map(str, top_vals))}")
        print(f"[ok] salvo em: {out_csv}")

if __name__ == "__main__":
    main()
