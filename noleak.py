#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_detector_noleak_spread.py
------------------------------
Detector NO-LEAK (predição sem olhar a coluna `numero base`) com
espalhamento reforçado nas faixas altas (60+, 75+, 90+).

Fluxo:
1) Treino (com planilhas QUE têm `numero base`):
   - Aprende priors de posição (densidade por decil).
   - Aprende confiabilidade de colunas (quais colunas tendem a estar
     preenchidas quando existe `numero base`).

2) Predição (em planilhas sem `numero base`):
   - Usa apenas os priors aprendidos + presença de colunas confiáveis
     + pesos por decil + bônus de espalhamento para 60/75/90+.
   - Extrai valores 0..99 das colunas confiáveis e gera TOP-K.

Exemplos:
  # Treinar o modelo
  python base_detector_noleak_spread.py --train saiu.xlsx --model ./out/model.pkl

  # Predizer (sem olhar numero base) e gerar Top-50
  python base_detector_noleak_spread.py --predict saiu_sem_numero_base.xlsx --model ./out/model.pkl --topk 50

Dependências:
  pip install pandas numpy openpyxl
"""

import argparse
import pickle
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# ===================== Utils de parsing dos blocos =====================

def normalize_name(s): 
    return str(s).strip().lower()

def find_block_starts(df: pd.DataFrame):
    """Detecta início de blocos pela presença de 'numero base' em qualquer coluna."""
    return df.index[df.astype(str).apply(
        lambda row: row.str_contains("numero base", case=False, na=False) 
        if hasattr(row, "str_contains") else row.str.contains("numero base", case=False, na=False)
    ).any(axis=1)].tolist()

def parse_meta_from_context(df: pd.DataFrame, start_idx: int, lookback: int = 4):
    """(Opcional) Parse de metadados nas linhas acima do cabeçalho. Não é obrigatório para predição."""
    s = []
    for r in range(max(0, start_idx - lookback), start_idx):
        vals = [str(x) for x in df.iloc[r].tolist() if pd.notna(x)]
        s.append(" ".join(vals))
    text = " ".join(s)
    # Exemplo de metadado: Porcentagem Faltante, se quiser usar depois
    meta = {}
    m = re.search(r'Porcentagem Faltante:\s*([\d\.,]+)%', text, re.IGNORECASE)
    if m:
        meta["pct_faltante"] = float(m.group(1).replace(",", "."))
    return meta

def build_block(df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    """Monta um DataFrame do bloco usando a linha de cabeçalho em start_idx."""
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
    """Extrai todos os blocos de um arquivo (cada bloco começa na linha que contém 'numero base')."""
    starts = find_block_starts(df)
    blocks = []
    for i, s in enumerate(starts):
        e = starts[i+1] if i+1 < len(starts) else len(df)
        meta = parse_meta_from_context(df, s, lookback=4)
        block_df = build_block(df, s, e)
        blocks.append({"start": s, "end": e, "meta": meta, "data": block_df})
    return blocks

def decil_of_row(ridx: int, n_rows: int) -> int:
    if n_rows <= 0: 
        return 1
    return int(np.ceil((ridx+1)/n_rows*10))


# ===================== Priors e pesos =====================

# Pesos “estruturais” por decil (1..10): 1º e 9º fortes; 2º e 10º fracos.
BASE_DECIL_WEIGHTS = np.array([1.00, -0.40, 0.10, 0.35, 0.30, 0.30, 0.10, 0.15, 0.60, -0.25], dtype=float)

# Espalhamento reforçado (como no re-run com 10/20):
SPREAD_60 = 0.20   # bônus para valores 60..74
SPREAD_75 = 0.40   # bônus para valores 75..89
SPREAD_90 = 0.60   # bônus para valores 90..99

def weights_by_pct_faltante(pct: float) -> np.ndarray:
    """Se desejar usar % faltante (quando disponível) para ajustar decil; opcional."""
    w = BASE_DECIL_WEIGHTS.copy()
    if pct is not None and not np.isnan(pct) and pct <= 10:
        # reforço clássico (opcional)
        w[0] += 0.10; w[8] += 0.05; w[1] -= 0.05; w[9] -= 0.05
    return w

def line_score_from_priors(dec_prior: np.ndarray, decil_weighted: np.ndarray, dec: int) -> float:
    """Combina prior de decil com peso normalizado do decil."""
    w_norm = (decil_weighted - decil_weighted.min()) / (decil_weighted.max() - decil_weighted.min() + 1e-9)
    return float(dec_prior[dec] * w_norm[dec])


# ===================== Treino (com `numero base`) =====================

def train_model(train_paths):
    """
    Aprende:
    - prior de decil (proporção de linhas com `numero base` por decil);
    - confiabilidade de colunas (probabilidade de a coluna estar preenchida nas linhas com `numero base`).
    """
    dec_counts = np.zeros(10, dtype=float)
    dec_total  = np.zeros(10, dtype=float)
    col_hits = defaultdict(float)
    col_tot  = defaultdict(int)

    for path in train_paths:
        df = pd.read_excel(path)
        blocks = extract_blocks(df)
        for b in blocks:
            dfb = b["data"]; n = len(dfb)
            if n == 0: 
                continue
            nb_cols = [c for c in dfb.columns if normalize_name(c) == "numero base"]
            if not nb_cols: 
                continue
            nb_mask = dfb[nb_cols[0]].notna().to_numpy()

            # prior de decil
            for ridx, is_nb in enumerate(nb_mask):
                dec = decil_of_row(ridx, n) - 1
                dec_total[dec] += 1
                if is_nb:
                    dec_counts[dec] += 1

            # confiabilidade de colunas (presença quando NB=1)
            idx_nb = np.where(nb_mask)[0]
            if len(idx_nb) == 0: 
                continue
            for col in dfb.columns:
                if col == nb_cols[0]: 
                    continue
                present = dfb[col].notna().to_numpy()[idx_nb].mean()
                col_hits[normalize_name(col)] += float(present)
                col_tot[normalize_name(col)]  += 1

    # prior de decil normalizado (0..1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dec_prior = np.where(dec_total > 0, dec_counts / dec_total, 0.0)
    if dec_prior.sum() > 0:
        dec_prior = dec_prior / (np.abs(dec_prior).max() or 1.0)
    else:
        # fallback: usa forma dos pesos estruturais
        m = BASE_DECIL_WEIGHTS - BASE_DECIL_WEIGHTS.min()
        dec_prior = m / (m.max() or 1.0)

    # confiabilidade por coluna
    col_reliability = {c: (col_hits[c] / max(1, col_tot[c])) for c in col_tot.keys()}

    return {"dec_prior": dec_prior, "col_reliability": col_reliability}


# ===================== Predição (sem `numero base`) =====================

def predict_file(xlsx_path, model, topk=50, alpha=0.6, use_pct_faltante=False):
    """
    Gera Top-K de candidatos 0..99 SEM olhar `numero base`.
    - dec_prior: prior de decil aprendido no treino
    - col_reliability: confiabilidade das colunas (aprendida)
    - alpha: peso do reforço de presença das colunas confiáveis na linha
    - use_pct_faltante: se True, ajusta o peso de decil com base em % faltante (quando disponível)
    """
    dec_prior = model["dec_prior"]
    col_rel   = model["col_reliability"]

    df = pd.read_excel(xlsx_path)
    blocks = extract_blocks(df)

    # escolher top colunas confiáveis (reduz ruído)
    top_cols = [c for c,_ in sorted(col_rel.items(), key=lambda x: x[1], reverse=True)[:6]]

    value_scores = defaultdict(float)

    for b in blocks:
        dfb = b["data"]; n = len(dfb)
        if n == 0:
            continue

        if use_pct_faltante:
            pct = b["meta"].get("pct_faltante")
            dec_w = weights_by_pct_faltante(pct)
        else:
            dec_w = BASE_DECIL_WEIGHTS.copy()

        for ridx in range(n):
            dec = decil_of_row(ridx, n) - 1

            # score base da linha (posição no bloco)
            line_score = line_score_from_priors(dec_prior, dec_w, dec)

            # reforço de presença de colunas confiáveis nessa linha
            present_boost = 0.0
            for col in dfb.columns:
                cn = normalize_name(col)
                if cn in top_cols and pd.notna(dfb.iloc[ridx][col]):
                    present_boost += col_rel[cn]

            row_score = line_score * (1.0 + alpha * present_boost)

            # extrair valores 0..99 das colunas confiáveis e somar o score
            for col in dfb.columns:
                cn = normalize_name(col)
                if cn not in top_cols:
                    continue
                val = dfb.iloc[ridx][col]
                if pd.isna(val):
                    continue
                try:
                    v = int(float(str(val).strip()))
                except Exception:
                    continue
                if not (0 <= v <= 99):
                    continue

                # bônus de espalhamento reforçado (como no re-run 10/20)
                spread = 0.0
                if v >= 90:
                    spread = SPREAD_90
                elif v >= 75:
                    spread = SPREAD_75
                elif v >= 60:
                    spread = SPREAD_60

                value_scores[v] += (row_score + spread)

    items = [{"valor": v, "score": s} for v, s in value_scores.items()]
    df_rank = pd.DataFrame(items).sort_values(["score", "valor"], ascending=[False, True]).reset_index(drop=True)
    top_vals = df_rank.head(topk)["valor"].tolist()
    return df_rank, top_vals


# ===================== CLI =====================

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    ap = argparse.ArgumentParser(description="Detector NO-LEAK com espalhamento reforçado (60/75/90+).")
    ap.add_argument("--train", nargs="*", help="Arquivos .xlsx COM `numero base` (treino).")
    ap.add_argument("--predict", nargs="*", help="Arquivos .xlsx SEM `numero base` (predição).")
    ap.add_argument("--model", type=str, required=True, help="Caminho do modelo .pkl para salvar/carregar.")
    ap.add_argument("--topk", type=int, default=50, help="Top-K de candidatos (default: 50).")
    ap.add_argument("--alpha", type=float, default=0.6, help="Peso do reforço por presença de colunas confiáveis (default: 0.6).")
    ap.add_argument("--use-pct-faltante", action="store_true", help="Usar % faltante (se houver) para ajustar peso por decil.")
    ap.add_argument("--outdir", type=str, default="out_noleak_spread", help="Saída (default: ./out_noleak_spread)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.train:
        model = train_model(args.train)
        save_model(model, args.model)
        print(f"[ok] Modelo treinado e salvo em: {args.model}")
        print("Top colunas por confiabilidade:")
        for k, v in sorted(model["col_reliability"].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {k:30s} -> {v:.3f}")

    if args.predict:
        model = load_model(args.model)
        for inp in args.predict:
            df_rank, top_vals = predict_file(
                inp, model, topk=args.topk, alpha=args.alpha, use_pct_faltante=args.use_pct_faltante
            )
            base = Path(inp).stem
            df_rank.to_csv(outdir / f"{base}_ranking_noleak_spread.csv", index=False)
            pd.DataFrame({"valor": top_vals}).to_csv(outdir / f"{base}_top{args.topk}_noleak_spread.csv", index=False)
            print(f">>> {base}: {', '.join(map(str, top_vals))}")

if __name__ == "__main__":
    main()
