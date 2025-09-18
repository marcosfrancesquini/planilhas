#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_detector.py
----------------
Detector de números "promissores" (0..99) a partir de planilhas com blocos que
contêm uma coluna "numero base". NÃO olhamos os valores do "numero base";
usamos a COOCORRÊNCIA de linhas (posições) onde há "numero base" com as demais colunas.
Este pipeline replica o método que levou a 16/20 acertos:

- Extração automática de blocos (cada bloco começa onde aparece "numero base").
- Parsing de metadados de bloco a partir do texto acima do cabeçalho
  (ex.: "Porcentagem Faltante: X%").
- Pesos por posição (decis 1..10): 1º e 9º fortes; 2º e 10º fracos.
- Reforço quando % faltante <= 10%.
- Bônus de espalhamento para valores 60+, 75+ e 90+ (para cobrir 0..99).
- Filtros de força: manter só valores que aparecem em >= N blocos, >= M colunas e
  com frequência total >= F nas linhas com "numero base".
- Geração do ranking completo e de um TOP-K por arquivo.

Uso (CLI):
    python base_detector.py saiu1.xlsx saiu2.xlsx saiu3.xlsx --topk 50 --min-blocos 5 --min-colunas 2 --min-freq 4 --outdir ./out

Requer:
    pip install pandas numpy openpyxl

Autor: você & ChatGPT
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# ------------- Configurações padrão -------------

# Pesos por decil (1..10). Ajuste fino pode ser feito via argumentos no futuro.
BASE_WEIGHTS = np.array([
    1.00,   # decil 1  (forte)
    -0.40,  # decil 2  (vale)
    0.10,   # decil 3
    0.35,   # decil 4
    0.30,   # decil 5
    0.30,   # decil 6
    0.10,   # decil 7
    0.15,   # decil 8
    0.60,   # decil 9  (forte)
    -0.25,  # decil 10 (vale)
], dtype=float)

# Bônus por espalhamento (para cobrir 0..99 e evitar travar em 0..58)
SPREAD_BONUS_60 = 0.08
SPREAD_BONUS_75 = 0.15
SPREAD_BONUS_90 = 0.25

# Reforço quando % faltante <= 10% (faixas mais padronizadas)
REINFORCE_PCT_FALTANTE = {
    "add_dec1": 0.10,    # reforça decil 1
    "add_dec9": 0.05,    # reforça decil 9
    "sub_dec2": 0.05,    # penaliza ainda mais decil 2
    "sub_dec10": 0.05,   # penaliza ainda mais decil 10
}


# ------------- Utilitários -------------

def normalize_name(s) -> str:
    return str(s).strip().lower()


def find_block_starts(df: pd.DataFrame):
    """Retorna índices de linhas que contêm um dos nomes de BASE_COL_CANDIDATES."""
    pats = sorted(BASE_COL_CANDIDATES)
    def row_has_base(row):
        for cell in row:
            cs = normalize_name(cell)
            if any((c in cs) or (cs == c) for c in pats):
                return True
        return False
    mask = df.apply(row_has_base, axis=1)
    return df.index[mask].tolist()

def parse_meta_from_context(df: pd.DataFrame, start_idx: int, lookback: int = 4) -> dict:
    """Parseia metadados textuais nas linhas imediatamente acima do cabeçalho do bloco."""
    s = []
    for r in range(max(0, start_idx - lookback), start_idx):
        vals = [str(x) for x in df.iloc[r].tolist() if pd.notna(x)]
        s.append(" ".join(vals))
    text = " ".join(s)
    meta = {}
    # Exemplos de metadados; adicione regex extras conforme necessário
    m = re.search(r'Porcentagem Faltante:\s*([\d\.,]+)%', text, re.IGNORECASE)
    if m:
        meta["pct_faltante"] = float(m.group(1).replace(",", "."))
    m = re.search(r'K\s*=\s*(\d+)\s*\(([\d\.,]+)%\)', text)
    if m:
        meta["K"] = int(m.group(1))
        meta["K_pct"] = float(m.group(2).replace(",", "."))
    m = re.search(r'Tamanho do bloco:\s*(\d+)', text, re.IGNORECASE)
    if m:
        meta["tamanho_bloco"] = int(m.group(1))
    m = re.search(r'Tamanho do ultimo bloco:\s*(\d+)', text, re.IGNORECASE)
    if m:
        meta["tamanho_ultimo_bloco"] = int(m.group(1))
    m = re.search(r'Número de linhas faltantes:\s*(\d+)', text, re.IGNORECASE)
    if m:
        meta["n_linhas_faltantes"] = int(m.group(1))
    meta["raw_meta"] = text
    return meta

def build_block(df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    """Usa a linha 'start_idx' como cabeçalho e retorna o bloco de dados limpo entre (start_idx, end_idx)."""
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

def extract_blocks(df: pd.DataFrame, lookback: int = 4):
    """Extrai todos os blocos, com metadados + DataFrame do bloco."""
    starts = find_block_starts(df)
    blocks = []
    for i, s in enumerate(starts):
        e = starts[i+1] if i+1 < len(starts) else len(df)
        meta = parse_meta_from_context(df, s, lookback=lookback)
        block_df = build_block(df, s, e)
        blocks.append({"start": s, "end": e, "meta": meta, "data": block_df})
    return blocks

def weights_by_pct_faltante(pct: float) -> np.ndarray:
    """Ajusta pesos por decil dependendo da % faltante (<=10% reforça 1 e 9, penaliza 2 e 10)."""
    w = BASE_WEIGHTS.copy()
    if pct is not None and not np.isnan(pct) and pct <= 10:
        w[0] += REINFORCE_PCT_FALTANTE["add_dec1"]
        w[8] += REINFORCE_PCT_FALTANTE["add_dec9"]
        w[1] -= REINFORCE_PCT_FALTANTE["sub_dec2"]
        w[9] -= REINFORCE_PCT_FALTANTE["sub_dec10"]
    return w

def decil_of_row(row_idx: int, n_rows: int) -> int:
    """Converte índice de linha (0-based) em decil 1..10 do bloco."""
    if n_rows <= 0:
        return 1
    dec = int(np.ceil((row_idx + 1) / n_rows * 10))
    return max(1, min(10, dec))

def score_candidates_for_file(path: Path,
                              min_blocks: int = 5,
                              min_cols: int = 2,
                              min_freq: int = 4,
                              topk: int = 50):
    """Gera ranking e topK de valores 0..99 para um arquivo .xlsx."""
    df = pd.read_excel(path)
    blocks = extract_blocks(df)

    score = defaultdict(float)
    freq = defaultdict(int)
    blocks_set = defaultdict(set)
    cols_set = defaultdict(set)

    for bi, b in enumerate(blocks):
        dfb = b["data"]
        meta = b["meta"]
        n = len(dfb)
        nb_cols = [c for c in dfb.columns if normalize_name(c) in BASE_COL_CANDIDATES]
        if not nb_cols:
            nb_mask = dfb.notna().any(axis=1) if FALLBACK_MODE == "uniform" else None
            if nb_mask is None:
                continue
        nb_mask = dfb[nb_cols[0]].notna()
        if nb_mask.sum() == 0:
            nb_mask = dfb.notna().any(axis=1) if FALLBACK_MODE == "uniform" else None
            if nb_mask is None:
                continue

        w = weights_by_pct_faltante(meta.get("pct_faltante"))

        for col in dfb.columns:
            if col == nb_cols[0]:
                continue
            ser = dfb[col]
            for ridx in np.where(nb_mask)[0]:
                val = ser.iloc[ridx]
                if pd.isna(val):
                    continue
                # converter para inteiro, ignorar o resto
                try:
                    v = int(float(str(val).strip()))
                except Exception:
                    continue
                if v < 0 or v > 99:
                    continue

                dec = decil_of_row(ridx, n)
                base = w[dec - 1]

                # bônus de espalhamento
                if v >= 90:
                    spread = SPREAD_BONUS_90
                elif v >= 75:
                    spread = SPREAD_BONUS_75
                elif v >= 60:
                    spread = SPREAD_BONUS_60
                else:
                    spread = 0.0

                score[v] += base + spread
                freq[v] += 1
                blocks_set[v].add(bi)
                cols_set[v].add(col)

    # montar dataframe de ranking bruto
    rows = []
    for v in score.keys():
        rows.append({
            "valor": v,
            "score": float(score[v]),
            "freq": int(freq[v]),
            "n_blocos": int(len(blocks_set[v])),
            "n_colunas": int(len(cols_set[v])),
        })
    df_sc = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    # aplicar filtros de força
    strong = df_sc.query("n_blocos >= @min_blocks and n_colunas >= @min_cols and freq >= @min_freq").copy()
    strong = strong.sort_values(["score", "n_blocos", "n_colunas", "freq", "valor"],
                                ascending=[False, False, False, False, True]).reset_index(drop=True)

    # completar até TOPK se necessário (priorizando >=75)
    top_vals = strong["valor"].tolist()
    if len(top_vals) < topk:
        rem = df_sc[~df_sc["valor"].isin(top_vals)].copy()
        rem["hi_bonus"] = np.where(rem["valor"] >= 75, 1, 0)
        rem = rem.sort_values(["hi_bonus", "score", "n_blocos", "n_colunas", "freq", "valor"],
                              ascending=[False, False, False, False, False, True])
        need = topk - len(top_vals)
        top_vals.extend(rem.head(need)["valor"].tolist())

    top_vals = top_vals[:topk]
    return df_sc, strong, top_vals


def main():
    ap = argparse.ArgumentParser(description="Detector de números promissores (0..99) por coocorrência de posição com 'numero base'.")
    ap.add_argument("inputs", nargs="+", help="Arquivos .xlsx de entrada")
    ap.add_argument("--topk", type=int, default=50, help="Tamanho do TOP-K por arquivo (default: 50)")
    ap.add_argument("--min-blocos", type=int, default=5, help="Mínimo de blocos distintos para um valor ser 'forte' (default: 5)")
    ap.add_argument("--min-colunas", type=int, default=2, help="Mínimo de colunas distintas (default: 2)")
    ap.add_argument("--min-freq", type=int, default=4, help="Frequência mínima total (default: 4)")
    ap.add_argument("--outdir", type=str, default="out", help="Diretório de saída para CSVs (default: ./out)")
    ap.add_argument("--base-col", type=str, default="numero base",
                help="Lista de nomes candidatos para a coluna base (separados por vírgula)")
    ap.add_argument("--fallback-mode", choices=["uniform","none"], default="uniform",
                help="Quando não houver coluna base: 'uniform' usa todas as linhas não vazias; 'none' ignora o bloco")
    ap.add_argument("--weights", type=str, default=None,
                help="10 pesos por decil separados por vírgula para substituir os padrões")
    ap.add_argument("--bonus-60", type=float, default=None, help="Bônus de espalhamento para valores >=60")
    ap.add_argument("--bonus-75", type=float, default=None, help="Bônus de espalhamento para valores >=75")
    ap.add_argument("--bonus-90", type=float, default=None, help="Bônus de espalhamento para valores >=90")

    
    args = ap.parse_args()

    # Aplicar overrides de CLI
    global BASE_COL_CANDIDATES, FALLBACK_MODE, BASE_WEIGHTS, SPREAD_BONUS_60, SPREAD_BONUS_75, SPREAD_BONUS_90
    BASE_COL_CANDIDATES = {normalize_name(s) for s in re.split(r"[,|;]+", args.base_col) if s.strip()}
    FALLBACK_MODE = args.fallback_mode
    if args.weights:
        ws = [float(x) for x in re.split(r"[\s,]+", args.weights.strip()) if x]
        if len(ws) != 10:
            raise SystemExit("Erro: --weights exige exatamente 10 números.")
        BASE_WEIGHTS = np.array(ws, dtype=float)
    if args.bonus_60 is not None:
        SPREAD_BONUS_60 = float(args.bonus_60)
    if args.bonus_75 is not None:
        SPREAD_BONUS_75 = float(args.bonus_75)
    if args.bonus_90 is not None:
        SPREAD_BONUS_90 = float(args.bonus_90)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for inp in args.inputs:
        path = Path(inp)
        if not path.exists():
            print(f"[WARN] Arquivo não encontrado: {path}")
            continue

        print(f"\n>>> Processando: {path.name}")
        df_sc, strong, top_vals = score_candidates_for_file(
            path,
            min_blocks=args.min_blocos,
            min_cols=args.min_colunas,
            min_freq=args.min_freq,
            topk=args.topk,
        )

        # salvar CSVs
        base = path.stem
        df_sc.to_csv(outdir / f"{base}_ranking_completo.csv", index=False)
        strong.to_csv(outdir / f"{base}_ranking_forte.csv", index=False)
        pd.DataFrame({"valor": top_vals}).to_csv(outdir / f"{base}_top{args.topk}.csv", index=False)

        print(f"Top{args.topk} para {base}:")
        print(", ".join(str(v) for v in top_vals))

    print("\nConcluído.")


if __name__ == "__main__":
    main()
