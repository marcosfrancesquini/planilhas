#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ordenar_planilhas_colunas_lado.py

Gera uma ÚNICA planilha (uma aba) onde, para cada aba do arquivo de entrada:
- primeira linha: "Nome da Planilha | Tamanho do bloco: x | Tamanho do ultimo bloco: y | Porcentagem Faltante: z% | Número de linhas faltantes: k"
  (essas 5 informações são colocadas em cinco células lado a lado, na MESMA linha)
- abaixo, em uma grade horizontal: 7 blocos lado a lado; cada bloco tem 3 colunas:
    [<critério>, numero base, Número]
  Os blocos ficam lado a lado. As linhas são preenchidas até o tamanho do maior bloco, com células vazias quando faltar.
- entre as seções (planilhas), são inseridas 2 linhas em branco.

Critérios (na ordem e sentido de ordenação):
  1) Máxima Freq. nos Blocos Completos (desc)
  2) Mínima Freq. nos Blocos Completos (asc)
  3) Frequência no Último Bloco Incompleto (desc)
  4) Maximo de vezes (desc)
  5) Minimo de vezes (asc)
  6) máximo grande (desc)
  7) mínimo pequeno (asc)

Uso:
  python ordenar_planilhas_colunas_lado.py --input caminho/arquivo.xlsx --output caminho/saida.xlsx [--sheet "Planilha Única"] [--zero-como-vazio]
    --zero-como-vazio  -> opcional; se presente, mostra vazio (em vez de 0) nas colunas "máximo grande" e "mínimo pequeno".

Dependências:
  pip install pandas openpyxl odfpy xlsxwriter
"""

import argparse
import re
import unicodedata
from typing import List, Tuple, Any, Dict, Optional

import numpy as np
import pandas as pd


CRITERIOS: List[Tuple[str, bool]] = [
    ("Máxima Freq. nos Blocos Completos", False),         # 1 desc
    ("Mínima Freq. nos Blocos Completos", True),          # 2 asc
    ("Frequência no Último Bloco Incompleto", False),     # 3 desc
    ("Maximo de vezes", False),                            # 4 desc
    ("Minimo de vezes", True),                             # 5 asc
    ("máximo grande", False),                              # 6 desc
    ("mínimo pequeno", True),                              # 7 asc
]


def _first_non_null(df: pd.DataFrame, candidates: List[str]) -> Any:
    for c in candidates:
        if c in df.columns:
            s = df[c].dropna()
            if len(s) > 0:
                return s.iloc[0]
    return np.nan


def _fmt_percent(val: Any) -> str:
    if pd.isna(val):
        return ""
    try:
        s = str(val).strip().replace("%", "").replace(",", ".")
        return f"{float(s):.2f}%"
    except Exception:
        return str(val)


def _pick_numero_base(df: pd.DataFrame) -> pd.Series:
    if "numero base" in df.columns:
        return df["numero base"]
    for alt in ["Número", "numero", "numero_base", "número base"]:
        if alt in df.columns:
            s = df[alt].copy()
            s.name = "numero base"
            return s
    return pd.Series(df.index + 1, index=df.index, name="numero base")


def _pick_numero_col(df: pd.DataFrame) -> pd.Series:
    """Tenta extrair a coluna 'Número' (ou variantes). Se não houver, usa 'numero base' como fallback."""
    for alt in ["Número", "numero", "num", "Numero", "NÚMERO", "número"]:
        if alt in df.columns:
            s = df[alt].copy()
            s.name = "Número"
            return s
    # fallback
    nb = _pick_numero_base(df)
    s = nb.copy()
    s.name = "Número"
    return s


def _coerce_numeric_col(series: pd.Series) -> pd.Series:
    if series.dtype != object:
        return series
    try:
        coerced = pd.to_numeric(
            series.astype(str)
                  .str.replace("%", "", regex=False)
                  .str.replace(",", ".", regex=False)
                  .str.strip(),
            errors="coerce"
        )
        return coerced
    except Exception:
        return series


def _norm_text(s: str) -> str:
    """Remove acentos, pontuação e padroniza para comparação robusta."""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)  # remove pontuação
    s = re.sub(r"\s+", " ", s)
    return s


def _build_col_resolver(df: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for c in df.columns:
        mapping[_norm_text(c)] = c
    return mapping


def _resolve_col(df: pd.DataFrame, canonical: str, resolver: Dict[str, str], extras: Optional[List[str]] = None) -> Optional[str]:
    candidates = [canonical]
    if extras:
        candidates += extras
    for cand in candidates:
        key = _norm_text(cand)
        if key in resolver:
            return resolver[key]
    key = _norm_text(canonical).replace("freq", "frequencia")
    if key in resolver:
        return resolver[key]
    return None


SYNONYMS: Dict[str, List[str]] = {
    "Máxima Freq. nos Blocos Completos": ["Maxima Freq. nos Blocos Completos", "Maxima Freq nos Blocos Completos", "Máxima Freq nos Blocos Completos"],
    "Mínima Freq. nos Blocos Completos": ["Minima Freq. nos Blocos Completos", "Minima Freq nos Blocos Completos", "Mínima Freq nos Blocos Completos"],
    "Frequência no Último Bloco Incompleto": ["Frequencia no Ultimo Bloco Incompleto", "Frequencia no Último Bloco Incompleto"],
    "Maximo de vezes": ["Máximo de vezes", "Maximo de Vezes"],
    "Minimo de vezes": ["Mínimo de vezes", "Minimo de Vezes"],
    "máximo grande": ["Maximo grande", "maximo grande"],
    "mínimo pequeno": ["Minimo pequeno", "minimo pequeno"],
}


def _get_header_values(df: pd.DataFrame, sheet_name: str) -> List[str]:
    tam_bloco = _first_non_null(df, ["Tamanho do Bloco", "Tamanho do bloco", "Tamanho do bloco: 69"])
    tam_ultimo = _first_non_null(df, ["Tamanho do Último Bloco", "Tamanho do ultimo bloco", " Tamanho do ultimo bloco: 66"])
    porc_falt = _first_non_null(df, ["Porcentagem Faltante (%)", "Porcentagem Faltante", " Porcentagem Faltante: 4,35%"])
    porc_falt_fmt = _fmt_percent(porc_falt)
    num_linhas_falt = _first_non_null(df, ["Número de linhas faltantes", " Número de linhas faltantes: 3"])

    return [
        f"{sheet_name}",
        f"Tamanho do bloco: {tam_bloco}",
        f"Tamanho do ultimo bloco: {tam_ultimo}",
        f"Porcentagem Faltante: {porc_falt_fmt}",
        f"Número de linhas faltantes: {num_linhas_falt}",
    ]


def _build_blocks_side_by_side(df: pd.DataFrame, zero_as_blank: bool) -> Dict[str, pd.DataFrame]:
    nb = _pick_numero_base(df)
    numcol = _pick_numero_col(df)
    blocks: Dict[str, pd.DataFrame] = {}

    resolver = _build_col_resolver(df)

    for crit, asc in CRITERIOS:
        real_col = _resolve_col(df, crit, resolver, SYNONYMS.get(crit))
        if real_col is None:
            ord_df = pd.DataFrame({crit: pd.Series(dtype=object), "numero base": pd.Series(dtype=object), "Número": pd.Series(dtype=object)})
        else:
            orig = df[real_col].copy()

            if zero_as_blank and crit in ["máximo grande", "mínimo pequeno"]:
                if pd.api.types.is_numeric_dtype(orig):
                    orig = orig.mask(orig == 0, other=np.nan)
                else:
                    orig = orig.mask(orig.astype(str).str.strip().isin({"0", "0.0", "0,0"}), other=np.nan)

            key = _coerce_numeric_col(df[real_col])
            ord_df = pd.DataFrame({crit: orig, "numero base": nb, "Número": numcol, "__key__": key})
            ord_df = ord_df.sort_values(by="__key__", ascending=asc, kind="mergesort", na_position="last").drop(columns=["__key__"])

        blocks[crit] = ord_df.reset_index(drop=True)

    return blocks


def _xlsx_value(v):
    try:
        if v is None:
            return None
        if isinstance(v, (float, int)) and (pd.isna(v) or (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))):
            return None
        if pd.isna(v):
            return None
        return v
    except Exception:
        return v


def _write_section(worksheet, start_row: int, header_cells: List[str], blocks: Dict[str, pd.DataFrame]) -> int:
    row = start_row
    for j, val in enumerate(header_cells):
        worksheet.write(row, j, _xlsx_value(val))

    max_len = max((len(df) for df in blocks.values()), default=0)
    row_blocks_header = row + 1

    start_col = 0
    for idx, (crit, _) in enumerate(CRITERIOS):
        bdf = blocks[crit].copy()
        if len(bdf) < max_len:
            pad = pd.DataFrame({crit: [np.nan]*(max_len - len(bdf)),
                                "numero base": [np.nan]*(max_len - len(bdf)),
                                "Número": [np.nan]*(max_len - len(bdf))})
            bdf = pd.concat([bdf, pad], ignore_index=True)

        worksheet.write(row_blocks_header, start_col + 0, _xlsx_value(crit))
        worksheet.write(row_blocks_header, start_col + 1, _xlsx_value("numero base"))
        worksheet.write(row_blocks_header, start_col + 2, _xlsx_value("Número"))

        col_vals = bdf[crit].tolist()
        col_nb = bdf["numero base"].tolist()
        col_num = bdf["Número"].tolist()
        for r_off in range(max_len):
            worksheet.write(row_blocks_header + 1 + r_off, start_col + 0, _xlsx_value(col_vals[r_off] if r_off < len(col_vals) else None))
            worksheet.write(row_blocks_header + 1 + r_off, start_col + 1, _xlsx_value(col_nb[r_off] if r_off < len(col_nb) else None))
            worksheet.write(row_blocks_header + 1 + r_off, start_col + 2, _xlsx_value(col_num[r_off] if r_off < len(col_num) else None))

        start_col += 3

    total_consumed = 1 + 1 + max_len
    return total_consumed


def process_file(input_path: str, output_path: str, sheet_name: str = "Planilha Única", zero_as_blank: bool = False) -> None:
    xls = pd.ExcelFile(input_path)
    writer = pd.ExcelWriter(output_path, engine="xlsxwriter")
    pd.DataFrame([]).to_excel(writer, sheet_name=sheet_name, index=False)
    worksheet = writer.sheets[sheet_name]

    current_row = 0
    for sname in xls.sheet_names:
        df = pd.read_excel(input_path, sheet_name=sname)
        header_cells = _get_header_values(df, sname)
        blocks = _build_blocks_side_by_side(df, zero_as_blank=zero_as_blank)
        consumed = _write_section(worksheet, current_row, header_cells, blocks)
        current_row += consumed + 2

    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Gera planilha única com blocos lado a lado por critério (3 colunas por bloco).")
    parser.add_argument("--input", "-i", required=True, help="Caminho do arquivo XLSX/ODS de entrada")
    parser.add_argument("--output", "-o", required=True, help="Caminho do XLSX de saída")
    parser.add_argument("--sheet", "-s", default="Planilha Única", help="Nome da aba única de saída")
    parser.add_argument("--zero-como-vazio", action="store_true", help="Se presente, mostra vazio em vez de 0 em 'máximo grande' e 'mínimo pequeno'.")
    args = parser.parse_args()

    process_file(args.input, args.output, args.sheet, args.zero_como_vazio)
    print(f"OK: gerado '{args.output}'")

if __name__ == "__main__":
    main()
