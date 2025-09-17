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
  6) máximo grande (desc)   *zeros viram vazio para exibição*
  7) mínimo pequeno (asc)   *zeros viram vazio para exibição*

Uso:
  python ordenar_planilhas_colunas_lado.py --input caminho/arquivo.xlsx --output caminho/saida.xlsx

Dependências:
  pip install pandas openpyxl odfpy xlsxwriter
"""

import argparse
from typing import List, Tuple, Any, Dict

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


def _build_blocks_side_by_side(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Para cada critério, gera um DataFrame de três colunas [<critério>, numero base, Número]
    já ordenado. Ordena por uma CHAVE numérica derivada do critério, mas mantém o VALOR
    ORIGINAL na coluna do critério (para não virar "0").

    Tratamento especial: para as colunas "máximo grande" e "mínimo pequeno",
    valores iguais a 0/0.0/"0"/"0,0" são tratados como vazios (NaN) para exibição.
    """
    nb = _pick_numero_base(df)
    numcol = _pick_numero_col(df)
    blocks: Dict[str, pd.DataFrame] = {}

    ZEROY = {"0", "0.0", "0,0"}

    for crit, asc in CRITERIOS:
        if crit in df.columns:
            orig = df[crit].copy()

            if crit in ["máximo grande", "mínimo pequeno"]:
                if pd.api.types.is_numeric_dtype(orig):
                    orig = orig.mask(orig == 0, other=np.nan)
                else:
                    orig = orig.mask(orig.astype(str).str.strip().isin(ZEROY), other=np.nan)

            key = _coerce_numeric_col(df[crit])
            ord_df = pd.DataFrame({crit: orig, "numero base": nb, "Número": numcol, "__key__": key})
            ord_df = ord_df.sort_values(by="__key__", ascending=asc, kind="mergesort", na_position="last").drop(columns=["__key__"])
        else:
            ord_df = pd.DataFrame({crit: pd.Series(dtype=object), "numero base": pd.Series(dtype=object), "Número": pd.Series(dtype=object)})

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
    """
    Escreve no worksheet:
      - linha de header (5 células lado a lado)
      - em seguida, linhas com 7 blocos de colunas [<critério>, numero base, Número], lado a lado
    Retorna o número total de linhas consumidas (inclui a linha do header + cabeçalhos dos blocos + dados).
    """
    # 1) Cabeçalho-resumo (5 células)
    row = start_row
    for j, val in enumerate(header_cells):
        worksheet.write(row, j, _xlsx_value(val))

    # 2) Preparar blocos
    max_len = max((len(df) for df in blocks.values()), default=0)
    row_blocks_header = row + 1

    # 3) Escrever blocos lado a lado
    start_col = 0
    for idx, (crit, _) in enumerate(CRITERIOS):
        bdf = blocks[crit].copy()
        if len(bdf) < max_len:
            pad = pd.DataFrame({crit: [np.nan]*(max_len - len(bdf)),
                                "numero base": [np.nan]*(max_len - len(bdf)),
                                "Número": [np.nan]*(max_len - len(bdf))})
            bdf = pd.concat([bdf, pad], ignore_index=True)

        # cabeçalhos das 3 colunas do bloco
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


def process_file(input_path: str, output_path: str, sheet_name: str = "Planilha Única") -> None:
    xls = pd.ExcelFile(input_path)
    writer = pd.ExcelWriter(output_path, engine="xlsxwriter")
    # cria aba única
    pd.DataFrame([]).to_excel(writer, sheet_name=sheet_name, index=False)
    worksheet = writer.sheets[sheet_name]

    current_row = 0
    for sname in xls.sheet_names:
        df = pd.read_excel(input_path, sheet_name=sname)
        header_cells = _get_header_values(df, sname)
        blocks = _build_blocks_side_by_side(df)
        consumed = _write_section(worksheet, current_row, header_cells, blocks)
        current_row += consumed + 2  # 2 linhas em branco entre planilhas

    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Gera planilha única com blocos lado a lado por critério (3 colunas por bloco).")
    parser.add_argument("--input", "-i", required=True, help="Caminho do arquivo XLSX/ODS de entrada")
    parser.add_argument("--output", "-o", required=True, help="Caminho do XLSX de saída")
    parser.add_argument("--sheet", "-s", default="Planilha Única", help="Nome da aba única de saída")
    args = parser.parse_args()

    process_file(args.input, args.output, args.sheet)
    print(f"OK: gerado '{args.output}'")

if __name__ == "__main__":
    main()
