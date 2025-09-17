#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisador de sorteios da Lotomania em blocos, gerando planilhas Excel.
v3: Colunas K e L mostram o NÚMERO (coluna E) quando passam no limiar:
    - K: =IF(Irow >= max_x, Erow, "")
    - L: =IF(Jrow <= min_x, Erow, "")
"""

import sys
import os
import pandas as pd
from typing import List, Tuple, Dict

def ler_sorteios_txt(caminho: str) -> List[List[int]]:
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    sorteios: List[List[int]] = []
    with open(caminho, 'r', encoding='utf-8') as f:
        for idx, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            raw = raw.replace('\t', ' ').replace(';', ' ')
            partes = [p for p in raw.split(' ') if p != '']
            linha: List[int] = []
            for p in partes:
                try:
                    n = int(p)
                except ValueError:
                    print(f"[AVISO] Linha {idx}: token inválido '{p}' – ignorado.", file=sys.stderr)
                    continue
                if 0 <= n <= 99:
                    linha.append(n)
            if linha:
                sorteios.append(linha)
    if not sorteios:
        raise ValueError("Nenhuma linha válida encontrada no arquivo.")
    return sorteios

def analise_por_bloco(linhas: List[List[int]], K: int):
    N = len(linhas)
    C = N // K
    r = N - C*K
    dados: Dict[int, tuple[int, int, int]] = {}
    if C == 0:
        for n in range(100):
            freq_incomp = 0
            for i in range(0, r):
                freq_incomp += sum(1 for x in linhas[i] if x == n)
            dados[n] = (0, 0, freq_incomp)
        return C, r, dados

    blocos_freq: List[List[int]] = []
    for j in range(C):
        i0 = j*K
        i1 = (j+1)*K
        hist = [0]*100
        for i in range(i0, i1):
            for x in linhas[i]:
                if 0 <= x <= 99:
                    hist[x] += 1
        blocos_freq.append(hist)

    hist_incomp = [0]*100
    if r > 0:
        i0 = C*K
        i1 = N
        for i in range(i0, i1):
            for x in linhas[i]:
                if 0 <= x <= 99:
                    hist_incomp[x] += 1

    for n in range(100):
        col = [blocos_freq[j][n] for j in range(C)]
        max_c = max(col) if col else 0
        min_c = min(col) if col else 0
        dados[n] = (max_c, min_c, hist_incomp[n] if r > 0 else 0)

    return C, r, dados

def gerar_planilhas(
    caminho_txt: str,
    faixa_percent: Tuple[float, float] = (3.0, 5.0),
    max_x: int = 10,
    min_x: int = 0,
    numeros_base: List[int] | None = None,
    saida_xlsx: str | None = None
) -> str:
    if numeros_base is None:
        numeros_base = []
    linhas = ler_sorteios_txt(caminho_txt)
    N = len(linhas)

    x, y = faixa_percent
    if x < 0 or y < 0 or x > 100 or y > 100 or x > y:
        raise ValueError("Faixa de porcentagem inválida. Use, por exemplo, 3 a 5.")

    if saida_xlsx is None:
        base = os.path.splitext(os.path.basename(caminho_txt))[0]
        saida_xlsx = f"analise_lotomania_{base}.xlsx"

    with pd.ExcelWriter(saida_xlsx, engine="xlsxwriter") as writer:
        for K in range(2, N+1):
            C = N // K
            r = N - C*K
            if r == 0:
                continue
            faltantes = K - r
            pct_faltante = (faltantes / K) * 100.0
            if pct_faltante + 1e-9 < x or pct_faltante - 1e-9 > y:
                continue

            C, r, dados = analise_por_bloco(linhas, K)

            registros = []
            for num in range(100):
                max_c, min_c, freq_r = dados[num]
                max_vezes = max_c - freq_r
                min_vezes = freq_r - min_c
                registro = {
                    "Tamanho do Bloco": K,                           # A
                    "Tamanho do Último Bloco": r,                    # B
                    "Porcentagem Faltante (%)": round(pct_faltante, 2), # C
                    "Número de linhas faltantes": faltantes,         # D
                    "Número": f"{num:02d}",                           # E
                    "Máxima Freq. nos Blocos Completos": max_c,      # F
                    "Mínima Freq. nos Blocos Completos": min_c,      # G
                    "Frequência no Último Bloco Incompleto": freq_r, # H
                    "Maximo de vezes": max_vezes,                    # I
                    "Minimo de vezes": min_vezes,                    # J
                    "máximo grande": "",                             # K (fórmula será escrita)
                    "mínimo pequeno": "",                            # L (fórmula será escrita)
                    "numero base": (num if num in numeros_base else "") # M
                }
                registros.append(registro)

            df = pd.DataFrame(registros, columns=[
                "Tamanho do Bloco","Tamanho do Último Bloco","Porcentagem Faltante (%)",
                "Número de linhas faltantes","Número",
                "Máxima Freq. nos Blocos Completos","Mínima Freq. nos Blocos Completos",
                "Frequência no Último Bloco Incompleto",
                "Maximo de vezes","Minimo de vezes",
                "máximo grande","mínimo pequeno","numero base"
            ])

            sheet_name = f"K={K} ({pct_faltante:.2f}%)"
            if len(sheet_name) > 31:
                sheet_name = f"K={K}"
            df.to_excel(writer, index=False, sheet_name=sheet_name)

            # Escrever fórmulas em K e L (linhas de dados começam na linha 2 no Excel)
            ws = writer.sheets[sheet_name]
            first_row = 2
            last_row = len(df) + 1
            for row in range(first_row, last_row + 1):
                # Usa IF (inglês). Excel PT-BR exibirá como SE ao abrir.
                formula_k = f'=IF(I{row}>={max_x},E{row},"")'
                formula_l = f'=IF(J{row}<={min_x},E{row},"")'
                ws.write_formula(row - 1, 10, formula_k)  # K = col 10 (0-based)
                ws.write_formula(row - 1, 11, formula_l)  # L = col 11

        # Ajuste de largura
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.set_column(0, 1, 22)
            ws.set_column(2, 3, 26)
            ws.set_column(4, 4, 10)
            ws.set_column(5, 8, 30)
            ws.set_column(9, 11, 18)
            ws.set_column(12, 12, 14)

    return os.path.abspath(saida_xlsx)

def parse_lista_numeros(txt: str) -> List[int]:
    if not txt.strip():
        return []
    itens = []
    for p in txt.replace(',', ' ').split():
        try:
            n = int(p)
        except ValueError:
            continue
        if 0 <= n <= 99:
            itens.append(n)
    return sorted(set(itens))

def main():
    print("=== Análise de Sorteios (Lotomania) — v3 ===")
    caminho_txt = input("Arquivo TXT com sorteios (ex.: ex.txt): ").strip() or "ex.txt"

    padrao_x, padrao_y = 3.0, 5.0
    faixa_str = input(f"Faixa % faltante para último bloco (x-y) [padrão {padrao_x}-{padrao_y}]: ").strip()
    if faixa_str:
        try:
            partes = faixa_str.replace('%', '').split('-')
            x = float(partes[0].replace(',', '.'))
            y = float(partes[1].replace(',', '.'))
        except Exception:
            print("Entrada inválida; usando padrão 3-5%.")
            x, y = padrao_x, padrao_y
    else:
        x, y = padrao_x, padrao_y

    max_x_default = 10
    min_x_default = 0
    s_max = input(f"Limiar 'Máximo Maior' (padrão {max_x_default}): ").strip()
    s_min = input(f"Limiar 'Mínimo Menor' (padrão {min_x_default}): ").strip()
    try:
        max_x = int(s_max) if s_max else max_x_default
    except Exception:
        max_x = max_x_default
    try:
        min_x = int(s_min) if s_min else min_x_default
    except Exception:
        min_x = min_x_default

    base_str = input("Números base (separados por vírgula ou espaço), ex.: 1,4  (opcional): ").strip()
    numeros_base = parse_lista_numeros(base_str)

    saida = input("Nome do arquivo Excel de saída (.xlsx) (opcional): ").strip()
    if not saida:
        saida = None
    else:
        if not saida.lower().endswith(".xlsx"):
            saida = saida + ".xlsx"

    try:
        caminho_saida = gerar_planilhas(
            caminho_txt=caminho_txt,
            faixa_percent=(x, y),
            max_x=max_x,
            min_x=min_x,
            numeros_base=numeros_base,
            saida_xlsx=saida
        )
        print(f"\nConcluído! Arquivo gerado: {caminho_saida}")
    except Exception as e:
        print(f"\nERRO: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
