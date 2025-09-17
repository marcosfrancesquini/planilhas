#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

def contar_linhas(caminho, encoding='utf-8'):
    total = 0
    with open(caminho, 'r', encoding=encoding, newline='') as f:
        for _ in f:
            total += 1
    return total

def processar(caminho, corte, encoding='utf-8'):
    total = contar_linhas(caminho, encoding=encoding)
    print(f"Total de linhas: {total}")

    if corte < 1 or corte >= total:
        print("⚠️ valor de corte inválido.")
        return

    base_dir = os.path.dirname(os.path.abspath(caminho))
    nome = os.path.basename(caminho)
    stem, _ = os.path.splitext(nome)

    arq_primeiras = os.path.join(base_dir, f"{stem}_primeiras_{corte}.txt")
    arq_linha_unica = os.path.join(base_dir, f"{stem}_linha_{corte+1}.txt")

    linha_n_mais_1 = None
    with open(caminho, 'r', encoding=encoding, newline='') as fin, \
         open(arq_primeiras, 'w', encoding=encoding, newline='') as fout_primeiras:
        for idx, linha in enumerate(fin, start=1):
            if idx <= corte:
                fout_primeiras.write(linha)
            elif idx == corte + 1:
                linha_n_mais_1 = linha
            else:
                pass

    with open(arq_linha_unica, 'w', encoding=encoding, newline='') as f:
        if linha_n_mais_1 is not None:
            f.write(linha_n_mais_1)

    print("✅ operação concluída:")
    print(f" - Linha {corte+1} salva em {arq_linha_unica}")
    print(f" - Primeiras {corte} linhas salvas em {arq_primeiras}")

if __name__ == "__main__":
    caminho = input("Digite o caminho do arquivo .txt: ").strip()
    corte = int(input("Digite o número da linha onde cortar (N): ").strip())
    processar(caminho, corte)
