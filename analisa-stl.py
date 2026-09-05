#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "trimesh",
#     "scipy",
#     "networkx",
# ]
# ///
"""
Analisa malhas STL e diz se são estanques (watertight), explicando o porquê.

Companheiro do simplifica-stl.py. Mesmas bibliotecas, mesmo esquema de
dependências inline (PEP 723): basta executar o arquivo.

Código de saída: 0 = estanque, 1 = não estanque, 2 = erro de leitura.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh


# ---------------------------------------------------------------- utilidades

def humaniza(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024


def formato_do_arquivo(caminho):
    with open(caminho, "rb") as fh:
        cabecalho = fh.read(5)
    return "ASCII" if cabecalho[:5].lower() == b"solid" else "binário"


def arestas_de_borda(malha):
    """Arestas usadas por uma única face — ou seja, o contorno de um buraco."""
    grupos = trimesh.grouping.group_rows(malha.edges_sorted, require_count=1)
    return malha.edges_sorted[grupos] if len(grupos) else np.empty((0, 2), dtype=np.int64)


def arestas_nao_manifold(malha):
    """Arestas compartilhadas por 3+ faces: topologia impossível de imprimir."""
    _, inverso, contagens = np.unique(
        malha.edges_sorted, axis=0, return_inverse=True, return_counts=True
    )
    return int((contagens > 2).sum())


def conta_bordas_fechadas(arestas):
    """Agrupa as arestas de borda em laços conexos = número de buracos."""
    if len(arestas) == 0:
        return 0, []
    pai = {}

    def raiz(x):
        pai.setdefault(x, x)
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    def une(a, b):
        ra, rb = raiz(a), raiz(b)
        if ra != rb:
            pai[ra] = rb

    for a, b in arestas:
        une(int(a), int(b))

    laços = {}
    for i, (a, _) in enumerate(arestas):
        laços.setdefault(raiz(int(a)), []).append(i)
    return len(laços), list(laços.values())


def solda_por_proximidade(malha, tol):
    """Funde vértices a menos de `tol` de distância e devolve a malha soldada.

    Não dá para usar o merge_vertices do trimesh aqui: ele funde por
    arredondamento de casas decimais, então dois vértices a 2e-7 de distância
    caem em baldes diferentes se estiverem na fronteira do arredondamento.
    Agrupar por distância real (KDTree + union-find) responde de fato qual é a
    menor tolerância que fecha a malha.
    """
    from scipy.spatial import cKDTree

    v = np.asarray(malha.vertices)
    pares = cKDTree(v).query_pairs(r=tol, output_type="ndarray")

    rotulo = np.arange(len(v))

    def raiz(x):
        while rotulo[x] != x:
            rotulo[x] = rotulo[rotulo[x]]
            x = rotulo[x]
        return x

    for a, b in pares:
        ra, rb = raiz(int(a)), raiz(int(b))
        if ra != rb:
            rotulo[ra] = rb

    canonico = np.array([raiz(i) for i in range(len(v))])
    faces = canonico[malha.faces]
    # colapsar vértices transforma alguns triângulos em linhas: descartá-los
    validas = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return trimesh.Trimesh(vertices=v, faces=faces[validas], process=False)


def tolerancia_que_fecha(malha, escala):
    """Menor tolerância de soldagem que torna a malha estanque, se existir.

    Se ela fecha com uma tolerância muito abaixo da escala da peça, os
    'buracos' eram frestas numéricas do float32 do STL, não falhas reais.
    """
    for expoente in range(-7, -1):
        tol = escala * (10.0 ** expoente)
        try:
            if solda_por_proximidade(malha, tol).is_watertight:
                return tol
        except Exception:
            continue
    return None


# ------------------------------------------------------------------- análise

def analisa(caminho, args, saida):
    caminho = Path(caminho)
    bruta = trimesh.load_mesh(str(caminho), file_type="stl", process=False)
    if isinstance(bruta, trimesh.Scene):
        bruta = trimesh.util.concatenate(tuple(bruta.geometry.values()))
    faces_arquivo = len(bruta.faces)

    malha = trimesh.load_mesh(str(caminho), file_type="stl", process=True)
    if isinstance(malha, trimesh.Scene):
        malha = trimesh.util.concatenate(tuple(malha.geometry.values()))
    if len(malha.faces) == 0:
        raise ValueError("arquivo não contém malha triangular")

    dim = malha.extents
    escala = float(np.max(dim)) if dim is not None and len(dim) else 1.0

    saida(f"\n=== {caminho.name} ===")
    saida(f"  arquivo        : {humaniza(caminho.stat().st_size)}, STL {formato_do_arquivo(caminho)}")
    saida(f"  triângulos     : {faces_arquivo:,}")
    saida(f"  vértices       : {len(malha.vertices):,} únicos "
          f"(o arquivo repete {faces_arquivo * 3:,} — STL não compartilha vértices)")
    saida(f"  dimensões      : {dim[0]:.3f} x {dim[1]:.3f} x {dim[2]:.3f}")

    # --- diagnóstico topológico
    bordas = arestas_de_borda(malha)
    n_buracos, laços = conta_bordas_fechadas(bordas)
    n_nao_manifold = arestas_nao_manifold(malha)

    areas = malha.area_faces
    n_degeneradas = int((areas <= 0).sum())
    n_orfaos = len(malha.vertices) - len(np.unique(malha.faces))
    grupos_dup = trimesh.grouping.group_rows(np.sort(malha.faces, axis=1), require_count=2)
    n_duplicadas = len(grupos_dup)

    try:
        corpos = malha.body_count
    except Exception:
        corpos = "?"

    estanque = bool(malha.is_watertight)

    saida("")
    if estanque:
        saida("  VEREDITO: ESTANQUE — a malha é um sólido fechado.")
    else:
        saida("  VEREDITO: NÃO ESTANQUE — a malha tem furos ou topologia inválida.")

    saida(f"  arestas abertas    : {len(bordas):,}"
          + (f"  formando {n_buracos} buraco(s)" if len(bordas) else ""))
    saida(f"  arestas não-manifold: {n_nao_manifold:,}"
          + ("   (3+ faces na mesma aresta)" if n_nao_manifold else ""))
    saida(f"  faces degeneradas  : {n_degeneradas:,}" + ("   (área zero)" if n_degeneradas else ""))
    saida(f"  faces duplicadas   : {n_duplicadas:,}")
    saida(f"  vértices órfãos    : {n_orfaos:,}")
    saida(f"  corpos separados   : {corpos}")
    saida(f"  normais coerentes  : {'sim' if malha.is_winding_consistent else 'NÃO'}")
    saida(f"  característica de Euler: {malha.euler_number}"
          + ("   (2 = esfera topológica, sem alças nem furos)"
             if malha.euler_number == 2 else ""))

    saida(f"  área               : {malha.area:.4f}")
    if estanque:
        saida(f"  volume             : {malha.volume:.4f}"
              f"   (normais para {'fora' if malha.volume > 0 else 'DENTRO — invertidas'})")
    else:
        saida("  volume             : indefinido (só faz sentido em malha fechada)")

    # --- tamanho dos buracos
    if len(bordas) and laços:
        v = malha.vertices
        comprimentos = np.linalg.norm(v[bordas[:, 0]] - v[bordas[:, 1]], axis=1)
        perims = sorted((float(comprimentos[idx].sum()) for idx in laços), reverse=True)
        saida("\n  buracos por perímetro (maiores primeiro):")
        for i, p in enumerate(perims[:args.max_buracos], 1):
            rel = 100.0 * p / escala if escala else 0.0
            marca = "fresta numérica" if rel < 0.1 else ("pequeno" if rel < 5 else "GRANDE")
            saida(f"    {i:2d}. perímetro {p:.5f}  ({rel:.3f}% da maior dimensão) — {marca}")
        if len(perims) > args.max_buracos:
            saida(f"    ... e mais {len(perims) - args.max_buracos} buraco(s). "
                  f"Use --max-buracos N para ver mais.")

    # --- fresta numérica x buraco real
    if not estanque and not args.rapido:
        saida("\n  testando se são apenas frestas de precisão do float32...")
        tol = tolerancia_que_fecha(malha, escala)
        if tol is not None:
            rel = 100.0 * tol / escala if escala else 0.0
            saida(f"    a malha FECHA soldando vértices a {tol:g} de distância "
                  f"({rel:.5f}% da maior dimensão).")
            saida("    => são frestas numéricas do float32, não falhas de modelagem;")
            saida("       a geometria está inteira, só os vértices da costura não coincidem.")
        else:
            saida("    não fecha em nenhuma tolerância até 1% da peça")
            saida("    => há buraco real de geometria (falta face).")

    return estanque


def main():
    p = argparse.ArgumentParser(
        prog="analisa-stl",
        description="Diz se um STL é estanque e, se não for, explica o que está quebrado.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""exemplos:
  analisa-stl.py peca.stl
  analisa-stl.py *.stl                 # analisa em lote
  analisa-stl.py --rapido peca.stl     # pula o teste de tolerância (mais rápido)

código de saída: 0 = todos estanques, 1 = algum não estanque, 2 = erro
""",
    )
    p.add_argument("arquivos", nargs="+", type=Path, help="arquivos STL a analisar")
    p.add_argument("--rapido", action="store_true",
                   help="não testa tolerâncias de fusão de vértices")
    p.add_argument("--max-buracos", type=int, default=10, metavar="N",
                   help="quantos buracos listar por arquivo (padrão: 10)")
    p.add_argument("-q", "--silencioso", action="store_true",
                   help="só o veredito por arquivo")
    args = p.parse_args()

    linhas = []
    saida = linhas.append if args.silencioso else (lambda m: print(m, flush=True))

    algum_furado = False
    erros = 0
    for arq in args.arquivos:
        try:
            if not arq.is_file():
                raise ValueError("arquivo não encontrado")
            ok = analisa(arq, args, saida)
            if args.silencioso:
                print(f"{'ESTANQUE    ' if ok else 'NÃO ESTANQUE'}  {arq}", flush=True)
            if not ok:
                algum_furado = True
        except Exception as e:
            print(f"erro em {arq}: {e}", file=sys.stderr)
            erros += 1

    if not args.silencioso:
        print()
    if erros:
        return 2
    return 1 if algum_furado else 0


if __name__ == "__main__":
    sys.exit(main())
