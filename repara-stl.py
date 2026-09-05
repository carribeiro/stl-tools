#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# # Versões fixas de propósito: a imagem de container roda sem rede, com
# # cache uv pré-populado, e resolver versão exigiria o índice. Fixar também
# # evita divergência numérica entre a máquina local e o servidor.
# dependencies = [
#     "numpy==2.5.2",
#     "trimesh==5.1.0",
#     "scipy==1.18.1",
#     "networkx==3.6.1",
# ]
# ///
"""
Repara malhas STL: solda frestas, remove lixo topológico e fecha buracos.

Companheiro do analisa-stl.py e do simplifica-stl.py. Rode o analisa antes e
depois para comparar. A ordem das etapas importa: soldar primeiro, limpar
degeneradas/duplicadas depois, resolver não-manifold em seguida e só então
preencher buracos — inverter isso costuma criar defeitos novos.

Código de saída: 0 = terminou estanque, 1 = melhorou mas ainda não estanque,
2 = erro.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh


def humaniza(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024


def estado(malha):
    """(abertas, nao_manifold, degeneradas, duplicadas) da malha atual."""
    _, cont = np.unique(malha.edges_sorted, axis=0, return_counts=True)
    abertas = int((cont == 1).sum())
    nao_manifold = int((cont > 2).sum())
    degeneradas = int((malha.area_faces <= 0).sum())
    duplicadas = len(trimesh.grouping.group_rows(np.sort(malha.faces, axis=1), require_count=2))
    return abertas, nao_manifold, degeneradas, duplicadas


def solda(malha, tol):
    """Funde vértices a menos de `tol` de distância (KDTree + union-find).

    Não uso merge_vertices do trimesh: ele funde por arredondamento de casas
    decimais, então vértices a 2e-7 caem em baldes diferentes se estiverem na
    fronteira do arredondamento.
    """
    from scipy.spatial import cKDTree

    v = np.asarray(malha.vertices)
    pares = cKDTree(v).query_pairs(r=tol, output_type="ndarray")
    if len(pares) == 0:
        return malha, 0

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
    validas = ((faces[:, 0] != faces[:, 1])
               & (faces[:, 1] != faces[:, 2])
               & (faces[:, 0] != faces[:, 2]))
    nova = trimesh.Trimesh(vertices=v, faces=faces[validas], process=False)
    nova.remove_unreferenced_vertices()
    return nova, len(v) - len(nova.vertices)


def resolve_nao_manifold(malha, max_iter, log):
    """Desfaz arestas com 3+ faces descartando as faces pior costuradas.

    Escolher pela área é enganoso: uma aleta espúria pode ser enorme e uma face
    legítima da casca pode ser minúscula. O critério que funciona é quantas das
    3 arestas da face são compartilhadas por exatamente 2 faces — uma aleta tem
    as outras duas arestas soltas (nota 0), enquanto uma face bem integrada tem
    nota 2. Mantemos as duas de maior nota e usamos a área só como desempate.
    """
    total = 0
    for _ in range(max_iter):
        arestas = malha.edges_sorted
        _, inverso, cont = np.unique(arestas, axis=0, return_inverse=True, return_counts=True)
        ruins = np.flatnonzero(cont > 2)
        if len(ruins) == 0:
            break

        face_da_aresta = np.repeat(np.arange(len(malha.faces)), 3)
        # 3 arestas por face, na ordem das faces: dá para dobrar em (n_faces, 3)
        nota = (cont == 2)[inverso].reshape(-1, 3).sum(axis=1)
        areas = malha.area_faces
        remover = set()

        for e in ruins:
            fs = [int(f) for f in face_da_aresta[inverso == e] if int(f) not in remover]
            if len(fs) <= 2:
                continue
            # pior costurada primeiro; entre iguais, a de menor área
            ordem = sorted(fs, key=lambda f: (nota[f], areas[f]))
            for f in ordem[: len(fs) - 2]:
                remover.add(f)

        if not remover:
            break
        manter = np.ones(len(malha.faces), dtype=bool)
        manter[list(remover)] = False
        malha.update_faces(manter)
        malha.remove_unreferenced_vertices()
        total += len(remover)
        log(f"    removidas {len(remover):,} faces excedentes")
    return malha, total


def laços_de_borda(malha):
    """Encadeia as arestas de borda em laços fechados, já no sentido da tampa.

    Cada aresta de borda é usada por uma única face, numa direção. A tampa
    precisa percorrê-la no sentido oposto para que a normal saia coerente com
    a vizinhança — por isso a inversão abaixo.
    """
    dirigidas = np.asarray(malha.edges)
    grupos = trimesh.grouping.group_rows(malha.edges_sorted, require_count=1)
    if len(grupos) == 0:
        return []

    saidas = {}
    for i in grupos:
        a, b = int(dirigidas[i][1]), int(dirigidas[i][0])
        saidas.setdefault(a, []).append(b)

    laços, usadas = [], set()
    for inicio, destinos in list(saidas.items()):
        for primeiro in destinos:
            if (inicio, primeiro) in usadas:
                continue
            laço, a, b = [], inicio, primeiro
            while (a, b) not in usadas:
                usadas.add((a, b))
                laço.append((a, b))
                seguintes = [x for x in saidas.get(b, []) if (b, x) not in usadas]
                if not seguintes:
                    break
                a, b = b, seguintes[0]
            if len(laço) >= 3:
                laços.append(laço)
    return laços


def preenche_buracos(malha, escala, max_rel):
    """Tampa cada laço de borda com um leque a partir do seu centroide.

    O fill_holes do trimesh só resolve furos de 3 ou 4 arestas; o leque lida
    com laços de qualquer tamanho, inclusive não planares.
    """
    laços = laços_de_borda(malha)
    if not laços:
        return malha, 0, 0

    v = np.asarray(malha.vertices)
    novos_vertices, novas_faces = [], []
    preenchidos = pulados = 0

    for laço in laços:
        perimetro = sum(float(np.linalg.norm(v[a] - v[b])) for a, b in laço)
        if max_rel and perimetro > max_rel * escala:
            pulados += 1
            continue
        centro = v[[a for a, _ in laço]].mean(axis=0)
        indice_centro = len(v) + len(novos_vertices)
        novos_vertices.append(centro)
        novas_faces.extend([a, b, indice_centro] for a, b in laço)
        preenchidos += 1

    if not novas_faces:
        return malha, 0, pulados

    nova = trimesh.Trimesh(
        vertices=np.vstack([v, np.array(novos_vertices)]),
        faces=np.vstack([malha.faces, np.array(novas_faces, dtype=np.int64)]),
        process=False,
    )
    return nova, preenchidos, pulados


def repara(entrada, args, log):
    entrada = Path(entrada)
    saida = Path(args.saida) if args.saida else entrada.with_name(f"{entrada.stem}-reparado.stl")
    if saida.exists() and not args.sobrescrever:
        raise ValueError(f"{saida} já existe; use --sobrescrever")

    malha = trimesh.load_mesh(str(entrada), file_type="stl", process=True)
    if isinstance(malha, trimesh.Scene):
        malha = trimesh.util.concatenate(tuple(malha.geometry.values()))
    if len(malha.faces) == 0:
        raise ValueError("arquivo não contém malha triangular")

    faces_iniciais = len(malha.faces)
    a0, n0, d0, p0 = estado(malha)
    escala = float(np.max(malha.extents))
    log(f"{entrada.name}: {faces_iniciais:,} faces, {humaniza(entrada.stat().st_size)}")
    log(f"  antes : abertas={a0:,} nao-manifold={n0:,} degeneradas={d0:,} duplicadas={p0:,} "
        f"estanque={malha.is_watertight}")

    # 1) soldar frestas numéricas
    if args.solda != "nao":
        tol = escala * 1e-5 if args.solda == "auto" else float(args.solda)
        log(f"  [1] soldando vértices a {tol:g} ({100 * tol / escala:.5f}% da peça)...")
        malha, fundidos = solda(malha, tol)
        log(f"      {fundidos:,} vértices fundidos")
    else:
        log("  [1] soldagem desativada")

    # 2-4) limpeza, não-manifold e buracos até convergir
    # Preencher um buraco pode criar face duplicada ou aresta não-manifold nova,
    # e remover uma face não-manifold pode abrir um buraco. Por isso os três
    # passos se repetem até o estado parar de mudar.
    for passe in range(1, args.passes + 1):
        anterior = estado(malha)
        log(f"  [passe {passe}]")

        antes = len(malha.faces)
        malha.update_faces(malha.nondegenerate_faces())
        malha.update_faces(malha.unique_faces())
        malha.remove_unreferenced_vertices()
        if antes != len(malha.faces):
            log(f"      limpeza: {antes - len(malha.faces):,} faces degeneradas/duplicadas")

        malha, removidas = resolve_nao_manifold(malha, args.max_iter, log)
        if removidas:
            log(f"      nao-manifold: {removidas:,} faces removidas")

        if not args.sem_preencher:
            malha, cheios, pulados = preenche_buracos(malha, escala, args.max_buraco)
            if cheios or pulados:
                log(f"      buracos: {cheios:,} preenchidos"
                    + (f", {pulados:,} grandes demais (acima de "
                       f"{100 * args.max_buraco:.0f}% da peça) — deixados abertos"
                       if pulados else ""))

        if malha.is_watertight:
            log(f"      convergiu: estanque no passe {passe}")
            break
        if estado(malha) == anterior:
            log(f"      estado parou de mudar no passe {passe}")
            break

    # 5) normais
    log("  [5] corrigindo orientação das normais...")
    trimesh.repair.fix_winding(malha)
    if malha.is_watertight:
        trimesh.repair.fix_inversion(malha)

    a1, n1, d1, p1 = estado(malha)
    ok = bool(malha.is_watertight)
    log(f"  depois: abertas={a1:,} nao-manifold={n1:,} degeneradas={d1:,} duplicadas={p1:,} "
        f"estanque={ok}")
    log(f"  faces : {faces_iniciais:,} -> {len(malha.faces):,} "
        f"({100.0 * len(malha.faces) / faces_iniciais:.2f}% do original)")

    if not ok and not args.forcar:
        log("  NÃO gravei: a malha ainda não está estanque. Use --forcar para gravar assim mesmo.")
        return False

    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_bytes(trimesh.exchange.stl.export_stl(malha))
    log(f"  -> {saida} ({humaniza(saida.stat().st_size)})")
    if ok:
        log(f"  volume: {malha.volume:.2f}")
    return ok


def main():
    p = argparse.ArgumentParser(
        prog="repara-stl",
        description="Repara malhas STL: solda frestas, limpa topologia e fecha buracos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""exemplos:
  repara-stl.py peca.stl
  repara-stl.py --solda 0.001 -o boa.stl peca.stl
  repara-stl.py --sem-preencher peca.stl      # só limpa topologia, não inventa faces

código de saída: 0 = ficou estanque, 1 = melhorou mas não fechou, 2 = erro
""",
    )
    p.add_argument("arquivo", type=Path)
    p.add_argument("-o", "--saida", type=Path, help="padrão: <nome>-reparado.stl")
    p.add_argument("--solda", default="auto", metavar="TOL|auto|nao",
                   help="tolerância de soldagem; auto = 0,001%% da peça (padrão: auto)")
    p.add_argument("--sem-preencher", action="store_true",
                   help="não preenche buracos (não inventa geometria)")
    p.add_argument("--max-iter", type=int, default=10,
                   help="iterações internas do passo não-manifold (padrão: 10)")
    p.add_argument("--passes", type=int, default=6,
                   help="passes de limpeza/não-manifold/buracos (padrão: 6)")
    p.add_argument("--max-buraco", type=float, default=0.5, metavar="FRAC",
                   help="não preenche laço com perímetro acima desta fração da "
                        "maior dimensão; 0 = preenche tudo (padrão: 0.5)")
    p.add_argument("--forcar", action="store_true", help="grava mesmo sem ficar estanque")
    p.add_argument("--sobrescrever", action="store_true")
    p.add_argument("-q", "--silencioso", action="store_true")
    args = p.parse_args()

    log = (lambda m: None) if args.silencioso else (lambda m: print(m, flush=True))
    try:
        if not args.arquivo.is_file():
            raise ValueError("arquivo não encontrado")
        return 0 if repara(args.arquivo, args, log) else 1
    except Exception as e:
        print(f"erro em {args.arquivo}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
