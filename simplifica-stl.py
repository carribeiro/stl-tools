#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# # Versões fixas de propósito: a imagem de container roda sem rede, com
# # cache uv pré-populado, e resolver versão exigiria o índice. Fixar também
# # evita divergência numérica entre a máquina local e o servidor.
# dependencies = [
#     "numpy==2.5.2",
#     "trimesh==5.1.0",
#     "fast-simplification==0.2.0",
# ]
# ///
"""
Simplifica malhas STL pela linha de comando (decimação quádrica).

Substitui o open3d, que não tem wheel para o Python 3.14 do Ubuntu 26.04.
O algoritmo é o mesmo do open3d.simplify_quadric_decimation (Garland-Heckbert),
via a biblioteca fast-simplification.

As dependências são declaradas inline (PEP 723) e o uv monta o ambiente
sozinho no primeiro uso. Basta executar o arquivo.
"""

import argparse
import datetime
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import fast_simplification


# ------------------------------------------------------------- instrumentação
# Este bloco é duplicado nos três scripts de propósito. Cada um precisa ser um
# arquivo único e autossuficiente: o roda-remoto.sh envia UM só arquivo e a
# imagem de container assa cada script isoladamente, então um módulo
# compartilhado quebraria a execução remota.

_INICIO = time.monotonic()
_RESUMO = []


def _pico_memoria_mib():
    """Pico de RSS do processo. Não custa nada: o kernel já mantém o número."""
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1048576 if sys.platform == "darwin" else kb / 1024


def _encerra(ferramenta, codigo):
    """Reporta tempo e memória, e anexa uma linha ao log de invocações."""
    segundos = time.monotonic() - _INICIO
    memoria = _pico_memoria_mib()
    print(f"[{ferramenta}] tempo {segundos:.1f}s · pico de memória "
          f"{memoria:,.0f} MiB", file=sys.stderr)

    caminho = os.environ.get("STL_TOOLS_LOG") or os.path.expanduser(
        "~/.local/state/stl-tools/invocacoes.jsonl")
    try:
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        with open(caminho, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "quando": datetime.datetime.now().astimezone().isoformat(
                    timespec="seconds"),
                "ferramenta": ferramenta,
                "argumentos": sys.argv[1:],
                "resultados": _RESUMO,
                "codigo": codigo,
                "segundos": round(segundos, 2),
                "pico_memoria_mib": round(memoria),
            }, ensure_ascii=False) + "\n")
    except OSError:
        pass  # o log é conveniência; nunca deve derrubar o processamento
    return codigo


def humaniza(n_bytes):
    for unidade in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unidade == "GB":
            return f"{n_bytes:.0f} {unidade}" if unidade == "B" else f"{n_bytes:.1f} {unidade}"
        n_bytes /= 1024


def topologia(malha):
    """(abertas, nao_manifold, duplicadas) — para comparar antes e depois.

    A decimação quádrica não garante saída manifold: quanto mais agressiva a
    redução, mais arestas com 3+ faces ela cria. Medir os dois lados é a única
    forma de o estrago não passar em silêncio.
    """
    _, cont = np.unique(malha.edges_sorted, axis=0, return_counts=True)
    duplicadas = len(trimesh.grouping.group_rows(
        np.sort(malha.faces, axis=1), require_count=2))
    return int((cont == 1).sum()), int((cont > 2).sum()), duplicadas


def carrega(caminho):
    """Carrega o STL como uma malha única, com vértices duplicados fundidos.

    STL é uma 'sopa de triângulos': cada face repete seus 3 vértices e nada é
    compartilhado. Sem fundir os vértices primeiro, a decimação não encontra
    arestas para colapsar e não simplifica quase nada. O process=True do
    trimesh faz essa fusão.
    """
    malha = trimesh.load_mesh(str(caminho), file_type="stl", process=True)
    if isinstance(malha, trimesh.Scene):
        malha = trimesh.util.concatenate(tuple(malha.geometry.values()))
    if not isinstance(malha, trimesh.Trimesh) or len(malha.faces) == 0:
        raise ValueError("arquivo não contém uma malha triangular utilizável")
    return malha


def caminho_saida(entrada, args, varias_entradas):
    entrada = Path(entrada)
    if args.saida:
        destino = Path(args.saida)
        if varias_entradas or destino.is_dir():
            destino.mkdir(parents=True, exist_ok=True)
            return destino / f"{entrada.stem}{args.sufixo}.stl"
        return destino
    return entrada.with_name(f"{entrada.stem}{args.sufixo}.stl")


def processa(entrada, args, varias_entradas, log):
    entrada = Path(entrada)
    saida = caminho_saida(entrada, args, varias_entradas)

    if saida.resolve() == entrada.resolve() and not args.sobrescrever:
        raise ValueError("saída igual à entrada; use --sobrescrever se for essa a intenção")
    if saida.exists() and not args.sobrescrever:
        raise ValueError(f"{saida} já existe; use --sobrescrever")

    malha = carrega(entrada)
    faces_antes = len(malha.faces)
    verts_antes = len(malha.vertices)
    tam_antes = entrada.stat().st_size

    if args.faces is not None:
        alvo = dict(target_count=min(args.faces, faces_antes))
    else:
        alvo = dict(target_reduction=1.0 - args.razao)

    log(f"{entrada.name}: {faces_antes:,} triângulos / {verts_antes:,} vértices "
        f"({humaniza(tam_antes)})")

    pontos, faces = fast_simplification.simplify(
        malha.vertices.astype(np.float64),
        malha.faces.astype(np.int32),
        agg=args.agressividade,
        lossless=args.lossless,
        preserve_border=args.preserva_borda,
        **({} if args.lossless else alvo),
    )

    nova = trimesh.Trimesh(vertices=pontos, faces=faces, process=True)
    nova.remove_unreferenced_vertices()
    if args.corrige_normais:
        nova.fix_normals()

    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_bytes(
        trimesh.exchange.stl.export_stl_ascii(nova).encode()
        if args.ascii
        else trimesh.exchange.stl.export_stl(nova)
    )

    faces_depois = len(nova.faces)
    tam_depois = saida.stat().st_size

    if not args.sem_conferir:
        ab0, nm0, dp0 = topologia(malha)
        ab1, nm1, dp1 = topologia(nova)
        if (nm1, dp1) > (nm0, dp0):
            log(f"  ATENÇÃO: a decimação degradou a topologia — "
                f"nao-manifold {nm0:,} -> {nm1:,}, duplicadas {dp0:,} -> {dp1:,}, "
                f"abertas {ab0:,} -> {ab1:,}.")
            log(f"  O dano cresce com a redução. Considere uma razão maior, ou "
                f"rodar o repara-stl.py depois e conferir com o analisa-stl.py.")
    pct = 100.0 * faces_depois / faces_antes if faces_antes else 0.0
    log(f"  -> {saida.name}: {faces_depois:,} triângulos ({pct:.1f}% do original), "
        f"{humaniza(tam_depois)} "
        f"[-{100 - 100.0 * tam_depois / tam_antes:.0f}% de tamanho]"
        f"{'' if nova.is_watertight else '  [ATENÇÃO: malha não é estanque]'}")
    _RESUMO.append({
        "arquivo": entrada.name, "saida": saida.name,
        "faces_antes": faces_antes, "faces_depois": faces_depois,
        "razao": round(faces_depois / faces_antes, 4) if faces_antes else None,
        "bytes_antes": tam_antes, "bytes_depois": tam_depois,
        "estanque": bool(nova.is_watertight),
        "topologia_antes": None if args.sem_conferir else list(topologia(malha)),
        "topologia_depois": None if args.sem_conferir else list(topologia(nova)),
    })
    return saida


def main():
    p = argparse.ArgumentParser(
        prog="simplifica-stl",
        description="Simplifica malhas STL por decimação quádrica.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""exemplos:
  simplifica-stl.py peca.stl                      # mantém 50% dos triângulos
  simplifica-stl.py -r 0.1 peca.stl               # mantém 10%
  simplifica-stl.py -n 20000 peca.stl             # alvo de 20 mil triângulos
  simplifica-stl.py -r 0.25 -o simplificados/ *.stl
  simplifica-stl.py --lossless peca.stl           # só remove redundância, sem perder forma
""",
    )
    p.add_argument("arquivos", nargs="+", type=Path, help="arquivos STL de entrada")

    alvo = p.add_mutually_exclusive_group()
    alvo.add_argument("-r", "--razao", type=float, default=0.5,
                      help="fração de triângulos a MANTER, entre 0 e 1 (padrão: 0.5)")
    alvo.add_argument("-n", "--faces", type=int,
                      help="número alvo de triângulos (alternativa a --razao)")

    p.add_argument("-o", "--saida", type=Path,
                   help="arquivo de saída (uma entrada) ou diretório (várias)")
    p.add_argument("-s", "--sufixo", default="-simplif",
                   help="sufixo do nome de saída (padrão: -simplif)")
    p.add_argument("-a", "--agressividade", type=float, default=7.0, metavar="0..10",
                   help="10 = rápido e mais grosseiro, 0 = lento e mais fiel (padrão: 7)")
    p.add_argument("--preserva-borda", action="store_true",
                   help="não colapsa arestas que tocam borda aberta")
    p.add_argument("--lossless", action="store_true",
                   help="remove apenas geometria redundante; ignora --razao/--faces")
    p.add_argument("--corrige-normais", action="store_true",
                   help="reorienta as normais depois de simplificar")
    p.add_argument("--sem-conferir", action="store_true",
                   help="não compara a topologia antes/depois (poupa alguns "
                        "segundos em malha grande, mas o estrago passa calado)")
    p.add_argument("--ascii", action="store_true", help="grava STL ASCII (padrão: binário)")
    p.add_argument("--sobrescrever", action="store_true", help="sobrescreve arquivos existentes")
    p.add_argument("-q", "--silencioso", action="store_true")

    args = p.parse_args()

    if not args.lossless and args.faces is None and not 0.0 < args.razao <= 1.0:
        p.error("--razao precisa estar entre 0 (exclusivo) e 1 (inclusive)")
    if args.faces is not None and args.faces < 4:
        p.error("--faces precisa ser pelo menos 4")
    if not 0.0 <= args.agressividade <= 10.0:
        p.error("--agressividade precisa estar entre 0 e 10")

    log = (lambda *a, **k: None) if args.silencioso else \
        (lambda msg: print(msg, file=sys.stderr, flush=True))

    varias = len(args.arquivos) > 1
    erros = 0
    for entrada in args.arquivos:
        try:
            if not entrada.is_file():
                raise ValueError("arquivo não encontrado")
            processa(entrada, args, varias, log)
        except Exception as e:
            print(f"erro em {entrada}: {e}", file=sys.stderr)
            erros += 1

    return 1 if erros else 0


if __name__ == "__main__":
    sys.exit(_encerra("simplifica-stl", main()))
