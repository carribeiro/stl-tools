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
Suaviza uma região da malha, deixando o resto intacto.

Feito para ruído de digitalização localizado — fita, massa, marcação — sem
achatar as formas que importam. Usa o filtro de Taubin, que alterna passos
laplacianos de sinal oposto e por isso **não encolhe a peça**: o laplaciano
puro contrai o volume a cada iteração, e encolher uma casca estraga o encaixe
de tudo que foi dimensionado em cima dela.

A região é uma faixa arbitrária em qualquer eixo, em fração da peça ou em
coordenadas do modelo, com transição gradual nas bordas — corte seco deixaria
vinco visível na fronteira.

Código de saída: 0 = gravou, 2 = erro.
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


# ------------------------------------------------------------- instrumentação
# Este bloco é duplicado nos scripts de propósito. Cada um precisa ser um
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


# ---------------------------------------------------------------- utilidades

def humaniza(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024


def carrega(caminho):
    """STL é sopa de triângulos: sem process=True nada é compartilhado e a
    vizinhança de vértices — que a suavização precisa — não existe."""
    malha = trimesh.load_mesh(str(caminho), file_type="stl", process=True)
    if isinstance(malha, trimesh.Scene):
        malha = trimesh.util.concatenate(tuple(malha.geometry.values()))
    if len(malha.faces) == 0:
        raise ValueError("arquivo não contém malha triangular")
    return malha


def rampa(t):
    """Smoothstep: 0 a 1 com derivada nula nas pontas.

    Interpolação linear deixaria descontinuidade na derivada nos extremos da
    faixa, o que aparece como marca na superfície.
    """
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def peso_da_faixa(coord, de, ate, transicao):
    """Peso por vértice: 1 dentro da faixa, 0 fora, com rampa nas duas bordas.

    A faixa é fechada dos dois lados de propósito — assim a mesma opção
    descreve topo, base ou um trecho no meio, sem caso especial.
    """
    if transicao <= 0:
        return ((coord >= de) & (coord <= ate)).astype(float)
    sobe = rampa((coord - (de - transicao)) / transicao)
    desce = rampa(((ate + transicao) - coord) / transicao)
    return np.minimum(sobe, desce)


def achata_para_float32(vertices):
    """Arredonda para a precisão que o STL vai gravar de qualquer forma.

    Assim o que medimos aqui é o que sai no arquivo: gravar em float32 depois
    de medir em float64 muda a topologia por trás da medição.
    """
    return np.asarray(vertices, dtype=np.float32).astype(np.float64)


def vertices_de_borda(malha, aneis):
    """Vértices na borda aberta, mais `aneis` camadas de vizinhos.

    Suavização não sabe lidar com borda: o vértice de borda tem vizinhança
    incompleta, então o laplaciano o arrasta para dentro e o furo ABRE. Numa
    malha com fendas pequenas isso é destrutivo — medimos uma fenda passar de
    0,25 mm para 18 mm. Congelar a borda e uma faixa em volta evita o efeito.
    """
    grupos = trimesh.grouping.group_rows(malha.edges_sorted, require_count=1)
    mascara = np.zeros(len(malha.vertices), dtype=bool)
    if len(grupos) == 0:
        return mascara
    mascara[np.unique(malha.edges_sorted[grupos])] = True

    arestas = malha.edges_unique
    for _ in range(max(aneis, 0)):
        nova = mascara.copy()
        nova[arestas[mascara[arestas[:, 0]], 1]] = True
        nova[arestas[mascara[arestas[:, 1]], 0]] = True
        mascara = nova
    return mascara


def suaviza(entrada, args, log):
    entrada = Path(entrada)
    saida = Path(args.saida) if args.saida else entrada.with_name(
        f"{entrada.stem}-suave.stl")
    if saida.exists() and not args.sobrescrever:
        raise ValueError(f"{saida} já existe; use --sobrescrever")

    malha = carrega(entrada)
    originais = malha.vertices.copy()
    extensao_total = float(np.max(malha.extents))
    eixo = "xyz".index(args.eixo)
    coord = originais[:, eixo]
    c0, c1 = float(coord.min()), float(coord.max())
    extensao = c1 - c0

    if args.absoluto:
        de, ate, transicao = args.de, args.ate, args.transicao
    else:
        de = c0 + extensao * args.de
        ate = c0 + extensao * args.ate
        transicao = extensao * args.transicao
    if ate < de:
        de, ate = ate, de

    log(f"{entrada.name}: {len(malha.faces):,} faces, "
        f"{humaniza(entrada.stat().st_size)}")
    log(f"  eixo {args.eixo}: de {c0:.2f} a {c1:.2f}")
    log(f"  faixa : efeito total entre {de:.2f} e {ate:.2f}, "
        f"com rampa de {transicao:.2f} nas bordas")

    peso = peso_da_faixa(coord, de, ate, transicao)

    if not args.mexer_na_borda:
        congelados = vertices_de_borda(malha, args.aneis_de_borda)
        if congelados.any():
            peso[congelados] = 0.0
            log(f"  borda aberta: {int(congelados.sum()):,} vértices congelados "
                f"({args.aneis_de_borda} anéis) — suavizar borda ABRE o furo")

    atingidos = int((peso > 0.01).sum())
    log(f"  vértices afetados: {atingidos:,} de {len(originais):,} "
        f"({100 * atingidos / len(originais):.1f}%)")
    if atingidos == 0:
        raise ValueError("a faixa escolhida não contém vértice nenhum; "
                         "revise --de/--ate")

    volume_antes = float(malha.volume) if malha.is_watertight else None

    log(f"  suavizando (Taubin, {args.iteracoes} iterações)...")
    trimesh.smoothing.filter_taubin(malha, lamb=args.lamb, nu=args.nu,
                                    iterations=args.iteracoes)
    suavizados = malha.vertices.copy()

    # Mistura: fora da faixa fica exatamente o original, dentro entra o suave.
    delta = peso[:, None] * (suavizados - originais)

    # Teto de deslocamento. O filtro diverge em pontos isolados de triangulação
    # ruim — medimos 123 vértices voando mais de 20 numa peça de 190, com
    # mediana de 0,065 e p99 de 0,20. Como suavizar é, por definição, mover
    # pouco, qualquer coisa grande é divergência e não suavização: limitar o
    # vetor preserva a direção e mata o estrago, sem esconder que ocorreu.
    limite = extensao_total * args.limite if not args.absoluto else args.limite
    normas = np.linalg.norm(delta, axis=1)
    estourados = normas > limite
    if estourados.any():
        delta[estourados] *= (limite / normas[estourados])[:, None]
        log(f"  ATENÇÃO: {int(estourados.sum()):,} vértices passaram do teto de "
            f"{limite:.3f} e foram limitados (máximo bruto era "
            f"{normas.max():.3f}).")
        log(f"     Isso é divergência do filtro em triangulação ruim, não "
            f"suavização. Reduza -n se o número for grande.")

    finais = originais + delta
    malha.vertices = achata_para_float32(finais)

    deslocamento = np.linalg.norm(finais - originais, axis=1)
    dentro = peso > 0.01
    log(f"  deslocamento: médio {deslocamento[dentro].mean():.4f} · "
        f"p99 {np.percentile(deslocamento[dentro], 99):.4f} · "
        f"máximo {deslocamento.max():.4f}")

    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_bytes(trimesh.exchange.stl.export_stl(malha))

    conferida = trimesh.load_mesh(str(saida), file_type="stl", process=True)
    volume_depois = float(conferida.volume) if conferida.is_watertight else None
    log(f"  -> {saida.name} ({humaniza(saida.stat().st_size)}), "
        f"{len(conferida.faces):,} faces")
    if volume_antes and volume_depois:
        log(f"  volume: {volume_antes:.1f} -> {volume_depois:.1f} "
            f"({100 * (volume_depois - volume_antes) / volume_antes:+.3f}%)")
    else:
        log("  volume não comparável (malha não estanque) — o Taubin preserva "
            "volume por construção, mas aqui não dá para confirmar pelo número")

    _RESUMO.append({
        "arquivo": entrada.name, "saida": saida.name,
        "faces": len(conferida.faces),
        "vertices_afetados": atingidos,
        "deslocamento_max": round(float(deslocamento.max()), 5),
        "vertices_limitados": int(estourados.sum()),
        "volume_antes": volume_antes, "volume_depois": volume_depois,
    })
    return 0


def main():
    p = argparse.ArgumentParser(
        prog="suaviza-stl",
        description="Suaviza uma faixa de altura da malha, preservando o resto.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""exemplos:
  suaviza-stl.py peca.stl                          # terço superior (padrão)
  suaviza-stl.py --de 0.6 --ate 1 -n 30 peca.stl   # 40% do topo, mais forte
  suaviza-stl.py --de 0.3 --ate 0.5 peca.stl       # só um trecho do meio
  suaviza-stl.py --eixo x --de 0 --ate 0.25 peca.stl   # um lado da peça
  suaviza-stl.py --absoluto --de 120 --ate 190 --transicao 8 peca.stl
  suaviza-stl.py --de 0 --ate 1 --transicao 0 peca.stl # a peça inteira

código de saída: 0 = gravou, 2 = erro
""",
    )
    p.add_argument("arquivo", type=Path)
    p.add_argument("-o", "--saida", type=Path, help="padrão: <nome>-suave.stl")
    p.add_argument("--eixo", choices=("x", "y", "z"), default="z",
                   help="eixo que define a faixa (padrão: z)")
    p.add_argument("--de", type=float, default=0.667, metavar="V",
                   help="início da faixa, em fração do eixo (padrão: 0.667)")
    p.add_argument("--ate", type=float, default=1.0, metavar="V",
                   help="fim da faixa, em fração do eixo (padrão: 1.0)")
    p.add_argument("--absoluto", action="store_true",
                   help="interpreta --de/--ate/--transicao em coordenadas do "
                        "modelo em vez de fração")
    p.add_argument("--transicao", type=float, default=0.1, metavar="V",
                   help="largura da rampa nas duas bordas da faixa "
                        "(padrão: 0.1). Zero deixa vinco visível.")
    p.add_argument("-n", "--iteracoes", type=int, default=20,
                   help="iterações do filtro (padrão: 20)")
    p.add_argument("--lamb", type=float, default=0.5,
                   help="passo positivo do Taubin (padrão: 0.5)")
    p.add_argument("--nu", type=float, default=0.53,
                   help="passo negativo do Taubin; precisa ser maior que --lamb "
                        "para não encolher (padrão: 0.53)")
    p.add_argument("--limite", type=float, default=0.01, metavar="V",
                   help="teto de deslocamento por vértice, em fração da maior "
                        "dimensão (padrão: 0.01). Segue --absoluto.")
    p.add_argument("--aneis-de-borda", type=int, default=3, metavar="N",
                   help="camadas de vértices congeladas em volta de borda "
                        "aberta (padrão: 3)")
    p.add_argument("--mexer-na-borda", action="store_true",
                   help="não congela a borda aberta; só use em malha fechada, "
                        "porque suavizar borda alarga o furo")
    p.add_argument("--sobrescrever", action="store_true")
    p.add_argument("-q", "--silencioso", action="store_true")
    args = p.parse_args()

    if not args.absoluto and not (0.0 <= args.de <= 1.0 and 0.0 <= args.ate <= 1.0):
        p.error("--de e --ate são frações entre 0 e 1; use --absoluto para "
                "coordenadas do modelo")
    if args.nu <= args.lamb:
        p.error("--nu precisa ser maior que --lamb, senão o filtro encolhe a peça")

    log = (lambda m: None) if args.silencioso else (lambda m: print(m, flush=True))
    try:
        if not args.arquivo.is_file():
            raise ValueError("arquivo não encontrado")
        return suaviza(args.arquivo, args, log)
    except Exception as e:
        print(f"erro em {args.arquivo}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(_encerra("suaviza-stl", main()))
