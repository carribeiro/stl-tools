#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# # Versões fixas de propósito: a imagem de container roda sem rede, com
# # cache uv pré-populado, e resolver versão exigiria o índice. Fixar também
# # evita divergência numérica entre a máquina local e o servidor.
# #
# # scikit-image entra aqui e não nos outros três: é a única dependência do
# # conjunto que traz marching cubes, e não há como extrair isosuperfície com
# # numpy/scipy sozinhos. Versão atual em 09/2026, wheel para cp314.
# dependencies = [
#     "numpy==2.5.2",
#     "trimesh==5.1.0",
#     "scipy==1.18.1",
#     "scikit-image==0.26.0",
# ]
# ///
"""
Transforma um STL sólido em casco oco com parede de espessura uniforme.

Companheiro do analisa-stl.py, do repara-stl.py e do simplifica-stl.py.
A malha de entrada precisa estar estanque: passe o repara-stl.py antes se o
analisa-stl.py acusar buraco.

A saída tem duas superfícies fechadas — a externa original, intacta, e a
interna deslocada para dentro com as normais invertidas. É a representação
padrão de peça oca e todo slicer entende.

Código de saída: 0 = gravou casca estanque, 1 = espessura não deixa cavidade
(não gravou), 2 = erro — inclui o caso raro de gravar e o arquivo sair não
estanque.
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
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure


# ------------------------------------------------------------- instrumentação
# Este bloco é duplicado nos quatro scripts de propósito. Cada um precisa ser um
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
    """Carrega o STL como malha única, com vértices duplicados fundidos.

    STL é uma 'sopa de triângulos': cada face repete seus 3 vértices e nada é
    compartilhado. Sem o process=True a malha nunca é estanque — todo par de
    faces vizinhas fica com vértices distintos — e o teste de estanqueidade
    abaixo reprovaria qualquer arquivo.
    """
    malha = trimesh.load_mesh(str(caminho), file_type="stl", process=True)
    if isinstance(malha, trimesh.Scene):
        malha = trimesh.util.concatenate(tuple(malha.geometry.values()))
    if not isinstance(malha, trimesh.Trimesh) or len(malha.faces) == 0:
        raise ValueError("arquivo não contém uma malha triangular utilizável")
    return malha


def conta_faces_do_arquivo(caminho):
    """Triângulos como gravados, sem fusão — é o número que o slicer vê."""
    bruta = trimesh.load_mesh(str(caminho), file_type="stl", process=False)
    if isinstance(bruta, trimesh.Scene):
        bruta = trimesh.util.concatenate(tuple(bruta.geometry.values()))
    return len(bruta.faces)


# ------------------------------------------------- distância exata à superfície

def distancia_ponto_triangulo(pontos, triangulos):
    """Distância exata de cada ponto ao seu triângulo (pareados 1 a 1).

    Projeta no plano e, quando a projeção cai fora, recua para a aresta ou o
    vértice mais próximo — as sete regiões de Voronoi do triângulo. Tudo
    vetorizado: são milhões de pares por chamada e um laço em Python levaria
    minutos onde isto leva segundos.
    """
    a, b, c = triangulos[:, 0], triangulos[:, 1], triangulos[:, 2]
    ab, ac = b - a, c - a

    ap = pontos - a
    d1 = (ab * ap).sum(1)
    d2 = (ac * ap).sum(1)
    bp = pontos - b
    d3 = (ab * bp).sum(1)
    d4 = (ac * bp).sum(1)
    cp = pontos - c
    d5 = (ab * cp).sum(1)
    d6 = (ac * cp).sum(1)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    # caso geral: a projeção cai dentro do triângulo
    soma = va + vb + vc
    inv = 1.0 / np.where(soma == 0.0, 1.0, soma)
    proximo = a + (vb * inv)[:, None] * ab + (vc * inv)[:, None] * ac

    def recua(mascara, ponto):
        if mascara.any():
            proximo[mascara] = ponto[mascara]

    # vértices
    perto_a = (d1 <= 0) & (d2 <= 0)
    perto_b = (d3 >= 0) & (d4 <= d3)
    perto_c = (d6 >= 0) & (d5 <= d6)
    # arestas: só valem onde nenhum vértice já capturou o ponto
    livre = ~(perto_a | perto_b | perto_c)

    def na_aresta(mascara, origem, direcao, num, den):
        if not mascara.any():
            return
        t = num[mascara] / np.where(den[mascara] == 0.0, 1.0, den[mascara])
        proximo[mascara] = origem[mascara] + t[:, None] * direcao[mascara]

    na_aresta(livre & (vc <= 0) & (d1 >= 0) & (d3 <= 0), a, ab, d1, d1 - d3)
    na_aresta(livre & (vb <= 0) & (d2 >= 0) & (d6 <= 0), a, ac, d2, d2 - d6)
    na_aresta(livre & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
              b, c - b, d4 - d3, (d4 - d3) + (d5 - d6))
    recua(perto_a, a)
    recua(perto_b, b)
    recua(perto_c, c)

    return np.linalg.norm(pontos - proximo, axis=1)


def indexa_superficie(malha, aresta_maxima):
    """Prepara a consulta de distância: triângulos + KDTree dos baricentros.

    O baricentro só localiza o triângulo; a distância em si é exata. Mas um
    triângulo muito maior que a grade pode ter o baricentro longe do ponto e
    escapar dos k vizinhos, e aí a distância sairia maior do que é — parede
    grossa demais num trecho, sem nenhum sinal de erro. Subdividir as faces
    grandes (subdivisão é coplanar, não muda a superfície) mantém a nuvem de
    baricentros densa em relação à grade e fecha esse buraco.
    """
    v, f = trimesh.remesh.subdivide_to_size(
        malha.vertices, malha.faces, max_edge=aresta_maxima)
    triangulos = v[f]
    return triangulos, cKDTree(triangulos.mean(axis=1))


def distancia_a_superficie(pontos, indice, vizinhos=16, bloco=25_000):
    """Menor distância de cada ponto à superfície, em blocos para caber na RAM.

    O bloco é pequeno de propósito: cada ponto arrasta 16 triângulos candidatos,
    e a conta ponto-triângulo cria mais de dez arrays intermediários desse
    tamanho. Medido, blocos de 200 mil pontos custavam meio giga de pico; 25 mil
    custam um décimo disso sem diferença de tempo perceptível.
    """
    triangulos, arvore = indice
    k = int(min(vizinhos, len(triangulos)))
    saida = np.empty(len(pontos), dtype=np.float64)
    for i in range(0, len(pontos), bloco):
        p = pontos[i:i + bloco]
        _, vizinhos_idx = arvore.query(p, k=k, workers=-1)
        vizinhos_idx = vizinhos_idx.reshape(len(p), k)
        d = distancia_ponto_triangulo(
            np.repeat(p, k, axis=0), triangulos[vizinhos_idx.ravel()])
        saida[i:i + bloco] = d.reshape(len(p), k).min(axis=1)
    return saida


# ------------------------------------------------------------ grade e campo

# A voxelização marca todo voxel que a superfície encosta, então a fronteira da
# ocupação fica cerca de um voxel para fora da superfície real, e a EDT mede a
# partir de centros de voxel, não da superfície. O campo bruto sai com esse
# viés (~0,75 voxel, medido em esfera, cubo e cilindro). Ele NÃO entra na
# espessura final: serve só para centrar a faixa que será recalculada com
# distância exata. Por isso um valor aproximado basta aqui.
VIES_EDT = 0.75

# Meia-largura da faixa recalculada, em voxels. Precisa ser maior que VIES_EDT
# para que os valores nas duas bordas da faixa fiquem do mesmo lado do nível
# antes e depois do refino — senão a emenda entre campo bruto e campo exato
# criaria uma isosuperfície fantasma ali.
FAIXA_VOXELS = 2.0


def amostra_triangulos(a, ab, ac, nu, nv):
    """Pontos numa malha paramétrica sobre cada triângulo, passo ≤ o pedido.

    Cada triângulo ganha a sua própria malha (nu+1) x (nv+1) em coordenadas
    baricêntricas, então um triângulo comprido e estreito é amostrado denso ao
    longo do comprimento e ralo na largura — que é o certo. Subdividir em 4
    como o trimesh faz é isotrópico: para acompanhar a aresta longa ele divide
    também a curta, e o custo cresce com a razão de aspecto, não com a área.
    Num cilindro de 512 faces isso era a diferença entre 2,3 milhões de pontos
    e 134 milhões.
    """
    total = (nu + 1) * (nv + 1)
    face = np.repeat(np.arange(len(a)), total)
    # índice local dentro da malha de cada triângulo, sem laço em Python
    local = np.arange(total.sum()) - np.repeat(
        np.concatenate([[0], np.cumsum(total)[:-1]]), total)
    # local percorre a malha (nu+1) x (nv+1) em ordem de linha: a divisão dá o
    # passo ao longo de ab, o resto dá o passo ao longo de ac
    largura = np.repeat(nv + 1, total)
    u = (local // largura) / np.repeat(nu, total)
    v = (local % largura) / np.repeat(nv, total)

    # Quem cai fora do triângulo é rebatido sobre a hipotenusa em vez de
    # descartado: descartar deixaria uma faixa sem amostra justamente ao longo
    # da aresta, que é onde a fresta na casca apareceria.
    fora = (u + v) > 1.0
    escala = np.where(fora, 1.0 / np.maximum(u + v, 1e-300), 1.0)
    u, v = u * escala, v * escala

    return a[face] + u[:, None] * ab[face] + v[:, None] * ac[face]


def ocupacao_solida(malha, pitch, orcamento=4_000_000):
    """Grade booleana do sólido: casca voxelizada + miolo preenchido.

    Devolve (ocupacao, origem), com origem = centro do voxel [0,0,0] no mundo.

    Não uso o malha.voxelized() do trimesh: ele subdivide a malha INTEIRA de uma
    vez antes de marcar voxel nenhum, e numa esfera de 20 mil faces a 256 voxels
    isso sozinho custou 1,9 GB — numa peça de 2 milhões de triângulos não
    caberia em máquina nenhuma. Aqui a superfície é amostrada em lotes, com o
    pico limitado pelo lote.
    """
    # A margem vazia serve a dois propósitos: garante que o preenchimento por
    # inundação tenha um "lado de fora" conectado em toda a borda da grade, e
    # evita que o marching cubes encoste no limite do array.
    margem = 3
    inferior, superior = malha.bounds
    origem = inferior - margem * pitch
    forma = tuple(int(np.ceil((superior[i] - inferior[i]) / pitch)) + 2 * margem + 1
                  for i in range(3))
    ocupacao = np.zeros(forma, dtype=bool)

    # Passo de amostragem abaixo de meio voxel: assim dois pontos vizinhos caem
    # no mesmo voxel ou em voxels adjacentes, e a casca não deixa fresta por
    # onde a inundação do miolo possa vazar.
    passo = 0.4 * pitch
    v = malha.vertices
    f = malha.faces
    a = v[f[:, 0]]
    ab = v[f[:, 1]] - a
    ac = v[f[:, 2]] - a
    nu = np.ceil(np.linalg.norm(ab, axis=1) / passo).astype(np.int64)
    nv = np.ceil(np.linalg.norm(ac, axis=1) / passo).astype(np.int64)
    np.maximum(nu, 1, out=nu)
    np.maximum(nv, 1, out=nv)

    # Lotear pelo número de pontos, não pelo de faces: uma face grande sozinha
    # gera mais pontos que milhares de faces pequenas.
    acumulado = np.cumsum((nu + 1) * (nv + 1))
    inicio = 0
    while inicio < len(f):
        ja_feito = acumulado[inicio - 1] if inicio else 0
        fim = max(inicio + 1,
                  int(np.searchsorted(acumulado, ja_feito + orcamento, side="right")))
        corte = slice(inicio, fim)
        pontos = amostra_triangulos(a[corte], ab[corte], ac[corte],
                                    nu[corte], nv[corte])
        idx = np.rint((pontos - origem) / pitch).astype(np.int64)
        ocupacao[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        inicio = fim

    # binary_fill_holes marca tudo que não alcança a borda da grade. Numa casca
    # fechada isso é exatamente o interior; numa casca furada o "interior" vaza
    # para fora e nada é preenchido — daí a conferência de volume em esvazia().
    return ndimage.binary_fill_holes(ocupacao), origem


def campo_de_profundidade(ocupacao, origem, pitch, espessura, indice, log):
    """Profundidade de cada voxel dentro do sólido, exata perto do nível alvo.

    A EDT dá o campo inteiro barato, mas enviesado. Recalcular tudo com
    distância exata seria caro; recalcular só a casca de voxels que o marching
    cubes vai atravessar custa pouco e é o que define a espessura da parede.
    """
    profundidade = ndimage.distance_transform_edt(
        ocupacao, sampling=pitch).astype(np.float32)

    # maior esfera que cabe na peça: acima disso não sobra cavidade nenhuma
    profundidade_maxima = float(profundidade.max()) - VIES_EDT * pitch

    faixa = np.abs(profundidade - (espessura + VIES_EDT * pitch)) <= FAIXA_VOXELS * pitch
    n_faixa = int(faixa.sum())
    if n_faixa:
        pontos = origem + np.argwhere(faixa) * pitch
        profundidade[faixa] = distancia_a_superficie(pontos, indice)
    log(f"      faixa recalculada com distância exata: {n_faixa:,} voxels")
    return profundidade, profundidade_maxima


def achata_para_float32(malha):
    """Aplica a precisão do STL à malha em memória e desfaz o que ela colapsar.

    O marching cubes coloca vértices arbitrariamente perto quando o nível passa
    rente a um vértice da grade — distâncias que o float32 do STL não
    representa. Gravar assim funde os dois vértices DENTRO do arquivo, e o
    triângulo entre eles vira face de área zero com aresta não-manifold: uma
    esfera de parede 2 mm saía com 24 delas e o analisa-stl.py reprovava o
    arquivo, embora a malha em memória parecesse perfeita.

    Arredondar aqui e colapsar as faces resultantes deixa a malha em memória
    idêntica à que vai para o disco. Colapsar aresta em malha fechada mantém a
    malha fechada — as duas faces que somem são exatamente as degeneradas.
    """
    v = np.asarray(malha.vertices, dtype=np.float32).astype(np.float64)
    unicos, inverso = np.unique(v, axis=0, return_inverse=True)
    faces = inverso.reshape(-1)[malha.faces]
    validas = ((faces[:, 0] != faces[:, 1])
               & (faces[:, 1] != faces[:, 2])
               & (faces[:, 0] != faces[:, 2]))
    nova = trimesh.Trimesh(vertices=unicos, faces=faces[validas], process=False)
    nova.remove_unreferenced_vertices()
    return nova


def superficie_interna(profundidade, origem, pitch, espessura):
    """Isosuperfície da profundidade = espessura, com as normais para dentro.

    allow_degenerate=False porque o marching cubes gera triângulos de área zero
    onde o nível passa exatamente por um vértice da grade, e face degenerada é
    justamente o que o analisa-stl.py aponta como defeito.
    """
    v, f, _, _ = measure.marching_cubes(
        profundidade, level=espessura, spacing=(pitch, pitch, pitch),
        allow_degenerate=False)
    interna = trimesh.Trimesh(vertices=v + origem, faces=f, process=True)
    if len(interna.faces) == 0:
        return interna
    interna = achata_para_float32(interna)

    # A superfície de uma cavidade tem as normais apontando para o vazio, não
    # para fora dela: o material está do lado de fora. Isso é o mesmo que dizer
    # que o volume com sinal da cavidade é NEGATIVO, e é o que faz o volume da
    # casca sair como externa menos cavidade. Não confio na convenção de
    # orientação do marching cubes — o sinal do volume responde de fato.
    trimesh.repair.fix_winding(interna)
    if interna.is_watertight and interna.volume > 0:
        interna.invert()
    return interna


def mede_espessura(interna, indice, amostra):
    """Mede a parede de verdade: distância dos vértices internos à casca externa.

    Conferir o parâmetro pedido não prova nada — o que importa é onde a
    superfície interna foi de fato parar. A medida usa a mesma distância exata
    ponto-triângulo, mas contra a malha externa original.
    """
    v = np.asarray(interna.vertices)
    if len(v) > amostra:
        # amostragem determinística: mesma entrada, mesmo relatório
        v = v[np.linspace(0, len(v) - 1, amostra, dtype=np.int64)]
    d = distancia_a_superficie(v, indice)
    return float(d.min()), float(d.mean()), float(d.max())


# ------------------------------------------------------------------ pipeline

def esvazia(entrada, args, log):
    entrada = Path(entrada)
    saida = Path(args.saida) if args.saida else entrada.with_name(
        f"{entrada.stem}-oco.stl")
    if saida.resolve() == entrada.resolve() and not args.sobrescrever:
        raise ValueError("saída igual à entrada; use --sobrescrever se for essa a intenção")
    if saida.exists() and not args.sobrescrever:
        raise ValueError(f"{saida} já existe; use --sobrescrever")

    faces_arquivo = conta_faces_do_arquivo(entrada)
    malha = carrega(entrada)
    dim = malha.extents
    escala = float(np.max(dim))

    log(f"{entrada.name}: {faces_arquivo:,} triângulos, {humaniza(entrada.stat().st_size)}")
    log(f"  dimensões : {dim[0]:.3f} x {dim[1]:.3f} x {dim[2]:.3f}")

    if not malha.is_watertight:
        raise ValueError(
            "a malha não é estanque — esvaziar uma malha aberta produz lixo, "
            "porque não há 'dentro' definido.\n"
            "        Rode ./analisa-stl.py para ver o defeito e ./repara-stl.py "
            "para consertar antes.")

    volume_solido = float(malha.volume)
    if volume_solido <= 0:
        raise ValueError("volume não positivo: as normais parecem invertidas; "
                         "rode ./repara-stl.py antes")

    pitch = escala / args.resolucao
    if args.espessura <= 2 * pitch:
        raise ValueError(
            f"espessura {args.espessura:g} é fina demais para a grade: o voxel "
            f"mede {pitch:.4f} e a parede precisa de pelo menos 2 voxels.\n"
            f"        Aumente --resolucao para {int(np.ceil(2.2 * escala / args.espessura))} "
            f"ou mais.")

    log(f"  espessura : {args.espessura:g}   voxel: {pitch:.4f} "
        f"({args.espessura / pitch:.1f} voxels por parede)")

    log("  [1] voxelizando e preenchendo o interior...")
    ocupacao, origem = ocupacao_solida(malha, pitch)
    n_voxels = int(ocupacao.sum())
    log(f"      grade {'x'.join(str(n) for n in ocupacao.shape)} "
        f"({ocupacao.size / 1e6:.1f} M voxels), {n_voxels:,} ocupados")

    # Rede de segurança para o caso de a malha passar no teste de estanqueidade
    # mas ter uma fenda menor que o voxel: a inundação escapa, o interior não é
    # preenchido e o resultado seria uma casca de espessura aleatória.
    volume_grade = n_voxels * pitch ** 3
    if volume_grade < 0.5 * volume_solido:
        raise ValueError(
            f"o preenchimento do interior vazou: a grade acusa volume "
            f"{volume_grade:.2f} contra {volume_solido:.2f} da malha.\n"
            f"        Costuma ser fenda menor que o voxel. Aumente --resolucao "
            f"ou rode ./repara-stl.py.")

    log("  [2] campo de distância...")
    indice = indexa_superficie(malha, aresta_maxima=4.0 * pitch)
    profundidade, profundidade_maxima = campo_de_profundidade(
        ocupacao, origem, pitch, args.espessura, indice, log)
    del ocupacao

    if profundidade_maxima <= args.espessura:
        log("")
        log(f"  ESPESSURA GRANDE DEMAIS: com {args.espessura:g} de parede não sobra "
            f"cavidade nenhuma.")
        log(f"  A maior esfera que cabe nesta peça tem raio ~{profundidade_maxima:.3f}, "
            f"que é o teto da espessura.")
        log(f"  Use algo abaixo disso (com folga: perto do teto a cavidade fica "
            f"minúscula) ou deixe a peça sólida.")
        log("  Não gravei nada.")
        _RESUMO.append({"arquivo": entrada.name, "gravou": False,
                        "motivo": "sem cavidade",
                        "espessura": args.espessura,
                        "espessura_maxima": round(profundidade_maxima, 4)})
        return 1

    log("  [3] extraindo a superfície interna (marching cubes)...")
    interna = superficie_interna(profundidade, origem, pitch, args.espessura)
    if len(interna.faces) == 0:
        log("")
        log(f"  ESPESSURA GRANDE DEMAIS: a cavidade não sobreviveu à grade "
            f"(nenhuma face extraída).")
        log(f"  Teto estimado da espessura: ~{profundidade_maxima:.3f}. Não gravei nada.")
        _RESUMO.append({"arquivo": entrada.name, "gravou": False,
                        "motivo": "cavidade vazia na grade",
                        "espessura": args.espessura,
                        "espessura_maxima": round(profundidade_maxima, 4)})
        return 1

    # Cada bolsão isolado é um vazio que não se comunica com os outros: importa
    # porque a peça oca vira várias câmaras estanques independentes.
    bolsoes = int(ndimage.label(profundidade > args.espessura)[1])
    del profundidade

    volume_cavidade = float(interna.volume) if interna.is_watertight else 0.0
    volume_cavidade = abs(volume_cavidade)
    log(f"      superfície interna: {len(interna.faces):,} triângulos, "
        f"{bolsoes} cavidade(s), volume {volume_cavidade:.3f}")

    log("  [4] medindo a parede que saiu (não o parâmetro pedido)...")
    e_min, e_med, e_max = mede_espessura(interna, indice, args.amostra)
    log(f"      espessura medida: mín {e_min:.4f} · média {e_med:.4f} · máx {e_max:.4f}"
        f"   (pedido {args.espessura:g}, erro médio {e_med - args.espessura:+.4f})")

    # A concatenação é o passo barato e exato: a superfície externa sai do
    # arquivo original sem um triângulo alterado, e a interna entra como segundo
    # corpo fechado, com as normais já invertidas. Reconstruir as duas por
    # marching cubes degradaria a externa sem nenhum ganho.
    casca = trimesh.util.concatenate([malha, interna])

    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_bytes(
        trimesh.exchange.stl.export_stl_ascii(casca).encode()
        if args.ascii else trimesh.exchange.stl.export_stl(casca))

    # STL guarda coordenadas em float32 e não compartilha vértices: gravar e
    # reler funde vértices que estavam separados em memória e pode mudar a
    # topologia. O estado que vale é o do arquivo, não o da malha em memória.
    conferida = trimesh.load_mesh(str(saida), file_type="stl", process=True)
    estanque = bool(conferida.is_watertight)
    volume_material = float(conferida.volume) if estanque else float(casca.volume)
    corpos = conferida.body_count
    economia = 100.0 * (1.0 - volume_material / volume_solido)

    log("")
    log(f"  -> {saida} ({humaniza(saida.stat().st_size)})")
    log(f"     triângulos      : {faces_arquivo:,} -> {len(conferida.faces):,} "
        f"({len(conferida.faces) / faces_arquivo:.1f}x)")
    log(f"     volume sólido   : {volume_solido:.3f}")
    log(f"     volume da casca : {volume_material:.3f}  (material de fato impresso)")
    log(f"     economia        : {economia:.1f}% de material")
    log(f"     corpos          : {corpos}   (2 = externa + cavidade, o esperado)")
    log(f"     estanque        : {'sim' if estanque else 'NÃO'}"
        + ("" if estanque else "   <-- rode ./analisa-stl.py para ver o que sobrou"))

    if len(conferida.faces) > 500_000:
        log("")
        log(f"     O marching cubes é denso por natureza e a superfície interna "
            f"saiu com {len(interna.faces):,} triângulos.")
        log(f"     Para aliviar: ./simplifica-stl.py -r 0.2 {saida.name}")

    if estanque:
        log("")
        log("  " + "=" * 68)
        log("  ATENÇÃO: esta casca é FECHADA e não tem furo de dreno.")
        log("  Em impressão de resina a cavidade alaga, a resina presa não escoa e a")
        log("  peça pode estourar na cura — ou colar na plataforma por sucção.")
        log(f"  {'Fure' if bolsoes == 1 else f'São {bolsoes} bolsões isolados: fure CADA UM'} "
            f"antes de imprimir, no CAD ou no slicer.")
        log("  Esta ferramenta não posiciona o furo de propósito: o lugar certo depende")
        log("  da orientação de impressão e de onde a marca é aceitável — quem conhece")
        log("  a peça decide melhor do que qualquer regra automática.")
        log("  " + "=" * 68)

    _RESUMO.append({
        "arquivo": entrada.name, "saida": saida.name, "gravou": True,
        "espessura": args.espessura, "resolucao": args.resolucao,
        "pitch": round(pitch, 6),
        "faces_antes": faces_arquivo, "faces_depois": len(conferida.faces),
        "volume_solido": round(volume_solido, 4),
        "volume_material": round(volume_material, 4),
        "economia_pct": round(economia, 2),
        "espessura_medida": {"min": round(e_min, 4), "media": round(e_med, 4),
                             "max": round(e_max, 4)},
        "bolsoes": bolsoes, "corpos": corpos, "estanque": estanque,
    })
    return 0 if estanque else 2


def main():
    p = argparse.ArgumentParser(
        prog="esvazia-stl",
        description="Transforma um STL sólido em casco oco de parede uniforme.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""exemplos:
  esvazia-stl.py -e 2 peca.stl                  # parede de 2 mm
  esvazia-stl.py -e 1.5 -r 384 peca.stl         # grade mais fina, mais detalhe
  esvazia-stl.py -e 3 -o oca.stl --sobrescrever peca.stl

como funciona:
  Voxeliza o sólido, mede a profundidade de cada voxel com um campo de
  distância e extrai por marching cubes a isosuperfície a `espessura` de
  profundidade. Essa superfície entra na saída com as normais invertidas,
  junto da externa ORIGINAL, que não é tocada.

  A profundidade é recalculada com distância exata ponto-triângulo numa faixa
  fina em volta do nível alvo. É por isso que a espessura da parede fica certa
  mesmo em grade grossa: --resolucao governa o detalhe da cavidade, não a
  precisão da parede.

--resolucao: voxels ao longo da MAIOR dimensão da peça (padrão: 256).
  Fidelidade contra memória — a grade é cúbica, então dobrar a resolução
  multiplica a memória por 8:
      128 -> ~0,03 GB      256 -> ~0,2 GB
      384 -> ~0,8 GB       512 -> ~1,8 GB
  Detalhe menor que um voxel não aparece na cavidade (a parede engrossa ali,
  o que não estraga a peça). Comece em 256 e só suba se a cavidade perder
  detalhe que importa.

saída densa: marching cubes gera muito triângulo. Se a contagem incomodar,
  ./simplifica-stl.py -r 0.2 peca-oca.stl  reduz depois, sem mexer na forma.

SEM FURO DE DRENO, de propósito: peça oca fechada alaga na impressão de
  resina e pode estourar na cura. O furo é decisão de quem conhece a peça e a
  orientação de impressão — automatizar isso erra o lugar e estraga a
  impressão ou enfraquece a parede. O script avisa; o furo é seu.

código de saída: 0 = gravou casca estanque, 1 = espessura não deixa cavidade,
  2 = erro (inclusive casca gravada que saiu não estanque)
""",
    )
    p.add_argument("arquivo", type=Path, help="arquivo STL de entrada")
    p.add_argument("-e", "--espessura", type=float, required=True, metavar="MM",
                   help="espessura da parede, nas unidades do modelo (obrigatório)")
    p.add_argument("-r", "--resolucao", type=int, default=256, metavar="N",
                   help="voxels na maior dimensão (padrão: 256)")
    p.add_argument("-o", "--saida", type=Path, help="padrão: <nome>-oco.stl")
    p.add_argument("--amostra", type=int, default=20000, metavar="N",
                   help="vértices amostrados ao medir a parede (padrão: 20000)")
    p.add_argument("--ascii", action="store_true", help="grava STL ASCII (padrão: binário)")
    p.add_argument("--sobrescrever", action="store_true",
                   help="sobrescreve arquivos existentes")
    p.add_argument("-q", "--silencioso", action="store_true")
    args = p.parse_args()

    if args.espessura <= 0:
        p.error("--espessura precisa ser maior que zero")
    if args.resolucao < 16:
        p.error("--resolucao precisa ser pelo menos 16")
    if args.resolucao > 1024:
        p.error("--resolucao acima de 1024 estoura a memória de qualquer máquina "
                "razoável (grade de mais de 1 bilhão de voxels)")
    if args.amostra < 100:
        p.error("--amostra precisa ser pelo menos 100")

    log = (lambda m: None) if args.silencioso else (lambda m: print(m, flush=True))
    try:
        if not args.arquivo.is_file():
            raise ValueError("arquivo não encontrado")
        return esvazia(args.arquivo, args, log)
    except Exception as e:
        print(f"erro em {args.arquivo}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(_encerra("esvazia-stl", main()))
