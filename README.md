# stl-tools

Ferramentas de linha de comando para inspecionar, reparar e simplificar malhas
STL. Cada script é um arquivo único e independente, sem instalação: as
dependências são declaradas inline (PEP 723) e o `uv` monta o ambiente sozinho
na primeira execução.

```bash
./analisa-stl.py peca.stl        # é estanque? se não, por quê?
./repara-stl.py peca.stl         # conserta
./simplifica-stl.py -r 0.1 peca.stl   # reduz para 10% dos triângulos
```

## Por que não open3d

A sugestão original era usar `open3d`, mas ele **não instala no Python 3.14** do
Ubuntu 26.04: a versão 0.19.0 só publica wheels até `cp312`, e compilar do
fonte exigiria CMake mais toda a stack de visualização.

O substituto é `fast-simplification`, que implementa a **mesma decimação
quádrica de Garland–Heckbert** do `open3d.simplify_quadric_decimation` — mesmo
algoritmo, não um sucedâneo pior — com wheel para `cp314` e ~1 MB contra os
~450 MB do open3d. O I/O e a topologia ficam com `trimesh`.

## Requisitos

Uso local: só o [`uv`](https://astral.sh/uv). Nada de `pip install`, nada de
venv manual.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Os scripts trazem `#!/usr/bin/env -S uv run --script`, então basta executá-los.

### Dentro de container (ou em qualquer ambiente já provisionado)

O bloco PEP 723 no topo de cada script é um comentário Python: sob um
interpretador comum ele é simplesmente ignorado. Então onde as dependências já
vêm instaladas na imagem, **nada precisa mudar nos scripts** — só a forma de
invocar, porque a shebang deixa de valer:

```bash
pip install -r requirements.txt
python analisa-stl.py peca.stl
```

Vale manter o bloco inline mesmo assim: ele continua servindo o uso local e
documenta, junto ao código, exatamente o que a imagem precisa instalar.

As versões no bloco PEP 723 e no `requirements.txt` são **fixas de propósito**.
Uma imagem sem rede, com cache uv pré-populado, não consegue resolver versão —
precisaria do índice. Fixar também garante que a máquina local e o servidor
usem o mesmo numpy/scipy: divergência numérica entre as pontas é cara de achar
justamente porque não parece bug.

---

## Os scripts

| Script | Para quê | Código de saída |
|---|---|---|
| `analisa-stl.py` | Diagnostica: é estanque? o que está quebrado? | 0 ok · 1 não estanque · 2 erro |
| `repara-stl.py` | Solda, limpa topologia, fecha buracos | 0 fechou · 1 melhorou · 2 erro |
| `simplifica-stl.py` | Reduz a contagem de triângulos | 0 ok · 1 erro em algum arquivo |
| `roda-remoto.sh` | Executa qualquer um deles em outra máquina | o do script remoto |

### `analisa-stl.py` — diagnóstico

Responde se a malha é estanque e, quando não é, explica a causa. Reporta
arestas abertas e quantos buracos elas formam (com o perímetro de cada um),
arestas não-manifold, faces degeneradas e duplicadas, vértices órfãos, corpos
separados, coerência das normais e a característica de Euler.

O diferencial é **separar buraco real de fresta numérica**. STL guarda
coordenadas em float32; vértices que deveriam coincidir muitas vezes diferem
nos últimos bits, e a malha *parece* cheia de furos embora a geometria esteja
inteira. O script solda os vértices por proximidade em tolerâncias crescentes e
diz se existe alguma que fecha a peça, e qual, em percentual da maior dimensão.

```
a malha FECHA soldando vértices a 2e-05 de distância (0.00100% da maior dimensão).
=> são frestas numéricas do float32, não falhas de modelagem.
```

Opções: `--rapido` (pula o teste de tolerância, bem mais rápido em malha
grande), `--max-buracos N`, `-q` (só o veredito, bom para lote).

### `repara-stl.py` — conserto

Cinco etapas, nesta ordem — inverter a ordem cria defeitos novos:

1. **Soldagem** por proximidade (KDTree), fechando as frestas de float32.
2. **Limpeza** de faces degeneradas (área zero) e duplicadas.
3. **Não-manifold**: desfaz arestas com 3+ faces.
4. **Buracos**: tampa cada laço de borda com um leque a partir do centroide.
5. **Normais**: reorienta o winding e corrige inversão.

As etapas 2 a 4 brigam entre si — preencher um buraco pode criar face duplicada,
remover uma face não-manifold pode abrir um buraco — então rodam num laço até o
estado parar de mudar (`--passes`, padrão 6).

Por segurança o script **não grava se não conseguiu fechar** a malha, a menos
que você passe `--forcar`. Use `--sem-preencher` quando não quiser que ele
invente geometria, e `--max-buraco FRAC` para não tampar vãos grandes demais
(padrão: recusa laços com perímetro acima de 50% da maior dimensão).

### `simplifica-stl.py` — decimação

Reduz triângulos por decimação quádrica, preservando a forma.

Alvo por fração (`-r 0.1` mantém 10%) ou por contagem (`-n 20000`).
Aceita vários arquivos com `-o pasta/` para lote. `-a` troca velocidade por
fidelidade (10 = rápido e grosseiro, 0 = lento e fiel). `--lossless` remove só
redundância sem alterar a forma. Ainda: `--preserva-borda`, `--corrige-normais`,
`--ascii`, `--sobrescrever`.

Medido numa esfera de 82k triângulos: reduzindo a **5%**, a malha continuou
estanque, com **0,11% de erro de volume** e desvio máximo de superfície de 1,4%
do raio. Arquivo 95% menor.

### `roda-remoto.sh` — execução remota

Uma malha de 2 milhões de triângulos não cabe confortavelmente num desktop
modesto. Este wrapper manda o script e os arquivos por `rsync` para uma máquina
maior, executa lá e traz o resultado de volta.

```bash
REMOTO_HOST=servidor REMOTO_USER=usuario REMOTO_KEY=~/.ssh/chave \
  ./roda-remoto.sh repara-stl.py peca.stl
```

É **síncrono e sem estado**: nada de daemon, fila ou spool. O script é reenviado
a cada chamada, então nunca há versão velha rodando no servidor. Variáveis:
`REMOTO_HOST` (obrigatória), `REMOTO_USER`, `REMOTO_KEY`, `REMOTO_JOBS`.
Opções: `-s/--servidor`, `-d/--destino`, `--manter`. Ele detecta sozinho como
rodar do outro lado — `uv` se houver, senão `python3` — e `-e/--exec` força uma
invocação específica, útil quando o script roda dentro de um container:

```bash
./roda-remoto.sh -e "docker exec -i malhas python3" repara-stl.py peca.stl
```

Quando o script roda dentro de um container, o diretório do job tem um caminho
no host e outro visto de dentro. Informe a base interna com `--dir-exec` e use
`{dir}` no `--exec` — o `rsync` continua usando o caminho do host:

```bash
REMOTO_JOBS=/srv/ferramenta/work ./roda-remoto.sh \
  -e "docker exec -w {dir} -i ferramenta" --dir-exec /work \
  repara-stl.py peca.stl
```

Nesse arranjo quem chama precisa estar no grupo `docker` do host, senão o
`docker exec` pede sudo e o job trava.

Descartei de propósito duas alternativas: montar o **armazenamento remoto no
servidor** via FUSE (o arquivo cruza a rede de qualquer jeito, leitura aleatória
por FUSE é lenta e gravar o resultado de volta por ela é arriscado) e uma **fila
de jobs com daemon** (só compensa com submissão desacompanhada ou concorrência;
para uso interativo é estado a mais para emperrar).

> **GPU não acelera nada disso.** `trimesh`, `scipy` e `fast-simplification` são
> todos CPU. O ganho de uma máquina maior é RAM e núcleos.

---

## Tempo, memória e log de invocações

Os três scripts reportam ao final, em `stderr`, quanto levaram e o pico de
memória do processo:

```
[repara-stl] tempo 208.4s · pico de memória 2.386 MiB
```

O pico vem do `ru_maxrss` do próprio processo, que o kernel já mantém — não há
amostragem nem custo. O `roda-remoto.sh` reporta separadamente o tempo total de
parede, transferência incluída, que é sempre maior que o do script.

Cada invocação também vira uma linha JSON em
`~/.local/state/stl-tools/invocacoes.jsonl` (mude com `STL_TOOLS_LOG`), com os
argumentos, o resultado resumido por arquivo, o código de saída, o tempo e a
memória. Serve para comparar execuções e dimensionar máquina:

```bash
jq -r '[.quando, .ferramenta, .segundos, .pico_memoria_mib] | @tsv' \
  ~/.local/state/stl-tools/invocacoes.jsonl | column -t
```

O log é conveniência: se o caminho não for gravável, ele é silenciosamente
ignorado e o processamento segue.

## Fluxo típico

```bash
./analisa-stl.py peca.stl          # 1. entender o problema
./repara-stl.py peca.stl           # 2. consertar
./analisa-stl.py peca-reparado.stl # 3. conferir
./simplifica-stl.py -r 0.1 peca-reparado.stl   # 4. aliviar
```

**Simplifique depois de reparar, nunca antes** — decimar em cima de topologia
inválida espalha o defeito. Mas **confira de novo depois de simplificar**: a
decimação quádrica não garante saída manifold, e o estrago cresce com a
redução. Medido numa malha real de 1,9 milhão de triângulos, partindo de zero
arestas não-manifold:

| redução para | não-manifold criadas |
|---|---|
| 90% | 40 |
| 50% | 881 |
| 10% | 2.210 |

Baixar a agressividade não resolve: com `-a 0` o algoritmo sequer alcança o
alvo (parou em 97,8% quando pedimos 10%). Reparar de novo depois zera as
não-manifold, mas em troca de muitas arestas abertas — numa redução agressiva
o dano não é recuperável. Se a peça precisa de topologia limpa, prefira uma
razão conservadora. O `simplifica-stl.py` avisa quando degrada.

## Armadilhas do formato STL

Coisas que custaram tempo e estão embutidas nos scripts:

- **STL é sopa de triângulos.** Cada face repete seus 3 vértices e nada é
  compartilhado. Sem fundir vértices primeiro, a decimação não acha aresta para
  colapsar e não simplifica quase nada.
- **`trimesh.merge_vertices()` funde por arredondamento de casas decimais**, não
  por distância: dois vértices a 2e-7 caem em baldes diferentes se estiverem na
  fronteira do arredondamento. Para saber a menor tolerância que fecha uma
  malha, é preciso soldar por proximidade real (KDTree + union-find); usar
  `digits_vertex` sugere tolerâncias grosseiras o bastante para deformar a peça.
- **`trimesh.repair.fill_holes()` só fecha furos de 3 ou 4 arestas.** Buracos
  maiores passam batido — daí o preenchimento próprio por leque.
- **Não escolha faces não-manifold pela área.** Uma aleta espúria pode ser
  enorme e uma face legítima pode ser minúscula. O critério que funciona é
  quantas das 3 arestas da face são compartilhadas por exatamente 2 faces:
  aleta tira 0, face bem costurada tira 2.
- **Cuidado ao trabalhar dentro de mount FUSE** (rclone, sshfs). Se o mount cai
  e volta, o shell que já estava dentro fica com o inode velho e qualquer
  comando morre com `Current directory does not exist` — parece erro do
  programa, mas não é. Conserto: `cd` de novo no caminho absoluto.

## Validação

Os scripts foram testados contra malhas quebradas de propósito — buraco real,
fresta de 1e-6, aleta não-manifold e as três combinadas. As quatro ficaram
estanques após o reparo, com **0,00% de erro de volume** contra a referência, e
a conferência foi feita pelo `analisa-stl.py`, que é código independente.

## Limitações conhecidas

- O preenchimento por leque tampa o buraco, mas não reconstrói curvatura: em
  vãos grandes a tampa fica visivelmente plana. Por isso o `--max-buraco`.
- Não há detecção de auto-interseção (faces que se atravessam). Uma malha pode
  passar como estanque e ainda assim ser problemática para o slicer.
- `roda-remoto.sh` é síncrono: fechou o terminal, perdeu o job. Se isso
  incomodar, o próximo passo é `nohup` no lado remoto com um id de job.
