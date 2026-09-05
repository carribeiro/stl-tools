#!/usr/bin/env bash
# Roda um script pesado em outra máquina: manda script + arquivos, executa lá,
# traz o resultado de volta.
#
# Padrão deliberadamente sem estado: nada de daemon, fila ou spool. Cada chamada
# é síncrona e independente, e o script é reenviado toda vez, então nunca há
# versão velha rodando no servidor.
#
#   REMOTO_HOST=servidor roda-remoto.sh repara-stl.py peca.stl --sobrescrever
#   REMOTO_HOST=servidor roda-remoto.sh analisa-stl.py *.stl
#
# Variáveis: REMOTO_HOST (obrigatória), REMOTO_USER, REMOTO_KEY, REMOTO_JOBS,
# REMOTO_EXEC (como invocar o script lá; padrão: detecta uv, senão python3).
#
# Quando o script roda dentro de um container, o caminho do job visto de dentro
# difere do caminho no host. Use {dir} no --exec e informe a base interna com
# --dir-exec:
#
#   REMOTO_JOBS=/srv/ferramenta/work roda-remoto.sh \
#     -e "docker exec -w {dir} -i ferramenta" --dir-exec /work analisa-stl.py p.stl

set -euo pipefail

HOST="${REMOTO_HOST:-}"
USUARIO="${REMOTO_USER:-}"
CHAVE="${REMOTO_KEY:-}"
RAIZ_JOBS="${REMOTO_JOBS:-~/jobs}"
EXEC="${REMOTO_EXEC:-}"
DIR_EXEC="${REMOTO_DIR_EXEC:-}"
DIR_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANTER=0
DESTINO=""

uso() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    cat <<EOF

opções:
  -s, --servidor HOST   host ssh (padrão: \$REMOTO_HOST)
  -e, --exec CMD        como rodar o script no servidor (padrão: uv run, ou
                        python3 se não houver uv). Ex.: "python3", ou
                        "docker exec -i malhas python3"
      --dir-exec BASE   base do diretório de trabalho como vista pelo --exec
                        (ex.: /work num container); use {dir} no --exec
  -d, --destino DIR     onde gravar os resultados (padrão: pasta do 1o arquivo)
      --manter          não apaga a pasta de trabalho remota
  -h, --ajuda
EOF
}

ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--servidor) HOST="$2"; shift 2 ;;
        -e|--exec)     EXEC="$2"; shift 2 ;;
        --dir-exec)    DIR_EXEC="$2"; shift 2 ;;
        -d|--destino)  DESTINO="$2"; shift 2 ;;
        --manter)      MANTER=1; shift ;;
        -h|--ajuda)    uso; exit 0 ;;
        *)             ARGS+=("$1"); shift ;;
    esac
done

[[ ${#ARGS[@]} -ge 1 ]] || { uso; exit 2; }
[[ -n "$HOST" ]] || { echo "defina o host: -s HOST ou REMOTO_HOST=..." >&2; exit 2; }

# --- localiza o script: caminho dado ou vizinho deste wrapper
SCRIPT="${ARGS[0]}"
[[ -f "$SCRIPT" ]] || SCRIPT="$DIR_SCRIPTS/${ARGS[0]}"
[[ -f "$SCRIPT" ]] || { echo "script não encontrado: ${ARGS[0]}" >&2; exit 2; }

SSH_OPTS=(-o ConnectTimeout=10)
[[ -n "$CHAVE" ]] && SSH_OPTS+=(-o IdentitiesOnly=yes -i "$CHAVE")
ALVO="${USUARIO:+$USUARIO@}$HOST"

# --- separa arquivos locais (viajam) de flags (passam direto)
ENTRADAS=()
ENVIADOS=()   # só os nomes-base dos arquivos que subiram
REMOTOS=()    # linha de comando remota: nomes-base + flags
for a in "${ARGS[@]:1}"; do
    if [[ -f "$a" ]]; then
        ENTRADAS+=("$a")
        ENVIADOS+=("$(basename "$a")")
        REMOTOS+=("$(basename "$a")")
    else
        REMOTOS+=("$a")
    fi
done

[[ -z "$DESTINO" && ${#ENTRADAS[@]} -gt 0 ]] && DESTINO="$(dirname "${ENTRADAS[0]}")"
DESTINO="${DESTINO:-.}"

JOB="$(date +%Y%m%d-%H%M%S)-$$"
SEGUNDOS_INICIO=$SECONDS
TRABALHO="$RAIZ_JOBS/$JOB"

echo ">> job $JOB em $ALVO"

ssh "${SSH_OPTS[@]}" "$ALVO" "mkdir -p $TRABALHO"

# Como executar lá: uv resolve as dependências do bloco PEP 723 na hora; num
# container elas já vêm instaladas na imagem, então basta o interpretador.
if [[ -z "$EXEC" ]]; then
    EXEC=$(ssh "${SSH_OPTS[@]}" "$ALVO" \
        'if command -v uv >/dev/null; then echo "uv run --quiet";
         elif command -v python3 >/dev/null; then echo python3;
         else echo NENHUM; fi')
    if [[ "$EXEC" == "NENHUM" ]]; then
        echo "servidor não tem uv nem python3; use -e/--exec para dizer como rodar" >&2
        exit 127
    fi
fi
echo ">> executando com: $EXEC"

echo ">> enviando script e ${#ENTRADAS[@]} arquivo(s)"
RSYNC_SSH="ssh ${SSH_OPTS[*]}"
rsync -a --info=progress2 -e "$RSYNC_SSH" \
      "$SCRIPT" ${ENTRADAS[@]+"${ENTRADAS[@]}"} "$ALVO:$TRABALHO/"

# --- monta a linha de comando remota preservando espaços em nomes
# O cd vale para o host; dentro de um container o job está em outro caminho.
DIR_VISIVEL="$TRABALHO"
[[ -n "$DIR_EXEC" ]] && DIR_VISIVEL="$DIR_EXEC/$JOB"
# ./ explícito: sem isso, um --exec que resolve por PATH (docker exec num
# container que já traz os scripts assados) rodaria a cópia da imagem em vez da
# que acabamos de enviar — perdendo a garantia de estar rodando esta versão.
# chmod: o rsync -a preserva o modo do arquivo local, e invocar por ./ exige o
# bit de execução. Sem isso, um script sem +x na origem falha do outro lado com
# "permission denied", que não deixa nada óbvio sobre a causa.
BASE_SCRIPT="$(printf '%q' "$(basename "$SCRIPT")")"
COMANDO="cd $TRABALHO && chmod +x ./$BASE_SCRIPT && ${EXEC//\{dir\}/$DIR_VISIVEL} ./$BASE_SCRIPT"
for r in ${REMOTOS[@]+"${REMOTOS[@]}"}; do
    COMANDO+=" $(printf '%q' "$r")"
done

echo ">> executando"
set +e
ssh "${SSH_OPTS[@]}" "$ALVO" "$COMANDO"
CODIGO=$?
set -e

echo ">> trazendo resultados para $DESTINO/"
mkdir -p "$DESTINO"
# exclui apenas o que nós mesmos enviamos; flags não são nomes de arquivo
EXCLUI=(--exclude "$(basename "$SCRIPT")")
for r in ${ENVIADOS[@]+"${ENVIADOS[@]}"}; do EXCLUI+=(--exclude "$r"); done
rsync -a --info=progress2 -e "$RSYNC_SSH" \
      "${EXCLUI[@]}" "$ALVO:$TRABALHO/" "$DESTINO/"

if [[ $MANTER -eq 0 ]]; then
    ssh "${SSH_OPTS[@]}" "$ALVO" "rm -rf $TRABALHO"
else
    echo ">> pasta remota mantida: $TRABALHO"
fi

echo ">> fim (código $CODIGO) · $((SECONDS - SEGUNDOS_INICIO))s no total, transferência incluída"
exit $CODIGO
