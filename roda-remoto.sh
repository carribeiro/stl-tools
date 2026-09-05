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
# Variáveis: REMOTO_HOST (obrigatória), REMOTO_USER, REMOTO_KEY, REMOTO_JOBS.

set -euo pipefail

HOST="${REMOTO_HOST:-}"
USUARIO="${REMOTO_USER:-}"
CHAVE="${REMOTO_KEY:-}"
RAIZ_JOBS="${REMOTO_JOBS:-~/jobs}"
DIR_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANTER=0
DESTINO=""

uso() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    cat <<EOF

opções:
  -s, --servidor HOST   host ssh (padrão: \$REMOTO_HOST)
  -d, --destino DIR     onde gravar os resultados (padrão: pasta do 1o arquivo)
      --manter          não apaga a pasta de trabalho remota
  -h, --ajuda
EOF
}

ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--servidor) HOST="$2"; shift 2 ;;
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
TRABALHO="$RAIZ_JOBS/$JOB"

echo ">> job $JOB em $ALVO"

ssh "${SSH_OPTS[@]}" "$ALVO" "command -v uv >/dev/null || {
    echo 'uv não está instalado no servidor. Instale com:' >&2
    echo '  curl -LsSf https://astral.sh/uv/install.sh | sh' >&2
    exit 127
}; mkdir -p $TRABALHO"

echo ">> enviando script e ${#ENTRADAS[@]} arquivo(s)"
RSYNC_SSH="ssh ${SSH_OPTS[*]}"
rsync -a --info=progress2 -e "$RSYNC_SSH" \
      "$SCRIPT" ${ENTRADAS[@]+"${ENTRADAS[@]}"} "$ALVO:$TRABALHO/"

# --- monta a linha de comando remota preservando espaços em nomes
COMANDO="cd $TRABALHO && uv run --quiet $(printf '%q' "$(basename "$SCRIPT")")"
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

echo ">> fim (código $CODIGO)"
exit $CODIGO
