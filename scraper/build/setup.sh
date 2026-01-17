#!/bin/bash -x
# =============================================================================
# Preamble
# =============================================================================
set -euo pipefail
MAIN_DIR="$(dirname "$BASH_SOURCE")"
cd "$MAIN_DIR"
SCRIPTDIR="$(pwd -P)"
cd - >/dev/null
cd "$MAIN_DIR/.."
ROOT_DIR="$(pwd -P)"
cd - >/dev/null

source "$ROOT_DIR"/env.sh

if [ "$BASE_IMAGE" == "" ]; then
  echo "BASE_IMAGE must be set in env.sh"
fi

chmod a+x "$SCRIPTDIR"/setup-inside.sh

rm "$ROOT_DIR"/Dockerfile || true

docker run --rm \
  -u 0:0 \
  -v "$ROOT_DIR:/app" \
  -w /app \
  --entrypoint bash \
  $BASE_IMAGE \
  /app/build/setup-inside.sh "$(id -u):$(id -g)"

export MYUID=$(id -u)
export MYGID=$(id -g)

cat "$SCRIPTDIR"/Dockerfile.in | envsubst > "$ROOT_DIR"/Dockerfile

