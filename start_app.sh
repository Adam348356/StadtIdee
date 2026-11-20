#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"

if [ ! -x .venv/bin/python ] || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    if [ -e .venv ]; then
        RECOVERY_DIR=".venv.incomplete.$$"
        mv .venv "$RECOVERY_DIR"
        echo "Die unvollständige Umgebung wurde unter $RECOVERY_DIR aufbewahrt."
    fi
    if ! python3 -m venv .venv; then
        echo "Die Python-Umgebung konnte nicht erstellt werden." >&2
        echo "Installiere unter Ubuntu zuerst: sudo apt install python3-venv" >&2
        exit 1
    fi
fi

if ! .venv/bin/python -c 'import streamlit, streamlit_drawable_canvas, tensorflow, cv2, PIL' >/dev/null 2>&1; then
    .venv/bin/python -m pip install -r requirements.txt
fi

CUSTOM_PORT=0
for argument in "$@"; do
    case "$argument" in
        --server.port|--server.port=*) CUSTOM_PORT=1 ;;
    esac
done

if [ "$CUSTOM_PORT" -eq 0 ]; then
    APP_PORT=$(.venv/bin/python -c 'import socket; s=socket.socket(); port=8501
while port < 8600:
    try:
        s.bind(("127.0.0.1", port)); break
    except OSError:
        port += 1
else:
    raise SystemExit("No free Streamlit port found in 8501–8599")
s.close(); print(port)')
    exec .venv/bin/python -m streamlit run app.py --server.port "$APP_PORT" "$@"
fi

exec .venv/bin/python -m streamlit run app.py "$@"
