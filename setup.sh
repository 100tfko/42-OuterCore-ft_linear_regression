#!/usr/bin/env bash

set -e

VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"

find_python() {
    local candidate major minor

    for candidate in python3.10 python3.11 python3.12 python3.13 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            major="$($candidate -c 'import sys; print(sys.version_info[0])' 2>/dev/null)"
            minor="$($candidate -c 'import sys; print(sys.version_info[1])' 2>/dev/null)"
            if [ "$major" -gt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; }; then
                printf '%s\n' "$candidate"
                return 0
            fi
        fi
    done

    return 1
}

PYTHON_BIN="$(find_python || true)"

if [ -z "$PYTHON_BIN" ]; then
    echo "Error: Python 3.10 or newer is required to create this environment."
    echo "Install python3.10+ and rerun ./setup.sh."
    exit 1
fi

PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))')"

# Crear el entorno si no existe o si su intérprete no coincide con el seleccionado
if [ ! -d "$VENV_DIR" ] || [ ! -x "$VENV_DIR/bin/python" ] || [ "$("$VENV_DIR/bin/python" -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))' 2>/dev/null)" != "$PYTHON_VERSION" ]; then
    if [ -d "$VENV_DIR" ]; then
        echo "Existing virtual environment is incompatible. Recreating it..."
        rm -rf "$VENV_DIR"
    fi

    echo "Creando entorno virtual..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"

    echo "Activando entorno..."
    source "$VENV_DIR/bin/activate"

    echo "Actualizando pip..."
    python -m pip install --upgrade pip

    echo "Instalando dependencias..."
    python -m pip install -r "$REQUIREMENTS"
else
    echo "Activando entorno existente..."
    source "$VENV_DIR/bin/activate"

    echo "Verificando e instalando dependencias..."
    python -m pip install -r "$REQUIREMENTS"
fi

echo
echo "Entorno virtual activado."
echo "Python: $(which python)"
echo "Pip:    $(which pip)"