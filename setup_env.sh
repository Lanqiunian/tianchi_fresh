#!/bin/bash

# --- Configuration ---
VENV_NAME="tianchi_project_env"
PYTHON_VERSION="3.9"
ANACONDA_MODULE="env/anaconda3/2022.05"
COMPILER_MODULE="compiler/gcc/10.1.0"

source /home-ssd/hpc/ini_module.bash

# --- Helper Functions ---
echoinfo() {
    echo "[INFO] $1"
}

echoerror() {
    echo "[ERROR] $1" >&2
}

# --- Main Script ---

echoinfo "Starting environment setup for Tianchi project..."

# 1. Purge existing modules
echoinfo "Purging existing modules..."
module purge
# ... (error checking) ...

# 2. Load necessary base modules
echoinfo "Loading compiler module: ${COMPILER_MODULE}..."
module load "${COMPILER_MODULE}"
# ... (error checking) ...

echoinfo "Loading Anaconda module: ${ANACONDA_MODULE}..."
module load "${ANACONDA_MODULE}"
if [ $? -ne 0 ]; then
    echoerror "Failed to load Anaconda module: ${ANACONDA_MODULE}. Cannot proceed."
    exit 1
fi

echoinfo "Modules loaded (initial):"
module list
echo "Initial PATH: $PATH"
echo "---------------------------------------------"

# 3. Check/Create Conda environment
# ... (你的现有逻辑不变) ...
if conda env list | grep -q "${VENV_NAME}"; then
    echoinfo "Conda environment '${VENV_NAME}' already exists."
else
    echoinfo "Conda environment '${VENV_NAME}' not found. Creating it now with Python ${PYTHON_VERSION}..."
    conda create -n "${VENV_NAME}" python="${PYTHON_VERSION}" -y
    # ... (error checking) ...
    echoinfo "Conda environment '${VENV_NAME}' created successfully."
fi
echo "---------------------------------------------"

# 4. Activate the Conda environment
CONDA_BASE_PATH=$(conda info --base)
CONDA_SH_PATH="${CONDA_BASE_PATH}/etc/profile.d/conda.sh"

if [ -f "${CONDA_SH_PATH}" ]; then
    echoinfo "Sourcing Conda script from ${CONDA_SH_PATH}..."
    source "${CONDA_SH_PATH}"
else
    echoerror "Conda setup script '${CONDA_SH_PATH}' not found. Activation might fail."
fi

echoinfo "Activating Conda environment: ${VENV_NAME}..."
conda activate "${VENV_NAME}"
if [ $? -ne 0 ]; then
    echoerror "Failed to activate Conda environment '${VENV_NAME}'."
    exit 1
fi

# ***** 新增/修改部分：显式调整 PATH *****
VENV_PATH=$(conda env list | grep "${VENV_NAME}" | awk '{print $NF}') # 获取虚拟环境的精确路径
if [ -z "${VENV_PATH}" ] || [ ! -d "${VENV_PATH}/bin" ]; then
    echoerror "Could not determine the bin path for environment ${VENV_NAME}."
else
    echoinfo "Virtual environment bin path: ${VENV_PATH}/bin"
    echoinfo "PATH before manual adjustment: $PATH"
    # 从 PATH 中移除旧的 Anaconda base bin (如果存在且在前面)
    # 并将虚拟环境的 bin 放在最前面
    # 这是一个相对安全的做法，避免重复添加
    CLEAN_PATH=$(echo "$PATH" | sed -e "s|${CONDA_BASE_PATH}/bin:||g" -e "s|:${CONDA_BASE_PATH}/bin||g" -e "s|${CONDA_BASE_PATH}/bin||g")
    export PATH="${VENV_PATH}/bin:${CLEAN_PATH}"
    echoinfo "PATH after manual adjustment: $PATH"
fi
# ***** 结束新增/修改部分 *****

echoinfo "Successfully activated Conda environment (as reported by conda): $(conda env list | grep '*' | awk '{print $1}')"
echo "Current Python version (after PATH adjustment): $(python --version)"
echo "Python executable (after PATH adjustment): $(which python)"
echo "Python executable (from sys.executable): $(python -c 'import sys; print(sys.executable)')"
echo "---------------------------------------------"

# 5. Advise on installing dependencies
# ... (你的现有逻辑不变) ...

echoinfo "Setup script finished."
echoinfo "If you sourced this script (e.g., 'source ./setup_script.sh'), the environment '${VENV_NAME}' should be active with the correct PATH."
echoinfo "If you just executed it (e.g., './setup_script.sh'), the PATH change will only affect this script's execution, not your parent shell."