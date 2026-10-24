#!/bin/bash
#SBATCH --job-name=qwen-sh
#SBATCH --output=/gscratch/scrubbed/parkmoll/logs/%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --account kohlab
#SBATCH --partition gpu-l40

# Usage:
#   Remote model (API):  sbatch --mem 8G run.sh
#   Local vLLM:          sbatch --gpus-per-node=l40:2 --mem 48G  run.sh --serve-vllm

set -uo pipefail


TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
PROJECT_DIR="/mmfs1/home/parkmoll/code/advocate-project/synthetic_hospital"
VLLM_PORT=30001
VLLM_LOG="/gscratch/scrubbed/parkmoll/logs/vllm-server.log"
VLLM_MODEL="Qwen/Qwen3.5-4B"
export SH_MODEL_TAG="qwen3-4b"

SERVE_VLLM="${SERVE_VLLM:-0}"
for arg in "$@"; do
    case "$arg" in
        --serve-vllm)    SERVE_VLLM=1 ;;
        --no-serve-vllm) SERVE_VLLM=0 ;;
        *) echo "Unknown argument: $arg"; exit 2 ;;
    esac
done

#---------------------------------------------------------------
# (Optional) Serve the model with vLLM
# ---------------------------------------------------------------

if [ "$SERVE_VLLM" = "1" ]; then

    source /gscratch/scrubbed/parkmoll/virtual_envs/vllm_env/bin/activate
    echo "Starting vLLM on port $VLLM_PORT (model: $VLLM_MODEL, log: $VLLM_LOG)"
    
    module load gcc
    module load cuda/13.4.2

    vllm serve $VLLM_MODEL \
        --port $VLLM_PORT \
        --max-model-len 32768 \
	--tensor-parallel-size 2 \
        --enable-auto-tool-choice --tool-call-parser qwen3_coder \
        > $VLLM_LOG 2>&1 &
    
    VLLM_PID=$!

    cleanup() {
        echo "Shutting down vLLM (PID=$VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null
        wait "$VLLM_PID" 2>/dev/null
    }
    trap cleanup EXIT

    echo "Waiting for vLLM to be ready..."
    until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "vLLM died during startup. Last log lines:"
            tail -n 50 "$VLLM_LOG"
            exit 1
        fi
        sleep 5
    done
    echo "vLLM is ready."
else
    echo "Skipping vLLM serving step."
fi

# ---------------------------------------------------------------
# Enter the apptainer, start the API, then run the agent
# ---------------------------------------------------------------
cd $PROJECT_DIR
mkdir -p logs results/traces "$PWD/sh-state"
module load apptainer

apptainer run \
    --env-file $PWD/.env \
    --bind $PWD/sh-state:/var/lib/synthetic-hospital \
    --bind $PWD/scripts:/app/scripts \
    --bind $PWD/eval:/app/eval \
    --bind $PWD/results:/app/results \
    --bind $PWD/data:/app/data synthetic_hospital.sif \
    sh -c '
        echo "Entering apptainer container..."
	entrypoint.sh load

        echo "Starting server"
        uvicorn epic_sim.app.main:app --host 127.0.0.1 --port 8000 > $SH_STATE_DIR/api.log 2>&1 &
        APP_PID=$!
        trap "kill $APP_PID 2>/dev/null" EXIT

	sleep 5
        echo "App is up (PID=$APP_PID)."
	echo "Running agent eval (model: $SH_MODEL_TAG)"
	python -m scripts.env_run --model $SH_MODEL_TAG --episodes 800 --budget 40 --split train
    '
