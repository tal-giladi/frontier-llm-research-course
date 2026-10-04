#!/usr/bin/env bash
# Lesson 09.3 main path: torchtitan 0.3.0 on one 8x H100 node. NOT RUN IN THIS BUILD (multi-GPU is not piloted).
#
#   bash labs/module-09/lesson-03/run_titan.sh            # from the course repository root
#
# Needs: 8 CUDA GPUs on one node, CUDA 12.x/13.x, a Hugging Face token with access to meta-llama/Llama-3.1-8B
# (the tokenizer; set HF_TOKEN), about 4-6 GPU-hours in total for both layouts plus the optional ones.
set -euo pipefail

COURSE="$(cd "$(dirname "$0")/../../.." && pwd)"
WORK="${WORK:-$HOME/m09-titan}"
mkdir -p "$WORK"
cd "$WORK"

# 1. torchtitan at the pinned tag, with the pinned torch (torchtitan 0.3.0 requires PyTorch 2.14.0 or newer)
if [ ! -d torchtitan ]; then
  git clone --branch v0.3.0 --depth 1 https://github.com/pytorch/torchtitan
fi
cd torchtitan
CUDA_TAG="${CUDA_TAG:-cu128}"     # check pytorch.org for the CUDA wheel tags published for torch 2.14.1
pip install "torch==2.14.1" --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
pip install -r requirements.txt
pip install -e .
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token "$HF_TOKEN"

# 2. the record of the environment (goes into the run cards)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$WORK/gpus.csv"
nvidia-smi topo -m | tee "$WORK/topo.txt"
python -c "import torch, torchtitan; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())" | tee "$WORK/versions.txt"

# 3. interconnect baseline: NCCL all-reduce bus bandwidth on this node (lesson 02.3's measurement, NCCL this time)
LAB_TARGET=solution python "$COURSE/labs/module-02/lesson-03/run_dp.py" --device cuda --world 8 --preset baseline0 --vocab 32768 \
  --batch 8 --seq 1024 --dtype bf16 --buckets 25 --steps 30 --no-fsdp | tee "$WORK/allreduce.txt"

# 4. dry-run both configurations first (fake process groups, one GPU, one step): catches config errors cheaply
export PYTHONPATH="$COURSE/labs/module-09/lesson-03:${PYTHONPATH:-}"
for cfg in m09_fsdp8 m09_fsdp4_tp2; do
  NGPU=8 COMM_MODE="fake_backend" MODULE=titan_configs CONFIG=$cfg ./run_train.sh 2>&1 | tail -3
done

# 5. the two layouts of the experiment contract, each twice, alternating (A B A B) so drift hits both
for rep in 1 2; do
  for cfg in m09_fsdp8 m09_fsdp4_tp2; do
    NGPU=8 LOG_RANK=0,1,2,3,4,5,6,7 MODULE=titan_configs CONFIG=$cfg ./run_train.sh 2>&1 | tee "$WORK/${cfg}_rep${rep}.log"
    cp -r "outputs/$cfg/profiling" "$WORK/${cfg}_rep${rep}_profiling" || true
  done
done

# 6. optional layouts for the capability matrix (record whether each composes; a failure is a result)
for cfg in m09_fsdp4_pp2 m09_fsdp4_cp2_32k; do
  NGPU=8 LOG_RANK=0,1,2,3,4,5,6,7 MODULE=titan_configs CONFIG=$cfg ./run_train.sh 2>&1 | tee "$WORK/${cfg}.log" || \
    echo "$cfg FAILED (record it in the matrix)" | tee -a "$WORK/failures.txt"
done

# 7. summarise: tokens/s, MFU and memory per rank with intervals; exposed communication from the traces
python "$COURSE/labs/module-09/lesson-03/titan_report.py" "$WORK" --skip 10 | tee "$WORK/report.txt"
