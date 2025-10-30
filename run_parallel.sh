#!/bin/bash

# 설정 변수들
EPS=0.12
ERROR_RATE=0.02
N_SAMPLES=1
DATA_DIR="RX_exp_sweep_1761111430"
MAX_PARALLEL=8  # 동시에 실행할 최대 프로세스 수

# 결과 저장 디렉토리 생성
RESULTS_DIR="simulation_results"
mkdir -p $RESULTS_DIR

echo "Starting 2000 parallel simulations..."
echo "Max parallel processes: $MAX_PARALLEL"

# 병렬 실행을 위한 함수
run_batch() {
    local start=$1
    local end=$2
    
    for ((i=$start; i<=$end; i++)); do
        {
            save_path="${RESULTS_DIR}/result_${i}.npy"
            echo "Starting job $i"
            
            python run_simulation.py \
                --eps $EPS \
                --error_rate $ERROR_RATE \
                --save_path $save_path \
                --data_dir $DATA_DIR \
                --n_samples $N_SAMPLES
            
            echo "Completed job $i"
        } &
        
        # MAX_PARALLEL 개수만큼만 동시 실행
        if (( i % MAX_PARALLEL == 0 )); then
            wait  # 모든 백그라운드 작업이 완료될 때까지 대기
        fi
    done
    wait  # 남은 작업들 완료 대기
}

# 전체 작업 실행
run_batch 1 2000

echo "All simulations completed!"

# 결과 병합
echo "Merging results..."
python - << EOF
import numpy as np
import os
from glob import glob

results_dir = "$RESULTS_DIR"
result_files = glob(os.path.join(results_dir, "result_*.npy"))

all_samples = []
successful_files = 0

for file in result_files:
    try:
        samples = np.load(file, allow_pickle=True).tolist()
        if isinstance(samples, list):
            all_samples.extend(samples)
        else:
            all_samples.append(samples)
        successful_files += 1
    except Exception as e:
        print(f"Error loading {file}: {e}")

print(f"Successfully loaded {successful_files} files")
print(f"Total samples collected: {len(all_samples)}")
np.save(os.path.join(results_dir, "all_results.npy"), np.array(all_samples, dtype=object))
print("Merged results saved as all_results.npy")
EOF