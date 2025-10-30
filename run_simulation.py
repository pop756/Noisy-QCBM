import numpy as np
import json
import argparse
import os
import time
import fcntl
from simulator.Noise_simulator import Operation, DiagonalGate, PauliNoise, Circuit, sample_noisy_IQP_once_streaming_final
from typing import List
from tqdm import tqdm

def safe_load_and_append(save_path, new_samples):
    """
    안전하게 파일을 로드하고 새로운 샘플을 추가한 후 저장하는 함수
    파일 락을 사용하여 동시 접근을 방지
    """
    max_retries = 10
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            # 파일 락을 위한 락 파일 경로
            lock_path = save_path + ".lock"
            
            # 락 파일 생성 및 락 획득
            with open(lock_path, 'w') as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                
                try:
                    # 기존 데이터 로드
                    if os.path.exists(save_path):
                        existing_samples = np.load(save_path, allow_pickle=True).tolist()
                        if not isinstance(existing_samples, list):
                            existing_samples = [existing_samples]
                    else:
                        existing_samples = []
                    
                    # 새로운 샘플 추가
                    if isinstance(new_samples, list):
                        existing_samples.extend(new_samples)
                    else:
                        existing_samples.append(new_samples)
                    
                    # 저장
                    np.save(save_path, np.array(existing_samples, dtype=object))
                    return len(existing_samples)
                    
                finally:
                    # 락 해제는 with 구문이 끝날 때 자동으로 됨
                    pass
            
            # 락 파일 삭제
            try:
                os.remove(lock_path)
            except:
                pass
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Save attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(retry_delay * (2 ** attempt))  # 지수적 백오프
            else:
                print(f"Failed to save after {max_retries} attempts: {e}")
                raise
    
    return 0

def main():
    parser = argparse.ArgumentParser(description='Run noisy IQP simulation')
    parser.add_argument('--eps', type=float, required=True, help='Epsilon threshold for parameter filtering')
    parser.add_argument('--error_rate', type=float, required=True, help='Error rate for noise')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save results')
    parser.add_argument('--data_dir', type=str, default='RX_exp_sweep_1761111430', help='Directory containing gates.json and params.npy')
    parser.add_argument('--n_samples', type=int, default=1, help='Number of samples to generate')
    
    args = parser.parse_args()
    
    # 임시로 새로운 샘플들을 저장할 리스트
    new_samples = []
    
    # Load data
    with open(f"{args.data_dir}/gates.json", "r") as f:
        gates_data = json.load(f)
    params = np.load(f"{args.data_dir}/params.npy")
    
    # Filter parameters by epsilon threshold
    eps_index = []
    for i, param in enumerate(params):
        if abs(param) > args.eps:
            eps_index.append(i)
    
    filtered_params = params[eps_index]
    filtered_gates = [gates_data[i] for i in eps_index]
    
    print(f"Filtered {len(filtered_params)} gates with |param| > {args.eps}")
    
    # Build circuit
    ops = []
    for i, g in enumerate(filtered_gates):
        ops.append(DiagonalGate(qubits=tuple(g[0]), angle=filtered_params[i]))
        for j in g[0]:
            ops.append(PauliNoise(qubit=j, pI=1-args.error_rate, 
                                pX=args.error_rate/3, pY=args.error_rate/3, pZ=args.error_rate/3))
    
    test_circuit = Circuit(n_qubits=45, ops=ops)
    
    # Generate samples
    print(f"Generating {args.n_samples} samples...")
    successful_samples = 0
    
    for sample_idx in tqdm(range(args.n_samples)):
        try:
            sample, dbg = sample_noisy_IQP_once_streaming_final(test_circuit, shots=1)
            new_samples.append(sample)
            successful_samples += 1
            
            # Save results periodically (every 10 samples)
            if (sample_idx + 1) % 10 == 0:
                total_samples = safe_load_and_append(args.save_path, new_samples)
                new_samples = []  # 저장 후 임시 리스트 초기화
                print(f"Saved batch. Total samples in file: {total_samples}")
                
        except Exception as e:
            print(f"Error during sampling {sample_idx}: {e}")
    
    # Final save - 남은 샘플들 저장
    if new_samples:
        total_samples = safe_load_and_append(args.save_path, new_samples)
        print(f"Final save completed. Total samples in file: {total_samples}")
    else:
        # 파일이 존재하는 경우 총 샘플 수 확인
        if os.path.exists(args.save_path):
            existing_samples = np.load(args.save_path, allow_pickle=True).tolist()
            total_samples = len(existing_samples) if isinstance(existing_samples, list) else 1
        else:
            total_samples = 0
    
    print(f"Completed! Generated {successful_samples}/{args.n_samples} successful samples")
    print(f"Total samples in file: {total_samples}")
    print(f"Results saved to: {args.save_path}")

if __name__ == "__main__":
    main()