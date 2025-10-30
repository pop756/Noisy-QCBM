import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
from tqdm import tqdm



###############################################################################
# Data structures
###############################################################################
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli

def apply_multi_z_phase(qc, qubits, theta):
    label = 'Z' * len(qubits)
    pauli = Pauli(label)
    qc.append(PauliEvolutionGate(pauli, time=theta), qubits)

@dataclass
class DiagonalGate:
    qubits: Tuple[int, ...]
    angle: float

@dataclass
class PauliNoise:
    qubit: int
    pI: float
    pX: float
    pY: float
    pZ: float

Operation = Union[DiagonalGate, PauliNoise]

@dataclass
class Circuit:
    n_qubits: int
    ops: List[Operation]

def merge_diagonal_ops(ops: List[DiagonalGate]) -> List[DiagonalGate]:
    acc: Dict[Tuple[int, ...], float] = {}
    for g in ops:
        key = tuple(sorted(g.qubits))
        acc[key] = acc.get(key, 0.0) + g.angle
    merged = []
    for key, angle in acc.items():
        if abs(angle) > 1e-12:
            merged.append(DiagonalGate(qubits=key, angle=angle))
    return merged


###############################################################################
# Noise modeling from Lemma 3
###############################################################################

def compute_lemma3_params(noise: PauliNoise) -> Tuple[float, PauliNoise, PauliNoise]:
    pX, pY, pZ = noise.pX, noise.pY, noise.pZ
    p = pZ + min(pX, pY)
    if p >= 0.5: p = 0.499999

    if pX >= pY:
        pX1 = abs(pX - pY) / (1 - 2*p)
        N1 = PauliNoise(noise.qubit, 1 - pX1, pX1, 0, 0)
        pX2 = min(pX, pY) / p if p > 0 else 0
        N2 = PauliNoise(noise.qubit, 1 - pX2, pX2, 0, 0)
    else:
        pY1 = abs(pY - pX) / (1 - 2*p)
        N1 = PauliNoise(noise.qubit, 1 - pY1, 0, pY1, 0)
        pY2 = min(pX, pY) / p if p > 0 else 0
        N2 = PauliNoise(noise.qubit, 1 - pY2, 0, pY2, 0)
    return p, N1, N2


def sample_noise_branch(noise: PauliNoise):
    p, N1, N2 = compute_lemma3_params(noise)
    if random.random() < 2*p:
        return "dephase", N2, p
    else:
        return "nondephase", N1, p


###############################################################################
# Helpers
###############################################################################

def init_b_state(n_qubits: int):
    return ['+' for _ in range(n_qubits)]

def z_eigenvalue(bit01: int) -> int:
    return +1 if bit01 == 0 else -1


###############################################################################
# Collapse propagation + diagonal gate reduction
###############################################################################

def streaming_process(circ_raw: Circuit, b_state: List[Union[str,int]]) -> Circuit:
    new_ops: List[DiagonalGate] = []

    for op in circ_raw.ops:
        if isinstance(op, PauliNoise):
            branch, sub_noise, p = sample_noise_branch(op)
            if branch == "dephase":
                q_c = op.qubit
                b_state[q_c] = random.choice([0,1])
                sign_c = z_eigenvalue(b_state[q_c])

                updated_ops: List[DiagonalGate] = []
                for g in new_ops:
                    if q_c in g.qubits:
                        remaining = tuple(q for q in g.qubits if q != q_c)
                        new_angle = g.angle * sign_c
                        if len(remaining) > 0:
                            remaining_sorted = tuple(sorted(remaining))
                            updated_ops.append(DiagonalGate(remaining_sorted, new_angle))
                    else:
                        updated_ops.append(g)
                new_ops = updated_ops
            else:
                continue

        elif isinstance(op, DiagonalGate):
            qs = list(op.qubits)
            classical_sign = 1
            quantum_qubits = []
            for q in qs:
                if b_state[q] == '+':
                    quantum_qubits.append(q)
                else:
                    classical_sign *= z_eigenvalue(b_state[q])

            if len(quantum_qubits) == 0:
                continue

            eff_angle = op.angle * classical_sign
            qq_sorted = tuple(sorted(quantum_qubits))
            new_ops.append(DiagonalGate(qq_sorted, eff_angle))

        else:
            raise ValueError("Unknown op type in streaming_process")

    merged_ops = merge_diagonal_ops(new_ops)
    circ3 = Circuit(n_qubits=circ_raw.n_qubits, ops=merged_ops)
    return circ3, b_state



###############################################################################
# Graph + component decomposition
###############################################################################

def build_interaction_graph(circ3: Circuit):
    g = {q:set() for q in range(circ3.n_qubits)}
    for op in circ3.ops:
        if isinstance(op, DiagonalGate):
            qs = list(op.qubits)
            for i in range(len(qs)):
                for j in range(i+1, len(qs)):
                    g[qs[i]].add(qs[j])
                    g[qs[j]].add(qs[i])
    return g

def connected_components(g: Dict[int,set]):
    seen = set()
    comps = []
    for start in g.keys():
        if start in seen: continue
        stack, comp = [start], []
        while stack:
            v = stack.pop()
            if v in seen: continue
            seen.add(v)
            comp.append(v)
            for w in g[v]:
                if w not in seen:
                    stack.append(w)
        comps.append(sorted(comp))
    return comps



        
import concurrent.futures

def run_subcircuit_and_sample(subcircuit: Circuit, subset: List[int], shots=1, timeout_sec=60):
    qc = QuantumCircuit(len(subset))
    for q in range(len(subset)):
        qc.h(q)

    for op in subcircuit.ops:
        local_qubits = [subset.index(q) for q in op.qubits]
        apply_multi_z_phase(qc, local_qubits, op.angle)

    for q in range(len(subset)):
        qc.h(q)

    qc.measure_all()
    sim = AerSimulator()
    compiled = transpile(qc, sim)

    # 내부 실행 함수를 별도 스레드로 감싸기
    def run_sim():
        job = sim.run(compiled, shots=shots)
        return job.result().get_counts()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_sim)
            counts = future.result(timeout=timeout_sec)  # 1분 제한
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"Simulation exceeded {timeout_sec} seconds and was stopped.")

    bitstrings = []
    if isinstance(counts, dict):
        for bitstr, freq in counts.items():
            for _ in range(freq):
                bitstrings.append(bitstr)
    else:
        bitstrings.append(counts)

    samples_out = []
    for bitstr in bitstrings:
        out_bits = {}
        for i, q_global in enumerate(subset[::-1]):
            out_bits[q_global] = int(bitstr[i])
        samples_out.append(out_bits)

    return samples_out




###############################################################################
# Sampling components and full run
###############################################################################
from random import shuffle
def sample_components(circ3: Circuit,
                      b_state: List[Union[str,int]],
                      components: List[List[int]],
                      shots: int):
    n = circ3.n_qubits
    all_results = [ [None]*n for _ in range(shots) ]

    for comp in components:
        comp_set = set(comp)
        plus_qubits = [q for q in comp if b_state[q] == '+']
        collapsed_qubits = [q for q in comp if b_state[q] != '+']

        if len(plus_qubits) > 0:
            sub_ops_raw = [
                op for op in circ3.ops
                if set(op.qubits).issubset(comp_set)
            ]
            sub_ops_merged = merge_diagonal_ops(sub_ops_raw)
            subc = Circuit(len(comp), sub_ops_merged)

            measured_list = run_subcircuit_and_sample(subc, comp, shots=shots)
            shuffle(measured_list)
            for shot_idx in range(shots):
                measured = measured_list[shot_idx]
                for q in comp:
                    if q in measured:
                        all_results[shot_idx][q] = measured[q]
                for q in collapsed_qubits:
                    if all_results[shot_idx][q] is None:
                        all_results[shot_idx][q] = random.randint(0,1)
        else:
            for shot_idx in range(shots):
                for q in comp:
                    all_results[shot_idx][q] = random.randint(0,1)

    for shot_idx in range(shots):
        for q in range(n):
            if all_results[shot_idx][q] is None:
                if b_state[q] == '+':
                    all_results[shot_idx][q] = 0
                else:
                    all_results[shot_idx][q] = random.randint(0,1)

    return all_results



def sample_noisy_IQP_once_streaming_final(circ_raw: Circuit, shots=1):
    b_state = init_b_state(circ_raw.n_qubits)
    circ3, b_state = streaming_process(circ_raw, b_state)
    graph = build_interaction_graph(circ3)
    comps = connected_components(graph)
    samples = sample_components(circ3, b_state, comps, shots=shots)
    debug = {
        "b_state_final": b_state,
        "circ3_ops": len(circ3.ops),
        "components": comps,
    }
    return samples, debug

