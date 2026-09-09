
from __future__ import annotations
import numpy as np

WINDOW = 31
AA20_STR = "ACDEFGHIKLMNPQRSTVWY"
AA20 = set(AA20_STR)


def norm_sequence(seq: str, window: int = WINDOW) -> str:
    seq = (seq or "").strip().upper().replace(" ", "")
    seq = "".join(ch if (ch in AA20 or ch == "X") else "X" for ch in seq)
    if len(seq) > window:
        seq = seq[:window]
    if len(seq) < window:
        seq = seq + "X" * (window - len(seq))
    return seq


def _build_blosum62_dict():
    aa_order = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    mat = np.array([
        [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
        [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
        [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
        [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
        [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
        [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
        [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
        [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
        [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
        [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
        [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
        [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
        [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
        [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
        [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
        [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
        [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
        [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
        [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
        [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
    ], dtype=np.float32)
    raw = {aa_order[i]: mat[i] for i in range(20)}
    out = {}
    for aa in AA20_STR:
        vec = np.zeros((20,), dtype=np.float32)
        raw_vec = raw[aa]
        for j, aa2 in enumerate(AA20_STR):
            vec[j] = raw_vec[aa_order.index(aa2)]
        out[aa] = vec
    out["X"] = np.zeros((20,), dtype=np.float32)
    return out

BLOSUM62_DICT = _build_blosum62_dict()


def blosum62_encode(seq: str) -> np.ndarray:
    seq = norm_sequence(seq)
    return np.stack([BLOSUM62_DICT.get(ch, BLOSUM62_DICT["X"]) for ch in seq], axis=0).astype(np.float32)


def aa_onehot_21(seq: str) -> np.ndarray:
    seq = norm_sequence(seq)
    mat = np.zeros((len(seq), 21), dtype=np.float32)
    for i, ch in enumerate(seq):
        mat[i, AA20_STR.index(ch) if ch in AA20_STR else 20] = 1.0
    return mat


def relative_position_and_center_flag(seq_len: int = WINDOW) -> np.ndarray:
    center_idx = seq_len // 2
    denom = float(center_idx) if center_idx > 0 else 1.0
    relpos = np.arange(-center_idx, seq_len-center_idx, dtype=np.float32)[:, None] / denom
    center_flag = np.zeros((seq_len, 1), dtype=np.float32)
    center_flag[center_idx, 0] = 1.0
    return np.concatenate([relpos, center_flag], axis=1).astype(np.float32)


def encode_sequence_features(seq: str) -> np.ndarray:
    seq = norm_sequence(seq)
    return np.concatenate([
        blosum62_encode(seq),
        aa_onehot_21(seq),
        relative_position_and_center_flag(len(seq)),
    ], axis=1).astype(np.float32)


def input_dim() -> int:
    return 43
