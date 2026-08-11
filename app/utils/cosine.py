import numpy as np

def pack_f32(x: np.ndarray) -> bytes:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x.tobytes()

def unpack_f32(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)
