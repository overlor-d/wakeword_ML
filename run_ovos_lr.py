#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import openwakeword


def oww_model_path(name: str) -> str:
    base = Path(openwakeword.__file__).resolve().parent
    p = base / "resources" / "models" / name
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    return str(p)

def resample_48k_to_16k(x: np.ndarray) -> np.ndarray:
    # downsample factor 3 (48k -> 16k)
    return x[::3].astype(np.float32, copy=False)

def rms_normalize(x: np.ndarray, target_rms: float = 0.03) -> np.ndarray:
    r = float(np.sqrt(np.mean(x*x) + 1e-12))
    g = float(np.clip(target_rms / (r + 1e-12), 0.1, 10.0))
    return np.tanh(x * g).astype(np.float32)

def quantize_if_needed(x: np.ndarray, inp_detail: dict) -> np.ndarray:
    dtype = inp_detail["dtype"]
    if dtype == np.float32:
        return x.astype(np.float32, copy=False)
    q = inp_detail.get("quantization_parameters", {})
    scales = q.get("scales", None)
    zero_points = q.get("zero_points", None)
    if scales is None or len(scales) == 0:
        return x.astype(dtype)
    scale = float(scales[0]); zp = int(zero_points[0])
    y = np.round(x / scale + zp)
    info = np.iinfo(dtype)
    return np.clip(y, info.min, info.max).astype(dtype)

def dequantize_if_needed(y: np.ndarray, out_detail: dict) -> np.ndarray:
    dtype = out_detail["dtype"]
    if dtype == np.float32:
        return y.astype(np.float32, copy=False)
    q = out_detail.get("quantization_parameters", {})
    scales = q.get("scales", None)
    zero_points = q.get("zero_points", None)
    if scales is None or len(scales) == 0:
        return y.astype(np.float32)
    scale = float(scales[0]); zp = int(zero_points[0])
    return ((y.astype(np.float32) - zp) * scale).astype(np.float32)


class OWWEmbedder:
    def __init__(self):

        self.mel = tflite.Interpreter(
            model_path=self.mel_path,
            num_threads=1,
            experimental_op_resolver_type=tflite.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        self.mel.allocate_tensors()
        self.mel_in = self.mel.get_input_details()[0]
        self.mel_out = self.mel.get_output_details()[0]
        self.mel_in_shape = tuple(self.mel_in["shape"])
        if len(self.mel_in_shape) == 2:
            self.audio_len = int(self.mel_in_shape[1])
        else:
            self.audio_len = int(self.mel_in_shape[1])

        self.emb = tflite.Interpreter(
            model_path=self.emb_path,
            num_threads=1,
            experimental_op_resolver_type=tflite.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        self.emb.allocate_tensors()
        self.emb_in = self.emb.get_input_details()[0]
        self.emb_out = self.emb.get_output_details()[0]

    def embed_16k(self, x: np.ndarray) -> np.ndarray:
        if x.size < self.audio_len:
            x = np.pad(x, (0, self.audio_len - x.size))
        elif x.size > self.audio_len:
            x = x[:self.audio_len]
        x = x.astype(np.float32)

        if len(self.mel_in_shape) == 2:
            xin = x.reshape(1, -1)
        else:
            xin = x.reshape(1, -1, 1)

        self.mel.set_tensor(self.mel_in["index"], quantize_if_needed(xin, self.mel_in))
        self.mel.invoke()
        mel_out = dequantize_if_needed(self.mel.get_tensor(self.mel_out["index"]), self.mel_out)

        emb_in_shape = tuple(self.emb_in["shape"])
        flat = mel_out.astype(np.float32).reshape(-1)
        need = int(np.prod(emb_in_shape))
        if flat.size != need:
            flat = flat[:need] if flat.size > need else np.pad(flat, (0, need - flat.size))
        emb_in = flat.reshape(emb_in_shape).astype(np.float32)

        self.emb.set_tensor(self.emb_in["index"], quantize_if_needed(emb_in, self.emb_in))
        self.emb.invoke()
        e = dequantize_if_needed(self.emb.get_tensor(self.emb_out["index"]), self.emb_out).reshape(-1).astype(np.float32)
        e /= float(np.linalg.norm(e) + 1e-12)
        return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ovos_lr_model.npz")
    ap.add_argument("--mic", type=int, default=9)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--hop_ms", type=int, default=100)
    ap.add_argument("--cooldown_s", type=float, default=1.5)
    ap.add_argument("--consecutive", type=int, default=2, help="nb de hops consécutifs au-dessus du seuil")
    args = ap.parse_args()

    pack = np.load(args.model)
    w = pack["w"].astype(np.float32)
    b = float(pack["b"])
    mu = pack["mu"].astype(np.float32)
    sdv = pack["sd"].astype(np.float32)
    thr = float(pack["threshold"])

    emb = OWWEmbedder()

    hop = int(round(args.sr * (args.hop_ms / 1000.0)))
    buf_len = args.sr  # 1s à 48k, on crop au besoin ensuite
    ring = np.zeros((buf_len,), dtype=np.float32)
    idx = 0
    filled = False

    last_fire = 0.0
    streak = 0

    def sigmoid(z: float) -> float:
        return 1.0 / (1.0 + np.exp(-z))

    def on_audio(indata, frames, time_info, status):
        nonlocal ring, idx, filled, last_fire, streak
        x = indata[:, 0].astype(np.float32)
        n = x.size
        if n >= ring.size:
            ring[:] = x[-ring.size:]
            idx = 0
            filled = True
        else:
            end = idx + n
            if end < ring.size:
                ring[idx:end] = x
                idx = end
            else:
                k = ring.size - idx
                ring[idx:] = x[:k]
                ring[:end - ring.size] = x[k:]
                idx = end - ring.size
                filled = True

    with sd.InputStream(device=args.mic, channels=1, samplerate=args.sr, blocksize=hop, callback=on_audio):
        while True:
            time.sleep(args.hop_ms / 1000.0)
            if not filled:
                continue

            # reconstituer buffer dans l’ordre
            x48 = np.concatenate([ring[idx:], ring[:idx]])
            x48 = rms_normalize(x48)

            x16 = resample_48k_to_16k(x48)
            e = emb.embed_16k(x16)

            xs = (e - mu) / sdv
            p = float(sigmoid(float(xs @ w + b)))

            if p >= thr:
                streak += 1
            else:
                streak = 0

            now = time.time()
            if streak >= args.consecutive and (now - last_fire) >= args.cooldown_s:
                last_fire = now
                streak = 0
                print("TRUE", flush=True)

if __name__ == "__main__":
    main()
