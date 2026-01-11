#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import numpy as np
import soundfile as sf
import tflite_runtime.interpreter as tflite
import openwakeword


# ---------- utils paths ----------
def oww_model_path(name: str) -> str:
    # openwakeword/resources/models/<name>
    base = Path(openwakeword.__file__).resolve().parent
    p = base / "resources" / "models" / name
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    return str(p)


# ---------- audio ----------
def load_mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    x = x.mean(axis=1)
    x = x - float(np.mean(x)) if x.size else x
    return x, int(sr)

def resample_linear(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr or x.size == 0:
        return x.astype(np.float32, copy=False)
    n_out = int(round(x.size * (target_sr / sr)))
    if n_out <= 1:
        return np.zeros((0,), dtype=np.float32)
    t_in = np.linspace(0.0, 1.0, num=x.size, endpoint=False, dtype=np.float32)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float32)
    return np.interp(t_out, t_in, x).astype(np.float32)

def rms_normalize(x: np.ndarray, target_rms: float = 0.03) -> np.ndarray:
    if x.size == 0:
        return x
    r = float(np.sqrt(np.mean(x * x) + 1e-12))
    g = float(np.clip(target_rms / (r + 1e-12), 0.1, 10.0))
    y = np.tanh(x * g).astype(np.float32)
    return y

def center_crop_energy(x: np.ndarray, sr: int, win_s: float) -> np.ndarray:
    win = int(round(win_s * sr))
    if win <= 0:
        return np.zeros((0,), dtype=np.float32)
    if x.size == 0:
        return np.zeros((win,), dtype=np.float32)

    frame = int(round(0.02 * sr))
    hop = int(round(0.01 * sr))
    frame = max(frame, 1)
    hop = max(hop, 1)

    x2 = x if x.size >= frame else np.pad(x, (0, frame - x.size))
    n_frames = 1 + (x2.size - frame) // hop
    rms = np.empty((n_frames,), dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        seg = x2[s:s + frame]
        rms[i] = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    i_max = int(np.argmax(rms))
    center = i_max * hop + frame // 2

    start = center - win // 2
    end = start + win

    if start < 0:
        x = np.pad(x, (-start, 0))
        start = 0
        end = win
    if end > x.size:
        x = np.pad(x, (0, end - x.size))
    return x[start:end].astype(np.float32)

def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # très léger, pour robustesse sans libs:
    # gain + bruit blanc + petit décalage temporel
    y = x.copy()

    gain = float(rng.uniform(0.7, 1.3))
    y *= gain

    noise = rng.normal(0.0, 0.003, size=y.shape).astype(np.float32)
    y = y + noise

    shift = int(rng.integers(-200, 200))
    y = np.roll(y, shift)

    return np.tanh(y).astype(np.float32)


# ---------- TFLite helpers ----------
def tflite_io(interp: Interpreter):
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    return inp, out

def quantize_if_needed(x: np.ndarray, inp_detail: dict) -> np.ndarray:
    dtype = inp_detail["dtype"]
    if dtype == np.float32:
        return x.astype(np.float32, copy=False)
    # quantized?
    q = inp_detail.get("quantization_parameters", {})
    scales = q.get("scales", None)
    zero_points = q.get("zero_points", None)
    if scales is None or len(scales) == 0:
        # fallback: cast
        return x.astype(dtype)
    scale = float(scales[0])
    zp = int(zero_points[0])
    y = np.round(x / scale + zp)
    info = np.iinfo(dtype)
    y = np.clip(y, info.min, info.max).astype(dtype)
    return y

def dequantize_if_needed(y: np.ndarray, out_detail: dict) -> np.ndarray:
    dtype = out_detail["dtype"]
    if dtype == np.float32:
        return y.astype(np.float32, copy=False)
    q = out_detail.get("quantization_parameters", {})
    scales = q.get("scales", None)
    zero_points = q.get("zero_points", None)
    if scales is None or len(scales) == 0:
        return y.astype(np.float32)
    scale = float(scales[0])
    zp = int(zero_points[0])
    return ((y.astype(np.float32) - zp) * scale).astype(np.float32)


class OWWEmbedder:
    def __init__(self):
        self.mel_path = oww_model_path("melspectrogram.tflite")
        self.emb_path = oww_model_path("embedding_model.tflite")

        self.mel = tflite.Interpreter(
            model_path=self.mel_path,
            num_threads=1,
            experimental_op_resolver_type=tflite.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )

        self.mel.allocate_tensors()
        self.mel_in, self.mel_out = tflite_io(self.mel)

        
        self.emb = tflite.Interpreter(
            model_path=self.emb_path,
            num_threads=1,
            experimental_op_resolver_type=tflite.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        self.emb.allocate_tensors()
        self.emb_in, self.emb_out = tflite_io(self.emb)

        # infer required audio length from mel model input shape
        self.mel_in_shape = tuple(self.mel_in["shape"])
        # handle [1, N] or [1, N, 1]
        if len(self.mel_in_shape) == 2:
            self.audio_len = int(self.mel_in_shape[1])
        elif len(self.mel_in_shape) == 3:
            self.audio_len = int(self.mel_in_shape[1])
        else:
            raise ValueError(f"Unexpected mel input shape: {self.mel_in_shape}")

    def embed(self, audio_16k: np.ndarray) -> np.ndarray:
        x = audio_16k
        # pad/crop to expected length
        if x.size < self.audio_len:
            x = np.pad(x, (0, self.audio_len - x.size))
        elif x.size > self.audio_len:
            x = x[:self.audio_len]
        x = x.astype(np.float32)

        # mel model expects shape like [1,N] or [1,N,1]
        if len(self.mel_in_shape) == 2:
            xin = x.reshape(1, -1)
        else:
            xin = x.reshape(1, -1, 1)

        xin_q = quantize_if_needed(xin, self.mel_in)
        self.mel.set_tensor(self.mel_in["index"], xin_q)
        self.mel.invoke()
        mel_out = self.mel.get_tensor(self.mel_out["index"])
        mel_out = dequantize_if_needed(mel_out, self.mel_out)

        # feed embedding model
        emb_in_shape = tuple(self.emb_in["shape"])
        # reshape safely: rely on total size match
        mel_flat = mel_out.astype(np.float32).reshape(-1)
        needed = int(np.prod(emb_in_shape))
        if mel_flat.size != needed:
            # fallback: pad/crop
            if mel_flat.size < needed:
                mel_flat = np.pad(mel_flat, (0, needed - mel_flat.size))
            else:
                mel_flat = mel_flat[:needed]
        emb_in = mel_flat.reshape(emb_in_shape).astype(np.float32)

        emb_in_q = quantize_if_needed(emb_in, self.emb_in)
        self.emb.set_tensor(self.emb_in["index"], emb_in_q)
        self.emb.invoke()
        e = self.emb.get_tensor(self.emb_out["index"])
        e = dequantize_if_needed(e, self.emb_out).reshape(-1).astype(np.float32)

        # L2 normalize
        e /= float(np.linalg.norm(e) + 1e-12)
        return e


# ---------- logistic regression (numpy) ----------
def train_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, lr: float = 0.1, steps: int = 2000) -> tuple[np.ndarray, float]:
    # X: (n,d) standardized
    n, d = X.shape
    w = np.zeros((d,), dtype=np.float32)
    b = 0.0

    for _ in range(steps):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        # gradients
        gw = (X.T @ (p - y)) / n + l2 * w
        gb = float(np.mean(p - y))
        w -= lr * gw.astype(np.float32)
        b -= lr * gb
    return w, float(b)

def scores_from_model(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    z = X @ w + b
    p = 1.0 / (1.0 + np.exp(-z))
    return p.astype(np.float32)

def auc_eer(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float, float]:
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    thr = np.unique(s)[::-1]
    thr = np.concatenate([[thr[0] + 1e-6], thr, [thr[-1] - 1e-6]])

    P = float(pos.size); N = float(neg.size)
    tpr_list=[]; fpr_list=[]
    eer=1.0; eer_thr=float(thr[0]); best=1e9

    for t in thr:
        pred = (s >= t).astype(np.int32)
        tp = float(np.sum((pred==1) & (y==1)))
        fp = float(np.sum((pred==1) & (y==0)))
        fn = P - tp
        tn = N - fp
        tpr = tp/(tp+fn+1e-12)
        fpr = fp/(fp+tn+1e-12)
        tpr_list.append(tpr); fpr_list.append(fpr)
        diff = abs(fpr - (1.0 - tpr))
        if diff < best:
            best = diff
            eer = (fpr + (1.0 - tpr))/2.0
            eer_thr = float(t)

    fpr_arr = np.array(fpr_list, dtype=np.float64)
    tpr_arr = np.array(tpr_list, dtype=np.float64)
    order = np.argsort(fpr_arr)
    auc = float(np.trapz(tpr_arr[order], fpr_arr[order]))
    return auc, float(eer), float(eer_thr)

def zero_fp_threshold(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    t = float(np.max(neg) + 1e-6)
    fnr = float(np.mean(pos < t))
    return t, fnr


def list_wavs(d: Path) -> list[Path]:
    return sorted([p for p in d.glob("*.wav") if p.is_file()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positive", required=True)
    ap.add_argument("--negative", required=True)
    ap.add_argument("--out", default="ovos_lr_model.npz")
    ap.add_argument("--report", default="ovos_lr_report.json")
    ap.add_argument("--augment", type=int, default=5, help="embeddings supplémentaires par fichier (défaut 5)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    pos_files = list_wavs(Path(args.positive))
    neg_files = list_wavs(Path(args.negative))
    if not pos_files or not neg_files:
        raise SystemExit("Dossiers vides.")

    rng = np.random.default_rng(args.seed)
    emb = OWWEmbedder()

    def make_set(files: list[Path], label: int) -> tuple[list[np.ndarray], list[int], list[str]]:
        X=[]; y=[]; names=[]
        for f in files:
            x, sr = load_mono(f)
            x = resample_linear(x, sr, 16000)
            x = rms_normalize(x)
            # 1 embedding "clean"
            xc = center_crop_energy(x, 16000, win_s=1.2)
            X.append(emb.embed(xc)); y.append(label); names.append(f.name)

            # augmentations
            for _ in range(args.augment):
                xa = augment(xc, rng)
                X.append(emb.embed(xa)); y.append(label); names.append(f.name + ":aug")
        return X, y, names

    Xp, yp, np_names = make_set(pos_files, 1)
    Xn, yn, nn_names = make_set(neg_files, 0)

    X = np.vstack(Xp + Xn).astype(np.float32)
    y = np.array(yp + yn, dtype=np.float32)

    # standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Xs = (X - mu) / sd

    w, b = train_logreg(Xs, y, l2=1e-3, lr=0.1, steps=2500)
    p = scores_from_model(Xs, w, b)

    p_pos = p[y == 1]
    p_neg = p[y == 0]

    auc, eer, eer_thr = auc_eer(p_pos, p_neg)
    zfp_thr, zfp_fnr = zero_fp_threshold(p_pos, p_neg)

    report = {
        "counts": {"pos_files": len(pos_files), "neg_files": len(neg_files), "train_examples": int(X.shape[0])},
        "metrics": {"auc": auc, "eer": eer, "eer_threshold": eer_thr, "zero_fp_threshold": zfp_thr, "fnr_at_zero_fp": zfp_fnr},
        "note": "zero_fp_threshold garantit 0 FP sur TES négatifs, pas sur la vraie vie. Pour fiabilité réelle: ajouter des négatifs ambiants."
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    np.savez(args.out, w=w, b=b, mu=mu, sd=sd, threshold=zfp_thr)

    print("OK")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
