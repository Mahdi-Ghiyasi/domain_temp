#!/usr/bin/env python3
"""
Standalone script for training DAEL on Digit5 dataset
Converts Jupyter notebook operations to server-runnable Python
"""

import os
import sys
import subprocess
import shutil
import random
from pathlib import Path
import argparse
import os
import os.path as osp
import subprocess
import argparse
import numpy as np
from scipy.ndimage import convolve
from tqdm import tqdm
import h5py


def run_command(cmd, check=True, shell=True):
    """Execute shell command and handle errors"""
    print(f"[CMD] {cmd}")
    result = subprocess.run(cmd, shell=shell, check=False, 
                          capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {cmd}")
    return result

def write_file(filepath, content):
    """Write content to file, creating directories if needed"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"[FILE] Written: {filepath}")

    
def install_requirements(work_dir):
    """Install Python requirements"""
    print("\n📦 Installing requirements...")
    os.chdir(osp.join(work_dir, "Dassl.pytorch"))
    
    # First install requirements
    run_command("pip install -r requirements.txt")
    
    # Try modern editable install first, fall back if needed
    try:
        run_command("pip install -e . --no-build-isolation")
    except RuntimeError:
        print("⚠️  Modern install failed, trying legacy method...")
        try:
            run_command("python setup.py develop --no-deps")
        except RuntimeError:
            print("⚠️  Legacy install failed, trying direct install...")
            run_command("pip install .")
    
    print("✅ Requirements installed")


def create_output_dirs(base_dir, num_tries=20):
    """Create output directories"""
    print("\n=== Creating Output Directories ===")
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    for i in range(1, num_tries + 1):
        (output_dir / f"try{i}").mkdir(exist_ok=True)
    print(f"Created {num_tries} output directories")

def create_config_files(dassl_root):
    """Create all necessary config files"""
    print("\n=== Creating Config Files ===")
    
    # Dataset config
    dataset_config = """#!/usr/bin/env python3
INPUT:
  SIZE: (224,224)
  PIXEL_MEAN: [0.5, 0.5, 0.5]
  PIXEL_STD: [0.5, 0.5, 0.5]
  TRANSFORMS: ["normalize"]

DATASET:
  NAME: "Digit5"

MODEL:
  BACKBONE:
    NAME: "cnn_digit5_m3sda"
"""
    write_file(dassl_root / "configs/datasets/da/digit5.yaml", dataset_config)
    
    # Trainer config
    trainer_config = """#!/usr/bin/env python3
DATALOADER:
  TRAIN_X:
    SAMPLER: "RandomDomainSampler"
    BATCH_SIZE: 20
  TRAIN_U:
    SAME_AS_X: False
    BATCH_SIZE: 20
  TEST:
    BATCH_SIZE: 20

OPTIM:
  NAME: "sgd"
  LR: 0.05
  STEPSIZE: [1]
  MAX_EPOCH: 1
  LR_SCHEDULER: "cosine"

TRAINER:
  DAEL:
    STRONG_TRANSFORMS: ["randaugment2", "normalize"]
"""
    write_file(dassl_root / "configs/trainers/da/dael/digit5.yaml", trainer_config)

def create_backbone_file(dassl_root):
    """Create custom backbone file"""
    print("\n=== Creating Backbone File ===")
    
    backbone_code = """
import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

class FeatureExtractor(Backbone):
    def __init__(self):
        super().__init__()

        backbone = models.densenet121(weights=None)
        self.features = backbone.features  # convolutional part only
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(backbone.classifier.in_features, 2048)
        self.bn_fc = nn.BatchNorm1d(2048)
        self._out_features = 2048

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn_fc(self.fc(x)))
        return x

@BACKBONE_REGISTRY.register()
def cnn_digit5_m3sda(**kwargs):
    return FeatureExtractor()
"""
    write_file(dassl_root / "dassl/modeling/backbone/cnn_digit5_m3sda.py", backbone_code)

def create_dataset_file(dassl_root):
    """Create Digit5 (DomainNet) dataset loader using HDF5"""
    print("\n=== Creating Dataset File (HDF5 / DomainNet) ===")
    
    dataset_code = """#!/usr/bin/env python3
import os, os.path as osp, glob, math
import numpy as np
import h5py
from PIL import Image

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase

TRAIN_X_KEYS = ("train_data","train_x","X_train","X","images_train")
TRAIN_Y_KEYS = ("train_label","train_y","y_train","y","labels_train","labels","label")
TEST_X_KEYS  = ("test_data","test_x","X_test","X","images_test")
TEST_Y_KEYS  = ("test_label","test_y","y_test","y","labels_test","labels","label")

def _get_h5_paths(dataset_dir, domain_name):
    \"\"\"Return all .h5 files for a given domain (base or pseudo-labeled).\"\"\"
    direct = osp.join(dataset_dir, f"{domain_name}.h5")
    if osp.isfile(direct):
        return [direct]
    pattern = osp.join(dataset_dir, domain_name, "**", "*.h5")
    return glob.glob(pattern, recursive=True)

def _find_key(m, keys, purpose=""):
    for k in keys:
        if k in m:
            return k
    raise KeyError(f"None of {keys} found in file ({purpose})")

def _get_split_arrays(m, split):
    \"\"\"Return (X, y) where X is an h5py.Dataset, y is a NumPy array.\"\"\"
    if split == "train":
        xk = _find_key(m, TRAIN_X_KEYS, "train_x")
        yk = _find_key(m, TRAIN_Y_KEYS, "train_y")
    else:
        xk = _find_key(m, TEST_X_KEYS, "test_x")
        yk = _find_key(m, TEST_Y_KEYS, "test_y")
    X = m[xk]                  # h5py.Dataset, we access per-sample later
    y = m[yk][:]               # small enough to load fully
    if y.ndim == 2 and y.shape[0] == 1:
        y = y.flatten()
    return X, y

def _cache_images(cache_dir, dname, split, X, y, base_name):
    \"\"\"Materialize images to PNGs under _cache/ so Dassl can use path-based loading.\"\"\"
    n = X.shape[0]
    h, w = X.shape[1], X.shape[2]
    subdir = osp.join(cache_dir, dname, split, base_name)
    os.makedirs(subdir, exist_ok=True)
    pairs = []
    for i in range(10):
        arr = X[i]  # this triggers reading chunk i from the h5 dataset
        label = int(y[i]) if i < len(y) else -1
        fname = f"img_{i:05d}_L{label}.png"
        fpath = osp.join(subdir, fname)
        if not osp.exists(fpath):
            if arr.ndim == 2 or arr.shape[2] == 1:
                Image.fromarray(arr.astype(np.uint8), mode="L").save(fpath)
            else:
                Image.fromarray(arr.astype(np.uint8), mode="RGB").save(fpath)
        pairs.append((fpath, label))
    return pairs

@DATASET_REGISTRY.register()
class Digit5(DatasetBase):
    \"\"\"DomainNet-as-Digit5: reads .h5 (HDF5) instead of .mat.

    Expected base files under <root>/digit5:
      clipart.h5, infograph.h5, painting.h5, sketch.h5, real.h5, quickdraw.h5
    and any pseudo-labeled variants like clipart1r.h5, etc.
    \"\"\"
    dataset_dir = "digit5"
    domains = []

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        if not osp.isdir(self.dataset_dir):
            raise FileNotFoundError(f"Digit5 root not found: {self.dataset_dir}")

        self.cache_dir = osp.join(self.dataset_dir, "_cache")
        self.domains = sorted({*cfg.DATASET.SOURCE_DOMAINS, *cfg.DATASET.TARGET_DOMAINS})
        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS)

        train_x = self._read_split(cfg.DATASET.SOURCE_DOMAINS, split="train", tag="train_x")
        train_u = self._read_split(cfg.DATASET.TARGET_DOMAINS, split="train", tag="train_u")
        test    = self._read_split(cfg.DATASET.TARGET_DOMAINS, split="test",  tag="test")

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _read_split(self, domains, split, tag):
        all_items = []
        for domain_idx, dname in enumerate(domains):
            paths = _get_h5_paths(self.dataset_dir, dname)
            if not paths:
                raise FileNotFoundError(f"No .h5 files for domain '{dname}'. Expected {osp.join(self.dataset_dir, dname + '.h5')}.")
            total = 0
            for p in paths:
                base = osp.splitext(osp.basename(p))[0]
                with h5py.File(p, "r") as m:
                    X, y = _get_split_arrays(m, split)
                    pairs = _cache_images(self.cache_dir, dname, split, X, y, base)
                for impath, label in pairs:
                    all_items.append(Datum(
                        impath=impath,
                        label=int(label),
                        domain=domain_idx,
                        classname=str(int(label)) if label >= 0 else "-1"
                    ))
                total += len(pairs)
            print(f"[Digit5] {tag} domain='{dname}': {total} cached samples from {len(paths)} file(s).")
        return all_items
"""
    write_file(dassl_root / "dassl/data/datasets/da/digit5.py", dataset_code)


def create_pseudolabel_script(dassl_root):
    """Create pseudo-label generation script (writes .h5 with h5py)"""
    print("\n=== Creating Pseudolabel Script (HDF5) ===")
    
    script_code = """#!/usr/bin/env python3
import os, os.path as osp, argparse
import numpy as np
from PIL import Image
import h5py
import torch

from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.engine.trainer import TrainerBase
from dassl.utils import load_checkpoint

def _safe_load_model(self, directory, epoch=None):
    if not directory:
        print("Note that load_model() is skipped as no pretrained model is given")
        return
    names = self.get_model_names()
    model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
    for name in names:
        model_path = osp.join(directory, name, model_file)
        if not osp.exists(model_path):
            raise FileNotFoundError(f"No model at {model_path}")
        checkpoint = load_checkpoint(model_path)
        state_dict  = checkpoint["state_dict"]
        ep          = checkpoint.get("epoch")
        val_result  = checkpoint.get("val_result")
        vr = "None" if val_result is None else f"{val_result:.1f}"
        print(f"Load {model_path} to {name} (epoch={ep}, val_result={vr})")
        self._models[name].load_state_dict(state_dict)
TrainerBase.load_model = _safe_load_model

def _bilinear():
    try:
        return Image.Resampling.BILINEAR
    except AttributeError:
        return Image.BILINEAR

def build_trainer_from_args(args):
    cfg = get_cfg_default()
    cfg.merge_from_file(args.dataset_config)
    cfg.merge_from_file(args.trainer_config)
    cfg.DATASET.ROOT = args.root
    cfg.DATASET.SOURCE_DOMAINS = args.source
    cfg.DATASET.TARGET_DOMAINS = [args.target]
    cfg.OUTPUT_DIR = args.model_dir
    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = args.workers
    if args.trainer_name:
        cfg.TRAINER.NAME = args.trainer_name

    trainer = build_trainer(cfg)

    if args.load_epoch is not None:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
    else:
        trainer.load_model(args.model_dir)

    if hasattr(trainer, "set_model_mode"):
        trainer.set_model_mode("eval")
    if hasattr(trainer, "F"): trainer.F.eval()
    if hasattr(trainer, "E"): trainer.E.eval()
    if hasattr(trainer, "model"): trainer.model.eval()

    return trainer

def read_rgb_uint8(impath, img_size):
    im = Image.open(impath).convert("RGB").resize((img_size, img_size), _bilinear())
    return np.asarray(im, dtype=np.uint8)

@torch.inference_mode()
def predict_paths_labels(trainer, loader, dedup=True):
    device = getattr(trainer, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    seen = {}
    for batch in loader:
        imgs = batch["img"].to(device, non_blocking=True)
        logits_or_probs = trainer.model_inference(imgs)
        _, preds = logits_or_probs.max(1)
        for pth, y in zip(batch["impath"], preds.detach().cpu().tolist()):
            if not dedup or pth not in seen:
                seen[pth] = y
    paths = sorted(seen.keys())
    preds = [seen[p] for p in paths]
    return paths, preds

def extract_paths_labels_from_loader(loader, dedup=True):
    seen = {}
    for batch in loader:
        labels = batch["label"].cpu().tolist()
        for pth, y in zip(batch["impath"], labels):
            if not dedup or pth not in seen:
                seen[pth] = y
    paths = sorted(seen.keys())
    labels = [seen[p] for p in paths]
    return paths, labels

def build_arrays(paths, labels, img_size):
    X = np.stack([read_rgb_uint8(p, img_size) for p in paths], axis=0).astype(np.uint8)
    # int16 so DomainNet labels (0..344) are safe
    y = np.asarray(labels, dtype=np.int16).reshape(1, -1)
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out-mat", required=True, help="Output .h5 path (name kept for backward-compat)")
    parser.add_argument("--source", nargs="+", required=True)
    parser.add_argument("--load-epoch", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dedup-train", action="store_true")
    parser.add_argument("--save-paths", action="store_true")
    parser.add_argument("--trainer-name", type=str, default=None)
    parser.add_argument("--dataset-config", default="configs/datasets/da/digit5.yaml")
    parser.add_argument("--trainer-config", default="configs/trainers/da/dael/digit5.yaml")
    args = parser.parse_args()

    out_path = osp.join(args.root, args.out_mat) if not osp.isabs(args.out_mat) else args.out_mat
    os.makedirs(osp.dirname(out_path), exist_ok=True)

    trainer = build_trainer_from_args(args)
    loader_train_u = trainer.train_loader_u
    loader_test    = trainer.test_loader

    print("Generating pseudo-labels for training data...")
    train_paths, train_preds = predict_paths_labels(trainer, loader_train_u, dedup=args.dedup_train)

    print("Extracting ground truth labels for test data...")
    test_paths, test_labels = extract_paths_labels_from_loader(loader_test, dedup=True)

    train_data, train_label = build_arrays(train_paths, train_preds, img_size=args.img_size)
    test_data,  test_label  = build_arrays(test_paths,  test_labels, img_size=args.img_size)

    print("train_data:",  train_data.shape, train_data.dtype)
    print("train_label:", train_label.shape, train_label.dtype)
    print("test_data:",   test_data.shape,  test_data.dtype)
    print("test_label:",  test_label.shape, test_label.dtype)

    # Write as HDF5
    print("Writing HDF5 to:", out_path)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("train_data",  data=train_data,  dtype="uint8")
        f.create_dataset("train_label", data=train_label, dtype="int16")
        f.create_dataset("test_data",   data=test_data,   dtype="uint8")
        f.create_dataset("test_label",  data=test_label,  dtype="int16")

        if args.save_paths:
            train_paths_ds = f.create_dataset("train_impaths", shape=(1, len(train_paths)), dtype=str_dtype)
            test_paths_ds  = f.create_dataset("test_impaths",  shape=(1, len(test_paths)),  dtype=str_dtype)
            train_paths_ds[0, :] = np.asarray(train_paths, dtype=object)
            test_paths_ds[0, :]  = np.asarray(test_paths,  dtype=object)

    print("Saved ->", out_path)

if __name__ == "__main__":
    main()
"""
    write_file(dassl_root / "tools/pseudolabel_to_mat.py", script_code)

def get_motion_blur_kernel(size, angle=0):
    """Same as before, but float32 for speed."""
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    angle_rad = np.deg2rad(angle)
    for i in range(size):
        offset = i - center
        x = int(center + offset * np.cos(angle_rad))
        y = int(center + offset * np.sin(angle_rad))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0

    s = kernel.sum()
    if s > 0:
        kernel /= s
    return kernel


def _apply_motion_blur_batch(batch, kernel, angle, mode="reflect", cval=0.0):
    """
    Apply motion blur to a batch of images.

    batch:  float32 array of shape (B,H,W) or (B,H,W,3)
    kernel: 2D float32 kernel (size,size)
    """
    # Build ND kernel matching batch dimensions but only acting on H,W
    if batch.ndim == 3:
        # (B,H,W) -> kernel shape (1,k,k)
        k_nd = kernel[np.newaxis, :, :]
    elif batch.ndim == 4:
        # (B,H,W,C) -> kernel shape (1,k,k,1)
        k_nd = kernel[np.newaxis, :, :, np.newaxis]
    else:
        raise ValueError(f"Unsupported batch ndim {batch.ndim}")

    blurred = convolve(batch, k_nd, mode=mode, cval=cval)
    return blurred

def motion_blur_dataset_h5(
    input_h5_path,
    output_h5_path,
    kernel_size_train=11,
    kernel_size_test=11,
    angle_train=45,
    angle_test=45,
    mode="reflect",
    cval=0.0,
    batch_size=256,
):
    """
    Fast batched version of motion blur for HDF5 datasets.

    Expects HDF5 layout:
        train_data  : (N_train, H, W, C) or (N_train, H, W)
        train_label : (1, N_train) or (N_train,)
        test_data   : (N_test,  H, W, C) or (N_test,  H, W)
        test_label  : (1, N_test) or (N_test,)
    """
    print(f"Loading HDF5: {input_h5_path}")
    k_train = get_motion_blur_kernel(kernel_size_train, angle_train)
    k_test  = get_motion_blur_kernel(kernel_size_test,  angle_test)

    with h5py.File(input_h5_path, "r") as src, h5py.File(output_h5_path, "w") as dst:
        # Copy file-level attributes
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        # Check required datasets
        if "train_data" not in src or "test_data" not in src:
            raise KeyError("Input file must contain 'train_data' and 'test_data' datasets")

        train_src = src["train_data"]
        test_src  = src["test_data"]

        train_shape, train_dtype = train_src.shape, train_src.dtype
        test_shape,  test_dtype  = test_src.shape,  test_src.dtype

        print(f"Train data: {train_shape}, dtype: {train_dtype}")
        print(f"Test  data: {test_shape},  dtype: {test_dtype}")

        # Create destination datasets (same shape/dtype)
        train_dst = dst.create_dataset("train_data", shape=train_shape, dtype=train_dtype)
        test_dst  = dst.create_dataset("test_data",  shape=test_shape,  dtype=test_dtype)

        # Copy labels directly
        for lbl_name in ["train_label", "test_label"]:
            if lbl_name in src:
                dst.create_dataset(lbl_name, data=src[lbl_name][...])
            else:
                print(f"Warning: '{lbl_name}' not found; skipping.")

        # Copy any other datasets unchanged (e.g. train_impaths, test_impaths)
        for name in src.keys():
            if name in ["train_data", "test_data", "train_label", "test_label"]:
                continue
            src.copy(name, dst)

        # ---- Process train_data in batches ----
        n_train = train_shape[0]
        print(f"\nApplying motion blur to TRAIN (kernel={kernel_size_train}, angle={angle_train}°, batch={batch_size})")
        for start in tqdm(range(0, n_train, batch_size), desc="Train"):
            end = min(n_train, start + batch_size)
            # (B, ...) slice
            batch = train_src[start:end].astype(np.float32)
            blurred = _apply_motion_blur_batch(batch, k_train, angle_train, mode=mode, cval=cval)
            # clip & cast back
            np.clip(blurred, 0, 255, out=blurred)
            train_dst[start:end] = blurred.astype(train_dtype)

        # ---- Process test_data in batches ----
        n_test = test_shape[0]
        print(f"\nApplying motion blur to TEST (kernel={kernel_size_test}, angle={angle_test}°, batch={batch_size})")
        for start in tqdm(range(0, n_test, batch_size), desc="Test"):
            end = min(n_test, start + batch_size)
            batch = test_src[start:end].astype(np.float32)
            blurred = _apply_motion_blur_batch(batch, k_test, angle_test, mode=mode, cval=cval)
            np.clip(blurred, 0, 255, out=blurred)
            test_dst[start:end] = blurred.astype(test_dtype)

    print(f"\nSaved blurred dataset to: {output_h5_path}")
    print("✅ Motion blur applied (fast batched version).")
    return output_h5_path

def train_model(work_dir, dassl_root, trainer, source_domains, target_domain, output_dir):
    """Run training"""
    os.chdir(dassl_root)
    cmd="nvidia-smi"
    run_command(cmd)
    cmd = f"""CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--trainer {trainer} \
--root {work_dir} \
--source-domains {' '.join(source_domains)} \
--target-domains {target_domain} \
--dataset-config-file configs/datasets/da/digit5.yaml \
--config-file configs/trainers/da/dael/digit5.yaml \
--output-dir {output_dir}"""
    run_command(cmd)

def generate_pseudolabels(work_dir, dassl_root, model_dir, target, out_mat, source_domains, load_epoch):
    """Generate pseudo-labels"""
    os.chdir(dassl_root)
    cmd = f"""CUDA_VISIBLE_DEVICES=0 python tools/pseudolabel_to_mat.py \
--trainer-name DAEL \
--root {work_dir} \
--model-dir {model_dir} \
--target {target} \
--out-mat {out_mat} \
--load-epoch {load_epoch} \
--source {' '.join(source_domains)}"""
    run_command(cmd)

def cleanup(work_dir, patterns):
    """Clean up files/directories"""
    for pattern in patterns:
        path = work_dir / pattern
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"[CLEANUP] Removed: {path}")

import shutil

def split_dataset_mat_in_half(
    in_path: str,
    out_path_a: str,
    out_path_b: str,
    *,
    shuffle: bool = True,
    stratify: bool = False,
    random_state: int = 42,
):
    """
    Clone a dataset .h5 file to two identical copies and save both .h5 files.

    NOTE: Still a clone, not an actual half-split. We rely on h5py.copy so we
    don't have to read the whole file into RAM.
    """
    # We ignore shuffle/stratify/random_state (kept only for API compatibility)
    with h5py.File(in_path, "r") as src, \
         h5py.File(out_path_a, "w") as fa, \
         h5py.File(out_path_b, "w") as fb:

        # copy all groups/datasets
        for name in src:
            src.copy(name, fa)
            src.copy(name, fb)

        # copy file-level attributes if any
        for k, v in src.attrs.items():
            fa.attrs[k] = v
            fb.attrs[k] = v

    print(f"✅ Cloned {in_path} → {out_path_a}, {out_path_b}")


def self_adapt_domain(work_dir, dassl_root, domain_name, iteration, current_try):
    """
    Self-adapt a single domain by:
    1. Splitting its pseudo-labeled version into 1a and 1b
    2. Training on both halves targeting the base version
    3. Generating new pseudo-labels
    4. Cleaning up intermediate files
    """
    print(f"\n{'─'*60}")
    print(f"🔄 SELF-ADAPTING: {domain_name}")
    print(f"{'─'*60}")
    
    digit5_dir = work_dir / "digit5"
    base_mat = f"{domain_name}1r.h5"
    split_a = f"{domain_name}1a.h5"
    split_b = f"{domain_name}1b.h5"
    target_mat = f"{domain_name}2.h5"
    output_mat = f"{domain_name}2r.h5"
    
    # Step 1: Remove cache
    cache_dir = digit5_dir / "_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"🗑️ Removed cache directory")
    
    # Step 2: Split into 1a and 1b
    print(f"\n📂 Splitting {base_mat} into halves...")
    split_dataset_mat_in_half(
        str(digit5_dir / base_mat),
        str(digit5_dir / split_a),
        str(digit5_dir / split_b),
        shuffle=True,
        stratify=True,
        random_state=123,
    )
    
    # Step 3: Train on both halves
    output_dir = f"output/try{current_try}"
    print(f"\n🚀 Training on {split_a} and {split_b} → {target_mat}")
    
    try:
        train_model(
            work_dir=work_dir,
            dassl_root=dassl_root,
            trainer="DAEL",
            source_domains=[f"{domain_name}1a", f"{domain_name}1b"],
            target_domain=f"{domain_name}2",
            output_dir=output_dir
        )
        
        # Step 4: Generate pseudo-labels for the adapted version
        print(f"\n🏷️ Generating pseudo-labels → {output_mat}")
        generate_pseudolabels(
            work_dir=work_dir,
            dassl_root=dassl_root,
            model_dir=output_dir,
            target=f"{domain_name}2",
            out_mat=str(digit5_dir / output_mat),
            source_domains=[f"{domain_name}1a", f"{domain_name}1b"],
            load_epoch=1
        )
        
        # Step 5: Cleanup intermediate files
        print(f"\n🗑️ Cleaning up intermediate files...")
        cleanup_files = [
            digit5_dir / f"{domain_name}1.h5",
            digit5_dir / base_mat,
            digit5_dir / split_a,
            digit5_dir / split_b,
        ]
        
        for f in cleanup_files:
            if f.exists():
                f.unlink()
                print(f"  ✓ Deleted {f.name}")
        
        # Also clean the output directory
        output_path = work_dir / output_dir
        if output_path.exists():
            shutil.rmtree(output_path)
            print(f"  ✓ Deleted {output_dir}/")
        
        print(f"✅ Self-adaptation complete for {domain_name}")
        return True
        
    except Exception as e:
        print(f"❌ Self-adaptation failed for {domain_name}: {e}")
        return False
import torch        
def benchmark_gpu():
    device = torch.device('cuda:0')
    
    print("=== GPU Information ===")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print("\n=== Memory Before Allocation ===")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Try to allocate memory in chunks
    print("\n=== Testing Memory Allocation ===")
    tensors = []
    chunk_size = 100  # MB
    
    try:
        for i in range(80):  # Try to allocate 8 GB
            tensor = torch.randn(chunk_size * 1024 * 256, device=device)  # ~100 MB
            tensors.append(tensor)
            if i % 10 == 0:
                print(f"Allocated {(i+1) * chunk_size} MB")
                print(f"  Current: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    except RuntimeError as e:
        print(f"\n!!! OOM at {len(tensors) * chunk_size} MB")
        print(f"Error: {e}")
    
    print("\n=== Final Memory Stats ===")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Peak: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Cleanup
    del tensors
    torch.cuda.empty_cache()
def main():
    parser = argparse.ArgumentParser(description="Train DAEL on Digit5 dataset")
    parser.add_argument("--work-dir", type=str, default="/",
                       help="Working directory")
    parser.add_argument("--skip-install", action="store_true",
                       help="Skip installation of requirements")
    parser.add_argument("--skip-clone", action="store_true",
                       help="Skip cloning Dassl.pytorch repo")
    parser.add_argument("--num-iterations", type=int, default=5,
                       help="Number of training iterations per domain")
    parser.add_argument("--max-epoch", type=int, default=1,
                       help="Max epochs per training run")
    args = parser.parse_args()
    benchmark_gpu()
    work_dir = Path(args.work_dir)
    dassl_root = work_dir / "Dassl.pytorch"
    
    print(f"\n{'='*80}")
    print("DAEL TRAINING PIPELINE - COMPLETE WORKFLOW WITH SELF-ADAPTATION")
    print(f"{'='*80}\n")
    print(f"Working directory: {work_dir}")
    print(f"Dassl root: {dassl_root}")
    print(f"Number of iterations: {args.num_iterations}")
    
    # ===== SETUP PHASE =====
    print(f"\n{'='*80}")
    print("PHASE 1: SETUP")
    print(f"{'='*80}")
    
    if not args.skip_clone:
        if not dassl_root.exists():
            print("\n[1.1] Cloning Dassl.pytorch repository...")
            os.chdir(work_dir)
            run_command("git clone https://github.com/KaiyangZhou/Dassl.pytorch")
        else:
            print("\n[1.1] Dassl.pytorch already exists, skipping clone")
    
    if not args.skip_install:
        print("\n[1.2] Installing requirements...")
        install_requirements(work_dir)
    else:
        print("\n[1.2] Skipping installation (--skip-install)")
    
    print("\n[1.3] Creating output directories...")
    create_output_dirs(work_dir, num_tries=20)
    
    print("\n[1.4] Creating configuration files...")
    create_config_files(dassl_root)
    
    print("\n[1.5] Creating custom backbone...")
    create_backbone_file(dassl_root)
    
    print("\n[1.6] Creating Digit5 dataset loader...")
    create_dataset_file(dassl_root)
    
    print("\n[1.7] Creating pseudolabel generation script...")
    create_pseudolabel_script(dassl_root)
    # ===== INITIAL TRAINING PHASE =====
    print(f"\n{'='*80}")
    print("PHASE 2: INITIAL TRAINING ON clipart")
    print(f"{'='*80}")
    work_dir = Path(args.work_dir)
    dassl_root = work_dir / "Dassl.pytorch"
    print(f"\n{'='*80}")
    print("DAEL TRAINING PIPELINE - EXACT WORKFLOW")
    print(f"{'='*80}\n")
    
    os.chdir(dassl_root)
    run_command("nvidia-smi")

    # ===== STEP 1: Initial Training =====
    print("\n[STEP 1] Training initial model with clipart as target...")
    run_command(
        "CUDA_VISIBLE_DEVICES=0 python tools/train.py "
        "--trainer DAEL "
        f"--root {work_dir} "
        "--source-domains infograph painting quickdraw sketch real "
        "--target-domains clipart "
        "--dataset-config-file configs/datasets/da/digit5.yaml "
        "--config-file configs/trainers/da/dael/digit5.yaml "
        "--output-dir output/try1"
    )
    
    # ===== STEP 2: Generate pseudo-labels for clipart =====
    print("\n[STEP 2] Generating pseudo-labels for clipart...")
    run_command(
        "python tools/pseudolabel_to_mat.py "
        f"--root {work_dir} "
        "--model-dir output/try1 "
        "--target clipart "
        "--out-mat digit5/clipart0p.h5 "
        "--load-epoch 1 "
        "--trainer-name DAEL "
        "--source infograph sketch painting quickdraw real"
    )
    
    # ===== STEP 3: Self-adapt each domain (infograph, sketch, painting, quickdraw, real) =====
    for domain in ["infograph", "sketch", "painting", "quickdraw", "real"]:
        run_command("nvidia-smi")
        print(f"\n[STEP 3.{domain}] Self-adapting {domain}...")
        
        # Clean cache
        cache_dir = work_dir / "digit5" / "_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        
        # Split dataset
        print(f"  Splitting {domain}.h5 into halves...")
        split_dataset_mat_in_half(
            str(work_dir / "digit5" / f"{domain}.h5"),
            str(work_dir / "digit5" / f"{domain}a.h5"),
            str(work_dir / "digit5" / f"{domain}b.h5"),
            shuffle=True,
            stratify=True,
            random_state=123,
        )

        motion_blur_dataset_h5(
            str(work_dir / "digit5" / f"{domain}.h5"),
            str(work_dir / "digit5" / f"{domain}1.h5"),
            kernel_size_train=13,   # light blur
            kernel_size_test=13,
            angle_train=45,
            angle_test=45,
        )
        
        # Train - UNIQUE OUTPUT DIR FOR EACH DOMAIN
        print(f"  Training on {domain}...")
        run_command(
            "CUDA_VISIBLE_DEVICES=0 python tools/train.py "
            "--trainer DAEL "
            f"--root {work_dir} "
            f"--source-domains {domain}a {domain}b "
            f"--target-domains {domain}1 "
            "--dataset-config-file configs/datasets/da/digit5.yaml "
            "--config-file configs/trainers/da/dael/digit5.yaml "
            f"--output-dir output/self_adapt_r1_{domain}"
        )
        
        # Generate pseudo-labels
        print(f"  Generating pseudo-labels for {domain}...")
        run_command(
            "python tools/pseudolabel_to_mat.py "
            f"--root {work_dir} "
            f"--model-dir output/self_adapt_r1_{domain} "
            f"--target {domain}1 "
            f"--out-mat digit5/{domain}1r.h5 "
            "--load-epoch 1 "
            "--trainer-name DAEL "
            f"--source {domain}a {domain}b"
        )
        
        # Cleanup
        print(f"  Cleaning up {domain} intermediate files...")
        for f in [f"{domain}a.h5", f"{domain}b.h5"]:
            file_path = work_dir / "digit5" / f
            if file_path.exists():
                file_path.unlink()
    
    # ===== STEP 4: Train clipart with round 1 pseudo-labels =====
    print("\n[STEP 4] Training clipart with round 1 pseudo-labels...")

    motion_blur_dataset_h5(
            str(work_dir / "digit5" / "clipart.h5"),
            str(work_dir / "digit5" / "clipart1.h5"),
            kernel_size_train=13,   # light blur
            kernel_size_test=13,
            angle_train=45,
            angle_test=45,
        )
    run_command(
        "CUDA_VISIBLE_DEVICES=0 python tools/train.py "
        "--trainer DAEL "
        f"--root {work_dir} "
        "--source-domains infograph1r sketch1r painting1r quickdraw1r clipart0p real1r "
        "--target-domains clipart1 "
        "--dataset-config-file configs/datasets/da/digit5.yaml "
        "--config-file configs/trainers/da/dael/digit5.yaml "
        "--output-dir output/try2"
    )
    
    run_command(
        "python tools/pseudolabel_to_mat.py "
        f"--root {work_dir} "
        "--model-dir output/try2 "
        "--target clipart1 "
        "--out-mat digit5/clipart1p.h5 "
        "--load-epoch 1 "
        "--trainer-name DAEL "
        "--source infograph1r sketch1r clipart0p painting1r quickdraw1r real1r"
    )
    
    # Delete clipart0p.mat
    (work_dir / "digit5" / "clipart0p.h5").unlink(missing_ok=True)
    
    # ===== STEP 5: Second round self-adaptation =====
    for domain in ["infograph", "sketch", "painting", "quickdraw", "real"]:
        print(f"\n[STEP 5.{domain}] Second round self-adaptation for {domain}...")
        
        cache_dir = work_dir / "digit5" / "_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        
        split_dataset_mat_in_half(
            str(work_dir / "digit5" / f"{domain}1r.h5"),
            str(work_dir / "digit5" / f"{domain}1a.h5"),
            str(work_dir / "digit5" / f"{domain}1b.h5"),
            shuffle=True,
            stratify=True,
            random_state=123,
        )

        motion_blur_dataset_h5(
            str(work_dir / "digit5" / f"{domain}.h5"),
            str(work_dir / "digit5" / f"{domain}2.h5"),
            kernel_size_train=35,   # light blur
            kernel_size_test=35,
            angle_train=45,
            angle_test=45,
        )
        
        run_command(
            "CUDA_VISIBLE_DEVICES=0 python tools/train.py "
            "--trainer DAEL "
            f"--root {work_dir} "
            f"--source-domains {domain}1a {domain}1b "
            f"--target-domains {domain}2 "
            "--dataset-config-file configs/datasets/da/digit5.yaml "
            "--config-file configs/trainers/da/dael/digit5.yaml "
            f"--output-dir output/self_adapt_r2_{domain}"
        )
        
        run_command(
            "python tools/pseudolabel_to_mat.py "
            f"--root {work_dir} "
            f"--model-dir output/self_adapt_r2_{domain} "
            f"--target {domain}2 "
            f"--out-mat digit5/{domain}2r.h5 "
            "--load-epoch 1 "
            "--trainer-name DAEL "
            f"--source {domain}1a {domain}1b"
        )
        
        # Cleanup
        for f in [f"{domain}1r.h5", f"{domain}1a.h5", f"{domain}1b.h5"]:
            (work_dir / "digit5" / f).unlink(missing_ok=True)
    
    # ===== STEP 6: Final clipart training (round 2) =====
    print("\n[STEP 6] Final clipart training with round 2 pseudo-labels...")
    motion_blur_dataset_h5(
            str(work_dir / "digit5" / "clipart.h5"),
            str(work_dir / "digit5" / "clipart2.h5"),
            kernel_size_train=35,   # light blur
            kernel_size_test=35,
            angle_train=45,
            angle_test=45,
        )
    run_command(
        "CUDA_VISIBLE_DEVICES=0 python tools/train.py "
        "--trainer DAEL "
        f"--root {work_dir} "
        "--source-domains infograph2r sketch2r painting2r quickdraw2r clipart1p real2r "
        "--target-domains clipart2 "
        "--dataset-config-file configs/datasets/da/digit5.yaml "
        "--config-file configs/trainers/da/dael/digit5.yaml "
        "--output-dir output/try3"
    )
    
    print("\n[STEP 7] Final training without clipart pseudo-labels...")
    run_command(
        "CUDA_VISIBLE_DEVICES=0 python tools/train.py "
        "--trainer DAEL "
        f"--root {work_dir} "
        "--source-domains infograph2r sketch2r painting2r quickdraw2r real2r "
        "--target-domains clipart2 "
        "--dataset-config-file configs/datasets/da/digit5.yaml "
        "--config-file configs/trainers/da/dael/digit5.yaml "
        "--output-dir output/try6"
    )
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()

