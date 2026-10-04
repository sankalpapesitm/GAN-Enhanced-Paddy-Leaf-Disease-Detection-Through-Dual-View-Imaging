"""
Paddy Leaf Disease Classification - Dual-View (T1 + B1) EfficientNet-B0
Local version (no Google Colab needed).

USAGE
-----
1) Install deps:
     pip install torch torchvision scikit-learn seaborn tqdm matplotlib pillow pandas numpy

2) Train + evaluate:
     python paddy_dualview_local.py --data "C:/path/to/Segregation final1.zip"
     python paddy_dualview_local.py --data "C:/path/to/dataset_folder"
     python paddy_dualview_local.py --data "..." --test "C:/path/to/test_folder"

3) Diagnose one leaf later (after training):
     python paddy_dualview_local.py --predict_top leaf_T1.jpg --predict_bottom leaf_B1.jpg

Dataset layout expected (inside the zip/folder):
     <root>/Brown Spot/xxx_T1.jpg, xxx_B1.jpg ...
     <root>/Healthy Paddy/...
     <root>/Hispa/...
     <root>/Leaf Blast/...
"""

import os
import re
import json
import time
import shutil
import random
import zipfile
import argparse
from pathlib import Path
from collections import defaultdict

# Fix Windows DLL loading order for PyTorch (must be imported before sklearn / matplotlib)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support)

# =============================================================================
# CONFIG
# =============================================================================
WORK_DIR        = Path("./paddy_work")          # everything is stored here
EXTRACT_DIR     = WORK_DIR / "dataset_raw"
SPLIT_DIR       = WORK_DIR / "Paddy_Split"
BEST_MODEL_PATH = WORK_DIR / "best_dual_efficientnet_b0.pth"
FINAL_MODEL_PATH = WORK_DIR / "Dual_EfficientNet_B0_Paddy_Disease.pth"
CLASS_JSON_PATH = WORK_DIR / "class_to_idx.json"

TARGET_CLASSES  = ["Brown Spot", "Healthy Paddy", "Hispa", "Leaf Blast"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SEED        = 42
TRAIN_RATIO = 0.80
IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_WORKERS = 2            # set to 0 if you get DataLoader errors on Windows
NUM_EPOCHS  = 20
LR          = 1e-4
WEIGHT_DECAY = 1e-4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# HELPERS: leaf-id / top-view detection
# =============================================================================
def extract_leaf_id(filepath: Path) -> str:
    """Strip T1/B1/D1/front/back etc. so both views of a leaf share one ID."""
    stem = filepath.stem
    clean = re.sub(
        r'([_-]?(t1|b1|d1|front|back|top|bottom|down|dorsal|ventral|adaxial|abaxial)|[_-][tbd])$',
        '', stem, flags=re.IGNORECASE)
    clean = re.sub(
        r'^((t1|b1|d1|front|back|top|bottom|down|dorsal|ventral|adaxial|abaxial)[_-]|[tbd][_-])',
        '', clean, flags=re.IGNORECASE)
    clean = clean.strip(' _-')
    return clean if clean else stem


def is_top_image(filepath: Path) -> bool:
    stem = filepath.stem.lower()
    return bool(re.search(r'([_-]t1?$|_t$|front|top|adaxial|^t1?[_-])', stem))


def pick_pair(files):
    """Return (top_path, bottom_path) from a list of files of one leaf."""
    if len(files) >= 2:
        tops = [f for f in files if is_top_image(f)]
        bottoms = [f for f in files if not is_top_image(f)]
        if tops and bottoms:
            return tops[0], bottoms[0]
        s = sorted(files)
        return s[0], s[1]
    return files[0], files[0]


# =============================================================================
# STEP 1: dataset extraction + class detection + integrity check
# =============================================================================
def prepare_dataset(data_arg: str):
    data_path = Path(data_arg)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    if data_path.is_file() and data_path.suffix.lower() == ".zip":
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        if not any(EXTRACT_DIR.iterdir()):
            print(f"Extracting {data_path} -> {EXTRACT_DIR} ...")
            with zipfile.ZipFile(data_path, "r") as z:
                z.extractall(EXTRACT_DIR)
        else:
            print(f"Using already-extracted data in {EXTRACT_DIR}")
        search_root = EXTRACT_DIR
    else:
        search_root = data_path

    expected = {c.lower().replace(" ", "").replace("_", "") for c in TARGET_CLASSES}
    found = None
    for root, dirs, _ in os.walk(search_root):
        norm = {d.lower().replace(" ", "").replace("_", ""): d for d in dirs}
        if len(expected.intersection(norm.keys())) >= 3:
            found = root
            break
    if found is None:
        raise RuntimeError("Could not locate the class folders "
                           f"{TARGET_CLASSES} inside {search_root}")

    dataset_path = Path(found)
    existing = {f.lower().strip().replace("_", " "): f
                for f in os.listdir(dataset_path) if (dataset_path / f).is_dir()}
    classes = []
    for t in TARGET_CLASSES:
        key = t.lower().strip().replace("_", " ")
        if key in existing:
            classes.append(existing[key])
    classes = sorted(set(classes))

    print("=" * 55)
    print(f"DATASET_PATH : {dataset_path}")
    for i, c in enumerate(classes):
        print(f"  [{i}] {c}")
    print("=" * 55)

    # count + integrity
    bad = set()
    total = 0
    print("Verifying images...")
    for cls in classes:
        files = [f for f in (dataset_path / cls).rglob("*")
                 if f.suffix.lower() in IMAGE_EXTENSIONS]
        print(f"{cls:<22} | {len(files)}")
        total += len(files)
        for f in files:
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:
                    im.load()
            except Exception:
                bad.add(str(f))
    print(f"TOTAL images: {total} | corrupted: {len(bad)}")
    return dataset_path, classes, bad


# =============================================================================
# STEP 2: leak-free pair-aware split
# =============================================================================
def split_dataset(dataset_path, classes, bad_paths):
    if SPLIT_DIR.exists():
        try:
            shutil.rmtree(SPLIT_DIR, ignore_errors=True)
        except Exception as e:
            print(f"Warning resetting {SPLIT_DIR}: {e}")
    rng = random.Random(SEED)
    summary = defaultdict(lambda: {"train": 0, "val": 0, "unique_leaves": 0})

    for cls in classes:
        files = [f for f in (dataset_path / cls).rglob("*")
                 if f.suffix.lower() in IMAGE_EXTENSIONS and str(f) not in bad_paths]
        groups = defaultdict(list)
        for f in files:
            groups[extract_leaf_id(f)].append(f)

        leaves = sorted(groups.keys())
        rng.shuffle(leaves)
        n_train = int(round(len(leaves) * TRAIN_RATIO))
        summary[cls]["unique_leaves"] = len(leaves)

        for split_name, assigned in (("train", leaves[:n_train]), ("val", leaves[n_train:])):
            target = SPLIT_DIR / split_name / cls
            target.mkdir(parents=True, exist_ok=True)
            for lid in assigned:
                for p in groups[lid]:
                    shutil.copy2(p, target / p.name)
                    summary[cls][split_name] += 1

    print("=" * 65)
    print(f"{'CLASS':<20} | {'LEAVES':<8} | {'TRAIN':<8} | {'VAL':<8}")
    print("=" * 65)
    for cls in classes:
        s = summary[cls]
        print(f"{cls:<20} | {s['unique_leaves']:<8} | {s['train']:<8} | {s['val']:<8}")

    # leakage test (per class, since ids are grouped per class)
    leaks = 0
    for cls in classes:
        tr = {extract_leaf_id(f) for f in (SPLIT_DIR / "train" / cls).rglob("*")
              if f.suffix.lower() in IMAGE_EXTENSIONS}
        va = {extract_leaf_id(f) for f in (SPLIT_DIR / "val" / cls).rglob("*")
              if f.suffix.lower() in IMAGE_EXTENSIONS}
        leaks += len(tr & va)
    print(f"Specimen overlap Train AND Val: {leaks} (expected 0)")


# =============================================================================
# STEP 3: dataset / model
# =============================================================================
class DualViewPaddyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []

        for cls_name in self.classes:
            files = [f for f in (self.root_dir / cls_name).rglob("*")
                     if f.suffix.lower() in IMAGE_EXTENSIONS]
            groups = defaultdict(list)
            for f in files:
                groups[extract_leaf_id(f)].append(f)
            for leaf_id, fl in groups.items():
                top, bottom = pick_pair(fl)
                self.samples.append((top, bottom, self.class_to_idx[cls_name], leaf_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        top_p, bot_p, label, _ = self.samples[idx]
        top = Image.open(top_p).convert("RGB")
        bot = Image.open(bot_p).convert("RGB")
        if self.transform:
            top, bot = self.transform(top), self.transform(bot)
        return (top, bot), label


class DualViewEfficientNetB0(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, dropout_rate=0.3):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = efficientnet_b0(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        n_feat = base.classifier[1].in_features  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(n_feat * 2, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(512, num_classes),
        )

    def extract_features(self, x):
        return torch.flatten(self.avgpool(self.features(x)), 1)

    def forward(self, img_top, img_bottom=None):
        if img_bottom is None:
            img_bottom = img_top
        fused = torch.cat([self.extract_features(img_top),
                           self.extract_features(img_bottom)], dim=1)
        return self.classifier(fused)


# =============================================================================
# STEP 4: training
# =============================================================================
def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    tr_losses, va_losses, tr_accs, va_accs = [], [], [], []
    best_val_acc = 0.0
    start = time.time()
    print(f"Training on {device} for {NUM_EPOCHS} epochs...\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for (top, bot), labels in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS} [Train]", leave=False):
            top, bot, labels = top.to(device), bot.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(top, bot)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * labels.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        tr_loss, tr_acc = run_loss / total, 100.0 * correct / total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for (top, bot), labels in val_loader:
                top, bot, labels = top.to(device), bot.to(device), labels.to(device)
                out = model(top, bot)
                v_loss += criterion(out, labels).item() * labels.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total += labels.size(0)
        va_loss, va_acc = v_loss / v_total, 100.0 * v_correct / v_total

        scheduler.step(va_acc)
        lr = optimizer.param_groups[0]["lr"]
        tr_losses.append(tr_loss); va_losses.append(va_loss)
        tr_accs.append(tr_acc);   va_accs.append(va_acc)

        msg = (f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | Train Loss {tr_loss:.4f} Acc {tr_acc:.2f}% | "
               f"Val Loss {va_loss:.4f} Acc {va_acc:.2f}% | LR {lr:.6f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            msg += f"  -> best saved ({best_val_acc:.2f}%)"
        print(msg)

    print(f"\nTraining done in {(time.time()-start)/60:.2f} min | best val acc {best_val_acc:.2f}%")
    return tr_losses, va_losses, tr_accs, va_accs


def plot_curves(tr_losses, va_losses, tr_accs, va_accs):
    ep = range(1, len(tr_accs) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
    a1.plot(ep, tr_accs, "o-", label="Train Acc"); a1.plot(ep, va_accs, "s-", label="Val Acc")
    a1.set_title("Accuracy"); a1.set_xlabel("Epoch"); a1.set_ylabel("%"); a1.grid(True, ls="--", alpha=.5); a1.legend()
    a2.plot(ep, tr_losses, "o-", label="Train Loss"); a2.plot(ep, va_losses, "s-", label="Val Loss")
    a2.set_title("Loss"); a2.set_xlabel("Epoch"); a2.set_ylabel("CE Loss"); a2.grid(True, ls="--", alpha=.5); a2.legend()
    plt.tight_layout()
    plt.savefig(WORK_DIR / "training_curves.png", dpi=300)
    plt.savefig(WORK_DIR / "accuracy_curve.png", dpi=300)
    plt.savefig(WORK_DIR / "loss_curve.png", dpi=300)
    plt.show()


# =============================================================================
# STEP 5: testing on a folder (T1/B1 pairs)
# =============================================================================
def evaluate_folder(model, test_dir, classes):
    test_path = Path(test_dir)
    files = [f for f in test_path.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"Test images found: {len(files)}")

    groups = defaultdict(lambda: {"files": [], "true_class": None, "leaf_id": None})
    for f in files:
        lid = extract_leaf_id(f)
        parent = f.parent.name.lower().replace(" ", "").replace("_", "")
        matched = next((c for c in classes
                        if c.lower().replace(" ", "").replace("_", "") == parent), None)
        key = f"{matched or 'UNKNOWN'}_{lid}"
        groups[key]["files"].append(f)
        groups[key]["true_class"] = matched
        groups[key]["leaf_id"] = lid

    rows = []
    model.eval()
    with torch.no_grad():
        for g in groups.values():
            top_p, bot_p = pick_pair(g["files"])
            t = eval_transform(Image.open(top_p).convert("RGB")).unsqueeze(0).to(device)
            b = eval_transform(Image.open(bot_p).convert("RGB")).unsqueeze(0).to(device)
            probs = torch.softmax(model(t, b), dim=1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            rec = {"Leaf_ID": g["leaf_id"], "Top_Image": top_p.name, "Bottom_Image": bot_p.name,
                   "Predicted_Class": classes[idx], "Confidence": float(probs[idx]),
                   "True_Class": g["true_class"]}
            for i, c in enumerate(classes):
                rec[f"Prob_{c}"] = float(probs[i])
            rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(WORK_DIR / "test_predictions.csv", index=False)
    print(f"Saved {WORK_DIR / 'test_predictions.csv'} ({len(df)} leaf pairs)")

    lab = df[df["True_Class"].notnull()]
    if len(lab) == 0:
        print(df[["Leaf_ID", "Top_Image", "Bottom_Image", "Predicted_Class", "Confidence"]].head(10))
        return df

    y_true, y_pred = lab["True_Class"].tolist(), lab["Predicted_Class"].tolist()
    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    print("=" * 60)
    print(f"Test leaf pairs : {len(lab)}")
    print(f"Accuracy        : {acc*100:.2f}%")
    print(f"Macro P / R / F1: {mp*100:.2f}% / {mr*100:.2f}% / {mf*100:.2f}%")
    print(f"Weighted F1     : {wf*100:.2f}%")
    print("=" * 60)
    print(classification_report(y_true, y_pred, labels=classes, target_names=classes, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(WORK_DIR / "test_confusion_matrix.png", dpi=300)
    plt.show()
    return df


# =============================================================================
# STEP 6: single leaf diagnosis
# =============================================================================
def diagnose_leaf_pair(model, classes, top_path, bottom_path, show=True):
    model.eval()
    img_top = Image.open(top_path).convert("RGB")
    img_bot = Image.open(bottom_path).convert("RGB")
    t = eval_transform(img_top).unsqueeze(0).to(device)
    b = eval_transform(img_bot).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t, b), dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    pred, conf = classes[idx], probs[idx] * 100

    print("=" * 55)
    print(f"Top (T1)    : {top_path}")
    print(f"Bottom (B1) : {bottom_path}")
    print(f"Diagnosis   : {pred}  ({conf:.2f}% confidence)")
    for i, c in enumerate(classes):
        print(f"   {c:<15} {probs[i]*100:6.2f}%")
    print("=" * 55)

    if show:
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle(f"Diagnosis: {pred} ({conf:.2f}%)", fontweight="bold")
        a1.imshow(img_top); a1.set_title("Top (T1)"); a1.axis("off")
        a2.imshow(img_bot); a2.set_title("Bottom (B1)"); a2.axis("off")
        a3.barh(range(len(classes)), probs * 100,
                color=["#2ca02c" if c == pred else "#1f77b4" for c in classes])
        a3.set_yticks(range(len(classes))); a3.set_yticklabels(classes)
        a3.set_xlim(0, 100); a3.set_xlabel("Probability (%)")
        plt.tight_layout(); plt.show()
    return pred, conf


def load_trained_model():
    if not CLASS_JSON_PATH.exists():
        raise FileNotFoundError("class_to_idx.json not found - train the model first.")
    with open(CLASS_JSON_PATH) as f:
        class_to_idx = json.load(f)
    classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    model = DualViewEfficientNetB0(num_classes=len(classes), pretrained=False).to(device)
    path = FINAL_MODEL_PATH if FINAL_MODEL_PATH.exists() else BEST_MODEL_PATH
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, classes


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=r"D:\Segregation final1.zip", help="Dataset ZIP or folder (with the 4 class folders)")

    parser.add_argument("--test", type=str, default=None, help="Optional test folder (class sub-folders with T1/B1 images)")
    parser.add_argument("--predict_top", type=str, default=None, help="Top (T1) image for single diagnosis")
    parser.add_argument("--predict_bottom", type=str, default=None, help="Bottom (B1) image for single diagnosis")
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    set_seed()

    print("=" * 55)
    print(f"PyTorch {torch.__version__} | Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: no GPU found - training on CPU will be slow.")
    print("=" * 55)

    # ---- inference-only mode ----
    if args.predict_top and not args.data:
        model, classes = load_trained_model()
        diagnose_leaf_pair(model, classes, args.predict_top, args.predict_bottom or args.predict_top)
        return

    if not args.data:
        parser.error("--data is required for training")

    # ---- prepare + split ----
    dataset_path, classes, bad = prepare_dataset(args.data)
    split_dataset(dataset_path, classes, bad)

    train_ds = DualViewPaddyDataset(SPLIT_DIR / "train", transform=train_transform)
    val_ds = DualViewPaddyDataset(SPLIT_DIR / "val", transform=eval_transform)
    print(f"Classes: {train_ds.classes}")
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")
    with open(CLASS_JSON_PATH, "w") as f:
        json.dump(train_ds.class_to_idx, f, indent=4)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin)

    # ---- model + train ----
    model = DualViewEfficientNetB0(num_classes=len(train_ds.classes), pretrained=True).to(device)
    history = train_model(model, train_loader, val_loader)
    plot_curves(*history)

    # ---- evaluate ----
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    test_dir = args.test if args.test else str(SPLIT_DIR / "val")
    print(f"\nEvaluating on: {test_dir}")
    df = evaluate_folder(model, test_dir, train_ds.classes)

    # ---- demo single diagnosis ----
    if len(df) > 0:
        first = df.iloc[0]
        # locate the files again
        for p in Path(test_dir).rglob("*"):
            if p.name == first["Top_Image"]:
                top_p = p
            if p.name == first["Bottom_Image"]:
                bot_p = p
        try:
            diagnose_leaf_pair(model, train_ds.classes, top_p, bot_p)
        except Exception as e:
            print("Demo diagnosis skipped:", e)

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\nSaved final model: {FINAL_MODEL_PATH}")
    print(f"All outputs are in: {WORK_DIR.resolve()}")


if __name__ == "__main__":   # required on Windows for DataLoader workers
    main()