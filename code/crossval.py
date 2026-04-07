"""
CO3519 Assessment 2 — K-Fold Cross-Validation
MSS Fernando | G21266967

Models:
  1. Custom CNN  — 5-fold  (~25 min)
  2. VGG16       — 3-fold  (~5-6 hrs)
  3. ResNet50    — 3-fold  (~4-5 hrs)

Run:
  python3 crossval.py

Results saved to: results/cv/
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as res_preprocess
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from PIL import Image

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE        = os.path.join(os.path.dirname(__file__), 'datasets')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'cv')
os.makedirs(RESULTS_DIR, exist_ok=True)

FER_TRAIN = os.path.join(BASE, 'FER2013', 'train')
FER_VAL   = os.path.join(BASE, 'FER2013', 'val')
FER_TEST  = os.path.join(BASE, 'FER2013', 'test')
CK_TRAIN  = os.path.join(BASE, 'CK_Plus', 'train')
CK_VAL    = os.path.join(BASE, 'CK_Plus', 'val')
CK_TEST   = os.path.join(BASE, 'CK_Plus', 'test')

EMOTIONS    = ['AN', 'FE', 'HA', 'NE', 'SA', 'SU']
LABELS      = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
NUM_CLASSES = 6

print('\n' + '='*65)
print('  CO3519 Assessment 2 — K-Fold Cross-Validation')
print('  MSS Fernando | G21266967')
print('='*65)
print(f'  GPU : {tf.config.list_physical_devices("GPU")}')
print()


# ── DATA LOADING ───────────────────────────────────────────────────────────────
def collect_file_paths(dirs):
    """Return (paths[], labels[]) from one or more split directories."""
    paths, labels = [], []
    for d in dirs:
        for cls_idx, cls in enumerate(EMOTIONS):
            cls_dir = os.path.join(d, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(cls_dir, fname))
                    labels.append(cls_idx)
    return np.array(paths), np.array(labels)


def load_images_into_memory(paths, img_size, preprocess_fn=None):
    """Load all images into a numpy array (use for small CNN only)."""
    X = []
    for p in paths:
        img = Image.open(p).convert('RGB').resize((img_size, img_size))
        arr = np.array(img, dtype=np.float32)
        if preprocess_fn is not None:
            arr = preprocess_fn(arr)
        else:
            arr = arr / 255.0
        X.append(arr)
    return np.array(X)


class FoldSequence(keras.utils.Sequence):
    """Keras Sequence that reads images from a list of file paths on-the-fly."""

    def __init__(self, paths, labels, img_size, batch_size,
                 augment=False, preprocess_fn=None):
        self.paths        = paths
        self.labels       = labels
        self.img_size     = img_size
        self.batch_size   = batch_size
        self.augment      = augment
        self.preprocess   = preprocess_fn
        self.indices      = np.arange(len(paths))
        if augment:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []
        for i in batch_idx:
            img = Image.open(self.paths[i]).convert('RGB').resize(
                (self.img_size, self.img_size))
            arr = np.array(img, dtype=np.float32)
            if self.augment:
                arr = self._augment(arr)
            if self.preprocess is not None:
                arr = self.preprocess(arr)
            else:
                arr = arr / 255.0
            X.append(arr)
            y.append(self.labels[i])
        return np.array(X), keras.utils.to_categorical(y, NUM_CLASSES)

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.indices)

    def _augment(self, arr):
        # Horizontal flip
        if np.random.rand() > 0.5:
            arr = arr[:, ::-1, :]
        # Brightness
        arr = np.clip(arr * np.random.uniform(0.80, 1.20), 0, 255)
        # Small rotation via numpy (lightweight)
        if np.random.rand() > 0.7:
            k = np.random.choice([-1, 1])
            arr = np.rot90(arr, k=k)
        return arr


# ── MODELS ─────────────────────────────────────────────────────────────────────
def build_custom_cnn(img_size=48):
    keras.backend.clear_session()
    m = keras.Sequential(name='CustomCNN')
    for f in [32, 64, 128]:
        m.add(layers.Conv2D(f, (3,3), activation='relu', padding='same'))
        m.add(layers.BatchNormalization())
        m.add(layers.Conv2D(f, (3,3), activation='relu', padding='same'))
        m.add(layers.BatchNormalization())
        m.add(layers.MaxPooling2D(2, 2))
        m.add(layers.Dropout(0.25))
    m.add(layers.GlobalAveragePooling2D())
    m.add(layers.Dense(256, activation='relu'))
    m.add(layers.BatchNormalization())
    m.add(layers.Dropout(0.50))
    m.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    m.build((None, img_size, img_size, 3))
    return m


def build_vgg16():
    keras.backend.clear_session()
    base = VGG16(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
    base.trainable = False
    inp = keras.Input(shape=(224, 224, 3))
    x   = base(inp, training=False)
    x   = layers.Flatten()(x)
    x   = layers.Dense(512, activation='relu')(x)
    x   = layers.Dropout(0.50)(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.30)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return keras.Model(inp, out, name='VGG16_FER'), base


def build_resnet50():
    keras.backend.clear_session()
    base = ResNet50(weights='imagenet', include_top=False,
                    input_shape=(224, 224, 3))
    base.trainable = False
    inp = keras.Input(shape=(224, 224, 3))
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(512, activation='relu')(x)
    x   = layers.Dropout(0.50)(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.30)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return keras.Model(inp, out, name='ResNet50_FER'), base


def get_callbacks(name, results_dir):
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5,
            restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5,
            patience=3, min_lr=1e-8, verbose=0),
    ]


# ── CV CHART & REPORT ──────────────────────────────────────────────────────────
def save_cv_results(model_name, dataset_name, fold_accs):
    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    k        = len(fold_accs)

    print(f'\n  {"─"*50}')
    print(f'  {model_name} | {dataset_name} | {k}-Fold CV')
    print(f'  {"─"*50}')
    for i, a in enumerate(fold_accs, 1):
        print(f'    Fold {i}: {a:.2f}%')
    print(f'  {"─"*50}')
    print(f'    Mean  : {mean_acc:.2f}%')
    print(f'    Std   : {std_acc:.2f}%')
    print(f'    95% CI: {mean_acc:.2f}% +/- {1.96*std_acc:.2f}%')

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#3498db' if a >= mean_acc else '#e74c3c' for a in fold_accs]
    bars = ax.bar([f'Fold {i}' for i in range(1, k+1)],
                  fold_accs, color=colors, alpha=0.85, width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.axhline(mean_acc, color='green', ls='--', lw=2,
               label=f'Mean = {mean_acc:.2f}%')
    ax.fill_between(range(-1, k+1),
                    mean_acc - std_acc, mean_acc + std_acc,
                    alpha=0.15, color='green', label=f'SD = {std_acc:.2f}%')
    ax.set_xlim(-0.5, k - 0.5)
    ax.set_ylim(max(0, min(fold_accs) - 10), min(100, max(fold_accs) + 10))
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(f'{model_name} — {dataset_name}  {k}-Fold CV',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR,
                              f'CV_{model_name}_{dataset_name}.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()

    rpt_path = os.path.join(RESULTS_DIR,
                            f'CV_{model_name}_{dataset_name}.txt')
    with open(rpt_path, 'w', encoding='utf-8') as f:
        f.write(f'{model_name} | {dataset_name} | {k}-Fold CV\n')
        f.write('='*50 + '\n')
        for i, a in enumerate(fold_accs, 1):
            f.write(f'Fold {i}: {a:.2f}%\n')
        f.write('-'*50 + '\n')
        f.write(f'Mean  : {mean_acc:.2f}%\n')
        f.write(f'Std   : {std_acc:.2f}%\n')
        f.write(f'95% CI: {mean_acc:.2f}% +/- {1.96*std_acc:.2f}%\n')

    print(f'    Chart  -> {chart_path}')
    print(f'    Report -> {rpt_path}')
    return mean_acc, std_acc


# ── CNN CV (in-memory, 5-fold) ─────────────────────────────────────────────────
def run_cnn_cv(dataset_name, train_dirs, test_dir, epochs, batch_size, k=5):
    print(f'\n{"="*65}')
    print(f'  Custom CNN — {dataset_name}  {k}-Fold CV')
    print(f'{"="*65}')

    paths_all, labels_all = collect_file_paths(train_dirs)
    paths_test, labels_test = collect_file_paths([test_dir])

    print(f'  Loading {len(paths_all)} train+val images into memory...')
    X_all  = load_images_into_memory(paths_all,  img_size=48)
    X_test = load_images_into_memory(paths_test, img_size=48)
    y_all, y_test = labels_all, labels_test
    print(f'  Test set: {len(X_test)} images')

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all), 1):
        print(f'\n  -- Fold {fold}/{k} --')
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all[va_idx], y_all[va_idx]

        cw_arr = compute_class_weight('balanced',
                                      classes=np.unique(y_tr), y=y_tr)
        cw = dict(zip(np.unique(y_tr).astype(int), cw_arr))

        model = build_custom_cnn(img_size=48)
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        y_tr_cat = keras.utils.to_categorical(y_tr, NUM_CLASSES)
        y_va_cat = keras.utils.to_categorical(y_va, NUM_CLASSES)

        model.fit(
            X_tr, y_tr_cat,
            validation_data=(X_va, y_va_cat),
            epochs=epochs, batch_size=batch_size,
            class_weight=cw,
            callbacks=get_callbacks('cnn', RESULTS_DIR),
            verbose=1
        )

        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred) * 100
        fold_accs.append(acc)
        print(f'  Fold {fold} test acc: {acc:.2f}%')
        keras.backend.clear_session()

    return save_cv_results('CustomCNN', dataset_name, fold_accs)


# ── TL CV (file-based generator, 3-fold) ──────────────────────────────────────
def run_tl_cv(model_name, dataset_name, train_dirs, test_dir,
              build_fn, preprocess_fn,
              p1_epochs, p2_epochs, batch_size, unfreeze_layers, k=3):
    print(f'\n{"="*65}')
    print(f'  {model_name} — {dataset_name}  {k}-Fold CV (two-phase)')
    print(f'{"="*65}')

    paths_all, labels_all = collect_file_paths(train_dirs)
    paths_test, labels_test = collect_file_paths([test_dir])
    print(f'  Train+val: {len(paths_all)} | Test: {len(paths_test)}')

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_accs = []

    test_seq = FoldSequence(paths_test, labels_test, img_size=224,
                            batch_size=batch_size, augment=False,
                            preprocess_fn=preprocess_fn)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(paths_all, labels_all), 1):
        print(f'\n  -- Fold {fold}/{k} --')
        p_tr, l_tr = paths_all[tr_idx], labels_all[tr_idx]
        p_va, l_va = paths_all[va_idx], labels_all[va_idx]

        cw_arr = compute_class_weight('balanced',
                                      classes=np.unique(l_tr), y=l_tr)
        cw = dict(zip(np.unique(l_tr).astype(int), cw_arr))

        train_seq = FoldSequence(p_tr, l_tr, img_size=224,
                                 batch_size=batch_size, augment=True,
                                 preprocess_fn=preprocess_fn)
        val_seq   = FoldSequence(p_va, l_va, img_size=224,
                                 batch_size=batch_size, augment=False,
                                 preprocess_fn=preprocess_fn)

        model, base = build_fn()

        # Phase 1 — frozen base
        print(f'  Phase 1: head only (lr=1e-3, max {p1_epochs} epochs)...')
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_seq, validation_data=val_seq,
                  epochs=p1_epochs, class_weight=cw,
                  callbacks=get_callbacks(model_name, RESULTS_DIR),
                  verbose=1)

        # Phase 2 — fine-tune top layers
        print(f'  Phase 2: fine-tune top {unfreeze_layers} layers '
              f'(lr=1e-5, max {p2_epochs} epochs)...')
        base.trainable = True
        for layer in base.layers[:-unfreeze_layers]:
            layer.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(1e-5),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_seq, validation_data=val_seq,
                  epochs=p2_epochs, class_weight=cw,
                  callbacks=get_callbacks(model_name, RESULTS_DIR),
                  verbose=1)

        # Evaluate on test
        y_pred = np.argmax(model.predict(test_seq, verbose=0), axis=1)
        acc = accuracy_score(labels_test, y_pred) * 100
        fold_accs.append(acc)
        print(f'  Fold {fold} test acc: {acc:.2f}%')
        keras.backend.clear_session()

    return save_cv_results(model_name, dataset_name, fold_accs)


# ── RUN ALL CV ─────────────────────────────────────────────────────────────────
all_results = {}

# 1. Custom CNN — FER2013 (5-fold)
m, s = run_cnn_cv(
    'FER2013',
    train_dirs=[FER_TRAIN, FER_VAL],
    test_dir=FER_TEST,
    epochs=30, batch_size=32, k=5
)
all_results['CustomCNN_FER2013'] = (m, s)

# 2. Custom CNN — CK+ (5-fold)
m, s = run_cnn_cv(
    'CK+',
    train_dirs=[CK_TRAIN, CK_VAL],
    test_dir=CK_TEST,
    epochs=50, batch_size=16, k=5
)
all_results['CustomCNN_CK'] = (m, s)

# 3. VGG16 — FER2013 (3-fold)
m, s = run_tl_cv(
    'VGG16', 'FER2013',
    train_dirs=[FER_TRAIN, FER_VAL],
    test_dir=FER_TEST,
    build_fn=build_vgg16,
    preprocess_fn=vgg_preprocess,
    p1_epochs=12, p2_epochs=15,
    batch_size=32, unfreeze_layers=8, k=3
)
all_results['VGG16_FER2013'] = (m, s)

# 4. VGG16 — CK+ (3-fold)
m, s = run_tl_cv(
    'VGG16', 'CK+',
    train_dirs=[CK_TRAIN, CK_VAL],
    test_dir=CK_TEST,
    build_fn=build_vgg16,
    preprocess_fn=vgg_preprocess,
    p1_epochs=15, p2_epochs=20,
    batch_size=16, unfreeze_layers=8, k=3
)
all_results['VGG16_CK'] = (m, s)

# 5. ResNet50 — FER2013 (3-fold)
m, s = run_tl_cv(
    'ResNet50', 'FER2013',
    train_dirs=[FER_TRAIN, FER_VAL],
    test_dir=FER_TEST,
    build_fn=build_resnet50,
    preprocess_fn=res_preprocess,
    p1_epochs=12, p2_epochs=15,
    batch_size=32, unfreeze_layers=30, k=3
)
all_results['ResNet50_FER2013'] = (m, s)

# 6. ResNet50 — CK+ (3-fold)
m, s = run_tl_cv(
    'ResNet50', 'CK+',
    train_dirs=[CK_TRAIN, CK_VAL],
    test_dir=CK_TEST,
    build_fn=build_resnet50,
    preprocess_fn=res_preprocess,
    p1_epochs=15, p2_epochs=20,
    batch_size=16, unfreeze_layers=30, k=3
)
all_results['ResNet50_CK'] = (m, s)

# ── FINAL SUMMARY ──────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print(f'  CROSS-VALIDATION SUMMARY')
print(f'{"="*65}')
print(f'  {"Model":<25} {"Dataset":<10} {"Mean":>8} {"Std":>7} {"95% CI":>22}')
print(f'  {"─"*60}')
for key, (mean, std) in all_results.items():
    parts  = key.rsplit('_', 1)
    model  = parts[0]
    ds     = 'CK+' if parts[1] == 'CK' else parts[1]
    ci     = f'{mean:.2f}% +/- {1.96*std:.2f}%'
    print(f'  {model:<25} {ds:<10} {mean:>7.2f}% {std:>6.2f}%   {ci}')
print(f'{"="*65}')

summary_path = os.path.join(RESULTS_DIR, 'CV_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('CO3519 Assessment 2 — Cross-Validation Summary\n')
    f.write('MSS Fernando | G21266967\n')
    f.write('='*60 + '\n\n')
    f.write(f'{"Model":<25} {"Dataset":<10} {"Mean":>8} {"Std":>7} {"95% CI":>22}\n')
    f.write('-'*60 + '\n')
    for key, (mean, std) in all_results.items():
        parts = key.rsplit('_', 1)
        model = parts[0]
        ds    = 'CK+' if parts[1] == 'CK' else parts[1]
        ci    = f'{mean:.2f}% +/- {1.96*std:.2f}%'
        f.write(f'{model:<25} {ds:<10} {mean:>7.2f}% {std:>6.2f}%   {ci}\n')

print(f'\nSummary -> {summary_path}')
print('DONE.')
print('='*65)
