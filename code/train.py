"""
CO3519 Assessment 2 — Facial Emotion Recognition using Deep Learning
MSS Fernando | G21266967

Models:
  1. Custom CNN    — built from scratch (baseline)
  2. VGG16         — Transfer Learning (CO3519 Lab 5)
  3. ResNet50      — Improved Transfer Learning

Run:
  python3 train.py

Results saved to: results/
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — required on Mac/Linux
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as res_preprocess
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.utils.class_weight import compute_class_weight

# ImageDataGenerator moved in TF 2.16+ / Keras 3 — handle all versions
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    try:
        from keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE        = os.path.join(os.path.dirname(__file__), 'datasets')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
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

# ── VERIFY PATHS ───────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('  CO3519 Assessment 2 — FER Deep Learning')
print('  MSS Fernando | G21266967')
print('='*65)
print('\nDataset verification:')
all_ok = True
for p in [FER_TRAIN, FER_VAL, FER_TEST, CK_TRAIN, CK_VAL, CK_TEST]:
    n = sum(len(f) for _, _, f in os.walk(p)) if os.path.exists(p) else 0
    ok = n > 0
    if not ok:
        all_ok = False
    print(f'  [{"OK" if ok else "MISSING"}] {n:>6} images | {p}')

if not all_ok:
    print('\nERROR: Fix missing paths before running.')
    sys.exit(1)

ck_train_count = sum(len(f) for _, _, f in os.walk(CK_TRAIN))
if ck_train_count < 500:
    print(f'\nWARNING: CK+ train has only {ck_train_count} images.')
    print('Expected 1,872 (augmented). Results may be poor.')

print(f'\nResults will save to: {RESULTS_DIR}/')
print(f'GPU: {tf.config.list_physical_devices("GPU")}')
print()

# ── DATA GENERATORS ────────────────────────────────────────────────────────────
# Custom CNN (48px) — rescale to [0,1]
cnn_fer_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True,
    zoom_range=0.10,
    brightness_range=[0.80, 1.20]
)
cnn_ck_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
    brightness_range=[0.75, 1.25],
    shear_range=0.10,
    fill_mode='nearest'
)
cnn_eval = ImageDataGenerator(rescale=1./255)

# VGG16 (224px) — uses its own preprocess_input (mean subtraction, BGR)
vgg_fer_aug = ImageDataGenerator(
    preprocessing_function=vgg_preprocess,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True,
    zoom_range=0.10,
    brightness_range=[0.80, 1.20]
)
vgg_ck_aug = ImageDataGenerator(
    preprocessing_function=vgg_preprocess,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
    brightness_range=[0.75, 1.25],
    shear_range=0.10,
    fill_mode='nearest'
)
vgg_eval = ImageDataGenerator(preprocessing_function=vgg_preprocess)

# ResNet50 (224px) — uses its own preprocess_input (mean subtraction, BGR)
res_fer_aug = ImageDataGenerator(
    preprocessing_function=res_preprocess,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True,
    zoom_range=0.10,
    brightness_range=[0.80, 1.20]
)
res_ck_aug = ImageDataGenerator(
    preprocessing_function=res_preprocess,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
    brightness_range=[0.75, 1.25],
    shear_range=0.10,
    fill_mode='nearest'
)
res_eval = ImageDataGenerator(preprocessing_function=res_preprocess)


def make_gens(train_p, val_p, test_p, img_size, batch=32,
              train_aug=None, eval_aug=None):
    kw = dict(
        target_size=(img_size, img_size),
        class_mode='categorical',
        classes=EMOTIONS,
        color_mode='rgb'
    )
    tr = train_aug.flow_from_directory(train_p, shuffle=True, seed=42,
                                       batch_size=batch, **kw)
    va = eval_aug.flow_from_directory(val_p,  shuffle=False,
                                      batch_size=batch, **kw)
    te = eval_aug.flow_from_directory(test_p, shuffle=False,
                                      batch_size=batch, **kw)
    return tr, va, te


def get_class_weights(gen):
    y  = gen.classes
    cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
    return dict(zip(np.unique(y).astype(int), cw))


# ── MODEL DEFINITIONS ──────────────────────────────────────────────────────────
def build_custom_cnn(img_size=48):
    """
    Custom CNN built from scratch — baseline model.
    Architecture: 3x (Conv->BN->Conv->BN->MaxPool->Dropout)
                  -> GAP -> Dense(256) -> Softmax(6)
    """
    m = keras.Sequential(name='CustomCNN')
    for f in [32, 64, 128]:
        m.add(layers.Conv2D(f, (3, 3), activation='relu', padding='same'))
        m.add(layers.BatchNormalization())
        m.add(layers.Conv2D(f, (3, 3), activation='relu', padding='same'))
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


def build_vgg16(img_size=224):
    """
    VGG16 Transfer Learning — CO3519 Lab 5 (Week 5).
    Head: Flatten -> Dense(512) -> Dropout(0.5) -> Dense(6)
    Returns (model, base) for two-phase training.
    """
    base = VGG16(weights='imagenet', include_top=False,
                 input_shape=(img_size, img_size, 3))
    base.trainable = False
    inp = keras.Input(shape=(img_size, img_size, 3))
    x   = base(inp, training=False)
    x   = layers.Flatten()(x)
    x   = layers.Dense(512, activation='relu')(x)
    x   = layers.Dropout(0.50)(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.30)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return keras.Model(inp, out, name='VGG16_FER'), base


def build_resnet50(img_size=224):
    """
    ResNet50 Transfer Learning — improved over ResNet18.
    Uses GlobalAveragePooling (not Flatten) to reduce overfitting.
    Lower initial LR (3e-4) works better for FER.
    Returns (model, base) for two-phase training.
    """
    base = ResNet50(weights='imagenet', include_top=False,
                    input_shape=(img_size, img_size, 3))
    base.trainable = False
    inp = keras.Input(shape=(img_size, img_size, 3))
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)   # added — matches ResNetOnly
    x   = layers.Dense(512, activation='relu')(x)
    x   = layers.Dropout(0.50)(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.30)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return keras.Model(inp, out, name='ResNet50_FER'), base


def get_callbacks(name):
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=6,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5,
            patience=3, min_lr=1e-8, verbose=1),
        keras.callbacks.ModelCheckpoint(
            os.path.join(RESULTS_DIR, f'{name}.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=0)
    ]


# ── EVALUATION UTILITIES ───────────────────────────────────────────────────────
def plot_curves(histories, model_name, dataset_name):
    if not isinstance(histories, list):
        histories = [histories]
    tl, vl, ta, va = [], [], [], []
    p1_end = len(histories[0].history['loss'])
    for h in histories:
        tl += h.history['loss']
        vl += h.history['val_loss']
        ta += h.history['accuracy']
        va += h.history['val_accuracy']
    ep = range(1, len(tl) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{model_name} — {dataset_name} Training History',
                 fontsize=14, fontweight='bold')
    for ax, tr, v, ylabel in [(ax1, tl, vl, 'Loss'),
                               (ax2, ta, va, 'Accuracy')]:
        ax.plot(ep, tr, 'b-', lw=2, label=f'Train {ylabel}')
        ax.plot(ep, v,  'r-', lw=2, label=f'Val {ylabel}')
        if len(histories) > 1:
            ax.axvline(x=p1_end, color='green', ls='--', lw=1.5,
                       label='Fine-tune start')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'{model_name}_{dataset_name}_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {path}')


def evaluate(model, test_gen, model_name, dataset_name):
    test_gen.reset()
    y_pred = np.argmax(model.predict(test_gen, verbose=0), axis=1)
    y_true = test_gen.classes
    acc    = accuracy_score(y_true, y_pred) * 100
    print(f'\n{"="*65}')
    print(f'  {model_name}  |  {dataset_name}')
    print(f'{"="*65}')
    print(f'Test Accuracy: {acc:.2f}%')
    report = classification_report(y_true, y_pred,
                                   target_names=LABELS, digits=4)
    print(report)

    # Save report to text file
    rpt_path = os.path.join(RESULTS_DIR,
                            f'report_{model_name}_{dataset_name}.txt')
    with open(rpt_path, 'w', encoding='utf-8') as f:
        f.write(f'{model_name} | {dataset_name}\n')
        f.write(f'Test Accuracy: {acc:.2f}%\n\n')
        f.write(report)

    # Confusion matrices
    cm  = confusion_matrix(y_true, y_pred)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, data, fmt, title in [
        (axes[0], cm,  'd',    'Counts'),
        (axes[1], cmn, '.2f', 'Normalised')
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=LABELS, yticklabels=LABELS, ax=ax)
        ax.set_title(f'{model_name} — {dataset_name} ({title})',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR,
                           f'CM_{model_name}_{dataset_name}.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {cm_path}')
    return acc


def show_samples(model, test_gen, model_name, dataset_name):
    SHOW   = ('HA', 'AN', 'SU')
    COLORS = {'HA': '#27ae60', 'AN': '#c0392b', 'SU': '#2980b9',
              'SA': '#8e44ad', 'FE': '#e67e22', 'NE': '#7f8c8d'}
    test_gen.reset()
    imgs, labs = [], []
    for bx, by in test_gen:
        imgs.append(bx)
        labs.append(by)
        if len(imgs) * test_gen.batch_size >= test_gen.n:
            break
    imgs   = np.vstack(imgs)[:test_gen.n]
    y_true = np.argmax(np.vstack(labs)[:test_gen.n], axis=1)
    y_pred = np.argmax(model.predict(imgs, verbose=0), axis=1)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'{model_name} — {dataset_name}: Sample Predictions',
                 fontsize=14, fontweight='bold')
    for row, emo in enumerate(SHOW):
        idx_emo = EMOTIONS.index(emo)
        correct = np.where((y_true == idx_emo) & (y_pred == idx_emo))[0]
        pool    = correct if len(correct) >= 4 else np.where(y_true == idx_emo)[0]
        for col, i in enumerate(pool[:4]):
            ax = axes[row, col]
            ax.imshow(imgs[i])
            pn = EMOTIONS[y_pred[i]]
            tn = EMOTIONS[y_true[i]]
            ax.set_title(
                f'Pred: {LABELS[EMOTIONS.index(pn)]}\n'
                f'True: {LABELS[EMOTIONS.index(tn)]}',
                fontsize=9, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor=COLORS.get(pn, '#333'), alpha=0.85)
            )
            if pn != tn:
                for s in ax.spines.values():
                    s.set_edgecolor('red')
                    s.set_linewidth(2)
            ax.axis('off')
        axes[row, 0].set_ylabel(LABELS[EMOTIONS.index(emo)],
                                fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR,
                        f'Samples_{model_name}_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {path}')


# ── TRAINING ───────────────────────────────────────────────────────────────────
results = {}

# ── 1/6: Custom CNN — FER2013 ──────────────────────────────────────────────
print('\n' + '='*65)
print('  TRAIN 1/6 — Custom CNN on FER2013')
print('='*65)
fer_tr_c, fer_va_c, fer_te_c = make_gens(
    FER_TRAIN, FER_VAL, FER_TEST, img_size=48, batch=32,
    train_aug=cnn_fer_aug, eval_aug=cnn_eval)
cw_fer = get_class_weights(fer_tr_c)

cnn_fer = build_custom_cnn(img_size=48)
cnn_fer.compile(optimizer=keras.optimizers.Adam(1e-3),
                loss='categorical_crossentropy', metrics=['accuracy'])
hist_cnn_fer = cnn_fer.fit(
    fer_tr_c, validation_data=fer_va_c,
    epochs=30, callbacks=get_callbacks('CustomCNN_FER2013'),
    class_weight=cw_fer, verbose=1
)
plot_curves(hist_cnn_fer, 'CustomCNN', 'FER2013')
results['CustomCNN_FER2013'] = evaluate(cnn_fer, fer_te_c, 'CustomCNN', 'FER2013')

# ── 2/6: Custom CNN — CK+ ──────────────────────────────────────────────────
print('\n' + '='*65)
print('  TRAIN 2/6 — Custom CNN on CK+')
print('='*65)
ck_tr_c, ck_va_c, ck_te_c = make_gens(
    CK_TRAIN, CK_VAL, CK_TEST, img_size=48, batch=16,
    train_aug=cnn_ck_aug, eval_aug=cnn_eval)
cw_ck = get_class_weights(ck_tr_c)

cnn_ck = build_custom_cnn(img_size=48)
cnn_ck.compile(optimizer=keras.optimizers.Adam(1e-3),
               loss='categorical_crossentropy', metrics=['accuracy'])
hist_cnn_ck = cnn_ck.fit(
    ck_tr_c, validation_data=ck_va_c,
    epochs=50, callbacks=get_callbacks('CustomCNN_CK'),
    class_weight=cw_ck, verbose=1
)
plot_curves(hist_cnn_ck, 'CustomCNN', 'CK+')
results['CustomCNN_CK'] = evaluate(cnn_ck, ck_te_c, 'CustomCNN', 'CK+')

# ── 3/6: VGG16 — FER2013 ───────────────────────────────────────────────────
print('\n' + '='*65)
print('  TRAIN 3/6 — VGG16 Transfer Learning on FER2013')
print('='*65)
fer_tr_e, fer_va_e, fer_te_e = make_gens(
    FER_TRAIN, FER_VAL, FER_TEST, img_size=224, batch=32,
    train_aug=vgg_fer_aug, eval_aug=vgg_eval)

vgg_fer, base_vgg_fer = build_vgg16(img_size=224)

print('\nPHASE 1: Training head only (base frozen)...')
vgg_fer.compile(optimizer=keras.optimizers.Adam(1e-3),
                loss='categorical_crossentropy', metrics=['accuracy'])
h_vgg_p1 = vgg_fer.fit(
    fer_tr_e, validation_data=fer_va_e,
    epochs=15, callbacks=get_callbacks('VGG16_FER2013_p1'),
    class_weight=cw_fer, verbose=1
)

print('\nPHASE 2: Fine-tuning top layers (lr=1e-5)...')
base_vgg_fer.trainable = True
for layer in base_vgg_fer.layers[:-8]:
    layer.trainable = False
vgg_fer.compile(optimizer=keras.optimizers.Adam(1e-5),
                loss='categorical_crossentropy', metrics=['accuracy'])
h_vgg_p2 = vgg_fer.fit(
    fer_tr_e, validation_data=fer_va_e,
    epochs=20, callbacks=get_callbacks('VGG16_FER2013_p2'),
    class_weight=cw_fer, verbose=1
)

plot_curves([h_vgg_p1, h_vgg_p2], 'VGG16', 'FER2013')
results['VGG16_FER2013'] = evaluate(vgg_fer, fer_te_e, 'VGG16', 'FER2013')
show_samples(vgg_fer, fer_te_e, 'VGG16', 'FER2013')

# ── 4/6: VGG16 — CK+ ───────────────────────────────────────────────────────
print('\n' + '='*65)
print('  TRAIN 4/6 — VGG16 Transfer Learning on CK+')
print('='*65)
ck_tr_e, ck_va_e, ck_te_e = make_gens(
    CK_TRAIN, CK_VAL, CK_TEST, img_size=224, batch=16,
    train_aug=vgg_ck_aug, eval_aug=vgg_eval)

vgg_ck, base_vgg_ck = build_vgg16(img_size=224)

print('\nPHASE 1: Training head only...')
vgg_ck.compile(optimizer=keras.optimizers.Adam(1e-3),
               loss='categorical_crossentropy', metrics=['accuracy'])
h_vgg_ck_p1 = vgg_ck.fit(
    ck_tr_e, validation_data=ck_va_e,
    epochs=20, callbacks=get_callbacks('VGG16_CK_p1'),
    class_weight=cw_ck, verbose=1
)

print('\nPHASE 2: Fine-tuning...')
base_vgg_ck.trainable = True
for layer in base_vgg_ck.layers[:-8]:
    layer.trainable = False
vgg_ck.compile(optimizer=keras.optimizers.Adam(1e-5),
               loss='categorical_crossentropy', metrics=['accuracy'])
h_vgg_ck_p2 = vgg_ck.fit(
    ck_tr_e, validation_data=ck_va_e,
    epochs=30, callbacks=get_callbacks('VGG16_CK_p2'),
    class_weight=cw_ck, verbose=1
)

plot_curves([h_vgg_ck_p1, h_vgg_ck_p2], 'VGG16', 'CK+')
results['VGG16_CK'] = evaluate(vgg_ck, ck_te_e, 'VGG16', 'CK+')
show_samples(vgg_ck, ck_te_e, 'VGG16', 'CK+')

# ── 5/6: ResNet50 — FER2013 ────────────────────────────────────────────────
print('\n' + '='*65)
print('  TRAIN 5/6 — ResNet50 Transfer Learning on FER2013')
print('='*65)
fer_tr_r, fer_va_r, fer_te_r = make_gens(
    FER_TRAIN, FER_VAL, FER_TEST, img_size=224, batch=32,
    train_aug=res_fer_aug, eval_aug=res_eval)

res_fer, base_res_fer = build_resnet50(img_size=224)

print('\nPHASE 1: Training head only (lr=3e-4)...')
res_fer.compile(optimizer=keras.optimizers.Adam(3e-4),
                loss='categorical_crossentropy', metrics=['accuracy'])
h_res_p1 = res_fer.fit(
    fer_tr_r, validation_data=fer_va_r,
    epochs=15, callbacks=get_callbacks('ResNet50_FER2013_p1'),
    class_weight=cw_fer, verbose=1
)

print('\nPHASE 2: Fine-tuning top 30 layers (lr=1e-5)...')
base_res_fer.trainable = True
for layer in base_res_fer.layers[:-30]:
    layer.trainable = False
res_fer.compile(optimizer=keras.optimizers.Adam(1e-5),
                loss='categorical_crossentropy', metrics=['accuracy'])
h_res_p2 = res_fer.fit(
    fer_tr_r, validation_data=fer_va_r,
    epochs=25, callbacks=get_callbacks('ResNet50_FER2013_p2'),
    class_weight=cw_fer, verbose=1
)

plot_curves([h_res_p1, h_res_p2], 'ResNet50', 'FER2013')
results['ResNet50_FER2013'] = evaluate(res_fer, fer_te_r, 'ResNet50', 'FER2013')
show_samples(res_fer, fer_te_r, 'ResNet50', 'FER2013')

# ── 6/6: ResNet50 — CK+ ────────────────────────────────────────────────────
print('\n' + '='*65)
print('  TRAIN 6/6 — ResNet50 Transfer Learning on CK+')
print('='*65)
ck_tr_r, ck_va_r, ck_te_r = make_gens(
    CK_TRAIN, CK_VAL, CK_TEST, img_size=224, batch=16,
    train_aug=res_ck_aug, eval_aug=res_eval)

res_ck, base_res_ck = build_resnet50(img_size=224)

print('\nPHASE 1: Training head only...')
res_ck.compile(optimizer=keras.optimizers.Adam(3e-4),
               loss='categorical_crossentropy', metrics=['accuracy'])
h_res_ck_p1 = res_ck.fit(
    ck_tr_r, validation_data=ck_va_r,
    epochs=20, callbacks=get_callbacks('ResNet50_CK_p1'),
    class_weight=cw_ck, verbose=1
)

print('\nPHASE 2: Fine-tuning...')
base_res_ck.trainable = True
for layer in base_res_ck.layers[:-30]:
    layer.trainable = False
res_ck.compile(optimizer=keras.optimizers.Adam(1e-5),
               loss='categorical_crossentropy', metrics=['accuracy'])
h_res_ck_p2 = res_ck.fit(
    ck_tr_r, validation_data=ck_va_r,
    epochs=35, callbacks=get_callbacks('ResNet50_CK_p2'),
    class_weight=cw_ck, verbose=1
)

plot_curves([h_res_ck_p1, h_res_ck_p2], 'ResNet50', 'CK+')
results['ResNet50_CK'] = evaluate(res_ck, ck_te_r, 'ResNet50', 'CK+')
show_samples(res_ck, ck_te_r, 'ResNet50', 'CK+')

# ── FINAL SUMMARY ──────────────────────────────────────────────────────────────
A1_FER = 50.39
A1_CK  = 95.83

acc_cnn_fer = results['CustomCNN_FER2013']
acc_cnn_ck  = results['CustomCNN_CK']
acc_vgg_fer = results['VGG16_FER2013']
acc_vgg_ck  = results['VGG16_CK']
acc_res_fer = results['ResNet50_FER2013']
acc_res_ck  = results['ResNet50_CK']

print(f'\n{"─"*65}')
print(f'{"Model":<42} {"FER2013":>10} {"CK+":>10}')
print(f'{"─"*65}')
print(f'{"SVM + LBP+HOG  (Assessment 1)":<42} {A1_FER:>9.2f}% {A1_CK:>9.2f}%')
print(f'{"Custom CNN     (Assessment 2)":<42} {acc_cnn_fer:>9.2f}% {acc_cnn_ck:>9.2f}%')
print(f'{"VGG16 TL       (Assessment 2)":<42} {acc_vgg_fer:>9.2f}% {acc_vgg_ck:>9.2f}%')
print(f'{"ResNet50 TL    (Assessment 2)":<42} {acc_res_fer:>9.2f}% {acc_res_ck:>9.2f}%')
print(f'{"─"*65}')

best_fer = max(acc_cnn_fer, acc_vgg_fer, acc_res_fer)
best_ck  = max(acc_cnn_ck,  acc_vgg_ck,  acc_res_ck)
print(f'\nBest FER2013: {best_fer:.2f}%  ({best_fer - A1_FER:+.2f}% vs A1 SVM)')
print(f'Best CK+    : {best_ck:.2f}%  ({best_ck - A1_CK:+.2f}% vs A1 SVM)')

# Bar chart
names  = ['SVM\n(A1)', 'Custom\nCNN', 'VGG16\nTL', 'ResNet50\nTL']
fer_v  = [A1_FER, acc_cnn_fer, acc_vgg_fer, acc_res_fer]
ck_v   = [A1_CK,  acc_cnn_ck,  acc_vgg_ck,  acc_res_ck]
x, w   = np.arange(4), 0.35

fig, ax = plt.subplots(figsize=(13, 7))
b1 = ax.bar(x - w/2, fer_v, w, label='FER2013', color='#3498db', alpha=0.85)
b2 = ax.bar(x + w/2, ck_v,  w, label='CK+',     color='#e74c3c', alpha=0.85)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.axvline(x=0.5, color='gray', ls='--', lw=1.5, alpha=0.6)
ax.text(0,   113, 'Assess. 1', ha='center', fontsize=9,
        style='italic', color='gray')
ax.text(2,   113, 'Assessment 2 — Deep Learning', ha='center',
        fontsize=9, style='italic', color='gray')
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_ylim(0, 120)
ax.set_title('Traditional ML vs Deep Learning — A1 vs A2',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
chart_path = os.path.join(RESULTS_DIR, 'A1_vs_A2_Comparison.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()

# Save colab_results.txt automatically
results_txt = os.path.join(os.path.dirname(__file__), 'colab_results.txt')
with open(results_txt, 'w', encoding='utf-8') as f:
    f.write('=== COLAB RESULTS ===\n\n')
    f.write(f'CustomCNN_FER2013_accuracy: {acc_cnn_fer:.2f}\n')
    f.write(f'CustomCNN_CK_accuracy: {acc_cnn_ck:.2f}\n')
    f.write(f'VGG16_FER2013_accuracy: {acc_vgg_fer:.2f}\n')
    f.write(f'VGG16_CK_accuracy: {acc_vgg_ck:.2f}\n')
    f.write(f'ResNet50_FER2013_accuracy: {acc_res_fer:.2f}\n')
    f.write(f'ResNet50_CK_accuracy: {acc_res_ck:.2f}\n\n')
    for key in ['VGG16_FER2013', 'ResNet50_FER2013', 'VGG16_CK', 'ResNet50_CK']:
        rpt = os.path.join(RESULTS_DIR, f'report_{key}.txt')
        if os.path.exists(rpt):
            f.write(f'\n{key} Classification Report:\n')
            with open(rpt, encoding='utf-8') as r:
                f.write(r.read())
            f.write('\n')

print(f'\ncolab_results.txt auto-saved -> {results_txt}')
print(f'All plots saved -> {RESULTS_DIR}/')
print('\nDONE. Give colab_results.txt to Claude Code to build the report.')
print('='*65)
