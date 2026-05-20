"""
Лабораторна робота №3 — ДКП-стиснення зображень.
Аналіз сегментів: Фон / Текстура / Деталі.
Розміри блоків: 8×8 / 16×16 / 32×32.
Якість: Q = 10, 25, 50, 75, 90.

Запуск локально:  python DiscreteCosineTransform.py
Запуск у Colab:   завантажте I04.BMP через files.upload(), решта без змін.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import heapq
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import warnings
warnings.filterwarnings("ignore")

# ── Colab / локальний запуск ───────────────────────────────────────────────────
try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ── шлях до зображення ────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE, "I04.BMP")

# ── папка для збереження фігур ─────────────────────────────────────────────────
OUT_DIR = os.path.join(BASE, "results", "dct_analysis")
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# HUFFMAN
# ══════════════════════════════════════════════════════════════════════════════

def get_huffman_codes(data):
    freq = Counter(data.flatten().astype(int))
    if not freq:
        return {}
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo, hi = heapq.heappop(heap), heapq.heappop(heap)
        for pair in lo[1:]: pair[1] = "0" + pair[1]
        for pair in hi[1:]: pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return dict(heapq.heappop(heap)[1:])


def calc_entropy_metrics(data, codes):
    total = data.size
    freq  = Counter(data.flatten().astype(int))
    probs = {s: f / total for s, f in freq.items()}
    entropy = -sum(p * np.log2(p) for p in probs.values() if p > 0)
    avg_len = sum(probs[s] * len(codes[s]) for s in probs)
    return entropy, avg_len


# ══════════════════════════════════════════════════════════════════════════════
# QUANTIZATION — стандартна формула JPEG (ідентична JPEG стандарту)
# ══════════════════════════════════════════════════════════════════════════════

JPEG_Q50 = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99],
], dtype=np.float32)


def get_q_matrix(quality: int) -> np.ndarray:
    """Стандартна матриця квантування JPEG для заданого Q (1–100)."""
    q     = max(1, min(100, quality))
    scale = 5000 / q if q < 50 else 200 - 2 * q
    qm    = np.floor((JPEG_Q50 * scale + 50) / 100).astype(np.float32)
    return np.clip(qm, 1, 255)


def get_q_matrix_for_block(quality: int, block_size: int) -> np.ndarray:
    """Матриця квантування для довільного розміру блоку (масштабується від 8×8)."""
    base8 = get_q_matrix(quality)
    if block_size == 8:
        return base8
    qm = cv2.resize(base8, (block_size, block_size),
                    interpolation=cv2.INTER_LINEAR)
    return np.clip(np.round(qm), 1, 255).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# DCT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def dct2(block: np.ndarray) -> np.ndarray:
    return cv2.dct(block.astype(np.float32))


def idct2(dct_block: np.ndarray) -> np.ndarray:
    return cv2.idct(dct_block.astype(np.float32))


def quantize(dct_block: np.ndarray, q_mat: np.ndarray) -> np.ndarray:
    return np.round(dct_block / q_mat).astype(np.int32)


def compression_ratio(quantized: np.ndarray, block_size: int) -> float:
    """Коефіцієнт стиснення: (розмір оригінального блоку) / (ненульові коеф.)."""
    total    = block_size * block_size
    nonzero  = np.count_nonzero(quantized)
    return total / nonzero if nonzero > 0 else float("inf")


# ══════════════════════════════════════════════════════════════════════════════
# ЗАВАНТАЖЕННЯ ЗОБРАЖЕННЯ
# ══════════════════════════════════════════════════════════════════════════════

def load_image():
    if IN_COLAB:
        uploaded = colab_files.upload()
        path = list(uploaded.keys())[0]
    else:
        path = IMAGE_PATH
    img_bgr  = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Зображення не знайдено: {path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr, img_gray


# ══════════════════════════════════════════════════════════════════════════════
# СЕГМЕНТИ
# ══════════════════════════════════════════════════════════════════════════════

SEGMENT_COORDS = {
    "Фон (рівномірний)":       (20,  20),
    "Текстура (регулярна)":    (140, 220),
    "Деталі (високочастотні)": (250, 240),
}

BLOCK_SIZES = [8, 16, 32]
Q_FACTORS   = [10, 25, 50, 75, 90]


# ══════════════════════════════════════════════════════════════════════════════
# ВІЗУАЛІЗАЦІЯ — карта сегментів
# ══════════════════════════════════════════════════════════════════════════════

def plot_segment_map(img_bgr):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    for name, (y, x) in SEGMENT_COORDS.items():
        ax.add_patch(plt.Rectangle((x, y), 8, 8,
                                   color="red", fill=False, lw=2))
        ax.text(x, y - 5, name, color="white",
                backgroundcolor="red", fontsize=9)
    ax.set_title("Карта вибору контрольних сегментів")
    ax.axis("off")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "segment_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Збережено: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# ВІЗУАЛІЗАЦІЯ — аналіз одного сегменту (блок + гістограма + спектр)
# ══════════════════════════════════════════════════════════════════════════════

def plot_segment_analysis(name, block):
    dct_block = dct2(block.astype(np.float32))
    short = name.split(" ")[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Статистичний аналіз структури: {name}", fontsize=13)

    axes[0].imshow(block, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Вихідний блок 8×8")

    axes[1].hist(block.flatten(), bins=16, color="darkblue", edgecolor="white")
    axes[1].set_title("Розподіл інтенсивностей пікселів")
    axes[1].set_xlabel("Яскравість (0–255)")
    axes[1].set_ylabel("Кількість")

    axes[2].imshow(np.log(np.abs(dct_block) + 1), cmap="plasma")
    axes[2].set_title("Енергетичний спектр ДКП (log)")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"segment_analysis_{short}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Збережено: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# ВІЗУАЛІЗАЦІЯ — порівняльні гістограми (пікселі → ДКП → після Q)
# ══════════════════════════════════════════════════════════════════════════════

def plot_histogram_comparison(name, block):
    """
    4 панелі:
      ① гістограма пікселів
      ② ДКП ДО квантування (всі 64 коефіцієнти, DC завжди додатній)
      ③ ДКП ПІСЛЯ Q=10
      ④ ДКП ПІСЛЯ Q=90
    """
    short     = name.split(" ")[0]
    dct_block = dct2(block.astype(np.float32))  # без центрування → DC > 0

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"Порівняння гістограм: пікселі → ДКП → після квантування\nСегмент: «{name}»",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # ① пікселі
    pixels = block.flatten().astype(float)
    axes[0].hist(pixels, bins=16, color="#2c7bb6", edgecolor="white", alpha=0.85)
    axes[0].set_title("① Гістограма яскравості\nпікселів (0–255)", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Яскравість")
    axes[0].set_ylabel("Кількість")
    axes[0].grid(True, alpha=0.3)
    _stat_box(axes[0], pixels)

    # ② ДКП ДО квантування — всі коефіцієнти з DC
    raw = dct_block.flatten().astype(float)
    axes[1].hist(raw, bins=20, color="#d7191c", edgecolor="white", alpha=0.85)
    axes[1].set_title("② ДКП ДО квантування\n(всі коефіцієнти, з DC)", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Значення ДКП-коефіцієнта")
    axes[1].set_ylabel("Кількість")
    axes[1].grid(True, alpha=0.3)
    _stat_box(axes[1], raw)

    # ③ ④ після квантування
    colors = {10: "#e66101", 90: "#1a9641"}
    syms   = {10: "③", 90: "④"}
    for col_idx, q in enumerate([10, 90], start=2):
        qmat  = get_q_matrix(q)
        qblk  = quantize(dct_block, qmat).flatten().astype(float)
        zeros = 100.0 * np.sum(qblk == 0) / len(qblk)
        axes[col_idx].hist(qblk, bins=min(len(np.unique(qblk)) + 1, 30),
                           color=colors[q], edgecolor="white", alpha=0.85)
        axes[col_idx].set_title(
            f"{syms[q]} ДКП ПІСЛЯ Q={q}\n({zeros:.0f}% нулів)",
            fontsize=10, fontweight="bold",
        )
        axes[col_idx].set_xlabel("Квантований коефіцієнт")
        axes[col_idx].set_ylabel("Кількість")
        axes[col_idx].grid(True, alpha=0.3)
        _stat_box(axes[col_idx], qblk)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"hist_compare_{short}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Збережено: {path}")


def _stat_box(ax, data):
    flat = data.flatten().astype(float)
    txt  = (f"μ={flat.mean():.1f}\nσ={flat.std():.1f}\n"
            f"min={flat.min():.0f}, max={flat.max():.0f}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


# ══════════════════════════════════════════════════════════════════════════════
# АНАЛІЗ: різні розміри блоків (8×8, 16×16, 32×32)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_block_sizes(img_gray):
    """Порівнює PSNR та CR для різних розмірів блоків та якостей."""
    # беремо центральну ділянку зображення
    h, w   = img_gray.shape
    cy, cx = h // 2, w // 2
    rows   = []

    for bs in BLOCK_SIZES:
        r0, c0   = cy - bs // 2, cx - bs // 2
        block    = img_gray[r0:r0 + bs, c0:c0 + bs].astype(np.float32)
        dct_blk  = dct2(block)

        for q in Q_FACTORS:
            qmat     = get_q_matrix_for_block(q, bs)
            quant    = quantize(dct_blk, qmat)
            cr       = compression_ratio(quant, bs)
            recon    = np.clip(idct2(quant.astype(np.float32) * qmat), 0, 255).astype(np.uint8)
            orig_u8  = block.astype(np.uint8)
            psnr_val = psnr_metric(orig_u8, recon)
            ssim_val = ssim_metric(orig_u8, recon)
            nonzero  = int(np.count_nonzero(quant))
            zeros_pct = 100.0 * (bs * bs - nonzero) / (bs * bs)

            rows.append({
                "Розмір блоку": f"{bs}×{bs}",
                "Q": q,
                "Нулів (%)": round(zeros_pct, 1),
                "CR": round(cr, 2),
                "PSNR (дБ)": round(psnr_val, 2),
                "SSIM": round(ssim_val, 4),
            })

    df = pd.DataFrame(rows)
    print("\nПорівняння розмірів блоків:")
    print(df.to_string(index=False))
    return df


def plot_block_size_comparison(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Вплив розміру блоку ДКП на якість та стиснення", fontsize=13, fontweight="bold")

    colors = {8: "#1f77b4", 16: "#ff7f0e", 32: "#2ca02c"}
    for bs in BLOCK_SIZES:
        sub = df[df["Розмір блоку"] == f"{bs}×{bs}"]
        axes[0].plot(sub["Q"], sub["PSNR (дБ)"], marker="o",
                     color=colors[bs], label=f"{bs}×{bs}")
        axes[1].plot(sub["Q"], sub["CR"], marker="s",
                     color=colors[bs], label=f"{bs}×{bs}")

    axes[0].set_title("PSNR (дБ) від якості Q")
    axes[0].set_xlabel("Якість Q"); axes[0].set_ylabel("PSNR (дБ)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Коефіцієнт стиснення від Q")
    axes[1].set_xlabel("Якість Q"); axes[1].set_ylabel("CR (×)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "block_size_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Збережено: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# ГОЛОВНИЙ АНАЛІЗ — таблиця метрик по сегментах
# ══════════════════════════════════════════════════════════════════════════════

def analyze_segments(img_gray):
    results = []
    for name, (y, x) in SEGMENT_COORDS.items():
        block    = img_gray[y:y + 8, x:x + 8].astype(np.float32)
        dct_blk  = dct2(block)  # DC > 0, без центрування

        for q in Q_FACTORS:
            qmat    = get_q_matrix(q)
            quant   = quantize(dct_blk, qmat)
            cr      = compression_ratio(quant, 8)
            recon   = np.clip(idct2(quant.astype(np.float32) * qmat), 0, 255).astype(np.uint8)
            orig_u8 = block.astype(np.uint8)

            codes    = get_huffman_codes(quant)
            ent, avg = calc_entropy_metrics(quant, codes)
            psnr_val = psnr_metric(orig_u8, recon)
            ssim_val = ssim_metric(orig_u8, recon)
            zeros    = 100.0 * np.sum(quant == 0) / 64

            results.append({
                "Сегмент": name, "Q": q,
                "Нулів (%)": round(zeros, 1),
                "CR": round(cr, 2),
                "Ентропія": round(ent, 3),
                "Сер. довж. коду": round(avg, 3),
                "PSNR (дБ)": round(psnr_val, 2),
                "SSIM": round(ssim_val, 4),
            })

    df = pd.DataFrame(results)
    print("\nКомплексні результати тестування компресії:")
    print(df.to_string(index=False))
    return df


def plot_quality_vs_cr(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Залежність якості та ентропії від ступеня стиснення", fontsize=13)

    cmap   = {"Фон (рівномірний)": "#1a9641",
               "Текстура (регулярна)": "#d7191c",
               "Деталі (високочастотні)": "#2c7bb6"}
    styles = {"Фон (рівномірний)": "o-",
               "Текстура (регулярна)": "s--",
               "Деталі (високочастотні)": "^:"}

    for name in SEGMENT_COORDS:
        sub = df[df["Сегмент"] == name]
        axes[0].plot(sub["Сер. довж. коду"], sub["PSNR (дБ)"],
                     styles[name], color=cmap[name], label=name)
        axes[1].plot(sub["Q"], sub["Ентропія"],
                     styles[name], color=cmap[name], label=name)

    axes[0].set_xlabel("Інформаційна ємність (біт/піксель)")
    axes[0].set_ylabel("PSNR (дБ)")
    axes[0].set_title("PSNR від середньої довжини коду")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Якість Q")
    axes[1].set_ylabel("Ентропія (біт)")
    axes[1].set_title("Ентропія ДКП від якості Q")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "quality_vs_compression.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Збережено: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Завантаження зображення…")
    img_bgr, img_gray = load_image()
    print(f"  {img_gray.shape[1]}×{img_gray.shape[0]} пікселів")

    print("\n1. Карта сегментів…")
    plot_segment_map(img_bgr)

    print("\n2. Аналіз кожного сегменту…")
    for name, (y, x) in SEGMENT_COORDS.items():
        block = img_gray[y:y + 8, x:x + 8]
        plot_segment_analysis(name, block)
        plot_histogram_comparison(name, block)

    print("\n3. Таблиця метрик (8×8 блоки)…")
    df_seg = analyze_segments(img_gray)

    print("\n4. Графіки якості та ентропії…")
    plot_quality_vs_cr(df_seg)

    print("\n5. Порівняння розмірів блоків (8×8, 16×16, 32×32)…")
    df_bs = analyze_block_sizes(img_gray)
    plot_block_size_comparison(df_bs)

    print(f"\nВсі результати збережено у: {OUT_DIR}")


if __name__ == "__main__":
    main()
