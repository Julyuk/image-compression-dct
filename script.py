import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
from collections import Counter
import heapq

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "I04.BMP")
BLOCK_SIZE = 8
QUALITY_FACTORS = [10, 25, 50, 75, 90]

# ──────────────────────────────────────────────
# Допоміжні функції
# ──────────────────────────────────────────────

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ──────────────────────────────────────────────
# ДКП (Дискретне косинусне перетворення)
# ──────────────────────────────────────────────

def dct2(block: np.ndarray) -> np.ndarray:
    return cv2.dct(block.astype(np.float32))


def idct2(block: np.ndarray) -> np.ndarray:
    return cv2.idct(block.astype(np.float32))


# Базова матриця квантування JPEG (яскравість, Q=50)
JPEG_Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float32)


def get_quantization_matrix(quality: int) -> np.ndarray:
    """Масштабує стандартну матрицю квантування JPEG за фактором якості (1–100)."""
    q = max(1, min(100, quality))
    scale = 5000 / q if q < 50 else 200 - 2 * q
    qm = np.floor((JPEG_Q50 * scale + 50) / 100).astype(np.float32)
    return np.clip(qm, 1, 255)


def quantize(dct_block: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    return np.round(dct_block / q_matrix).astype(np.int32)


def dequantize(q_block: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    return (q_block * q_matrix).astype(np.float32)


def process_image_dct(gray: np.ndarray, block_size: int = 8, quality: int = 50):
    """
    Виконує ДКП → квантування → зворотне квантування → зворотне ДКП.
    Повертає: відновлене зображення, масив квантованих коефіцієнтів, матрицю квантування.
    """
    h, w = gray.shape
    h_pad = (block_size - h % block_size) % block_size
    w_pad = (block_size - w % block_size) % block_size
    padded = np.pad(gray.astype(np.float32), ((0, h_pad), (0, w_pad)), mode="edge") - 128.0
    hp, wp = padded.shape
    q_matrix = get_quantization_matrix(quality)

    all_coeffs = []
    reconstructed = np.zeros((hp, wp), dtype=np.float32)

    for i in range(0, hp, block_size):
        for j in range(0, wp, block_size):
            block = padded[i:i + block_size, j:j + block_size]
            dct_block = dct2(block)
            q_block = quantize(dct_block, q_matrix)
            all_coeffs.extend(q_block.flatten().tolist())
            rec_block = idct2(dequantize(q_block, q_matrix))
            reconstructed[i:i + block_size, j:j + block_size] = rec_block

    reconstructed = np.clip(reconstructed + 128.0, 0, 255).astype(np.uint8)[:h, :w]
    return reconstructed, np.array(all_coeffs, dtype=np.int32), q_matrix


# ──────────────────────────────────────────────
# Крок 5: Кодування Хаффмана
# ──────────────────────────────────────────────

class _HNode:
    __slots__ = ("sym", "freq", "left", "right")

    def __init__(self, sym, freq):
        self.sym = sym
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def _build_tree(freq_dict: dict) -> "_HNode":
    heap = [_HNode(s, f) for s, f in freq_dict.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        l = heapq.heappop(heap)
        r = heapq.heappop(heap)
        merged = _HNode(None, l.freq + r.freq)
        merged.left = l
        merged.right = r
        heapq.heappush(heap, merged)
    return heap[0] if heap else None


def _collect_codes(node: "_HNode", prefix: str = "", codes: dict = None) -> dict:
    if codes is None:
        codes = {}
    if node is None:
        return codes
    if node.sym is not None:
        codes[node.sym] = prefix or "0"
        return codes
    _collect_codes(node.left, prefix + "0", codes)
    _collect_codes(node.right, prefix + "1", codes)
    return codes


def huffman_encode(coefficients: np.ndarray):
    """
    Будує таблицю частот, дерево Хаффмана та словник кодів.
    Повертає: словник кодів, загальну кількість закодованих бітів, частотний словник.
    """
    freq = Counter(coefficients.tolist())
    if len(freq) == 1:
        sym = next(iter(freq))
        return {sym: "0"}, len(coefficients), freq

    tree = _build_tree(freq)
    codes = _collect_codes(tree)
    encoded_bits = sum(len(codes[s]) * cnt for s, cnt in freq.items())
    return codes, encoded_bits, freq


# ──────────────────────────────────────────────
# Крок 6: Рівень стиснення та PSNR
# ──────────────────────────────────────────────

def compression_stats(n_pixels: int, encoded_bits: int):
    original_bits = n_pixels * 8
    ratio = original_bits / encoded_bits if encoded_bits > 0 else 0.0
    return ratio, original_bits, encoded_bits


def calc_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse_val = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse_val < 1e-12:
        return float("inf")
    return 10.0 * math.log10(255.0 ** 2 / mse_val)


def calc_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))


def calc_ssim(original: np.ndarray, recon: np.ndarray) -> float:
    """Спрощений SSIM (без windowed)."""
    x = original.astype(np.float64)
    y = recon.astype(np.float64)
    C1, C2 = 6.5025, 58.5225  # (0.01*255)^2, (0.03*255)^2
    mx, my = x.mean(), y.mean()
    sx = x.std() ** 2
    sy = y.std() ** 2
    sxy = np.mean((x - mx) * (y - my))
    num = (2 * mx * my + C1) * (2 * sxy + C2)
    den = (mx ** 2 + my ** 2 + C1) * (sx + sy + C2)
    return float(num / den)


# ──────────────────────────────────────────────
# Крок 7: Оцінка якості відновлених зображень
# ──────────────────────────────────────────────

def evaluate_quality(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Обчислює набір метрик якості відновленого зображення."""
    p = calc_psnr(original, reconstructed)
    m = calc_mse(original, reconstructed)
    s = calc_ssim(original, reconstructed)
    mae = float(np.mean(np.abs(original.astype(np.float64) - reconstructed.astype(np.float64))))
    max_err = float(np.max(np.abs(original.astype(np.int32) - reconstructed.astype(np.int32))))
    return {"PSNR (дБ)": p, "MSE": m, "RMSE": math.sqrt(m), "MAE": mae,
            "Max error": max_err, "SSIM": s}


# ──────────────────────────────────────────────
# Головна функція
# ──────────────────────────────────────────────

def run_lab3(image_path: str, out_dir: str, quality_factors: list = None):
    if quality_factors is None:
        quality_factors = QUALITY_FACTORS
    ensure_dir(out_dir)

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Зображення не знайдено: {image_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(out_dir, "00_original.png"), gray)

    results = {}
    reconstructed_map = {}

    print("=" * 55)
    print("  Лабораторна робота №3: ДКП + Хаффман + PSNR")
    print("=" * 55)

    for q in quality_factors:
        rec, coeffs, q_matrix = process_image_dct(gray, BLOCK_SIZE, q)

        # Крок 5: Кодування Хаффмана
        codes, enc_bits, freq = huffman_encode(coeffs)

        # Крок 6: Рівень стиснення та PSNR
        ratio, orig_bits, _ = compression_stats(gray.size, enc_bits)
        psnr_val = calc_psnr(gray, rec)

        # Крок 7: Повна оцінка якості
        quality_metrics = evaluate_quality(gray, rec)

        # Ентропія Хаффмана
        total = len(coeffs)
        entropy_val = -sum((c / total) * math.log2(c / total) for c in freq.values() if c > 0)
        avg_code_len = enc_bits / total if total > 0 else 0

        results[q] = {
            "rec": rec,
            "codes": codes,
            "enc_bits": enc_bits,
            "orig_bits": orig_bits,
            "ratio": ratio,
            "psnr": psnr_val,
            "quality_metrics": quality_metrics,
            "num_symbols": len(freq),
            "entropy": entropy_val,
            "avg_code_len": avg_code_len,
            "freq": freq,
        }
        reconstructed_map[q] = rec

        cv2.imwrite(os.path.join(out_dir, f"rec_q{q:03d}.png"), rec)

        print(f"\n  Q = {q}")
        print(f"    Унікальних символів:  {len(freq)}")
        print(f"    Ентропія:             {entropy_val:.4f} біт/симв")
        print(f"    Сер. довжина коду:    {avg_code_len:.4f} біт/симв")
        print(f"    Оригінал (біт):       {orig_bits}")
        print(f"    Стиснено (біт):       {enc_bits}")
        print(f"    Коеф. стиснення:      {ratio:.4f}x")
        for k, v in quality_metrics.items():
            print(f"    {k:<20}: {v:.4f}")

    # ── Рисунок 1: відновлені зображення ──────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Оригінал", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for idx, q in enumerate(quality_factors):
        r = results[q]
        ax = axes[positions[idx][0], positions[idx][1]]
        ax.imshow(reconstructed_map[q], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Q={q}  PSNR={r['psnr']:.1f} дБ\nСтиснення={r['ratio']:.2f}x", fontsize=10)
        ax.axis("off")

    plt.suptitle("Рисунок 1. Відновлені зображення при різних факторах якості", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_reconstructed.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Рисунок 2: PSNR vs Q ──────────────────────────────
    qs = list(results.keys())
    psnrs = [results[q]["psnr"] for q in qs]

    plt.figure(figsize=(9, 5))
    plt.plot(qs, psnrs, "b-o", linewidth=2, markersize=8)
    for q, p in zip(qs, psnrs):
        plt.annotate(f"{p:.1f}", (q, p), textcoords="offset points", xytext=(5, 6), fontsize=10)
    plt.xlabel("Фактор якості Q", fontsize=12)
    plt.ylabel("PSNR (дБ)", fontsize=12)
    plt.title("Рисунок 2. Залежність PSNR від фактора якості", fontsize=13)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_psnr_vs_quality.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Рисунок 3: Коефіцієнт стиснення vs Q ─────────────
    ratios = [results[q]["ratio"] for q in qs]

    plt.figure(figsize=(9, 5))
    plt.plot(qs, ratios, "r-s", linewidth=2, markersize=8)
    for q, r in zip(qs, ratios):
        plt.annotate(f"{r:.2f}x", (q, r), textcoords="offset points", xytext=(5, 6), fontsize=10)
    plt.xlabel("Фактор якості Q", fontsize=12)
    plt.ylabel("Коефіцієнт стиснення", fontsize=12)
    plt.title("Рисунок 3. Залежність коефіцієнта стиснення від фактора якості", fontsize=13)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_compression_ratio.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Рисунок 4: Розподіл довжин кодів Хаффмана (Q=50) ─
    demo_q = 50
    demo_codes = results[demo_q]["codes"]
    lengths = [len(v) for v in demo_codes.values()]

    plt.figure(figsize=(9, 5))
    plt.hist(lengths, bins=range(1, max(lengths) + 2), edgecolor="black", alpha=0.75, color="steelblue")
    plt.xlabel("Довжина коду Хаффмана (біт)", fontsize=12)
    plt.ylabel("Кількість символів", fontsize=12)
    plt.title(f"Рисунок 4. Розподіл довжин кодів Хаффмана (Q={demo_q})", fontsize=13)
    plt.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_huffman_code_lengths.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Рисунок 5: Карти помилок (Q=10 і Q=90) ───────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for idx, q in enumerate([10, 90]):
        diff = np.abs(gray.astype(np.float64) - reconstructed_map[q].astype(np.float64))
        im = axes[idx].imshow(diff, cmap="hot", vmin=0, vmax=60)
        axes[idx].set_title(f"Q={q}: Карта абсолютних помилок\n"
                            f"PSNR={results[q]['psnr']:.1f} дБ, Max err={results[q]['quality_metrics']['Max error']:.0f}",
                            fontsize=11)
        axes[idx].axis("off")
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    plt.suptitle("Рисунок 5. Карти абсолютних помилок відновлення", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "05_error_maps.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Рисунок 6: Гістограми оригінал vs відновлений ────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(gray.flatten(), bins=256, range=(0, 255), color="royalblue", alpha=0.8)
    axes[0].set_title("Гістограма оригінального зображення", fontsize=12)
    axes[0].set_xlabel("Яскравість")
    axes[0].set_ylabel("Кількість пікселів")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(reconstructed_map[50].flatten(), bins=256, range=(0, 255), color="seagreen", alpha=0.8)
    axes[1].set_title("Гістограма відновленого зображення (Q=50)", fontsize=12)
    axes[1].set_xlabel("Яскравість")
    axes[1].set_ylabel("Кількість пікселів")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Рисунок 6. Порівняння гістограм оригіналу та відновленого зображення", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "06_histograms.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Рисунок 7: SSIM, MSE, MAE vs Q ───────────────────
    metrics_names = ["SSIM", "MSE", "RMSE", "MAE"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = ["mediumorchid", "tomato", "darkorange", "steelblue"]
    for ax, mname, col in zip(axes.flat, metrics_names, colors):
        vals = [results[q]["quality_metrics"][mname if mname != "RMSE" else "RMSE"] for q in qs]
        ax.plot(qs, vals, color=col, marker="o", linewidth=2, markersize=8)
        for q, v in zip(qs, vals):
            ax.annotate(f"{v:.3f}", (q, v), textcoords="offset points", xytext=(4, 5), fontsize=9)
        ax.set_title(mname, fontsize=12)
        ax.set_xlabel("Фактор якості Q")
        ax.grid(True, alpha=0.35)
    plt.suptitle("Рисунок 7. Метрики якості відновленого зображення залежно від Q", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "07_quality_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Рисунок 8: Залежність ентропії та сер. довжини коду ─
    entropies = [results[q]["entropy"] for q in qs]
    avg_lens = [results[q]["avg_code_len"] for q in qs]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(qs, entropies, "b-o", linewidth=2, markersize=8, label="Ентропія (біт/симв)")
    ax1.set_xlabel("Фактор якості Q", fontsize=12)
    ax1.set_ylabel("Ентропія (біт/символ)", fontsize=12, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2 = ax1.twinx()
    ax2.plot(qs, avg_lens, "r-s", linewidth=2, markersize=8, label="Сер. довжина коду")
    ax2.set_ylabel("Середня довжина коду Хаффмана", fontsize=12, color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left")
    plt.title("Рисунок 8. Ентропія та середня довжина коду Хаффмана залежно від Q", fontsize=12)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "08_entropy_vs_codelength.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Текстовий звіт ────────────────────────────────────
    report_path = os.path.join(out_dir, "results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Лабораторна робота №3 — Результати\n")
        f.write("=" * 55 + "\n")
        for q in quality_factors:
            r = results[q]
            f.write(f"\nФактор якості Q = {q}\n")
            f.write(f"  Унікальних символів:      {r['num_symbols']}\n")
            f.write(f"  Ентропія:                 {r['entropy']:.4f} біт/симв\n")
            f.write(f"  Сер. довжина коду:        {r['avg_code_len']:.4f} біт/симв\n")
            f.write(f"  Оригінальний розмір:      {r['orig_bits']} біт\n")
            f.write(f"  Стиснений розмір:         {r['enc_bits']} біт\n")
            f.write(f"  Коеф. стиснення:          {r['ratio']:.4f}x\n")
            for k, v in r["quality_metrics"].items():
                f.write(f"  {k:<22}: {v:.4f}\n")

    print(f"\nРезультати збережено у: {out_dir}")
    return results


if __name__ == "__main__":
    run_lab3(IMAGE_PATH, OUT_DIR, QUALITY_FACTORS)
