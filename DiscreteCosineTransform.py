import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from collections import Counter
import heapq
from google.colab import files
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import warnings

warnings.filterwarnings('ignore')

# функція для генерації кодів Хаффмана
def get_huffman_codes(data):
    freq = Counter(data.flatten().astype(int))
    if not freq: return {}
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo, hi = heapq.heappop(heap), heapq.heappop(heap)
        for pair in lo[1:]: pair[1] = '0' + pair[1]
        for pair in hi[1:]: pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return dict(heapq.heappop(heap)[1:])

# розрахунок ентропії та середньої довжини коду
def calc_metrics(data, codes):
    total = data.size
    freq = Counter(data.flatten().astype(int))
    probs = {s: f/total for s, f in freq.items()}
    entropy = -sum(p * np.log2(p) for p in probs.values() if p > 0)
    avg_len = sum(probs[s] * len(codes[s]) for s in probs.keys())
    return entropy, avg_len

# матриця квантування за стандартом JPEG
def get_q_matrix(q):
    base = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    return base * q

# завантаження зображення
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
img_bgr = cv2.imread(img_path)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# координати сегментів: фон, текстура, деталі
coords = {'Фон (рівномірний)': (20, 20), 'Текстура (регулярна)': (140, 220), 'Деталі (високочастотні)': (250, 240)}
segments = {n: img_gray[y:y+8, x:x+8] for n, (y, x) in coords.items()}

# візуалізація вибору зон на оригіналі
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
for n, (y, x) in coords.items():
    plt.gca().add_patch(plt.Rectangle((x, y), 8, 8, color='red', fill=False, lw=2))
    plt.text(x, y-5, n, color='white', backgroundcolor='red', fontsize=9)
plt.title('Карта вибору контрольних сегментів')
plt.axis('off')
plt.show()

results_list = []
q_factors = [0.1, 1.0, 5.0, 10.0]

# аналіз кожного обраного блоку
for name, block in segments.items():
    # аналіз спектру та гістограми
    dct_orig = cv2.dct(block.astype(np.float32))
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Статистичний аналіз структури: {name}', fontsize=14)
    ax[0].imshow(block, cmap='gray', vmin=0, vmax=255); ax[0].set_title('Вихідний блок 8x8')
    ax[1].hist(block.flatten(), bins=16, color='darkblue'); ax[1].set_title('Розподіл інтенсивностей')
    ax[2].imshow(np.log(np.abs(dct_orig)+1), cmap='plasma'); ax[2].set_title('Енергетичний спектр ДКП')
    plt.show()

    # цикл дослідження впливу квантування
    for q in q_factors:
        q_mat = get_q_matrix(q)
        quantized = np.round(dct_orig / q_mat) # квантування спектру
        
        # кодування квантованих значень Хаффманом
        q_codes = get_huffman_codes(quantized)
        q_ent, q_avg = calc_metrics(quantized, q_codes)
        
        # відновлення зображення із втратами
        reconstructed = cv2.idct(quantized * q_mat)
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        
        # розрахунок метрик якості відновлення
        psnr_val = psnr_metric(block, reconstructed)
        ssim_val = ssim_metric(block, reconstructed)
        
        results_list.append({
            'Сегмент': name, 'Фактор Q': q, 'Ентропія ДКП': q_ent, 
            'Довжина коду': q_avg, 'Якість PSNR': psnr_val, 'Схожість SSIM': ssim_val
        })

# вивід фінальної таблиці з метриками
df = pd.DataFrame(results_list)
print("\nКомплексні результати тестування компресії:")
display(df.round(4))

# графік залежності якості від ступеня стиснення
plt.figure(figsize=(10, 6))
for name in coords.keys():
    data = df[df['Сегмент'] == name]
    plt.plot(data['Довжина коду'], data['Якість PSNR'], marker='s', linestyle='--', label=name)
plt.xlabel('Інформаційна ємність (біт/піксель)')
plt.ylabel('Рівень якості PSNR (дБ)')
plt.title('Порівняльна характеристика ефективності стиснення')
plt.legend(); plt.grid(True)
plt.show()