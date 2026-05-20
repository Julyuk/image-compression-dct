"""
Генерація звіту: запускає основний аналіз ДКП та зберігає результати у results/.
Запуск: python generate_report.py
"""
import os
from script import run_lab3

BASE    = os.path.dirname(os.path.abspath(__file__))
IMG     = os.path.join(BASE, "I04.BMP")
OUT_DIR = os.path.join(BASE, "results")

if __name__ == "__main__":
    run_lab3(IMG, OUT_DIR)
    print(f"Звіт збережено у: {OUT_DIR}")
