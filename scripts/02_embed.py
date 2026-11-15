"""
Крок 2. Кодуємо тексти статей моделлю SPECTER2 і зберігаємо вектори в embeddings.npy.

Формат входу -- title + " [SEP] " + abstract: саме так модель навчалася, тому токен [SEP]
обов'язковий. Пулінг -- CLS (див. common.load_model), нормалізація -- L2 (одиничні вектори).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    EMBEDDINGS_PATH,
    MODEL_NAME,
    PARQUET_PATH,
    PROJECT_ROOT,
    build_text,
    load_model,
    section,
)

BATCH_SIZE = 64


def main() -> None:
    if not os.path.exists(PARQUET_PATH):
        sys.exit(f"Немає {PARQUET_PATH}. Спочатку запустіть scripts/01_prepare_data.py")

    section("Крок 2. Ембеддинги SPECTER2")

    df = pd.read_parquet(PARQUET_PATH)
    print(f"Прочитано записів із parquet: {len(df)}")

    # title + [SEP] + abstract -- формат, на якому навчався SPECTER2
    texts = [build_text(row.title, row.abstract) for row in df.itertuples(index=False)]
    print(f"Приклад тексту для кодування:\n  {texts[0][:220]} ...")

    print(f"\nЗавантажую модель {MODEL_NAME} (pooling=cls) ...")
    model = load_model()
    print(f"Максимальна довжина входу: {model.max_seq_length} токенів")
    print(f"Пристрій: {model.device}")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nОброблено текстів:      {len(texts)}")
    print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
    print(f"Норма першого вектора:   {norms[0]:.6f}")
    print(f"Норми (min/max):         {norms.min():.6f} / {norms.max():.6f}")

    os.makedirs(os.path.join(PROJECT_ROOT, "embeddings"), exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    size_mb = os.path.getsize(EMBEDDINGS_PATH) / 1024 / 1024
    print(f"\nЗбережено в {EMBEDDINGS_PATH} ({size_mb:.1f} MB, dtype={embeddings.dtype})")


if __name__ == "__main__":
    main()
