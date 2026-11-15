"""
Крок 3. Створюємо індекс arxiv-papers у Pinecone і завантажуємо вектори з метаданими.

Метрика індексу -- cosine (обґрунтування у README, Частина 1).
abstract у метаданих обрізаний до 500 символів: Pinecone обмежує метадані одного вектора
40 KB, повний текст лишається в parquet і підтягується за arxiv_id після пошуку.
"""

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    EMBEDDINGS_PATH,
    INDEX_NAME,
    PARQUET_PATH,
    VECTOR_DIM,
    ensure_index,
    get_pinecone,
    index_stats,
    section,
)

BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів
ABSTRACT_LIMIT = 500
AUTHORS_LIMIT = 200


def main() -> None:
    for path in (PARQUET_PATH, EMBEDDINGS_PATH):
        if not os.path.exists(path):
            sys.exit(f"Немає {path}. Запустіть кроки 01 і 02 перед цим скриптом.")

    section("Крок 3. Завантаження в Pinecone")

    df = pd.read_parquet(PARQUET_PATH).reset_index(drop=True)
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Записів у parquet: {len(df)}, ембеддингів: {embeddings.shape}")

    if len(df) != len(embeddings):
        sys.exit("Кількість записів і векторів не збігається -- перезапустіть 02_embed.py")
    if embeddings.shape[1] != VECTOR_DIM:
        sys.exit(f"Очікувалася розмірність {VECTOR_DIM}, отримано {embeddings.shape[1]}")

    pc = get_pinecone()
    index, created = ensure_index(pc, INDEX_NAME, dimension=VECTOR_DIM, metric="cosine")

    if not created:
        existing = index_stats(index).get("total_vector_count", 0)
        if existing:
            print(f"  В індексі вже є {existing} векторів -- роблю upsert поверх (ті самі id).")

    # --- Батчеве завантаження ---------------------------------------------- #
    total = 0
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Upsert у Pinecone", unit=" батч"):
        chunk = df.iloc[start:start + BATCH_SIZE]
        vectors = []
        for offset, (_, row) in enumerate(chunk.iterrows()):
            i = start + offset
            vectors.append({
                "id": f"paper_{i}",
                "values": embeddings[i].tolist(),
                "metadata": {
                    "arxiv_id": str(row["id"]),
                    "title":    row["title"],
                    "abstract": row["abstract"][:ABSTRACT_LIMIT],
                    "authors":  row["authors"][:AUTHORS_LIMIT],
                    "year":     int(row["year"]),
                    "category": row["category"],
                },
            })
        index.upsert(vectors=vectors)
        total += len(vectors)

    print(f"\nВідправлено векторів: {total}")

    stats = index_stats(index)
    print(f"Усього векторів в індексі '{INDEX_NAME}': {stats.get('total_vector_count')}")
    print(f"Розмірність індексу: {stats.get('dimension')}")
    print(f"Заповненість (index_fullness): {stats.get('index_fullness')}")

    sample = index.fetch(ids=["paper_0"])
    vector = sample.vectors.get("paper_0") if hasattr(sample, "vectors") else None
    if vector is not None:
        print("\nПриклад збережених метаданих (paper_0):")
        for key, value in vector.metadata.items():
            text = str(value)
            print(f"  {key}: {text[:120]}{'...' if len(text) > 120 else ''}")


if __name__ == "__main__":
    main()
