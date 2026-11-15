"""
Крок 4. Три типи пошуку:
  A) чистий семантичний пошук у Pinecone;
  B) пошук із фільтрами за метаданими (категорія + рік);
  C) порівняння метрик схожості (cosine / dot product / L2) на локальних ембеддингах.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    EMBEDDINGS_PATH,
    INDEX_NAME,
    MODEL_NAME,
    PARQUET_PATH,
    encode_query,
    get_pinecone,
    hit,
    index_stats,
    load_model,
    matches_of,
    section,
)

TOP_K = 5

QUERY_SEMANTIC = "teaching machines to recognize objects in pictures"
QUERY_RL = "reinforcement learning agent learns a policy from rewards"
QUERY_METRICS = "neural network for natural language understanding"


def print_hits(matches, abstracts: dict) -> None:
    """Друкує видачу Pinecone; повний abstract підтягуємо з parquet за arxiv_id."""
    if not matches:
        print("  (порожня видача)")
        return
    for rank, match in enumerate(matches, start=1):
        _doc_id, score, meta = hit(match)
        full = abstracts.get(meta["arxiv_id"], meta.get("abstract", ""))
        print(f"  {rank}. [{score:.4f}] {meta['title']}")
        print(f"     {meta['category']} | {int(meta['year'])} | {meta['authors'][:70]}")
        print(f"     {full[:200].strip()}...")


def print_local_top(title: str, order, scores, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    for rank, i in enumerate(order, start=1):
        row = df.iloc[i]
        print(f"  {rank}. [{scores[i]:+.4f}] paper_{i} | {row['category']} | {row['year']} "
              f"| {row['title'][:70]}")


def main() -> None:
    for path in (PARQUET_PATH, EMBEDDINGS_PATH):
        if not os.path.exists(path):
            sys.exit(f"Немає {path}. Запустіть кроки 01-03 перед цим скриптом.")

    df = pd.read_parquet(PARQUET_PATH).reset_index(drop=True)
    abstracts = dict(zip(df["id"], df["abstract"]))   # для повного тексту анотації

    pc = get_pinecone()
    index = pc.Index(INDEX_NAME)
    print(f"Підключено до індексу '{INDEX_NAME}': "
          f"{index_stats(index).get('total_vector_count')} векторів")
    print(f"Завантажую модель {MODEL_NAME} (pooling=cls) ...")
    model = load_model()

    # ------------------------------------------------------------------ #
    # A. Чистий семантичний пошук
    # ------------------------------------------------------------------ #
    section("A. Чистий семантичний пошук (без фільтрів)")
    print(f"Запит: \"{QUERY_SEMANTIC}\"")
    print("Жодне слово запиту не є термінологічним: 'pictures' замість 'images',")
    print("'teaching machines' замість 'training models'. Повнотекстовий пошук тут безсилий.\n")

    query_vector = encode_query(model, QUERY_SEMANTIC)
    result = index.query(vector=query_vector.tolist(), top_k=TOP_K, include_metadata=True)
    print_hits(matches_of(result), abstracts)

    # ------------------------------------------------------------------ #
    # B. Пошук із фільтрами за метаданими
    # ------------------------------------------------------------------ #
    section("B. Пошук із фільтрами за метаданими")
    max_year = int(df["year"].max())
    recent_from = max_year - 4          # "останні 5 років" включно з max_year
    rl_vector = encode_query(model, QUERY_RL)
    print(f"Запит для обох прикладів: \"{QUERY_RL}\"")

    filter_a = {"category": {"$eq": "cs.LG"}, "year": {"$gte": recent_from}}
    print(f"\nПриклад A -- filter={filter_a}  (cs.LG, {recent_from}-{max_year})")
    matches_a = matches_of(index.query(vector=rl_vector.tolist(), top_k=TOP_K,
                                       include_metadata=True, filter=filter_a))
    print_hits(matches_a, abstracts)

    filter_b = {"year": {"$lt": 2015}}
    print(f"\nПриклад B -- filter={filter_b}  (будь-яка категорія, до 2015 року)")
    matches_b = matches_of(index.query(vector=rl_vector.tolist(), top_k=TOP_K,
                                       include_metadata=True, filter=filter_b))
    print_hits(matches_b, abstracts)

    ids_a = {hit(m)[0] for m in matches_a}
    ids_b = {hit(m)[0] for m in matches_b}
    scores_a = [hit(m)[1] for m in matches_a]
    scores_b = [hit(m)[1] for m in matches_b]
    print("\nПорівняння A і B:")
    print(f"  спільних документів: {len(ids_a & ids_b)}")
    if scores_a and scores_b:
        print(f"  середня схожість A: {np.mean(scores_a):.4f} | B: {np.mean(scores_b):.4f}")
    print(f"  категорії A: {sorted({hit(m)[2]['category'] for m in matches_a})}")
    print(f"  категорії B: {sorted({hit(m)[2]['category'] for m in matches_b})}")
    print(f"  роки A: {sorted({int(hit(m)[2]['year']) for m in matches_a})}")
    print(f"  роки B: {sorted({int(hit(m)[2]['year']) for m in matches_b})}")
    print("  Фільтр -- це жорстке обмеження простору пошуку (pre-filter), а не переранжування:")
    print("  кожна видача -- найкраще В МЕЖАХ СВОГО підпростору, тому скори з різних фільтрів")
    print("  не порівнюються між собою як 'релевантність для користувача'. Перетин A і B нульовий")
    print("  за побудовою: умови за роком взаємно виключні.")
    print("  Показово, що середня схожість B не нижча за A, хоча B -- це статті 2002-2014 років:")
    print("  запит сформульований класичною RL-мовою ('agent', 'policy', 'rewards'), і старі")
    print("  роботи описують себе тими самими простими словами, тоді як сучасний cs.LG-зріз")
    print("  вужчий і термінологічніший (MARL, successor features, BAMDP shaping).")
    print("  Висновок: фільтр змінює СКЛАД кандидатів, а не те, як модель міряє схожість.")

    # ------------------------------------------------------------------ #
    # C. Порівняння метрик схожості на локальних ембеддингах
    # ------------------------------------------------------------------ #
    section("C. Порівняння метрик схожості на локальних ембеддингах")
    embeddings = np.load(EMBEDDINGS_PATH)
    q = encode_query(model, QUERY_METRICS)
    print(f"Запит: \"{QUERY_METRICS}\"")
    print(f"Ембеддинги: {embeddings.shape}, норми в [{np.linalg.norm(embeddings, axis=1).min():.4f}, "
          f"{np.linalg.norm(embeddings, axis=1).max():.4f}], норма запиту {np.linalg.norm(q):.4f}")

    dot = embeddings @ q
    cosine = dot / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q))
    l2 = np.linalg.norm(embeddings - q, axis=1)

    top_cos = np.argsort(-cosine)[:TOP_K]
    top_dot = np.argsort(-dot)[:TOP_K]
    top_l2 = np.argsort(l2)[:TOP_K]

    print_local_top("Cosine similarity (більше -- краще):", top_cos, cosine, df)
    print_local_top("Dot product (більше -- краще):", top_dot, dot, df)
    print_local_top("L2 distance (менше -- краще):", top_l2, l2, df)

    print("\nПорівняння списків:")
    print(f"  cosine vs dot: однаковий порядок = {list(top_cos) == list(top_dot)}, "
          f"спільних {len(set(top_cos) & set(top_dot))}/{TOP_K}")
    print(f"  cosine vs L2:  однаковий порядок = {list(top_cos) == list(top_l2)}, "
          f"спільних {len(set(top_cos) & set(top_l2))}/{TOP_K}")
    print(f"  максимальне відхилення |cosine - dot| по всьому корпусу: "
          f"{np.abs(cosine - dot).max():.2e}")
    print(f"  перевірка тотожності L2^2 = 2 - 2*cos: max|різниця| = "
          f"{np.abs(l2 ** 2 - (2 - 2 * cosine)).max():.2e}")

    # Контрприклад: що буде, якщо вектори НЕ нормалізовані
    rng = np.random.default_rng(0)
    scales = rng.uniform(0.5, 3.0, size=len(embeddings)).astype(np.float32)
    scaled = embeddings * scales[:, None]
    dot_scaled = scaled @ q
    cos_scaled = dot_scaled / (np.linalg.norm(scaled, axis=1) * np.linalg.norm(q))
    top_dot_scaled = np.argsort(-dot_scaled)[:TOP_K]
    top_cos_scaled = np.argsort(-cos_scaled)[:TOP_K]
    print("\nКонтрприклад -- ті самі вектори, помножені на випадковий масштаб 0.5-3.0:")
    print(f"  dot    top-5: {[f'paper_{i}' for i in top_dot_scaled]}")
    print(f"  cosine top-5: {[f'paper_{i}' for i in top_cos_scaled]}")
    print(f"  dot зберіг вихідний топ-5: {list(top_dot_scaled) == list(top_cos)}; "
          f"cosine зберіг: {list(top_cos_scaled) == list(top_cos)}")
    print("  Висновок: рівність cosine і dot тримається рівно доти, доки всі вектори одиничні.")


if __name__ == "__main__":
    main()
