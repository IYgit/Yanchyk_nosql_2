"""
Крок 6 (Частина 5, додаткова). Гібридний пошук: BM25 + вектори через Reciprocal Rank Fusion.

RRF не порівнює скори різних методів (вони в різних шкалах -- BM25 необмежений, cosine у [-1, 1]),
а працює лише з позиціями в ранжованих списках:

    score(d) = sum_i  1 / (k + rank_i(d)),   rank з 1, k -- константа згладжування.
"""

import os
import re
import sys

import pandas as pd
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
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

TOP_K = 10      # беремо ширше, щоб RRF міг переранжувати
SHOW_K = 5
RRF_K = 60

QUERIES = [
    ("точний термін",                "BERT fine-tuning"),
    ("ім'я автора",                  "Yann LeCun convolutional networks"),
    ("перефразування без термінів",  "making computers understand human emotions from text"),
]

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    return TOKEN_RE.findall(text.lower())


def search_bm25(bm25: BM25Okapi, query: str, top_k: int = TOP_K) -> list:
    """Повертає [(doc_key, score)] за спаданням BM25-скору."""
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [(f"paper_{i}", float(scores[i])) for i in order]


def search_vector(index, model, query: str, top_k: int = TOP_K) -> list:
    """Повертає [(doc_key, cosine_score)] з Pinecone."""
    vector = encode_query(model, query).tolist()
    result = index.query(vector=vector, top_k=top_k, include_metadata=False)
    return [(doc_id, float(score))
            for doc_id, score, _meta in (hit(m) for m in matches_of(result))]


def rrf_fuse(ranked_lists: dict, k: int = RRF_K) -> list:
    """
    Reciprocal Rank Fusion. ranked_lists: {назва_методу: [(doc_key, score), ...]}.
    Повертає [(doc_key, rrf_score, {метод: позиція})] за спаданням RRF-скору.
    """
    fused, positions = {}, {}
    for method, hits in ranked_lists.items():
        for rank, (doc_key, _score) in enumerate(hits, start=1):
            fused[doc_key] = fused.get(doc_key, 0.0) + 1.0 / (k + rank)
            positions.setdefault(doc_key, {})[method] = rank
    order = sorted(fused.items(), key=lambda item: -item[1])
    return [(doc_key, score, positions[doc_key]) for doc_key, score in order]


def row_of(df: pd.DataFrame, doc_key: str) -> pd.Series:
    return df.iloc[int(doc_key.split("_")[1])]


def print_ranking(title: str, hits: list, df: pd.DataFrame, score_fmt: str = "{:.4f}") -> None:
    print(f"\n  {title}")
    if not hits:
        print("    (порожня видача)")
        return
    for rank, entry in enumerate(hits[:SHOW_K], start=1):
        doc_key, score = entry[0], entry[1]
        row = row_of(df, doc_key)
        extra = ""
        if len(entry) > 2:
            extra = "  <- позиції: " + ", ".join(f"{m}#{p}" for m, p in sorted(entry[2].items()))
        print(f"    {rank}. [{score_fmt.format(score)}] {row['title'][:66]}")
        print(f"       {row['category']} | {row['year']} | {row['authors'][:60]}{extra}")


def main() -> None:
    if not os.path.exists(PARQUET_PATH):
        sys.exit(f"Немає {PARQUET_PATH}. Запустіть кроки 01-03 перед цим скриптом.")

    section("Крок 6. Гібридний пошук BM25 + вектори (RRF)")

    df = pd.read_parquet(PARQUET_PATH).reset_index(drop=True)

    # 1. Локальний BM25-індекс за заголовками й анотаціями
    corpus = [tokenize(f"{row.title} {row.abstract}") for row in df.itertuples(index=False)]
    bm25 = BM25Okapi(corpus)
    print(f"BM25-індекс: {len(corpus)} документів, "
          f"{sum(len(d) for d in corpus)} токенів, "
          f"словник {len({t for d in corpus for t in d})} термінів")

    # 2. Pinecone + модель
    pc = get_pinecone()
    index = pc.Index(INDEX_NAME)
    print(f"Pinecone '{INDEX_NAME}': "
          f"{index_stats(index).get('total_vector_count')} векторів")
    print(f"Завантажую модель {MODEL_NAME} (pooling=cls) ...")
    model = load_model()

    summary = []

    # 3. Три демо-запити
    for kind, query in QUERIES:
        section(f"Запит ({kind}): \"{query}\"")

        bm25_hits = search_bm25(bm25, query)
        vector_hits = search_vector(index, model, query)
        hybrid_hits = rrf_fuse({"bm25": bm25_hits, "vector": vector_hits})

        print_ranking("BM25 (top-5):", bm25_hits, df, "{:.2f}")
        print_ranking("Векторний пошук (top-5):", vector_hits, df)
        print_ranking(f"Гібрид RRF, k={RRF_K} (top-5):", hybrid_hits, df, "{:.5f}")

        bm25_top = [d for d, _ in bm25_hits[:SHOW_K]]
        vector_top = [d for d, _ in vector_hits[:SHOW_K]]
        hybrid_top = [d for d, _, _ in hybrid_hits[:SHOW_K]]
        only_hybrid = [d for d in hybrid_top if d not in bm25_top and d not in vector_top]

        both_top_k = {d for d, _ in bm25_hits} & {d for d, _ in vector_hits}
        print(f"\n  Документів, знайдених обома методами в межах top-{TOP_K}: {len(both_top_k)}")
        print(f"  Перетини top-5: BM25 ∩ vector = {len(set(bm25_top) & set(vector_top))}, "
              f"hybrid ∩ BM25 = {len(set(hybrid_top) & set(bm25_top))}, "
              f"hybrid ∩ vector = {len(set(hybrid_top) & set(vector_top))}")
        if only_hybrid:
            print("  Документи, що є лише в top-5 гібриду (у кожного з методів вони були "
                  "нижче 5-ї позиції):")
            for doc_key in only_hybrid:
                row = row_of(df, doc_key)
                pos = dict(hybrid_hits[hybrid_top.index(doc_key)][2])
                print(f"    - {row['title'][:66]} ({pos})")
        else:
            print("  Унікальних для гібриду документів у top-5 немає.")

        summary.append((kind, query, bm25_top, vector_top, hybrid_top))

    # 4. Вплив параметра k
    section("Вплив параметра k у RRF")
    # Беремо запит, де списки BM25 і векторного пошуку ПЕРЕТИНАЮТЬСЯ: лише тоді k
    # щось вирішує. Якщо перетину немає, RRF за будь-якого k просто чергує лідерів.
    kind, query = QUERIES[2]
    print(f"Запит ({kind}): \"{query}\"")
    bm25_hits = search_bm25(bm25, query)
    vector_hits = search_vector(index, model, query)
    both = {d for d, _ in bm25_hits} & {d for d, _ in vector_hits}
    print(f"Документів, які знайшли ОБА методи (у межах top-{TOP_K}): {len(both)}")
    for k in (1, 10, 60, 1000):
        hits = rrf_fuse({"bm25": bm25_hits, "vector": vector_hits}, k=k)
        print(f"\n  k={k}:")
        for rank, (doc_key, score, pos) in enumerate(hits[:SHOW_K], start=1):
            mark = "обидва" if len(pos) == 2 else "один"
            print(f"    {rank}. [{score:.5f}] ({mark}: "
                  f"{', '.join(f'{m}#{p}' for m, p in sorted(pos.items()))}) "
                  f"{row_of(df, doc_key)['title'][:46]}")
    print("\n  За малого k перша позиція одного списку майже недосяжна: 1/(1+1)=0.5 проти")
    print("  1/(1+3)+1/(1+2)=0.58 -- тобто перевагу отримує той, хто просто був №1 десь.")
    print("  За великого k різниця між сусідніми ранками стирається (1/(60+1) ~ 1/(60+3)),")
    print("  і вирішальним стає те, чи знайшли документ ОБА методи, а не на якій позиції.")
    print("  Саме тому k=60 (значення з оригінальної статті Cormack et al.) на практиці")
    print("  працює як 'консенсусний' режим, а k=1 -- як 'почергова видача лідерів'.")

    # 5. Чому авторський запит не працює
    section("Чому запит за іменем автора провалюється в обох методах")
    author_query = QUERIES[1][1]
    print(f"Запит: \"{author_query}\"")
    hits_in_text = sum(1 for row in df.itertuples(index=False)
                       if "lecun" in f"{row.title} {row.abstract}".lower())
    hits_in_authors = sum(1 for row in df.itertuples(index=False)
                          if "lecun" in row.authors.lower())
    print(f"Статей, де 'LeCun' зустрічається в title+abstract: {hits_in_text}")
    print(f"Статей, де 'LeCun' є серед авторів:                {hits_in_authors}")
    print("І BM25, і векторний індекс побудовані на title+abstract, тому поле authors")
    print("для пошуку просто не існує. Перевірка: додаємо authors у BM25-корпус.\n")

    corpus_with_authors = [
        tokenize(f"{row.title} {row.abstract} {row.authors}")
        for row in df.itertuples(index=False)
    ]
    bm25_authors = BM25Okapi(corpus_with_authors)
    hits_plain = search_bm25(bm25, author_query)
    hits_authors = search_bm25(bm25_authors, author_query)
    print_ranking("BM25 по title+abstract:", hits_plain, df, "{:.2f}")
    print_ranking("BM25 по title+abstract+authors:", hits_authors, df, "{:.2f}")
    lecun_plain = sum(1 for d, _ in hits_plain[:SHOW_K] if "lecun" in row_of(df, d)["authors"].lower())
    lecun_authors = sum(1 for d, _ in hits_authors[:SHOW_K] if "lecun" in row_of(df, d)["authors"].lower())
    print(f"\n  Статей саме LeCun у top-5: без authors -- {lecun_plain}, з authors -- {lecun_authors}")

    # 6. Підсумкова таблиця
    section("Підсумок: 3 запити x 3 методи (top-1 кожного методу)")
    for kind, query, bm25_top, vector_top, hybrid_top in summary:
        print(f"\n[{kind}] \"{query}\"")
        for method, top in (("BM25  ", bm25_top), ("Vector", vector_top), ("Hybrid", hybrid_top)):
            row = row_of(df, top[0])
            print(f"  {method}: {row['title'][:72]}")


if __name__ == "__main__":
    main()
