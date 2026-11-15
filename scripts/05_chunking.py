"""
Крок 5. Chunking довгих анотацій двома стратегіями + окремі індекси під кожну.

  fixed-size  -- фіксована кількість слів із перекриттям (не зважає на межі речень);
  semantic    -- накопичуємо цілі речення, поки не досягнемо ліміту слів.

Кожен чанк кодується у форматі "title [SEP] chunk": заголовок дає чанку контекст статті,
і це той самий формат входу, на якому навчався SPECTER2.

Якщо безкоштовний тір Pinecone не дозволяє створити додаткові індекси, скрипт автоматично
переходить на namespaces всередині основного індексу -- статті лишаються в default namespace,
тому кроки 4 і 6 не ламаються.
"""

import os
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    INDEX_NAME,
    MODEL_NAME,
    PARQUET_PATH,
    VECTOR_DIM,
    build_text,
    encode_query,
    ensure_index,
    get_pinecone,
    hit,
    load_model,
    matches_of,
    section,
)

N_PAPERS = 30            # статті з найдовшими анотаціями
CHUNK_WORDS = 60         # цільовий розмір чанка у словах
OVERLAP_WORDS = 15       # перекриття для fixed-size
BATCH_SIZE = 100
TOP_K = 5

INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"

TEST_QUERIES = [
    "how the proposed method was evaluated on benchmark datasets",
    "theoretical guarantees and convergence proofs",
    "limitations and future work",
]

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# --------------------------------------------------------------------------- #
# Стратегії розбиття
# --------------------------------------------------------------------------- #
def chunk_fixed(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list:
    """Фіксована кількість слів із перекриттям. Межі речень ігноруються."""
    words = text.split()
    if len(words) <= size:
        return [text.strip()]
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def chunk_semantic(text: str, max_words: int = CHUNK_WORDS) -> list:
    """Накопичуємо цілі речення, поки не перевищимо ліміт слів. Речення не ріжемо."""
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and current_len + n > max_words:
            chunks.append(" ".join(current))
            current, current_len = [sentence], n
        else:
            current.append(sentence)
            current_len += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def safe_id(arxiv_id: str) -> str:
    """Старі arXiv id мають вигляд 'math/0605123' -- приводимо до безпечного для Pinecone вигляду."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(arxiv_id))


def starts_mid_sentence(chunk: str) -> bool:
    """Чанк починається посеред речення, якщо перший значущий символ -- маленька літера."""
    stripped = chunk.lstrip()
    return bool(stripped) and stripped[0].islower()


def ends_mid_sentence(chunk: str) -> bool:
    stripped = chunk.rstrip()
    return bool(stripped) and stripped[-1] not in ".!?"


def chunk_stats(chunks: list) -> str:
    lengths = [len(c.split()) for c in chunks]
    cut_start = sum(starts_mid_sentence(c) for c in chunks)
    cut_end = sum(ends_mid_sentence(c) for c in chunks)
    return (f"чанків {len(chunks)}, слів у чанку: сер. {np.mean(lengths):.1f} "
            f"(min {min(lengths)}, max {max(lengths)}), "
            f"починаються посеред речення {cut_start} ({cut_start / len(chunks):.0%}), "
            f"обриваються посеред речення {cut_end} ({cut_end / len(chunks):.0%})")


# --------------------------------------------------------------------------- #
# Робота з Pinecone
# --------------------------------------------------------------------------- #
class Target:
    """Куда пишемо чанки: або окремий індекс, або namespace в основному індексі."""

    def __init__(self, index, name: str, namespace: str = ""):
        self.index = index
        self.name = name
        self.namespace = namespace

    def label(self) -> str:
        return self.name if not self.namespace else f"{self.name} (namespace '{self.namespace}')"

    def upsert(self, vectors) -> None:
        if self.namespace:
            self.index.upsert(vectors=vectors, namespace=self.namespace)
        else:
            self.index.upsert(vectors=vectors)

    def query(self, vector, top_k: int):
        kwargs = {"vector": vector, "top_k": top_k, "include_metadata": True}
        if self.namespace:
            kwargs["namespace"] = self.namespace
        return self.index.query(**kwargs)


def make_target(pc, preferred_name: str, fallback_namespace: str) -> Target:
    """Пробуємо окремий індекс; якщо тір не дозволяє -- падаємо на namespace."""
    try:
        index, _created = ensure_index(pc, preferred_name, dimension=VECTOR_DIM, metric="cosine")
        return Target(index, preferred_name)
    except Exception as exc:  # ліміт індексів на безкоштовному тірі тощо
        print(f"  Не вдалося створити індекс '{preferred_name}': {type(exc).__name__}: "
              f"{str(exc)[:200]}")
        print(f"  Фолбек: пишу чанки в namespace '{fallback_namespace}' індексу '{INDEX_NAME}'.")
        return Target(pc.Index(INDEX_NAME), INDEX_NAME, namespace=fallback_namespace)


def build_records(papers: pd.DataFrame, chunker, strategy: str) -> list:
    """Створює список словників-чанків (без векторів) для однієї стратегії."""
    records = []
    for _, row in papers.iterrows():
        chunks = chunker(row["abstract"])
        for position, text in enumerate(chunks):
            records.append({
                "id": f"{strategy}_{safe_id(row['id'])}_{position}",
                "text": text,
                "metadata": {
                    "arxiv_id":  str(row["id"]),
                    "title":     row["title"],
                    "text":      text[:1000],
                    "chunk_id":  position,
                    "n_chunks":  len(chunks),
                    "year":      int(row["year"]),
                    "category":  row["category"],
                    "strategy":  strategy,
                },
            })
    return records


def upload(target: Target, records: list, model) -> None:
    texts = [build_text(r["metadata"]["title"], r["text"]) for r in records]
    vectors_array = model.encode(
        texts, batch_size=64, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    for start in tqdm(range(0, len(records), BATCH_SIZE),
                      desc=f"Upsert -> {target.label()}", unit=" батч"):
        batch = [
            {"id": r["id"], "values": vectors_array[start + k].tolist(), "metadata": r["metadata"]}
            for k, r in enumerate(records[start:start + BATCH_SIZE])
        ]
        target.upsert(batch)
    print(f"  Завантажено {len(records)} чанків у {target.label()}")


def search_chunks(target: Target, model, query: str) -> None:
    vector = encode_query(model, query).tolist()
    matches = matches_of(target.query(vector, TOP_K))
    if not matches:
        print("  (порожня видача)")
        return
    for rank, match in enumerate(matches, start=1):
        _doc_id, score, meta = hit(match)
        print(f"  {rank}. [{score:.4f}] {meta['title'][:70]}")
        print(f"     чанк {int(meta['chunk_id']) + 1}/{int(meta['n_chunks'])} | "
              f"{meta['category']} | {int(meta['year'])}")
        print(f"     \"{meta['text'][:180].strip()}...\"")


def main() -> None:
    if not os.path.exists(PARQUET_PATH):
        sys.exit(f"Немає {PARQUET_PATH}. Запустіть кроки 01-03 перед цим скриптом.")

    section("Крок 5. Chunking: fixed-size vs semantic")

    df = pd.read_parquet(PARQUET_PATH).reset_index(drop=True)
    lengths = df["abstract"].str.split().str.len()
    papers = df.loc[lengths.nlargest(N_PAPERS).index].copy()
    print(f"Узято {len(papers)} статей із найдовшими анотаціями: "
          f"{int(lengths[papers.index].min())}-{int(lengths[papers.index].max())} слів "
          f"(медіана по корпусу {int(lengths.median())})")

    fixed_records = build_records(papers, chunk_fixed, "fixed")
    semantic_records = build_records(papers, chunk_semantic, "semantic")

    print(f"\nFIXED (size={CHUNK_WORDS}, overlap={OVERLAP_WORDS}): "
          f"{chunk_stats([r['text'] for r in fixed_records])}")
    print(f"SEMANTIC (max_words={CHUNK_WORDS}):        "
          f"{chunk_stats([r['text'] for r in semantic_records])}")

    print("\nВплив overlap на кількість чанків (fixed-size, ті самі 30 статей):")
    total_words = int(lengths[papers.index].sum())
    for overlap in (0, 15, 30, 45):
        chunks = [c for _, row in papers.iterrows()
                  for c in chunk_fixed(row["abstract"], CHUNK_WORDS, overlap)]
        emitted = sum(len(c.split()) for c in chunks)
        print(f"  overlap={overlap:>2} слів -> чанків {len(chunks):>3}, "
              f"закодовано слів {emitted} ({emitted / total_words:.2f}x від обсягу тексту)")

    print("\nПриклад: перші два чанки однієї статті кожною стратегією")
    example_id = papers.iloc[0]["id"]
    for strategy, records in (("fixed", fixed_records), ("semantic", semantic_records)):
        print(f"\n  [{strategy}] {papers.iloc[0]['title'][:70]}")
        for record in [r for r in records if r["metadata"]["arxiv_id"] == example_id][:2]:
            flags = []
            if starts_mid_sentence(record["text"]):
                flags.append("починається посеред речення")
            if ends_mid_sentence(record["text"]):
                flags.append("обривається посеред речення")
            print(f"    чанк {record['metadata']['chunk_id']}: \"{record['text'][:150]}...\"")
            print(f"      {len(record['text'].split())} слів"
                  f"{' | ' + ', '.join(flags) if flags else ' | межі речень збережені'}")

    # --- Індекси та завантаження ------------------------------------------- #
    section("Створення індексів і завантаження чанків")
    pc = get_pinecone()
    print(f"Завантажую модель {MODEL_NAME} (pooling=cls) ...")
    model = load_model()

    target_fixed = make_target(pc, INDEX_FIXED, "chunks-fixed")
    target_semantic = make_target(pc, INDEX_SEMANTIC, "chunks-semantic")

    upload(target_fixed, fixed_records, model)
    upload(target_semantic, semantic_records, model)

    # --- Пошук по чанках ---------------------------------------------------- #
    section("Пошук по чанках: порівняння стратегій")
    for query in TEST_QUERIES:
        print(f"\nЗапит: \"{query}\"")
        print(f"\n  --- {target_fixed.label()} ---")
        search_chunks(target_fixed, model, query)
        print(f"\n  --- {target_semantic.label()} ---")
        search_chunks(target_semantic, model, query)


if __name__ == "__main__":
    main()
