"""
Спільні хелпери для всіх скриптів завдання.

Головна причина існування цього модуля — єдиний спосіб кодування текстів.
Документи (02_embed.py) і запити (04/05/06) мають кодуватися абсолютно однаково,
інакше скалярний добуток у Pinecone втрачає сенс. Тому модель створюється
в одному місці — у load_model().
"""

import os
import sys
import time

# Вивід скриптів містить український текст, а при перенаправленні в файл Windows
# використовує локальну кодову сторінку -> UnicodeEncodeError. Фіксуємо UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
MAX_SEQ_LENGTH = 512

INDEX_NAME = "arxiv-papers"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

PARQUET_PATH = os.path.join(PROJECT_ROOT, "data", "arxiv_subset.parquet")
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "embeddings", "embeddings.npy")


def load_env() -> None:
    """Читає .env з кореня проєкту і доповнює змінні, яких очікують сторонні бібліотеки."""
    from dotenv import load_dotenv

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

    # kaggle/kagglehub очікують KAGGLE_KEY, а в .env ключ названий KAGGLE_API_TOKEN
    if "KAGGLE_API_TOKEN" in os.environ and not os.environ.get("KAGGLE_KEY"):
        os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]


def build_text(title: str, abstract: str) -> str:
    """
    Формат входу, на якому навчався SPECTER2: title + [SEP] + abstract.
    Картка моделі: "concatenate title and abstract with the [SEP] token".
    """
    return f"{title} [SEP] {abstract}"


def load_model():
    """
    Збирає SentenceTransformer вручну, з CLS-пулінгом.

    SentenceTransformer("allenai/specter2_base") спрацював би, але в репозиторії моделі
    немає sentence-transformers конфігурації, тому бібліотека молча додає MEAN-пулінг.
    Картка моделі ж бере саме CLS-токен:
        "take the first token in the batch as the embedding" -> output.last_hidden_state[:, 0, :]
    Тому пулінг задаємо явно.
    """
    from sentence_transformers import SentenceTransformer, models

    word_embedding_model = models.Transformer(MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH)
    # у sentence-transformers 6.x метод перейменовано, у 3.x/5.x живе стара назва
    get_dim = getattr(word_embedding_model, "get_embedding_dimension", None) \
        or word_embedding_model.get_word_embedding_dimension
    pooling_model = models.Pooling(get_dim(), pooling_mode="cls")
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


def encode_query(model, query: str):
    """Кодує текстовий запит тією ж моделлю і тією ж нормалізацією, що й документи."""
    return model.encode(query, normalize_embeddings=True, show_progress_bar=False)


def get_pinecone():
    """Ініціалізує клієнт Pinecone."""
    from pinecone import Pinecone

    load_env()
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        sys.exit("PINECONE_API_KEY не знайдено. Додайте його у .env (див. .env.example).")
    return Pinecone(api_key=api_key)


def list_index_names(pc) -> list:
    """Назви наявних індексів (API трохи різниться між версіями клієнта)."""
    indexes = pc.list_indexes()
    try:
        return list(indexes.names())
    except AttributeError:
        return [idx["name"] for idx in indexes]


def index_stats(index) -> dict:
    """describe_index_stats() як звичайний dict (у різних версіях клієнта різні типи)."""
    stats = index.describe_index_stats()
    for attr in ("to_dict", "_asdict"):
        if hasattr(stats, attr):
            try:
                return dict(getattr(stats, attr)())
            except Exception:
                pass
    if isinstance(stats, dict):
        return dict(stats)
    return {
        "total_vector_count": getattr(stats, "total_vector_count", None),
        "dimension": getattr(stats, "dimension", None),
        "index_fullness": getattr(stats, "index_fullness", None),
    }


def matches_of(result) -> list:
    """Список збігів із відповіді index.query() (dict-доступ або атрибут, залежно від версії)."""
    try:
        return list(result["matches"])
    except (TypeError, KeyError, IndexError):
        return list(getattr(result, "matches", None) or [])


def hit(match):
    """Нормалізує один збіг до (id, score, metadata)."""
    if isinstance(match, dict):
        return match.get("id"), match.get("score"), dict(match.get("metadata") or {})
    try:
        return match["id"], match["score"], dict(match["metadata"] or {})
    except (TypeError, KeyError):
        return (
            getattr(match, "id", None),
            getattr(match, "score", None),
            dict(getattr(match, "metadata", None) or {}),
        )


def ensure_index(pc, name: str, dimension: int = VECTOR_DIM, metric: str = "cosine"):
    """
    Створює serverless-індекс, якщо його ще немає, і чекає готовності.
    Повертає (index, created_flag).
    """
    from pinecone import ServerlessSpec

    existing = list_index_names(pc)
    created = False
    if name not in existing:
        print(f"  Створюю індекс '{name}' (dim={dimension}, metric={metric}) ...")
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        created = True
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
        print(f"  Індекс '{name}' готовий.")
    else:
        print(f"  Індекс '{name}' уже існує — підключаюся.")
    return pc.Index(name), created


def section(title: str) -> None:
    """Заголовок блоку у виводі скрипта."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
