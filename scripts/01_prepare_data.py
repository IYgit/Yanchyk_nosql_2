"""
Крок 1. Читаємо JSONL-снапшот arXiv, чистимо і зберігаємо підмножину в parquet.

Відхилення від скелета із завдання (свідоме, пояснене в README):
скелет бере ПЕРШІ 10 000 рядків файлу. Снапшот відсортований за arXiv id, тому перші
10 000 записів -- це квітень-травень 2007 року, майже виключно physics/math і нуль cs.LG.
На такій підмножині фільтри Частини 3 ("cs.LG за останні 5 років") повертають порожню
видачу, а демо-запити Частини 5 (BERT, LeCun, емоції в тексті) не мають релевантних статей.
Тому робимо стратифіковану вибірку за (група категорій x роковий бакет) через reservoir
sampling в один потоковий прохід по файлу.
"""

import json
import os
import random
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PROJECT_ROOT, load_env  # noqa: E402

SNAPSHOT_NAME = "arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "arxiv_subset.parquet")
KAGGLE_DATASET = "Cornell-University/arxiv"

MAX_RECORDS = 10_000
SEED_AUTHOR = "lecun"      # цільовий seed для авторського запиту з Частини 5
SEED_AUTHOR_QUOTA = 30
EXTRA_POOL = 15_000        # резерв, з якого добираємо, якщо якась комірка недобрана
RANDOM_SEED = 42

# Скільки статей хочемо з кожної групи категорій
GROUP_QUOTAS = {
    "cs.LG":    1600,
    "cs.CL":    1200,
    "cs.CV":    1200,
    "cs.AI":     600,
    "cs.other":  900,
    "stat":      500,
    "physics":  1600,
    "math":     1400,
    "other":     970,
}

# Як розкладаємо квоту групи за роками:
# 30% до 2015 року (для фільтра B), 50% після 2018 (для фільтра A)
YEAR_BUCKETS = [
    ("<=2010",    None, 2010, 0.15),
    ("2011-2014", 2011, 2014, 0.15),
    ("2015-2018", 2015, 2018, 0.20),
    ("2019-2022", 2019, 2022, 0.25),
    (">=2023",    2023, None, 0.25),
]

PHYSICS_PREFIXES = (
    "hep-", "astro-ph", "cond-mat", "gr-qc", "quant-ph", "nucl-",
    "physics.", "nlin.", "math-ph", "chao-dyn", "acc-phys", "atom-ph",
    "plasm-ph", "supr-con", "mtrl-th", "comp-gas", "adap-org", "patt-sol",
    "solv-int", "alg-geom", "dg-ga", "funct-an", "q-alg",
)

random.seed(RANDOM_SEED)


# --------------------------------------------------------------------------- #
# Пошук / завантаження снапшоту
# --------------------------------------------------------------------------- #
def locate_snapshot() -> str:
    """Шукає снапшот локально, інакше тягне датасет із Kaggle через kagglehub."""
    for candidate in (
        os.path.join(PROJECT_ROOT, SNAPSHOT_NAME),
        os.path.join(PROJECT_ROOT, "data", SNAPSHOT_NAME),
    ):
        if os.path.exists(candidate):
            print(f"Знайдено локальний снапшот: {candidate}")
            return candidate

    load_env()
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        sys.exit(
            "Немає Kaggle-креденшелів. Додайте у .env KAGGLE_USERNAME і KAGGLE_API_TOKEN\n"
            "(або покладіть kaggle.json у ~/.kaggle/), див. .env.example."
        )

    import kagglehub

    print(f"Завантажую датасет '{KAGGLE_DATASET}' з Kaggle (~1.2 GB) ...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Датасет у кеші: {path}")

    for root, _dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".json"):
                return os.path.join(root, name)
    sys.exit(f"У завантаженому датасеті не знайдено .json файлу (шукав у {path}).")


# --------------------------------------------------------------------------- #
# Нормалізація полів (структура -- як у скелеті завдання)
# --------------------------------------------------------------------------- #
def extract_year(paper: dict) -> int:
    """
    Беремо рік із першої версії статті -- це дата публікації на arXiv.
    update_date -- дата останнього оновлення, вона може бути на роки пізніше.
    Формат created: "Mon, 2 Apr 2007 19:18:42 GMT"
    """
    try:
        versions = paper.get("versions", [])
        if versions:
            created = versions[0]["created"]
            # Рік стоїть на 4-й позиції після split по пробілу
            return int(created.split()[3])
    except (IndexError, ValueError, KeyError):
        pass
    # Запасний варіант: update_date у форматі "YYYY-MM-DD"
    return int(paper.get("update_date", "2000-01-01")[:4])


def format_authors(paper: dict) -> str:
    """
    authors_parsed -- структурований список [["Прізвище", "Ініціали", ""]].
    Збираємо у читабельний рядок "Прізвище І., Прізвище І."
    Якщо authors_parsed відсутній -- беремо сирий рядок authors.
    """
    parsed = paper.get("authors_parsed", [])
    if parsed:
        parts = []
        for entry in parsed[:10]:  # не більше 10 авторів
            last = entry[0].strip() if len(entry) > 0 else ""
            first = entry[1].strip() if len(entry) > 1 else ""
            if last:
                parts.append(f"{last} {first}".strip())
        return ", ".join(parts)
    # Запасний варіант: сирий рядок авторів
    return paper.get("authors", "").replace("\n", " ")


def category_group(primary_category: str) -> str:
    """Зводимо ~150 arXiv-категорій до 9 груп, за якими стратифікуємо вибірку."""
    cat = primary_category
    if cat in ("cs.LG", "cs.CL", "cs.CV", "cs.AI"):
        return cat
    if cat == "cmp-lg":          # історична назва computational linguistics
        return "cs.CL"
    if cat.startswith("cs."):
        return "cs.other"
    if cat.startswith("stat."):
        return "stat"
    if cat.startswith(PHYSICS_PREFIXES):
        return "physics"
    if cat.startswith("math.") or cat == "math":
        return "math"
    return "other"


def year_bucket(year: int) -> str:
    for name, lo, hi, _weight in YEAR_BUCKETS:
        if (lo is None or year >= lo) and (hi is None or year <= hi):
            return name
    return ">=2023"


def build_cell_quotas() -> dict:
    """Квота на кожну комірку (група категорій, роковий бакет)."""
    quotas = {}
    for group, group_quota in GROUP_QUOTAS.items():
        for name, _lo, _hi, weight in YEAR_BUCKETS:
            quotas[(group, name)] = max(1, round(group_quota * weight))
    return quotas


# --------------------------------------------------------------------------- #
# Reservoir sampling
# --------------------------------------------------------------------------- #
class Reservoir:
    """Рівномірна випадкова вибірка фіксованого розміру з потоку невідомої довжини."""

    def __init__(self, size: int):
        self.size = size
        self.items = []
        self.seen = 0

    def offer(self, item) -> None:
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
        else:
            j = random.randrange(self.seen)
            if j < self.size:
                self.items[j] = item


def main() -> None:
    input_file = locate_snapshot()
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)

    cell_quotas = build_cell_quotas()
    cells = {key: Reservoir(quota) for key, quota in cell_quotas.items()}
    author_seed = Reservoir(SEED_AUTHOR_QUOTA)
    extra = Reservoir(EXTRA_POOL)

    total_lines = 0
    skipped = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Читаємо датасет", unit=" рядків"):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            abstract = paper.get("abstract", "").strip()
            title = paper.get("title", "").strip()
            paper_id = str(paper.get("id", "")).strip()

            # Пропускаємо записи без анотації, заголовка або id
            if not abstract or not title or not paper_id:
                skipped += 1
                continue

            # categories може містити кілька категорій через пробіл: "cs.LG cs.AI"
            # Беремо першу як основну
            categories = (paper.get("categories") or "").split()
            primary_category = categories[0] if categories else "unknown"

            year = extract_year(paper)
            authors = format_authors(paper)

            record = {
                "id":       paper_id,
                "title":    title.replace("\n", " ").strip(),
                "abstract": abstract.replace("\n", " ").strip(),
                "authors":  authors,
                "year":     year,
                "category": primary_category,
            }

            key = (category_group(primary_category), year_bucket(year))
            if key in cells:
                cells[key].offer(record)
            extra.offer(record)
            if SEED_AUTHOR in authors.lower():
                author_seed.offer(record)

    # --- Збираємо фінальну вибірку ----------------------------------------- #
    chosen, seen_ids = [], set()

    def add(record) -> bool:
        if record["id"] in seen_ids or len(chosen) >= MAX_RECORDS:
            return False
        seen_ids.add(record["id"])
        chosen.append(record)
        return True

    for record in author_seed.items:          # цільовий seed -- першим
        add(record)
    for key in sorted(cells):                 # страти
        for record in cells[key].items:
            add(record)
    random.shuffle(extra.items)
    for record in extra.items:                # добір, якщо комірки недобрані
        if len(chosen) >= MAX_RECORDS:
            break
        add(record)

    df = pd.DataFrame(chosen)
    words = df["abstract"].str.split().str.len()

    # --- Звіт --------------------------------------------------------------- #
    print(f"\nПрочитано рядків: {total_lines}, пропущено (без title/abstract): {skipped}")
    print(f"Авторський seed ('{SEED_AUTHOR}'): знайдено {author_seed.seen}, "
          f"узято {len(author_seed.items)}")
    underfilled = [k for k, c in cells.items() if len(c.items) < cell_quotas[k]]
    if underfilled:
        print(f"Недобраних комірок: {len(underfilled)} з {len(cells)} -- добрано з резерву")

    print(f"\nЗавантажено статей: {len(df)}")
    print("\nРозподіл за категоріями (топ-10):")
    print(df["category"].value_counts().head(10))
    print("\nРозподіл за групами категорій:")
    print(df["category"].map(category_group).value_counts())
    print("\nРозподіл за роками:")
    print(df["year"].value_counts().sort_index().to_string())
    print(f"\nСтатей до 2015 року: {(df['year'] < 2015).sum()}, "
          f"з 2021 року: {(df['year'] >= 2021).sum()}")
    print(f"Статей у категорії cs.LG: {(df['category'] == 'cs.LG').sum()}")
    print(f"\nДовжина анотації (слів): медіана {int(words.median())}, "
          f"максимум {int(words.max())}, понад 300 слів: {(words > 300).sum()}")
    print("\nПриклад запису:")
    print(df.iloc[0].to_dict())

    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nЗбережено в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
