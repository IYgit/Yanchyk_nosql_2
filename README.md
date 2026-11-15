# Завдання 2. Семантичний пошук за науковими статтями

Пайплайн семантичного пошуку по 10 000 анотацій arXiv: від сирого JSONL-снапшоту до
векторного індексу в Pinecone, пошуку з фільтрами за метаданими, двох стратегій chunking
і гібридного пошуку BM25 + вектори через Reciprocal Rank Fusion.

* Модель ембеддингів: `allenai/specter2_base` (768 вимірів, CLS-пулінг, L2-нормалізація)
* Векторна БД: Pinecone serverless (`aws / us-east-1`), метрика `cosine`
* Датасет: [arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) (Cornell University)

---

## Структура репозиторію

```
.
├── .env                      # API-ключі (не комітиться)
├── .env.example              # шаблон .env
├── .gitignore
├── requirements.txt
├── data/
│   └── arxiv_subset.parquet  # 10 000 статей (не комітиться)
├── embeddings/
│   └── embeddings.npy        # 10000 x 768 float32 (не комітиться)
├── outputs/                  # збережений вивід усіх скриптів
├── scripts/
│   ├── common.py             # спільні хелпери: модель, Pinecone, формат входу
│   ├── 01_prepare_data.py
│   ├── 02_embed.py
│   ├── 03_load_to_pinecone.py
│   ├── 04_search.py
│   ├── 05_chunking.py
│   └── 06_hybrid_search.py
└── README.md
```

## Як запустити

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1                  # Windows PowerShell

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp .env.example .env                         # і підставити власні ключі

python scripts/01_prepare_data.py            # датасет -> data/arxiv_subset.parquet
python scripts/02_embed.py                   # ембеддинги -> embeddings/embeddings.npy
python scripts/03_load_to_pinecone.py        # індекс arxiv-papers
python scripts/04_search.py                  # семантичний / з фільтрами / порівняння метрик
python scripts/05_chunking.py                # chunking + окремі індекси під чанки
python scripts/06_hybrid_search.py           # BM25 + вектори через RRF
```

`.env`:

```
PINECONE_API_KEY=...
KAGGLE_USERNAME=...
KAGGLE_API_TOKEN=...
```

`01_prepare_data.py` сам завантажує датасет через `kagglehub`, якщо снапшоту немає локально,
тому `kaggle.json` не обов'язковий — достатньо двох змінних оточення.

---

## Два свідомі відхилення від скелета із завдання

**1. Стратифікована вибірка замість перших 10 000 рядків.**
Снапшот відсортований за arXiv id, тому перші 10 000 записів — це квітень–травень 2007 року:
майже виключно physics/math і фактично нуль статей `cs.LG`. На такій підмножині фільтр із
Частини 3 («cs.LG за останні 5 років») повертає порожню видачу, а демо-запити Частини 5
(BERT, LeCun, емоції в тексті) не мають жодного релевантного документа — тобто половина
завдання стає непоказовою. Тому `01_prepare_data.py` робить один потоковий прохід по всьому
файлу і набирає 10 000 статей reservoir sampling'ом по комірках
(9 груп категорій × 5 рокових бакетів), плюс маленький цільовий seed (≤30 статей, де серед
авторів є LeCun) — щоб авторський запит Частини 5 мав на чому проявитися.
Поля, їх нормалізація та звіт у консоль — ті самі, що у скелеті.

**2. CLS-пулінг замість дефолтного mean-пулінгу.**
`SentenceTransformer("allenai/specter2_base")` працює, але в репозиторії моделі немає
конфігурації sentence-transformers, тому бібліотека мовчки додає MEAN-пулінг. Картка ж моделі
бере саме CLS-токен: *«take the first token in the batch as the embedding»*. Тому в
`scripts/common.py` модель збирається вручну: `models.Transformer(max_seq_length=512)` +
`models.Pooling(pooling_mode="cls")`. Усі скрипти кодують і документи, і запити через цю
одну функцію — інакше вектори запиту і бази лежали б у різних просторах.

---

## Частина 1 — Підготовка даних і вибір інструментів

### 1.1. Чим Pinecone відрізняється від Qdrant і Chroma

**Модель розгортання.** Pinecone існує тільки як керований SaaS: свого сервера ми не
підіймаємо, а працюємо з HTTP/gRPC API, обираючи лише хмару і регіон (у serverless-режимі
storage і compute розділені, тому індекс масштабується автоматично й оплачується за обсягом
читань/записів). Qdrant — це самостійний сервер на Rust, який запускається
одним Docker-контейнером, кластеризується в Kubernetes і за потреби береться як керований
Qdrant Cloud; ми самі обираємо залізо, параметри HNSW, тип квантизації та шардинг. Chroma
проєктувалася як «SQLite для ембеддингів»: типовий режим — вбудована бібліотека в тому ж
Python-процесі, що й застосунок, із персистенцією в локальну папку; є також режим
клієнт-сервер і керований хостинг, але архітектурно це один вузол.

**Ліцензія.** Pinecone — закритий комерційний продукт; вихідного коду немає, міграція
означає переливання векторів через API. Qdrant і Chroma — відкритий код під Apache 2.0:
їх можна форкнути, запустити on-prem у закритому контурі й не залежати від вендора.

**Продуктивність.** Pinecone дає передбачувану затримку і чесну роботу з мільярдами векторів
«з коробки», але ціною того, що тонких налаштувань майже немає: метрику й розмірність індексу не
змінити після створення, а параметри ANN-графа від нас сховані. Qdrant у публічних
ANN-бенчмарках стабільно в лідерах і головне — керований: HNSW-параметри (`m`, `ef_construct`),
скалярна/продуктова/бінарна квантизація, mmap проти in-memory, фільтрація, вбудована в обхід
графа, нативні розріджені вектори для гібридного пошуку. Chroma на корпусі в десятки й сотні
тисяч векторів відповідає миттєво, але вона одновузлова: ні шардингу, ні реплікації, і на
мільйонах векторів упирається в пам'ять одного процесу.

**Коли що брати.** Pinecone — продакшн-сервіс, де команда не хоче утримувати інфраструктуру
й готова платити за SLA та еластичність (саме цей сценарій у завданні: безкоштовний тір,
нуль налаштувань, індекс живий через хвилину). Qdrant — коли дані не можна віддавати
зовнішньому вендору (медицина, фінанси, GDPR-резидентність), коли потрібен контроль над
recall/latency і економіка на великих обсягах, або коли потрібен справжній гібридний пошук
з розрідженими векторами в одній БД. Chroma — прототип, локальний RAG, ноутбук,
інтеграційні тести: нуль інфраструктури, `pip install` і працює.

### 1.2. Чому `specter2_base`, а не універсальна `all-MiniLM-L6-v2`

Картка SPECTER2 прямо описує, під що модель навчена: вона *«trained on over 6M triplets of
scientific paper citations»* — тобто сигнал навчання це граф цитувань наукових статей, а не
загальна «схожість речень». Далі в карті перелічені формати задач, під які є адаптери, і
серед них є рівно наші два: **Proximity** — *«encoding papers as queries/candidates in link
prediction and nearest neighbor search»*, і **Adhoc Query** — *«short raw text queries for
search tasks»*. Тобто пошук статей за коротким текстовим запитом — це цільова задача моделі,
а не побічне застосування. Базова модель — `bert-base-uncased + adapters`, *«finetuned from
model: allenai/scibert»*, тобто словник і передтренування вже «наукові»: токенізатор SciBERT
не розсипає терміни на дрібні subword-уламки так, як це робить загальний
`bert-base-uncased`.

`all-MiniLM-L6-v2` у своїй карті позиціонується інакше: *«Our model is intended to be used
as a sentence and short paragraph encoder»*, навчена на *«1B sentence pairs»* з Reddit
(726M пар), S2ORC, WikiAnswers, PAQ, Stack Exchange, MS MARCO — це універсальний
загальнодоменний енкодер. Два практичні наслідки для нашої задачі. По-перше, довжина:
*«By default, input text longer than 256 word pieces is truncated»*, а анотація arXiv разом
із заголовком — це типово 200–350 слів, тобто 300–500 word pieces; MiniLM фізично відрізав
би половину абстракту, тоді як у SPECTER2 *«max input length = 512»*. По-друге,
розмірність і спеціалізація: MiniLM віддає 384 виміри загального простору, де «attention»
із Reddit і «attention» із статті про трансформери живуть поряд, тоді як SPECTER2 віддає 768
вимірів простору, натренованого так, щоб статті, які цитують одна одну, лежали близько —
що і є визначенням релевантності для наукового пошуку.

Чесне застереження: MiniLM у ~5 разів менша й швидша, і на дуже коротких загальних
запитах буває не гіршою. Але для корпусу абстрактів і запитів «за змістом статті»
доменна модель і довший контекст важливіші за швидкість.

### 1.3. Що в картці моделі написано про метрику схожості

Прямої фрази «використовуйте cosine» в картці `allenai/specter2_base` немає. 
Картка фіксує інше, не менш важливе: як саме отримати
вектор (*«take the first token in the batch as the embedding»*, тобто CLS, а не mean),
у якому форматі подати документ (`title + tokenizer.sep_token + abstract`) і яка межа
довжини (*«max input length = 512»*). Непрямий вказівник на метрику — спосіб навчання:
SPECTER/SPECTER2 вчилися triplet-лоссом на дистанції між CLS-векторами запитної статті,
процитованої (позитив) і непроцитованої (негатив), а proximity-задачі SciRepEval, на яких
модель оцінювали, — це пошук найближчих сусідів.

Чому це критично при створенні індексу: у Pinecone метрика задається один раз при
`create_index` і змінити її потім неможливо — тільки перестворити індекс і перезалити всі
вектори. Якщо метрика індексу не та, під яку навчалася модель, пошук не падає з помилкою —
він просто тихо ранжує гірше. У нашому випадку
`02_embed.py` нормалізує вектори (`normalize_embeddings=True`), а для одиничних векторів
cosine, dot product і L2 дають **однаковий порядок** (виведення — у Частині 6, п. 3), тому
вибір між ними безпечний. Узято `cosine`: вона інваріантна до норми, тому індекс не
зламається, якщо в нього колись доллють вектори, забувши нормалізувати.

### 1.4. Чому для нормалізованих ембеддингів cosine ≡ dot product

За визначенням

$$\cos(a,b)=\frac{a\cdot b}{\lVert a\rVert\,\lVert b\rVert}.$$

Якщо обидва вектори одиничні, тобто $\lVert a\rVert=\lVert b\rVert=1$, знаменник дорівнює 1 і

$$\cos(a,b)=a\cdot b .$$

Косинус — це і є скалярний добуток, поділений на норми; нормалізація просто виконує це
ділення заздалегідь, один раз на етапі кодування, а не на кожному порівнянні. Практична
вигода: dot product — це одна інструкція BLAS без ділення і без обчислення кореня, тому
ANN-індекси на нормалізованих векторах працюють швидше, а семантика скору зберігається
(діапазон $[-1,1]$). Емпірична перевірка в `04_search.py` показує максимальне відхилення
`|cosine - dot|` по всьому корпусу на рівні машинної точності float32.

### Вивід кроків 1–2

**Крок 1.** Прохід по всьому снапшоту: 3 233 620 рядків за 40 с, 0 записів відкинуто через
відсутній заголовок чи анотацію. З 225 знайдених статей за участі LeCun узято 30 (seed).
Підсумкова вибірка — 10 000 статей: `cs.LG` 1613, `cs.CV` 1213, `cs.CL` 1101, `cs.AI` 602,
physics 1600, math 1400, stat 501, решта — інші категорії; роки 1992–2026, з них
**2995 статей до 2015 року** (для фільтра B) і **3880 з 2021 року** (для фільтра A).
Довжина анотації: медіана 151 слово, максимум 328 слів, 26 анотацій довші за 300 слів.

<details>
<summary>Вивід 01_prepare_data.py</summary>

```text
Знайдено локальний снапшот: C:\Work\GoIT\NoSQL and Verctor Databases\task_2\arxiv-metadata-oai-snapshot.json
Читаємо датасет: 0 рядків [00:00, ? рядків/s]
Читаємо датасет: 3233620 рядків [00:40, 79210.53 рядків/s]
Прочитано рядків: 3233620, пропущено (без title/abstract): 0
Авторський seed ('lecun'): знайдено 225, узято 30
Завантажено статей: 10000
Розподіл за категоріями (топ-10):
category
cs.LG       1613
cs.CV       1213
cs.CL       1101
cs.AI        602
stat.ME      210
eess.SY      167
quant-ph     166
math.CO      139
stat.ML      135
math.AP      135
Name: count, dtype: int64
Розподіл за групами категорій:
category
cs.LG       1613
physics     1600
math        1400
cs.CV       1213
cs.CL       1200
other        971
cs.other     900
cs.AI        602
stat         501
Name: count, dtype: int64
Розподіл за роками:
year
1992      2
1993      5
1994     34
1995     33
1996     31
1997     32
1998     33
1999     31
2000     47
2001     36
2002     63
2003     47
2004     69
2005     87
2006     88
2007    128
2008    181
2009    207
2010    343
2011    226
2012    345
2013    428
2014    499
2015    348
2016    397
2017    519
2018    734
2019    506
2020    621
2021    654
2022    719
2023    461
2024    580
2025    705
2026    761
Статей до 2015 року: 2995, з 2021 року: 3880
Статей у категорії cs.LG: 1613
Довжина анотації (слів): медіана 151, максимум 328, понад 300 слів: 26
Приклад запису:
{'id': '2209.14905', 'title': 'Variance Covariance Regularization Enforces Pairwise Independence in   Self-Supervised Representations', 'abstract': "Self-Supervised Learning (SSL) methods such as VICReg, Barlow Twins or W-MSE avoid collapse of their joint embedding architectures by constraining or regularizing the covariance matrix of their projector's output. This study highlights important properties of such strategy, which we coin Variance-Covariance regularization (VCReg). More precisely, we show that {\\em VCReg combined to a MLP projector enforces pairwise independence between the features of the learned representation}. This result emerges by bridging VCReg applied on the projector's output to kernel independence criteria applied on the projector's input. We empirically validate our findings where (i) we put in evidence which projector's characteristics favor pairwise independence, (ii) we demonstrate pairwise independence to be beneficial for out-of-domain generalization, (iii) we demonstrate that the scope of VCReg goes beyond SSL by using it to solve Independent Component Analysis. This provides the first theoretical motivation and explanation of MLP projectors in SSL.", 'authors': 'Mialon Grégoire, Balestriero Randall, LeCun Yann', 'year': 2022, 'category': 'cs.LG'}
Збережено в C:\Work\GoIT\NoSQL and Verctor Databases\task_2\data\arxiv_subset.parquet
```

</details>

**Крок 2.** 10 000 текстів `title [SEP] abstract`, батчі по 64, CPU (14 потоків torch),
34 хв 52 с. Розмірність 768, норми всіх векторів — рівно 1.000000, файл 29.3 MB float32.

<details>
<summary>Вивід 02_embed.py</summary>

```text
==============================================================================
Крок 2. Ембеддинги SPECTER2
==============================================================================
Прочитано записів із parquet: 10000
Приклад тексту для кодування:
  Variance Covariance Regularization Enforces Pairwise Independence in   Self-Supervised Representations [SEP] Self-Supervised Learning (SSL) methods such as VICReg, Barlow Twins or W-MSE avoid collapse of their joint embe ...
Завантажую модель allenai/specter2_base (pooling=cls) ...
Loading weights: 100%|██████████| 199/199 [00:00<00:00, 44159.91it/s]
Максимальна довжина входу: 512 токенів
Пристрій: cpu
Batches: 100%|██████████| 157/157 [34:52<00:00, 13.33s/it]
Оброблено текстів:      10000
Розмірність ембеддингів: 768
Норма першого вектора:   1.000000
Норми (min/max):         1.000000 / 1.000000
Збережено в C:\Work\GoIT\NoSQL and Verctor Databases\task_2\embeddings\embeddings.npy (29.3 MB, dtype=float32)
```

</details>

---

## Частина 2 — Завантаження даних і метадані

Індекс `arxiv-papers` створено як serverless (`aws / us-east-1`), `dimension=768`,
`metric=cosine`; скрипт чекає `status.ready`, тому наступний крок не стартує на ще
не піднятому індексі. Завантаження — 50 батчів по 200 векторів (ліміт, який рекомендує
Pinecone), 1 хв 42 с на 10 000 векторів. Після заливки `describe_index_stats()` показує
`total_vector_count = 10000`.

Метадані кожного вектора: `arxiv_id`, `title`, `abstract` (перші 500 символів),
`authors` (перші 200 символів), `year` (int — щоб працювали числові оператори `$gte`/`$lt`)
і `category` (рядок — для `$eq`). Обрізання не втрата даних: повний текст анотації лишається
в `data/arxiv_subset.parquet`, і `04_search.py` підтягує його за `arxiv_id` уже після пошуку.
Причина обрізання — ліміт Pinecone у 40 KB метаданих на вектор; при 10 000 векторів повні
абстракти ще влізли б, але на мільйонах така схема просто дорога: метадані зберігаються
разом з індексом і збільшують його обсяг, тоді як parquet-файл коштує як звичайний файл.

Ідемпотентність: id формуються як `paper_<номер рядка в parquet>`, тому повторний запуск
скрипта не дублює вектори, а перезаписує ті самі ключі (`upsert`, не `insert`). Якщо індекс
уже існує, скрипт про це повідомляє і не намагається створити його вдруге.

<details>
<summary>Вивід 03_load_to_pinecone.py</summary>

```text
==============================================================================
Крок 3. Завантаження в Pinecone
==============================================================================
Записів у parquet: 10000, ембеддингів: (10000, 768)
  Створюю індекс 'arxiv-papers' (dim=768, metric=cosine) ...
  Індекс 'arxiv-papers' готовий.
Upsert у Pinecone: 100%|██████████| 50/50 [01:42<00:00,  2.05s/ батч]
Відправлено векторів: 10000
Усього векторів в індексі 'arxiv-papers': 10000
Розмірність індексу: 768
Заповненість (index_fullness): 0.0
Приклад збережених метаданих (paper_0):
  abstract: Self-Supervised Learning (SSL) methods such as VICReg, Barlow Twins or W-MSE avoid collapse of their joint embedding arc...
  arxiv_id: 2209.14905
  authors: Mialon Grégoire, Balestriero Randall, LeCun Yann
  category: cs.LG
  title: Variance Covariance Regularization Enforces Pairwise Independence in   Self-Supervised Representations
  year: 2022
```

</details>

---

## Частина 3 — Пошукові запити

### Що показали три типи пошуку

**A. Чистий семантичний пошук.** Запит `"teaching machines to recognize objects in pictures"`
навмисно не містить жодного терміна з предметної області: слова `pictures` немає в корпусі
в цьому значенні взагалі, статті кажуть `image` і `visual`. Топ-5: *Interpretable Image
Recognition with Hierarchical Prototypes* (cs.CV, 2019, 0.9247), *A Novel Template-Based
Learning Model* (cs.LG, 2011), *Multiple Object Recognition with Visual Attention*
(cs.LG, 2014), *Building Machines That Learn and Think Like People* (cs.AI, 2016),
*A Roadmap for the Development of the "SP Machine"* (cs.AI, 2017). Тобто модель зв'язала
«pictures» з «image/visual», а «teaching machines» — з «learning model». Повнотекстовий
пошук за цим запитом знайшов би хіба що статті зі словом `objects`.

**B. Пошук із фільтрами.** Запит однаковий —
`"reinforcement learning agent learns a policy from rewards"`, різні фільтри:

| | Фільтр | Топ-5: роки | Топ-5: категорії | Середня схожість |
|---|---|---|---|---|
| A | `category = cs.LG` і `year >= 2022` | 2022–2024 | cs.LG | 0.9417 |
| B | `year < 2015` | 2002–2014 | cs.AI, cs.LG | 0.9463 |

Перетин видач — нульовий **за побудовою**: умови за роком взаємно виключні. Фільтр у
Pinecone — це pre-filter: він відрізає простір кандидатів ще до ANN-пошуку, а не
переранжовує готову видачу. Тому кожен список — найкраще **в межах свого підпростору**,
і скори з двох різних фільтрів не є мірою «кращої відповіді для користувача».
Показово, що середня схожість старих статей (B) навіть трохи вища: запит сформульований
класичною RL-мовою («agent», «policy», «rewards»), і роботи 2002–2014 описують себе тими
самими простими словами, тоді як сучасний cs.LG-зріз термінологічно вужчий (MADDPG,
successor features, BAMDP shaping, monitored MDP).

**C. Порівняння метрик** — запит `"neural network for natural language understanding"`,
усі 10 000 локальних ембеддингів:

| Метрика | Топ-1 | Скор топ-1 | Порядок топ-5 |
|---|---|---|---|
| cosine | *Natural Language Understanding with Distributed Representation* | +0.9553 | — |
| dot product | те саме | +0.9553 | ідентичний cosine |
| L2 distance | те саме | 0.2990 | ідентичний cosine |

### Обов'язкові відповіді

**Чи збігаються топ-5 для cosine і dot product і чому?**
Збігаються повністю — і склад, і порядок (`однаковий порядок = True`, `спільних 5/5`), а
максимальне розходження самих значень по всьому корпусу становить `2.98e-07`, тобто рівень
машинної точності float32. Причина проста: `02_embed.py` кодує з `normalize_embeddings=True`,
тому норми всіх 10 000 векторів і вектора запиту дорівнюють 1.0 (це видно у виводі:
`норми в [1.0000, 1.0000]`). А при $\lVert a\rVert=\lVert b\rVert=1$ косинус за визначенням
дорівнює скалярному добутку: $\cos(a,b)=\frac{a\cdot b}{1\cdot 1}=a\cdot b$. Залишкові
`1e-07` — це не різниця метрик, а накопичення помилки округлення при діленні на норму,
яка сама обчислена у float32.

**Чи відрізняються результати для L2 і чому?**
Склад і порядок топ-5 — той самий (`однаковий порядок = True`), відрізняються тільки
значення і їхній напрямок: для топ-1 cosine дає `+0.9553` («більше — краще»), а L2 дає
`0.2990` («менше — краще»). Це не збіг: для одиничних векторів
$\lVert a-b\rVert_2^{2}=2-2\cos(a,b)$, що скрипт перевіряє чисельно —
`max|L2² − (2 − 2·cos)| = 1.13e-06`. Корінь із $2-2\cos$ — монотонно спадна функція
косинуса, а монотонне перетворення не змінює порядку сортування. Отже на нормалізованих
векторах L2 ранжує тотожно cosine; різниця лише в шкалі скорів (L2 ∈ [0, 2] проти
cosine ∈ [−1, 1]) і в тому, що пороги та сортування треба розвертати.

**Що сталося б, якби ембеддинги не були нормалізовані?**
Рівність зникає, і скрипт це демонструє експериментально: ті самі вектори помножено на
випадковий масштаб із діапазону 0.5–3.0 (що імітує модель без нормалізації), після чого
топ-5 за dot product змінився **повністю** — жодного спільного документа з початковою
видачею (`dot зберіг вихідний топ-5: False`), тоді як cosine дав точно той самий список
(`cosine зберіг: True`). Математично: $a\cdot b=\lVert a\rVert\,\lVert b\rVert\cos(a,b)$,
тому без нормалізації dot product ранжує за добутком «схожість × довжина вектора» і
систематично піднімає догори просто довгі вектори. У текстових моделях довжина вектора
корелює з довжиною й лексичною насиченістю тексту, тобто видача зсувалася б у бік довгих
абстрактів незалежно від змісту запиту. L2 постраждав би так само:
$\lVert a-b\rVert^2=\lVert a\rVert^2+\lVert b\rVert^2-2a\cdot b$ — доданок
$\lVert a\rVert^2$ карає документи за власну норму. Інваріантним до масштабу залишається
лише косинус, і саме тому нормалізація на етапі кодування робить вибір метрики в Pinecone
безпечним.

<details>
<summary>Вивід 04_search.py</summary>

```text
Підключено до індексу 'arxiv-papers': 10000 векторів
Завантажую модель allenai/specter2_base (pooling=cls) ...
Loading weights: 100%|██████████| 199/199 [00:00<00:00, 49729.89it/s]
==============================================================================
A. Чистий семантичний пошук (без фільтрів)
==============================================================================
Запит: "teaching machines to recognize objects in pictures"
Жодне слово запиту не є термінологічним: 'pictures' замість 'images',
'teaching machines' замість 'training models'. Повнотекстовий пошук тут безсилий.
  1. [0.9247] Interpretable Image Recognition with Hierarchical Prototypes
     cs.CV | 2019 | Hase Peter, Chen Chaofan, Li Oscar, Rudin Cynthia
     Vision models are interpretable when they classify objects on the basis of features that a person can directly understand. Recently, methods relying on visual feature prototypes have been developed fo...
  2. [0.9168] A Novel Template-Based Learning Model
     cs.LG | 2011 | Abolghasemi-Dahaghani Mohammadreza, Didehvar Farzad, Nowroozi Alireza
     This article presents a model which is capable of learning and abstracting new concepts based on comparing observations and finding the resemblance between the observations. In the model, the new obse...
  3. [0.9157] Multiple Object Recognition with Visual Attention
     cs.LG | 2014 | Ba Jimmy, Mnih Volodymyr, Kavukcuoglu Koray
     We present an attention-based model for recognizing multiple objects in images. The proposed model is a deep recurrent neural network trained with reinforcement learning to attend to the most relevant...
  4. [0.9120] Building Machines That Learn and Think Like People
     cs.AI | 2016 | Lake Brenden M., Ullman Tomer D., Tenenbaum Joshua B., Gershman Samuel
     Recent progress in artificial intelligence (AI) has renewed interest in building systems that learn and think like people. Many advances have come from using deep neural networks trained end-to-end in...
  5. [0.9111] A Roadmap for the Development of the "SP Machine" for Artificial   Intelligence
     cs.AI | 2017 | Wolff J Gerard
     This paper describes a roadmap for the development of the "SP Machine", based on the "SP Theory of Intelligence" and its realisation in the "SP Computer Model". The SP Machine will be developed initia...
==============================================================================
B. Пошук із фільтрами за метаданими
==============================================================================
Запит для обох прикладів: "reinforcement learning agent learns a policy from rewards"
Приклад A -- filter={'category': {'$eq': 'cs.LG'}, 'year': {'$gte': 2022}}  (cs.LG, 2022-2026)
  1. [0.9455] Optimistic Linear Support and Successor Features as a Basis for Optimal   Policy Transfer
     cs.LG | 2022 | Alegre Lucas N., Bazzan Ana L. C., da Silva Bruno C.
     In many real-world applications, reinforcement learning (RL) agents might have to solve multiple tasks, each one typically modeled via a reward function. If reward functions are expressed linearly, an...
  2. [0.9442] Revisiting the Gumbel-Softmax in MADDPG
     cs.LG | 2023 | Tilbury Callum Rhys, Christianos Filippos, Albrecht Stefano V.
     MADDPG is an algorithm in multi-agent reinforcement learning (MARL) that extends the popular single-agent method, DDPG, to multi-agent scenarios. Importantly, DDPG is an algorithm designed for continu...
  3. [0.9406] Monitored Markov Decision Processes
     cs.LG | 2024 | Parisi Simone, Mohammedalamen Montaser, Kazemipour Alireza, Taylor Mat
     In reinforcement learning (RL), an agent learns to perform a task by interacting with an environment and receiving feedback (a numerical reward) for its actions. However, the assumption that rewards a...
  4. [0.9396] Distributional Successor Features Enable Zero-Shot Policy Optimization
     cs.LG | 2024 | Zhu Chuning, Wang Xinqi, Han Tyler, Du Simon S., Gupta Abhishek
     Intelligent agents must be generalists, capable of quickly adapting to various tasks. In reinforcement learning (RL), model-based RL learns a dynamics model of the world, in principle enabling transfe...
  5. [0.9384] BAMDP Shaping: a Unified Framework for Intrinsic Motivation and Reward   Shaping
     cs.LG | 2024 | Lidayan Aly, Dennis Michael, Russell Stuart
     Intrinsic motivation and reward shaping guide reinforcement learning (RL) agents by adding pseudo-rewards, which can lead to useful emergent behaviors. However, they can also encourage counterproducti...
Приклад B -- filter={'year': {'$lt': 2015}}  (будь-яка категорія, до 2015 року)
  1. [0.9523] Selecting the State-Representation in Reinforcement Learning
     cs.LG | 2013 | Maillard Odalric-Ambrym, Munos Rémi, Ryabko Daniil
     The problem of selecting the right state-representation in a reinforcement learning problem is considered. Several models (functions mapping past observations to a finite set) of the observations are...
  2. [0.9498] A Multiagent Reinforcement Learning Algorithm with Non-linear Dynamics
     cs.LG | 2014 | Abdallah Sherief, Lesser Victor
     Several multiagent reinforcement learning (MARL) algorithms have been proposed to optimize agents decisions. Due to the complexity of the problem, the majority of the previously developed MARL algorit...
  3. [0.9464] Reinforcement Learning via AIXI Approximation
     cs.LG | 2010 | Veness Joel, Ng Kee Siong, Hutter Marcus, Silver David
     This paper introduces a principled approach for the design of a scalable general reinforcement learning agent. This approach is based on a direct approximation of AIXI, a Bayesian optimality notion fo...
  4. [0.9431] Learning from Scarce Experience
     cs.AI | 2002 | Peshkin Leonid, Shelton Christian R.
     Searching the space of policies directly for the optimal policy has been one popular method for solving partially observable reinforcement learning problems. Typically, with each change of the target...
  5. [0.9397] Decision Making Agent Searching for Markov Models in Near-Deterministic   World
     cs.AI | 2011 | Matuz Gabor, Lorincz Andras
     Reinforcement learning has solid foundations, but becomes inefficient in partially observed (non-Markovian) environments. Thus, a learning agent -born with a representation and a policy- might wish to...
Порівняння A і B:
  спільних документів: 0
  середня схожість A: 0.9417 | B: 0.9463
  категорії A: ['cs.LG']
  категорії B: ['cs.AI', 'cs.LG']
  роки A: [2022, 2023, 2024]
  роки B: [2002, 2010, 2011, 2013, 2014]
  Фільтр -- це жорстке обмеження простору пошуку (pre-filter), а не переранжування:
  кожна видача -- найкраще В МЕЖАХ СВОГО підпростору, тому скори з різних фільтрів
  не порівнюються між собою як 'релевантність для користувача'. Перетин A і B нульовий
  за побудовою: умови за роком взаємно виключні.
  Показово, що середня схожість B не нижча за A, хоча B -- це статті 2002-2014 років:
  запит сформульований класичною RL-мовою ('agent', 'policy', 'rewards'), і старі
  роботи описують себе тими самими простими словами, тоді як сучасний cs.LG-зріз
  вужчий і термінологічніший (MARL, successor features, BAMDP shaping).
  Висновок: фільтр змінює СКЛАД кандидатів, а не те, як модель міряє схожість.
==============================================================================
C. Порівняння метрик схожості на локальних ембеддингах
==============================================================================
Запит: "neural network for natural language understanding"
Ембеддинги: (10000, 768), норми в [1.0000, 1.0000], норма запиту 1.0000
Cosine similarity (більше -- краще):
  1. [+0.9553] paper_953 | cs.CL | 2015 | Natural Language Understanding with Distributed Representation
  2. [+0.9348] paper_637 | cs.CL | 2014 | Learning Phrase Representations using RNN Encoder-Decoder for   Statis
  3. [+0.9326] paper_3298 | cs.LG | 2017 | Survey of reasoning using Neural networks
  4. [+0.9324] paper_1062 | cs.CL | 2019 | Neural Text Generation from Rich Semantic Representations
  5. [+0.9321] paper_814 | cs.CL | 2017 | Effective Spoken Language Labeling with Deep Recurrent Neural Networks
Dot product (більше -- краще):
  1. [+0.9553] paper_953 | cs.CL | 2015 | Natural Language Understanding with Distributed Representation
  2. [+0.9348] paper_637 | cs.CL | 2014 | Learning Phrase Representations using RNN Encoder-Decoder for   Statis
  3. [+0.9326] paper_3298 | cs.LG | 2017 | Survey of reasoning using Neural networks
  4. [+0.9324] paper_1062 | cs.CL | 2019 | Neural Text Generation from Rich Semantic Representations
  5. [+0.9321] paper_814 | cs.CL | 2017 | Effective Spoken Language Labeling with Deep Recurrent Neural Networks
L2 distance (менше -- краще):
  1. [+0.2990] paper_953 | cs.CL | 2015 | Natural Language Understanding with Distributed Representation
  2. [+0.3610] paper_637 | cs.CL | 2014 | Learning Phrase Representations using RNN Encoder-Decoder for   Statis
  3. [+0.3672] paper_3298 | cs.LG | 2017 | Survey of reasoning using Neural networks
  4. [+0.3677] paper_1062 | cs.CL | 2019 | Neural Text Generation from Rich Semantic Representations
  5. [+0.3685] paper_814 | cs.CL | 2017 | Effective Spoken Language Labeling with Deep Recurrent Neural Networks
Порівняння списків:
  cosine vs dot: однаковий порядок = True, спільних 5/5
  cosine vs L2:  однаковий порядок = True, спільних 5/5
  максимальне відхилення |cosine - dot| по всьому корпусу: 2.98e-07
  перевірка тотожності L2^2 = 2 - 2*cos: max|різниця| = 1.13e-06
Контрприклад -- ті самі вектори, помножені на випадковий масштаб 0.5-3.0:
  dot    top-5: ['paper_1144', 'paper_4309', 'paper_1812', 'paper_1454', 'paper_3424']
  cosine top-5: ['paper_953', 'paper_637', 'paper_3298', 'paper_1062', 'paper_814']
  dot зберіг вихідний топ-5: False; cosine зберіг: True
  Висновок: рівність cosine і dot тримається рівно доти, доки всі вектори одиничні.
```

</details>

---

## Частина 4 — Chunking

Узято 30 статей із найдовшими анотаціями — 299–328 слів (медіана по корпусу — 151 слово).
Дві стратегії з однаковим цільовим розміром 60 слів:

| | Чанків | Слів у чанку (сер./min/max) | Починаються посеред речення | Обриваються посеред речення |
|---|---|---|---|---|
| `fixed` (60 слів, overlap 15) | 210 | 57.0 / 29 / 60 | **146 (70%)** | **170 (81%)** |
| `semantic` (цілі речення, ≤60 слів) | 203 | 45.7 / 10 / 143 | **0 (0%)** | **0 (0%)** |

Чанки залито в два окремі індекси — `arxiv-chunks-fixed` і `arxiv-chunks-semantic`
(безкоштовний тір дозволив створити їх разом з основним; у скрипті передбачено фолбек на
namespaces, якщо ліміт індексів буде вичерпано). Кожен чанк кодується як
`title [SEP] chunk`: заголовок дає чанку контекст статті, з якої його вирізано.

### Обов'язкові відповіді

**Яка стратегія дає більш осмислені чанки?**
Semantic — і це видно не лише в статистиці, а прямо у видачі. На запит
`"how the proposed method was evaluated on benchmark datasets"` обидва індекси підняли
ті самі дві статті, але fixed віддає фрагменти, що починаються з середини фрази:
*«large volume of test subjects. As part of the work, synthetic image datasets…»*,
*«we found out that the pre-processing of categorizing the news is still done by…»*; а
semantic на тих самих місцях віддає цілі думки: *«As part of the work, synthetic image
datasets of five different types of seed images namely, canola, rough rice, sorghum, soy
and wheat are applied to three different self-supervised…»*. Для користувача, який читає
сніпет, і для LLM, якій цей чанк піде в контекст, друге придатне, а перше — ні.
Ціна semantic-стратегії — нерівномірність: 45.7 слова в середньому замість 57, тобто
частина чанків недовантажена, а один чанк вийшов на 143 слова, бо це одне довге речення,
яке стратегія принципово не ріже (ліміт у 60 слів перевищується, якщо речення саме довше).
Fixed натомість дає рівну довжину й передбачувану кількість векторів, але купує це
розрізаними реченнями.

**Чи є випадки розрізаних речень і як це впливає на ембеддинги?**
Так, і це не окремі випадки, а норма для fixed-стратегії: **70%** чанків починаються з
маленької літери (тобто з середини речення) і **81%** обриваються без кінцевої точки.
Для semantic — 0% і 0%. Вплив на ембеддинги подвійний. По-перше, трансформер отримує
синтаксично зламаний вхід: `[CLS]`-вектор формується з токенів, серед яких немає початку
предикативної конструкції, тому модель кодує «обривок», а не твердження — і такий вектор
систематично зсунутий від вектора повного речення з тим самим змістом. По-друге, рветься
кореференція: якщо речення *«It outperforms the baseline by 4 points»* потрапило в чанк без
попереднього, то `it` не має референта ні для моделі, ні для користувача, і чанк стає
«релевантним до всього» — класичне джерело хибнопозитивних спрацювань. Практичний компроміс,
який це лікує, — саме перекриття: навіть розрізане речення з високою ймовірністю повністю
входить у сусідній чанк.

**Як розмір overlap впливає на кількість чанків і покриття тексту?**
Залежність нелінійна й швидко стає дорогою — скрипт міряє її на тих самих 30 статтях
(9275 слів тексту):

| Overlap | Чанків | Закодовано слів | Роздування |
|---|---|---|---|
| 0 | 176 | 9 275 | 1.00× |
| 15 | 210 (+19%) | 11 975 | 1.29× |
| 30 | 296 (+68%) | 17 255 | 1.86× |
| 45 | 543 (+208%) | 32 360 | 3.49× |

Кількість чанків росте як $\lceil N/(S-O)\rceil$, тобто при наближенні overlap до розміру
чанка знаменник прямує до нуля, і крива вистрілює: на overlap 45 з 60 (75%) той самий текст
кодується 3.5 раза. Платиш за це тричі — часом ембеддингу, обсягом індексу і грошима за
запити, — а віддача спадна: покриття тексту вже при overlap 0 дорівнює 100% (жодне слово не
втрачено), overlap лікує не покриття, а **межі**: ймовірність того, що речення або думка
опиниться розрізаною в *усіх* чанках, де вона зустрічається. Тому розумний діапазон —
10–20% від розміру чанка (у нас 15 з 60 = 25%, що вже трохи забагато, але дає запас на
довгі речення), а якщо межі не ріжуться принципово (semantic-стратегія), перекриття можна
взагалі не використовувати.

<details>
<summary>Вивід 05_chunking.py</summary>

```text
==============================================================================
Крок 5. Chunking: fixed-size vs semantic
==============================================================================
Узято 30 статей із найдовшими анотаціями: 299-328 слів (медіана по корпусу 151)
FIXED (size=60, overlap=15): чанків 210, слів у чанку: сер. 57.0 (min 29, max 60), починаються посеред речення 146 (70%), обриваються посеред речення 170 (81%)
SEMANTIC (max_words=60):        чанків 203, слів у чанку: сер. 45.7 (min 10, max 143), починаються посеред речення 0 (0%), обриваються посеред речення 0 (0%)
Вплив overlap на кількість чанків (fixed-size, ті самі 30 статей):
  overlap= 0 слів -> чанків 176, закодовано слів 9275 (1.00x від обсягу тексту)
  overlap=15 слів -> чанків 210, закодовано слів 11975 (1.29x від обсягу тексту)
  overlap=30 слів -> чанків 296, закодовано слів 17255 (1.86x від обсягу тексту)
  overlap=45 слів -> чанків 543, закодовано слів 32360 (3.49x від обсягу тексту)
Приклад: перші два чанки однієї статті кожною стратегією
  [fixed] The ANTS problem
    чанк 0: "We introduce the Ants Nearby Treasure Search (ANTS) problem, which models natural cooperative foraging behavior such as that performed by ants around ..."
      60 слів | обривається посеред речення
    чанк 1: "The treasure is placed at a target location by an adversary and the agents' goal is to find it as fast as possible as a function of both k and D, wher..."
      60 слів | обривається посеред речення
  [semantic] The ANTS problem
    чанк 0: "We introduce the Ants Nearby Treasure Search (ANTS) problem, which models natural cooperative foraging behavior such as that performed by ants around ..."
      45 слів | межі речень збережені
    чанк 1: "The treasure is placed at a target location by an adversary and the agents' goal is to find it as fast as possible as a function of both k and D, wher..."
      56 слів | межі речень збережені
==============================================================================
Створення індексів і завантаження чанків
==============================================================================
Завантажую модель allenai/specter2_base (pooling=cls) ...
Loading weights: 100%|██████████| 199/199 [00:00<00:00, 44354.69it/s]
  Створюю індекс 'arxiv-chunks-fixed' (dim=768, metric=cosine) ...
  Індекс 'arxiv-chunks-fixed' готовий.
  Створюю індекс 'arxiv-chunks-semantic' (dim=768, metric=cosine) ...
  Індекс 'arxiv-chunks-semantic' готовий.
Batches: 100%|██████████| 4/4 [00:13<00:00,  3.42s/it]
Upsert -> arxiv-chunks-fixed: 100%|██████████| 3/3 [00:03<00:00,  1.21s/ батч]
  Завантажено 210 чанків у arxiv-chunks-fixed
Batches: 100%|██████████| 4/4 [00:18<00:00,  4.53s/it]
Upsert -> arxiv-chunks-semantic: 100%|██████████| 3/3 [00:03<00:00,  1.10s/ батч]
  Завантажено 203 чанків у arxiv-chunks-semantic
==============================================================================
Пошук по чанках: порівняння стратегій
==============================================================================
Запит: "how the proposed method was evaluated on benchmark datasets"
  --- arxiv-chunks-fixed ---
  1. [0.9200] Classification of Seeds using Domain Randomization on Self-Supervised
     чанк 5/7 | cs.CV | 2021
     "large volume of test subjects. As part of the work, synthetic image datasets of five different types of seed images namely, canola, rough rice, sorghum, soy and wheat are applied t..."
  2. [0.9141] Approach to Predicting News -- A Precise Multi-LSTM Network With BERT
     чанк 3/7 | cs.CL | 2022
     "we found out that the pre-processing of categorizing the news is still done by human labor. However, human labor may cause mistakes and cannot work for a long time. The growing dem..."
  3. [0.9117] Classification of Seeds using Domain Randomization on Self-Supervised
     чанк 2/7 | cs.CV | 2021
     "attributes of a seed such as size, shape, area, color and texture to identify the seed type, a process that is tedious and labor-intensive. Advances in the areas of computer vision..."
  4. [0.9047] Approach to Predicting News -- A Precise Multi-LSTM Network With BERT
     чанк 4/7 | cs.CL | 2022
     "do as good as humans or even better, using machines can reduce humans' burden and cut down costs. Therefore, in this work, we build a predictive model to classify the category of n..."
  5. [0.9042] Classification of Seeds using Domain Randomization on Self-Supervised
     чанк 1/7 | cs.CV | 2021
     "The first step toward Seed Phenotyping i.e. the comprehensive assessment of complex seed traits such as growth, development, tolerance, resistance, ecology, yield, and the measurem..."
  --- arxiv-chunks-semantic ---
  1. [0.9192] Classification of Seeds using Domain Randomization on Self-Supervised
     чанк 5/6 | cs.CV | 2021
     "As part of the work, synthetic image datasets of five different types of seed images namely, canola, rough rice, sorghum, soy and wheat are applied to three different self-supervis..."
  2. [0.9104] Approach to Predicting News -- A Precise Multi-LSTM Network With BERT
     чанк 3/6 | cs.CL | 2022
     "However, human labor may cause mistakes and cannot work for a long time. The growing demands for automatic machines in the near decades show that while the machine can do as good a..."
  3. [0.9090] Classification of Seeds using Domain Randomization on Self-Supervised
     чанк 2/6 | cs.CV | 2021
     "Generally, a plant re-searcher inspects the visual attributes of a seed such as size, shape, area, color and texture to identify the seed type, a process that is tedious and labor-..."
  4. [0.9085] Approach to Predicting News -- A Precise Multi-LSTM Network With BERT
     чанк 2/6 | cs.CL | 2022
     "It also shows that the "made-up news" has caused a great deal of confusion in Taiwanese society and has serious impacts on global stability. Although there are several applications..."
  5. [0.9045] Approach to Predicting News -- A Precise Multi-LSTM Network With BERT
     чанк 6/6 | cs.CL | 2022
     "Each document embedding is an input for the final predicting model, which contains two Dense layers and one Activation layer. And each document embedding is transformed into 1 vect..."
Запит: "theoretical guarantees and convergence proofs"
  --- arxiv-chunks-fixed ---
  1. [0.8977] From Hopf algebras to rough paths and regularity structures
     чанк 7/7 | math.RA | 2021
     "written as a master's thesis supervised by Peter Friz and Sylvie Paycha and submitted to TU Berlin on July 29 2016. As of 2021, it does not provide novel research material, but can..."
  2. [0.8971] Combinatorial geometry of neural codes, neural data analysis, and neur
     чанк 7/7 | q-bio.NC | 2022
     "focus on combinatorial threshold linear networks (CTLNs), which are TLNs defined from a directed graph. We prove that if the graph of a CTLN is a directed acyclic graph, then all t..."
  3. [0.8957] The ANTS problem
     чанк 4/7 | cs.DC | 2017
     "analysis aims at establishing connections between the time complexity and the initial knowledge held by agents (e.g., regarding their total number k), as they commence the search...."
  4. [0.8942] Combinatorial geometry of neural codes, neural data analysis, and neur
     чанк 5/7 | q-bio.NC | 2022
     "nodes and the Radon rank. We apply these to analyze calcium imaging data from a larval zebrafish. In Chapter 7, we explore the underlying rank in more detail, establish connections..."
  5. [0.8904] Combinatorial geometry of neural codes, neural data analysis, and neur
     чанк 1/7 | q-bio.NC | 2022
     "This dissertation explores applications of discrete geometry in mathematical neuroscience. We begin with convex neural codes, which model the activity of hippocampal place cells an..."
  --- arxiv-chunks-semantic ---
  1. [0.9023] Combinatorial geometry of neural codes, neural data analysis, and neur
     чанк 7/7 | q-bio.NC | 2022
     "We prove that if the graph of a CTLN is a directed acyclic graph, then all trajectories of the CTLN approach a fixed point...."
  2. [0.9008] Combinatorial geometry of neural codes, neural data analysis, and neur
     чанк 5/7 | q-bio.NC | 2022
     "In Chapter 7, we explore the underlying rank in more detail, establish connections to oriented matroid theory, and show that computing underlying rank is also NP-hard. Finally, we..."
  3. [0.8985] From Hopf algebras to rough paths and regularity structures
     чанк 7/7 | math.RA | 2021
     "This work was written as a master's thesis supervised by Peter Friz and Sylvie Paycha and submitted to TU Berlin on July 29 2016. As of 2021, it does not provide novel research mat..."
  4. [0.8913] A Probabilistic Characterization of the Dominance Order on Partitions
     чанк 3/7 | math.CO | 2015
     "Given non-negative integers $j$ and $t$, let $P(\nu,j,t)$ be the probability that the total number of balls in $F$ is $j$ and that no row of $F$ contains more that $t$ balls. We sh..."
  5. [0.8906] Combinatorial geometry of neural codes, neural data analysis, and neur
     чанк 1/7 | q-bio.NC | 2022
     "This dissertation explores applications of discrete geometry in mathematical neuroscience. We begin with convex neural codes, which model the activity of hippocampal place cells an..."
Запит: "limitations and future work"
  --- arxiv-chunks-fixed ---
  1. [0.8964] Classification of Seeds using Domain Randomization on Self-Supervised
     чанк 7/7 | cs.CV | 2021
     "frameworks in question, achieves an accuracy of 77% on the test dataset which is only ~13% less than the accuracy of 90% achieved by ResNet-50 trained on 100% of the labels...."
  2. [0.8963] Classification of Seeds using Domain Randomization on Self-Supervised
     чанк 6/7 | cs.CV | 2021
     "Latent (BYOL) where ResNet-50 is used as the backbone in each of the networks. When the self-supervised models are fine-tuned with only 5% of the labels from the synthetic dataset,..."
  3. [0.8949] Classification of Seeds using Domain Randomization on Self-Supervised
     чанк 2/7 | cs.CV | 2021
     "attributes of a seed such as size, shape, area, color and texture to identify the seed type, a process that is tedious and labor-intensive. Advances in the areas of computer vision..."
  4. [0.8938] Approach to Predicting News -- A Precise Multi-LSTM Network With BERT
     чанк 3/7 | cs.CL | 2022
     "we found out that the pre-processing of categorizing the news is still done by human labor. However, human labor may cause mistakes and cannot work for a long time. The growing dem..."
  5. [0.8905] A geometrical imaging of the real gap between economies of China and t
     чанк 4/7 | q-fin.GN | 2017
     "different countries depend on the path that each country passes through and local metric. To address distance between economies of China and of the US we need to know their utility..."
  --- arxiv-chunks-semantic ---
  1. [0.8988] Classification of Seeds using Domain Randomization on Self-Supervised
     чанк 6/6 | cs.CV | 2021
     "When the self-supervised models are fine-tuned with only 5% of the labels from the synthetic dataset, results show that MoCo, the model that yields the best performance of the self..."
  2. [0.8967] Approach to Predicting News -- A Precise Multi-LSTM Network With BERT
     чанк 3/6 | cs.CL | 2022
     "However, human labor may cause mistakes and cannot work for a long time. The growing demands for automatic machines in the near decades show that while the machine can do as good a..."
  3. [0.8918] Approach to Predicting News -- A Precise Multi-LSTM Network With BERT
     чанк 2/6 | cs.CL | 2022
     "It also shows that the "made-up news" has caused a great deal of confusion in Taiwanese society and has serious impacts on global stability. Although there are several applications..."
  4. [0.8909] CFRNet: Cycle-Consistent Fixed-Point Training for Real-Time Blind Face
     чанк 1/7 | cs.CV | 2026
     "Blind face restoration on consumer devices has to balance image quality against speed and memory...."
  5. [0.8906] CFRNet: Cycle-Consistent Fixed-Point Training for Real-Time Blind Face
     чанк 4/7 | cs.CV | 2026
     "Instead of training the network for one pass and then running it several times by hand, we train it to act as a fixed-point operator, so that applying it again to a restored face d..."
```

</details>

---

## Частина 5 — Гібридний пошук через RRF (додаткова частина)

BM25-індекс побудований локально за `title + abstract` усіх 10 000 статей
(1 692 584 токени, словник 39 845 термінів), векторне плече — той самий індекс
`arxiv-papers`, об'єднання — Reciprocal Rank Fusion:

$$\text{RRF}(d)=\sum_{i}\frac{1}{k+\text{rank}_i(d)},\qquad k=60.$$

RRF свідомо не чіпає скори: BM25 дає необмежену шкалу (тут 9–24), cosine — [−1, 1],
і їх не можна ні додавати, ні усереднювати без відкаліброваної нормалізації. Позиції ж
порівнянні завжди.

### Порівняльна таблиця: 3 запити × 3 методи (топ-1 і вердикт)

| Запит | BM25 | Векторний | Гібрид (RRF) |
|---|---|---|---|
| **«BERT fine-tuning»**<br>(точний термін) | *BERT-LSH: Reducing Absolute Compute For Attention* (18.14)<br>✅ усі 5 — про BERT/NLP | *Quantity vs. Quality: Hyperparameter Optimization for Deep RL* (0.9357)<br>❌ жодної статті про BERT | *BERT-LSH* (0.01639)<br>✅ правильний топ-1, видача чергує обидва списки |
| **«Yann LeCun convolutional networks»**<br>(ім'я автора) | *Introduction to Latent Variable Energy-Based Models* (24.13)<br>⚠️ 1 з 5 статей LeCun — і то тому, що прізвище згадано в анотації | *Deep Learning: From Basics to Building Deep Neural Networks* (0.9472)<br>❌ 0 статей LeCun, але тематично влучно | *Comparison between layer-to-layer network training…* (0.03016)<br>⚠️ підняв єдиний документ, знайдений обома |
| **«making computers understand human emotions from text»**<br>(перефразування) | *Facial Expression Classification using Fusion of Deep Neural Networks* (18.36)<br>❌ топ-1 про обличчя, №4 про голос — не текст | *A Sentiment-and-Semantics-Based Approach for Emotion Detection in Text* (0.9523)<br>✅ усі 5 — cs.CL про емоції в тексті | *A Sentiment-and-Semantics-Based Approach…* (0.03178)<br>✅ правильний топ-1 + 3 консенсусні статті вгорі |

### Відповіді

**Який метод дав кращий результат і чому?**
Жоден — і це і є головний результат. На запиті **«BERT fine-tuning»** BM25 безальтернативно
кращий: `BERT` — це рідкісний токен із високим IDF, і п'ять статей, де він стоїть у
заголовку й анотації, знаходяться точно. Векторний пошук тут провалився повністю: у топ-5
немає жодної статті про BERT, замість них — оптимізація гіперпараметрів і meta-black-box
бенчмарки зі схожістю 0.930–0.936. Причина не в «поганій моделі», а в розбіжності задач:
SPECTER2 навчена кодувати **документ** (заголовок + повна анотація, 6M триплетів цитувань),
а дві службові словоформи — це не документ; CLS-вектор такого огрика потрапляє в густу
«серединну» область простору, де відстані до тисяч ML-статей майже однакові (звідси
підозріло рівні 0.93). На запиті **«making computers understand human emotions from text»**
картина дзеркальна: у запиті немає ні слова `sentiment`, ні `emotion detection`, ні `NLP`,
і BM25 чіпляється за єдиний спільний токен `emotion`, через що піднімає статті про **емоції
на обличчях** (cs.CV) і **в голосі** (eess.AS) — формально токен є, фактично не та
модальність. Векторний пошук віддає п'ять cs.CL-статей саме про текст. Гібрид в обох
випадках дає правильний топ-1 і — головне — **ніде не провалюється катастрофічно**: він
гірший за переможця на його власному полі (бо чергує його з результатами слабшого методу),
але не має «мертвих зон». Для продакшн-пошуку, де тип запиту невідомий заздалегідь, це
цінніше за найкращий середній скор.

**Чи є документи в топ-5 гібридного пошуку, яких немає в топ-5 окремих методів?**
У наших трьох запитах — ні. Документ, знайдений **лише одним**
методом на позиціях 6–10, пробитися в топ-5 гібриду не може в принципі: `1/(60+6)=0.01515`
завжди менше за `1/(60+5)=0.01538` п'ятого місця конкурента. А ось документ, знайдений
**обома** методами поза їхніми топ-5 (скажімо, на 6-й і 7-й позиціях), легко обійшов би
лідерів: `1/66 + 1/67 = 0.0301` проти `1/61 = 0.0164` — просто в цих трьох запитах таких
подвійних «темних конячок» не трапилося (перетини top-10 склали 0, 1 і 3 документи).
Зате чітко видно саму механіку, яка це робить — **підйом консенсусних документів, яких
бракувало в топ-5 одного з методів**:

* запит про LeCun: *Comparison between layer-to-layer network training and conventional
  network training using Deep CNN* стояв у BM25 лише на **10-й** позиції (тобто поза топ-5),
  а у векторному пошуку на 3-й — і RRF підняв його на **1-е місце гібриду** (0.03016),
  вище за лідерів обох списків, бо це єдиний документ, який знайшли обидва методи;
* запит про емоції: *Measuring Cultural Relativity of Emotional Valence and Arousal* був
  у векторному пошуку на **6-й** позиції (поза топ-5), у BM25 — на 3-й, і в гібриді вийшов
  на **2-е місце**; *Rule-based Emotion Detection on Social Media* був у BM25 на 8-й — і
  піднявся на 3-є.

Тобто RRF не вигадує нових документів, а переносить угору ті, щодо яких **два незалежні
сигнали згодні**, навіть якщо кожен окремо поставив їх невисоко. Саме це й робить гібрид
стійкішим: випадкова помилка одного методу не має підтвердження другим.

**Як зміна параметра k впливає на видачу (k=60 vs k=1)?**
Скрипт міряє це на запиті про емоції (єдиному з трьох, де списки перетинаються — 3 спільні
документи; якщо перетину немає, RRF за будь-якого k просто чергує лідерів, і k ні на що не
впливає):

| | k=1 | k=10 | k=60 | k=1000 |
|---|---|---|---|---|
| №1 | *Sentiment-and-Semantics* (обидва) | *Sentiment-and-Semantics* (обидва) | *Sentiment-and-Semantics* (обидва) | *Sentiment-and-Semantics* (обидва) |
| №2 | ***Facial Expression* (лише BM25 #1)** | *Measuring Cultural Relativity* (обидва) | *Measuring Cultural Relativity* (обидва) | *Measuring Cultural Relativity* (обидва) |
| №3 | *Rule-based Emotion* (обидва) | *Rule-based Emotion* (обидва) | *Rule-based Emotion* (обидва) | *Rule-based Emotion* (обидва) |
| скор №1 | 0.66667 | 0.15758 | 0.03178 | 0.00199 |

При `k=1` вага позиції падає катастрофічно швидко: `1/(1+1)=0.5`, `1/(1+2)=0.333`,
`1/(1+3)=0.25` — перша позиція варта більше, ніж друга й четверта разом. Тому документ
*Facial Expression Classification*, якого векторний пошук узагалі не знайшов, пролізає на
2-е місце гібриду тільки за рахунок того, що був №1 у BM25 — а консенсусна
*Measuring Cultural Relativity* (BM25 #3 + vector #6) з нею не конкурує (0.393 проти 0.5).
Фактично `k=1` — це «почергова видача лідерів», тобто гібрид майже без зиску від об'єднання.
При `k=60` сусідні ранги майже нерозрізненні (`1/61 = 0.01639` проти `1/63 = 0.01587`,
різниця 3%), тому один голос за будь-яку позицію коштує приблизно однаково, а **два голоси
завжди переважають один** — і три консенсусні документи займають весь топ-3. При `k=1000`
ця логіка вироджується в чистий підрахунок голосів: скори 0.00199 (два методи) проти
0.00100 (один), позиції в межах списку вже не мають значення взагалі. Звідси й практичний
висновок: `k≈60` з оригінальної статті Cormack et al. — компроміс, у якому ранг ще
враховується, але консенсус двох систем важливіший за перше місце в одній.

<details>
<summary>Вивід 06_hybrid_search.py</summary>

```text
==============================================================================
Крок 6. Гібридний пошук BM25 + вектори (RRF)
==============================================================================
BM25-індекс: 10000 документів, 1692584 токенів, словник 39845 термінів
Pinecone 'arxiv-papers': 10000 векторів
Завантажую модель allenai/specter2_base (pooling=cls) ...
Loading weights: 100%|██████████| 199/199 [00:00<00:00, 49579.24it/s]
==============================================================================
Запит (точний термін): "BERT fine-tuning"
==============================================================================
  BM25 (top-5):
    1. [18.14] BERT-LSH: Reducing Absolute Compute For Attention
       cs.CL | 2024 | Li Zezheng, Yip Kingston
    2. [17.03] B-PROP: Bootstrapped Pre-training with Representative Words Predic
       cs.IR | 2021 | Ma Xinyu, Guo Jiafeng, Zhang Ruqing, Fan Yixing, Li Yingyan,
    3. [16.86] Adversarial Training for Large Neural Language Models
       cs.CL | 2020 | Liu Xiaodong, Cheng Hao, He Pengcheng, Chen Weizhu, Wang Yu,
    4. [16.66] A BERT-based Hierarchical Classification Model with Applications i
       cs.CL | 2025 | Liu Kun, Liu Tuozhen, Wang Feifei, Pan Rui
    5. [16.34] ANDES at SemEval-2020 Task 12: A jointly-trained BERT multilingual
       cs.CL | 2020 | Pérez Juan Manuel, Arango Aymé, Luque Franco
  Векторний пошук (top-5):
    1. [0.9357] Quantity vs. Quality: On Hyperparameter Optimization for Deep   Re
       cs.LG | 2020 | Hertel Lars, Baldi Pierre, Gillen Daniel L.
    2. [0.9331] On Training in Imagination
       cs.LG | 2026 | Timor Nadav, Shwartz-Ziv Ravid, Goldblum Micah, LeCun Yann,
    3. [0.9322] Loaded DiCE: Trading off Bias and Variance in Any-Order Score Func
       cs.LG | 2019 | Farquhar Gregory, Whiteson Shimon, Foerster Jakob
    4. [0.9313] MetaBox-v2: A Unified Benchmark Platform for Meta-Black-Box Optimi
       cs.LG | 2025 | Ma Zeyuan, Gong Yue-Jiao, Guo Hongshu, Qiu Wenjie, Ma Sijie,
    5. [0.9307] Small Experiments, Cheaper Decisions: A Case Study in Staged Promo
       cs.CL | 2026 | Polania Felipe Chavarro
  Гібрид RRF, k=60 (top-5):
    1. [0.01639] BERT-LSH: Reducing Absolute Compute For Attention
       cs.CL | 2024 | Li Zezheng, Yip Kingston  <- позиції: bm25#1
    2. [0.01639] Quantity vs. Quality: On Hyperparameter Optimization for Deep   Re
       cs.LG | 2020 | Hertel Lars, Baldi Pierre, Gillen Daniel L.  <- позиції: vector#1
    3. [0.01613] B-PROP: Bootstrapped Pre-training with Representative Words Predic
       cs.IR | 2021 | Ma Xinyu, Guo Jiafeng, Zhang Ruqing, Fan Yixing, Li Yingyan,  <- позиції: bm25#2
    4. [0.01613] On Training in Imagination
       cs.LG | 2026 | Timor Nadav, Shwartz-Ziv Ravid, Goldblum Micah, LeCun Yann,   <- позиції: vector#2
    5. [0.01587] Adversarial Training for Large Neural Language Models
       cs.CL | 2020 | Liu Xiaodong, Cheng Hao, He Pengcheng, Chen Weizhu, Wang Yu,  <- позиції: bm25#3
  Документів, знайдених обома методами в межах top-10: 0
  Перетини top-5: BM25 ∩ vector = 0, hybrid ∩ BM25 = 3, hybrid ∩ vector = 2
  Унікальних для гібриду документів у top-5 немає.
==============================================================================
Запит (ім'я автора): "Yann LeCun convolutional networks"
==============================================================================
  BM25 (top-5):
    1. [24.13] Introduction to Latent Variable Energy-Based Models: A Path Toward
       cs.LG | 2023 | Dawid Anna, LeCun Yann
    2. [10.50] Characterizing Types of Convolution in Deep Convolutional Recurren
       cs.LG | 2017 | Huang Che-Wei, Narayanan Shrikanth. S.
    3. [9.89] Graph Convolutional Networks: analysis, improvements and results
       cs.LG | 2019 | Ullah Ihsan, Manzo Mario, Shah Mitul, Madden Michael
    4. [9.67] Structural Compression of Convolutional Neural Networks
       cs.CV | 2017 | Abbasi-Asl Reza, Yu Bin
    5. [9.62] From Interpretable Filters to Predictions of Convolutional Neural
       cs.LG | 2022 | Henna Shagufta, Alcaraz Juan Miguel Lopez
  Векторний пошук (top-5):
    1. [0.9472] Deep Learning: From Basics to Building Deep Neural Networks with P
       cs.LG | 2022 | Vazan Milad
    2. [0.9416] PRKAN: Parameter-Reduced Kolmogorov-Arnold Networks
       cs.LG | 2025 | Ta Hoang-Thang, Thai Duy-Quy, Tran Anh, Sidorov Grigori, Gel
    3. [0.9396] Comparison between layer-to-layer network training and conventiona
       cs.CV | 2023 | Bhyravabhottla Kiran Kumar Ashish, Lee WonSook
    4. [0.9378] DiracNets: Training Very Deep Neural Networks Without Skip-Connect
       cs.CV | 2017 | Zagoruyko Sergey, Komodakis Nikos
    5. [0.9374] Deep Residual Networks with Exponential Linear Unit
       cs.CV | 2016 | Shah Anish, Kadam Eashan, Shah Hena, Shinde Sameer, Shingade
  Гібрид RRF, k=60 (top-5):
    1. [0.03016] Comparison between layer-to-layer network training and conventiona
       cs.CV | 2023 | Bhyravabhottla Kiran Kumar Ashish, Lee WonSook  <- позиції: bm25#10, vector#3
    2. [0.01639] Introduction to Latent Variable Energy-Based Models: A Path Toward
       cs.LG | 2023 | Dawid Anna, LeCun Yann  <- позиції: bm25#1
    3. [0.01639] Deep Learning: From Basics to Building Deep Neural Networks with P
       cs.LG | 2022 | Vazan Milad  <- позиції: vector#1
    4. [0.01613] Characterizing Types of Convolution in Deep Convolutional Recurren
       cs.LG | 2017 | Huang Che-Wei, Narayanan Shrikanth. S.  <- позиції: bm25#2
    5. [0.01613] PRKAN: Parameter-Reduced Kolmogorov-Arnold Networks
       cs.LG | 2025 | Ta Hoang-Thang, Thai Duy-Quy, Tran Anh, Sidorov Grigori, Gel  <- позиції: vector#2
  Документів, знайдених обома методами в межах top-10: 1
  Перетини top-5: BM25 ∩ vector = 0, hybrid ∩ BM25 = 2, hybrid ∩ vector = 3
  Унікальних для гібриду документів у top-5 немає.
==============================================================================
Запит (перефразування без термінів): "making computers understand human emotions from text"
==============================================================================
  BM25 (top-5):
    1. [18.36] Facial Expression Classification using Fusion of Deep Neural Netwo
       cs.CV | 2022 | Phan Kim Ngan, Nguyen Hong-Hai, Huynh Van-Thong, Kim Soo-Hyu
    2. [18.04] A Portrait of Emotion: Empowering Self-Expression through AI-Gener
       cs.AI | 2023 | Lee Yoon Kyung, Park Yong-Ha, Hahn Sowon
    3. [16.07] Measuring Cultural Relativity of Emotional Valence and Arousal usi
       cs.CL | 2013 | Bann Eugene Yuta, Bryson Joanna J.
    4. [15.68] Towards Interpretable and Transferable Speech Emotion Recognition:
       eess.AS | 2021 | Das Sneha, Lønfeldt Nicole Nadine, Pagsberg Anne Katrine, Cl
    5. [14.78] A Sentiment-and-Semantics-Based Approach for Emotion Detection in
       cs.CL | 2017 | Gupta Umang, Chatterjee Ankush, Srikanth Radhakrishnan, Agra
  Векторний пошук (top-5):
    1. [0.9523] A Sentiment-and-Semantics-Based Approach for Emotion Detection in
       cs.CL | 2017 | Gupta Umang, Chatterjee Ankush, Srikanth Radhakrishnan, Agra
    2. [0.9424] Rule-based Emotion Detection on Social Media: Putting Tweets on
       cs.CL | 2014 | Tromp Erik, Pechenizkiy Mykola
    3. [0.9367] Quantifying the Affective Gap: A Zero-Shot Evaluation of LLMs on F
       cs.CL | 2026 | Obiuwevwi Lawrence, Rechowicz Krzysztof J., Johnson Jessica
    4. [0.9304] Evaluating the Capabilities of Large Language Models for Multi-lab
       cs.CL | 2024 | Belay Tadesse Destaw, Azime Israel Abebe, Ayele Abinew Ali,
    5. [0.9249] AIMA at SemEval-2024 Task 10: History-Based Emotion Recognition in
       cs.CL | 2025 | Abootorabi Mohammad Mahdi, Ghazizadeh Nona, Dalili Seyed Ars
  Гібрид RRF, k=60 (top-5):
    1. [0.03178] A Sentiment-and-Semantics-Based Approach for Emotion Detection in
       cs.CL | 2017 | Gupta Umang, Chatterjee Ankush, Srikanth Radhakrishnan, Agra  <- позиції: bm25#5, vector#1
    2. [0.03102] Measuring Cultural Relativity of Emotional Valence and Arousal usi
       cs.CL | 2013 | Bann Eugene Yuta, Bryson Joanna J.  <- позиції: bm25#3, vector#6
    3. [0.03083] Rule-based Emotion Detection on Social Media: Putting Tweets on
       cs.CL | 2014 | Tromp Erik, Pechenizkiy Mykola  <- позиції: bm25#8, vector#2
    4. [0.01639] Facial Expression Classification using Fusion of Deep Neural Netwo
       cs.CV | 2022 | Phan Kim Ngan, Nguyen Hong-Hai, Huynh Van-Thong, Kim Soo-Hyu  <- позиції: bm25#1
    5. [0.01613] A Portrait of Emotion: Empowering Self-Expression through AI-Gener
       cs.AI | 2023 | Lee Yoon Kyung, Park Yong-Ha, Hahn Sowon  <- позиції: bm25#2
  Документів, знайдених обома методами в межах top-10: 3
  Перетини top-5: BM25 ∩ vector = 1, hybrid ∩ BM25 = 4, hybrid ∩ vector = 2
  Унікальних для гібриду документів у top-5 немає.
==============================================================================
Вплив параметра k у RRF
==============================================================================
Запит (перефразування без термінів): "making computers understand human emotions from text"
Документів, які знайшли ОБА методи (у межах top-10): 3
  k=1:
    1. [0.66667] (обидва: bm25#5, vector#1) A Sentiment-and-Semantics-Based Approach for E
    2. [0.50000] (один: bm25#1) Facial Expression Classification using Fusion
    3. [0.44444] (обидва: bm25#8, vector#2) Rule-based Emotion Detection on Social Media:
    4. [0.39286] (обидва: bm25#3, vector#6) Measuring Cultural Relativity of Emotional Val
    5. [0.33333] (один: bm25#2) A Portrait of Emotion: Empowering Self-Express
  k=10:
    1. [0.15758] (обидва: bm25#5, vector#1) A Sentiment-and-Semantics-Based Approach for E
    2. [0.13942] (обидва: bm25#3, vector#6) Measuring Cultural Relativity of Emotional Val
    3. [0.13889] (обидва: bm25#8, vector#2) Rule-based Emotion Detection on Social Media:
    4. [0.09091] (один: bm25#1) Facial Expression Classification using Fusion
    5. [0.08333] (один: bm25#2) A Portrait of Emotion: Empowering Self-Express
  k=60:
    1. [0.03178] (обидва: bm25#5, vector#1) A Sentiment-and-Semantics-Based Approach for E
    2. [0.03102] (обидва: bm25#3, vector#6) Measuring Cultural Relativity of Emotional Val
    3. [0.03083] (обидва: bm25#8, vector#2) Rule-based Emotion Detection on Social Media:
    4. [0.01639] (один: bm25#1) Facial Expression Classification using Fusion
    5. [0.01613] (один: bm25#2) A Portrait of Emotion: Empowering Self-Express
  k=1000:
    1. [0.00199] (обидва: bm25#5, vector#1) A Sentiment-and-Semantics-Based Approach for E
    2. [0.00199] (обидва: bm25#3, vector#6) Measuring Cultural Relativity of Emotional Val
    3. [0.00199] (обидва: bm25#8, vector#2) Rule-based Emotion Detection on Social Media:
    4. [0.00100] (один: bm25#1) Facial Expression Classification using Fusion
    5. [0.00100] (один: bm25#2) A Portrait of Emotion: Empowering Self-Express
  За малого k перша позиція одного списку майже недосяжна: 1/(1+1)=0.5 проти
  1/(1+3)+1/(1+2)=0.58 -- тобто перевагу отримує той, хто просто був №1 десь.
  За великого k різниця між сусідніми ранками стирається (1/(60+1) ~ 1/(60+3)),
  і вирішальним стає те, чи знайшли документ ОБА методи, а не на якій позиції.
  Саме тому k=60 (значення з оригінальної статті Cormack et al.) на практиці
  працює як 'консенсусний' режим, а k=1 -- як 'почергова видача лідерів'.
==============================================================================
Чому запит за іменем автора провалюється в обох методах
==============================================================================
Запит: "Yann LeCun convolutional networks"
Статей, де 'LeCun' зустрічається в title+abstract: 1
Статей, де 'LeCun' є серед авторів:                35
І BM25, і векторний індекс побудовані на title+abstract, тому поле authors
для пошуку просто не існує. Перевірка: додаємо authors у BM25-корпус.
  BM25 по title+abstract:
    1. [24.13] Introduction to Latent Variable Energy-Based Models: A Path Toward
       cs.LG | 2023 | Dawid Anna, LeCun Yann
    2. [10.50] Characterizing Types of Convolution in Deep Convolutional Recurren
       cs.LG | 2017 | Huang Che-Wei, Narayanan Shrikanth. S.
    3. [9.89] Graph Convolutional Networks: analysis, improvements and results
       cs.LG | 2019 | Ullah Ihsan, Manzo Mario, Shah Mitul, Madden Michael
    4. [9.67] Structural Compression of Convolutional Neural Networks
       cs.CV | 2017 | Abbasi-Asl Reza, Yu Bin
    5. [9.62] From Interpretable Filters to Predictions of Convolutional Neural
       cs.LG | 2022 | Henna Shagufta, Alcaraz Juan Miguel Lopez
  BM25 по title+abstract+authors:
    1. [21.14] Deep Convolutional Networks on Graph-Structured Data
       cs.LG | 2015 | Henaff Mikael, Bruna Joan, LeCun Yann
    2. [20.40] Efficient Object Localization Using Convolutional Networks
       cs.CV | 2014 | Tompson Jonathan, Goroshin Ross, Jain Arjun, LeCun Yann, Bre
    3. [20.11] OverFeat: Integrated Recognition, Localization and Detection using
       cs.CV | 2013 | Sermanet Pierre, Eigen David, Zhang Xiang, Mathieu Michael,
    4. [18.85] Introduction to Latent Variable Energy-Based Models: A Path Toward
       cs.LG | 2023 | Dawid Anna, LeCun Yann
    5. [17.28] A Closer Look at Spatiotemporal Convolutions for Action Recognitio
       cs.CV | 2017 | Tran Du, Wang Heng, Torresani Lorenzo, Ray Jamie, LeCun Yann
  Статей саме LeCun у top-5: без authors -- 1, з authors -- 5
==============================================================================
Підсумок: 3 запити x 3 методи (top-1 кожного методу)
==============================================================================
[точний термін] "BERT fine-tuning"
  BM25  : BERT-LSH: Reducing Absolute Compute For Attention
  Vector: Quantity vs. Quality: On Hyperparameter Optimization for Deep   Reinforc
  Hybrid: BERT-LSH: Reducing Absolute Compute For Attention
[ім'я автора] "Yann LeCun convolutional networks"
  BM25  : Introduction to Latent Variable Energy-Based Models: A Path Towards   Au
  Vector: Deep Learning: From Basics to Building Deep Neural Networks with Python
  Hybrid: Comparison between layer-to-layer network training and conventional   ne
[перефразування без термінів] "making computers understand human emotions from text"
  BM25  : Facial Expression Classification using Fusion of Deep Neural Network in
  Vector: A Sentiment-and-Semantics-Based Approach for Emotion Detection in   Text
  Hybrid: A Sentiment-and-Semantics-Based Approach for Emotion Detection in   Text
```

</details>

---

## Частина 6 — Аналіз і висновки

### 6.1. Семантичний пошук vs пошук за ключовими словами

Результати частин 3–5 дають досить різкий поділ.

**Де виграв семантичний пошук.** Запит з Частини 3
`"teaching machines to recognize objects in pictures"` не містить жодного слова, яким
релевантні статті себе описують: у корпусі пишуть `image recognition`, `visual attention`,
`learning model`, а не `pictures` і `teaching machines`. Вектор знайшов їх усі, BM25 за
таким запитом зачепився б хіба за `objects`. Те саме в Частині 5 на запиті
`"making computers understand human emotions from text"`: векторний пошук дав п'ять
cs.CL-статей саме про емоції в тексті, а BM25, маючи спільним лише токен `emotion`,
підняв статтю про емоції **на обличчях** (cs.CV) і про емоції **в голосі** (eess.AS) —
токен присутній, модальність не та. Це і є типовий профіль задачі для векторів: запит
природною мовою, інша лексика, ніж у документі, і релевантність визначається змістом,
а не словами.

**Де виграв пошук за ключовими словами.** Запит `"BERT fine-tuning"` з Частини 5 — повна
протилежність. BM25 видав п'ять статей, де `BERT` стоїть у заголовку й анотації
(*BERT-LSH*, *B-PROP*, *BERT-based Hierarchical Classification*, *ANDES at SemEval*),
а векторний пошук не знайшов **жодної** статті про BERT: у топ-5 опинилися оптимізація
гіперпараметрів і meta-black-box бенчмарки зі схожістю 0.930–0.936. Причина подвійна.
По-перше, `BERT` — токен з дуже високим IDF: він зустрічається рідко, і там, де
зустрічається, він майже гарантує релевантність — рівно той випадок, для якого
придуманий BM25. По-друге, SPECTER2 навчена кодувати **документ** (заголовок + анотація,
6M триплетів цитувань), а двослівний фрагмент — не документ: його CLS-вектор падає в густу
серединну область простору, де відстані до тисяч ML-статей майже однакові (звідси
підозріло рівні 0.93 у всіх п'ятьох результатах). Для пошуку за точними термінами,
абревіатурами, назвами моделей, ідентифікаторами, номерами теорем і кодами помилок
ключовий пошук не просто не гірший — він правильніший.

**Третій випадок — коли не виграє ніхто.** Запит `"Yann LeCun convolutional networks"`
провалили обидва методи: BM25 підняв одну статтю LeCun (і лише тому, що прізвище випадково
згадане в анотації), вектор — жодної. У корпусі 35 статей за участі LeCun, але і
BM25-індекс, і векторний індекс побудовані на `title + abstract`, тобто поля `authors`
для пошуку просто не існує. Коли ми додали `authors` у BM25-корпус, у топ-5 стало **5 з 5**
статей LeCun (*Deep Convolutional Networks on Graph-Structured Data*, *Efficient Object
Localization*, *OverFeat*, …). Висновок, який не зводиться до «семантика проти токенів»:
жодна метрика схожості не замінить правильно спроєктовану схему індексації — атрибутивні
запити (автор, рік, журнал, DOI) закриваються структурованим полем і фільтром, а не
ембеддингом.

**Загальне правило.** Чим вища ймовірність, що автор релевантного документа вжив **саме ті
слова**, що й користувач, — тим більше шансів у BM25; чим більше запит описує зміст
**іншими словами** — тим більше у вектора. Операційно це три перевірки: (1) чи є в запиті
рідкісні високо-IDF токени (назви, абревіатури, ідентифікатори) → ключовий пошук;
(2) чи є запит перефразуванням, питанням або описом ситуації → семантичний;
(3) чи є те, що шукають, атрибутом, а не змістом → структурований фільтр. Оскільки в
реальному пошуку тип запиту заздалегідь невідомий, практичний дефолт — гібрид: у наших
трьох запитах RRF дав правильний топ-1 там, де правий був BM25, і там, де правий був вектор,
програвши переможцю на його полі кілька позицій, але не маючи жодної катастрофічної видачі.

### 6.2. Вплив розміру чанка

Розмір чанка — це компроміс між **точністю локалізації** і **повнотою контексту**, і обидва
полюси псують пошук по-різному.

**Занадто маленький чанк (10–15 слів).** Вектор перестає описувати думку і починає описувати
фразу. По-перше, падає відношення сигналу до шуму: ембеддинг такого фрагмента майже повністю
визначається двома-трьома випадковими термінами, і один і той самий фрагмент
«we evaluate our approach on three benchmark datasets» буде однаково близький до тисяч
статей — видача стає ненадійною і повною майже-дублікатів. По-друге, зникає кореференція:
фрагмент «it outperforms the baseline by 4 points» не містить ні того, що таке `it`, ні на
якій задачі — користувач отримує уривок, який неможливо інтерпретувати без сусідніх чанків.
По-третє, кількість чанків (а значить, вектори, запити й гроші) зростає обернено до розміру,
і разом із нею зростає ризик, що релевантна стаття потрапить у топ через випадковий
службовий фрагмент, а не через свою суть.

**Занадто великий чанк (500+ слів).** Тут протилежна проблема — **розмиття**. Ембеддинг
фіксованої розмірності (768) змушений усереднити кілька різних тем; у підсумку вектор не
близький ні до одної з них. Якщо в чанку описані і метод, і датасет, і споріднені роботи,
то за запитом про метод він проиграє чанку, присвяченому тільки методу. Додатково вмикається
жорстка межа моделі: `max input length = 512` токенів, тобто ~350–400 англійських слів —
усе, що довше, просто **тихо відрізається** токенізатором, і друга половина чанка не впливає
на вектор узагалі. Це найковарніший випадок: помилки немає, індекс наповнюється, але частина
корпусу фактично не проіндексована.

**Оптимум.** Універсального числа немає, він визначається трьома речами: (1) вікном моделі —
чанк не має сенсу робити довшим за `max_seq_length`; (2) природною одиницею сенсу в
документі — для абстрактів це 2–4 речення (приблизно 50–100 слів), для технічної
документації — розділ, для коду — функція; (3) тим, що робитиме споживач видачі — якщо
користувач читає сніпет, чанк має бути самодостатнім (ближче до 100 слів), якщо чанк іде
у контекст LLM, краще дрібніші чанки з перекриттям, бо модель сама зшиє їх. Практичне
правило, яке підтвердилось і на наших даних: цільовий розмір 1/4–1/2 вікна моделі,
перекриття 10–20 % розміру чанка, а межі ставити по реченнях, а не по словах.

### 6.3. Невідповідна метрика: що якби індекс був із `euclidean`

Нехай $a$ і $b$ — нормалізовані вектори, $\lVert a\rVert=\lVert b\rVert=1$. Розкриємо
квадрат евклідової відстані:

$$\lVert a-b\rVert_2^{2}=(a-b)\cdot(a-b)=\lVert a\rVert^{2}+\lVert b\rVert^{2}-2\,(a\cdot b)
=2-2\,(a\cdot b)=2-2\cos(a,b).$$

Отже

$$\lVert a-b\rVert_2=\sqrt{2-2\cos(a,b)},\qquad
\cos(a,b)=1-\frac{\lVert a-b\rVert_2^{2}}{2}.$$

Права частина — **строго монотонно спадна** функція від $\cos(a,b)$: чим більший косинус,
тим менша відстань, похідна $\frac{d}{d\cos}\sqrt{2-2\cos}=-\frac{1}{\sqrt{2-2\cos}}<0$ на
всьому діапазоні $\cos\in[-1,1)$. Монотонне перетворення не може змінити порядок
відсортованого списку, тому **топ-K за L2 на одиничних векторах тотожний топ-K за cosine** —
не просто схожий, а той самий і в тому ж порядку (що й підтверджує `04_search.py`:
`cosine vs L2: однаковий порядок = True, спільних 5/5`, а чисельна перевірка тотожності дала
`max|L2² − (2 − 2·cos)| = 1.13e-06` на всіх 10 000 векторах).

Тобто індекс із `euclidean` на нормалізованій моделі — не катастрофа, а «працює випадково».
Проблеми починаються в трьох місцях. Перше: **скор втрачає звичну семантику**. Pinecone
повертає відстань, а не схожість, тому «більше — краще» перетворюється на «менше — краще»;
усі пороги відсікання (`score > 0.7`), моніторинг і UI, написані під косинус, починають
працювати навпаки, причому діапазон тепер $[0, 2]$. Друге: **втрачається інваріантність**.
Рівність тримається рівно доти, доки кожен вектор одиничний. Варто комусь залити партію
векторів без `normalize_embeddings=True` (або перейти на модель, яка не нормалізує вихід),
як L2 почне плутати «довгий вектор» із «далеким»: у $\lVert a-b\rVert^2=\lVert a\rVert^2+
\lVert b\rVert^2-2a\cdot b$ з'являється доданок $\lVert a\rVert^2$, який ранжує документи
за їхньою власною нормою, а не за схожістю із запитом. Косинус у цій ситуації залишився б
коректним. Третє: **метрику не змінити** — у Pinecone вона фіксується при `create_index`,
тому «тимчасово неправильний» вибір означає перестворення індексу і повне перезаливання
всіх векторів.

Окремо варто сказати, що для **ненормалізованих** векторів три метрики розходяться
принципово: dot product тоді винагороджує довгі вектори (у текстових моделях довжина
корелює з довжиною тексту й частотністю), cosine дивиться лише на кут, а L2 змішує обидва
фактори. Саме тому крок нормалізації в `02_embed.py` — не косметика, а умова, за якої
вибір метрики перестає впливати на ранжування.

### 6.4. Обмеження Pinecone Starter і що робити на 10 мільйонах статей

**З чим зіткнулися фактично.**

1. **Ліміт метаданих 40 KB на вектор** — єдине обмеження, яке прямо змінило код: `abstract`
   у метаданих обрізано до 500 символів, повний текст лишився в parquet і підтягується за
   `arxiv_id` після пошуку (`04_search.py`). На 10 000 статей повні абстракти ще влізли б,
   але схема «вектор + мінімум метаданих, текст поруч» масштабується, а «вектор + усе
   всередині» — ні.
2. **Метрика і розмірність незмінні після `create_index`.** Це означає, що рішення з
   Частини 1 (cosine, 768) не можна «поправити потім» — лише перестворити індекс і
   перезалити всі вектори. Для 10 000 векторів це 1 хв 42 с, для 10 мільйонів — доба роботи
   й окремий бюджет.
3. **Кількість індексів.** В умові сказано, що Starter дає один індекс, але фактично
   безкоштовний тір дозволив створити три (`arxiv-papers`, `arxiv-chunks-fixed`,
   `arxiv-chunks-semantic`) — Частина 4 пройшла без фолбеку. Оскільки покладатися на це
   не можна, у `05_chunking.py` залишено автоматичний перехід на namespaces всередині
   основного індексу, якщо створення чергового індексу відпаде за квотою.
4. **Регіон і хмара зафіксовані** (`aws / us-east-1`) — вибору немає, що для реальних
   проєктів із вимогами до резидентності даних було б блокером.
5. **Пропускна здатність запису.** Заливка пішла лише батчами (50 × 200 векторів);
   поштучний `upsert` 10 000 разів упирався б у rate limit і тривав десятки хвилин.
   Також безкоштовний тір не має реплік, тож на час заливки індекс — єдина точка відмови.

Окремо варто сказати, що найдорожчим ресурсом виявився не Pinecone, а **CPU**: ембеддинг
10 000 анотацій зайняв 34 хв 52 с на 14 потоках, тобто ~4.8 тексту за секунду. Саме це, а
не ліміти БД, визначає, наскільки великий корпус реально обробити.

**Що робити на 10 мільйонах статей.**

*Ембеддинг (головне вузьке місце).* У нашому темпі 10M текстів — це ≈24 доби CPU. Тому
обов'язково GPU: одна A10/4090 дає 500–1000 текстів/с, тобто 3–6 годин; далі — шардинг по
кількох воркерах із чекпоінтами, щоб падіння на 7-й годині не обнуляло роботу. Ембеддинги
писати не одним `.npy` (30 GB у float32), а партіями parquet/`np.memmap` по 100k записів,
з ідемпотентними id — тоді пайплайн відновлюється з місця обриву. Варто також зберігати
float16 (15 GB) — втрата точності на рівні 1e-3 не впливає на ранжування.

*Зберігання і пошук.* 10M × 768 × 4 B = 30 GB «сирих» векторів, і це вже за межами
безкоштовного тіру (100k векторів). Два шляхи. Перший — Pinecone serverless на платному
плані: storage і compute розділені, платиш за обсяг і за читання, шардинг робить сервіс.
Другий — self-hosted Qdrant/Milvus: дорожче в підтримці, але дає скалярну або бінарну
квантизацію (int8 → 7.5 GB, binary + rescoring → ~1 GB у пам'яті при recall@10 близько
0.95), mmap замість RAM, контроль над `m`/`ef_construct` і, головне, передбачувану вартість
на такому обсязі. Якщо дані вже в Postgres, pgvector з HNSW закриває задачу без окремої БД.

*Архітектура пошуку.* (а) **Шардинг за namespace** — за категорією (`cs.*`, `physics`,
`math`) або роком: більшість наукових запитів і так обмежені доменом, а менший простір
пошуку означає і дешевший фільтр, і вищий recall; (б) **двоступеневий retrieval** — ANN
віддає top-200, cross-encoder (наприклад, `ms-marco-MiniLM` або SPECTER2-adhoc) переранжовує
top-20: це закриває саме ту слабкість, яку ми бачили на запиті «BERT fine-tuning», де
dense-модель не розрізняла близькі за темою статті; (в) **гібрид на серверній стороні** —
локальний `rank_bm25` на 10M документів нереалістичний (на 10k документів у нас 1.69M
токенів у пам'яті Python; на 10M це ~1.7B токенів і десятки гігабайт), тому BM25 має жити
в інвертованому індексі (OpenSearch/Elasticsearch) або як sparse-вектори в самій векторній
БД, а RRF виконуватися над двома серверними видачами; (г) **метадані** — у векторній БД
тримати тільки id і поля, за якими реально фільтрують (рік, категорія), а тексти, авторів
і повні записи — у parquet на об'єктному сховищі або в Postgres.

*Експлуатація.* Інкрементальне доіндексування щодня (arXiv дає ~20k нових статей на
місяць) замість перезаливання; вимірювання `recall@k` проти brute-force на вибірці з
10–50k векторів після кожної зміни параметрів індексу; моніторинг latency p95 і вартості
на 1000 запитів. І обов'язково — відділення «тексту правди» (parquet/Postgres) від індексу:
індекс має бути відтворюваним із даних за одну команду, бо на такому обсязі його
доведеться перебудовувати не раз.

