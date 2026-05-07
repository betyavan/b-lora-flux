# Datasets

Данные, используемые экспериментами из `plan.md`. Последовательно: источник → применение → статус.

Все данные версионируются через DVC; в репозитории хранятся `.dvc`-файлы с хэшами,
фактические данные подгружаются через `dvc pull`. Конфигурация путей: `configs/data/datasets.yaml`.

**Протокол B-LoRA §5.1 (количественное сравнение со сторонними методами):** маленький *paired* evaluation set —
пул контентных объектов (по нескольку изображений на объект), пул стилевых референсов, затем **50 случайных пар**
`(content, style)` с фиксированным `seed`, метрики — среднее косинусное сходство **DINO ViT-B/8** между выходом и
референсами (как Table 1 в оригинале). В этом файле это оформлено как **DS8**; остальные DS (DS1–DS7) используются
для тренировок/абляций/FID и **подстраиваются под DS8** там, где пересекаются по роли.

---

## Сводная таблица

| ID  | Датасет                          | Размер                                      | Лицензия       | Фаза (plan.md)          | Статус |
|-----|----------------------------------|---------------------------------------------|----------------|--------------------------|--------|
| DS1 | Styles (Van Gogh + Monet)        | 2 × 4 = 8 изображений                       | public domain  | Phase 1, 2, 3, 4, 5       | [x] собрано |
| DS2 | MS-COCO val2017 (подвыборка)     | 100 промптов (fixed)                        | CC-BY          | Phase 4 inference        | [x] собрано |
| DS3 | ArtBench-10 (жанровые подмножества) | 2 × **500** изображений *(reduced)*     | CC-BY          | Phase 4 FID reference    | [x] собрано |
| DS4 | Block analysis prompts           | 200 пар (object, color)                     | —              | Phase 0                  | [x] собрано |
| DS5 | DreamBooth canonical subjects    | **8** предметов: dog (~30 img) + 7 для DS8  | Apache 2.0     | Phase 1b; **DS8 content** | [x] собрано |
| DS6 | Limitations prompts              | 5 + 5 = 10 промптов                         | CC-BY (COCO)   | Phase 4b (L01, L03)      | [x] собрано |
| DS7 | Center-cropped styles            | 8 изображений                               | public domain  | Phase 4b (L02)           | [x] собрано |
| DS8 | B-LoRA paired eval (§5.1)        | **8** объектов × **8** стилей → **50** пар *(reduced)* | DS1 + DS5 | **plan.md** Phase **5.2**; опц. Phase 6 M02 | [x] собрано |

---

## DS8 — Paired evaluation (аналог набора из B-LoRA §5.1)

### Что воспроизводим из статьи

| Элемент оригинала | Значение | Как закрепить в репозитории |
|-------------------|----------|----------------------------|
| Контент-пул | **8** объектов из DS5 (dog + 7 предметов DreamBooth) | папки `data/eval_content/{subject_id}/` |
| Стиль-пул | **8** референсов из DS1 (ван Гог × 4 + Моне × 4) | `data/eval_styles/{style_id}.jpg` или подпапки |
| Количественная выборка | **50** случайных пар из 8×8 = 64 возможных | манифест JSON + `seed=42` |
| User study (опционально) | **30** пар из того же манифеста | список индексов `pair_ix` в JSON |

Источники контента в оригинале: работы по персонализации **[15], [33], [45], [52]** — наборы вида «несколько фото одного экземпляра» (не один снимок на всю выборку). У нас те же классы источников, практически через **DreamBooth dataset** от Google **[45]** (несколько subject-папок) + при необходимости открытые демонаборы Textual Inversion / multi-concept.

### Структура каталогов (целевая)

```text
data/eval_content/
  dog/          ← из DS5 (`data/dreambooth_dog/`, после копирования или symlink)
  cat/
  stuffed_toy_green/
  ...          ← остальные subject из DreamBooth repo / аналоги
data/eval_styles/
  s01.jpg      ← можно копировать/линковать DS1 как style_001…008
  ...
  s25.jpg      ← до 17 доп. из открытых примеров [22]/[48] или собственная курировка
experiments/data/b_lora_eval_pairs.json   # manifest 50 пар + метаданные
```

После сборки добавьте ключи в `configs/data/datasets.yaml` (`eval_content`, `eval_styles`, `eval_pairs_manifest`) — см. текущее соглашение в начале этого файла.

### Маппинг с уже имеющимися DS

| Существующий DS | Роль в DS8 |
|-----------------|------------|
| **DS1** | **8 своих стилевых референсов** (соответствие «20+5» из статьи: у нас **расширенный «свой» блок = 8** картин WikiArt с фиксацией лицензии). При желании строго приблизиться к числу «5»: выделите любые **5** из DS8 как основной «own» блок, остальные DS1 оставьте для Phase 4/6 без обязательного включения в манифест. |
| **DS5** | **Ровно один subject** контент-пула (`eval_content/dog`). Остальные **22 subject** тем же способом: репозиторий Google DreamBooth `dataset/` (subject на папку), лицензия Apache 2.0 — перечень имён совпадает с их README (dog, cat, plushy, … — зафиксировать список в манифесте при загрузке). |

### Стиль-пул: как добить до 25

1. Положить **все 8 файлов DS1** в `data/eval_styles/` со стабильными id `wikivg_01` … `wikimo_08` (или `s01`…`s08`).
2. Добирать **`25 − 8 = 17`** изображениями из:
   - визуальных примеров в репозиториях/supplements **StyleAligned [22]** и **StyleDrop [48]** (сохранить ссылку и лицензию в таблице-реестре ниже — завести `experiments/data/eval_assets_registry.yaml` или секцию в `b_lora_eval_pairs_schema`);
   - или **собственных** фото/арта с явным разрешением, если нужен полностью контролируемый пайплайн.

**Принятый протокол для ВКР** (*reduced protocol*, зафиксировано): Nc = **8** объектов из DS5, Ns = **8** стилей из DS1, **min(50, 8×8) = 50** пар без повторения пары `(subject_id, style_id)` при `seed=42`. Таблицы сравнимы по методологии с оригиналом; абсолютные значения DINO не сопоставимы напрямую с Table 1 статьи — оговорить в тексте ВКР.

### Манифест пар `experiments/data/b_lora_eval_pairs.json`

Рекомендуемый формат (версионирование упрощает воспроизводимость):

```json
{
  "protocol": "b_lora_tab1_style_transfer",
  "seed": 42,
  "n_pairs": 50,
  "pairs": [
    {
      "pair_id": 0,
      "content_subject_id": "dog",
      "content_repr_path": "data/eval_content/dog/img_00.jpg",
      "style_id": "wikivg_01",
      "style_repr_path": "data/eval_styles/wikivg_01.jpg"
    }
  ],
  "user_study_pair_ids": [0, 3, 7, 12, 15, 18, 21, 24, 27, 30, 33, 37, 40, 42, 45, 47, 48, 49, 11, 19, 22, 25, 28, 31, 34, 38, 41, 44, 2, 9]
}
```

Подбор **user_study_pair_ids**: 30 случайных **без возвращения** из `pair_id ∈ [0, 49]` с тем же `seed` (или второй производный seed), как в приложении оригинальной статьи.

**Генерация:** скрипт `scripts/data/build_eval_pairs.py` (создать): читает список доступных subject/style id с диска, фильтрует пустые, семплирует 50 пар, пишет JSON; второй режим `--validate-only` проверяет существование путей.

**DVC:** `experiments/data/b_lora_eval_pairs.json.dvc` после фиксации первой версии.

---

## DS1 — Стилевые референсы (Van Gogh + Monet)

**Источник**: WikiArt, изображения в public domain (автор умер >70 лет назад).

Использование в **DS8**: см. таблицу маппинга выше; файлы можно **не дублировать физически**, если в манифесте указать канонический путь `data/styles/{van_gogh,monet}/img{1..4}.jpg` как `style_repr_path`.

**Состав**:

| Стиль    | img | Название                                | Год  |
|----------|-----|------------------------------------------|------|
| Van Gogh | 1   | Звёздная ночь                            | 1889 |
| Van Gogh | 2   | Красные виноградники                     | 1888 |
| Van Gogh | 3   | Подсолнухи                               | 1888 |
| Van Gogh | 4   | Автопортрет с перевязанным ухом          | 1889 |
| Monet    | 1   | Стог сена в Живерни                      | 1886 |
| Monet    | 2   | Водяные лилии — вечерний эффект          | 1899 |
| Monet    | 3   | Мост Ватерлоо                            | 1903 |
| Monet    | 4   | Мадам Моне с ребёнком                    | 1875 |

**Обработка**: resize до 1024×1024 с центральным кропом, RGB, JPEG quality 95.

**Путь**: `data/styles/{van_gogh,monet}/img{1..4}.jpg`. DVC-файл: `data/styles.dvc`.

**Caption**: `a sks` во всех 8 файлах (унифицирован в Issue I08).

---

## DS2 — MS-COCO val2017 (промпты для инференса)

**Источник**: `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`, лицензия CC-BY.

**Протокол отбора** (зафиксировать в скрипте для воспроизводимости):
1. Скачать `captions_val2017.json`.
2. Отсортировать записи по `image_id` по возрастанию.
3. Для каждого уникального `image_id` взять первую по порядку caption.
4. Взять первые 100 записей.

**Путь**: `data/coco_prompts.txt` (по одной строке на промпт). DVC-файл: `data/coco_prompts.txt.dvc`.

---

## DS3 — ArtBench-10 (референс для FID)

**Источник**: HuggingFace Hub, `Doub7e/ArtBench-10`, лицензия CC-BY.
~60 000 изображений в 10 жанрах по 6 000 на жанр.

**Использование в Phase 4 (FID)**:
- Van Gogh → подмножество **post\_impressionism**.
- Monet → подмножество **impressionism**.
- Из каждого подмножества для FID берётся по **500 изображений** *(reduced protocol для ВКР; оговорить как «approximate FID» в тексте; достаточно для оценки порядка метрики и сравнения конфигураций между собой)*.

> **Важно**: FID с референсной выборкой в 500 изображений менее стабилен, чем стандартный (2048+), но приемлем для ВКР при явном упоминании ограничения. Сравнение методов производится на одинаковых условиях.

> **Важно**: FID считается отдельно по каждому стилю с жанрово-соответствующей подвыборкой,
> а не против всего ArtBench-10. Иначе метрика измеряет "близость к усреднённому искусству",
> а не к целевому стилю.

**Путь**: `data/artbench10/{impressionism,post_impressionism}/`. DVC-файл: `data/artbench10.dvc`.

---

## DS4 — Block analysis prompts (Phase 0)

Воспроизводит протокол Section 4.1 оригинальной статьи. В статье — 400 пар промптов, генерированных ChatGPT; здесь — 200 пар (компромисс с бюджетом).

**Состав**:
- 50 случайных повседневных объектов (cat, guitar, chair, car, bunny, tiger, umbrella, bicycle, …).
- 20 цветов (red, blue, green, yellow, orange, purple, pink, black, white, brown, grey, cyan, magenta, gold, silver, navy, beige, olive, teal, maroon).

**Шаблоны**:
- `P_content = "A photo of a {object}"` × 100 пар (разные пары object, object').
- `P_style = "A photo of a {color} {object}"` × 100 пар (object фиксирован, меняются цвета).

**Генерация**: отдельный скрипт `scripts/analysis/generate_block_prompts.py`, результат — `experiments/data/block_analysis_prompts.json`.

**Формат**:
```json
[
  {"id": 0, "type": "content", "p": "A photo of a cat",    "p_hat": "A photo of a tiger"},
  {"id": 1, "type": "style",   "p": "A photo of a red car", "p_hat": "A photo of a blue car"},
  ...
]
```

---

## DS5 — DreamBooth canonical subjects (Phase 1b + контент-пул DS8)

**Источник**: `https://github.com/google/dreambooth/tree/main/dataset/` (Apache 2.0) — несколько предметных папок (**dog**, cat, plush toy, watercolor, vase, …; точный перечень в README репозитория).

**Назначение**:
- Phase 1b: стабильный **single-subject** контент для поиска Θ_content (**dog** как канонический набор — **5 изображений**, репозиторий Google DreamBooth содержит только few-shot reference set, не полный тренировочный набор).
- **DS8** (контентный пул): **8** предметов — `dog` (канонический) + 7 дополнительных: `cat`, `backpack`, `bowl`, `can`, `clock`, `vase`, `bear`. Каждая папка становится `data/eval_content/<subject>/`. Выбирается по одному представительному кадру на пару в манифесте DS8.

**Путь**: основной наблюдаемый путь собаки — `data/dreambooth_dog/img{1..N}.jpg`; остальные subjects — напр. `data/dreambooth_subjects/<name>/`. DVC-файлы: по одному на subject или общий архив — на усмотрение размера.

**Caption для dog**: `a sks dog` (протокол DreamBooth). Для других subjects — `a sks <class>` в соответствии с README DreamBooth.

---

## DS6 — Limitations prompts (Phase 4b)

Не отдельный датасет, а фиксированные подвыборки из COCO-формата.

### L01 — Color leakage (5 промптов с характерными цветами)

```
a red apple on a wooden table
a yellow school bus parked on a street
blue jeans folded on a chair
a green parrot sitting on a branch
an orange pumpkin in a field
```

### L03 — Complex scenes (5 промптов с множественными объектами)

```
a crowded kitchen with multiple people cooking together
a busy street market with vendors and shoppers
a living room with many furniture pieces and decorations
a classroom full of children sitting at desks with books and pencils
a dining table set with many plates, glasses, and silverware for a dinner party
```

**Путь**: `experiments/data/limitations_prompts.json` (создать).

---

## DS7 — Center-cropped styles (Phase 4b, L02)

Для L02 проверяется рекомендация статьи (Appendix B): центр-кроп стилевого референса снижает background leakage.

**Генерация**: из каждого изображения DS1 создаётся центральный квадратный кроп 512×512 → `data/styles_cropped/{style}/img{1..4}.jpg`.

**Протокол**: fine-tune B-LoRA на обоих вариантах (full-frame DS1 vs DS7), сравнить DINO-object на сгенерированных изображениях с L03-промптами.

**Скрипт**: `scripts/data/make_center_crops.py`.

---

## Применение в фазах (обратный индекс)

| Фаза plan.md            | Использует                          |
|-------------------------|-------------------------------------|
| Шаг 0 (инфраструктура)  | —                                   |
| Phase 0                 | DS4                                 |
| Phase 1 (diag\_d)       | DS1 (Van Gogh img1)                 |
| Phase 1b (Θ\_content)   | DS5 (DreamBooth dog)                |
| Phase 2 (D-A/B/C/P)     | DS1 (Van Gogh img1)                 |
| Phase 3 (Alpha)         | DS1, DS2 (инференс)                 |
| Phase 4.1–4.4 (Group E) | DS1 (train), DS2 (inference prompts), DS3 (FID ref) |
| Phase 4b (L01)          | DS1, DS6.L01                        |
| Phase 4b (L02)          | DS1 vs DS7, DS6.L03                 |
| Phase 4b (L03)          | DS1, DS6.L03                        |
| Phase 5.1 (B-LoRA-SDXL, VG) | DS1 (Van Gogh img1–4) |
| Phase 5.2 (§5.1 Table 1)    | **DS8** (+ D08/I01/I10); baselines как в методичке времени |
| Phase 6 (Mixing)        | DS1, DS5                            |
| Phase 6 (M02)           | **DS8** (количественная матрица по манифесту, по желанию) |

---

## Прогресс

- **Готово**: DS1–DS8 (8 / 8) — все датасеты собраны (2026-05-07).
- **Зафиксированные решения (2026-05-06)**:
  - DS8 → *reduced protocol* (Nc=8, Ns=8, 50 пар, seed=42) — **не менять без пересмотра**
  - DS3 → 500 img/жанр (approximate FID) — упомянуть в тексте ВКР
  - DS5 → 5–6 img/subject (Google DreamBooth few-shot set, Apache 2.0)
  - DS8 стили → только DS1 (без внешних StyleAligned/StyleDrop источников)
