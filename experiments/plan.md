# Experiment Plan

Последовательный план: от реализации инфраструктуры до финальных экспериментов.
Статусы: `[ ]` pending · `[~]` running · `[x]` done · `[!]` failed

Общие параметры (действуют во всех фазах, если не указано иначе): base = FLUX.1-dev,
lr = 5e-5, optimizer = Adam, prompt = "a sks", steps = 1000, rank r = 16, seed = 42,
inference: steps = 28, guidance scale = 3.5, alpha = 1.0.

Описание используемых данных — `experiments/datasets.md`.

### Согласование протоколов оценки (Phase 4 vs B-LoRA §5.1 / DS8)

| Протокол | Данные | Задача | Метрики (ядро) | Где в плане |
|----------|--------|--------|----------------|-------------|
| **A — генерация по тексту (основной FLUX-пайплайн)** | DS1 × 8 картин, **DS2** (100 текстовых промптов), **DS3** для FID | Тот же финальный эксперимент, что описан ниже как Phase 4: стиль задаётся картиной DS1, контент главы — промпты из COCO | по таблицам Phase 4: DINO-/CLIP-сигналы, **FID**, LPIPS (как уже задано столбцами) | Phase 4 |
| **B — парный style-transfer (реплика §5.1)** | **DS8**: пул контента `eval_content/`, стилей `eval_styles/`, манифест **50 пар** `b_lora_eval_pairs.json`, `seed = 42` | Как Table 1 в B-LoRA: для каждой пары (контент-референс, стиль-референс) перенос стиля в изображение; усреднение по 50 парам | **Cosine similarity DINO ViT-B/8** выходного изображения с эмбеддингами **стилевого** и **контентного** референса (две строки таблицы: style / content) | Phase **5.2** (обязательно на SDXL); **Phase 6 M02**, если используете DS8 вместо/вместе с матрицей 3×3 |

Протоколы **взаимно дополняют**: Phase 4 не заменяет §5.1 (нет фиксированных пар изображений-референсов и иной генеративный сценарий). Для главы следует явно подписать, какая таблица к какому протоколу относится.

---

## Шаг 0 — Инфраструктура

Реализуется до любых экспериментов. Делится на две части: код/конфиги и данные.

### 0.1 — Код и конфиги

| ID  | Артефакт                                          | Описание                                                                 | Статус |
|-----|---------------------------------------------------|--------------------------------------------------------------------------|--------|
| I01 | `scripts/eval/compute_metrics.py` (DINO ViT-B/8)  | Cosine similarity DINO-output↔ref (style/content) как в §5.1 Table 1; метрики для Phase 4 (столбцы таблицы) | [x]    |
| I02 | `scripts/analysis/block_analysis.py`              | Prompt-injection анализ блоков FLUX + CLIP-сходство                       | [x]    |
| I03 | `scripts/eval/limitations_eval.py`                | Скрипт количественной оценки ограничений (color/background/complexity)    | [x]    |
| I04 | Конфиги Phase 1b (`dc_content_*.yaml`, 3 шт.)     | Диагностика Θ_content                                                     | [x]    |
| I05 | Конфиги Phase 2b (`dp*.yaml`, 4 шт.)              | Ablation training prompt                                                  | [x]    |
| I06 | Конфиги Phase 4.3 SplitFlux (`e03_*.yaml`, 8 шт.) | Реализация конкурирующего метода из статьи SplitFlux                      | [x]    |
| I07 | Конфиги IP-Adapter-FLUX (`e04_*.yaml`, 2 шт.)     | Дополнительный baseline (включается при наличии стабильной реализации)    | [ ]    |
| I08 | Caption файлы                                      | Единообразный prompt "a sks" во всех 8 файлах                             | [x]    |
| I09 | `configs/experiments/base_flux_lora.yaml`         | lr = 5e-5 (исправлено)                                                    | [x]    |
| I10 | `scripts/data/build_eval_pairs.py` (+ `--validate-only`) | Сборка/проверка манифеста **50 пар** для DS8 (`b_lora_eval_pairs.json`), `seed=42` | [x]    |

### 0.2 — Данные (см. `datasets.md`)

| ID  | Артефакт                                          | Назначение                                                                | Статус |
|-----|---------------------------------------------------|---------------------------------------------------------------------------|--------|
| D01 | `data/styles/` (DS1)                              | 8 стилевых референсов Van Gogh + Monet                                    | [x]    |
| D02 | `data/coco_prompts.txt` (DS2)                     | 100 промптов из COCO val2017 с фиксированным протоколом отбора            | [x]    |
| D03 | `data/artbench10/{impressionism,post_impressionism}/` (DS3) | Структурировать ArtBench-10 по жанрам для FID-референса         | [x]    |
| D04 | `scripts/analysis/generate_block_prompts.py` + `experiments/data/block_analysis_prompts.json` (DS4) | 200 пар промптов для Phase 0 | [x]    |
| D05 | `data/dreambooth_*` / subject-папки (DS5)         | Canonical subjects: минимум **dog** для Phase 1b; полный DS5 — источник `eval_content/*` для DS8 | [x]    |
| D06 | `experiments/data/limitations_prompts.json` (DS6) | 5 + 5 промптов для L01 и L03                                              | [x]    |
| D07 | `scripts/data/make_center_crops.py` + `data/styles_cropped/` (DS7) | Центр-кропы стилей для L02                                | [x]    |
| D08 | DS8 — `data/eval_content/`, `data/eval_styles/`, `experiments/data/b_lora_eval_pairs.json` (+ опц. DVC, `eval_assets_registry.yaml`) | Парная выборка **§5.1**: 23/25 объектов как в `datasets.md`, **50 пар** для Phase 5.2 и опц. M02 | [x]    |

---

## Phase 0 — Анализ блоков FLUX (prompt-injection + CLIP)

Воспроизводит Section 4.1 оригинальной статьи. Для каждого блока подменяется текстовый эмбеддинг и измеряется
CLIP-сходство выхода с альтернативным промптом. Определяет целевые блоки для всех последующих фаз.

**Промпты:** `P_content = "A photo of a {object}"`, `P_style = "A photo of a {color} {object}"`, 200 пар.

| ID    | Блоки                                 | Выход                                                 | Статус |
|-------|---------------------------------------|-------------------------------------------------------|--------|
| P0-DS | Double-stream 0–18                    | CLIP-сходство content/style по 19 блокам              | [x]    |
| P0-SS | Single-stream 0–37 (все 38 блоков)    | CLIP-сходство content/style по 38 блокам              | [x]    |
| P0-R  | Сводный отчёт `output/analysis/block_analysis_report.md` | Ранжирование DS+SS, фазовые границы, выбор блоков | [x]    |

---

## Phase 1 — Диагностика гипотезы double-stream (группа `diag_d`)

Проверка: несут ли double-stream блоки стилевые признаки? Устанавливает baseline без LoRA.

| ID  | Конфиг                              | Блоки              | r  | Steps | DINO-style | CLIP-style | CLIP-content | FID    | LPIPS | Статус |
|-----|-------------------------------------|--------------------|----|-------|------------|------------|--------------|--------|-------|--------|
| E00 | `e00_no_lora_baseline.yaml`         | нет LoRA           | —  | —     | 0.046      | 0.463      | 0.255        | 283.3  | 0.840 | [x]    |
| D01 | `d01_double_stream_1000steps.yaml`  | double-stream 0–18 | 16 | 1000  | **0.191**  | **0.486**  | 0.254        | **246.3** | **0.790** | [x] |
| D02 | `d02_double_stream_2000steps.yaml`  | double-stream 0–18 | 16 | 2000  | 0.162      | 0.483      | 0.250        | 257.5  | 0.790 | [x]    |
| D03 | `d03_double_stream_rank32.yaml`     | double-stream 0–18 | 32 | 1000  | 0.102      | 0.467      | 0.255        | 271.8  | 0.810 | [x]    |

---

## Phase 1b — Поиск Θ_content

Проверка возможности выделить отдельный адаптер содержания (ΔW⁴ в терминах оригинальной статьи).

| ID    | Конфиг                        | Блоки (кандидат content)     | r  | Steps | Reconstruction | Content isolation | Статус |
|-------|-------------------------------|------------------------------|----|-------|----------------|-------------------|--------|
| DC-01 | `dc_content_early_ds.yaml`    | double-stream [0–8]          | 16 | 1000  | —              | —                 | [ ]    |
| DC-02 | `dc_content_late_ds.yaml`     | double-stream [9–18]         | 16 | 1000  | —              | —                 | [ ]    |
| DC-03 | `dc_content_from_p0.yaml`     | ds\_05--ds\_12 (content-долина Phase 0) | 16 | 1000  | —              | —                 | [ ]    |

---

## GATE 1 (после Phase 0 + Phase 1 + Phase 1b)

Решение о дальнейшем плане:

- **Сценарий A** — double-stream подтверждены, Θ_content выделен → полный план Phase 2–6.
- **Сценарий B** — double-stream подтверждены, Θ_content не выделяется → Phase 2–5 без Phase 3b и Phase 6.
- **Сценарий C** — double-stream не дают переноса → возврат к Phase 0 с пересмотром целевых блоков.

---

## Phase 2 — Аблации гиперпараметров

### Phase 2.1 (D-A) — диапазон double-stream блоков

Van Gogh img1, r = 32, 1000 steps.

| ID   | Конфиг                      | Блоки        | DINO-style | CLIP-style | CLIP-content | FID | LPIPS | Статус |
|------|-----------------------------|--------------|------------|------------|--------------|-----|-------|--------|
| DA01 | `da01_ds_blocks_0_6.yaml`   | DS [0–6]     | —          | —          | —            | —   | —     | [ ]    |
| DA02 | `da02_ds_blocks_0_12.yaml`  | DS [0–12]    | —          | —          | —            | —   | —     | [ ]    |
| DA03 | `da03_ds_blocks_0_18.yaml`  | DS [0–18]    | —          | —          | —            | —   | —     | [ ]    |
| DA04 | `da04_ds_blocks_6_18.yaml`  | DS [6–18]    | —          | —          | —            | —   | —     | [ ]    |

### Phase 2.2 (D-B) — ранг LoRA

Блоки = лучший из D-A, 1000 steps.

| ID   | Конфиг                 | r     | DINO-style | CLIP-style | CLIP-content | FID | LPIPS | Статус |
|------|------------------------|-------|------------|------------|--------------|-----|-------|--------|
| DB01 | `db01_rank_4.yaml`     | 4     | —          | —          | —            | —   | —     | [ ]    |
| DB02 | `db02_rank_16.yaml`    | 16    | —          | —          | —            | —   | —     | [ ]    |
| DB03 | `db03_rank_32.yaml`    | 32    | —          | —          | —            | —   | —     | [ ]    |
| DB04 | `db04_rank_64.yaml`    | 64    | —          | —          | —            | —   | —     | [ ]    |

### Phase 2.3 (D-C) — число шагов обучения

Блоки = лучший из D-A, r = лучший из D-B.

| ID   | Конфиг                 | Steps | DINO-style | CLIP-style | CLIP-content | FID | LPIPS | Статус |
|------|------------------------|-------|------------|------------|--------------|-----|-------|--------|
| DC01 | `dc01_steps_500.yaml`  | 500   | —          | —          | —            | —   | —     | [ ]    |
| DC02 | `dc02_steps_1000.yaml` | 1000  | —          | —          | —            | —   | —     | [ ]    |
| DC03 | `dc03_steps_2000.yaml` | 2000  | —          | —          | —            | —   | —     | [ ]    |
| DC04 | `dc04_steps_4000.yaml` | 4000  | —          | —          | —            | —   | —     | [ ]    |

### Phase 2.4 (D-P) — training prompt

Блоки = лучший из D-A, r = лучший из D-B, steps = лучший из D-C.

| ID   | Конфиг                         | Training prompt                | DINO-style | CLIP-content | Статус |
|------|--------------------------------|--------------------------------|------------|--------------|--------|
| DP01 | `dp01_prompt_sks.yaml`         | "a sks"                        | —          | —            | [ ]    |
| DP02 | `dp02_prompt_sks_class.yaml`   | "a sks painting"               | —          | —            | [ ]    |
| DP03 | `dp03_prompt_v.yaml`           | "a [v]"                        | —          | —            | [ ]    |
| DP04 | `dp04_prompt_v_class.yaml`     | "a [v] painting in [s] style"  | —          | —            | [ ]    |

---

## Phase 3 — Alpha (lora_scale)

Тестируется инференсом без переобучения (через переменную `LORA_SCALE`). Базовая конфигурация = финальная из Phase 2.

### Phase 3.1 — общий lora_scale

| ID  | alpha | DINO-style | DINO-content | CLIP-style | Статус |
|-----|-------|------------|--------------|------------|--------|
| G01 | 0.3   | —          | —            | —          | [ ]    |
| G02 | 0.5   | —          | —            | —          | [ ]    |
| G03 | 0.7   | —          | —            | —          | [ ]    |
| G04 | 1.0   | —          | —            | —          | [ ]    |
| G05 | 1.5   | —          | —            | —          | [ ]    |
| G06 | 2.0   | —          | —            | —          | [ ]    |

### Phase 3.2 — раздельный α_style / α_content (только при Сценарии A)

| ID   | α_style | α_content | DINO-style | CLIP-content | Color preservation | Статус |
|------|---------|-----------|------------|--------------|--------------------|--------|
| GS01 | 0.4     | 1.0       | —          | —            | —                  | [ ]    |
| GS02 | 0.5     | 1.0       | —          | —            | —                  | [ ]    |
| GS03 | 0.7     | 1.0       | —          | —            | —                  | [ ]    |
| GS04 | 1.0     | 0.5       | —          | —            | —                  | [ ]    |

---

## Phase 4 — Финальное сравнение методов

**Протокол A** (таблица выше «Согласование протоколов»): финальная конфигурация (блоки + r + steps + prompt + alpha) применяется ко всем 4 изображениям каждого из двух направлений стиля (Van Gogh / Monet из **DS1**). Инференс по тексту — **DS2**, 100 первых промптов; **seed = 42**. Количественно — столбцы DINO-/CLIP-/FID/LPIPS как ниже (**не** смешивать построчно с метриками §5.1 без пояснения).

### Phase 4.1 — B-LoRA-FLUX (предлагаемый метод)

| ID      | Конфиг                                  | Стиль    | img | DINO-style | CLIP-style | CLIP-content | FID | LPIPS | Статус |
|---------|-----------------------------------------|----------|-----|------------|------------|--------------|-----|-------|--------|
| E01-1   | `e01_blora_flux_van_gogh_img1.yaml`     | Van Gogh | 1   | —          | —          | —            | —   | —     | [ ]    |
| E01-2   | `e01_blora_flux_van_gogh_img2.yaml`     | Van Gogh | 2   | —          | —          | —            | —   | —     | [ ]    |
| E01-3   | `e01_blora_flux_van_gogh_img3.yaml`     | Van Gogh | 3   | —          | —          | —            | —   | —     | [ ]    |
| E01-4   | `e01_blora_flux_van_gogh_img4.yaml`     | Van Gogh | 4   | —          | —          | —            | —   | —     | [ ]    |
| E01M-1  | `e01_blora_flux_monet_img1.yaml`        | Monet    | 1   | —          | —          | —            | —   | —     | [ ]    |
| E01M-2  | `e01_blora_flux_monet_img2.yaml`        | Monet    | 2   | —          | —          | —            | —   | —     | [ ]    |
| E01M-3  | `e01_blora_flux_monet_img3.yaml`        | Monet    | 3   | —          | —          | —            | —   | —     | [ ]    |
| E01M-4  | `e01_blora_flux_monet_img4.yaml`        | Monet    | 4   | —          | —          | —            | —   | —     | [ ]    |

### Phase 4.2 — Full-LoRA-FLUX (наивный baseline)

| ID      | Конфиг                                  | Стиль    | img | DINO-style | CLIP-style | CLIP-content | FID | LPIPS | Статус |
|---------|-----------------------------------------|----------|-----|------------|------------|--------------|-----|-------|--------|
| E02-1   | `e02_full_lora_flux_van_gogh_img1.yaml` | Van Gogh | 1   | —          | —          | —            | —   | —     | [ ]    |
| E02-2   | `e02_full_lora_flux_van_gogh_img2.yaml` | Van Gogh | 2   | —          | —          | —            | —   | —     | [ ]    |
| E02-3   | `e02_full_lora_flux_van_gogh_img3.yaml` | Van Gogh | 3   | —          | —          | —            | —   | —     | [ ]    |
| E02-4   | `e02_full_lora_flux_van_gogh_img4.yaml` | Van Gogh | 4   | —          | —          | —            | —   | —     | [ ]    |
| E02M-1  | `e02_full_lora_flux_monet_img1.yaml`    | Monet    | 1   | —          | —          | —            | —   | —     | [ ]    |
| E02M-2  | `e02_full_lora_flux_monet_img2.yaml`    | Monet    | 2   | —          | —          | —            | —   | —     | [ ]    |
| E02M-3  | `e02_full_lora_flux_monet_img3.yaml`    | Monet    | 3   | —          | —          | —            | —   | —     | [ ]    |
| E02M-4  | `e02_full_lora_flux_monet_img4.yaml`    | Monet    | 4   | —          | —          | —            | —   | —     | [ ]    |

### Phase 4.3 — SplitFlux (конкурирующий метод из литературного обзора)

| ID      | Конфиг                                  | Стиль    | img | DINO-style | CLIP-style | CLIP-content | FID | LPIPS | Статус |
|---------|-----------------------------------------|----------|-----|------------|------------|--------------|-----|-------|--------|
| E03-1   | `e03_splitflux_van_gogh_img1.yaml`      | Van Gogh | 1   | —          | —          | —            | —   | —     | [ ]    |
| E03-2   | `e03_splitflux_van_gogh_img2.yaml`      | Van Gogh | 2   | —          | —          | —            | —   | —     | [ ]    |
| E03-3   | `e03_splitflux_van_gogh_img3.yaml`      | Van Gogh | 3   | —          | —          | —            | —   | —     | [ ]    |
| E03-4   | `e03_splitflux_van_gogh_img4.yaml`      | Van Gogh | 4   | —          | —          | —            | —   | —     | [ ]    |
| E03M-1  | `e03_splitflux_monet_img1.yaml`         | Monet    | 1   | —          | —          | —            | —   | —     | [ ]    |
| E03M-2  | `e03_splitflux_monet_img2.yaml`         | Monet    | 2   | —          | —          | —            | —   | —     | [ ]    |
| E03M-3  | `e03_splitflux_monet_img3.yaml`         | Monet    | 3   | —          | —          | —            | —   | —     | [ ]    |
| E03M-4  | `e03_splitflux_monet_img4.yaml`         | Monet    | 4   | —          | —          | —            | —   | —     | [ ]    |

### Phase 4.4 — IP-Adapter-FLUX (опционально)

| ID     | Конфиг                             | Стиль    | Статус |
|--------|------------------------------------|----------|--------|
| E04-VG | `e04_ipadapter_van_gogh.yaml`      | Van Gogh | [ ]    |
| E04-M  | `e04_ipadapter_monet.yaml`         | Monet    | [ ]    |

### Сводная таблица (заполняется по завершении Phase 4)

Среднее по 4 изображениям, Van Gogh:

| Метод            | DINO-style ↑ | CLIP-style ↑ | CLIP-content ↑ | FID ↓ | LPIPS ↓ |
|------------------|--------------|--------------|----------------|-------|---------|
| No-LoRA (E00)    | —            | —            | —              | —     | —       |
| B-LoRA-FLUX      | —            | —            | —              | —     | —       |
| Full-LoRA-FLUX   | —            | —            | —              | —     | —       |
| SplitFlux        | —            | —            | —              | —     | —       |
| IP-Adapter-FLUX  | —            | —            | —              | —     | —       |

Среднее по 4 изображениям, Monet:

| Метод            | DINO-style ↑ | CLIP-style ↑ | CLIP-content ↑ | FID ↓ | LPIPS ↓ |
|------------------|--------------|--------------|----------------|-------|---------|
| No-LoRA (E00)    | —            | —            | —              | —     | —       |
| B-LoRA-FLUX      | —            | —            | —              | —     | —       |
| Full-LoRA-FLUX   | —            | —            | —              | —     | —       |
| SplitFlux        | —            | —            | —              | —     | —       |
| IP-Adapter-FLUX  | —            | —            | —              | —     | —       |

---

## Phase 4b — Анализ ограничений метода

Качественное и количественное исследование проблемных случаев. На каждый случай — 5 генераций.

| ID  | Ограничение         | Протокол                                                            | Метрика       | Статус |
|-----|---------------------|---------------------------------------------------------------------|---------------|--------|
| L01 | Color leakage       | 5 объектов с характерными цветами; α_style = 0.5 vs 1.0             | ΔE colordiff  | [ ]    |
| L02 | Background leakage  | 5 стилей с выраженным фоном; full-frame vs center-cropped train     | DINO-object   | [ ]    |
| L03 | Complex scenes      | 5 сложных промптов COCO; сравнение fidelity сгенерированной сцены    | CLIP-content  | [ ]    |

---

## Phase 5 — Кросс-архитектурное сравнение (B-LoRA-SDXL)

Базовая модель совпадает с оригинальной статьей (**SDXL**). Разделено на два подпротокола.

### Phase 5.1 — Совпадение с Phase 4 по курированному подмножеству DS1 (Van Gogh)

Те же 4 эталонных картины Van Gogh, что в Phase 4 для VG, только пайплайн SDXL+B-LoRA. Monet здесь можно не включать при ограничении по времени (тогда текст ВКР: «единый курированный поднабор как в Phase 4, без расширения на второй художника на SDXL»).

| ID    | Конфиг                              | Стиль    | img | DINO-style | CLIP-style | CLIP-content | FID | LPIPS | Статус |
|-------|-------------------------------------|----------|-----|------------|------------|--------------|-----|-------|--------|
| F01-1 | `e04_blora_sdxl_van_gogh_img1.yaml` | Van Gogh | 1   | —          | —          | —            | —   | —     | [ ]    |
| F01-2 | `e04_blora_sdxl_van_gogh_img2.yaml` | Van Gogh | 2   | —          | —          | —            | —   | —     | [ ]    |
| F01-3 | `e04_blora_sdxl_van_gogh_img3.yaml` | Van Gogh | 3   | —          | —          | —            | —   | —     | [ ]    |
| F01-4 | `e04_blora_sdxl_van_gogh_img4.yaml` | Van Gogh | 4   | —          | —          | —            | —   | —     | [ ]    |

### Phase 5.2 — Парная оценка по **§5.1 / DS8** (аналог Table 1)

Источники пар и числа объектов стиля/контента — см. **`datasets.md` (DS8)**. Для каждой из **50** строк манифеста `experiments/data/b_lora_eval_pairs.json`: обучить/склеить метод (см. оригинал: наш B-LoRA; опционально ZipLoRA, StyleAligned, StyleDrop, DB-LoRA+ControlNet), сгенерировать выход; усреднить две метрики **DINO ViT-B/8 cosine** (output↔style ref, output↔content ref). При сравнении с литературой указывать, использован ли *full* пул (23×25) или *reduced* из `datasets.md`.

| ID   | Описание | Метрики | Статус |
|------|----------|---------|--------|
| F02  | Батч-прогон по 50 парам DS8: B-LoRA-SDXL (+ baselines по возможности) | Table 1–style (DINO-style sim), Table 1–content (DINO-content sim); опц. поднабор **30** пар для user study | [ ]    |

**Зависимости:** D08, I01 (две косинусные метрики по референс-картинкам), I10.

---

## Phase 6 — Style-content mixing (только при Сценарии A)

Выполняется, если Phase 1b подтвердила выделение Θ_content.

| ID  | Протокол                                                          | Выход              | Статус |
|-----|-------------------------------------------------------------------|--------------------|--------|
| M01 | Матрица 3×3 (3 content × 3 style), swap Θ_content и Θ_style       | Figure в главе 4   | [ ]    |
| M02 | Количественная оценка DINO-style и DINO-content: **либо** усреднение по ячейкам M01, **либо** тот же протокол, что **F02** по манифесту DS8 (выбрать один и зафиксировать в тексте ВКР) | Таблица в главе 4  | [ ]    |

---

## Прогресс

- **Шаг 0.1** (Код и конфиги): 9/10 — I01–I06, I08, I09, I10 ✓ (I07 IP-Adapter опц. — pending)
- **Шаг 0.2** (Данные): 8/8 — D01–D08 ✓ **все датасеты собраны**
- **Phase 0** (Block analysis): 3/3 ✓
- **Phase 1** (diag_d): 4/4 ✓ — лучший: d01 (r=16, 1000 steps), DINO-style=0.191, FID=246.3
- **Phase 1b** (Θ_content): 0/3
- **Phase 2** (D-A/B/C/P): 0/16
- **Phase 3** (Alpha): 0/10 (6 + 4 условно)
- **Phase 4** (Group E): 0/26 (24 основных + 2 IP-Adapter опц.)
- **Phase 4b** (Limitations): 0/3
- **Phase 5** (SDXL): 0/5 (F01 ×4 + **F02** батч DS8)
- **Phase 6** (Mixing): 0/2 (условно)
- **Итого экспериментов:** 7 / **72**
- **Инфраструктура:** 11 / **18** (3 код/конфиги + **8** данные)
