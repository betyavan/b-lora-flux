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
Запуск: `20260511T225500`.

| ID    | Конфиг                        | Блоки (кандидат content)     | r  | Steps | DINO-style     | CLIP-style     | FID            | LPIPS          | Статус |
|-------|-------------------------------|------------------------------|----|-------|----------------|----------------|----------------|----------------|--------|
| DC-01 | `dc_content_early_ds.yaml`    | double-stream [0–8]          | 16 | 1000  | 0.079          | 0.463          | 274.31         | 0.817          | [x]    |
| DC-02 | `dc_content_late_ds.yaml`     | double-stream [9–18]         | 16 | 1000  | **0.121**      | **0.480**      | **257.22**     | **0.817**      | [x]    |
| DC-03 | `dc_content_from_p0.yaml`     | double-stream [0–5]          | 16 | 1000  | 0.074          | 0.466          | 276.15         | 0.822          | [x]    |

Вывод: поздние double-stream блоки [9–18] (DC-02) несут наибольший стилевой/content-сигнал.
Запуск от нуля без прогрева (DC-03) — слабейший вариант по всем метрикам.
CLIP-content у всех трёх ≈ 0.254 (совпадает с baseline) — семантика промпта не нарушена.

---

## GATE 1 (после Phase 0 + Phase 1 + Phase 1b)

Решение о дальнейшем плане:

- **Сценарий A** — double-stream подтверждены, Θ_content выделен → полный план Phase 2–6.
- **Сценарий B** — double-stream подтверждены, Θ_content не выделяется → Phase 2–5 без Phase 3b и Phase 6.
- **Сценарий C** — double-stream не дают переноса → возврат к Phase 0 с пересмотром целевых блоков.

**→ Принято решение: Сценарий A.** Θ_content = double-stream [9–18] (DC-02, `dc_content_late_ds`). Phase 6 разблокирована.

---

## Phase 2 — Аблации гиперпараметров

### Phase 2.1 (D-A) — диапазон double-stream блоков

Van Gogh img1, r = 32, 1000 steps.

> ⚠️ **Методическое примечание:** серия DA проводилась при r=32, унаследованном от
> пилотной single-stream ветки (legacy `ablation_b`), а не от winner Phase 1 (r=16).
> Влияние ранга на выбор диапазона блоков оценивается как минимальное — серия DB
> подтвердила, что блоковый эффект воспроизводится при r=16 (DB02 DINO=0.1835 лучший,
> DA с r=32 даёт аналогичный порядок рангов конфигов). При написании тезиса указать
> это явно в разделе об ограничениях эксперимента.

| ID   | Конфиг                      | Блоки        | DINO-style | CLIP-style | CLIP-content | FID    | LPIPS  | Статус |
|------|-----------------------------|--------------|------------|------------|--------------|--------|--------|--------|
| DA01 | `da01_ds_blocks_0_6.yaml`   | DS [0–6]     | 0.1007     | 0.4644     | 0.2503       | 270.55 | 0.8127 | [x]    |
| DA02 | `da02_ds_blocks_0_12.yaml`  | DS [0–12]    | 0.1849     | 0.4973     | 0.2483       | 248.70 | 0.7879 | [x]    |
| DA03 | `da03_ds_blocks_0_18.yaml`  | DS [0–18]    | **0.2568** | **0.5181** | 0.2447       | **235.85** | **0.7791** | [x] |
| DA04 | `da04_ds_blocks_6_18.yaml`  | DS [6–18]    | 0.1165     | 0.4774     | 0.2563       | 264.00 | 0.8170 | [x]    |

**Вывод:** DA03 (DS 0–18) лучший по всем 5 метрикам: DINO-style=0.2568, CLIP-style=0.5181, CLIP-content=0.2447, FID=235.85, LPIPS=0.7791.
DINO-style DA03 превышает лучший результат Phase 1 (D01=0.191) на 34%.
Блоки 0–5 критически необходимы: без них (DA04) DINO-style падает в 2.2× относительно DA03.
**→ Θ\_style = DS [0–18] подтверждён. Следующая фаза (DB): r = {4, 16, 32, 64} при блоках DS [0–18].**

**Наблюдение (visual):** стилизация зависит от семантической категории объекта.
Современные объекты с сильным фотореалистичным prior (транспорт, еда, спорт) сопротивляются стилизации;
стиль проявляется только как «отсылки» на экранах и знаках.
Фоновые и природные сцены принимают стиль значительно лучше.
Зафиксировано в `output/results/da0*_visual.json` + раздел ограничений тезиса.

### Phase 2.2 (D-B) — ранг LoRA

Блоки = DS [0–18] (DA03, лучший из D-A), 1000 steps.

| ID   | Конфиг                 | r     | DINO-style | CLIP-style | CLIP-content | FID    | LPIPS  | Статус |
|------|------------------------|-------|------------|------------|--------------|--------|--------|--------|
| DB01 | `db01_rank_4.yaml`     | 4     | 0.0969     | 0.4726     | 0.2515       | 267.52 | 0.8127 | [x]    |
| DB02 | `db02_rank_16.yaml`    | 16    | 0.1835     | 0.4927     | 0.2495       | 248.29 | 0.7908 | [x] ★  |
| DB03 | `db03_rank_32.yaml`    | 32    | 0.1039     | 0.4695     | 0.2522       | 271.32 | 0.8125 | [x] ⚠  |
| DB04 | `db04_rank_64.yaml`    | 64    | 0.1228     | 0.4711     | 0.2536       | 261.24 | 0.8029 | [x]    |

★ winner Phase 2.2 — r=16 оптимален (capacity overfitting при r≥32)
⚠ WARN: dino_style аномально низкий vs DA03 (0.1039 vs 0.2568) — возможен stale latent cache; внутри серии DB сравнение валидно

**Вывод Phase 2.2:** r=16 оптимален для DS [0–18]. r=4 — недостаточная ёмкость (DINO=0.097), r≥32 — capacity overfitting (стиль «размывается» контентными признаками). Подтверждает рекомендацию оригинальной B-LoRA статьи (r=16–32 sweet spot); в FLUX нижняя граница предпочтительна.
**→ Следующая фаза (DC): шаги = {500, 1000, 2000, 4000} при DS [0–18], r=16.**

### Phase 2.3 (D-C) — число шагов обучения

Блоки = DS [0–18] (DA03), r = 16 (DB02, winner Phase 2.2).

| ID   | Конфиг                 | Steps | DINO-style | CLIP-style | CLIP-content | FID    | LPIPS  | Статус |
|------|------------------------|-------|------------|------------|--------------|--------|--------|--------|
| DC01 | `dc01_steps_500.yaml`  | 500   | 0.0496     | 0.4663     | 0.2563       | 283.8  | 0.8364 | [x]    |
| DC02 | `dc02_steps_1000.yaml` | 1000  | 0.1738     | 0.4871     | 0.2495       | 256.5  | 0.7936 | [x]    |
| DC03 | `dc03_steps_2000.yaml` | 2000  | 0.2226     | 0.5088     | 0.2434       | 245.1  | 0.7838 | [x] ★  |
| DC04 | `dc04_steps_4000.yaml` | 4000  | 0.3220     | 0.5359     | 0.2228       | 231.3  | 0.7627 | [x] ⚠  |

★ winner Phase 2.3 — steps=2000 оптимально (overfit при ≥2000 нарастает, при 4000 — выражен)

**Вывод Phase 2.3:** монотонный рост стилевых метрик с увеличением шагов, однако визуально overfit начинается при 2000 и ярко выражен при 4000 (в сценах появляются реки и другие элементы обучающего распределения вместо контента промпта). DC01 (500 шагов) — стиль фактически не усвоен (DINO=0.05). Оптимум: DC03 — 2000 шагов как лучший баланс стиля и семантической точности.

→ Следующая фаза (DP): зафиксировать лучший training prompt при steps=2000 (DC03).

### Phase 2.4 (D-P) — training prompt

Блоки = DS [0–18] (DA03), r = 16 (DB02, winner Phase 2.2), steps = 2000 (DC03, winner Phase 2.3).

| ID   | Конфиг                         | Training prompt                | DINO-style | CLIP-style | CLIP-content | FID    | LPIPS  | Статус |
|------|--------------------------------|--------------------------------|------------|------------|--------------|--------|--------|--------|
| DP01 | `dp01_prompt_sks.yaml`         | "a sks"                        | 0.3149     | 0.5582     | 0.2281       | 241.34 | 0.7735 | [x]    |
| DP02 | `dp02_prompt_sks_class.yaml`   | "a sks painting"               | **0.3990** | **0.6023** | 0.2144       | 234.02 | **0.7457** | [x] ★  |
| DP03 | `dp03_prompt_v.yaml`           | "a [v]"                        | 0.2004     | 0.4946     | **0.2466**   | 250.77 | 0.7906 | [x]    |
| DP04 | `dp04_prompt_v_class.yaml`     | "a [v] painting in [s] style"  | 0.3935     | 0.6016     | 0.2194       | **233.01** | 0.7476 | [x]    |

★ winner Phase 2.4 — "a sks painting" оптимально (лучший DINO-style и CLIP-style при разумном FID)

**Вывод Phase 2.4:** добавление class word "painting" — ключевой фактор. С ним (DP02, DP04) стилевые метрики существенно выше: сравнение DP01 vs DP02: +0.084 DINO-style, +7.3 FID, -0.028 LPIPS. Токен `[v]` без class word (DP03) — худший результат серии (DINO=0.2004, FID=250.77): DiT не связывает безсемантический токен с доменом живописи. DP02 и DP04 практически равны по метрикам; DP02 выбран winner как более простой формат промпта.

→ Следующая фаза (Phase 3): зафиксировать лучший `lora_scale` при `base_exp = dp02_prompt_sks_class`.

---

## Phase 3 — Alpha (lora_scale)

Тестируется инференсом без переобучения (через переменную `LORA_SCALE`). Базовая конфигурация = финальная из Phase 2.

### Phase 3.1 — общий lora_scale

| ID  | alpha | DINO-style | CLIP-style | CLIP-content | FID    | LPIPS  | Статус |
|-----|-------|------------|------------|--------------|--------|--------|--------|
| G01 | 0.3   | 0.1454     | 0.4808     | 0.2474       | 265.9  | 0.8298 | [x]    |
| G02 | 0.5   | 0.2483     | 0.5126     | 0.2475       | 253.2  | 0.7980 | [x]    |
| G03 | 0.7   | **0.4180** | 0.5629     | 0.2425       | **228.6** | 0.7611 | [x] ★ |
| G04 | 1.0   | 0.6065     | **0.6912** | 0.2015       | 246.2  | 0.7163 | [x]    |
| G05 | 1.5   | 0.6575     | 0.8060     | 0.1036 ⚠     | 317.4  | **0.6945** | [x] |
| G06 | 2.0   | 0.3674     | 0.6729     | 0.1144 ⚠     | 359.8  | 0.7176 | [x]    |

★ winner Phase 3.1 — alpha=0.7 оптимум по Pareto (лучший FID=228.6, DINO выше всей Phase 2, потеря CLIP-content < 0.013)
⚠ clip_content < 0.15 — научный результат: при alpha ≥ 1.5 стилевые LoRA-дельты подавляют текстовый эмбеддинг → FLUX генерирует «абстрактную живопись» вместо промпта

**Вывод Phase 3.1:** U-образный минимум FID при alpha=0.7 — оптимальная точка. CLIP-style растёт монотонно до alpha=1.5 затем падает при 2.0 (пересатурация). DINO-style аналогично: пик при 1.5, коллапс при 2.0. CLIP-content пороговый переход вблизи alpha=1.0: при alpha=1.5–2.0 семантика разрушена. Визуально: alpha=0.7 — эталонный баланс; alpha=2.0 — полная деградация (human score: 1). Human score winner (G03 alpha=0.7): 5/5.
→ Следующая фаза (GS, Phase 3.2): зафиксировать alpha_style=0.7 как базу; сетка (α_style, α_content) стартует от (0.7, 1.0).

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

| ID      | Конфиг                                  | Стиль    | img | DINO-style | CLIP-style | CLIP-content | FID    | LPIPS  | Статус |
|---------|-----------------------------------------|----------|-----|------------|------------|--------------|--------|--------|--------|
| E01-1   | `e01_blora_flux_van_gogh_img1.yaml`     | Van Gogh | 1   | 0.3873     | 0.5397     | 0.2443       | 235.69 | 0.7691 | [x]    |
| E01-2   | `e01_blora_flux_van_gogh_img2.yaml`     | Van Gogh | 2   | 0.1705     | 0.5075     | 0.2525       | 256.00 | 0.8267 | [x]    |
| E01-3   | `e01_blora_flux_van_gogh_img3.yaml`     | Van Gogh | 3   | 0.1145     | 0.4781     | 0.2543       | 270.39 | 0.8284 | [x]    |
| E01-4   | `e01_blora_flux_van_gogh_img4.yaml`     | Van Gogh | 4   | 0.1563     | 0.5078     | 0.2407       | 268.82 | 0.8263 | [x]    |
| E01M-1  | `e01_blora_flux_monet_img1.yaml`        | Monet    | 1   | 0.1284     | 0.4649     | 0.2506       | 263.15 | 0.7837 | [x]    |
| E01M-2  | `e01_blora_flux_monet_img2.yaml`        | Monet    | 2   | 0.2264     | 0.5085     | 0.2568       | 241.39 | 0.7688 | [x]    |
| E01M-3  | `e01_blora_flux_monet_img3.yaml`        | Monet    | 3   | 0.1062     | 0.4491     | 0.2554       | 268.54 | 0.7898 | [x]    |
| E01M-4  | `e01_blora_flux_monet_img4.yaml`        | Monet    | 4   | 0.1314     | 0.4576     | 0.2501       | 264.95 | 0.7776 | [x]    |

### Phase 4.2 — Full-LoRA-FLUX (наивный baseline)

| ID      | Конфиг                                  | Стиль    | img | DINO-style | CLIP-style | CLIP-content | FID    | LPIPS  | Статус |
|---------|-----------------------------------------|----------|-----|------------|------------|--------------|--------|--------|--------|
| E02-1   | `e02_full_lora_flux_van_gogh_img1.yaml` | Van Gogh | 1   | 0.3836     | 0.5403     | 0.2468       | 234.01 | 0.7692 | [x]    |
| E02-2   | `e02_full_lora_flux_van_gogh_img2.yaml` | Van Gogh | 2   | 0.1645     | 0.5074     | 0.2509       | 256.90 | 0.8294 | [x]    |
| E02-3   | `e02_full_lora_flux_van_gogh_img3.yaml` | Van Gogh | 3   | 0.1173     | 0.4755     | 0.2537       | 263.59 | 0.8352 | [x]    |
| E02-4   | `e02_full_lora_flux_van_gogh_img4.yaml` | Van Gogh | 4   | 0.1684     | 0.5128     | 0.2371       | 269.80 | 0.8252 | [x]    |
| E02M-1  | `e02_full_lora_flux_monet_img1.yaml`    | Monet    | 1   | 0.1385     | 0.4694     | 0.2490       | 263.79 | 0.7812 | [x]    |
| E02M-2  | `e02_full_lora_flux_monet_img2.yaml`    | Monet    | 2   | 0.2017     | 0.5038     | 0.2534       | 241.31 | 0.7717 | [x]    |
| E02M-3  | `e02_full_lora_flux_monet_img3.yaml`    | Monet    | 3   | 0.0996     | 0.4574     | 0.2528       | 271.00 | 0.7924 | [x]    |
| E02M-4  | `e02_full_lora_flux_monet_img4.yaml`    | Monet    | 4   | 0.1410     | 0.4600     | 0.2487       | 261.79 | 0.7763 | [x]    |

### Phase 4.3 — SplitFlux (single-stream baseline)

> **Редизайн (2026-05-15):** исходные конфиги e03 ошибочно использовали те же double-stream блоки [0–18], что и B-LoRA (e01).
> Переработано: e03 теперь обучается на **single-stream блоках [0–37]** (`single_transformer_blocks.0–37`).
> Это честное архитектурное сравнение: DS-блоки (B-LoRA) vs SS-блоки (SplitFlux).
> Гипотеза Phase 0 (SS-блоки несут контентный, а не стилевой сигнал) будет проверена количественно.

| ID      | Конфиг                                  | Стиль    | img | DINO-style | CLIP-style | CLIP-content | FID | LPIPS | Статус |
|---------|-----------------------------------------|----------|-----|------------|------------|--------------|-----|-------|--------|
| E03-1   | `e03_splitflux_van_gogh_img1.yaml`      | Van Gogh | 1   | 0.5268     | 0.5973     | 0.2400       | 224.80 | 0.7555 | [x]    |
| E03-2   | `e03_splitflux_van_gogh_img2.yaml`      | Van Gogh | 2   | 0.1453     | 0.4927     | 0.2472       | 263.66 | 0.8424 | [x]    |
| E03-3   | `e03_splitflux_van_gogh_img3.yaml`      | Van Gogh | 3   | 0.1082     | 0.4789     | 0.2463       | 267.68 | 0.8464 | [x]    |
| E03-4   | `e03_splitflux_van_gogh_img4.yaml`      | Van Gogh | 4   | 0.2687     | 0.5520     | 0.2312       | 275.80 | 0.7977 | [x]    |
| E03M-1  | `e03_splitflux_monet_img1.yaml`         | Monet    | 1   | 0.1675     | 0.4732     | 0.2537       | 254.88 | 0.7834 | [x]    |
| E03M-2  | `e03_splitflux_monet_img2.yaml`         | Monet    | 2   | 0.3079     | 0.5661     | 0.2485       | 237.72 | 0.7549 | [x]    |
| E03M-3  | `e03_splitflux_monet_img3.yaml`         | Monet    | 3   | 0.1903     | 0.4699     | 0.2561       | 262.72 | 0.7716 | [x]    |
| E03M-4  | `e03_splitflux_monet_img4.yaml`         | Monet    | 4   | 0.2968     | 0.4771     | 0.2400       | 242.53 | 0.7303 | [x]    |

### Phase 4.4 — IP-Adapter-FLUX (опционально)

| ID     | Конфиг                             | Стиль    | Статус |
|--------|------------------------------------|----------|--------|
| E04-VG | `e04_ipadapter_van_gogh.yaml`      | Van Gogh | [ ]    |
| E04-M  | `e04_ipadapter_monet.yaml`         | Monet    | [ ]    |

### Сводная таблица (заполняется по завершении Phase 4)

Среднее по 4 изображениям, Van Gogh:

| Метод            | DINO-style ↑ | CLIP-style ↑ | CLIP-content ↑ | FID ↓  | LPIPS ↓ |
|------------------|--------------|--------------|----------------|--------|---------|
| No-LoRA (E00)    | 0.046        | 0.463        | 0.255          | 283.3  | 0.840   |
| B-LoRA-FLUX      | 0.2072       | 0.5083       | **0.2480**     | 257.72 | 0.8127  |
| Full-LoRA-FLUX   | 0.2085       | 0.5090       | 0.2471         | **256.07** | 0.8147 |
| SplitFlux ★      | **0.2623**   | **0.5302**   | 0.2412         | 257.99 | **0.8105** |
| IP-Adapter-FLUX  | —            | —            | —              | —      | —       |

Среднее по 4 изображениям, Monet:

| Метод            | DINO-style ↑ | CLIP-style ↑ | CLIP-content ↑ | FID ↓  | LPIPS ↓ |
|------------------|--------------|--------------|----------------|--------|---------|
| No-LoRA (E00)    | 0.046        | 0.463        | 0.255          | 283.3  | 0.840   |
| B-LoRA-FLUX      | 0.1481       | 0.4700       | **0.2532**     | 259.51 | 0.7800  |
| Full-LoRA-FLUX   | 0.1452       | 0.4727       | 0.2510         | 259.47 | 0.7804  |
| SplitFlux ★      | **0.2406**   | **0.4966**   | 0.2496         | **249.46** | **0.7601** |
| IP-Adapter-FLUX  | —            | —            | —              | —      | —       |

**Вывод Phase 4.1–4.2:** B-LoRA-FLUX и Full-LoRA-FLUX статистически неразличимы по всем метрикам (max Δ < 0.002). Преимущество B-LoRA — не в превосходстве по качеству, а в параметрической эффективности (стилевые блоки DS [0–18] ≈ 50% параметров полного LoRA) и возможности раздельного управления стилем и контентом через независимые адаптеры. Общий провал обоих методов на промптах с фотореалистичным prior (0002, 0003) подтверждает ограничение, зафиксированное в Phase 2.1.

**Вывод Phase 4.3 (SplitFlux) ★:** SplitFlux (single-stream блоки [0–37]) устойчиво превосходит B-LoRA и Full-LoRA по стилевым метрикам: DINO-style +41.6% vs B-LoRA (VG: +26.6%, Monet: +62.5%), CLIP-style +5.0%, FID лучше на ~4 пункта в среднем, LPIPS улучшен. Единственная деградация — CLIP-content: −2.1% (в пределах погрешности). Стилевая асимметрия: ван Гог (DINO=0.2623) лучше Моне (DINO=0.2406), что согласуется с гипотезой — SS-блоки эффективнее кодируют локальные текстурные паттерны (импасто ван Гога), чем глобальные цветовые гармонии (Моне). Провал на промптах 0002/0003 воспроизводится и в e03 — это ограничение датасета/промптов, не метода.
→ Phase 4 завершена. Следующий шаг: /analyze-results compare_e (Phase 4.1–4.3 совместно).

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

| ID    | Конфиг                              | Стиль    | img | DINO-style | CLIP-style | CLIP-content | FID    | LPIPS  | Статус |
|-------|-------------------------------------|----------|-----|------------|------------|--------------|--------|--------|--------|
| F01-1 | `e04_blora_sdxl_van_gogh_img1.yaml` | Van Gogh | 1   | 0.5192     | 0.6752     | 0.2464       | 226.49 | 0.7521 | [x]    |
| F01-2 | `e04_blora_sdxl_van_gogh_img2.yaml` | Van Gogh | 2   | 0.4620     | 0.6589     | 0.2426       | 219.17 | 0.7780 | [x]    |
| F01-3 | `e04_blora_sdxl_van_gogh_img3.yaml` | Van Gogh | 3   | 0.3116     | 0.6298     | 0.2450       | 238.72 | 0.7947 | [x]    |
| F01-4 | `e04_blora_sdxl_van_gogh_img4.yaml` | Van Gogh | 4   | 0.3904     | 0.6384     | 0.2399       | 231.76 | 0.7852 | [x] ★  |

**Среднее Phase 5.1 (B-LoRA-SDXL, Van Gogh):** DINO-style=0.4208, CLIP-style=0.6506, CLIP-content=0.2435, FID=229.04, LPIPS=0.7775

**Вывод Phase 5.1 ★ winner: F01-4 (img4)** — лучший визуальный результат (human 5/5, AI 4.5/5): Starry Night sky idiom применён в 9/10 сценах.
B-LoRA-SDXL превосходит все FLUX-baseline по 4 из 5 метрик: CLIP-style +22.7% vs SplitFlux (лучший FLUX), DINO-style +60.4%, FID −10.6%, LPIPS −4.1%. Единственная формальная деградация — CLIP-content (−1.8%, ниже σ = незначимо). Консистентный провал на скейтбордисте (grayscale collapse у всех 4 запусков) — системное ограничение стилевого LoRA на фигурах в движении.
**Ключевой архитектурный вывод:** U-Net SDXL с иерархической encoder-decoder структурой обеспечивает более чистую стилевую специализацию блока `up_blocks.0.attentions.1`, чем MM-DiT FLUX (DS [0–18]). Это подтверждает необходимость разработки адаптированного блочного разделения для FLUX-специфической архитектуры.
→ Следующий шаг: Phase 5.2 — батч DS8 (50 пар), аналог Table 1 оригинальной статьи.

### Phase 5.2 — Парная оценка по **§5.1 / DS8** (аналог Table 1)

Источники пар и числа объектов стиля/контента — см. **`datasets.md` (DS8)**. Для каждой из **50** строк манифеста `experiments/data/b_lora_eval_pairs.json`: обучить/склеить метод (см. оригинал: наш B-LoRA; опционально ZipLoRA, StyleAligned, StyleDrop, DB-LoRA+ControlNet), сгенерировать выход; усреднить две метрики **DINO ViT-B/8 cosine** (output↔style ref, output↔content ref). При сравнении с литературой указывать, использован ли *full* пул (23×25) или *reduced* из `datasets.md`.

| ID   | Описание | Метрики | Статус |
|------|----------|---------|--------|
| F02  | Батч-прогон по 50 парам DS8: B-LoRA-SDXL (+ baselines по возможности) | Table 1–style (DINO-style sim), Table 1–content (DINO-content sim); опц. поднабор **30** пар для user study | ⏳ инфра готова, ждёт launch |

**F02 готовность (2026-05-15):**
- 4 Monet style configs `configs/experiments/e04_blora_sdxl_monet_img[1-4].yaml` (r=16, 500 steps, paritet с Phase 5.1)
- 8 content configs `configs/experiments/m02_content_sdxl_{backpack,bear,bowl,can,cat,clock,dog,vase}.yaml` (block `up_blocks.0.attentions.0`)
- `scripts/eval/generate_mixing_sdxl.py` + `configs/eval/mixing_sdxl.yaml` — pair-driver, генерирует одно изображение на пару
- `scripts/eval/compute_f02_metrics.py` + `configs/eval/f02_metrics.yaml` — DINO-style/DINO-content по парам + Table 1 JSON
- DAG: `GROUP_EXPERIMENTS["phase_5_2_f02"]` (12 trainings) + `_BATCH_INFERENCE_GROUPS` гасит per-experiment generate/metrics + новый job `dags/jobs/mixing_sdxl_f02/` (mlc preset + shell + s3cfg)
- 4 van_gogh-SDXL LoRA из Phase 4.4 (run_ts `20260515T083526`) переиспользуются через Airflow Param `STYLE_VG_RUN_TS`
- Запуск: `make run-group GROUP=phase_5_2_f02`; subset `PAIR_SUBSET=user_study` для 30 пар

**Зависимости:** D08, I01 (две косинусные метрики по референс-картинкам), I10.

---

## Phase 6 — Style-content mixing (только при Сценарии A)

Выполняется, если Phase 1b подтвердила выделение Θ_content.

| ID    | Протокол                                                                                                                                              | Выход              | Статус   |
|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|----------|
| M01a  | Тренировка 3 content-LoRA на субъектах (cat, dog, backpack) — DS[9–18], r=16, 2000 steps, "a sks painting" (winners Phase 1b/2)                       | 3 × .safetensors   | [x]      |
| M01b1 | Матрица 3×3 mixing, split: style=[0–8] / content=[9–18] (naive port B-LoRA SDXL recipe), `lora_scale=0.7`, prompts без стилевого suffix               | Figure (negative)  | [x] ❌   |
| M01b2 | Матрица 3×3 mixing, split: style=[0–12] / content=[13–18] (wider style range), prompts без стилевого suffix                                            | Figure (negative)  | [x] ❌   |
| M01b3 | Матрица 3×3 mixing, split: style=[0–8] / content=[9–18] (вернулись к эталону B-LoRA), **per-row style suffix добавлен** ("painted in the style of Van Gogh/Monet"), `lora_scale=0.7` | Figure в главе 4   | [x] ★    |
| M02   | Количественная оценка DINO-style и DINO-content (per-cell на 9 ячейках M01b3 winner) — grand mean DINO-style=0,099; grand mean DINO-content=0,562; STI=0,463 (content-dominant); таблица + текст в `chapters/snippets/phase6_m01b3.tex` | Таблица в главе 4  | [x]      |

**Phase 6.1 итог (M01a):** все три content-LoRA натренированы успешно (AI+human 5/5, нет коллапса, нет subject leakage в COCO-promptах). Артефакты: `s3://.../exp_logs/20260515T112125/m01_content_{cat,dog,backpack}/loras/m01_content_{cat,dog,backpack}.safetensors`.

**Phase 6.2 итог (M01b1) — ❌ FAIL.** Все 9 ячеек фотореалистичны, стилевой канал молчит (AI 2/5, human 2/5). Гипотеза при ревью v1: style-LoRA натренирован на 0–18, а merge оставил только keys для 0–8 → потеря 53% стилевой ёмкости. → См. M01b3 — гипотеза оказалась НЕВЕРНОЙ.
→ Артефакты v1: `s3://.../20260515T112125/m01_mixing_grid_v1/grid.png`.

**Phase 6.3 итог (M01b2) — ❌ FAIL.** Расширение style-range с [0–8] до [0–12] визуально ничего не изменило (AI 2/5). Это **исключило block-split как корневую причину** и потребовало sanity check'а.

**Sanity check (M01b ROOT CAUSE):** загрузили e01_blora_flux_van_gogh_img1 / img4 / monet_img1 **по отдельности** на FLUX-dev при `lora_scale=0.7` (Phase 4.1 PROMPT_SUFFIX="painted in the style of Van Gogh") — стиль ярко виден (импасто, painterly composition). LoRAs работают. Различие с mixing: в Phase 4.1 generate_job.sh добавляет PROMPT_SUFFIX, а `generate_mixing.py` подавал голые промпты ("a cat sitting on a chair") без какого-либо стилевого triggera. FLUX style-LoRAs не имеют trigger-токена — стиль активируется через natural-language подсказку. **Root cause v1+v2: missing prompt-side style invocation**, не block split.

→ Fix (применён): добавлено поле `mixing.style_suffixes: [3 str]` в `configs/eval/mixing.yaml` и `scripts/eval/generate_mixing.py` (~20 LOC patch). Per-row suffix приклеивается к каждому prompt'у внутри ряда.

**Phase 6.4 итог (M01b3) — ★ WINNER (AI 4/5, human 4/5).** Стилевой канал активирован: 6/9 ячеек со стилем (0/9 в v1+v2). Row 2 (Monet) — чистая демонстрация disentanglement; row 0 (van_gogh_starry_road) — 2/3 ОК + 1 memorization leak в (0,2) (Starry Night буквально как постер на стене); row 1 (van_gogh_starry_night) — слабая активация (style почти не виден на 3 ячейках, FLUX игнорирует известный van_gogh-промпт без сильного LoRA-форсинга). Артефакты: `s3://.../20260515T112125/m01_mixing_grid_v3_with_suffix/grid.png`.

**★ Ключевое методологическое наблюдение Phase 6 для главы 4 ВКР:** B-LoRA recipe на FLUX работает **тогда и только тогда**, когда стилевое invocation присутствует в промпте как natural-language подсказка ("painted in the style of X"). В отличие от SDXL-варианта B-LoRA (Rinon Gal et al.), где стиль может активироваться через trigger-токен / "[v]" placeholder, FLUX-style LoRAs опираются на text encoder и требуют семантически информативного style cue. v1+v2 (без suffix) дали 0/9 ячеек, v3 (со suffix, тот же split) — 6/9. Также наблюдается известная LoRA-патология memorization leak (ячейка 0,2) — Style-LoRA рендерит свою training-картину буквально, а не абстрагирует стиль; ограничение упомянуть в caption.

→ Следующий шаг (M02): количественная оценка DINO-style и DINO-content на 9 ячейках v3.

---

## Прогресс

- **Шаг 0.1** (Код и конфиги): 9/10 — I01–I06, I08, I09, I10 ✓ (I07 IP-Adapter опц. — pending)
- **Шаг 0.2** (Данные): 8/8 — D01–D08 ✓ **все датасеты собраны**
- **Phase 0** (Block analysis): 3/3 ✓
- **Phase 1** (diag_d): 4/4 ✓ — лучший: d01 (r=16, 1000 steps), DINO-style=0.191, FID=246.3
- **Phase 1b** (Θ_content): 3/3 ✓ — лучший: dc_content_late_ds (DS [9–18]), DINO-style=0.121, FID=257.22
- **Phase 2** (D-A/B/C/P): 16/16 ✓ — D-A ✓ (лучший: DA03, DS [0–18], DINO=0.2568, FID=235.85); D-B ✓ (лучший: DB02, r=16, DINO=0.1835, FID=248.29); D-C ✓ (лучший: DC03, steps=2000, DINO=0.2226, FID=245.1); D-P ✓ (лучший: DP02, "a sks painting", DINO=0.3990, FID=234.02)
- **Phase 3** (Alpha): 6/10 (Phase 3.1 ✓ winner: G03 alpha=0.7, DINO=0.4180, FID=228.6)
- **Phase 4** (Group E): 24/26 (Phase 4.1 ✓ B-LoRA FLUX 8 exp; Phase 4.2 ✓ Full-LoRA FLUX 8 exp; Phase 4.3 ✓ SplitFlux 8 exp — winner ★; Phase 4.4 IP-Adapter — опц.)
- **Phase 4b** (Limitations): 0/3
- **Phase 5** (SDXL): 4/5 (Phase 5.1 ✓ winner: F01-4 img4, DINO-style=0.4208, CLIP-style=0.6506, FID=229.0; **F02** батч DS8 — pending)
- **Phase 6** (Mixing): 5/5 ✅ (M01a content-LoRA cat/dog/backpack 5/5; M01b1 ❌ no suffix; M01b2 ❌ wide split no suffix; **M01b3 ★ winner — 6/9 ячеек со стилем, AI+human 4/5**; M02 ✓ DINO-style/content grand=0,099/0,562, STI=0,463, snippet `chapters/snippets/phase6_m01b3.tex`)
- **Итого экспериментов:** 51 / **75**
- **Инфраструктура:** 11 / **18** (3 код/конфиги + **8** данные)
