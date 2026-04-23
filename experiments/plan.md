# Experiment Plan

Auto-generated structure, statuses synced by `scripts/update_exp_plan.py`. Last updated: 2026-04-24

Статусы: `[ ]` pending · `[~]` running · `[x]` done · `[!]` failed

---

## Соответствие оригинальной статье B-LoRA

> Оригинальная статья (SDXL, UNet): rank=64, lr=5e-5, Adam, 1000 steps, 1 image, prompt="A [v]",
> метрика — **DINO ViT-B/8 cosine similarity**, базовые методы — ZipLoRA / StyleAligned / DB-LoRA / StyleDrop.
> Блоки стиля/контента идентифицированы через **prompt-injection + CLIP-анализ на 400 парах промптов**
> по всем Up-blocks (Section 4.1 статьи). Метод использует **два** адаптера: ΔW⁴ (content) и ΔW⁵ (style),
> применяемые независимо или в комбинации для трёх сценариев: reference-based style transfer,
> text-based stylization, consistent style generation.

**Расхождения с оригиналом и статус их устранения:**

| # | Расхождение | Критичность | Статус |
|---|-------------|-------------|--------|
| 1 | ~~Аблации A/B/C использовали single-stream блоки (19–37) — неверно для FLUX~~ | 🔴 | fixed через D-серию |
| 2 | Метрика DINO ViT-B/8 отсутствует | 🔴 | Issue #1 [ ] |
| 3 | Ранг r=64 не тестировался | 🟡 | fixed: DB04 добавлен |
| 4 | LR=1e-4 вместо 5e-5 | 🟡 | fixed: d01/d02/d03 + base_flux_lora.yaml = 5e-5 |
| 5 | Общий lora_scale вместо раздельного α для style/content | 🟠 | частично: Phase 3; полное решение требует Θ_content |
| 6 | **Нет prompt-injection анализа блоков** (Section 4.1 статьи) | 🔴 | Phase 0, **теперь обязательно до E-серии** |
| 7 | **Обучается только Θ_style; нет Θ_content** | 🔴 | Phase 1b добавлена; диагностика Θ_content |
| 8 | **SplitFlux не в baseline** (противоречит заявленной научной новизне) | 🔴 | добавлен в Phase 4 (E03) |
| 9 | Нет ablation по training prompt ("A [v]" vs "A [v] class") | 🟠 | Phase 2b (D-P) |
| 10 | Нет анализа limitations (color leakage / background / complex scenes) | 🟠 | Phase 4b |
| 11 | Исчерпывающая матрица пар блоков (8×8, Fig. 19–20) не перебирается | 🟡 | вынесено в scope limitations (бюджет) |
| 12 | Style-content mixing / swapping демонстрация отсутствует | 🟡 | Phase 6 (опционально, после Θ_content) |

---

## Приоритетный порядок фаз

Из-за изменений выше меняется зависимость между фазами:

```
Phase 0 (block analysis) ─┐
                          ├──► GATE 1 ──► Phase 2 (D-A/B/C) ─┐
Phase 1 (diag_d) ─────────┤                                  │
                          │                                  ├──► Phase 3 (Alpha G) ──► Phase 4 (E) ──► Phase 5 (F)
Phase 1b (Θ_content) ─────┘                                  │                                              │
                                         Phase 2b (D-P) ─────┘                                              ▼
                                                                                                   Phase 4b (Limitations)
                                                                                                   Phase 6 (Mixing demo)
```

**Правило**: Phase 0, 1, 1b — параллельны, все должны быть выполнены до GATE 1.
**GATE 1** — решение о структуре E-серии принимается только после получения результатов всех трёх фаз.

---

## Phase 0 — Prompt-injection анализ блоков FLUX (методология статьи, раздел 4.1)

> **Цель:** Воспроизвести процедуру идентификации стилевых/контентных блоков из статьи для FLUX.
> Для каждого из 19 double-stream (0–18) и 38 single-stream блоков (0–37) инжектируем альтернативный промпт
> и измеряем CLIP-сходство между выходом и инжектированным промптом.
> Промпты: P_content = "A photo of a {object}", P_style = "A photo of a {color} {object}".
> Минимум 200 пар (компромисс между 400 из статьи и бюджетом).
> Выводим матрицу 57×2 (content response, style response) → определяем целевые блоки.

| ID   | Описание                                      | Блоки           | #пар | Статус |
|------|-----------------------------------------------|-----------------|------|--------|
| P0-DS| Prompt injection, double-stream (0–18)        | DS 0–18         | 200  | [ ]    |
| P0-SS| Prompt injection, single-stream sample        | SS {0,10,20,30,37} | 200 | [ ]    |
| P0-R | Сводный отчёт + выбор target-блоков           | —               | —    | [ ]    |

> ⚠️ Требует скрипта `scripts/analysis/block_analysis.py` (Issue #4).
> **Приоритет изменён**: теперь обязательно до GATE 1, блокирует Phase 4.
> Обоснование: в оригинальной статье выбор блоков (2, 4, 5) следует именно из этого анализа, а не из функциональной аналогии.

---

## Phase 1 — Диагностика: правильные блоки + baseline (группа diag_d)

> **Цель:** Проверить гипотезу о том, что double-stream блоки (0–18) несут стиль в FLUX,
> и установить baseline CLIP/DINO/FID без LoRA.
> Запуск: `GROUP=diag_d` в `blora_flux_group_pipeline`.

| ID   | Конфиг                              | Блоки              | r  | Steps | CLIP-style | CLIP-content | DINO-style | FID   | LPIPS | Статус |
|------|-------------------------------------|--------------------|----|-------|------------|--------------|------------|-------|-------|--------|
| E00  | e00_no_lora_baseline.yaml           | нет LoRA           | —  | —     | —          | —            | —          | —     | —     | [ ]    |
| D01  | d01_double_stream_1000steps.yaml    | double-stream 0–18 | 16 | 1000  | —          | —            | —          | —     | —     | [ ]    |
| D02  | d02_double_stream_2000steps.yaml    | double-stream 0–18 | 16 | 2000  | —          | —            | —          | —     | —     | [ ]    |
| D03  | d03_double_stream_rank32.yaml       | double-stream 0–18 | 32 | 1000  | —          | —            | —          | —     | —     | [ ]    |

---

## Phase 1b — Диагностика Θ_content (новая фаза)

> **Цель:** Проверить, можно ли в FLUX выделить подмножество блоков, специализирующееся на контенте,
> чтобы воспроизвести парадигму двух адаптеров из оригинальной статьи (ΔW⁴ content, ΔW⁵ style).
> Если Phase 0 укажет конкретные "content-блоки" — берутся они; иначе тестируется разбиение double-stream
> на ранние (0–8, кандидат на content) vs поздние (9–18, кандидат на style).

| ID   | Конфиг (планируется)                | Блоки (предполагаемый content) | r  | Steps | Reconstruction? | Content-isolation? | Статус |
|------|-------------------------------------|--------------------------------|----|-------|-----------------|--------------------|--------|
| DC-01| dc_content_early_ds.yaml            | DS [0–8]                       | 16 | 1000  | —               | —                  | [ ]    |
| DC-02| dc_content_late_ds.yaml             | DS [9–18]                      | 16 | 1000  | —               | —                  | [ ]    |
| DC-03| dc_content_from_p0.yaml             | по результатам Phase 0         | 16 | 1000  | —               | —                  | [ ]    |

> **GATE 1b**: если ни одна конфигурация не даёт разделения (content-только vs style-только) → фиксируем
> в scope limitations, что Θ_content не найден, и ограничиваем работу одним Θ_style.
> Если разделение найдено — Θ_content включается в Phase 3 (раздельный α), Phase 4 (E-baseline),
> Phase 6 (style-content mixing).

---

## GATE 1 (после Phase 0 + Phase 1 + Phase 1b)

Решение о структуре Phase 2–6:

- **Сценарий A (гипотеза double-stream подтверждена + Θ_content найден):** полный план со всеми фазами.
- **Сценарий B (double-stream подтверждён, Θ_content не выделяется):** Phase 2–5 выполняются,
  Phase 3 без раздельного α, Phase 6 исключается, scope limitations обновляется.
- **Сценарий C (double-stream не даёт переноса стиля):** возврат к анализу Phase 0,
  пересмотр целевых блоков (возможно, mid single-stream).

---

## Phase 2 — Аблации с правильными блоками (double-stream, после GATE 1)

> Параметры по умолчанию из статьи: lr=5e-5, prompt="A [v]", Adam.

### Аблация D-A — выбор диапазона double-stream блоков (Van Gogh img1, r=32, 1000 steps)

| ID    | Конфиг                     | Блоки (double-stream) | CLIP-style | CLIP-content | DINO-style | FID | LPIPS | Статус |
|-------|----------------------------|-----------------------|------------|--------------|------------|-----|-------|--------|
| DA01  | da01_ds_blocks_0_6.yaml    | [0–6]                 | —          | —            | —          | —   | —     | [ ]    |
| DA02  | da02_ds_blocks_0_12.yaml   | [0–12]                | —          | —            | —          | —   | —     | [ ]    |
| DA03  | da03_ds_blocks_0_18.yaml   | [0–18] (все)          | —          | —            | —          | —   | —     | [ ]    |
| DA04  | da04_ds_blocks_6_18.yaml   | [6–18]                | —          | —            | —          | —   | —     | [ ]    |

> Лучший диапазон → используется в аблациях D-B и D-C.

### Аблация D-B — ранг LoRA (лучшие блоки из DA, 1000 steps)

| ID    | Конфиг                 | Ранг r | CLIP-style | CLIP-content | DINO-style | FID | LPIPS | Статус |
|-------|------------------------|--------|------------|--------------|------------|-----|-------|--------|
| DB01  | db01_rank_4.yaml       | 4      | —          | —            | —          | —   | —     | [ ]    |
| DB02  | db02_rank_16.yaml      | 16     | —          | —            | —          | —   | —     | [ ]    |
| DB03  | db03_rank_32.yaml      | 32     | —          | —            | —          | —   | —     | [ ]    |
| DB04  | db04_rank_64.yaml      | **64** | —          | —            | —          | —   | —     | [ ]    |

> r=64 — значение из оригинальной статьи, обязательно включить.

### Аблация D-C — число шагов (лучшие блоки + ранг из DA/DB)

| ID    | Конфиг                 | Steps | CLIP-style | CLIP-content | DINO-style | FID | LPIPS | Статус |
|-------|------------------------|-------|------------|--------------|------------|-----|-------|--------|
| DC01  | dc01_steps_500.yaml    | 500   | —          | —            | —          | —   | —     | [ ]    |
| DC02  | dc02_steps_1000.yaml   | 1000  | —          | —            | —          | —   | —     | [ ]    |
| DC03  | dc03_steps_2000.yaml   | 2000  | —          | —            | —          | —   | —     | [ ]    |
| DC04  | dc04_steps_4000.yaml   | 4000  | —          | —            | —          | —   | —     | [ ]    |

---

## Phase 2b — Аблация training prompt (D-P, новая фаза)

> Воспроизводит Appendix C / Fig. 21 оригинальной статьи. Нужно учесть, что FLUX использует T5-XXL
> (другая чувствительность к промпту, чем у CLIP в SDXL), поэтому вывод статьи (без class-name лучше)
> требует независимой проверки.

| ID    | Конфиг                    | Training prompt                   | Inference prompt              | CLIP-content | DINO-style | Statu |
|-------|---------------------------|-----------------------------------|-------------------------------|--------------|------------|-------|
| DP01  | dp01_prompt_sks.yaml      | "a sks" (текущий)                 | standard                      | —            | —          | [ ]   |
| DP02  | dp02_prompt_sks_class.yaml| "a sks painting"                  | standard                      | —            | —          | [ ]   |
| DP03  | dp03_prompt_v.yaml        | "a [v]"                           | standard                      | —            | —          | [ ]   |
| DP04  | dp04_prompt_v_class.yaml  | "a [v] painting in [s] style"     | standard                      | —            | —          | [ ]   |

> Базовая конфигурация — лучшая из D-A/B/C. Лучший prompt → используется в Phase 3, 4, 5.

---

## Phase 3 — Аблация Alpha / lora_scale (группа G)

> Из статьи: α=0.4–0.5 для стилевого адаптера лучше сохраняет оригинальные цвета объекта.
> Тестируем через `LORA_SCALE` env var при генерации (без переобучения — только инференс).

### Phase 3a — общий lora_scale (всегда выполняется)

| ID    | LORA_SCALE | Базовый конфиг       | DINO-style | DINO-content | CLIP-style | Статус |
|-------|------------|----------------------|------------|--------------|------------|--------|
| G01   | 0.3        | лучший из D-A/B/C/P  | —          | —            | —          | [ ]    |
| G02   | 0.5        | лучший из D-A/B/C/P  | —          | —            | —          | [ ]    |
| G03   | 0.7        | лучший из D-A/B/C/P  | —          | —            | —          | [ ]    |
| G04   | 1.0        | лучший из D-A/B/C/P  | —          | —            | —          | [ ]    |
| G05   | 1.5        | лучший из D-A/B/C/P  | —          | —            | —          | [ ]    |
| G06   | 2.0        | лучший из D-A/B/C/P  | —          | —            | —          | [ ]    |

### Phase 3b — раздельный α для style/content (только если Θ_content найден в Phase 1b)

> Воспроизводит Fig. 22 и Appendix B оригинала. Рекомендация статьи: α_style ∈ [0.4, 0.5]
> при α_content = 1.0 для решения проблемы color leakage.

| ID    | α_style | α_content | Базовый конфиг | DINO-style | CLIP-content | Color preservation | Статус |
|-------|---------|-----------|----------------|------------|--------------|--------------------|--------|
| GS01  | 0.4     | 1.0       | best           | —          | —            | —                  | [ ]    |
| GS02  | 0.5     | 1.0       | best           | —          | —            | —                  | [ ]    |
| GS03  | 0.7     | 1.0       | best           | —          | —            | —                  | [ ]    |
| GS04  | 1.0     | 0.5       | best           | —          | —            | —                  | [ ]    |

---

## Phase 4 — Финальное сравнение методов (группа E)

> Финальная конфигурация из D-A/B/C/P/G применяется ко всем 4 стилевым изображениям двух стилей.
> Набор промптов: 100 из COCO val2017; seed=42; steps=28; guidance=3.5.

### Phase 4.1 — B-LoRA-FLUX (предлагаемый метод)

| ID      | Конфиг                                  | Стиль    | img | CLIP-style | DINO-style | FID | LPIPS | Статус |
|---------|-----------------------------------------|----------|-----|------------|------------|-----|-------|--------|
| E01-1   | e01_blora_flux_van_gogh_img1.yaml       | Van Gogh | 1   | —          | —          | —   | —     | [ ]    |
| E01-2   | e01_blora_flux_van_gogh_img2.yaml       | Van Gogh | 2   | —          | —          | —   | —     | [ ]    |
| E01-3   | e01_blora_flux_van_gogh_img3.yaml       | Van Gogh | 3   | —          | —          | —   | —     | [ ]    |
| E01-4   | e01_blora_flux_van_gogh_img4.yaml       | Van Gogh | 4   | —          | —          | —   | —     | [ ]    |
| E01M-1  | e01_blora_flux_monet_img1.yaml          | Monet    | 1   | —          | —          | —   | —     | [ ]    |
| E01M-2  | e01_blora_flux_monet_img2.yaml          | Monet    | 2   | —          | —          | —   | —     | [ ]    |
| E01M-3  | e01_blora_flux_monet_img3.yaml          | Monet    | 3   | —          | —          | —   | —     | [ ]    |
| E01M-4  | e01_blora_flux_monet_img4.yaml          | Monet    | 4   | —          | —          | —   | —     | [ ]    |

### Phase 4.2 — Full-LoRA-FLUX (наивный baseline)

| ID      | Конфиг                                  | Стиль    | img | CLIP-style | DINO-style | FID | LPIPS | Статус |
|---------|-----------------------------------------|----------|-----|------------|------------|-----|-------|--------|
| E02-1   | e02_full_lora_flux_van_gogh_img1.yaml   | Van Gogh | 1   | —          | —          | —   | —     | [ ]    |
| E02-2   | e02_full_lora_flux_van_gogh_img2.yaml   | Van Gogh | 2   | —          | —          | —   | —     | [ ]    |
| E02-3   | e02_full_lora_flux_van_gogh_img3.yaml   | Van Gogh | 3   | —          | —          | —   | —     | [ ]    |
| E02-4   | e02_full_lora_flux_van_gogh_img4.yaml   | Van Gogh | 4   | —          | —          | —   | —     | [ ]    |
| E02M-1  | e02_full_lora_flux_monet_img1.yaml      | Monet    | 1   | —          | —          | —   | —     | [ ]    |
| E02M-2  | e02_full_lora_flux_monet_img2.yaml      | Monet    | 2   | —          | —          | —   | —     | [ ]    |
| E02M-3  | e02_full_lora_flux_monet_img3.yaml      | Monet    | 3   | —          | —          | —   | —     | [ ]    |
| E02M-4  | e02_full_lora_flux_monet_img4.yaml      | Monet    | 4   | —          | —          | —   | —     | [ ]    |

### Phase 4.3 — SplitFlux (конкурирующий метод — критично для научной новизны)

> **Критичный baseline**: в введении диплома (`chapter1_introduction.tex`) научная новизна явно
> формулируется как отстройка от SplitFlux ("в отличие от SplitFlux, применяющего single-stream блоки").
> Без численного сравнения заявленная новизна не подтверждена.

| ID      | Конфиг (создать)                        | Стиль    | img | CLIP-style | DINO-style | FID | LPIPS | Статус |
|---------|-----------------------------------------|----------|-----|------------|------------|-----|-------|--------|
| E03-1   | e03_splitflux_van_gogh_img1.yaml        | Van Gogh | 1   | —          | —          | —   | —     | [ ]    |
| E03-2   | e03_splitflux_van_gogh_img2.yaml        | Van Gogh | 2   | —          | —          | —   | —     | [ ]    |
| E03-3   | e03_splitflux_van_gogh_img3.yaml        | Van Gogh | 3   | —          | —          | —   | —     | [ ]    |
| E03-4   | e03_splitflux_van_gogh_img4.yaml        | Van Gogh | 4   | —          | —          | —   | —     | [ ]    |
| E03M-1  | e03_splitflux_monet_img1.yaml           | Monet    | 1   | —          | —          | —   | —     | [ ]    |
| E03M-2  | e03_splitflux_monet_img2.yaml           | Monet    | 2   | —          | —          | —   | —     | [ ]    |
| E03M-3  | e03_splitflux_monet_img3.yaml           | Monet    | 3   | —          | —          | —   | —     | [ ]    |
| E03M-4  | e03_splitflux_monet_img4.yaml           | Monet    | 4   | —          | —          | —   | —     | [ ]    |

> Реализация: по конфигурации из статьи SplitFlux (single-stream блоки, их параметры).
> Issue #6: создать конфиги и валидировать соответствие оригинальной методике.

### Phase 4.4 — IP-Adapter-FLUX (опционально)

| ID      | Конфиг (создать)                        | Метод            | Стиль    | img | Статус |
|---------|-----------------------------------------|------------------|----------|-----|--------|
| E04-VG  | e04_ipadapter_van_gogh.yaml             | IP-Adapter-FLUX  | Van Gogh | all | [ ]    |
| E04-M   | e04_ipadapter_monet.yaml                | IP-Adapter-FLUX  | Monet    | all | [ ]    |

> Включается, если для FLUX.1-dev появится стабильная реализация IP-Adapter.

### Сводная таблица результатов (заполняется по завершении Phase 4)

Среднее по 4 изображениям (Van Gogh):

| Метод             | CLIP-style ↑ | DINO-style ↑ | FID ↓ | LPIPS ↓ |
|-------------------|-------------|-------------|-------|---------|
| No-LoRA (E00)     | —           | —           | —     | —       |
| B-LoRA-FLUX       | —           | —           | —     | —       |
| Full-LoRA-FLUX    | —           | —           | —     | —       |
| SplitFlux         | —           | —           | —     | —       |
| IP-Adapter-FLUX   | —           | —           | —     | —       |

Среднее по 4 изображениям (Monet):

| Метод             | CLIP-style ↑ | DINO-style ↑ | FID ↓ | LPIPS ↓ |
|-------------------|-------------|-------------|-------|---------|
| No-LoRA (E00)     | —           | —           | —     | —       |
| B-LoRA-FLUX       | —           | —           | —     | —       |
| Full-LoRA-FLUX    | —           | —           | —     | —       |
| SplitFlux         | —           | —           | —     | —       |
| IP-Adapter-FLUX   | —           | —           | —     | —       |

---

## Phase 4b — Анализ ограничений метода (воспроизводит Appendix B статьи)

> Качественное и количественное исследование проблемных случаев. Минимум 3 случая по 5 генераций каждый.

| ID   | Ограничение                                      | Протокол                                                      | Метрика         | Статус |
|------|--------------------------------------------------|---------------------------------------------------------------|-----------------|--------|
| L01  | Color leakage (цвет объекта уезжает в стиль)     | 5 объектов с characteristic colors, сравнение α_style=0.5 vs 1.0 | ΔE colordiff   | [ ]    |
| L02  | Background leakage                               | 5 стилей с выраженным фоном, сравнение full vs center-cropped train| DINO-object  | [ ]    |
| L03  | Complex scenes (много объектов)                  | 5 сложных промптов (COCO), сравнение generation fidelity      | CLIP-content    | [ ]    |

> По каждому limitation — описание + 1–2 figures с качественным сравнением + предлагаемый workaround.
> Формат: раздел в главе 4 диплома (новый подраздел "Ограничения метода").

---

## Phase 5 — Кросс-архитектурное сравнение (группа F)

> B-LoRA-SDXL на тех же промптах и стилях. Проверка переносимости подхода.

| ID    | Конфиг                              | Метод       | Стиль    | img | CLIP-style | DINO-style | FID | LPIPS | Статус |
|-------|-------------------------------------|-------------|----------|-----|------------|------------|-----|-------|--------|
| F01-1 | e04_blora_sdxl_van_gogh_img1.yaml   | B-LoRA-SDXL | Van Gogh | 1   | —          | —          | —   | —     | [ ]    |
| F01-2 | e04_blora_sdxl_van_gogh_img2.yaml   | B-LoRA-SDXL | Van Gogh | 2   | —          | —          | —   | —     | [ ]    |
| F01-3 | e04_blora_sdxl_van_gogh_img3.yaml   | B-LoRA-SDXL | Van Gogh | 3   | —          | —          | —   | —     | [ ]    |
| F01-4 | e04_blora_sdxl_van_gogh_img4.yaml   | B-LoRA-SDXL | Van Gogh | 4   | —          | —          | —   | —     | [ ]    |

---

## Phase 6 — Style-content mixing demo (опционально, только при Θ_content)

> Воспроизводит Fig. 1, 27, 28 оригинальной статьи: обмен style и content между двумя стилизованными изображениями.
> Выполняется, только если Phase 1b подтвердила выделение Θ_content.

| ID   | Протокол                                                           | Результат                       | Статус |
|------|--------------------------------------------------------------------|---------------------------------|--------|
| M01  | 3×3 матрица (3 content × 3 style), swap Θ_content и Θ_style        | Figure в главе 4                | [ ]    |
| M02  | Количественная оценка DINO-content (vs content ref) + DINO-style   | Таблица в главе 4               | [ ]    |

> Если Phase 1b даёт негативный результат — фаза исключается, в scope limitations добавляется
> "style-content mixing невозможен без Θ_content; задача оставлена для будущих работ".

---

## Scope limitations (что явно вынесено за рамки работы)

Эти эксперименты из оригинальной статьи **сознательно не воспроизводятся**, с фиксированным обоснованием:

| Пропущенный этап | Обоснование | Куда добавить в диплом |
|------------------|-------------|------------------------|
| Exhaustive 8×8 матрица пар блоков (Fig. 19–20) | Вычислительный бюджет: 64 конфигурации × 1000 шагов × A100-час ≈ 26 GPU·часов только на эту аблацию; вместо этого используется целевой prompt-injection (Phase 0) и проверка 4 диапазонов (D-A) | Глава 4, "Scope" |
| User study (34 участника, 1020 ответов) | Замена на комбинированную DINO + CLIP + FID оценку; для ВКР количественные метрики признаются достаточными | Глава 4, "Протокол оценки" |
| B-LoRA for Personalization (App. D; multiple content images) | За рамками задачи single-image style transfer; требует изменения постановки | Глава 4, "Scope" |
| Baselines ZipLoRA / StyleDrop / StyleAligned / InstantStyle | Нет стабильных реализаций для FLUX.1; сравнение с ближайшим прямым конкурентом (SplitFlux) покрывает научную новизну | Глава 4, "Сравниваемые методы" |
| Style-content mixing (Fig. 1, 27, 28) | Зависит от Phase 1b: включается только при подтверждении Θ_content | Условно, Phase 6 |

---

## Технический долг / Issue tracker

| # | Проблема | Критичность | Статус |
|---|----------|-------------|--------|
| 1 | Добавить DINO ViT-B/8 в `compute_metrics.py` | 🔴 критично | [ ] |
| 2 | Проверить training prompt в caption файлах | 🟡 важно | [x] fixed: "a sks painting" → "a sks" (все 8 файлов) |
| 3 | Новые конфиги D-A/B/C нужны после GATE 1 | 🟡 важно | [ ] |
| 4 | Скрипт `scripts/analysis/block_analysis.py` для Phase 0 | 🔴 критично | [ ] |
| 5 | Конфиги E-серии нужно обновить под финальные гиперпараметры | 🟡 важно | [ ] |
| 6 | Конфиги SplitFlux baseline (E03-\*) | 🔴 критично | [ ] |
| 7 | Конфиги Phase 1b (dc_content_\*.yaml) | 🟠 высокий | [ ] |
| 8 | Конфиги Phase 2b (dp\*.yaml — prompt ablation) | 🟠 высокий | [ ] |
| 9 | Скрипт анализа limitations (L01–L03) | 🟠 высокий | [ ] |
| 10 | Конфиги IP-Adapter-FLUX (E04-\*) | 🟡 средний | [ ] |
| 11 | Обновить главы 3 и 4 диплома под расширенный план (включить Phase 0 как обязательную, Phase 1b, SplitFlux, Limitations, Scope) | 🟡 важно | [ ] |

---

## Прогресс

- **Phase 0** (Prompt-injection block analysis): 0/3 — P0-DS, P0-SS, P0-R
- **Phase 1** (diag_d): 0/4 — e00, d01, d02, d03
- **Phase 1b** (Θ_content diagnostic): 0/3 — dc01, dc02, dc03
- **Phase 2** (D-A/B/C ablations): 0/12 — pending (конфиги после GATE 1)
- **Phase 2b** (D-P prompt ablation): 0/4 — pending
- **Phase 3** (Alpha): 0/6 (Phase 3a) + 0/4 (Phase 3b, условно) = 0/10
- **Phase 4** (Group E comparison): 0/24 — 8 B-LoRA + 8 Full-LoRA + 8 SplitFlux, +2 IP-Adapter (опц.)
- **Phase 4b** (Limitations): 0/3 — L01, L02, L03
- **Phase 5** (Group F cross-arch): 0/4 — F01-1..4
- **Phase 6** (Mixing demo): 0/2 — M01, M02 (условно)
- **Итого:** 0 / 69 (был 0/43; +26 экспериментов после ревизии)
