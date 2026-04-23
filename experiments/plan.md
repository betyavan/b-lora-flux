# Experiment Plan

Auto-generated structure, statuses synced by `scripts/update_exp_plan.py`. Last updated: 2026-04-23

Статусы: `[ ]` pending · `[~]` running · `[x]` done · `[!]` failed

---

## Соответствие оригинальной статье B-LoRA

> Оригинальная статья (SDXL, UNet): rank=64, lr=5e-5, Adam, 1000 steps, 1 image, prompt="A [v]",
> метрика — **DINO ViT-B/8 cosine similarity**, базовые методы — ZipLoRA / StyleAligned / DB-LoRA / StyleDrop.
> Блоки стиля/контента идентифицированы через prompt-injection + CLIP-анализ по всем Up-blocks.

**Расхождения, которые нужно устранить:**
1. ~~Аблации A/B/C использовали single-stream блоки (19–37) — неверно для FLUX~~ → фиксим через D-серию
2. Метрика DINO ViT-B/8 отсутствует → добавляем в `compute_metrics.py` (Issue #1)
3. Ранг r=64 не тестировался → добавляем B05 в аблацию B
4. LR=1e-4 выше бумажного 5e-5 → новые конфиги используют 5e-5
5. Alpha (lora_scale) не варьировался → добавляем аблацию Alpha (группа G)
6. Нет анализа блоков по методологии статьи → планируется как Phase 0

---

## Phase 0 — Анализ блоков FLUX (методология статьи, раздел 4.1)

> **Цель:** Воспроизвести процедуру идентификации стилевых/контентных блоков из статьи для FLUX.
> Для каждого из 19 double-stream блоков (0–18) инжектируем альтернативный промпт и измеряем CLIP-сходство.
> Промпты: P_content = "A photo of a {object}", P_style = "A photo of a {color} {object}".
> Выводим матрицу 19×2 (content response, style response) → определяем целевые блоки.

| ID   | Описание                                | Статус |
|------|-----------------------------------------|--------|
| P0   | FLUX block analysis via prompt injection | [ ]    |

> ⚠️ Требует отдельного скрипта `scripts/analysis/block_analysis.py` (пока не реализован).
> Приоритет: после GATE 1 — результаты D-серии укажут, нужен ли отдельный анализ.

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

**GATE 1** (после diag_d):
- Если D01/D02/D03 показывают DINO-style > E00 + визуальный перенос стиля → **гипотеза подтверждена**,
  переходим к аблациям D-new (правильные блоки, rank, steps).
- Если всё равно нет переноса → реализуем Phase 0 (block analysis) и пересматриваем блоки.

---

## Phase 2 — Аблации с правильными блоками (double-stream, после GATE 1)

> После GATE 1 аблации A/B/C переделываются с блоками double-stream 0–18.
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

## Phase 3 — Аблация Alpha / lora_scale (группа G)

> Из статьи: α=0.4–0.5 для стилевого адаптера лучше сохраняет оригинальные цвета объекта.
> Тестируем через `LORA_SCALE` env var при генерации (без переобучения — только инференс).

| ID    | LORA_SCALE | Базовый конфиг       | DINO-style | DINO-content | CLIP-style | Статус |
|-------|------------|----------------------|------------|--------------|------------|--------|
| G01   | 0.3        | лучший из D-A/B/C    | —          | —            | —          | [ ]    |
| G02   | 0.5        | лучший из D-A/B/C    | —          | —            | —          | [ ]    |
| G03   | 0.7        | лучший из D-A/B/C    | —          | —            | —          | [ ]    |
| G04   | 1.0        | лучший из D-A/B/C    | —          | —            | —          | [ ]    |
| G05   | 1.5        | лучший из D-A/B/C    | —          | —            | —          | [ ]    |
| G06   | 2.0        | лучший из D-A/B/C    | —          | —            | —          | [ ]    |

---

## Phase 4 — Финальное сравнение методов (группа E)

> Финальная конфигурация из D-A/B/C/G применяется ко всем 4 стилевым изображениям двух стилей.
> Сравнение: B-LoRA-FLUX vs Full-LoRA-FLUX (DB-LoRA equivalent).

| ID      | Конфиг                                  | Метод          | Стиль    | img | CLIP-style | DINO-style | FID | LPIPS | Статус |
|---------|-----------------------------------------|----------------|----------|-----|------------|------------|-----|-------|--------|
| E01-1   | e01_blora_flux_van_gogh_img1.yaml       | B-LoRA-FLUX    | Van Gogh | 1   | —          | —          | —   | —     | [ ]    |
| E01-2   | e01_blora_flux_van_gogh_img2.yaml       | B-LoRA-FLUX    | Van Gogh | 2   | —          | —          | —   | —     | [ ]    |
| E01-3   | e01_blora_flux_van_gogh_img3.yaml       | B-LoRA-FLUX    | Van Gogh | 3   | —          | —          | —   | —     | [ ]    |
| E01-4   | e01_blora_flux_van_gogh_img4.yaml       | B-LoRA-FLUX    | Van Gogh | 4   | —          | —          | —   | —     | [ ]    |
| E02-1   | e02_full_lora_flux_van_gogh_img1.yaml   | Full-LoRA-FLUX | Van Gogh | 1   | —          | —          | —   | —     | [ ]    |
| E02-2   | e02_full_lora_flux_van_gogh_img2.yaml   | Full-LoRA-FLUX | Van Gogh | 2   | —          | —          | —   | —     | [ ]    |
| E02-3   | e02_full_lora_flux_van_gogh_img3.yaml   | Full-LoRA-FLUX | Van Gogh | 3   | —          | —          | —   | —     | [ ]    |
| E02-4   | e02_full_lora_flux_van_gogh_img4.yaml   | Full-LoRA-FLUX | Van Gogh | 4   | —          | —          | —   | —     | [ ]    |
| E01M-1  | e01_blora_flux_monet_img1.yaml          | B-LoRA-FLUX    | Monet    | 1   | —          | —          | —   | —     | [ ]    |
| E01M-2  | e01_blora_flux_monet_img2.yaml          | B-LoRA-FLUX    | Monet    | 2   | —          | —          | —   | —     | [ ]    |
| E01M-3  | e01_blora_flux_monet_img3.yaml          | B-LoRA-FLUX    | Monet    | 3   | —          | —          | —   | —     | [ ]    |
| E01M-4  | e01_blora_flux_monet_img4.yaml          | B-LoRA-FLUX    | Monet    | 4   | —          | —          | —   | —     | [ ]    |
| E02M-1  | e02_full_lora_flux_monet_img1.yaml      | Full-LoRA-FLUX | Monet    | 1   | —          | —          | —   | —     | [ ]    |
| E02M-2  | e02_full_lora_flux_monet_img2.yaml      | Full-LoRA-FLUX | Monet    | 2   | —          | —          | —   | —     | [ ]    |
| E02M-3  | e02_full_lora_flux_monet_img3.yaml      | Full-LoRA-FLUX | Monet    | 3   | —          | —          | —   | —     | [ ]    |
| E02M-4  | e02_full_lora_flux_monet_img4.yaml      | Full-LoRA-FLUX | Monet    | 4   | —          | —          | —   | —     | [ ]    |

Среднее по 4 изображениям (Van Gogh):

| Метод          | CLIP-style ↑ | DINO-style ↑ | FID ↓ | LPIPS ↓ |
|----------------|-------------|-------------|-------|---------|
| No-LoRA (E00)  | —           | —           | —     | —       |
| B-LoRA-FLUX    | —           | —           | —     | —       |
| Full-LoRA-FLUX | —           | —           | —     | —       |

Среднее по 4 изображениям (Monet):

| Метод          | CLIP-style ↑ | DINO-style ↑ | FID ↓ | LPIPS ↓ |
|----------------|-------------|-------------|-------|---------|
| No-LoRA (E00)  | —           | —           | —     | —       |
| B-LoRA-FLUX    | —           | —           | —     | —       |
| Full-LoRA-FLUX | —           | —           | —     | —       |

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

## Технический долг / Issue tracker

| # | Проблема | Критичность | Статус |
|---|----------|-------------|--------|
| 1 | Добавить DINO ViT-B/8 в `compute_metrics.py` | 🔴 критично | [ ] |
| 2 | Проверить training prompt в caption файлах — должен быть "A [v]" | 🟡 важно | [ ] |
| 3 | Новые конфиги D-A/B/C нужны после GATE 1 | 🟡 важно | [ ] |
| 4 | Скрипт `scripts/analysis/block_analysis.py` для Phase 0 | 🟠 желательно | [ ] |
| 5 | Конфиги E-серии нужно обновить под финальные гиперпараметры | 🟡 важно | [ ] |

---

## Прогресс

- **Phase 0** (Block Analysis): 0/1 — [ ] pending
- **Phase 1** (diag_d): 0/4 — e00, d01, d02, d03
- **Phase 2** (D-A/B/C ablations): 0/12 — [ ] pending (конфиги после GATE 1)
- **Phase 3** (Alpha ablation): 0/6 — [ ] pending
- **Phase 4** (Group E comparison): 0/16 — [ ] pending
- **Phase 5** (Group F cross-arch): 0/4 — [ ] pending
- **Итого:** 0 / 43
