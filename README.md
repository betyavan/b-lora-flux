# B-LoRA для переноса стиля в DiT-моделях (FLUX.1)

Выпускная квалификационная работа. Адаптация метода B-LoRA на архитектуру FLUX.1 (MM-DiT) для разделения стиля и содержания при переносе стиля одним референсным изображением.

## Идея

[B-LoRA](papers/b-lora.pdf) показал, что в U-Net SDXL разные блоки специализируются на стиле и содержании. В этой работе мы проверяем аналогичную гипотезу для FLUX.1: поздние single-stream блоки (30–37) кодируют стиль, ранние double-stream блоки (0–5) — содержание. При инференсе подключается только `Θ_style`, что позволяет воспроизводить художественную манеру по текстовому промпту.

## Структура проекта

```
DIPLOMA/
├── configs/
│   ├── data/datasets.yaml          # пути к датасетам
│   ├── download.yaml               # параметры загрузки
│   ├── eval/
│   │   ├── generate.yaml           # инференс
│   │   └── metrics.yaml            # оценка качества
│   └── experiments/                # 32 конфига ai-toolkit
│       ├── base_flux_lora.yaml     # базовый шаблон
│       ├── e01_blora_flux_*.yaml   # группа E: B-LoRA-FLUX
│       ├── e02_full_lora_flux_*.yaml
│       ├── e04_blora_sdxl_*.yaml
│       ├── a0{1-4}_blocks_*.yaml   # аблация A: диапазон блоков
│       ├── b0{1-4}_rank_*.yaml     # аблация B: ранг r
│       └── c0{1-4}_steps_*.yaml    # аблация C: число шагов
├── dags/
│   ├── blora_flux_pipeline.py         # одиночный эксперимент
│   ├── blora_flux_group_pipeline.py   # параллельный запуск группы
│   ├── plugins/job_runner_wrapper.py  # обёртка оператора
│   └── jobs/                          # train / generate / metrics
│       ├── train_blora/
│       ├── generate_blora/
│       └── metrics_blora/
├── data/
│   ├── styles/                     # стилевые изображения (DVC)
│   │   ├── van_gogh/img{1-4}/
│   │   └── monet/img{1-4}/
│   ├── coco_val2017/               # промпты для оценки (DVC)
│   └── artbench10/                 # референс для FID (DVC)
├── experiments/
│   └── plan.md                     # прогресс 32 экспериментов
├── scripts/
│   ├── download_datasets.py        # загрузка COCO + ArtBench
│   ├── update_exp_plan.py          # синхронизация ClearML → plan.md
│   └── eval/
│       ├── generate_images.py      # генерация с LoRA-адаптером
│       └── compute_metrics.py      # CLIP-style / CLIP-content / FID / LPIPS
├── src/
│   └── ai-toolkit/                 # git submodule (ostris/ai-toolkit)
├── thesis/                         # LaTeX-исходники ВКР
├── infra.env.template
└── pyproject.toml
```

## Установка

```bash
# клонировать с субмодулем
git clone --recurse-submodules <repo>
cd DIPLOMA

# зависимости
pip install poetry && poetry install

# настроить корпоративные переменные
cp infra.env.template infra.env
# отредактировать infra.env, затем:
make configure    # рендерит preset YAML из шаблонов

# данные (требует dvc remote)
dvc pull
```

## Быстрый старт

### Запуск через Airflow (основной способ)

```bash
# Один эксперимент
make run EXP=e01_blora_flux_van_gogh_img1

# Группа экспериментов (параллельно)
make run-group GROUP=ablation_a

# Статус и результаты
make status
make pull-results EXP=e01_blora_flux_van_gogh_img1
```

### Локальный запуск (без Airflow)

Используется для разработки и отладки.

#### 1. Обучить LoRA-адаптер

```bash
cd src/ai-toolkit
python run.py ../../configs/experiments/e01_blora_flux_van_gogh_img1.yaml
```

Адаптер сохраняется в `output/e01_blora_flux_van_gogh_img1/`.

#### 2. Сгенерировать изображения

```bash
python scripts/eval/generate_images.py \
  generate.lora_path=output/e01_blora_flux_van_gogh_img1/e01_blora_flux_van_gogh_img1.safetensors \
  generate.exp_name=e01_blora_flux_van_gogh_img1
```

Изображения сохраняются в `output/generated/e01_blora_flux_van_gogh_img1/`.

#### 3. Вычислить метрики

```bash
python scripts/eval/compute_metrics.py \
  metrics.generated_dir=output/generated/e01_blora_flux_van_gogh_img1 \
  metrics.style_ref=data/styles/van_gogh/img1/reference.jpg \
  metrics.prompt_file=data/coco_prompts.txt \
  metrics.artbench_dir=data/artbench10/van_gogh \
  metrics.exp_name=e01_blora_flux_van_gogh_img1
```

Результаты печатаются в консоль и логируются в ClearML.

#### 4. Обновить таблицу прогресса

```bash
python scripts/update_exp_plan.py
```

Читает метрики из ClearML и обновляет `experiments/plan.md`.

## Эксперименты

### Аблационные исследования (Van Gogh, img1)

| Группа | Что варьируется | Конфиги |
|--------|-----------------|---------|
| A | диапазон стилевых блоков: [34–37], [30–37], [24–37], [19–37] | `a01–a04` |
| B | ранг LoRA: r ∈ {4, 8, 16, 32} | `b01–b04` |
| C | число шагов: {100, 250, 500, 1000} | `c01–c04` |

### Основное сравнение (группа E, Van Gogh + Monet × 4 изображения)

| Метод | Конфиг | Примечание |
|-------|--------|------------|
| **B-LoRA-FLUX** | `e01_blora_flux_*` | предлагаемый метод |
| Full-LoRA-FLUX | `e02_full_lora_flux_*` | baseline без блочного разделения |
| B-LoRA-SDXL | `e04_blora_sdxl_*` | оригинальный B-LoRA на SDXL |

### Метрики

| Метрика | Направление | Что измеряет |
|---------|-------------|--------------|
| CLIP-style ↑ | выше лучше | соответствие стилевому референсу |
| CLIP-content ↑ | выше лучше | сохранение семантики промпта |
| FID ↓ | ниже лучше | близость к стилевому распределению ArtBench-10 |
| LPIPS ↓ | ниже лучше | перцептивное сходство с референсом |

## Воспроизводимость

- Все генерации: `seed=42`, `steps=28`, `guidance_scale=3.5`
- Конфигурации обучения фиксированы в `configs/experiments/`
- Данные версионированы через DVC (`data/*.dvc`)
- Метрики логируются в ClearML, проект `blora-flux-eval`

## Стек

Python 3.10 · PyTorch 2.1 · Diffusers ≥0.30 · PEFT ≥0.12 · Hydra 1.3 · ClearML 1.16 · DVC 3.54 · ai-toolkit 0.9.4 · Airflow 2.x · boto3 · s3cmd · pre-commit
