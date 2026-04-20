# Литературный обзор
## «Применение B-LoRA для улучшения качества переноса стиля в DiT-моделях — FLUX»

---

## Часть I. Обзор статей по темам

### Тема 1: Основы диффузионных моделей

| Статья | Год | Ключевой вклад |
|---|---|---|
| Ho et al. (DDPM) | 2020 | Марковская цепь зашумления/денойзинга; вариационная нижняя граница |
| Song et al. (DDIM) | 2021 | Детерминированный ускоренный сэмплинг (10–50× быстрее DDPM) |
| Rombach et al. (LDM/SD) | 2022 | Диффузия в латентном пространстве; cross-attention кондиционирование |
| Gatys et al. (NST) | 2016 | Формальное определение стиля через матрицы Грама; первый нейросетевой перенос стиля |

### Тема 2: Диффузионные трансформеры и масштабирование

| Статья | Год | Ключевой вклад |
|---|---|---|
| Vaswani et al. | 2017 | Архитектура Transformer; multi-head self-attention |
| Peebles & Xie (DiT) | 2023 | Transformer-бэкбон вместо U-Net; предсказуемое масштабирование |
| Esser et al. (SD3/MM-DiT) | 2024 | Мультимодальный DiT; двухпоточная архитектура; rectified flow |
| Black Forest Labs (FLUX.1) | 2024 | 12B MM-DiT + параллельные attention-блоки; SOTA T2I |

### Тема 3: Rectified Flow и Flow Matching

| Статья | Год | Ключевой вклад |
|---|---|---|
| Lipman et al. (Flow Matching) | 2022/23 | Обучение CNF без симуляции; OT-пути |
| Liu et al. (Rectified Flow) | 2022/23 | Прямолинейный ODE-транспорт между распределениями |

### Тема 4: LoRA и PEFT

| Статья | Год | Ключевой вклад |
|---|---|---|
| Hu et al. (LoRA) | 2022 | ΔW = AB, r ≪ d; сокращение параметров в 10 000× |
| Liu et al. (DoRA) | 2024 | Декомпозиция на величину и направление; улучшенная LoRA |
| Yeh et al. (LyCORIS) | 2024 | Таксономия и библиотека вариантов LoRA для SD |
| **Frenkel et al. (B-LoRA)** | **2024** | **Поблочная LoRA для разделения стиля/содержания** |

### Тема 5: Персонализация текст-изображение

| Статья | Год | Ключевой вклад |
|---|---|---|
| Gal et al. (Textual Inversion) | 2023 | Оптимизация токен-эмбеддинга; без изменения весов |
| Ruiz et al. (DreamBooth) | 2023 | Полное дообучение + prior preservation loss |
| Ye et al. (IP-Adapter) | 2023 | Разделённый cross-attention адаптер для image-prompt |
| Wang et al. (InstantStyle) | 2024 | Инъекция стиля в конкретные блоки; без обучения |

### Тема 6: Перенос стиля в диффузионных моделях

| Статья | Год | Ключевой вклад |
|---|---|---|
| Hertz et al. (StyleAligned) | 2024 | Разделённые attention-статистики для согласованного стиля |
| Qi et al. (DEADiff) | 2024 | Q-Former + нереконструктивное обучение; диссоциация стиль/контент |
| Huang et al. (IC-LoRA) | 2024 | In-context LoRA на FLUX для задач стиля/идентичности |
| Dalva et al. (FluxSpace) | 2025 | Семантическое редактирование через attention-пространства FLUX |

### Тема 7: Диссоциация стиля/содержания в FLUX (новые работы)

| Статья | Год | Ключевой вклад |
|---|---|---|
| **Frenkel et al. (B-LoRA)** | **2024** | **Поблочная LoRA в SDXL (исток метода)** |
| Yang et al. (SplitFlux) | 2025 | Идеи B-LoRA + RCA + VGRA на single-stream блоках FLUX |
| Dalva et al. (FluxSpace) | 2025 | Без дообучения; семантические пространства FLUX |

---

## Часть II. Сравнительная таблица методов переноса стиля

| Метод | Базовая модель | Подход | Требует обучения | Вход стиля | Сохранение содержания |
|---|---|---|---|---|---|
| Gatys NST (2016) | VGG | Итеративная оптимизация (Грам) | Нет (on-image) | Изображение | Явный content loss |
| Textual Inversion (2023) | SD 1.x | Оптимизация токен-эмбеддинга | Да (per-concept) | 3–5 изображений | Неявное через промпт |
| DreamBooth (2023) | SD 1.x/2.x | Полное дообучение + prior loss | Да (per-subject) | 3–5 изображений | Prior preservation |
| IP-Adapter (2023) | SDXL | Разделённый cross-attention | Да (общее) | Изображение | Сильное |
| StyleAligned (2024) | SDXL | Общие attention-статистики + AdaIN | Нет | Изображение | Умеренное |
| InstantStyle (2024) | SDXL | Инъекция в блоки стиля | Нет | Изображение | Сильное |
| DEADiff (2024) | SDXL | Q-Former + нереконструктивное | Да (общее) | Изображение | Сильное |
| **B-LoRA (2024)** | **SDXL** | **Поблочная LoRA, 2 блока, 1 фото** | **Да (~5 мин/стиль)** | **1 изображение** | **Сильное** |
| IC-LoRA (2024) | FLUX.1-dev | In-context LoRA | Да (малый датасет) | Набор изображений | Задачезависимо |
| FluxSpace (2025) | FLUX.1-dev | Attention-пространство FLUX | Нет | Изображение | Умеренное |
| SplitFlux (2025) | FLUX.1-dev | Поблочная LoRA + RCA + VGRA | Да (~5 мин/стиль) | 1 изображение | Сильное (RCA) |
| **B-LoRA-FLUX (предлагаемый)** | **FLUX.1-dev** | **Поблочная LoRA на MM-DiT** | **Да (~5 мин/стиль)** | **1 изображение** | **Оценивается** |

---

## Часть III. Анализ пробелов

### Пробел 1 — B-LoRA никогда не применялся к FLUX / MM-DiT
B-LoRA разработан и валидирован исключительно для SDXL с U-Net бэкбоном. Архитектура FLUX принципиально иная: двухпоточные (MM) и однопоточные трансформерные блоки, параллельные attention-слои, rectified flow. Отображение блоков из SDXL (блоки 4 и 5) не переносится на 57-блочную однопоточную архитектуру FLUX напрямую.

### Пробел 2 — SplitFlux требует дополнительных компонентов
SplitFlux (ноябрь 2025) расширяет идеи B-LoRA на FLUX, но вводит RCA и VGRA, отступая от минималистичной идеи B-LoRA. Нет работы, оценивающей, достаточно ли исходной методологии B-LoRA — без дополнительных компонентов — при правильном выборе блоков FLUX.

### Пробел 3 — Отсутствие стандартизированной оценки переноса стиля для FLUX
Метрики FID, DINO, CLIP-Text разрабатывались для U-Net моделей и не учитывают специфику стиля (мазки кисти, цветовая палитра, художественная техника) в высококачественных генерациях FLUX масштаба 12B параметров.

### Пробел 4 — Нет систематического анализа функциональной специализации блоков FLUX
Ни FluxSpace, ни SplitFlux не проводят интерпретируемого анализа того, какие именно архитектурные компоненты MM-DiT кодируют стиль vs содержание. Это понимание необходимо для принципиального выбора блоков в B-LoRA.

---

## Часть IV. Научная новизна (черновик)

Настоящая работа впервые осуществляет систематическую адаптацию метода B-LoRA — ранее разработанного для архитектуры SDXL на основе U-Net — к диффузионным трансформерным моделям на основе выпрямленных потоков, в частности к модели FLUX.1. В отличие от существующих подходов, предлагается методология выбора блоков MM-DiT архитектуры FLUX, ответственных за кодирование стиля и содержания, на основе анализа функциональной специализации отдельных трансформерных слоёв, что позволяет достичь неявного разделения стиля и содержания при дообучении на одном изображении без дополнительных архитектурных компонентов. Разработанный подход обеспечивает более высокое качество переноса художественного стиля по сравнению с существующими адаптерными методами (IP-Adapter, InstantStyle) и не требует обучающих наборов данных, что существенно снижает вычислительные затраты. Кроме того, предложена методика оценки качества переноса стиля, адаптированная к особенностям генерации в моделях масштаба FLUX, обеспечивающая объективное сравнение с актуальными методами и воспроизводимость результатов.

---

## Часть V. BibTeX — все статьи

```bibtex
@inproceedings{ho2020ddpm,
  title     = {Denoising Diffusion Probabilistic Models},
  author    = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  pages     = {6840--6851},
  year      = {2020}
}

@inproceedings{song2021ddim,
  title     = {Denoising Diffusion Implicit Models},
  author    = {Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  booktitle = {International Conference on Learning Representations},
  year      = {2021}
}

@inproceedings{rombach2022ldm,
  title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
  author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {10684--10695},
  year      = {2022}
}

@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob
               and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017}
}

@inproceedings{peebles2023dit,
  title     = {Scalable Diffusion Models with Transformers},
  author    = {Peebles, William and Xie, Saining},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages     = {4195--4205},
  year      = {2023}
}

@inproceedings{lipman2023flow,
  title     = {Flow Matching for Generative Modeling},
  author    = {Lipman, Yaron and Chen, Ricky T. Q. and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matt},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}

@inproceedings{liu2022rectifiedflow,
  title     = {Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow},
  author    = {Liu, Xingchao and Gong, Chengyue and Liu, Qiang},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}

@inproceedings{esser2024sd3,
  title     = {Scaling Rectified Flow Transformers for High-Resolution Image Synthesis},
  author    = {Esser, Patrick and Kulal, Sumith and Blattmann, Andreas and others},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {PMLR},
  volume    = {235},
  year      = {2024}
}

@misc{blackforest2024flux,
  title        = {{FLUX}.1},
  author       = {{Black Forest Labs}},
  year         = {2024},
  howpublished = {\url{https://github.com/black-forest-labs/flux}}
}

@inproceedings{hu2022lora,
  title     = {{LoRA}: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan
               and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle = {International Conference on Learning Representations},
  year      = {2022}
}

@inproceedings{liu2024dora,
  title     = {{DoRA}: Weight-Decomposed Low-Rank Adaptation},
  author    = {Liu, Shih-Yang and Wang, Chien-Yi and Yin, Hongxu and Molchanov, Pavlo
               and Wang, Yu-Chiang Frank and Cheng, Kwang-Ting and Chen, Min-Hung},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages     = {32100--32121},
  year      = {2024},
  volume    = {235},
  series    = {Proceedings of Machine Learning Research}
}

@inproceedings{yeh2024lycoris,
  title     = {Navigating Text-To-Image Customization: From {LyCORIS} Fine-Tuning to Model Evaluation},
  author    = {Yeh, Shih-Ying and Hsieh, Yu-Guan and Gao, Zhidong and Yang, Bernard B. W.
               and Oh, Giyeong and Gong, Yanmin},
  booktitle = {International Conference on Learning Representations},
  year      = {2024}
}

@inproceedings{frenkel2024blora,
  title     = {Implicit Style-Content Separation using {B-LoRA}},
  author    = {Frenkel, Yarden and Vinker, Yael and Shamir, Ariel and Cohen-Or, Daniel},
  booktitle = {Computer Vision -- ECCV 2024},
  series    = {Lecture Notes in Computer Science},
  volume    = {15079},
  pages     = {185--202},
  publisher = {Springer},
  year      = {2024}
}

@inproceedings{ruiz2023dreambooth,
  title     = {{DreamBooth}: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation},
  author    = {Ruiz, Nataniel and Li, Yuanzhen and Jampani, Varun and Pritch, Yael
               and Rubinstein, Michael and Aberman, Kfir},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {22500--22510},
  year      = {2023}
}

@inproceedings{gal2023textualinversion,
  title     = {An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion},
  author    = {Gal, Rinon and Alaluf, Yuval and Atzmon, Yuval and Patashnik, Or
               and Bermano, Amit H. and Chechik, Gal and Cohen-Or, Daniel},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}

@article{ye2023ipadapter,
  title   = {{IP-Adapter}: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models},
  author  = {Ye, Hu and Zhang, Jun and Liu, Sibo and Han, Xiao and Yang, Wei},
  journal = {arXiv preprint arXiv:2308.06721},
  year    = {2023}
}

@article{wang2024instantstyle,
  title   = {{InstantStyle}: Free Lunch towards Style-Preserving in Text-to-Image Generation},
  author  = {Wang, Haofan and Wang, Qixun and Bai, Xu and Qin, Zekui and Chen, Anthony},
  journal = {arXiv preprint arXiv:2404.02733},
  year    = {2024}
}

@inproceedings{hertz2024stylealigned,
  title     = {Style Aligned Image Generation via Shared Attention},
  author    = {Hertz, Amir and Voynov, Andrey and Fruchter, Shlomi and Cohen-Or, Daniel},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {4775--4785},
  year      = {2024}
}

@inproceedings{gatys2016nst,
  title     = {Image Style Transfer Using Convolutional Neural Networks},
  author    = {Gatys, Leon A. and Ecker, Alexander S. and Bethge, Matthias},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages     = {2414--2423},
  year      = {2016}
}

@inproceedings{radford2021clip,
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  author    = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle = {International Conference on Machine Learning},
  pages     = {8748--8763},
  volume    = {139},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  year      = {2021}
}

@inproceedings{zhang2023controlnet,
  title     = {Adding Conditional Control to Text-to-Image Diffusion Models},
  author    = {Zhang, Lvmin and Rao, Anyi and Agrawala, Maneesh},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2023}
}

@inproceedings{qi2024deadiff,
  title     = {{DEADiff}: An Efficient Stylization Diffusion Model with Disentangled Representations},
  author    = {Qi, Tianhao and Fang, Shancheng and Wu, Yanze and Xie, Hongtao
               and Liu, Jiawei and Chen, Lang and He, Qian and Zhang, Yongdong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {8693--8702},
  year      = {2024}
}

@article{huang2024iclora,
  title   = {In-Context {LoRA} for Diffusion Transformers},
  author  = {Huang, Lianghua and Wang, Wei and Wu, Zhi-Fan and others},
  journal = {arXiv preprint arXiv:2410.23775},
  year    = {2024}
}

@inproceedings{dalva2024fluxspace,
  title     = {{FluxSpace}: Disentangled Semantic Editing in Rectified Flow Transformers},
  author    = {Dalva, Yusuf and Venkatesh, Kavana and Yanardag, Pinar},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2025}
}

@article{yang2025splitflux,
  title   = {{SplitFlux}: Learning to Decouple Content and Style from a Single Image},
  author  = {Yang, Yitong and others},
  journal = {arXiv preprint arXiv:2511.15258},
  year    = {2025}
}
```
