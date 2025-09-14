# Хакатон  Data Fusion Contest 2025 - предсказание категории товаров по их названию и описанию

## Задача соревнования
Необходимо разработать алгоритм для автоматической категоризации товаров по их названию и атрибутам, даже в условиях неполной разметки.
Система категорий устроена в виде иерархического дерева (до 5 уровней вложенности), а данные о товарах поступают с множества торговых площадок.

Ссылка на соревнование: https://ods.ai/competitions/data-fusion2025-labelcraft

## Структура проекта
- Labelling_LLM.ipynb - jupyter-ноутбук для доразметки данных с помощью LLM

- Training.ipynb - jupyter-нотубук для обучения модели

- pipeline.py - основной скрипт для подготовки данных.

- data/raw - папка с исходными данными

- data/processed - папка с обработанными данными

  ## Навыки и технологии
#### Теги: машинное обучение, RecSys
![Polars](https://img.shields.io/badge/Polars-black?style=flat&logo=polars&logoColor=orange)
![implicit](https://img.shields.io/badge/implicit-black?style=flat&logoColor=orange)
![SciPy](https://img.shields.io/badge/SciPy-black?style=flat&logo=scipy&logoColor=orange)
![NumPy](https://img.shields.io/badge/NumPy-black?style=flat&logo=numpy&logoColor=orange)
