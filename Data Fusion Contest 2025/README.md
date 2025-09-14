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
![src](https://img.shields.io/badge/src-black?style=flat&logoColor=orange)
![transformers](https://img.shields.io/badge/transformers-black?style=flat&logo=transformers&logoColor=orange)
![a](https://img.shields.io/badge/torch-black?style=flat&logo=pytorch&logoColor=orange)
![Pandas](https://img.shields.io/badge/pandas-black?style=flat&logo=pandas&logoColor=orange)
![NumPy](https://img.shields.io/badge/numpy-black?style=flat&logo=numpy&logoColor=orange)
![scikit-learn](https://img.shields.io/badge/sklearn-black?style=flat&logo=scikitlearn&logoColor=orange)
