# Загружаем необходимые библиотеки
import pandas as pd
from src.category_tree.category_tree import CategoryTree
from sklearn.model_selection import GroupKFold, train_test_split

#########################
# Зададим константы для сплитов
RANDOM_STATE = 42
TEST_PART_SIZE = 0.1  # доля теста/валидации при разбиениях

# Загружаем структуру дерева категорий
category_tree = CategoryTree(category_tree_path='data/raw/category_tree.csv')

########################

# ------------------ Шаг 1: Разделение размеченных данных ------------------
# Цель: из размеченных данных выделить корректные (leaf) метки, некорректные (non-leaf),
# Удалить редкие категории и создать train/val сплиты с учётом группировки по категории.

# Загружаем только нужные колонки из размеченного набора
labeled_train = pd.read_parquet('data/raw/labeled_train.parquet', columns=['source_name', 'cat_id', 'hash_id', 'attributes'])

# bad_labeled: те записи, у которых категория НЕ является листовой (в дереве категорий)
bad_labeled = labeled_train[~labeled_train['cat_id'].isin(category_tree.leaf_nodes)]

# good_labeled: записи с листовыми категориями — потенциально пригодные для обучения
good_labeled = labeled_train[labeled_train['cat_id'].isin(category_tree.leaf_nodes)]

# Сохраняем плохие метки в отдельный файл, чтобы не терять информацию и освободить память
bad_labeled.to_parquet('data/processed/bad_labeled.parquet', index=False)
del bad_labeled

# Исключаем категории, в которых только 1 пример — такие классы малоинформативны для обучения
cat_id_samples_cnt = good_labeled['cat_id'].value_counts()
one_sample_cats = cat_id_samples_cnt[cat_id_samples_cnt == 1].index.values
good_labeled = good_labeled[~good_labeled['cat_id'].isin(one_sample_cats)]

# Выполняем групповую перекрёстную проверку (GroupKFold), чтобы разделить по категориям,
# и при этом избежать утечки категорий между train и oos (out-of-sample).
# n_splits = int(1 / TEST_PART_SIZE) — например, при TEST_PART_SIZE=0.1 получится 10 фолдов.
train_idx, test_idx = next(
    GroupKFold(n_splits=int(1 / TEST_PART_SIZE), shuffle=True, random_state=RANDOM_STATE).split(
        good_labeled, groups=good_labeled['cat_id']
    )
)

# Получаем выборки: df_train (для дальнейшего разделения) и df_oos (hold-out out-of-sample)
df_train, df_oos = good_labeled.iloc[train_idx], good_labeled.iloc[test_idx]

# Дополнительно разбиваем df_train на тренировочную часть и in-sample валидацию (is).
# Используем стратификацию по категории, чтобы сохранить распределение классов.
df_train, df_is = train_test_split(
    df_train, test_size=TEST_PART_SIZE, stratify=df_train['cat_id'], random_state=RANDOM_STATE
)

# Помечаем типы частей — полезно для downstream (например, различать is и oos в валидации)
df_is['part_type'] = "is"
df_oos['part_type'] = "oos"

# Объединяем is и oos в один валидационный набор (df_val).
df_val = pd.concat([df_is, df_oos], axis=0)
val_hash_ids = df_val['hash_id'].values.tolist()  # список hash_id, попавших в валидацию

# Подготавливаем и сохраняем файл валидации. Колонки: hash_id, source_name, cat_id, part, part_type
df_val['part'] = "val"
df_val = df_val[['hash_id', 'source_name', 'cat_id', 'part', 'part_type']]
df_val.to_parquet('data/processed/val.parquet', index=False)

# Освобождаем память — удаляем большие объекты, которые больше не нужны
del labeled_train, df_train, one_sample_cats, df_val, df_is, df_oos, train_idx, test_idx, cat_id_samples_cnt

# ------------------ Шаг 2: Обработка unlabeled_special_prize ------------------
# Цель: из специального набора выделить только те примеры, которые имеют листовую категорию,
# чтобы потом их можно было использовать как "хорошие" псевдоразмеченные данные.

unlabeled_special_prize = pd.read_parquet('data/raw/unlabeled_special_prize.parquet', columns=['source_name', 'cat_id', 'hash_id', 'attributes'])

# Здесь bad_prize не создаём (закомментировано) — оставляем только хорошие (leaf) примеры
good_prize = unlabeled_special_prize[unlabeled_special_prize['cat_id'].isin(category_tree.leaf_nodes)]  # Leaf category samples

# Удаляем исходный объект, чтобы освободить память
del unlabeled_special_prize

# ------------------ Шаг 3: Удаляем из unlabeled_train те элементы, что уже есть в good_prize ------------------
# Цель: избежать дублей — если в unlabeled_train есть те же hash_id, что в good_prize, их удаляем.

unlabeled_train = pd.read_parquet('data/raw/unlabeled_train.parquet', columns=['source_name', 'hash_id', 'attributes'])

# Фильтруем по hash_id: оставляем только те, которые НЕ присутствуют в good_prize
unlabeled_train = unlabeled_train[~unlabeled_train['hash_id'].isin(good_prize['hash_id'])]

# ------------------ Шаг 4: Объединяем размеченные и "special prize" разметки ------------------
# good_labeled теперь дополняем примерами из good_prize (т.е. дополнительные корректные метки)
good_labeled = pd.concat([good_labeled, good_prize])

del good_prize  # освободили память

# ------------------ Шаг 5: Псевдоразметка unlabeled через совпадение attributes ------------------
# Идея: сопоставить неразмеченные записи с размеченными по полю attributes;
# если атрибуты совпадают — использовать наиболее частую категорию для этого набора атрибутов/названий.

# Копируем good_labeled и нормализуем названия (lowercase) — удобнее для группировок по title
good_labeled_temp = good_labeled.copy()
good_labeled_temp['source_name'] = good_labeled_temp['source_name'].str.lower()

# Для каждой уникальной title в good_labeled выбираем наиболее частую категорию (most common cat_id).
# Это нужно, чтобы у одного title (если он встречается в разных записях) был единый доминирующий cat_id.
most_common_cat_id = good_labeled_temp.groupby('source_name')['cat_id'].agg(
    lambda x: x.value_counts().index[0] if len(x) > 0 else None
).reset_index()
most_common_cat_id.rename(columns={'cat_id': 'most_common_cat_id'}, inplace=True)

# Мержим back — теперь в каждой строке good_labeled_temp для title будет назначена самая частая категория
good_labeled_temp = good_labeled_temp.merge(most_common_cat_id, on='source_name', how='left')
good_labeled_temp['cat_id'] = good_labeled_temp['most_common_cat_id']
good_labeled_temp.drop(columns=['most_common_cat_id'], inplace=True)

# Начинаем формировать unlabeled_join — копия unlabeled_train для дальнейших операций
unlabeled_join = unlabeled_train.copy()
del unlabeled_train  # освободили память

# Нормализуем attributes в lowercase, чтобы сравнение было корректным
good_labeled_temp['attributes'] = good_labeled_temp['attributes'].str.lower()
unlabeled_join['attributes'] = unlabeled_join['attributes'].str.lower()

# Удаляем дубликаты по title — мы хотим одну репрезентативную строку на title
good_labeled_temp = good_labeled_temp.drop_duplicates(['source_name'])
unlabeled_join = unlabeled_join.drop_duplicates(['source_name'])

# Убираем строки с "пустыми" attributes (строка "[{}]") — они неинформативны для сопоставления
good_labeled_temp = good_labeled_temp[good_labeled_temp['attributes'] != "[{}]"]
unlabeled_join = unlabeled_join[unlabeled_join['attributes'] != "[{}]"]

# Выполняем join по колонке attributes: если для атрибутов найдено соответствие — берём cat_id из размеченных
unlabeled_join = unlabeled_join.merge(
    good_labeled_temp[['attributes', 'cat_id']], 
    on='attributes', 
    how='left'
).dropna(subset=['cat_id'])  # оставляем только те, где нашли категорию

del good_labeled_temp  # free memory

# В некоторых случаях одному title может соответствовать несколько категорий (из-за разных attribute-match)
# Нормализуем title в lowercase и выбираем наиболее частую (mode) категорию для каждой title
unlabeled_join['source_name'] = unlabeled_join['source_name'].str.lower()
most_common_cat_id = unlabeled_join.groupby('source_name')['cat_id'].agg(lambda x: x.mode().iloc[0]).reset_index()
most_common_cat_id.rename(columns={'cat_id': 'most_common_cat_id'}, inplace=True)

# Присваиваем финальную категорию (most common per title) и убираем временную колонку
unlabeled_join = unlabeled_join.merge(most_common_cat_id, on='source_name', how='left')
unlabeled_join['cat_id'] = unlabeled_join['most_common_cat_id']
unlabeled_join.drop(columns=['most_common_cat_id'], inplace=True)
unlabeled_join = unlabeled_join.drop_duplicates(['source_name'])  # оставляем по одному title

# После успешной псевдоразметки attributes колонка больше не нужна — удаляем
unlabeled_join.drop(columns=['attributes'], inplace=True)
good_labeled.drop(columns=['attributes'], inplace=True)  # и в исходном good_labeled attributes тоже удаляем (уже не нужны)

del most_common_cat_id  # очистка временной переменной

# ------------------ Шаг 6: Подгружаем дополнительные наборы, сгенерированные/доработанные LLM ------------------
# unlabeled_dorazmetka — часть неразмеченных данных, доразмеченных LLM (например, категория 460)
unlabeled_dorazmetka = pd.read_parquet('data/raw/unlabeled_dorazmetka.parquet', columns=['source_name', 'hash_id', 'cat_id'])

# new_items — полностью сгенерированные LLM новые элементы; переименуем колонку item_name -> source_name
new_items = pd.read_parquet('data/raw/new_items.parquet').rename(columns={'item_name': 'source_name'})

# ------------------ Шаг 7: Формирование финального тренировочного набора ------------------
# Исключаем из good_labeled те элементы, которые попали в валидацию (val_hash_ids)
good_labeled = good_labeled[~good_labeled['hash_id'].isin(val_hash_ids)]

# Объединяем все источники, чтобы получить максимально объёмный и разнообразный train набор:
# - good_labeled: размеченные (и некоторые из special prize)
# - unlabeled_join: псевдоразмеченные по attributes
# - unlabeled_dorazmetka: доразмеченные LLM
# - new_items: полностью сгенерированные items LLM
train = pd.concat([good_labeled, unlabeled_join, unlabeled_dorazmetka, new_items], ignore_index=True, copy=False)

# Отмечаем партию/тип для этих строк — пригодится для отслеживания происхождения в анализе
train['part_type'] = "is"
train['part'] = "train"

# Удаляем дубликаты по hash_id — сохраняем по одному экземпляру каждого уникального хэша
train.drop_duplicates(subset=['hash_id'], inplace=True)

# Оставляем только нужные колонки для обучения и сбрасываем индекс
train = train[['source_name', 'cat_id', 'part', 'part_type']]

# Перемешиваем данные (shuffle) для улучшения обобщающей способности модели при обучении
train = train.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Сохраняем финальный тренировочный датасет в parquet
train.to_parquet('data/processed/train.parquet')