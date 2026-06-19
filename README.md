# TDA 3D MNIST Classification

<!-- Meta & Language -->
[![License](https://img.shields.io/github/license/yrmint/ml-app-arch)](#)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](#)

<!-- Machine Learning & Data Science -->
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](#)
[![Optuna](https://img.shields.io/badge/Optuna-002C76?logo=optuna&logoColor=white)](#)

<!-- Infrastructure -->
[![uv](https://img.shields.io/badge/uv-Lightning_Fast-261230?logo=python&logoColor=white)](#)


**Топологический анализ данных и классификация 3D MNIST** с использованием алгоритмов машинного обучения

## О проекте

Проект посвящен предварительной обработке трехмерного датасета [3D MNIST](https://www.kaggle.com/datasets/daavoo/3d-mnist/data), представленного в виде облаков точек (point clouds) и воксельных сеток на их основе.

Основным документом проекта, объединяющим инициализацию, фильтрацию и подготовку данных, является `cubical_complex_pipeline.ipynb`.

## Структура проекта

```plaintext
├── data/
│   └── MNIST3d/
│       ├── train_point_clouds.h5  		# Тренировочный датасет
│       ├── test_point_clouds.h5   		# Тестовый датасет
│       └── voxelgrid.py           		# Модуль с классом VoxelGrid для вокселизации
├── feature-extraction/
	├──alpha_complex.ipynb				# Первый эксперимент с альфа-комлпексом и векторизацией через ландшафты персистентности
	└──cubical_complex_pipeline.ipynb	# Второй эксперимент с кубическим комплексом и множественными способами векторизации
├── ml/
	├── models/							# Сохраненные классификаторы
	└── optuna_search.py				# Оптимизация параметров классификатора
├── utils/                         		# Вспомогательные функции
├── main.ipynb                     		# Основной ноутбук проекта (точка входа)
└── README.md                      		# Документация проекта
```

## Результаты

- Разработаны независимые пайплайны на основе альфа- и кубического комплексов для классификации трехмерных цифр. В качестве классификатора используется `XGBClassifier`.

- Альфа-комплекс с ландшафтами персистентности позволил успешно классифицировать только цифры `0,1,8,9` из-за выраженных различий в их топологической структуре, которые сохранялись при построении персистентных гомологий.

- Для кубического комплекса с различными векторизациями и фильтрами достигнут `F1-score = 0.81` на тестовой выборке (прирост 10% относительно baseline [CNN](https://www.kaggle.com/code/mattop/3d-mnist-digits-0-9-tensorflow-cnn) с `F1 = 0.73`). Это подтверждает эффективность методов TDA в извлечении дискриминативных признаков из трехмерных изображений


## Технологический Стэк

- **Язык:** `Python 3.12`
- **Пакетный менеджер:** `uv`
- **ML** `Scikit-learn`, `XGBoost`, `Optuna-dashboard`
- **TDA:** `gudhi`,`giotto`
- **Utils:** `h5py`, `matplotlib`

## Установка и запуск

### Предварительные требования
- [Python](https://www.python.org/downloads/)
- Пакетный менеджер [uv](https://github.com/astral-sh/uv)


### 1. Клонирование репозитория

```commandline
git clone https://github.com/KnightKotOR/topology-mnist3d.git
```

### 2. Установка зависимостей и настройка venv

```commandline
uv python install 3.12
```

```commandline
uv sync --all-groups
```