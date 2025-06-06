# Классификация товаров по категориям (нейросеть без фреймворка)

Данный проект представляет собой реализацию простой нейронной сети без использования фреймворков (таких как PyTorch или TensorFlow). Модель предназначена для классификации товаров по категориям на основе входного CSV-файла с названиями товаров.

# 📌 Описание

- 📥 Входной файл: `input.csv`, содержащий колонку с названиями товаров (`title`)
- 📤 Выходной файл: `output.csv` с колонками [`Товар`][`Категория`] 
- 🧠 Модель реализована вручную с использованием NumPy
- 📊 Предусмотрена загрузка данных, обучение и предсказание

# Установка через GitHub
1. git clone https://github.com/CJArthur/Category-model.git
2. cd Category-model
3. pip install -r requirements.txt