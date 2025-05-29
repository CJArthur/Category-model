import pandas as pd
import pickle
import argparse
import csv
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default = "app/input.csv", help='Path to the test data file')
    parser.add_argument('--output-path', type=str, default = "app/output.csv", help='Path to the output file')
    args = parser.parse_args()

    # Загрузка тестовых данных
    test_data = pd.read_csv(args.input_path, header=None, encoding="cp1251")
    X_test = test_data.iloc[:, 0].tolist()

    # Загрузка TF-IDF векторизатора
    with open("app/models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    X_test_tfidf = vectorizer.transform(X_test)

    # Загрузка модели
    with open("app/models/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Загрузка LabelEncoder
    with open("app/models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # Предсказание и декодирование меток
    y_pred = model.predict(X_test_tfidf)
    predicted_labels = le.inverse_transform(y_pred)

    # Загрузим предсказания
    with open(args.output_path, "w", newline="", encoding="cp1251") as csvfile:
        writer = csv.writer(csvfile, delimiter = ';')
        writer.writerow(["Товар", "Категория"])  # Заголовки столбцов

        for item, label in zip(X_test, predicted_labels):
            writer.writerow([" ".join(str(item).split()).capitalize(), label])

    # Построим график распределения
    plt.figure(figsize=(7, 5))
    sns.countplot(x = predicted_labels, order=pd.Series(predicted_labels).value_counts().index)
    plt.xticks(rotation=0, ha='center')
    plt.title("Количество товаров в категориях")
    plt.tight_layout()
    plt.show()

    print("Загружен input файл:", args.input_path)
    print("Пример данных:", X_test[:5])