import pandas as pd
import pickle
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


# Очистка текста: приведение к нижнему регистру, удаление цифр, пунктуации и лишних пробелов
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # удаляем цифры
    text = text.translate(str.maketrans('', '', string.punctuation))  # удаляем пунктуацию
    text = re.sub(r'\s+', ' ', text).strip()  # убираем лишние пробелы
    return text

if __name__ == "__main__":
    # Загружаем тренировочные данные
    df = pd.read_parquet("app/dataset.parquet")
    df.columns = ["title", "category"]

    # Чистим тексты
    df["title"] = df["title"].apply(clean_text)

    # Разделяем данные на признаки и метки
    X = df["title"]
    y = df["category"]

    # Разделяем на тренировочную и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Настраиваем TF-IDF векторизатор
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        #stop_words='russian'
    )

    # Векторизуем данные
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    # Преобразуем метки классов в числовые
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)


    model = xgb.XGBClassifier(use_label_encoder=False)
    model.fit(X_train_tfidf, y_train)

    # Обучаем модель XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X_train_tfidf, y_train)

    le = LabelEncoder()
    y_train = le.fit_transform(df['category'])

    # Сохраняем модель и векторизатор
    with open("app/models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("app/models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("app/models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)


    print("✅ Обучение завершено и модель сохранена.")

