import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


def load_data(path: str):
    return pd.read_csv(path)


def train_test_split_data(df):
    X = df.drop(columns="crop")
    y = df["crop"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_single_feature_model(X_train, X_test, y_train, y_test, feature):
    model = LogisticRegression(
        multi_class="multinomial",
        max_iter=1000
    )
    model.fit(X_train[[feature]], y_train)
    y_pred = model.predict(X_test[[feature]])

    f1 = metrics.f1_score(y_test, y_pred, average="weighted")
    return f1


def evaluate_features(df, features=("N", "P", "K", "ph")):
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    results = {}
    for feature in features:
        results[feature] = train_single_feature_model(
            X_train, X_test, y_train, y_test, feature
        )

    return results
