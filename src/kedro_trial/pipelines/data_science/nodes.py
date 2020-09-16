import logging
from typing import Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

area = 75


pop_a = mpatches.Patch(color="#BB6B5A", label="High")
pop_b = mpatches.Patch(color="#E5E88B", label="Medium")
pop_c = mpatches.Patch(color="#8CCB9B", label="Low")


def colormap(risk_list):
    cols = []
    for l in risk_list:
        if l == 0:
            cols.append("#BB6B5A")
        elif l == 2:
            cols.append("#E5E88B")
        elif l == 1:
            cols.append("#8CCB9B")
    return cols


def two_d_compare(
    X_test: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray, model_name: str
):
    y_pred = label_encoder.fit_transform(y_pred)
    y_test = label_encoder.fit_transform(y_test)

    area = (12 * np.random.rand(40)) ** 2
    plt.subplots(ncols=2, figsize=(10, 4))
    plt.suptitle(
        "Actual vs Predicted data : "
        + model_name
        + ". Accuracy : %.2f" % accuracy_score(y_test, y_pred)
    )

    plt.subplot(121)
    plt.scatter(
        X_test["ESTINCOME"],
        X_test["DAYSSINCELASTTRADE"],
        alpha=0.8,
        c=colormap(y_test),
    )
    plt.title("Actual")
    plt.legend(handles=[pop_a, pop_b, pop_c])

    plt.subplot(122)
    plt.scatter(
        X_test["ESTINCOME"], X_test["DAYSSINCELASTTRADE"], alpha=0.8, c=colormap(y_pred)
    )
    plt.title("Predicted")
    plt.legend(handles=[pop_a, pop_b, pop_c])

    plt.show()


def three_d_compare(
    X_test: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray, model_name: str
):
    x = X_test["TOTALDOLLARVALUETRADED"]
    y = X_test["ESTINCOME"]
    z = X_test["DAYSSINCELASTTRADE"]
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(
        "Actual vs Predicted (3D) data : "
        + model_name
        + ". Accuracy : %.2f" % accuracy_score(y_test, y_pred)
    )

    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(x, y, z, c=colormap(y_test), marker="o")
    ax.set_xlabel("TOTAL DOLLAR VALUE TRADED")
    ax.set_ylabel("ESTIMATED INCOME")
    ax.set_zlabel("DAYS SINCE LAST TRADE")
    plt.legend(handles=[pop_a, pop_b, pop_c])
    plt.title("Actual")

    ax = fig.add_subplot(122, projection="3d")
    ax.scatter(x, y, z, c=colormap(y_pred), marker="o")
    ax.set_xlabel("TOTAL DOLLAR VALUE TRADED")
    ax.set_ylabel("ESTIMATED INCOME")
    ax.set_zlabel("DAYS SINCE LAST TRADE")
    plt.legend(handles=[pop_a, pop_b, pop_c])
    plt.title("Predicted")

    plt.show()


def model_metrics(y_test, y_pred):
    print(
        "Decoded values of Churnrisk after applying inverse of label encoder : "
        + str(np.unique(y_pred))
    )

    skplt.metrics.plot_confusion_matrix(
        y_test, y_pred, text_fontsize="small", cmap="Greens", figsize=(6, 4)
    )
    plt.show()

    print(
        "The classification report for the model : \n\n"
        + classification_report(y_test, y_pred)
    )


def split_data(data: pd.DataFrame, parameters: Dict) -> List:
    """Splits data into training and test sets.

        Args:
            data: Source data.
            parameters: Parameters defined in parameters.yml.

        Returns:
            A list containing split data.

    """
    # print(columns)
    X = data.drop("CHURNRISK", axis=1).values
    y = data["CHURNRISK"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    columns = data.columns
    columns = columns.drop(["CHURNRISK"])
    # 教師データとテストデータに分割される際にNumPy配列に変換されてしまっているためDataFrameに戻す
    X_train_df = pd.DataFrame(X_train)
    X_train_df.columns = columns
    X_test_df = pd.DataFrame(X_test)
    X_test_df.columns = columns
    y_train_df = pd.DataFrame(y_train)
    y_train_df.columns = ["CHURNRISK"]
    y_test_df = pd.DataFrame(y_test)
    y_test_df.columns = ["CHURNRISK"]

    return X_train_df, X_test_df, y_train_df, y_test_df


def numerical_transformer(data: pd.DataFrame) -> pd.DataFrame:
    """数値を含む列を識別して標準化する

        Args:
            data: Source data.
        Returns:
            Preprocessed data.
    """

    scaler_numerical = StandardScaler()

    transformed_coustomers = scaler_numerical.fit_transform(data)
    print(transformed_coustomers)

    columns = data.columns
    # DataFrameに変換
    data_df = pd.DataFrame(data)
    data_df.columns = columns
    return data_df


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> LinearRegression:
    """Train the linear regression model.

        Args:
            X_train: Training data of independent features.
            y_train: Training data for price.

        Returns:
            Trained model.

    """
    randomForestClassifier = RandomForestClassifier(
        n_estimators=100, max_depth=2, random_state=0
    )

    randomForestClassifier.fit(X_train, y_train)
    return randomForestClassifier


def evaluate_model(
    rfc_model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame
):
    """Calculate the coefficient of determination and log the result.

        Args:
            rfc_model: Trained model.
            X_test: Testing data of independent features.
            y_test: Testing data for price.

    """
    model_name = "Random Forest Classifier"
    y_pred_rfc = rfc_model.predict(X_test)

    # 推論結果と実際との差を可視化
    # two_d_compare(X_test, y_test.values, y_pred_rfc, model_name)
    # three_d_compare(X_test, y_test.values, y_pred_rfc, model_name)

    # y_test = label_encoder.inverse_transform(y_test)
    # y_pred_rfc = label_encoder.inverse_transform(y_pred_rfc)
    model_metrics(y_test, y_pred_rfc)

    uniqueValues, occurCount = np.unique(y_test, return_counts=True)
    frequency_actual = (occurCount[0], occurCount[2], occurCount[1])

    uniqueValues, occurCount = np.unique(y_pred_rfc, return_counts=True)
    frequency_predicted_rfc = (occurCount[0], occurCount[2], occurCount[1])

    n_groups = 3
    fig, ax = plt.subplots(figsize=(10, 5))
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 0.8

    rects1 = plt.bar(
        index, frequency_actual, bar_width, alpha=opacity, color="g", label="Actual"
    )

    rects6 = plt.bar(
        index + bar_width,
        frequency_predicted_rfc,
        bar_width,
        alpha=opacity,
        color="purple",
        label="Random Forest - Predicted",
    )

    plt.xlabel("Churn Risk")
    plt.ylabel("Frequency")
    plt.title("Actual vs Predicted frequency.")
    plt.xticks(index + bar_width, ("High", "Medium", "Low"))
    plt.legend()

    plt.tight_layout()
    plt.show()

    # オリジナル
    # y_pred = rfc_model.predict(X_test)
    # score = r2_score(y_test, y_pred)
    # logger = logging.getLogger(__name__)
    # logger.info("Model has a coefficient R^2 of %.3f.", score)

