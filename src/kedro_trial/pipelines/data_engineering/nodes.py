import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def drop_unused_col(customers: pd.DataFrame) -> pd.DataFrame:
    """不要な列(ID)を削除する

        Args:
            customers: Source data.
        Returns:
            Preprocessed data.

    """

    # drop()メソッドは指定した部分を削除したデータを返す
    # 呼ぶだけでは元データに影響を与えないことに注意
    customers = customers.drop(["ID"], axis=1)

    return customers


def impute_missing_value(customers: pd.DataFrame) -> pd.DataFrame:
    """欠損値を補完する

        Args:
            customers: Source data.
        Returns:
            Preprocessed data.
    """
    columns = customers.columns
    imputer = SimpleImputer(strategy="most_frequent")

    imputer = imputer.fit(customers.values)

    imputed_data = imputer.transform(customers.values)
    # print(imputed_data)

    # SimpleImputerで補完されたデータはNumPy配列なのでDataFrameに変換
    imputed_data_df = pd.DataFrame(imputed_data)
    # 列名の設定
    imputed_data_df.columns = columns
    # print(imputed_data_df)
    return imputed_data_df


def categorical_transform(customers: pd.DataFrame) -> pd.DataFrame:
    """カテゴリー・データが含まれる列を指定して、数値型の列に変換する

        Args:
            customers: Source data.
        Returns:
            Preprocessed data.
    """
    # デバッグ用.DataFrameを全列表示させるように設定
    pd.set_option("display.max_columns", 100)

    # "GENDER"を"F","M"から整数(0,1)に変換
    gender_mapping = {"F": 0, "M": 1}
    customers["GENDER"] = customers["GENDER"].map(gender_mapping)
    # print(customers["GENDER"])
    # "HOMEOWNER"を"N","Y"から整数(0,1)に変換
    homeowner_mapping = {"N": 0, "Y": 1}
    customers["HOMEOWNER"] = customers["HOMEOWNER"].map(homeowner_mapping)
    # print(customers["HOMEOWNER"])

    # "CHURNRISK"を整数に変換
    churnrisk_mapping = {"High": 0, "Low": 1, "Medium": 2}
    customers["CHURNRISK"] = customers["CHURNRISK"].map(churnrisk_mapping)

    # "STATUS"のダミーラベル化
    categoricalColumns = ["STATUS"]
    dummy_coded_data = pd.get_dummies(customers[categoricalColumns])
    # print(dummy_coded_data)

    customers = customers.drop(categoricalColumns, axis=1)
    transformed_customers = pd.concat([customers, dummy_coded_data], axis=1)
    # print(transformed_customers)

    return transformed_customers

