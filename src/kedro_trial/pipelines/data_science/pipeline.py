from kedro.pipeline import Pipeline, node

from kedro_trial.pipelines.data_science.nodes import (
    evaluate_model,
    numerical_transformer,
    split_data,
    train_model,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["preprocessed_customers", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
            ),
            node(
                func=numerical_transformer,
                inputs=["X_train"],
                outputs="numerical_transformed_X_train",
            ),
            node(
                func=train_model,
                inputs=["numerical_transformed_X_train", "y_train"],
                outputs="rfc_model",
            ),
            node(
                func=evaluate_model,
                inputs=["rfc_model", "X_test", "y_test"],
                outputs=None,
            ),
        ]
    )
