# import mlflow
# import mlflow.sklearn
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     classification_report,
# )
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler

# from src.config import (
#     SEGMENTED_DATA_PATH,
#     MLFLOW_TRACKING_URI,
#     EXPERIMENT_NAME,
#     RANDOM_STATE,
#     TEST_SIZE,
#     FIGURES_DIR,
#     METRICS_DIR,
# )
# from src.features import FEATURE_COLUMNS


# def load_segmented_data() -> pd.DataFrame:
#     if not SEGMENTED_DATA_PATH.exists():
#         raise FileNotFoundError(
#             "No existe el dataset segmentado. Ejecuta primero: python -m src.clustering"
#         )
#     return pd.read_csv(SEGMENTED_DATA_PATH)

# def save_confusion_matrix(y_test, y_pred, model_name: str):
#     cm = confusion_matrix(y_test, y_pred)

#     plt.figure(figsize=(7, 5))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#     plt.title(f"Matriz de confusión - {model_name}")
#     plt.xlabel("Predicción")
#     plt.ylabel("Valor real")
#     plt.tight_layout()

#     output_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
#     plt.savefig(output_path)
#     plt.close()
#     return output_path


# def train_and_log_model(model_name: str, model, X_train, X_test, y_train, y_test):
#     with mlflow.start_run(run_name=model_name):
#         pipeline = Pipeline([
#             ("scaler", StandardScaler()),
#             ("model", model),
#         ])

#         pipeline.fit(X_train, y_train)
#         y_pred = pipeline.predict(X_test)

#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
#         recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
#         f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

#         mlflow.log_param("model_name", model_name)
#         mlflow.log_metric("accuracy", accuracy)
#         mlflow.log_metric("precision", precision)
#         mlflow.log_metric("recall", recall)
#         mlflow.log_metric("f1", f1)

#         report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
#         report_df = pd.DataFrame(report).transpose()
#         report_path = METRICS_DIR / f"classification_report_{model_name}.csv"
#         report_df.to_csv(report_path)
#         mlflow.log_artifact(str(report_path), artifact_path="metrics")

#         cm_path = save_confusion_matrix(y_test, y_pred, model_name)
#         mlflow.log_artifact(str(cm_path), artifact_path="figures")

#         input_example = X_test.head(3)
#         mlflow.sklearn.log_model(
#             sk_model=pipeline,
#             artifact_path="model",
#             input_example=input_example,
#         )

#         print(f"Modelo: {model_name}")
#         print(f"Accuracy: {accuracy:.4f}")
#         print(f"F1: {f1:.4f}")


# def train_models():
#     FIGURES_DIR.mkdir(parents=True, exist_ok=True)
#     METRICS_DIR.mkdir(parents=True, exist_ok=True)

#     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#     mlflow.set_experiment(EXPERIMENT_NAME)

#     df = load_segmented_data()

#     X = df[FEATURE_COLUMNS]
#     y = df["customer_segment"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=TEST_SIZE,
#         random_state=RANDOM_STATE,
#         stratify=y,
#     )

#     models = {
#         "logistic_regression": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
#         "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
#         "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
#     }

#     for model_name, model in models.items():
#         train_and_log_model(model_name, model, X_train, X_test, y_train, y_test)


# if __name__ == "__main__":
#     train_models()