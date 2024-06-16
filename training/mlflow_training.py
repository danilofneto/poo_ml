import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

import pandas as pd


def create_experiment():
    # Provide an Experiment description that will appear in the UI
    experiment_description = (
        "Experimento de teste de deploy no sagemaker"
        "Este experimento tem o objetivo de fazer deploy de um modelo baseline no SageMaker"
    )

    # Provide searchable tags that define characteristics of the Runs that
    # will be in this Experiment
    experiment_tags = {
        "project_name": "Sagemaker-deploy",
        "mlflow.note.content": experiment_description,
    }

    # Create the Experiment, providing a unique name
    new_experiment = mlflow.create_experiment(
        name="Iris-Dev", tags=experiment_tags
    )


def mlflow_run_model(experiment_name, X_train, X_test, y_train, y_test):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_metric('accuracy', accuracy)
        mlflow.sklearn.log_model(model, 'model')
        mlflow.end_run()

