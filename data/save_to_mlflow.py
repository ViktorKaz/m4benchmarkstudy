import mlflow

def save_to_mlflow(model_name, dts_name, accuracy, f1, fpr, tpr, auc, MLFLOW_EXPERIMENT_ID=None):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_id=MLFLOW_EXPERIMENT_ID)

    with mlflow.start_run():
        tag = f"{model_name}-dataset:{dts_name}"
        mlflow.set_tag("mlflow.runName", tag )
        mlflow.log_param("dataset_name",dts_name)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('f1', f1)
        mlflow.log_metric('fpr', fpr)
        mlflow.log_metric('tpr', tpr)
        mlflow.log_metric('auc', auc)
        mlflow.end_run()
