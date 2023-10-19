import mlflow

def save_to_mlflow(model_name, dts_name, accuracy, f1, fpr, tpr, auc, MLFLOW_RUN_ID=None):
    mlflow.set_tracking_uri("http://localhost:5000")

    if MLFLOW_RUN_ID is None:
        with mlflow.start_run():
            tag = f"{model_name}-dataset:{dts_name}"
            mlflow.set_tag("mlflow.runName", tag )
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('f1', f1)
            mlflow.log_metric('fpr', fpr)
            mlflow.log_metric('tpr', tpr)
            mlflow.log_metric('auc', auc)
            mlflow.end_run()
    else:
        with mlflow.start_run(run_id=MLFLOW_RUN_ID):

            tag = f"{model_name}-dataset:{dts_name}"
            mlflow.set_tag("mlflow.runName", tag )
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('f1', f1)
            mlflow.log_metric('fpr', fpr)
            mlflow.log_metric('tpr', tpr)
            mlflow.log_metric('auc', auc)
            mlflow.end_run()