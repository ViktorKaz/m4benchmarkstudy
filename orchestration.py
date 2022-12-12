from load_dataset import LoadM4Dataset
from model_registry import ModelRegistry
from models import (lasso_regression, 
                    random_forest_regressor_pipeline,
                    svm_regressor_pipeline,
                    k_neighbours_regressor,
                    logistic_regression,
                    rf_classifier,
                    svm_classifier,
                    k_neighbours_classifier,
                    rf_hmm_exogenous,
                    rf_cusum_exogenous)
from evaluation_registry import DCEvaluator
import mlflow 


data = LoadM4Dataset(main_dir_path="Dataset", dts_frequency=['Yearly','Hourly','Quarterly','Weekly','Daily', 'Monthly'])
# models = ModelRegistry([regressor_pipe, classifcation_pipe,expgenous_pipe])
models = [rf_cusum_exogenous,
                        rf_hmm_exogenous,
                        lasso_regression, 
                        random_forest_regressor_pipeline,
                        svm_regressor_pipeline,
                        k_neighbours_regressor,
                        logistic_regression,
                        rf_classifier,
                        svm_classifier,
                        k_neighbours_classifier
                        ]


data_gen = data.load_dts_sequentially()
# models_gen = models.load_models_sequentially()


mlflow.set_tracking_uri("http://localhost:5000")

evaluator = DCEvaluator()
class Orchestrator:
    def __init__(self,models,data_gen, evaluator):
        self.models = models
        self.data_gen = data_gen
        self.evaluator = evaluator
    def run(self, run_id=0):
        runs = mlflow.search_runs("0")
        
        for y_train, y_test, dts_id in self.data_gen:
            for model in self.models:
                

                try:
                    tag = f"{model.get_model_name()}-dataset:{dts_id}"
                    if 'tags.mlflow.runName' in runs:
                        if tag in runs['tags.mlflow.runName'].values:
                            print(f'#### {tag} present in database. SKIPPING ####')
                            continue

                        
                        
                    built_model = model.build(y_train=y_train, y_test=y_test)
                    print(f'****Fitting {model.get_model_name()} on dataset {dts_id}')
                    fh = model.get_fh(y_test=y_test)
                    built_model.fit(y=y_train, fh=fh)
                    predictions = built_model.predict()

                    #evaluate
                    accuracy, f1,fpr, tpr, area_under_the_curve = self.evaluator.evaluate(y_train, y_test, predictions)
                    with mlflow.start_run(run_id=run_id):
                        mlflow.set_tag("mlflow.runName", tag )
                        mlflow.log_metric('accuracy', accuracy)
                        mlflow.log_metric('f1', float(f1))
                    
                        mlflow.log_metric('area_under_the_curve', float(area_under_the_curve))
                        mlflow.end_run()
                except ValueError as err:
                    print(f'###########!!!!!!!###########Error fitting: {tag}')
                    print(err)
                    with mlflow.start_run(run_id=run_id):
                        mlflow.set_tag("mlflow.runName", tag )
                        mlflow.log_param('run_error', True)
                        mlflow.end_run()
                    
if __name__ == "__main__":
    orchestrator  = Orchestrator(models=models,data_gen=data_gen, evaluator=evaluator)
    orchestrator.run(run_id=0)