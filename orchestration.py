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


data = LoadM4Dataset(main_dir_path="Dataset", dts_frequency='Daily')
# models = ModelRegistry([regressor_pipe, classifcation_pipe,expgenous_pipe])
models = ModelRegistry([rf_cusum_exogenous,
                        rf_hmm_exogenous,
                        lasso_regression, 
                        random_forest_regressor_pipeline,
                        svm_regressor_pipeline,
                        k_neighbours_regressor,
                        logistic_regression,
                        rf_classifier,
                        svm_classifier,
                        k_neighbours_classifier
                        ])


data_gen = data.load_dts_sequentially()
models_gen = models.load_models_sequentially()


mlflow.set_tracking_uri("http://localhost:5000")

evaluator = DCEvaluator()
class Orchestrator:
    def __init__(self,models_gen,data_gen, evaluator):
        self.models_gen = models_gen
        self.data_gen = data_gen
        self.evaluator = evaluator
    def run(self):
        total_number_datasets = data.get_num_lines_in_dts()
        for model in self.models_gen:
            current_dts = 0
            for dts in data.load_dts_sequentially():
            # for i in range(1,total_number_datasets):
            # for i in range(1,5):
                with mlflow.start_run():
                    mlflow.set_tag("mlflow.runName", f"{model.get_model_name()}-dataset:{current_dts}" )
                    y_train, y_test = dts
                    built_model = model.build(y_train=y_train, y_test=y_test)
                    print(f'****Fitting {model.get_model_name()} on dataset {current_dts}/{total_number_datasets}, {current_dts/total_number_datasets}% completed*****')
                    fh = model.get_fh(y_test=y_test)
                    try:
                        built_model.fit(y=y_train, fh=fh)
                        predictions = built_model.predict()
                    except ValueError:
                        print(f'###########!!!!!!!###########Error:{ValueError}')
                        continue
                    #evaluate
                    accuracy, f1,fpr, tpr, area_under_the_curve = self.evaluator.evaluate(y_train, y_test, predictions)
                    mlflow.log_metric('accuracy', accuracy)
                    mlflow.log_metric('f1', float(f1))
                    # mlflow.log_metric('fpr', fpr)
                    # mlflow.log_metric('tpr', tpr)
                    mlflow.log_metric('area_under_the_curve', float(area_under_the_curve))
                    # mlflow.sklearn.log_model(sk_model=built_model, artifact_path="models",registered_model_name=f"model:{model.get_model_name()}-dataset:{i}")
                    current_dts +=1
if __name__ == "__main__":
    orchestrator  = Orchestrator(models_gen,data_gen, evaluator)
    orchestrator.run()