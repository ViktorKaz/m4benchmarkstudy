from load_dataset import LoadM4Dataset
from model_registry import ModelRegistry
from models import RegressorPipe, SupervisedClassificationPipe, ExogenousPipe


regressor_pipe = RegressorPipe(model_name='regressor_pipeline')
classifcation_pipe = SupervisedClassificationPipe(model_name='supervised_classification_pipeline')
expgenous_pipe = ExogenousPipe(model_name='exogenous_pipeline')
data = LoadM4Dataset(main_dir_path="Dataset", dts_frequency='Daily')
models = ModelRegistry([regressor_pipe, classifcation_pipe,expgenous_pipe])


data_gen = data.load_dts_sequentially()
models_gen = models.load_models_sequentially()

for model in models_gen:
    # for i in range(1,data.get_num_lines_in_dts()):
    for i in range(1,5):
        y_train, y_test = next(data_gen)
        built_model = model.build(y_train=y_train, y_test=y_test)
        print(f'fitting {model.get_model_name()} on {i}')
        fh = model.get_fh(y_test=y_test)
        built_model.fit(y=y_train, fh=fh)
