from load_dataset import LoadM4Dataset
from model_registry import ModelRegistry
from models import RegressorPipe


regressor_pipe = RegressorPipe(model_name='regressor_pipeline')
data = LoadM4Dataset(main_dir_path="Dataset", dts_frequency='Daily')
models = ModelRegistry([regressor_pipe])


data_gen = data.load_dts_sequentially()
models_gen = models.load_models_sequentially()

for model in models_gen:
    for i in range(1,data.get_num_lines_in_dts()):
        y_train, y_test = next(data_gen)
        built_model = model.build(y_train)
        built_model.fit(y_train)
        print(f'fitted {model.get_model_name()} on {i}')
