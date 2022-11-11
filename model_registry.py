class ModelRegistry:
    """
    Registry of all models

    Paramters
    ---------
    models: list
        list of machine learning models
    """
    def __init__(self, models):
        self.models=models
    
    def get_number_of_models(self):
        return len(self.models)
    def load_models_sequentially(self):
        for model in self.models:
            yield model

