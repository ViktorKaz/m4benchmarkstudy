import pandas as pd
import mlflow
import numpy as np

#get runs from mlflow
mlflow.set_tracking_uri("http://localhost:5000")
runs = mlflow.search_runs("0")
runs['estimator_name']=runs['tags.mlflow.runName'].apply(lambda x : str(x).split('-')[0])
# pivot_results = pd.pivot_table(data=runs.fillna(0), values=['tags.mlflow.runName'], index=['estimator_name'],aggfunc='count')

# runs.head().to_csv('runs_head.csv')

pivot_results = pd.pivot_table(data=runs.fillna(0), values=['metrics.accuracy', 'metrics.f1', 'metrics.area_under_the_curve' ], index=['estimator_name'],aggfunc=np.average)
pivot_results.to_csv('results.csv')