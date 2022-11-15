from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import pandas as pd
from utils import converter
class DCEvaluator:
    

    def evaluate(self, y_train, y_test, y_pred):
        y_true = converter(y_train, y_test)

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        area_under_the_curve  = auc(fpr,tpr)

        return accuracy, f1,fpr, tpr, area_under_the_curve