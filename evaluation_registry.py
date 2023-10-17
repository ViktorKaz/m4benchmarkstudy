from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import pandas as pd
from utils import converter
class DCEvaluator:
    

    def evaluate(self, y_train, y_test, y_pred):
        y_true_dc = converter(y_train, y_test)
        y_pred_dc = converter(y_train, y_pred)

        accuracy = accuracy_score(y_true_dc, y_pred_dc)
        f1 = f1_score(y_true_dc, y_pred_dc)
        fpr, tpr, thresholds = roc_curve(y_true_dc, y_pred_dc, pos_label=1)
        area_under_the_curve  = auc(fpr,tpr)

        return accuracy, f1,fpr, tpr, area_under_the_curve