from tkinter.messagebox import NO
import numpy as np
import pandas as pd



# 線形モデルの定義
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class Model2Linear(object):
    """
    線形モデル
    """
    def __init__(self):
        """
        コンストラクタ
        """
        self.model = None
        self.scaler = None

    def fit(self, tr_x,tr_y, va_x, va_y):
        """
        標準化、学習
        """
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)

        tr_x = self.scaler.transform(tr_x)
        self.model = LogisticRegression(solver="lbfgs", C=1.0)
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        """
        標準化ののち、推論
        """
        x = self.scaler.transform(x)
        return self.model.predict_proba(x)[:, 1]
        