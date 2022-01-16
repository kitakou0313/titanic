from tkinter.messagebox import NO
import numpy as np
import pandas as pd
import lightgbm as lgb

class ModleGBDT(object):
    """
    lightGBMによる推論
    """
    def init(self):
        """
        コンストラクタ
        """
        self.model = None
    def fit(self, tr_x, tr_y, va_x, va_y):
        """
        学習
        """
        params = {'objective': 'binary', 'seed': 71, 'verbose': 0, 'metrics': 'binary_logloss'}
        num_round = 100

        dtrain = lgb.Dataset(tr_x, tr_y)
        dvalid = lgb.Dataset(va_x, va_y)
        
        self.model = lgb.train(
            params, dtrain, num_boost_round=num_round,valid_names=["train", "valid"], valid_sets=[dtrain, dvalid]
        )

    def predict(self, x):
        """
        推論用メソッド
        """
        return self.model.predict(x)
        


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
        