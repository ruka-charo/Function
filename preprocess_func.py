import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.style.use('seaborn')

import category_encoders as ce


# Label or OneHot
def encoding(encode_type, X_train, X_test, category_features):
    # エンコードのタイプを決定
    if encode_type == 'label':
        oe = ce.OrdinalEncoder(cols=category_features)
    elif encode_type == 'onehot':
        oe = ce.OneHotEncoder(cols=category_features)

    # 訓練データとテストデータを結合する
    mix_df = pd.concat([X_train, X_test])
    # 変換
    mix_tra = oe.fit_transform(mix_df)
    # データの分割
    X_train, X_test = mix_df[:X_train.shape[0]], mix_df[X_train.shape[0]:]

    return X_train, X_test


# lightgbm(sklearnAPI) 変数重要度の可視化
def lgb_importance(X_train, lgb_model):
    features = X_train.columns
    importances = lgb_model.feature_importances_
    indices = np.argsort(-importances)
    plt.bar(np.array(features)[indices], np.array(importances[indices]))
    plt.xticks(rotation=90)
    plt.show()
