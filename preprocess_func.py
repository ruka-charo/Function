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
    X_train_en, X_test_en = mix_tra[:X_train.shape[0]], mix_tra[X_train.shape[0]:]

    return X_train_en, X_test_en


# RandomForest, lightgbm(sklearnAPI) 変数重要度の可視化
def models_importance(X_train, model):
    features = X_train.columns
    importances = model.feature_importances_
    indices = np.argsort(-importances)
    plt.bar(np.array(features)[indices], np.array(importances[indices]))
    plt.xticks(rotation=90)
    plt.xlabel('features')
    plt.ylabel('importance')
    plt.title('Features Importance')
    plt.show()


# 提出用データフレーム作成
def answer_csv(y_pred, id):
    pred_df = pd.DataFrame(y_pred)
    ans = pd.DataFrame(id)
    ans = ans.merge(pred_df, right_index=True, left_index=True)

    return ans
