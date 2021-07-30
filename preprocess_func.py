import pandas as pd
import numpy as np

import category_encoders as ce
from sklearn.model_selection import KFold


'''Label or OneHot'''
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


'''Target Encoding'''
def target_encoding(cat_cols, train_x, train_y, test_x, n_splits=4, random_state=1):
    # =======================================
    # cat_cols: 変換したいカテゴリデータ
    # train_x: 訓練データの説明変数
    # train_y: 訓練データの目的変数
    # test_x: テストデータ
    # n_splits: encodingのためのfold数(4〜10)
    # =======================================

    for c in cat_cols:
        # テストデータの変換用に、学習データ全体で各カテゴリにおけるtargetの平均を計算
        data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
        target_mean = data_tmp.groupby(c)['target'].mean()# 各カテゴリごとの平均
        # テストデータのカテゴリを変換
        test_x[c] = test_x[c].map(target_mean)

        # 学習データの変換後の値を格納する配列を準備
        tmp = np.repeat(np.nan, train_x.shape[0])

        # 学習データを分割
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for idx_1, idx_2 in kf.split(train_x):
            # out-of-foldで各カテゴリにおける目的変数の平均を計算
            target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
            # 変換後の値を配列に格納
            tmp[idx_2] = train_x[c].iloc[idx_2].map(target_mean)

        # 変換後のデータでもとの変数を置換
        train_x[c] = tmp

    return train_x, test_x


# for分の中で学習、評価をする必要あり
def k_target_encoding(cat_cols, train_x, train_y, test_x, val_split=4, n_splits=4, random_state1=1, random_state2=2):
    kf = KFold(n_splits=val_splits, shuffle=True, random_state=random_state1) # 評価用
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        # 学習データからバリデーションデータを分ける
        tr_x, va_x = train_x.iloc[tr_idx].copy(), train_x.iloc[va_idx].copy()
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # カテゴリ変数をtarget encoding
        for c in cat_cols:
            # バリデーションデータの変換用に、学習データ全体で各カテゴリにおけるtargetの平均を計算
            data_tmp = pd.DataFrame({c: tr_x[c], 'target': tr_y})
            target_mean = data_tmp.groupby(c)['target'].mean()# 各カテゴリごとの平均
            # バリデーションデータのカテゴリを変換
            va_x[c] = va_x[c].map(target_mean)

            # 学習データの変換後の値を格納する配列を準備
            tmp = np.repeat(np.nan, tr_x.shape[0])

            # 学習データを分割
            kf_encoding = KFold(n_splits=n_splits, shuffle=True, random_state=random_state2)# target encoding用
            for idx_1, idx_2 in kf_encoding.split(tr_x):
                # out-of-foldで各カテゴリにおける目的変数の平均を計算
                target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
                # 変換後の値を配列に格納
                tmp[idx_2] = tr_x[c].iloc[idx_2].map(target_mean)

            # 変換後のデータでもとの変数を置換
            tr_x[c] = tmp

    return tr_x, va_x, tr_y, va_y # このままでは使えないがメモ


'''lightgbm(sklearnAPI) 変数重要度の可視化'''
def lgb_importance(X_train, lgb_model):
    features = X_train.columns
    importances = lgb_model.feature_importances_
    indices = np.argsort(-importances)
    plt.bar(np.array(features)[indices], np.array(importances[indices]))
    plt.xticks(rotation=90)
    plt.show()
