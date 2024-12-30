import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import joblib

df = pd.read_csv(r"../data/df_featured.csv")

# コロナ時期をfilter
df = df[~(df["Year"].isin([2020,2021,2022]))].reset_index(drop=True)

train_df = df[df['Year'] != 2024]  # 2024年以外をトレーニングデータに
test_df = df[df['Year'] == 2024]   # 2024年をテストデータに

# 予測に使う変数
X_columns = ['Average_Temperature (℃)', 'Total_Precipitation (mm)',
       'Average_wind_speed(m/s)', 'Weekday_Monday', 'Weekday_Saturday', 'Weekday_Sunday',
       'Weekday_Thursday', 'Weekday_Tuesday', 'Weekday_Wednesday',
       'Opponent_DeNA', 'Opponent_オリックス', 'Opponent_ヤクルト', 'Opponent_ロッテ',
       'Opponent_中日', 'Opponent_巨人', 'Opponent_広島', 'Opponent_日本ハム',
       'Opponent_楽天', 'Opponent_西武', 'Opponent_阪神', 'Is_Holiday',
       'Match_Number', 'Rain_Zero_Flag']

X_train = train_df[X_columns]
y_train = train_df['occupancy']
X_test = test_df[X_columns]
y_test = test_df['occupancy']

# 決定木回帰モデルの作成
model = DecisionTreeRegressor(random_state=42)

# モデルの訓練
model.fit(X_train, y_train)

# テストデータを使って予測
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, 1) 

# モデルを保存する
joblib.dump(model, '../results/models/decision_tree_model.pkl')

# 予測結果をDataFrameに格納
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

# 結果をCSVファイルとして保存
results_df.to_csv('../results/predictions/decision_tree_predictions.csv', index=False)