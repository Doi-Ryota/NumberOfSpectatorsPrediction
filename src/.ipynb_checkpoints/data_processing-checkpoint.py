# -*- coding: utf-8 -*-

import jpholiday
import pandas as pd

"""ソフトバンクデータの読み込み・前処理を行う関数"""
def process_softbank_data(file_path):
    # データの読み込み
    softbank = pd.read_csv(file_path, encoding="utf-8")

    # 年月日を適切な形式に変換
    softbank["FormattedDate"] = pd.to_datetime(
        softbank["Year"].astype(str) + "/" +
        softbank["Date"].str.extract(r"(\d+)月(\d+)日")[0].fillna('0') +
        "/" + softbank["Date"].str.extract(r"(\d+)月(\d+)日")[1].fillna('0'),
        format="%Y/%m/%d"
    )

    # 曜日を追加
    softbank["Weekday"] = softbank["FormattedDate"].dt.day_name()

    return softbank

def process_weather_data(file_path):
    """天気データの読み込み・前処理を行う関数"""
    # データの読み込み
    weather = pd.read_csv(file_path, encoding="shift_jis")

    # 列名をリネーム
    weather.columns = [
        "yyyy/mm/dd",
        "Average_Temperature (℃)",
        "Total_Precipitation (mm)",
        "Average_wind_speed(m/s)"
    ]

    # Date列をdatetime型に変換
    weather["yyyy/mm/dd"] = pd.to_datetime(weather["yyyy/mm/dd"])

    return weather

def merge_datasets(softbank_df, weather_df):
    """ソフトバンクデータと天気データをマージする関数"""
    # マージ処理: 結合キーをdatetime型に合わせる
    merged_df = pd.merge(softbank_df, weather_df, how="left", left_on="FormattedDate", right_on="yyyy/mm/dd")
    return merged_df

def save_to_csv(df, output_path):
    """データフレームをCSVとして出力する関数"""
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

# ファイルパスの設定
softbank_file = r"..\data\softbank_audience_full_data.csv"
weather_file = r"..\data\weather.csv"
output_file = r"..\data\final_data.csv"

# 前処理の実行
softbank_df = process_softbank_data(softbank_file)
weather_df = process_weather_data(weather_file)

# データのマージ
df = merge_datasets(softbank_df, weather_df)

def preprocess_data(df):
    # "中止"データを排除
    df = df[df['Score'] != '中止'].copy()  # copy()を追加して警告を回避

    # 該当のドームのみ
    df = df[df["Venue"].isin(['ヤフオクドーム', 'PayPayドーム', 'みずほPayPay'])].copy()

    # 日付の処理
    df['Date'] = pd.to_datetime(df['FormattedDate'])
    df['Weekday'] = pd.to_datetime(df['FormattedDate']).dt.day_name()

    # 結果の数値化
    df['Result'] = df['Result'].apply(lambda x: 0 if x == '●' else 1)

    # スコアの分割
    df[['Home_Score', 'Away_Score']] = df['Score'].str.split(' - ', expand=True)
    df['Home_Score'] = pd.to_numeric(df['Home_Score'])
    df['Away_Score'] = pd.to_numeric(df['Away_Score'])

    # 観客数の数値化
    df['Audience'] = pd.to_numeric(df['Audience'])

    # 降水量、気温、雲量の数値化
    df['Total_Precipitation (mm)'] = pd.to_numeric(df['Total_Precipitation (mm)'])
    df['Average_Temperature (℃)'] = pd.to_numeric(df['Average_Temperature (℃)'])
    df[ "Average_wind_speed(m/s)"] = pd.to_numeric(df[ "Average_wind_speed(m/s)"])

    # 対戦チームをエンコード
    n = df["Weekday"].nunique()
    df = pd.get_dummies(df, columns=['Weekday'], drop_first=False)
    # 最後のn列をint型に変換
    df.iloc[:, -n:] = df.iloc[:, -n:].astype(int)

    # 曜日をエンコード
    n = df["Opponent"].nunique()
    df = pd.get_dummies(df, columns=['Opponent'], drop_first=False)
    # 最後のn列をint型に変換
    df.iloc[:, -n:] = df.iloc[:, -n:].astype(int) 

    # ゲーム時間の分に変換
    df['GameTime'] = df['GameTime'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

    # 不要な列の削除
    df.drop(columns=['FormattedDate', 'Score','yyyy/mm/dd', "Venue"], inplace=True)

    return df

df_preprocessed = preprocess_data(df)
save_to_csv(df_preprocessed, r"..\data\df_preprocessed.csv")