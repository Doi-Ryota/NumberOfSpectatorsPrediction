import pandas as pd
import jpholiday
df = pd.read_csv(r"..\data\df_preprocessed.csv")
def feature_engineering(df):
    # Date列をdatetime型に変換
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 祝日フラグの作成
    df['Is_Holiday'] = df['Date'].apply(lambda x: 1 if jpholiday.is_holiday(x) else 0)
    
    # 試合数の作成（同年内で何試合目か）
    df['Match_Number'] = df.groupby('Year').cumcount() + 1
    
    # コロナ期間の削除（2020年2月29日～2022年3月24日）
    df['Is_Covid_Restricted'] = df['Date'].apply(lambda x: 1 if pd.Timestamp('2020-02-29') <= x <= pd.Timestamp('2022-03-24') else 0)
    df = df[df['Is_Covid_Restricted'] == 0].drop(columns=['Is_Covid_Restricted'])  # コロナ期間のデータ削除
    
    # 雨量0のフラグ
    df['Rain_Zero_Flag'] = df['Total_Precipitation (mm)'].apply(lambda x: 1 if x == 0 else 0)

    # ドームの定員
    df['Capacity'] = df['Year'].apply(lambda x: 38530 if x < 2019 else 40178)

    # 占有率
    df["Occupancy"] = df["Audience"] / df['Capacity']
    
    return df
df_featured = feature_engineering(df)
df_featured.to_csv(r"..\data\df_featured.csv",index=False)