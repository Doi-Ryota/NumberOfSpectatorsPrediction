# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# User-Agentを設定
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36"
}

# 年のループ範囲
years = list(range(15, 25))  # 2015年〜2024年

# データ格納用リスト
data = []

# 各年のデータをループして取得
for year in years:
    # 2024年は特別なURL
    if year == 24:
        url = "https://baseball-freak.com/audience/hawks.html"
    else:
        url = f"https://baseball-freak.com/audience/{year}/hawks.html"

    print(f"Processing: {url}")

    # ページを取得
    response = requests.get(url, headers=headers)

    # レスポンスの確認
    if response.status_code == 200:
        # BeautifulSoupでHTML解析
        soup = BeautifulSoup(response.text, 'html.parser')

        # tschedule内の <tr> を取得
        boxes = soup.find(class_='tschedule')

        # 存在しない場合スキップ
        if not boxes:
            print(f"No data found for year {year}")
            continue

        # 各行データをループ
        for box in boxes.find_all('tr'):
            # データ収集を行う
            cells = box.find_all('td')
            if len(cells) >= 8:  # 必要なデータが揃っている場合のみ処理
                date = cells[0].text.strip()  # 日付
                audience = cells[1].text.strip().replace(" 人", "").replace(",", "")
                audience = int(audience) if audience.isdigit() else 0  # 空文字列チェック
                result = cells[2].text.strip()  # 勝敗
                score = cells[3].text.strip()  # スコア
                opponent = cells[4].text.strip()  # 対戦相手
                pitcher = cells[5].text.strip()  # 投手名
                game_time = cells[6].text.strip()  # 試合時間
                venue = cells[7].text.strip()  # 開催場所

                # データを格納
                data.append({
                    "Year": 2000 + year if year != 24 else 2024,  # 年を適切に変換
                    "Date": date,
                    "Audience": audience,
                    "Result": result,
                    "Score": score,
                    "Opponent": opponent,
                    "Pitcher": pitcher,
                    "GameTime": game_time,
                    "Venue": venue
                })

        # サーバーへの負荷を軽減するためにスリープ
        time.sleep(2)  # 2秒の間隔を設ける
    else:
        print(f"Failed to fetch data for year {year}, status code: {response.status_code}")

# DataFrameに変換
df = pd.DataFrame(data)

display(df)

# CSVに保存
df.to_csv("softbank_audience_full_data.csv", index=False, encoding='utf-8-sig')
print("データをCSVに保存しました: softbank_audience_full_data.csv")



