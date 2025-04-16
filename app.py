import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
from deep_translator import GoogleTranslator
import re

# サイトにアクセス
url = "https://rara.jp/hotaruika-toyama/"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# クチコミを取得
comments = soup.find_all("table", class_="layer")
print(f"取得したコメント数: {len(comments)}")

# データ保存用のリスト
data = []

# BARTモデルのゼロショット分類器をロード（MPS対応）
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
labels = ["none", "few", "moderate", "many", "very many"]  # 英語ラベルで分類

for comment in comments:
    # 日付を抽出
    date_elem = comment.find("div", style="float:left;")
    if date_elem:
        date_text = date_elem.text
        # 「投稿日: YYYY年MM月DD日 HH:MM」を抽出
        date_match = re.search(r"投稿日: (\d{4}年\d{2}月\d{2}日 \d{2}:\d{2})", date_text)
        date = date_match.group(1) if date_match else "不明"
    else:
        date = "不明"

    # コメントを抽出
    text_elem = comment.find("td", style="font-size:15px;vertical-align:top;")
    text = text_elem.find("span").text.strip() if text_elem and text_elem.find("span") else ""

    if date != "不明" and text:
        print(f"日付: {date}, コメント: {text[:50]}...")

        # 日本語を英語に翻訳
        try:
            text_en = GoogleTranslator(source="ja", target="en").translate(text)
        except Exception as e:
            print(f"翻訳エラー: {e}")
            text_en = text  # 翻訳失敗時は原文を使用

        # BARTでゼロショット分類
        result = classifier(text_en, candidate_labels=labels, multi_label=False)
        predicted_label_en = result["labels"][0]
        
        # 英語ラベルを日本語に変換
        label_map = {
            "none": "なし",
            "few": "少ない",
            "moderate": "普通",
            "many": "多い",
            "very many": "非常に多い"
        }
        predicted_label = label_map[predicted_label_en]
        print(f"分類結果: {predicted_label}")

        # データに追加
        data.append({
            "date": date,
            "comment": text,
            "surge_level": predicted_label
        })
    else:
        print(f"スキップ: 日付={date}, コメント={text[:50]}...")

# データフレームに変換
df = pd.DataFrame(data)
print(f"保存データ数: {len(df)}")
if len(df) == 0:
    print("警告: データが空です")

# CSVに保存
df.to_csv("hotaruika_surge_data.csv", index=False, encoding="utf-8-sig")
print("データが保存されました: hotaruika_surge_data.csv")