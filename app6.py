import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai

# 環境変数からAPIキーを読み込み
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Google Gemini APIの設定
genai.configure(api_key=GOOGLE_API_KEY)

# ベースURL
base_url = "https://rara.jp/hotaruika-toyama/"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# データ保存用のリスト
data = []

# ページネーション形式の候補
pagination_formats = [
    "/hotaruika-toyama/page{}/",
    "/hotaruika-toyama/?page={}",
    "/hotaruika-toyama/{}/"
]

# 利用可能なモデルを確認（デバッグ用）
def list_available_models():
    try:
        models = genai.list_models()
        print("利用可能なモデル:")
        for model in models:
            print(f" - {model.name}")
        return models
    except Exception as e:
        print(f"モデル一覧取得エラー: {e}")
        return []

# モデル一覧を取得して表示
available_models = list_available_models()

def classify_with_gemini(text):
    """Google Gemini APIを使用してテキストを分類"""
    try:
        # 利用可能なモデル名に応じて調整
        # 通常は 'gemini-1.0-pro' または 'gemini-pro'
        model_name = None
        for model in available_models:
            if 'gemini' in model.name and 'pro' in model.name:
                model_name = model.name
                break
        
        if not model_name:
            model_name = 'gemini-1.5-pro'  # デフォルト値
            
        print(f"使用するモデル: {model_name}")
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        富山のホタルイカの湧き具合について書かれた以下のコメントを分析し、次のカテゴリのいずれかに分類してください：
        
        - なし: ホタルイカが全く見られない、いない
        - 少ない: わずか、数匹、ちらほら程度
        - 普通: そこそこ、まあまあ、例年並み、30匹程度
        - 多い: たくさん、いっぱい、40-60匹程度
        - 非常に多い: すごい数、大量、70匹以上
        - 不明: 湧き具合について言及していない、または判断できない
        
        コメント: {text}
        
        分類結果を「なし」「少ない」「普通」「多い」「非常に多い」「不明」のいずれかで返してください。それ以外の文章は含めないでください。
        """
        
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        # 結果が長い場合や説明が含まれている場合に正規表現で抽出
        categories = ["なし", "少ない", "普通", "多い", "非常に多い", "不明"]
        for category in categories:
            if category in result:
                return category
                
        return result if result in categories else "不明"
    except Exception as e:
        print(f"Gemini API エラー: {e}")
        return "不明"

def is_irrelevant(text):
    """無関係なコメントを判定（簡易フィルタとして残す）"""
    return bool(re.search(
        r'youtube|youtu\.be|https?:|放送|テレビ|チャンネル|アドバイス|ご注意',
        text, re.IGNORECASE
    )) or len(text) < 5

def get_pagination_urls(base_url, max_pages=5):
    """ページネーションリンクを取得"""
    urls = [base_url]
    try:
        response = requests.get(base_url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"アクセス失敗: {response.status_code}")
            return urls
        soup = BeautifulSoup(response.text, "html.parser")
        pagination_div = soup.find("div", style="font-size:13px;margin-bottom:4px;")
        if pagination_div:
            page_links = pagination_div.find_all("a", class_="n")
            for link in page_links:
                href = link.get("href")
                if href and len(urls) < max_pages:
                    if href.startswith("/"):
                        full_url = f"https://rara.jp{href}"
                    elif href.startswith("http"):
                        full_url = href
                    else:
                        page_num = re.search(r'link(\d+)', href)
                        if page_num:
                            page_num = int(page_num.group(1))
                            for fmt in pagination_formats:
                                candidate_url = f"https://rara.jp{fmt.format(page_num)}"
                                try:
                                    check_response = requests.head(candidate_url, headers=headers, timeout=5)
                                    if check_response.status_code == 200:
                                        full_url = candidate_url
                                        break
                                except requests.RequestException:
                                    continue
                            else:
                                print(f"ページ{page_num}の有効なURLが見つかりませんでした")
                                continue
                        else:
                            continue
                    if full_url not in urls:
                        urls.append(full_url)
        for page in range(2, max_pages + 1):
            for fmt in pagination_formats:
                candidate_url = f"https://rara.jp{fmt.format(page)}"
                try:
                    check_response = requests.head(candidate_url, headers=headers, timeout=5)
                    if check_response.status_code == 200 and candidate_url not in urls:
                        urls.append(candidate_url)
                except requests.RequestException:
                    continue
        print(f"検出したページリンク: {urls}")
    except requests.RequestException as e:
        print(f"アクセスエラー: {e}")
    return urls[:max_pages]

# テスト例文で動作確認
test_comment = "10時半〜12時までで30〜40杯取れました。"
print(f"テスト分類結果: {classify_with_gemini(test_comment)}")

# メイン処理を実行
try:
    # ページネーションリンクを取得
    urls_to_scrape = get_pagination_urls(base_url)
    
    # コメント収集と処理
    for url in urls_to_scrape:
        print(f"\nURLを処理中: {url}")
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"アクセス失敗: {response.status_code}")
                continue
        except requests.RequestException as e:
            print(f"アクセスエラー: {e}")
            continue
        
        soup = BeautifulSoup(response.text, "html.parser")
        comments = soup.find_all("table", class_="layer")
        print(f"取得したコメント数: {len(comments)}")
        
        if not comments:
            print("コメントが見つかりませんでした。次のページへ")
            continue
        
        for comment in comments:
            date_elem = comment.find("div", style="float:left;")
            if date_elem:
                date_text = date_elem.text
                date_match = re.search(r"投稿日: (\d{4}年\d{2}月\d{2}日 \d{2}:\d{2})", date_text)
                date = date_match.group(1) if date_match else "不明"
            else:
                date = "不明"
            
            text_elem = comment.find("td", style="font-size:15px;vertical-align:top;")
            text = text_elem.find("span").text.strip() if text_elem and text_elem.find("span") else ""
            
            if date != "不明" and text and not is_irrelevant(text):
                print(f"\n日付: {date}")
                print(f"コメント: {text[:50]}...")
                
                # Gemini APIを使用して分類
                predicted_label = classify_with_gemini(text)
                print(f"Gemini分類結果: {predicted_label}")
                
                data.append({
                    "date": date,
                    "comment": text,
                    "surge_level": predicted_label
                })
                
                # APIリクエスト制限に対応するため少し待機
                time.sleep(0.5)
            else:
                print(f"スキップ: 日付={date}, コメント={text[:50]}...")
        
        # 次のページへ進む前に待機
        time.sleep(3)
    
    # データフレームに変換
    df = pd.DataFrame(data)
    print(f"\n保存データ数: {len(df)}")
    
    # CSVに保存
    df.to_csv("hotaruika_surge_data_gemini.csv", index=False, encoding="utf-8-sig")
    print("データが保存されました: hotaruika_surge_data_gemini.csv")

except Exception as e:
    print(f"エラーが発生しました: {e}")
    # エラーが発生した場合でも収集したデータを保存
    if data:
        df = pd.DataFrame(data)
        df.to_csv("hotaruika_surge_data_gemini_error.csv", index=False, encoding="utf-8-sig")
        print(f"エラー前のデータ({len(df)}件)が保存されました: hotaruika_surge_data_gemini_error.csv")