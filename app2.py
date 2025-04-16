import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

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

# キーワードとスコアの定義
keyword_patterns = {
    "なし": {
        "patterns": [
            r'なし|無し|いない|0匹|ゼロ|いなかった|見.*?ない|姿.*?ない|見当たらず|皆無|気配.*?無|全く.*?いない|全滅|おらぬ|影も見えず|ゼロだった',
            r'イカなし|ホタルイカ.*?無|全くいない|全く採れない|イカとは会えず|打ち上げも無し|イカの気配なし|全く姿見えず'
        ],
        "score": 6
    },
    "少ない": {
        "patterns": [
            r'少な|わずか|数匹|ちらほら|[1-9]\s*(?:匹|杯)|ぽつぽつ|ほとんどいない|少々|ちょっとだけ|一匹|二匹|三匹|少しづつ',
            r'5匹ほど|10匹未満|数える程度|あまりいなかった|期待より少な|1時間で[1-9]\s*(?:匹|杯)'
        ],
        "score": 4
    },
    "普通": {
        "patterns": [
            r'普通|まあまあ|そこそこ|1[0-9]\s*(?:匹|杯)|2[0-9]\s*(?:匹|杯)|例年並|標準的|悪くない|それなり|まずまず',
            r'1時間で2[0-9]\s*(?:匹|杯)|20数匹|そこそこ掬えた|まあまあ楽しめた'
        ],
        "score": 4
    },
    "多い": {
        "patterns": [
            r'多い|たくさん|いっぱい|[4-6][0-9]\s*(?:匹|杯)|掬うのに.*?楽しい|堪能|たっぷり|充実|大漁',
            r'かなり取れた|多かった|50以上|良い感じに獲れた|1時間で[3-5][0-9]\s*(?:匹|杯)'
        ],
        "score": 4
    },
    "非常に多い": {
        "patterns": [
            r'非常に|めっちゃ|すごい|爆|大量|[7-9][0-9]\s*(?:匹|杯)|1[0-9][0-9]\s*(?:匹|杯)|イカだらけ|数えきれない|圧倒的',
            r'爆寄り|大量発生|掬い放題|過去一の量|網が重くなる|10分で[4-9][0-9]\s*(?:匹|杯)'
        ],
        "score": 4
    },
    "不明": {
        "patterns": [
            r'変わり無し|変化無し|様子見|待機|気配なさそう|どんな感じ|少し動き|雰囲気|帰る|粘る|撤収|パトカー|スルメ|鍵をかけて|月齢|ゴールデンウィーク|情報に感謝',
            r'波.*?(高い|あり)|濁り|駐車場|マナー|焚き火|巡回|禁止|準備|天気|どう|どんな|でしょう|\?|？'
        ],
        "score": 1
    }
}

def extract_count(text):
    """コメントから数量を抽出"""
    number_match = re.search(r'(\d+)\s*(?:匹|杯|個|くらい|前後|ほど)|(?:一|二|三|四|五|六|七|八|九)\s*(?:匹|杯)', text, re.IGNORECASE)
    if re.search(r'ゼロ|0匹|0杯', text, re.IGNORECASE):
        return 0
    if number_match:
        count_str = number_match.group(0)
        num_map = {
            "一匹": 1, "一杯": 1, "二匹": 2, "二杯": 2, "三匹": 3, "三杯": 3,
            "四匹": 4, "四杯": 4, "五匹": 5, "五杯": 5, "六匹": 6, "六杯": 6,
            "七匹": 7, "七杯": 7, "八匹": 8, "八杯": 8, "九匹": 9, "九杯": 9
        }
        if count_str in num_map:
            return num_map[count_str]
        return int(re.search(r'\d+', count_str).group())
    return None

def is_irrelevant(text):
    """無関係なコメントを判定"""
    return bool(re.search(
        r'youtube|youtu\.be|https?:|放送|テレビ|チャンネル|雷|注意|警報|アドバイス|ご注意|感謝|頑張|どう|どんな|でしょう|\?|？|スルメ|パトカー|鍵をかけて|月齢|ゴールデンウィーク|情報に感謝|待機|様子見|帰る|撤収|準備|天気|波.*?(高い|あり)|濁り|駐車場|マナー|焚き火|巡回|禁止',
        text, re.IGNORECASE
    ))

def split_sentences(text):
    """コメントを文単位で分割"""
    return [s.strip() for s in re.split(r'[。！？\n]', text) if s.strip()]

def keyword_score(text):
    """キーワードに基づくスコアリング"""
    scores = {
        "なし": 0,
        "少ない": 0,
        "普通": 0,
        "多い": 0,
        "非常に多い": 0,
        "不明": 0
    }
    for category, config in keyword_patterns.items():
        for pattern in config["patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                scores[category] += config["score"]
    return scores

def classify_comment(text):
    """コメントをイカの湧き量に基づいて分類"""
    print(f"\nコメント: {text[:50]}...")
    
    # 無関係なコメントをフィルタリング
    if is_irrelevant(text) or len(text) < 5:
        print("分類: 不明（無関係なコメントまたは短すぎる）")
        return "不明"
    
    # 文単位で数量をチェック（後半の文を優先）
    sentences = split_sentences(text)
    count = None
    for sentence in reversed(sentences):  # 最新の状態を優先
        count = extract_count(sentence)
        if count is not None:
            break
    
    if count is not None:
        if count == 0:
            print(f"分類: なし（数量: {count}匹）")
            return "なし"
        elif count <= 10:
            print(f"分類: 少ない（数量: {count}匹）")
            return "少ない"
        elif count <= 30:
            print(f"分類: 普通（数量: {count}匹）")
            return "普通"
        elif count <= 70:
            print(f"分類: 多い（数量: {count}匹）")
            return "多い"
        else:
            print(f"分類: 非常に多い（数量: {count}匹）")
            return "非常に多い"
    
    # ゼロやなしの明示的な表現
    if re.search(r'ゼロ|なし|無し|全くいない|皆無|いなかった|見つからない|見当たらない|気配.*?なし|全く.*?見.*?ない|おらぬ', text, re.IGNORECASE):
        print("分類: なし（明示的な表現）")
        return "なし"
    
    # キーワードスコアリング
    keyword_scores = keyword_score(text)
    print(f"キーワードスコア: {keyword_scores}")
    max_score = max(keyword_scores.values())
    max_categories = [cat for cat, score in keyword_scores.items() if score == max_score]
    
    if max_score >= 4 and len(max_categories) == 1:
        print(f"分類: {max_categories[0]}（キーワードスコア）")
        return max_categories[0]
    elif max_score >= 4 and "不明" not in max_categories:
        print(f"分類: {max_categories[0]}（キーワードスコア、複数候補から選択）")
        return max_categories[0]
    
    # 文単位でキーワードスコアリング（後半優先）
    for sentence in reversed(sentences):
        sentence_scores = keyword_score(sentence)
        max_score = max(sentence_scores.values())
        max_categories = [cat for cat, score in sentence_scores.items() if score == max_score]
        if max_score >= 4 and len(max_categories) == 1:
            print(f"分類: {max_categories[0]}（文単位キーワードスコア）")
            return max_categories[0]
        elif max_score >= 4 and "不明" not in max_categories:
            print(f"分類: {max_categories[0]}（文単位キーワードスコア、複数候補から選択）")
            return max_categories[0]
    
    print("分類: 不明（明確な分類基準なし）")
    return "不明"

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
        if len(urls) == 1:
            print("警告: 追加のページリンクが見つかりませんでした。1ページのみ処理します")
    except requests.RequestException as e:
        print(f"アクセスエラー: {e}")
    return urls[:max_pages]

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
        
        if date != "不明" and text:
            print(f"\n日付: {date}")
            predicted_label = classify_comment(text)
            data.append({
                "date": date,
                "comment": text,
                "surge_level": predicted_label
            })
        else:
            print(f"スキップ: 日付={date}, コメント={text[:50]}...")
    
    time.sleep(3)

# データフレームに変換
df = pd.DataFrame(data)
print(f"\n保存データ数: {len(df)}")
if len(df) == 0:
    print("警告: データが空です")
elif len(df) <= 50:
    print("警告: 1ページ分のデータ（50件程度）のみ取得。追加ページが存在しない可能性があります")

# CSVに保存
df.to_csv("hotaruika_surge_data.csv", index=False, encoding="utf-8-sig")
print("データが保存されました: hotaruika_surge_data.csv")