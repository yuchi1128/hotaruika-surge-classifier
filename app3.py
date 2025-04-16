import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import json
import os
from urllib.parse import urljoin

# OpenAIライブラリがインストールされていない場合に備えてAPIをオプションとして扱う
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("警告: OpenAIライブラリが見つかりません。シンプルな分類モードで実行します。")
    OPENAI_AVAILABLE = False

base_url = "https://rara.jp/hotaruika-toyama/"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# APIキー設定 - 環境変数から取得するか、直接設定
api_key = os.environ.get("GROK_API_KEY", "YOUR_GROK_API_KEY")

# OpenAIクライアント初期化（利用可能な場合）
if OPENAI_AVAILABLE:
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
    except Exception as e:
        print(f"OpenAIクライアント初期化エラー: {e}")
        OPENAI_AVAILABLE = False

# APIが使えない場合のフォールバック分類関数
def simple_classify_comment(text):
    """APIが使えない場合のシンプルな分類機能"""
    text = text.lower()
    
    # なし: ホタルイカが全く見られなかった
    if any(word in text for word in ["ゼロ", "0匹", "0杯", "なし", "いない", "見えない"]):
        return "なし"
    
    # 数値が直接含まれているケース
    number_matches = re.findall(r'(\d+)(?:匹|杯)', text)
    if number_matches:
        count = int(number_matches[0])
        if count == 0:
            return "なし"
        elif count <= 10:
            return "少ない"
        elif count <= 30:
            return "普通"
        elif count <= 70:
            return "多い"
        else:  # 71匹以上
            return "非常に多い"
    
    # 表現による分類
    if any(word in text for word in ["たくさん", "いっぱい", "多い"]):
        return "多い"
    elif any(word in text for word in ["ちらほら", "少し", "少ない"]):
        return "少ない"
    elif any(word in text for word in ["そこそこ", "まあまあ"]):
        return "普通"
    elif any(word in text for word in ["大量", "爆", "すごい"]):
        return "非常に多い"
    
    # 該当なしの場合
    return "不明"

def classify_comment_with_grok(text):
    """Grok APIを使用してコメントを分類"""
    print(f"\nコメント: {text[:50]}...")
    
    # APIが使えない場合はシンプル分類を使用
    if not OPENAI_AVAILABLE:
        result = simple_classify_comment(text)
        print(f"シンプル分類結果: {result}")
        return result
    
    prompt = """
あなたはホタルイカの湧き量を分類する専門家です。以下のコメントを読み、ホタルイカの湧き量を以下の5段階で分類してください：
- なし: ホタルイカが全く見られなかった、0匹、または「いない」「ゼロ」「なし」などの表現。
- 少ない: 1〜10匹程度、または「数匹」「ちらほら」「少しだけ」などの表現。
- 普通: 11〜30匹程度、または「まあまあ」「そこそこ」「例年並み」などの表現。
- 多い: 31〜70匹程度、または「たくさん」「いっぱい」「堪能」などの表現。
- 非常に多い: 71匹以上、または「大量」「爆寄り」「イカだらけ」などの表現。
- 不明: 湧き量に関する情報が不明確、または天気（「波が高い」）、待機（「様子見」）、リンク（「youtube」）、質問（「どうですか？」）などの無関係なコメント。

**ルール**:
- コメントが時間経過を含む場合（例: 「最初はゼロだったが後で28匹」）、最新の状態を優先してください。
- 絵文字（例: 😭）や口語表現（「イカゼロ」「めっちゃ多い」）を考慮し、文脈を正確に解釈してください。
- 数量（例: 「50匹」）が明示されている場合、それを最優先で分類に使用してください。
- 無関係なコメントは必ず「不明」に分類してください。
- 不明は最後の選択肢とし、可能な限り他のカテゴリを優先してください。

**具体例**:
- 「10分で50匹ぐらいのペースですね」→ 多い（50匹は31〜70匹の範囲）
- 「0時から2時までゼロでしたが…28匹でした」→ 普通（最新の28匹を優先、11〜30匹）
- 「人も🦑も少なし😭」→ なし（「少なし」は「いない」を意味）
- 「波少し高い気配なし満潮の3時頃でしょ」→ 不明（天気情報と質問）
- 「308杯でした」→ 非常に多い（71匹以上）
- 「ポツポツ1時間、集魚ライトで集めて40匹現在は、止まりました。」→ 多い（40匹は31〜70匹）
- 「1:15 ゼロ 採れる日はこの時間帯でも採れますか？」→ なし（ゼロと質問）
- 「100匹まで粘って終了です」→ 非常に多い（71匹以上）
- 「濁りはほんの少しマシになった程度、まだまだ濁ってます」→ 不明（天気情報）

**コメント**: {text}

**分類**: 以下の形式で回答してください：
```json
{{
    "surge_level": "なし|少ない|普通|多い|非常に多い|不明",
    "reason": "分類の理由を簡潔に説明"
}}
"""
    try:
        response = client.chat.completions.create(
            model="grok-3",
            messages=[
                {"role": "system", "content": "あなたは正確な分類を行うアシスタントです。"},
                {"role": "user", "content": prompt.format(text=text)}
            ],
            max_tokens=200,
            temperature=0.3
        )
        result = json.loads(response.choices[0].message.content)
        surge_level = result["surge_level"]
        reason = result["reason"]
        print(f"分類: {surge_level}（理由: {reason}）")
        return surge_level
    except Exception as e:
        print(f"APIエラー: {e}")
        print("APIエラーのためシンプル分類を使用します")
        result = simple_classify_comment(text)
        print(f"シンプル分類結果: {result}")
        return result

def get_pagination_urls(base_url, max_pages=5):
    """ページネーションURLを取得する"""
    urls = [base_url]
    
    try:
        print(f"ベースURL({base_url})にアクセス中...")
        response = requests.get(base_url, headers=headers, timeout=10)
        print(f"ステータスコード: {response.status_code}")
        
        if response.status_code != 200:
            print(f"アクセス失敗: {response.status_code}")
            return urls
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 提供されたHTML構造に基づいてページネーション要素を探す
        pagination_div = soup.find("div", style="font-size:13px;margin-bottom:4px;")
        
        if pagination_div:
            print("ページネーション要素を検出しました")
            # クラス 'n' を持つリンクを探す
            page_links = pagination_div.find_all("a", class_="n")
            
            if page_links:
                print(f"検出されたリンク数: {len(page_links)}")
                
                for link in page_links:
                    # リンクテキストがページ番号かチェック（>>, << などを除外）
                    link_text = link.text.strip()
                    if link_text.isdigit() and link_text != "1":  # 1ページ目は既にリストにある
                        # href属性を取得（'link2', 'link3'など）
                        href = link.get('href')
                        
                        # 相対URLを絶対URLに変換
                        absolute_url = urljoin(base_url, href)
                        
                        # 最大ページ数を超えなければリストに追加
                        page_num = int(link_text)
                        if page_num <= max_pages and absolute_url not in urls:
                            print(f"ページ{page_num}のURLを追加: {absolute_url}")
                            urls.append(absolute_url)
                            
                            if len(urls) >= max_pages:
                                break
            else:
                print("ページリンクが見つかりません")
        else:
            print("ページネーション要素が見つかりません")
    
    except requests.RequestException as e:
        print(f"アクセスエラー: {e}")
    
    # URLを検証
    validated_urls = []
    for url in urls:
        try:
            print(f"URLを検証中: {url}")
            test_response = requests.head(url, headers=headers, timeout=5)
            if test_response.status_code == 200:
                print(f"有効なURL: {url}")
                validated_urls.append(url)
            else:
                print(f"無効なURL ({test_response.status_code}): {url}")
        except requests.RequestException as e:
            print(f"URL検証エラー: {url} - {e}")
    
    print(f"処理する有効なURL一覧: {validated_urls}")
    return validated_urls[:max_pages]

def main():
    print("ホタルイカ湧き量データスクレイピング開始")
    
    # 現在の作業ディレクトリを表示
    print(f"作業ディレクトリ: {os.getcwd()}")
    
    # データを保存するリスト（全ページ分）
    all_data = []
    
    # 最大5ページまで処理
    urls_to_scrape = get_pagination_urls(base_url, max_pages=5)
    
    for i, url in enumerate(urls_to_scrape):
        print(f"\n[{i+1}/{len(urls_to_scrape)}] URLを処理中: {url}")
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            print(f"ステータスコード: {response.status_code}")
            
            if response.status_code != 200:
                print(f"アクセス失敗: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, "html.parser")
            
            # コメントテーブルを探す
            comments = soup.find_all("table", class_="layer")
            print(f"取得したコメント数: {len(comments)}")

            if not comments:
                print("コメントが見つかりません。HTML構造を確認します...")
                # HTMLの頭部を表示して構造確認
                html_preview = soup.prettify()[:1000]
                print(f"HTML preview: {html_preview}")
                continue

            for comment_idx, comment in enumerate(comments):
                try:
                    print(f"\nコメント {comment_idx+1}/{len(comments)} 処理中...")
                    
                    # 日付の取得
                    date_elem = comment.find("div", style=lambda s: s and "float:left" in s)
                    
                    if date_elem:
                        date_text = date_elem.text
                        date_match = re.search(r"投稿日: (\d{4}年\d{2}月\d{2}日 \d{2}:\d{2})", date_text)
                        date = date_match.group(1) if date_match else "不明"
                    else:
                        date = "不明"

                    # コメントテキストの取得
                    text_elem = comment.find("td", style=lambda s: s and "font-size:15px" in s)
                    if text_elem and text_elem.find("span"):
                        text = text_elem.find("span").text.strip()
                    else:
                        text = ""
                    
                    if date != "不明" and text:
                        print(f"日付: {date}")
                        predicted_label = classify_comment_with_grok(text)
                        entry = {
                            "date": date,
                            "comment": text,
                            "surge_level": predicted_label,
                            "source_url": url,
                            "page_number": i+1
                        }
                        all_data.append(entry)
                        print(f"データ追加: {date}, {predicted_label}")
                    else:
                        print(f"スキップ: 日付={date}, コメント長={len(text)}文字")
                except Exception as e:
                    print(f"コメント処理中にエラー: {e}")

            # 現在までの進捗状況を表示
            print(f"\n現在の総データ数: {len(all_data)}")
            
            # 一時バックアップを保存（ページごとの個別ファイルは作成しない）
            if all_data:
                backup_df = pd.DataFrame(all_data)
                backup_filename = "hotaruika_backup.csv"
                backup_df.to_csv(backup_filename, index=False, encoding="utf-8-sig")
                print(f"バックアップデータを保存しました: {backup_filename} (総レコード数: {len(all_data)})")
            
        except requests.RequestException as e:
            print(f"アクセスエラー: {e}")
        except Exception as e:
            print(f"ページ処理中に予期しないエラー: {e}")
            
        # サーバー負荷軽減のための待機
        if i < len(urls_to_scrape) - 1:
            wait_time = 3
            print(f"{wait_time}秒待機中...")
            time.sleep(wait_time)

    # 全データをDataFrameに変換
    df = pd.DataFrame(all_data)
    print(f"\n最終データ数: {len(df)}")
    
    if len(df) == 0:
        print("警告: データが空です。HTML構造が変更されているか、アクセスが制限されている可能性があります。")
    elif len(df) <= 50:
        print("警告: データが少ないです（50件以下）。追加ページが存在しないか、処理に問題がある可能性があります。")

    # 最終CSVを保存
    final_filename = "hotaruika_surge_data.csv"
    df.to_csv(final_filename, index=False, encoding="utf-8-sig")
    print(f"最終データを保存しました: {final_filename} (総レコード数: {len(df)})")

if __name__ == "__main__":
    main()