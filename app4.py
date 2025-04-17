import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import json
import os
from urllib.parse import urljoin
import logging
import unicodedata
import traceback

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hotaruika_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# OpenAIライブラリがインストールされていない場合に備えてAPIをオプションとして扱う
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    logger.info("OpenAIライブラリが利用可能です")
except ImportError:
    logger.warning("OpenAIライブラリが見つかりません。シンプルな分類モードで実行します。")
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
        # APIが動作するか簡単なチェック
        test_response = client.chat.completions.create(
            model="grok-3",
            messages=[{"role": "user", "content": "こんにちは"}],
            max_tokens=10
        )
        logger.info("Grok APIの接続確認に成功しました")
    except Exception as e:
        logger.error(f"OpenAIクライアント初期化エラー: {e}")
        OPENAI_AVAILABLE = False

def normalize_text(text):
    """テキストを正規化する（全角→半角、余分なスペースの削除など）"""
    # 全角→半角変換
    text = unicodedata.normalize('NFKC', text)
    # 余分なスペースの削除
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_numbers(text):
    """テキストから数値表現を抽出"""
    # 「10分で50匹ぐらい」「50匹くらい」「約50匹」などのパターンを検出
    patterns = [
        r'(\d+)\s*(?:匹|杯|尾|個|つ)',  # 基本パターン
        r'約\s*(\d+)',                   # 「約30」などのパターン
        r'(\d+)\s*(?:くらい|ぐらい|ほど|程度|位)', # 「30くらい」などのパターン
        r'(\d+)\s*(?:ずつ|づつ)',        # 「10匹ずつ」などのパターン
    ]
    
    numbers = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                numbers.append(int(match.group(1)))
            except ValueError:
                continue
    
    return numbers if numbers else None

def detect_negation(text):
    """否定表現を検出"""
    negation_patterns = [
        r'(?:全く|まったく|全然|皆無|ない|いない|見えない|見当たらない|ゼロ|0|不在)',
        r'(?:居ない|出ない|姿が見えない|あたり[一つ\d+]も)',
        r'(?:気配.*?(?:なし|無し|ない|無い|皆無))',
        r'(?:なし|無し|ゼロ|0匹|0杯)',
        r'(?:イカ.*?(?:なし|無し|いない|居ない|見えない|見当たらない|ゼロ|0))',
        r'(?:いかいない|イカいない|イカゼロ|イカはいない)',
        r'(?:一匹も.*?(?:なし|無し|いない|見えない|取れない))',
        r'(?:帰|諦|撤収)',
        r'(?:取れ.*?(?:なかった|ゼロ|0|ない))',
    ]
    
    for pattern in re.compile('|'.join(negation_patterns), re.IGNORECASE).finditer(text):
        return True
    return False

def detect_small_amount(text):
    """少量表現を検出"""
    small_patterns = [
        r'(?:少し|少ない|ちらほら|わずか|かろうじて|数匹|数杯|数個|ポツポツ)',
        r'(?:少量|少なめ|少なかった|僅か|乏しい)',
        r'(?:\d{1}匹|\d{1}杯)',  # 1〜9匹という表現
        r'(?:何とか\d+匹)',      # 「何とか3匹」などの表現
    ]
    
    for pattern in re.compile('|'.join(small_patterns), re.IGNORECASE).finditer(text):
        return True
    return False

def detect_normal_amount(text):
    """通常量表現を検出"""
    normal_patterns = [
        r'(?:普通|そこそこ|まあまあ|それなり|平均的|例年並み|並み|標準|通常)',
        r'(?:ふつう|ノーマル|いつも通り|いつも程度)',
        r'(?:そこそこ|10.*?20|十数|十匹程度)',
    ]
    
    for pattern in re.compile('|'.join(normal_patterns), re.IGNORECASE).finditer(text):
        return True
    return False

def detect_large_amount(text):
    """多量表現を検出"""
    large_patterns = [
        r'(?:多い|たくさん|いっぱい|多数|多め|増えてきた|いっぱい|アップ)',
        r'(?:よく獲れる|よく取れる|取れ出した|取り放題|豊富)',
        r'(?:堪能|夢中|集中|忙しい)',
        r'(?:増加|増えてきた|増えた)',
    ]
    
    for pattern in re.compile('|'.join(large_patterns), re.IGNORECASE).finditer(text):
        return True
    return False

def detect_very_large_amount(text):
    """非常に多い表現を検出"""
    very_large_patterns = [
        r'(?:大量|爆|すごい|凄い|すごく|凄く|かなり|相当|超|非常に多い)',
        r'(?:うじゃうじゃ|たっぷり|山ほど|溢れる|あふれる|イカだらけ)',
        r'(?:群れ|最高|多すぎ|沢山|過去最高|記録的)',
        r'(?:大漁|豊漁|収穫祭|すごい数)',
        r'(?:押し寄せる|集まり|集合|密集)',
    ]
    
    for pattern in re.compile('|'.join(very_large_patterns), re.IGNORECASE).finditer(text):
        return True
    return False

def detect_unrelated(text):
    """無関係なコメントを検出"""
    unrelated_patterns = [
        r'(?:天気|波|濁り|風|雨|雪|気温)',
        r'(?:質問|何処|どこ|どう|でしょうか\?)',
        r'(?:情報|求む|教えて|誰か)',
        r'(?:移動|向かう|出発|予定|明日)',
        r'(?:トイレ|駐車場|混雑|混み具合)',
        r'(?:泊まる|ホテル|宿泊)',
        r'(?:どうも|ありがとう|感謝)',
    ]
    
    # コメントが短く、ホタルイカに関する情報がない場合
    if len(text) < 15 and not re.search(r'(?:イカ|いか|ホタルイカ|掬|すくい|匹|杯)', text):
        return True
    
    # 明らかに質問や情報提供の依頼の場合
    if re.search(r'(?:ですか\?|\?|どう|情報求む|教えて|どなたか)', text):
        # ただし「〜はいないですか?」のような形式の報告は除外
        if not re.search(r'(?:いない|居ない|ゼロ|無い|なし)(?:です|でした)(?:か|ね|よ)', text):
            return True
    
    # 無関係なトピックが中心の場合
    unrelated_count = 0
    for pattern in re.compile('|'.join(unrelated_patterns), re.IGNORECASE).finditer(text):
        unrelated_count += 1
    
    # 無関係なトピックが多く、イカに関する具体的な言及がない場合
    if unrelated_count >= 2 and not re.search(r'(?:匹|杯|獲れ|取れ|収穫|掬|すくい)', text):
        return True
    
    return False

def improved_simple_classify(text):
    """改良版シンプル分類関数"""
    # テキストの正規化
    text = normalize_text(text)
    logger.debug(f"正規化されたテキスト: {text}")
    
    # 数値を抽出
    numbers = extract_numbers(text)
    logger.debug(f"抽出された数値: {numbers}")
    
    # 無関係なコメントかチェック
    if detect_unrelated(text):
        return "不明", "無関係なコメントまたは質問のため"
    
    # 数値に基づく分類（最優先）
    if numbers:
        max_num = max(numbers)
        if max_num == 0:
            return "なし", f"数値表現: {max_num}匹"
        elif max_num <= 10:
            return "少ない", f"数値表現: {max_num}匹"
        elif max_num <= 30:
            return "普通", f"数値表現: {max_num}匹"
        elif max_num <= 70:
            return "多い", f"数値表現: {max_num}匹"
        else:  # 71匹以上
            return "非常に多い", f"数値表現: {max_num}匹"
    
    # テキスト表現に基づく分類
    # 「なし」の検出（最優先）
    if detect_negation(text):
        return "なし", "否定表現を検出"
    
    # 「非常に多い」の検出
    if detect_very_large_amount(text):
        return "非常に多い", "大量表現を検出"
    
    # 「多い」の検出
    if detect_large_amount(text):
        return "多い", "多量表現を検出"
    
    # 「普通」の検出
    if detect_normal_amount(text):
        return "普通", "通常量表現を検出"
    
    # 「少ない」の検出
    if detect_small_amount(text):
        return "少ない", "少量表現を検出"
    
    # 判断できない場合
    return "不明", "判断基準に合致しない"

def classify_comment_with_grok(text):
    """Grok APIを使用してコメントを分類"""
    logger.info(f"\nコメント: {text[:50]}...")
    
    # APIが使えない場合はシンプル分類を使用
    if not OPENAI_AVAILABLE:
        result, reason = improved_simple_classify(text)
        logger.info(f"シンプル分類結果: {result}（理由: {reason}）")
        return result
    
    prompt = """
あなたはホタルイカの湧き量を分類する専門家です。以下のコメントを読み、ホタルイカの湧き量を以下の5段階で分類してください：
- なし: ホタルイカが全く見られなかった、0匹、または「いない」「ゼロ」「なし」などの表現。
- 少ない: 1〜10匹程度、または「数匹」「ちらほら」「少しだけ」などの表現。
- 普通: 11〜30匹程度、または「まあまあ」「そこそこ」「例年並み」などの表現。
- 多い: 31〜70匹程度、または「たくさん」「いっぱい」「堪能」などの表現。
- 非常に多い: 71匹以上、または「大量」「爆寄り」「イカだらけ」などの表現。
- 不明: 湧き量に関する情報が不明確、または天気（「波が高い」）、待機（「様子見」）、リンク（「youtube」）、質問（「どうですか？」）などの無関係なコメント。

**重要なルール**:
- 数値が明示されている場合は、それを最優先で使用してください。例えば「10分で50匹ぐらい」は「多い」に分類。
- 数値が複数ある場合は、最も新しい状況や最も多い数値を優先してください。
- 否定的な表現（「いない」「ゼロ」「全く見えない」など）があれば「なし」に分類。
- 質問文や情報を求めるコメントは「不明」に分類。
- 湧き量に関する情報がないコメントは「不明」に分類。
- すくうのに忙しいといった表現は湧き量が多いことを示唆します。

**コメント**: {text}

**分類**: 以下の形式で回答してください：
```json
{{
    "surge_level": "なし|少ない|普通|多い|非常に多い|不明",
    "reason": "分類の理由を簡潔に説明"
}}
```
"""
    try:
        # まずシンプル分類も実行して比較のために保存
        simple_result, simple_reason = improved_simple_classify(text)
        logger.info(f"シンプル分類結果: {simple_result}（理由: {simple_reason}）")
        
        # Grok APIを使用
        response = client.chat.completions.create(
            model="grok-3",
            messages=[
                {"role": "system", "content": "あなたは正確な分類を行うアシスタントです。"},
                {"role": "user", "content": prompt.format(text=text)}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        # JSONをパース
        try:
            result = json.loads(response.choices[0].message.content)
            surge_level = result["surge_level"]
            reason = result["reason"]
            logger.info(f"API分類結果: {surge_level}（理由: {reason}）")
            
            # シンプル分類と比較して不一致がある場合はログに記録
            if simple_result != surge_level and simple_result != "不明" and surge_level != "不明":
                logger.warning(f"分類不一致: API={surge_level}, シンプル={simple_result}, テキスト={text[:50]}...")
            
            return surge_level
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"APIレスポンスのパースエラー: {e}")
            logger.error(f"APIレスポンス: {response.choices[0].message.content}")
            return simple_result
            
    except Exception as e:
        logger.error(f"APIエラー: {e}")
        logger.error(traceback.format_exc())
        logger.info("APIエラーのためシンプル分類を使用します")
        return simple_result

def get_pagination_urls(base_url, max_pages=5):
    """ページネーションURLを取得する"""
    urls = [base_url]
    
    try:
        logger.info(f"ベースURL({base_url})にアクセス中...")
        response = requests.get(base_url, headers=headers, timeout=10)
        logger.info(f"ステータスコード: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"アクセス失敗: {response.status_code}")
            return urls
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 提供されたHTML構造に基づいてページネーション要素を探す
        pagination_div = soup.find("div", style="font-size:13px;margin-bottom:4px;")
        
        if pagination_div:
            logger.info("ページネーション要素を検出しました")
            # クラス 'n' を持つリンクを探す
            page_links = pagination_div.find_all("a", class_="n")
            
            if page_links:
                logger.info(f"検出されたリンク数: {len(page_links)}")
                
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
                            logger.info(f"ページ{page_num}のURLを追加: {absolute_url}")
                            urls.append(absolute_url)
                            
                            if len(urls) >= max_pages:
                                break
            else:
                logger.warning("ページリンクが見つかりません")
        else:
            logger.warning("ページネーション要素が見つかりません")
    
    except requests.RequestException as e:
        logger.error(f"アクセスエラー: {e}")
    
    # URLを検証
    validated_urls = []
    for url in urls:
        try:
            logger.info(f"URLを検証中: {url}")
            test_response = requests.head(url, headers=headers, timeout=5)
            if test_response.status_code == 200:
                logger.info(f"有効なURL: {url}")
                validated_urls.append(url)
            else:
                logger.warning(f"無効なURL ({test_response.status_code}): {url}")
        except requests.RequestException as e:
            logger.error(f"URL検証エラー: {url} - {e}")
    
    logger.info(f"処理する有効なURL一覧: {validated_urls}")
    return validated_urls[:max_pages]

def main():
    logger.info("ホタルイカ湧き量データスクレイピング開始")
    
    # 現在の作業ディレクトリを表示
    logger.info(f"作業ディレクトリ: {os.getcwd()}")
    
    # データを保存するリスト（全ページ分）
    all_data = []
    
    # 最大5ページまで処理
    urls_to_scrape = get_pagination_urls(base_url, max_pages=5)
    
    for i, url in enumerate(urls_to_scrape):
        logger.info(f"\n[{i+1}/{len(urls_to_scrape)}] URLを処理中: {url}")
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            logger.info(f"ステータスコード: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"アクセス失敗: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, "html.parser")
            
            # コメントテーブルを探す
            comments = soup.find_all("table", class_="layer")
            logger.info(f"取得したコメント数: {len(comments)}")

            if not comments:
                logger.warning("コメントが見つかりません。HTML構造を確認します...")
                # HTMLの頭部を表示して構造確認
                html_preview = soup.prettify()[:1000]
                logger.debug(f"HTML preview: {html_preview}")
                continue

            for comment_idx, comment in enumerate(comments):
                try:
                    logger.info(f"\nコメント {comment_idx+1}/{len(comments)} 処理中...")
                    
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
                        logger.info(f"日付: {date}")
                        predicted_label = classify_comment_with_grok(text)
                        entry = {
                            "date": date,
                            "comment": text,
                            "surge_level": predicted_label,
                            "source_url": url,
                            "page_number": i+1
                        }
                        all_data.append(entry)
                        logger.info(f"データ追加: {date}, {predicted_label}")
                    else:
                        logger.warning(f"スキップ: 日付={date}, コメント長={len(text)}文字")
                except Exception as e:
                    logger.error(f"コメント処理中にエラー: {e}")
                    logger.error(traceback.format_exc())

            # 現在までの進捗状況を表示
            logger.info(f"\n現在の総データ数: {len(all_data)}")
            
            # 一時バックアップを保存（ページごとの個別ファイルは作成しない）
            if all_data:
                backup_df = pd.DataFrame(all_data)
                backup_filename = "hotaruika_backup.csv"
                backup_df.to_csv(backup_filename, index=False, encoding="utf-8-sig")
                logger.info(f"バックアップデータを保存しました: {backup_filename} (総レコード数: {len(all_data)})")
            
        except requests.RequestException as e:
            logger.error(f"アクセスエラー: {e}")
        except Exception as e:
            logger.error(f"ページ処理中に予期しないエラー: {e}")
            logger.error(traceback.format_exc())
            
        # サーバー負荷軽減のための待機
        if i < len(urls_to_scrape) - 1:
            wait_time = 3
            logger.info(f"{wait_time}秒待機中...")
            time.sleep(wait_time)

    # 全データをDataFrameに変換
    df = pd.DataFrame(all_data)
    logger.info(f"\n最終データ数: {len(df)}")
    
    if len(df) == 0:
        logger.warning("警告: データが空です。HTML構造が変更されているか、アクセスが制限されている可能性があります。")
    elif len(df) <= 50:
        logger.warning("警告: データが少ないです（50件以下）。追加ページが存在しないか、処理に問題がある可能性があります。")

    # 最終CSVを保存
    final_filename = "hotaruika_surge_data.csv"
    df.to_csv(final_filename, index=False, encoding="utf-8-sig")
    logger.info(f"最終データを保存しました: {final_filename} (総レコード数: {len(df)})")
    
    # 分類結果の統計
    if not df.empty:
        classification_stats = df["surge_level"].value_counts()
        logger.info(f"\n分類結果の統計:\n{classification_stats}")

if __name__ == "__main__":
    main()