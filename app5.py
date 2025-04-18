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
    logger.warning("OpenAIライブラリが見つかりません。シンプル分類モードで実行します。")
    OPENAI_AVAILABLE = False

base_url = "https://rara.jp/hotaruika-toyama/"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# APIキー設定 - 環境変数から取得
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
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_numbers(text):
    """テキストから数値表現を抽出（改良版）"""
    patterns = [
        r'(\d+)\s*(?:匹|杯|尾|個|つ)',  # 基本パターン
        r'約\s*(\d+)',                   # 「約30」
        r'(\d+)\s*(?:くらい|ぐらい|ほど|程度|位)', # 「30くらい」
        r'(\d+)\s*(?:ずつ|づつ)',        # 「10匹ずつ」
        r'(\d+)(?:〜|~)(\d+)',          # 「2〜3匹」
        r'(?:二|三|四|五|六|七|八|九|十)\s*(?:匹|杯|尾|個|つ)', # 漢数字
    ]
    numbers = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                num = int(match.group(1))
                numbers.append(num)
            except (ValueError, IndexError):
                # 範囲指定の場合（例: 2〜3匹）は平均を取る
                if match.group(0).find('〜') != -1 or match.group(0).find('~') != -1:
                    try:
                        num1 = int(match.group(1))
                        num2 = int(match.group(2))
                        numbers.append((num1 + num2) // 2)
                    except (ValueError, IndexError):
                        continue
                # 漢数字の場合
                elif re.match(r'(?:二|三|四|五|六|七|八|九|十)', match.group(0)):
                    kanji_map = {'二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
                    for kanji, value in kanji_map.items():
                        if kanji in match.group(0):
                            numbers.append(value)
                            break
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
        r'(?:\d{1}\s*(?:匹|杯|尾|個|つ))',  # 1〜9匹
        r'(?:何とか\d+\s*(?:匹|杯))',      # 「何とか3匹」
    ]
    for pattern in re.compile('|'.join(small_patterns), re.IGNORECASE).finditer(text):
        return True
    return False

def detect_normal_amount(text):
    """通常量表現を検出"""
    normal_patterns = [
        r'(?:普通|そこそこ|まあまあ|それなり|平均的|例年並み|並み|標準|通常)',
        r'(?:ふつう|ノーマル|いつも通り|いつも程度)',
        r'(?:十数|十匹程度|10.*?20\s*(?:匹|杯))',
    ]
    for pattern in re.compile('|'.join(normal_patterns), re.IGNORECASE).finditer(text):
        return True
    return False

def detect_large_amount(text):
    """多量表現を検出"""
    large_patterns = [
        r'(?:多い|たくさん|いっぱい|多数|多め|増えてきた|アップ)',
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
    """無関係なコメントを検出（改良版）"""
    # 短いコメントでもホタルイカ関連なら処理続行
    if len(text) < 15 and not re.search(r'(?:イカ|いか|ホタルイカ|掬|すくい|匹|杯)', text, re.IGNORECASE):
        return True
    # 質問文だが報告を含む場合は除外
    if re.search(r'(?:ですか\?|\?|どう|情報求む|教えて|どなたか)', text) and not re.search(r'(?:いない|居ない|ゼロ|無い|なし)(?:です|でした)(?:か|ね|よ)', text):
        return True
    unrelated_patterns = [
        r'(?:天気|波|濁り|風|雨|雪|気温)',
        r'(?:移動|向かう|出発|予定|明日)',
        r'(?:トイレ|駐車場|混雑|混み具合)',
        r'(?:泊まる|ホテル|宿泊)',
    ]
    unrelated_count = sum(1 for pattern in unrelated_patterns if re.search(pattern, text, re.IGNORECASE))
    return unrelated_count >= 2 and not re.search(r'(?:匹|杯|獲れ|取れ|収穫|掬|すくい)', text, re.IGNORECASE)

def score_comment(text):
    """コメントをスコアリングして湧き量を評価"""
    score = 0
    if detect_very_large_amount(text):
        score += 3
    if detect_large_amount(text):
        score += 2
    if detect_normal_amount(text):
        score += 1
    if detect_small_amount(text):
        score -= 1
    if detect_negation(text):
        score -= 3
    numbers = extract_numbers(text)
    if numbers:
        max_num = max(numbers)
        if max_num >= 71:
            score += 3
        elif max_num >= 31:
            score += 2
        elif max_num >= 11:
            score += 1
        elif max_num > 0:
            score -= 1
        else:
            score -= 3
    return score

def improved_simple_classify(text):
    """改良版シンプル分類関数"""
    text = normalize_text(text)
    logger.debug(f"正規化されたテキスト: {text}")
    
    if detect_unrelated(text):
        return "不明", "無関係なコメントまたは質問"
    
    score = score_comment(text)
    numbers = extract_numbers(text)
    logger.debug(f"スコア: {score}, 抽出された数値: {numbers}")
    
    if score <= -2:
        return "なし", "否定表現またはゼロ"
    elif score == -1:
        return "少ない", "少量表現または1〜10匹"
    elif score == 0 or score == 1:
        return "普通", "通常量または11〜30匹"
    elif score == 2:
        return "多い", "多量または31〜70匹"
    else:
        return "非常に多い", "大量または71匹以上"

def classify_comment_with_grok(text):
    """Grok APIを使用してコメントを分類（改良版）"""
    logger.info(f"\nコメント: {text[:50]}...")
    
    # シンプル分類を先に実行
    simple_result, simple_reason = improved_simple_classify(text)
    logger.info(f"シンプル分類結果: {simple_result}（理由: {simple_reason}）")
    
    if not OPENAI_AVAILABLE:
        return simple_result
    
    prompt = """
あなたはホタルイカの湧き量を分類する専門家です。以下のコメントを読み、ホタルイカの湧き量を以下の5段階で分類してください：
- なし: 0匹、「いない」「ゼロ」「なし」など。
- 少ない: 1〜10匹、「数匹」「ちらほら」など。
- 普通: 11〜30匹、「そこそこ」「まあまあ」など。
- 多い: 31〜70匹、「たくさん」「いっぱい」など。
- 非常に多い: 71匹以上、「大量」「爆寄り」など。
- 不明: 湧き量が不明、または天気や質問など無関係なコメント。

**ルール**:
- 数値がある場合は最優先（例: 「50匹」→ 多い）。
- 否定表現（「いない」「ゼロ」）は「なし」。
- 質問や天気情報は「不明」。
- 「堪能」「忙しい」は「多い」以上を示唆。
- 例:
  - 「2時間で2匹」→ 少ない
  - 「60匹獲れた」→ 多い
  - 「イカなし、人なし」→ なし
  - 「波が高い、様子見」→ 不明
  - 「大量発生してました」→ 非常に多い

**コメント**: {text}

**分類**:
```json
{{
    "surge_level": "なし|少ない|普通|多い|非常に多い|不明",
    "reason": "分類の理由"
}}
"""
    for attempt in range(3):  # 最大3回リトライ
        try:
            response = client.chat.completions.create(
                model="grok-3",
                messages=[
                    {"role": "system", "content": "正確な分類を行うアシスタント"},
                    {"role": "user", "content": prompt.format(text=text)}
                ],
                max_tokens=200,
                temperature=0.3
            )
            result = json.loads(response.choices[0].message.content)
            surge_level = result["surge_level"]
            reason = result["reason"]
            logger.info(f"API分類結果: {surge_level}（理由: {reason}）")

            # シンプル分類と比較
            if simple_result != surge_level and simple_result != "不明" and surge_level != "不明":
                logger.warning(f"分類不一致: API={surge_level}, シンプル={simple_result}, テキスト={text[:50]}...")

            return surge_level
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.error(f"API試行{attempt+1}失敗: {e}")
            logger.error(f"APIレスポンス: {response.choices[0].message.content if 'response' in locals() else 'なし'}")
            if attempt == 2:
                logger.info("API失敗、シンプル分類を使用")
                return simple_result
            time.sleep(1)  # リトライ前に待機

def get_pagination_urls(base_url):
    """ページネーションURLを動的に取得"""
    urls = [base_url]
    try:
        logger.info(f"ベースURL({base_url})にアクセス中...")
        response = requests.get(base_url, headers=headers, timeout=10)
        logger.info(f"ステータスコード: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"アクセス失敗: {response.status_code}")
            return urls

        soup = BeautifulSoup(response.text, "html.parser")
        pagination_div = soup.find("div", style="font-size:13px;margin-bottom:4px;")

        if pagination_div:
            logger.info("ページネーション要素を検出しました")
            page_links = pagination_div.find_all("a", class_="n")
            for link in page_links:
                if link.text.strip().isdigit():
                    href = link.get('href')
                    absolute_url = urljoin(base_url, href)
                    if absolute_url not in urls:
                        logger.info(f"ページ{link.text}のURLを追加: {absolute_url}")
                        urls.append(absolute_url)
        else:
            logger.warning("ページネーション要素が見つかりません")

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
        return validated_urls
    except requests.RequestException as e:
        logger.error(f"ページネーション取得エラー: {e}")
        return urls

def post_process_classification(df):
    """分類結果の後処理で誤分類を修正"""
    logger.info("分類結果の後処理を開始")
    for idx, row in df.iterrows():
        text = row['comment']
        current_label = row['surge_level']
        numbers = extract_numbers(text)
        if numbers and current_label == "不明":
            max_num = max(numbers)
            if max_num == 0:
                df.at[idx, 'surge_level'] = "なし"
                logger.info(f"後処理: {text[:50]}... -> なし (数値: {max_num})")
            elif max_num <= 10:
                df.at[idx, 'surge_level'] = "少ない"
                logger.info(f"後処理: {text[:50]}... -> 少ない (数値: {max_num})")
            elif max_num <= 30:
                df.at[idx, 'surge_level'] = "普通"
                logger.info(f"後処理: {text[:50]}... -> 普通 (数値: {max_num})")
            elif max_num <= 70:
                df.at[idx, 'surge_level'] = "多い"
                logger.info(f"後処理: {text[:50]}... -> 多い (数値: {max_num})")
            else:
                df.at[idx, 'surge_level'] = "非常に多い"
                logger.info(f"後処理: {text[:50]}... -> 非常に多い (数値: {max_num})")
        if re.search(r'(?:大量|爆寄り|イカだらけ)', text, re.IGNORECASE) and current_label not in ["多い", "非常に多い"]:
            df.at[idx, 'surge_level'] = "非常に多い"
            logger.info(f"後処理: {text[:50]}... -> 非常に多い (大量表現)")
        if re.search(r'(?:ゼロ|いない|なし|居ない)', text, re.IGNORECASE) and current_label != "なし":
            df.at[idx, 'surge_level'] = "なし"
            logger.info(f"後処理: {text[:50]}... -> なし (否定表現)")
    return df

def main():
    logger.info("ホタルイカ湧き量データスクレイピング開始")

    logger.info(f"作業ディレクトリ: {os.getcwd()}")

    all_data = []
    urls_to_scrape = get_pagination_urls(base_url)

    for i, url in enumerate(urls_to_scrape):
        logger.info(f"\n[{i+1}/{len(urls_to_scrape)}] URLを処理中: {url}")

        try:
            response = requests.get(url, headers=headers, timeout=15)
            logger.info(f"ステータスコード: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"アクセス失敗: {response.status_code}")
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            comments = soup.find_all("table", class_="layer")
            logger.info(f"取得したコメント数: {len(comments)}")

            if not comments:
                logger.warning("コメントが見つかりません。HTML構造を確認します...")
                html_preview = soup.prettify()[:1000]
                logger.debug(f"HTML preview: {html_preview}")
                continue

            for comment_idx, comment in enumerate(comments):
                try:
                    logger.info(f"\nコメント {comment_idx+1}/{len(comments)} 処理中...")

                    date_elem = comment.find("div", style=lambda s: s and "float:left" in s)
                    if date_elem:
                        date_text = date_elem.text
                        date_match = re.search(r"投稿日: (\d{4}年\d{2}月\d{2}日 \d{2}:\d{2})", date_text)
                        date = date_match.group(1) if date_match else "不明"
                    else:
                        date = "不明"

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

            logger.info(f"\n現在の総データ数: {len(all_data)}")

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

        if i < len(urls_to_scrape) - 1:
            wait_time = 3
            logger.info(f"{wait_time}秒待機中...")
            time.sleep(wait_time)

    df = pd.DataFrame(all_data)
    logger.info(f"\n最終データ数: {len(df)}")

    if len(df) == 0:
        logger.warning("警告: データが空です。HTML構造が変更されているか、アクセスが制限されている可能性があります。")
    elif len(df) <= 50:
        logger.warning("警告: データが少ないです（50件以下）。追加ページが存在しないか、処理に問題がある可能性があります。")

    # 分類結果の後処理
    df = post_process_classification(df)

    final_filename = "hotaruika_surge_data.csv"
    df.to_csv(final_filename, index=False, encoding="utf-8-sig")
    logger.info(f"最終データを保存しました: {final_filename} (総レコード数: {len(df)})")

    if not df.empty:
        classification_stats = df["surge_level"].value_counts()
        logger.info(f"\n分類結果の統計:\n{classification_stats}")

if __name__ == "__main__":
    main()