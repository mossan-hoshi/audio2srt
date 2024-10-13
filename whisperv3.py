from faster_whisper import WhisperModel
from pathlib import Path
import srt
from datetime import timedelta
from typing import Dict, List
import csv
from dataclasses import dataclass


@dataclass
class Word:
    word: str
    start: float
    end: float


def load_replace_terms(file_path: str = "./replace_terms.csv") -> Dict[str, str]:
    """置換用語をCSVファイルから読み込む"""
    replace_terms_dict = {}
    try:
        with open(file_path, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            replace_terms_dict = {row[0]: row[1] for row in reader if len(row) >= 2}
    except FileNotFoundError:
        print(f"警告: {file_path} が見つかりません。空の置換辞書を使用します。")
    return replace_terms_dict


DEFAULT_REPLACE_TERMS = load_replace_terms()


def transcribe_audio(model: WhisperModel, audio_file_path: Path, language: str = None) -> List[Word]:
    """音声ファイルを文字起こしし、単語リストを返す"""
    transcribe_args = {
        "audio": audio_file_path,
        "beam_size": 5,
        "word_timestamps": True,
    }
    if language:
        transcribe_args["language"] = language

    segments, info = model.transcribe(**transcribe_args)
    print(f"検出された言語: '{info.language}' (確率: {info.language_probability})")
    return [
        Word(word=word.word, start=word.start, end=word.end)
        for segment in segments
        for word in segment.words
    ]


def replace_terms(
    words: List[Word], term_replace_dict: Dict[str, str] = {}
) -> List[Word]:
    """単語リスト内の特定の用語を置換する"""
    # 全ての単語を連結した文字列を作成
    concat_str = "".join(word.word for word in words)
    # 各単語の開始インデックスを計算
    word_start_indices = [
        sum(len(words[i].word) for i in range(j)) for j in range(len(words))
    ]

    # 置換辞書の各エントリに対して処理を行う
    for term_to_search, term_to_replace in term_replace_dict.items():
        start_index = 0
        while True:
            # 検索語を連結文字列内で探す
            found_index = concat_str.find(term_to_search, start_index)
            if found_index == -1:
                break

            # 検索語の終了インデックスを計算
            end_index = found_index + len(term_to_search)
            # 影響を受ける単語のインデックスを特定
            affected_word_indices = [
                i
                for i, word_start in enumerate(word_start_indices)
                if (word_start <= found_index < word_start + len(words[i].word))
                or (word_start < end_index <= word_start + len(words[i].word))
                or (found_index <= word_start < end_index)
            ]

            # 置換処理を実行
            replace_term(words, word_start_indices, term_to_replace, found_index, end_index, affected_word_indices)

            # 連結文字列を更新
            concat_str = (
                concat_str[:found_index] + term_to_replace + concat_str[end_index:]
            )
            # 単語の開始インデックスを調整
            length_diff = len(term_to_replace) - (end_index - found_index)
            word_start_indices = [
                index + length_diff if index > found_index else index
                for index in word_start_indices
            ]
            # 次の検索開始位置を設定
            start_index = found_index + len(term_to_replace)

    return words

def replace_term(words, word_start_indices, replace_term, found_index, end_index, affected_word_indices):
    """単語リスト内の特定の用語を置換する補助関数"""
    for idx, i in enumerate(affected_word_indices):
        word = words[i]
        word_start = word_start_indices[i]
        # 置換対象の相対的な開始位置を計算
        relative_start = max(0, found_index - word_start)
        # 置換対象の相対的な終了位置を計算
        relative_end = min(len(word.word), end_index - word_start)

        if idx == 0:
            # 最初の影響を受ける単語の場合、置換を行う
            word.word = (
                        word.word[:relative_start]
                        + replace_term
                        + word.word[relative_end:]
                    )
        else:
            # それ以外の影響を受ける単語の場合、該当部分を削除
            word.word = word.word[:relative_start] + word.word[relative_end:]


def create_subtitle(index: int, start: float, end: float, content: str) -> srt.Subtitle:
    """字幕オブジェクトを作成する"""
    return srt.Subtitle(
        index=index,
        start=timedelta(seconds=start),
        end=timedelta(seconds=end),
        content=content.strip(),
    )


def generate_srt_segments(
    words: List[Word],
    char_num: int = 48,
    max_line_str_num: int = 24,
    gap_seconds_threshold: int = 3,
) -> List[srt.Subtitle]:
    """単語リストからSRT形式の字幕セグメントを生成する"""
    srt_segments = []  # 生成されたSRT字幕セグメントのリスト
    current_segment = ""  # 現在処理中の字幕セグメントのテキスト
    segment_start = None  # 現在の字幕セグメントの開始時間
    segment_end = None  # 現在の字幕セグメントの終了時間
    segment_index = 1  # 字幕セグメントのインデックス

    for word in words:
        if segment_start is None:
            segment_start = word.start  # 最初の単語の場合、セグメント開始時間を設定
        elif (
            word.start - segment_end > gap_seconds_threshold
            or len(current_segment) + len(word.word) > char_num
        ):
            # 新しいセグメントを開始する条件：
            # 1. 前の単語との間隔が閾値を超える
            # 2. 現在のセグメントの文字数が上限を超える
            srt_segments.append(
                create_subtitle(
                    segment_index,
                    segment_start,
                    segment_end,
                    "\n".join(
                        [
                            current_segment[i : i + max_line_str_num]
                            for i in range(0, len(current_segment), max_line_str_num)
                        ]
                    ),
                )
            )
            segment_index += 1  # 次のセグメントのインデックスを増やす
            current_segment = ""  # 新しいセグメントのテキストをリセット
            segment_start = word.start  # 新しいセグメントの開始時間を設定

        current_segment += word.word  # 現在の単語をセグメントに追加
        segment_end = word.end  # セグメントの終了時間を更新

        print(f"  [{word.start:.2f}s -> {word.end:.2f}s] {word.word}")  # デバッグ用出力

    if current_segment:
        # 最後のセグメントを追加
        srt_segments.append(
            create_subtitle(segment_index, segment_start, segment_end, current_segment)
        )

    return srt_segments
def main(
    audio_file_path: Path,
    model: WhisperModel,
    term_replace_dict: Dict[str, str],
    char_num: int = 48,
    max_line_str_num: int = 24,
    gap_seconds_threshold: int = 3,
    language: str = None,
):
    """メイン処理関数"""
    # 音声ファイルを文字起こし
    words = transcribe_audio(model, audio_file_path, language)
    
    # 特定の用語を置換
    words = replace_terms(words, term_replace_dict)
    
    # SRT形式のセグメントを生成
    srt_segments = generate_srt_segments(
        words, char_num, max_line_str_num, gap_seconds_threshold
    )

    # SRTセグメントを文字列に変換
    srt_content = srt.compose(srt_segments)

    # SRTファイル保存
    output_file = audio_file_path.with_suffix(".srt")
    with open(output_file, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt_content)

    print(f"SRTファイルを保存しました: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Whisperモデルを使用して音声を文字起こしする"
    )
    parser.add_argument("audio_file", help="文字起こしする音声ファイルのパス")
    parser.add_argument(
        "--model",
        default="large-v3",
        help="使用するWhisperモデル (デフォルト: large-v3)",
    )
    parser.add_argument(
        "--device", default="cuda", help="計算に使用するデバイス (デフォルト: cuda)"
    )
    parser.add_argument(
        "--compute_type",
        default="float16",
        help="モデルの計算タイプ (デフォルト: float16)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="文字起こしに使用する言語 (デフォルト: 自動検出)",
    )
    parser.add_argument(
        "--replace_terms_path",
        default="./replace_terms.csv",
        help="置換用語のCSVファイルパス (デフォルト: ./replace_terms.csv)",
    )
    parser.add_argument(
        "--char_num",
        type=int,
        default=48,
        help="1行あたりの最大文字数 (デフォルト: 48)",
    )
    parser.add_argument(
        "--max_line_str_num",
        type=int,
        default=24,
        help="最大行数 (デフォルト: 24)",
    )
    parser.add_argument(
        "--gap_seconds_threshold",
        type=int,
        default=3,
        help="セグメント間の最大間隔 (デフォルト: 3秒)",
    )

    args = parser.parse_args()

    audio_file_path = Path(args.audio_file)

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    term_replace_dict = load_replace_terms(args.replace_terms)

    main(audio_file_path, model, term_replace_dict, args.char_num, args.max_line_str_num, args.gap_seconds_threshold, language=args.language)
