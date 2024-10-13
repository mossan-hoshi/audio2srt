from faster_whisper import WhisperModel
from pathlib import Path
import srt
import re
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


def transcribe_audio(
    model: WhisperModel, audio_file_path: Path, language: str = None
) -> List[Word]:
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

    for term_to_search, term_to_replace in term_replace_dict.items():
        # 正規表現をコンパイル
        pattern = re.compile(term_to_search)

        # 文字列全体で置換を実行し、置換の詳細情報を取得
        def replacement(match):
            # 置換後の文字列を取得（キャプチャグループを反映）
            replaced_text = match.expand(term_to_replace)
            # 置換位置を取得
            start_idx = match.start()
            end_idx = match.end()

            # 影響を受ける単語のインデックスを特定
            affected_word_indices = [
                i
                for i, word_start in enumerate(word_start_indices)
                if (word_start < end_idx and word_start + len(words[i].word) > start_idx)
            ]

            # 単語リストを更新
            replace_term(
                words,
                word_start_indices,
                replaced_text,
                start_idx,
                end_idx,
                affected_word_indices,
            )

            # 置換による長さの変化を計算
            length_diff = len(replaced_text) - (end_idx - start_idx)
            # 単語開始インデックスを更新
            for i in range(len(word_start_indices)):
                if word_start_indices[i] > start_idx:
                    word_start_indices[i] += length_diff

            return replaced_text

        # 置換を実行
        concat_str = pattern.sub(replacement, concat_str)

    return words


def replace_term(
    words,
    word_start_indices,
    replaced_text,
    found_index,
    end_index,
    affected_word_indices,
):
    """単語リスト内の特定の用語を置換する補助関数"""
    # 置換後のテキストを文字リストに変換
    replaced_chars = list(replaced_text)
    replaced_char_index = 0

    for i in affected_word_indices:
        word = words[i]
        word_start = word_start_indices[i]
        word_end = word_start + len(word.word)

        # 単語内の置換範囲を計算
        overlap_start = max(word_start, found_index)
        overlap_end = min(word_end, end_index)

        # 単語内の相対的な開始・終了位置
        relative_start = overlap_start - word_start
        relative_end = overlap_end - word_start

        # 置換後の文字を該当部分に反映
        num_chars_to_replace = relative_end - relative_start
        chars_to_insert = replaced_chars[
            replaced_char_index : replaced_char_index + num_chars_to_replace
        ]

        word.word = (
            word.word[:relative_start] + "".join(chars_to_insert) + word.word[relative_end:]
        )

        replaced_char_index += num_chars_to_replace


def create_subtitle(index: int, start: float, end: float, content: str) -> srt.Subtitle:
    """字幕オブジェクトを作成する"""
    return srt.Subtitle(
        index=index,
        start=timedelta(seconds=start),
        end=timedelta(seconds=end),
        content=content.strip(),
    )


def split_into_lines(text: str, max_line_length: int) -> List[str]:
    """テキストを行に分割し、英単語が行をまたがないようにする"""
    lines = []
    index = 0
    text_length = len(text)
    while index < text_length:
        current_line = ''
        line_start_index = index
        while index < text_length and len(current_line) < max_line_length:
            current_char = text[index]
            current_line += current_char
            index += 1

        # 英単語が行末で分割されていないか確認
        if index < text_length and text[index - 1].isalpha():
            # 次の文字もアルファベットなら、英単語が分割されている可能性あり
            if text[index].isalpha():
                # 英単語の先頭を探すためにバックトラック
                back_index = index - 1
                while back_index >= line_start_index and text[back_index].isalpha():
                    back_index -= 1
                # 行の先頭で英単語が始まっている場合はそのまま
                if back_index >= line_start_index:
                    # 単語を次の行に移動
                    shift = back_index - line_start_index + 1
                    current_line = current_line[:shift]
                    index = back_index + 1
        lines.append(current_line)
    return lines


def generate_srt_segments(
    words: List[Word],
    char_num: int = 48,
    max_line_str_num: int = 24,
    max_lines_per_segment: int = 2,
    gap_seconds_threshold: int = 3,
) -> List[srt.Subtitle]:
    """単語リストからSRT形式の字幕セグメントを生成し、英単語が行をまたがないようにする"""
    srt_segments = []
    current_segment = ""
    segment_start = None
    segment_end = None
    segment_index = 1

    for word in words:
        if segment_start is None:
            segment_start = word.start

        # Create potential new segment content
        potential_segment = current_segment + word.word
        potential_lines = split_into_lines(potential_segment.strip(), max_line_str_num)
        if (
            word.start - (segment_end if segment_end is not None else segment_start) > gap_seconds_threshold
            or len(potential_segment) > char_num
            or len(potential_lines) > max_lines_per_segment
        ):
            # 現在のセグメントを行に分割
            content_lines = split_into_lines(current_segment.strip(), max_line_str_num)
            content = '\n'.join(content_lines)
            srt_segments.append(
                create_subtitle(segment_index, segment_start, segment_end, content)
            )
            segment_index += 1
            current_segment = word.word
            segment_start = word.start
        else:
            current_segment = potential_segment

        segment_end = word.end

        print(f"  [{word.start:.2f}s -> {word.end:.2f}s] {word.word}")

    if current_segment:
        content_lines = split_into_lines(current_segment.strip(), max_line_str_num)
        content = '\n'.join(content_lines)
        srt_segments.append(
            create_subtitle(segment_index, segment_start, segment_end, content)
        )

    return srt_segments


def main(
    audio_file_path: Path,
    model: WhisperModel,
    term_replace_dict: Dict[str, str],
    char_num: int = 48,
    max_line_str_num: int = 24,
    max_lines_per_segment: int = 2,
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
        words,
        char_num,
        max_line_str_num,
        max_lines_per_segment,
        gap_seconds_threshold,
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
        help="1セグメントあたりの最大文字数 (デフォルト: 48)",
    )
    parser.add_argument(
        "--max_line_str_num",
        type=int,
        default=24,
        help="1行の最大文字数 (デフォルト: 24)",
    )
    parser.add_argument(
        "--max_lines_per_segment",
        type=int,
        default=2,
        help="セグメントの最大行数 (デフォルト: 2)",
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

    term_replace_dict = load_replace_terms(args.replace_terms_path)

    main(
        audio_file_path,
        model,
        term_replace_dict,
        args.char_num,
        args.max_line_str_num,
        args.max_lines_per_segment,
        args.gap_seconds_threshold,
        language=args.language,
    )