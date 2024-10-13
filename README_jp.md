faster-whisperを使って。Whisper-Large-V3で音声文字起こしをして。事前に指定した用語変換やフィラーワードの削除等をした上でセグメントをいい感じに再構築してSRTファイルとして出力するPythonスクリプトを紹介します

【[解説記事](https://zenn.dev/mossan_hoshi/articles/20241011_faster_whisper_srt) 】

## 目次
1. はじめに
2. 環境構築・使用方法
3. 各処理の解説
   1. 用語辞書の読み込み、モデルの初期化
   2. 文字起こし( `transcribe_audio() ` )
   3. 用語の置換処理( `replace_terms()` )
   4. srt化( `generate_srt_segments()` ) & 保存
4. 結論

## 1 はじめに
### 背景
Youtubeの字幕用にWhisper V3を使おうと思ったのですが、デフォルトのWhisperの出力だと「フィラーワード（ `あー` や `ええと` など ）を含む」「セグメントの粒度が大きい(画面に納まりきらない)」「専門用語を間違える」等が発生し使い勝手が悪かったので結果に対する後処理でこれらを修正できるようにしました

### 処理内容
以下の手順で処理を行います
1. Whisperによる文字起こし実施
2. 用語置換(フィラーワード除去を含む)
    - CSVで置換辞書を定義しておきます 
    - wordをまたぐ場合にも対応しています
3. srt化＆ファイル保存
    - 指定した文字数に納まるようセグメントを区切りなおします

## 2 環境構築・使用方法

1. **Pythonのインストール**: 今回は3.11.3を使いましたが、faster_whisperが使えればバージョンは問いません
- poetry を居r手置いてください `pip install poetry`

2. リポジトリンクローン

https://github.com/mossan-hoshi/audio2srt

```bash
gh repo clone mossan-hoshi/audio2srt
```

or

```bash
git clone https://github.com/mossan-hoshi/audio2srt.git
```

1. **仮想環境の作成**

- Window

```bash
poetry install
```

1. **CSVファイルの準備**: 用語設定のためのCSVファイルreplace_terms.csvを必要に応じて編集します。以下のフォーマットで作成してください。

- 1列目：検出文字列
- 2列目：置換後文字列（削除したい場合は空を指定）

[用語設定CSVファイルサンプル]

```csv
ブレンダー,Blender
ちょっと,
はいえーと,
あー,
えーと,
はい,
うん,
なんか,
ですね,
普通に,
ご視聴ありがとうございました,
サイクル図,Cycles
EV,EeVee
スクラプティング,スカルプティング
チャットGPD,ChatGPT
```


6. **スクリプトの実行**: 最後に、以下のコマンドを実行してスクリプトを実行します。

```bash
usage: whisperv3.py [-h] [--model MODEL] [--device DEVICE]
                    [--compute_type COMPUTE_TYPE]
                    [--language LANGUAGE]
                    [--replace_terms_path REPLACE_TERMS_PATH]        
                    [--char_num CHAR_NUM]
                    [--max_line_str_num MAX_LINE_STR_NUM]
                    [--gap_seconds_threshold GAP_SECONDS_THRESHOLD]  
                    audio_file
```

| オプション名                        | 内容                                         | デフォルト値               |
|-----------------------------------|--------------------------------------------|--------------------------|
| audio_file                        | 文字起こしする音声ファイルのパス              |                          |
| --model MODEL                     | 使用するWhisperモデル                        | large-v3                 |
| --device DEVICE                   | 計算に使用するデバイス                      | cuda                     |
| --compute_type COMPUTE_TYPE       | モデルの計算タイプ                          | float16                  |
| --language LANGUAGE               | 文字起こしに使用する言語                    | 自動検出                 |
| --replace_terms_path REPLACE_TERMS_PATH | 置換用語のCSVファイルパス                  | ./replace_terms.csv      |
| --char_num CHAR_NUM               | 1行あたりの最大文字数                      | 48                       |
| --max_line_str_num MAX_LINE_STR_NUM | 最大行数                                   | 24                       |
| --gap_seconds_threshold GAP_SECONDS_THRESHOLD | セグメント間の最大間隔秒数                    | 3                      |
