【[日本語](./README_jp.md) / [解説記事](https://zenn.dev/mossan_hoshi/articles/20241011_faster_whisper_srt) 】

This is a Python script that uses faster-whisper to transcribe audio usin

g Whisper-Large-V3. It performs specified term replacements and removes filler words before reconstructing segments nicely and outputting them as an SRT file.
## Table of Contents
1. Introduction
2. Environment Setup and Usage
3. Explanation of Each Process
   1. Loading the Term Dictionary and Initializing the Model
   2. Transcription (`transcribe_audio()`)
   3. Term Replacement (`replace_terms()`)
   4. SRT Generation (`generate_srt_segments()`) & Saving
4. Conclusion

## 1 Introduction
### Background
I wanted to use Whisper V3 for YouTube subtitles, but the default output from Whisper included "filler words" (like "uh" and "um"), had large segment sizes (which didn't fit on the screen), and misidentified technical terms. To improve usability, I implemented post-processing to correct these issues.

### Processing Steps
The processing is carried out in the following steps:
1. Perform transcription using Whisper.
2. Replace terms (including removing filler words).
    - A CSV file defines the replacement dictionary.
    - It also supports replacements that span multiple words.
3. Generate SRT and save the file.
    - Segments are restructured to fit within the specified character limit.

## 2 Environment Setup and Usage

1. **Python Installation**: We used version 3.11.3, but any version that supports faster_whisper will work.
- Please install poetry using `pip install poetry`.

2. Clone the repository:

https://github.com/mossan-hoshi/audio2srt

```bash
gh repo clone mossan-hoshi/audio2srt
```

or

```bash
git clone https://github.com/mossan-hoshi/audio2srt.git
```
# Start of Selection

1. **Creating a Virtual Environment**

- Window

```bash
poetry install
```
# Start of Selection

1. **Preparing the CSV File**: Edit the CSV file `replace_terms.csv` as needed for term settings. Please create it in the following format:

- Column 1: Detected string or regular expression pattern
- Column 2: Replacement string or regular expression pattern (specify empty if you want to delete)


[Sample Term Setting CSV File]

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

1. **Running the Script**: Finally, execute the following command to run the script.

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

| Option Name                        | Description                                  | Default Value            |
|-----------------------------------|----------------------------------------------|--------------------------|
| audio_file                        | Path to the audio file to be transcribed     |                          |
| --model MODEL                     | Whisper model to use                          | large-v3                 |
| --device DEVICE                   | Device to use for computation                 | cuda                     |
| --compute_type COMPUTE_TYPE       | Type of computation for the model             | float16                  |
| --language LANGUAGE               | Language to use for transcription              | auto-detect              |
| --replace_terms_path REPLACE_TERMS_PATH | Path to the CSV file for replacement terms   | ./replace_terms.csv      |
| --char_num CHAR_NUM               | Maximum number of characters per line         | 48                       |
| --max_line_str_num MAX_LINE_STR_NUM | Maximum number of lines                       | 24                       |
| --gap_seconds_threshold GAP_SECONDS_THRESHOLD | Maximum gap seconds between segments          | 3                       |
