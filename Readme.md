# MP3 to Text Transcription and Summarization

This project provides a Python script to transcribe speech from an MP3 file, and generate a summary using a locally hosted instance of the LLaMA model via an API. The transcription is performed using OpenAI's Whisper model, and summarization is done by calling a locally running Ollama API with the LLaMA 3.1 model so your data is safe with us.

## Features

- **MP3 to WAV conversion**: Converts the input MP3 file to WAV format using the `pydub` library and `ffmpeg`.
- **Whisper transcription**: Uses the Whisper model to transcribe the speech from the WAV file into text.
- **Summarization**: Sends the transcription to a locally hosted Ollama API using the LLaMA 3.1 model to generate a summary of the transcription.

## Prerequisites

- Python 3.7+
- Required Python libraries:
  - `whisper`
  - `pydub`
  - `requests`
  - `json`
  - `ffmpeg` (installed locally and path added to `AudioSegment.converter`)
- A locally running Ollama API with the LLaMA 3.1 model. Ensure the endpoint URL is correctly set in the script or you can use our model which is locally hosted

## Installation

1. Install the required Python libraries:
    ```bash
    pip install whisper pydub requests
    ```

2. Download and install `ffmpeg`:
    - Download from [ffmpeg official website](https://ffmpeg.org/download.html).
    - Extract the files and set the path to `ffmpeg` in the script:
      ```python
      AudioSegment.converter = ("<your-path-to-ffmpeg>/ffmpeg.exe")
      ```

3. Make sure you have the Ollama API running locally at `http://localhost:8080`. The script uses the following URL:
    ```python
    ollamaUrl = "http://localhost:8080/api/generate"
    ```
   or use :
   ```python
   ollamaUrl= "https://llm.cialabs.org/api/generate"
   ``` 

## How to Run

1. Run the script with an MP3 file path:
    ```bash
    python main.py
    ```

2. Enter the path of the MP3 file when prompted:
    ```
    enter the path of your audio (MP3) file:
    ```

3. The script will:
    - Convert the MP3 file to WAV format.
    - Transcribe the WAV file to text using Whisper.
    - Send the transcription text to Ollama for summarization.
    - Display the transcription and the generated summary.

## Customization

- **Whisper Model**: You can change the Whisper model by editing the line:
    ```python
    model = whisper.load_model("base")
    ```
  Replace `"base"` with `"tiny"`, `"small"`, `"medium"`, or `"large"` as needed.


## Error Handling

- If the MP3 file is not found, the script will output:
    ```
    MP3 file not found!
    ```

- If there's an error during transcription or summarization, the script will print the error message.

## Dependencies

- [Whisper](https://github.com/openai/whisper): For transcription.
- [PyDub](https://github.com/jiaaro/pydub): For audio file conversion.
- [ffmpeg](https://ffmpeg.org/): Required by PyDub for MP3 to WAV conversion.
- [Ollama](https://ollama.com/): For LLaMA-based summarization.


