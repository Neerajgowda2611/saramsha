import whisper
from pyannote.audio import Pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import numpy as np
from pyAudioAnalysis import audioSegmentation as aS
from sklearn.cluster import AgglomerativeClustering
from pydub import AudioSegment
import requests
import json

ollamaUrl="https://llm.cialabs.org/api/generate"
AudioSegment.converter= ("C:\\Users\\NEERAJ\\Desktop\\ffmpeg-master-latest-win64-gpl-shared\\ffmpeg-master-latest-win64-gpl-shared\\bin\\ffmpeg.exe")

# Function to convert MP3 to WAV
def mp3ToWav(mp3_file, wav_file):
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format="wav")
    
# Load Whisper model for audio-to-text conversion
def loadWhisper():
    model = whisper.load_model("base")  
    return model

# Convert audio to text using Whisper
def audioToText(audio_file):
    model = loadWhisper()
    result = model.transcribe(audio_file)
    return result['text']


# Load PyAnnote for speaker diarization (optional)
def diarizeAudio(audio_file):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")
    diarization = pipeline(audio_file)
    return diarization

def summarizeText(text):
    headers={
       "Content-Type":"application/json"
   }
    data={
       "model":"llama3.1",
       "prompt": f"summarize the following text : {text}"
   }
    try:
        response = requests.post(ollamaUrl, headers=headers, json=data, stream=True)
        summary = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    result = json.loads(line)
                    if result.get("done"):
                        break
                    summary += result.get("response", "")
                except ValueError as e:
                    print(f"Error during parsing: {str(e)}")
                    return None
        
        # If summary is not empty, return it
        if summary:
            return summary.strip()
        else:
            return "No summary generated"
    except Exception as e:
        print(f"Error during summarization: {str(e)}")
        return None
    

# Main function to process an MP3 file
def processMp3File(mp3_file_path):
    # Convert MP3 to WAV
    wav_file_path = "temp.wav"
    mp3ToWav(mp3_file_path, wav_file_path)

    # Step 1: Convert WAV to text using Whisper
    try:
        text = audioToText(wav_file_path)
        print("Transcription: ", text)
    except Exception as e:
        print(f"Error converting audio to text: {str(e)}")
        return

    # # Speaker Diarization using PyAnnote (local)
    # try:
    #     diarization = diarizeAudio(wav_file_path)
    #     print("Speaker Diarization: ", diarization)
    # except Exception as e:
    #     print(f"Error in diarization: {str(e)}")
    
    
    #Summarization of the text
    try:
        summary=summarizeText(text)
        if summary:
            print("Here's the SUMMARY !: ",summary)
    except Exception as e:
        print(f"error summarization : {str(e)}")
    

    # Clean up the temporary WAV file
    if os.path.exists(wav_file_path):
        os.remove(wav_file_path)

# Run the code with an MP3 file 
if __name__ == '__main__':
    mp3_file_path = r'C:\Users\NEERAJ\Desktop\saramsha\Saramshaa\aud.mp3'  # Replace with the path to your MP3 file

    # Ensure the file exists
    if os.path.exists(mp3_file_path):
        processMp3File(mp3_file_path)
    else:
        print("MP3 file not found!")
