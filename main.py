import whisper
from pyannote.audio import Pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import numpy as np
from pyAudioAnalysis import audioSegmentation as aS
from sklearn.cluster import AgglomerativeClustering
from pydub import AudioSegment

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

def extract_features(wav_file_path):
    try:
        # Perform speaker diarization to get segments
        segments = aS.speaker_diarization(wav_file_path, n_speakers=10, mid_window=2.0, mid_step=0.5, lda_dim=35)
        
        # Extract features from segments
        features = np.array([seg[0] for seg in segments if isinstance(seg[0], (list, np.ndarray))])
        if features.shape[0] == 0:
            raise ValueError("No valid features extracted.")
        return features
    except Exception as e:
        print(f"Error during feature extraction: {str(e)}")
        return None

def estimate_number_of_speakers(features):
    try:
        # Perform clustering to estimate number of speakers
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0).fit(features)
        num_speakers = len(set(clustering.labels_))
        return num_speakers
    except Exception as e:
        print(f"Error during clustering: {str(e)}")
        return None

def diarizeAudio(wav_file_path, num_speakers):
    try:
        # Perform speaker diarization using the estimated number of speakers
        segments = aS.speaker_diarization(wav_file_path, n_speakers=num_speakers)
        
        # Initialize list to hold timestamps and speaker labels
        timestamps = []
        start_time = 0
        frame_duration = 0.2  # Each frame is ~0.2 seconds
        
        # Extract and format timestamps
        for i in range(1, len(segments)):
            if segments[i] != segments[i-1] or i == len(segments) - 1:
                end_time = i * frame_duration
                timestamps.append((start_time, end_time, f"Speaker {int(segments[i-1]) + 1}"))
                start_time = end_time
        
        return timestamps
    except Exception as e:
        print(f"Error during speaker diarization: {str(e)}")
        return None

# Function to print diarization results in a readable format
def printDiarization(timestamps):
    if timestamps:
        for start, end, speaker in timestamps:
            print(f"{speaker}: from {start:.2f}s to {end:.2f}s")
    else:
        print("No speaker segments detected.")


# Load PyAnnote for speaker diarization (optional)
def diarizeAudio(audio_file):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")
    diarization = pipeline(audio_file)
    return diarization


# Main function to process an MP3 file
def processMp3File(mp3_file_path):
    # Convert MP3 to WAV
    wav_file_path = "C:\\Users\\NEERAJ\\Desktop\\temp.wav"
    mp3ToWav(mp3_file_path, wav_file_path)

    # Step 1: Convert WAV to text using Whisper
    try:
        text = audioToText(wav_file_path)
        print("Transcription: ", text)
    except Exception as e:
        print(f"Error converting audio to text: {str(e)}")
        return

    # Speaker Diarization using PyAnnote (local)
    try:
        diarization = diarizeAudio(wav_file_path)
        print("Speaker Diarization: ", diarization)
    except Exception as e:
        print(f"Error in diarization: {str(e)}")

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
