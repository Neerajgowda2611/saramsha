import os
from audio_processing.mp3_to_wav import mp3_to_wav
from audio_processing.transcription import audio_to_text
from audio_processing.summarization import summarize_text

def process_mp3_file(mp3_file_path):
    # Convert MP3 to WAV
    wav_file_path = "temp.wav"
    mp3_to_wav(mp3_file_path, wav_file_path)

    try:
        text = audio_to_text(wav_file_path)
        print("Transcription: ", text)

        summary = summarize_text(text)
        print("Summary: ", summary)
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
    finally:
        # Clean up the temporary WAV file
        if os.path.exists(wav_file_path):
            os.remove(wav_file_path)

if __name__ == '__main__':
    mp3_file_path = '/home/silent/saramsha/backend/temp/aud.mp3'  # Replace with your actual path
    if os.path.exists(mp3_file_path):
        process_mp3_file(mp3_file_path)
    else:
        print("MP3 file not found!")
