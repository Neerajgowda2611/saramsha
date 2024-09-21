import whisper

def load_whisper():
    model = whisper.load_model("base", device="cpu")
    return model

def audio_to_text(audio_file):
    model = load_whisper()
    result = model.transcribe(audio_file)
    return result['text']
