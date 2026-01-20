import queue
import sys
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

def stt_vosk() -> str:
    """
    Blocking microphone capture.
    Speak one sentence and pause.
    Returns recognized text.
    """

    model = Model("stt/vosk-model-small-en-us-0.15")
    recognizer = KaldiRecognizer(model, 16000)

    audio_queue: queue.Queue[bytes] = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(bytes(indata))

    print("Listening...")

    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    print(f"Heard: {text}")
                    return text
