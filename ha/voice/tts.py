import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

# NOTE:
# This version runs Piper *in-process* via its Python API (onnxruntime),
# not by shelling out to the `piper` binary.

def tts(final_text: str):

    if final_text:
        # Load Piper voice once (move this to module scope if reused)
        voice = PiperVoice.load("./voice/en_US-amy-medium.onnx")

        sample_rate = voice.config.sample_rate

        # Piper yields int16 PCM chunks
        with sd.OutputStream(
            samplerate=int(sample_rate),
            channels=1,
            dtype="int16",
            blocksize=4096,
        ) as stream:
            # Piper API changed across versions.
            # Newer versions expose `synthesize_stream`, older ones expose `synthesize`.
            # Handle both safely.

            stream_iter = (
                voice.synthesize_stream(final_text)
                if hasattr(voice, "synthesize_stream")
                else voice.synthesize(final_text)
            )

            def _chunk_to_pcm_bytes(chunk: object) -> bytes:
                """Normalize Piper AudioChunk to raw PCM bytes (int16 mono).

                In current piper-tts, `synthesize_stream()` yields `piper.voice.AudioChunk`.
                That object exposes **`.samples` as an int16 numpy array**.
                """
                if chunk is None:
                    return b""

                # Fast path: already bytes
                if isinstance(chunk, (bytes, bytearray, memoryview)):
                    return bytes(chunk)

                # Current Piper AudioChunk contract
                # chunk.samples: np.ndarray[int16]
                # Current Piper AudioChunk contract (observed):
                # chunk.audio_float_array: np.ndarray[float32] in [-1, 1]
                if hasattr(chunk, "audio_float_array"):
                    audio = chunk.audio_float_array
                    if isinstance(audio, np.ndarray):
                        pcm = np.clip(audio, -1.0, 1.0)
                        pcm = (pcm * 32767.0).astype(np.int16, copy=False)
                        return pcm.tobytes()
                # DEBUG: print actual chunk structure once to identify Piper API shape
                print("Piper chunk type:", type(chunk))
                print("Piper chunk attrs:", [a for a in dir(chunk) if not a.startswith('_')])
                raise TypeError(
                    "Unsupported Piper chunk type; expected bytes or AudioChunk with .samples"
                )

            for chunk in stream_iter:
                pcm_bytes = _chunk_to_pcm_bytes(chunk)
                if not pcm_bytes:
                    continue
                audio = np.frombuffer(pcm_bytes, dtype=np.int16)
                stream.write(audio)


if __name__ == "__main__":
    tts("Hello")
