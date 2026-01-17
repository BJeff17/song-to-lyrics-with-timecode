from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from pydub import AudioSegment
from pydub.silence import split_on_silence

model_id = "openai/whisper-tiny"
cache_dir = "./models/whisper-tiny"

import numpy as np

def audiosegment_to_numpy(audio):
    samples = np.array(audio.get_array_of_samples())

    # Convert stereo → mono if needed
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))
        samples = samples.mean(axis=1)

    return samples.astype(np.float32), audio.frame_rate


model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

processor = AutoProcessor.from_pretrained(
    model_id,
    cache_dir=cache_dir
)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if torch.cuda.is_available() else -1,
    return_timestamps=True
)

audio = AudioSegment.from_file("Alec Benjamin - Let Me Down Slowly (Lyrics)(MP3_160K).mp3")
audio_segments = split_on_silence(audio, min_silence_len=700, silence_thresh=-40)
audio_segments_np = [audiosegment_to_numpy(segment)[0] for segment in audio_segments]
result = pipe(audio_segments_np[0])
print(result)
