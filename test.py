from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

model_id = "openai/whisper-small"
cache_dir = "./models/whisper-small"

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

result = pipe("Alec Benjamin - Let Me Down Slowly (Lyrics)(MP3_160K).mp3")
print(result)
