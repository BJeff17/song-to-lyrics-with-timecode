from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import pydub
import multiprocessing

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
result = pipe(
    "Alec Benjamin - Let Me Down Slowly (Lyrics)(MP3_160K).mp3",
    return_timestamps="word",
    chunk_length_s=30,
    stride_length_s=5,
    generate_kwargs={
        "task": "transcribe",
        "language": "en"
    }
)



def merge_timestamps(result, threshold=0.5):
    merged = []
    current_segment = None

    for segment in result['chunks']:
        for word_info in segment:
            word, start, end = word_info["text"], word_info["timestamp"][0], word_info["timestamp"][1]
            if current_segment is None:
                current_segment = {
                    'text': word,
                    'start': start,
                    'end': end
                }
            else:
                if start <= current_segment['end'] + threshold:  # 0.5 seconds gap threshold
                    current_segment['text'] += ' ' + word
                    current_segment['end'] = end
                else:
                    merged.append(current_segment)
                    current_segment = {
                        'text': word,
                        'start': start,
                        'end': end
                    }
    if current_segment is not None:
        merged.append(current_segment)

    return merged

def print_and_play_sync(result):
    merged_timestamps = merge_timestamps(result)

    for segment in merged_timestamps:
        text = segment['text']
        start = segment['start']
        end = segment['end']
        print(f"[{start:.2f} - {end:.2f}]: {text}")

        # Play the audio segment synchronously
        audio = pydub.AudioSegment.from_file("Alec Benjamin - Let Me Down Slowly (Lyrics)(MP3_160K).mp3")
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        segment_audio = audio[start_ms:end_ms]
        play_process = multiprocessing.Process(target=segment_audio.play)
        play_process.start()
        play_process.join()  
print(result)
print_and_play_sync(result)