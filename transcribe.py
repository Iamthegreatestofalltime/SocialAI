import os
import json
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from loguru import logger as log
import time
from tqdm import tqdm

def setup_whisper():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-small"

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = WhisperProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device
    )

    return pipe

def transcribe_video(pipe, video_path):
    try:
        result = pipe(video_path, generate_kwargs={"language": "english"})
        return result["text"]
    except Exception as e:
        log.error(f"Error transcribing video {video_path}: {str(e)}")
        return ""

@torch.inference_mode()
def main(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    log.info("Setting up Whisper model...")
    pipe = setup_whisper()

    log.info("Starting transcription process...")
    transcriptions = {}
    video_files = [f for f in os.listdir(input_folder) if f.endswith((".mp4", ".mov", ".avi", ".mkv"))]

    for i, filename in enumerate(tqdm(video_files, desc="Transcribing videos")):
        video_path = os.path.join(input_folder, filename)
        video_id = os.path.splitext(filename)[0]

        log.info(f"Transcribing video {i+1}/{len(video_files)}: {filename}")
        start_time = time.time()

        transcription = transcribe_video(pipe, video_path)
        transcriptions[video_id] = transcription

        end_time = time.time()
        duration = end_time - start_time
        estimated_time_remaining = duration * (len(video_files) - i - 1)

        log.info(f"Completed in {duration:.2f} seconds. Estimated time remaining: {estimated_time_remaining:.2f} seconds")

    # Save transcriptions to a JSON file
    json_path = os.path.join(output_folder, "transcriptions.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=4)
    log.success(f"Transcriptions saved to {json_path}")

    # Create a DataFrame and save to Excel
    df = pd.DataFrame.from_dict(transcriptions, orient='index', columns=['Transcription'])
    df.index.name = 'Video ID'
    df.reset_index(inplace=True)
    excel_path = os.path.join(output_folder, "transcriptions.xlsx")
    df.to_excel(excel_path, index=False)
    log.success(f"Transcriptions saved to Excel: {excel_path}")

    log.success("Transcription process completed!")

if __name__ == "__main__":
    input_folder = "downloaded_videos"  # Folder containing the downloaded videos
    output_folder = "transcriptions"    # Folder to save transcriptions
    main(input_folder, output_folder)