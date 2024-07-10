import asyncio
import json
import os
import shutil
from typing import Dict, List
from loguru import logger as log
from scrapfly import ScrapeConfig, ScrapflyClient, ScrapeApiResponse
import pyktok as pyk
import time
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import cv2
from PIL import Image

SCRAPFLY = ScrapflyClient(key="scp-live-84ec83b142f94af8ba60ab0395960901")

js_scroll_function = """
function scrollToEnd(i) {
    if (window.innerHeight + window.scrollY >= document.body.scrollHeight) {
        console.log("Reached the bottom.");
        return;
    }
    window.scrollTo(0, document.body.scrollHeight);
    if (i < 15) {
        setTimeout(() => scrollToEnd(i + 1), 3000);
    } else {
        console.log("Reached the end of iterations.");
    }
}
scrollToEnd(0);
"""

def parse_channel(response: ScrapeApiResponse):
    xhr_calls = response.scrape_result["browser_data"]["xhr_call"]
    post_calls = [c for c in xhr_calls if "/api/post/item_list/" in c["url"]]
    post_data = []
    for post_call in post_calls:
        try:
            data = json.loads(post_call["response"]["body"])["itemList"]
            post_data.extend(data)
        except Exception as e:
            log.error(f"Post data couldn't load: {e}")
    parsed_data = []
    for post in post_data:
        result = {
            'createTime': post.get('createTime'),
            'desc': post.get('desc'),
            'id': post.get('id'),
            'stats': post.get('stats'),
            'video': post.get('video')
        }
        parsed_data.append(result)
    return parsed_data

def setup_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def caption_image(processor, model, image):
    try:
        if isinstance(image, str):  # If image is a file path
            raw_image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):  # If image is already a PIL Image
            raw_image = image.convert('RGB')
        else:
            raise ValueError("Input must be either a file path or a PIL Image object")

        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        log.error(f"Error captioning image: {str(e)}")
        return ""

def extract_frames(video_path, interval=2):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    
    success, frame = cap.read()
    frame_count = 0
    while success:
        if frame_count % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append(pil_image)
        frame_count += 1
        success, frame = cap.read()
    
    cap.release()
    return frames

def analyze_video_frames(blip_processor, blip_model, video_path, interval=2):
    frames = extract_frames(video_path, interval)
    frame_descriptions = []
    
    for i, frame in enumerate(frames):
        caption = caption_image(blip_processor, blip_model, frame)
        frame_descriptions.append(f"Frame {i+1}: {caption}")
    
    return " ".join(frame_descriptions)

async def scrape_channel(url: str) -> List[Dict]:
    log.info(f"Scraping channel page with the URL {url} for post data")
    response = await SCRAPFLY.async_scrape(ScrapeConfig(
        url, asp=True, country="GB", render_js=True, rendering_wait=2000, js=js_scroll_function
    ))
    data = parse_channel(response)
    log.success(f"Scraped {len(data)} posts data")
    return data

def extract_video_ids(data):
    return [item['id'] for item in data if 'id' in item]

def download_tiktok_content(username, content_ids, output_dir, limit=None):
    pyk.specify_browser('chrome')  # or 'firefox' if you prefer
    log.info("Browser specified. Starting downloads.")
    
    downloaded_files = []
    for i, content_id in enumerate(content_ids[:limit]):
        if limit and i >= limit:
            break
        
        video_url = f"https://www.tiktok.com/@{username}/video/{content_id}"
        photo_url = f"https://www.tiktok.com/@{username}/photo/{content_id}"
        
        retries = 2
        content_type = "video"
        current_url = video_url
        
        while retries > 0:
            try:
                log.info(f"Attempting to download {content_type} {i+1}: {content_id}")
                pyk.save_tiktok(current_url, True, os.path.join(output_dir, f"{username}_content.csv"), 'chrome')
                log.success(f"Downloaded {content_type} {i+1}: {content_id}")
                
                # Find the downloaded file and move it to the output directory
                for file in os.listdir():
                    if file.endswith(".mp4") and content_id in file:
                        new_filename = f"{content_id}.mp4"
                        shutil.move(file, os.path.join(output_dir, new_filename))
                        downloaded_files.append(new_filename)
                        break
                
                break
            except Exception as e:
                log.error(f"Error downloading {content_type} {content_id}: {str(e)}")
                retries -= 1
                if retries > 0 and content_type == "video":
                    log.info("Switching to photo URL")
                    content_type = "photo"
                    current_url = photo_url
                elif retries > 0:
                    log.info(f"Retrying in 5 seconds... ({retries} attempts left)")
                    time.sleep(5)
                else:
                    log.error(f"Failed to download content {content_id} after all attempts")
        
        time.sleep(2)
    
    return downloaded_files

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
async def main(username):
    output_dir = f"{username}_content"
    os.makedirs(output_dir, exist_ok=True)

    # Scrape channel
    channel_data = await scrape_channel(f"https://www.tiktok.com/@{username}")
    
    # Save channel data to JSON file
    json_file_path = os.path.join(output_dir, f"{username}_channel_data.json")
    with open(json_file_path, "w", encoding="utf-8") as file:
        json.dump(channel_data, file, indent=2, ensure_ascii=False)
    
    # Extract content IDs
    content_ids = extract_video_ids(channel_data)
    
    log.info(f"Total content IDs extracted: {len(content_ids)}")
    
    # Download content
    downloaded_files = download_tiktok_content(username, content_ids, output_dir)

    # Setup Whisper and BLIP models
    log.info("Setting up Whisper and BLIP models...")
    whisper_pipe = setup_whisper()
    blip_processor, blip_model = setup_blip()

    # Transcribe and analyze videos
    log.info("Starting transcription and frame analysis process...")
    transcriptions = {}
    frame_analyses = {}
    for filename in tqdm(downloaded_files, desc="Processing videos"):
        video_path = os.path.join(output_dir, filename)
        video_id = os.path.splitext(filename)[0]
        
        if os.path.exists(video_path):
            transcription = transcribe_video(whisper_pipe, video_path)
            transcriptions[video_id] = transcription
            
            log.info(f"Analyzing frames for video: {filename}")
            frame_analysis = analyze_video_frames(blip_processor, blip_model, video_path)
            frame_analyses[video_id] = frame_analysis
        else:
            log.warning(f"File not found: {video_path}")

    # Create Excel file from JSON data, transcriptions, and frame analyses
    excel_data = []
    for item in channel_data:
        video_id = item.get('id')
        row = {
            'ID': video_id,
            'Link': f"https://www.tiktok.com/@{username}/video/{video_id}",
            'Description': item.get('desc'),
            'Create Time': item.get('createTime'),
            'Likes': item.get('stats', {}).get('diggCount'),
            'Comments': item.get('stats', {}).get('commentCount'),
            'Shares': item.get('stats', {}).get('shareCount'),
            'Views': item.get('stats', {}).get('playCount'),
            'Duration': item.get('video', {}).get('duration'),
            'Transcription': transcriptions.get(video_id, ""),
            'Frame Analysis': frame_analyses.get(video_id, "")
        }
        excel_data.append(row)
    
    df = pd.DataFrame(excel_data)
    excel_file_path = os.path.join(output_dir, f"{username}_tiktok_data.xlsx")
    df.to_excel(excel_file_path, index=False, engine='openpyxl')
    log.success(f"Excel file created: {excel_file_path}")

if __name__ == "__main__":
    username = "essentialdetailingcarmel"  # Replace with the desired username
    asyncio.run(main(username))