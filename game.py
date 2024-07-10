import instaloader
import pandas as pd
import os
import re
import json
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import glob
import cv2
import numpy as np

# Initialize Instaloader
ig = instaloader.Instaloader(
    download_pictures=True,
    download_videos=True,
    download_video_thumbnails=False,
    download_geotags=False,
    download_comments=False,
    save_metadata=False,
    compress_json=False,
    post_metadata_txt_pattern=''
)

def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def find_media_file(directory, timestamp, is_video):
    extension = "mp4" if is_video else "jpg"
    pattern = os.path.join(directory, f"{timestamp}_UTC*.{extension}")
    files = glob.glob(pattern)
    if files:
        return files[0]  # Return the first matching file
    return None

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

# Setup Whisper model
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

# Setup BLIP model
def setup_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def transcribe_video(pipe, video_path):
    try:
        result = pipe(video_path, generate_kwargs={"language": "english"})
        return result["text"]
    except Exception as e:
        print(f"Error transcribing video {video_path}: {str(e)}")
        return ""

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
        print(f"Error captioning image: {str(e)}")
        return ""
    
def load_image_captions(json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}

def save_image_captions(json_path, captions):
    with open(json_path, 'w') as f:
        json.dump(captions, f, indent=2)

def main():
    # Input the Instagram username
    dp = input("Username: ")

    os.makedirs(dp, exist_ok=True)
    ig.dirname_pattern = dp

    # Load the profile
    try:
        profile = instaloader.Profile.from_username(ig.context, dp)
    except instaloader.exceptions.ProfileNotExistsException:
        print(f"The profile '{dp}' does not exist.")
        exit()

    # Setup Whisper and BLIP models
    print("Setting up Whisper and BLIP models...")
    whisper_pipe = setup_whisper()
    blip_processor, blip_model = setup_blip()

    json_path = os.path.join(dp, f'{dp}_image_captions.json')
    image_captions = load_image_captions(json_path)

    # Prepare lists to store data
    post_details = []

    for index, post in enumerate(tqdm(profile.get_posts(), desc="Processing posts")):
        try:
            likes = post.likes
            plays = post.video_view_count if post.is_video else None
            description = post.caption
            shortcode = post.shortcode
            timestamp = post.date_utc.strftime("%Y-%m-%d_%H-%M-%S")

            if post.is_video:
                print(f"Downloading video from post {index + 1} with {likes} likes")
                ig.download_post(post, target=dp)
                
                video_path = find_media_file(dp, timestamp, is_video=True)
                
                if video_path:
                    print(f"Found video file: {video_path}")
                    transcription = transcribe_video(whisper_pipe, video_path)
                    print("Analyzing video frames...")
                    frame_analysis = analyze_video_frames(blip_processor, blip_model, video_path)
                    image_caption = frame_analysis
                    print("Video frame analysis completed.")
                else:
                    print(f"Video file not found for post {index + 1}")
                    transcription = ""
                    image_caption = ""
            else:
                print(f"Downloading image from post {index + 1} with {likes} likes")
                ig.download_post(post, target=dp)
                
                image_path = find_media_file(dp, timestamp, is_video=False)
                
                if image_path:
                    print(f"Found image file: {image_path}")
                    if image_path in image_captions:
                        image_caption = image_captions[image_path]
                    else:
                        image_caption = caption_image(blip_processor, blip_model, image_path)
                        image_captions[image_path] = image_caption
                        save_image_captions(json_path, image_captions)
                    transcription = ""
                else:
                    print(f"Image file not found for post {index + 1}")
                    image_caption = ""
                    transcription = ""

            post_details.append({
                "index": index + 1,
                "shortcode": shortcode,
                "timestamp": timestamp,
                "likes": likes,
                "plays": plays,
                "description": description,
                "video_transcription": transcription,
                "image_caption": image_caption
            })
            
        except Exception as e:
            print(f"Error processing post: {e}")

    # Final save of image captions
    save_image_captions(json_path, image_captions)

    # Convert the list to a DataFrame
    df = pd.DataFrame(post_details)

    # Save the DataFrame to an Excel file in the same directory
    excel_path = os.path.join(dp, f'{dp}_post_details.xlsx')
    df.to_excel(excel_path, index=False, engine='openpyxl')

    print(f"Post details for {dp} saved to {excel_path}")
    print(f"Image captions saved to {json_path}")

if __name__ == "__main__":
    main()