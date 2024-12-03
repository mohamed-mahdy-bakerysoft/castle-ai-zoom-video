import os
from pathlib import Path
import subprocess
from typing import List
import cv2
import numpy as np
import math
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from face_bounding_box_detection import get_bounding_box
import logging

logging.basicConfig(level=logging.INFO)

class ZoomEffect:
    def __init__(self, start_time: float, end_time: float, zoom_in_duration: float,  scale: float, zoom_out_duration: float = 0, lag_time=None):
        self.start_time = start_time
        self.end_time = end_time
        if lag_time is not None:
            self.lag_time = lag_time   
        else:
            self.lag_time = end_time - start_time - zoom_in_duration - zoom_out_duration
        self.zoom_in_duration = zoom_in_duration
        self.zoom_out_duration = zoom_out_duration
        self.scale = scale
        self.total_duration = zoom_in_duration + zoom_out_duration + self.lag_time

    def get_scale_at_time(self, current_time: float) -> float:
        time_in_effect = current_time - self.start_time
        if 0 <= time_in_effect <= self.zoom_in_duration:
            progress = time_in_effect / self.zoom_in_duration
            return 1.0 + (self.scale - 1.0) * progress
        elif 0 <= time_in_effect - self.zoom_in_duration <= self.zoom_out_duration:
            time_in_zoom_out = time_in_effect - self.zoom_in_duration
            progress = time_in_zoom_out / self.zoom_out_duration
            return self.scale - (self.scale - 1.0) * progress
        return 1.0
    
    def get_scale_at_time_with_lag(self, current_time: float, scale=None) -> float:
        if scale is None:
            scale = self.scale
        time_in_effect = current_time - self.start_time
        if 0 <= time_in_effect <= self.zoom_in_duration:
            progress = time_in_effect / self.zoom_in_duration
            return 1.0 + (scale - 1.0) * progress
        elif self.zoom_in_duration <= time_in_effect <= self.lag_time + self.zoom_in_duration:
            return scale
        elif 0 <= time_in_effect - self.zoom_in_duration - self.lag_time <= self.zoom_out_duration:
            time_in_zoom_out = time_in_effect - self.zoom_in_duration - self.lag_time
            progress = time_in_zoom_out / self.zoom_out_duration
            return scale - (scale - 1.0) * progress
        return 1.0

def apply_zoom(frame: np.ndarray, scale: float, center_x: int = None, center_y: int=None) -> np.ndarray:
    if scale == 1.0:
        return frame
    height, width = frame.shape[:2]
    if center_x is None and center_y is None:
        center_x, center_y = width / 2, height / 2 -  (height / 4)
    M = np.float32([
        [scale, 0, center_x * (1 - scale)],
        [0, scale, center_y * (1 - scale)]
    ])
    return cv2.warpAffine(frame, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def extract_audio(input_video: str, output_audio: str):
    command = ['ffmpeg', '-i', input_video, '-vn', '-acodec', 'aac', '-y', output_audio]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logging.error("Audio extraction failed: %s", result.stderr.decode())
        raise RuntimeError("Failed to extract audio")

def process_frames_worker(frame_queue, out, zoom_scales):
    while True:
        try:
            frame_data = frame_queue.get()
            if frame_data is None:
                frame_queue.task_done()
                break  # Exit loop when sentinel is received

            frame_count, frame = frame_data
            current_scale = zoom_scales[frame_count]
            
            if current_scale != 1.0:
                frame = apply_zoom(frame, current_scale)

            out.write(frame)
            frame_queue.task_done()

        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            frame_queue.task_done()  # Ensure task_done is called even in case of error

def process_bounding_boxes(frame_queue, output_queue, zoom_scales):
    while True:
        try:
            frame_data = frame_queue.get()
            if frame_data is None:
                frame_queue.task_done()
                break  # Exit loop when sentinel is received

            frame_count, frame = frame_data
            current_scale = zoom_scales[frame_count]
            refined_scale, _, _ = get_bounding_box(frame)
            if current_scale != 1.0:
                   
                if refined_scale is not None:
                    output_queue.put((frame_count, math.ceil(refined_scale * 10) / 10))
                else:
                    output_queue.put((frame_count, current_scale))
            else:
                output_queue.put((frame_count, current_scale))
            frame_queue.task_done()

        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            frame_queue.task_done()  # Ensure task_done is called even in case of error
            

def process_video(video_path: str, zoom_effects: List[ZoomEffect]) -> str:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_dir = Path("temp_output")
    temp_dir.mkdir(exist_ok=True, parents=True)
    temp_video = str(temp_dir / "temp_video.mp4")
    temp_audio = str(temp_dir / "temp_audio.aac")
    final_output = str(temp_dir / f"output_{Path(video_path).stem}.mp4")

    status_text = st.empty()
    status_text.text("Extracting audio...")

    try:
        extract_audio(video_path, temp_audio)
    except RuntimeError as e:
        logging.error("An error occurred during audio extraction: %s", e)
        return

    out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not out.isOpened():
        raise RuntimeError("Failed to initialize video writer")

    zoom_scales = [1.0] * total_frames
    for effect in zoom_effects:
        start_frame = int(effect.start_time * fps)
        end_frame = min(total_frames, start_frame + int(effect.total_duration * fps))
        for frame_num in range(start_frame, end_frame):
            current_time = frame_num / fps
            zoom_scales[frame_num] = effect.get_scale_at_time_with_lag(current_time)

    progress_bar = st.progress(0)
    output_queue = Queue(maxsize = total_frames)
    frame_queue = Queue(maxsize=400)

    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.submit(process_bounding_boxes, frame_queue, output_queue, zoom_scales)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_queue.put((frame_count, frame))
            frame_count += 1

            if frame_count % (total_frames // 20) == 0:
                progress_bar.progress(frame_count / total_frames)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")

        cap.release()
        # Signal that no more frames will be added
        frame_queue.put(None)
        # Wait for the processing to complete
        frame_queue.join()
    
    processed_scales = {}
    while not output_queue.empty():
        frame_count, processed_scale = output_queue.get()
        processed_scales[frame_count] = processed_scale
    
    
    for effect in zoom_effects:
        start_frame = int(effect.start_time * fps) + int(effect.zoom_in_duration*fps)
        end_frame = min(total_frames, start_frame + int((effect.total_duration - effect.zoom_in_duration) * fps))
        values = [processed_scales[key] for key in range(start_frame, end_frame)]
        min_zoom_scale = min(values)
        for frame_num in range(start_frame, end_frame):
            zoom_scales[frame_num] = min_zoom_scale
        effect.scale = min_zoom_scale
        
    
    for effect in zoom_effects:
        start_frame = int(effect.start_time * fps) 
        end_frame = min(total_frames, start_frame + int(effect.zoom_in_duration * fps))
        for frame_num in range(start_frame, end_frame):
            current_time = frame_num / fps
            zoom_scales[frame_num] = effect.get_scale_at_time_with_lag(current_time)
    
    
    cap = cv2.VideoCapture(video_path)
    frame_queue = Queue(maxsize=400)
    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.submit(process_frames_worker, frame_queue, out, zoom_scales)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_queue.put((frame_count, frame))
            frame_count += 1

            if frame_count % (total_frames // 20) == 0:
                progress_bar.progress(frame_count / total_frames)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")

        cap.release()

        # Signal that no more frames will be added
        frame_queue.put(None)
        
        # Wait for the processing to complete
        frame_queue.join()
        out.release()

        logging.info("Beginning audio-video combination.")
        status_text.text("Combining video with audio...")

        ffmpeg_command = [
            'ffmpeg', '-i', temp_video, '-i', temp_audio, '-c:v', 'libx264',
            '-c:a', 'copy', '-shortest', '-movflags', '+faststart', '-y', final_output
        ]
        result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            logging.error("FFmpeg error during combination: %s", result.stderr.decode())
            raise RuntimeError("Failed to combine video and audio")

        logging.info("Combination completed successfully. Cleaning up temporary files.")
        if os.path.exists(temp_video):
            os.remove(temp_video)
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

        status_text.text("Processing complete!")
        return final_output
    
