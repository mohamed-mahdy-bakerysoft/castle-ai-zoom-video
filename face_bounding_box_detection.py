import mediapipe as mp
import cv2
import streamlit as st
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty

import logging

logging.basicConfig(level=logging.INFO)

HEIGHT_PADDING = 0.6
WIDTH_PADDING = 0.3


def expand_bounding_box(
    bbox, frame_height, frame_width, height_padding=0.2, width_padding=0.3
):
    """
    Expands the bounding box to include the full head (top of the forehead to the chin).

    Args:
        bbox (tuple): The bounding box for the face (x, y, w, h).
        frame_height (int): Height of the video frame.
        frame_width (int): Width of the video frame.
        padding (float): Fraction of padding to add around the face (default: 0.2).

    Returns:
        tuple: Expanded bounding box (x, y, w, h).
    """
    x, y, w, h = bbox
    expanded_h = int(
        h * (1 + height_padding)
    )  # Increase height by 20% or any desired value
    expanded_y = max(
        0, y - int(height_padding * h)
    )  # Ensure it doesn't go out of the top of the frame

    expanded_w = int(
        w * (1 + width_padding)
    )  # Increase height by 20% or any desired value
    expanded_x = max(0, x - int(width_padding * w))
    expanded_h = min(expanded_h, frame_height - expanded_y)
    expanded_w = min(expanded_w, frame_width - expanded_x)

    return (expanded_x, expanded_y, expanded_w, expanded_h)


# def get_bounding_box(frame):
#     # Convert frame to RGB (required by MediaPipe)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process frame to detect faces
#     results = face_detection.process(rgb_frame)

#     width = frame.shape[1]
#     height = frame.shape[0]
#     if results.detections:
#         # Process the first detected face
#         for detection in results.detections:
#             bboxC = detection.location_data.relative_bounding_box
#             x, y, w, h = (
#                 int(bboxC.xmin * width),
#                 int(bboxC.ymin * height),
#                 int(bboxC.width * width),
#                 int(bboxC.height * height),
#             )

#             # x_new, y_new, w_new, h_new = expand_bounding_box(
#             #     (x, y, w, h), height, width, height_padding=HEIGHT_PADDING, width_padding=WIDTH_PADDING
#             # )
#             # center_x, center_y = x_new + w_new // 2, y_new + h_new // 2
#             # max_up = center_y
#             # max_down = height - center_y
#             # max_left = center_x
#             # max_right = width - center_x
            
#             # scale_top = center_y / (center_y - y_new) if center_y != y_new else float('inf')
#             # scale_bottom = (height - center_y) / ((y_new + h_new) - center_y) if (y_new + h_new) != center_y else float('inf')
#             # scale_left = center_x / (center_x - x_new) if center_x != x_new else float('inf')
#             # scale_right = (width - center_x) / ((x_new + w_new) - center_x) if (x_new + w_new) != center_x else float('inf')
            
#             #min_scale = min(scale_top, scale_bottom, scale_left, scale_right)
#             return (x, y, w, h)#min_scale, center_x, center_y, True
#     else:
#         return None


# def process_bounding_boxes(frame_queue, output_queue):
#     while True:
#         try:
#             frame_data = frame_queue.get()
#             if frame_data is None:
#                 frame_queue.task_done()
#                 break  # Exit loop when sentinel is received

#             frame_count, frame = frame_data
#             coordinates = get_bounding_box(frame)
#             # if current_scale != 1.0:
#             output_queue.put(
#                 (
#                     frame_count,
#                     coordinates
#                     )
#             )
#             frame_queue.task_done()

#         except Exception as e:
#             logging.error(f"Error processing frame: {e}")
#             frame_queue.task_done()  # Ensure task_done is called even in case of error


# def get_bounding_box_coordinates(video_path):
#     cap = cv2.VideoCapture(video_path)
#     progress_bar = st.progress(0)

#     output_queue = Queue(maxsize=st.session_state.total_frames)
#     frame_queue = Queue(maxsize=400)
#     status_text = st.empty()
#     with ThreadPoolExecutor(max_workers=12) as executor:
#         executor.submit(process_bounding_boxes, frame_queue, output_queue)
#         frame_count = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_queue.put((frame_count, frame))
#             frame_count += 1

#             if frame_count % (st.session_state.total_frames // 20) == 0:
#                 progress_bar.progress(frame_count / st.session_state.total_frames)
#                 status_text.text(f"Processing frame {frame_count}/{st.session_state.total_frames}")

#         cap.release()
#         # Signal that no more frames will be added
#         frame_queue.put(None)
#         # Wait for the processing to complete
#         frame_queue.join()


#     # processed_scales = {}
#     # processed_centers = {}
#     # frame_face_flags = {}
#     bounding_box_coordinates = {}
#     while not output_queue.empty():
#         frame_count, bounding_box = output_queue.get()
#         bounding_box_coordinates[frame_count] = bounding_box
            
#     return bounding_box_coordinates
import mediapipe as mp
import cv2
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import logging
import time
from threading import Lock

logging.basicConfig(level=logging.INFO)

# Initialize MediaPipe face detection once, will be shared by threads
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)

def get_bounding_box(frame):
    if frame is None:
        return None
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    width = frame.shape[1]
    height = frame.shape[0]
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = (
                int(bboxC.xmin * width),
                int(bboxC.ymin * height),
                int(bboxC.width * width),
                int(bboxC.height * height),
            )
            return (x, y, w, h)
    return None

def process_frames(frame_queue, output_queue):
    while True:
        try:
            frame_data = frame_queue.get()
            if frame_data is None:
                frame_queue.task_done()
                break

            frame_count, frame = frame_data
            bbox = get_bounding_box(frame)
            output_queue.put((frame_count, bbox))
            frame_queue.task_done()
            
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            frame_queue.task_done()

def get_bounding_box_coordinates(video_path, num_threads=12):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_queue = Queue(maxsize=num_threads * 3)
    output_queue = Queue()
    processed_frames = 0
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Start worker threads
        workers = [
            executor.submit(process_frames, frame_queue, output_queue)
            for _ in range(num_threads)
        ]
        
        frame_count = 0
        status_text.text("Starting processing...")
        bounding_box_coordinates = {}
        
        while cap.isOpened() or not output_queue.empty():
            # Read and queue frames
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Don't wait if queue is full
                    if not frame_queue.full():
                        frame_queue.put((frame_count, frame))
                        frame_count += 1
                else:
                    cap.release()
                    # Signal threads to finish
                    for _ in range(num_threads):
                        frame_queue.put(None)
            
            # Process results
            while not output_queue.empty():
                frame_idx, bbox = output_queue.get()
                bounding_box_coordinates[frame_idx] = bbox
                processed_frames += 1
                
                # Update progress from main thread
                if processed_frames % 5 == 0:
                    progress = processed_frames / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {processed_frames}/{total_frames}")
            
            # Brief sleep to prevent CPU spinning
            time.sleep(0.001)

        # Wait for all frames to be processed
        frame_queue.join()

    # Final validation
    if len(bounding_box_coordinates) != total_frames:
        missing_frames = set(range(total_frames)) - set(bounding_box_coordinates.keys())
        st.warning(f"Missing {len(missing_frames)} frames out of {total_frames}")
        
    no_detection_frames = sum(1 for v in bounding_box_coordinates.values() if v is None)
    st.info(f"Frames with no face detection: {no_detection_frames}/{total_frames}")
    
    return bounding_box_coordinates

# import mediapipe as mp
# import cv2
# import streamlit as st
# from concurrent.futures import ThreadPoolExecutor
# from queue import Queue
# import logging

# logging.basicConfig(level=logging.INFO)

# # Initialize MediaPipe face detection once, will be shared by threads
# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(
#     model_selection=0, min_detection_confidence=0.5
# )

# def get_bounding_box(frame):
#     if frame is None:
#         return None
        
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_detection.process(rgb_frame)

#     width = frame.shape[1]
#     height = frame.shape[0]
#     if results.detections:
#         for detection in results.detections:
#             bboxC = detection.location_data.relative_bounding_box
#             x, y, w, h = (
#                 int(bboxC.xmin * width),
#                 int(bboxC.ymin * height),
#                 int(bboxC.width * width),
#                 int(bboxC.height * height),
#             )
#             return (x, y, w, h)
#     return None

# def process_frames(frame_queue, output_queue):
#     while True:
#         try:
#             frame_data = frame_queue.get()
#             if frame_data is None:
#                 frame_queue.task_done()
#                 break

#             frame_count, frame = frame_data
#             bbox = get_bounding_box(frame)
#             output_queue.put((frame_count, bbox))
#             frame_queue.task_done()
            
#         except Exception as e:
#             logging.error(f"Error processing frame: {e}")
#             frame_queue.task_done()

# def get_bounding_box_coordinates(video_path, num_threads=12):
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     frame_queue = Queue(maxsize=400)
#     output_queue = Queue()
    
#     with ThreadPoolExecutor(max_workers=num_threads) as executor:
#         # Start worker threads
#         workers = [
#             executor.submit(process_frames, frame_queue, output_queue)
#             for _ in range(num_threads)
#         ]
        
#         # Read and queue frames
#         frame_count = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_queue.put((frame_count, frame))
#             frame_count += 1

#             if frame_count % (total_frames // 20) == 0:
#                 progress_bar.progress(frame_count / total_frames)
#                 status_text.text(f"Reading frame {frame_count}/{total_frames}")

#         # Signal threads to finish
#         for _ in range(num_threads):
#             frame_queue.put(None)

#         # Wait for all frames to be processed
#         frame_queue.join()

#     cap.release()

#     # Collect results
#     bounding_box_coordinates = {}
#     while not output_queue.empty():
#         frame_idx, bbox = output_queue.get()
#         bounding_box_coordinates[frame_idx] = bbox

#     # Final validation
#     if len(bounding_box_coordinates) != total_frames:
#         missing_frames = set(range(total_frames)) - set(bounding_box_coordinates.keys())
#         st.warning(f"Missing {len(missing_frames)} frames out of {total_frames}")
        
#     no_detection_frames = sum(1 for v in bounding_box_coordinates.values() if v is None)
#     st.info(f"Frames with no face detection: {no_detection_frames}/{total_frames}")

#     return bounding_box_coordinates


# import mediapipe as mp
# import cv2
# import streamlit as st
# from multiprocessing import Pool, Manager
# import logging
# import numpy as np
# from typing import Dict, Optional, Tuple

# logging.basicConfig(level=logging.INFO)

# def init_worker():
#     """Initialize MediaPipe face detection in each worker process"""
#     global face_detection
#     mp_face_detection = mp.solutions.face_detection
#     face_detection = mp_face_detection.FaceDetection(
#         model_selection=0, min_detection_confidence=0.5
#     )

# def get_bounding_box(frame):
#     """Get bounding box using global face_detection instance"""
#     if frame is None:
#         return None
        
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_detection.process(rgb_frame)

#     width = frame.shape[1]
#     height = frame.shape[0]
#     if results.detections:
#         for detection in results.detections:
#             bboxC = detection.location_data.relative_bounding_box
#             x, y, w, h = (
#                 int(bboxC.xmin * width),
#                 int(bboxC.ymin * height),
#                 int(bboxC.width * width),
#                 int(bboxC.height * height),
#             )
#             return (x, y, w, h)
#     return None

# def process_chunk(chunk_data):
#     """Process a chunk of frames"""
#     start_idx, frames = chunk_data
#     results = {}
    
#     for i, frame in enumerate(frames):
#         frame_idx = start_idx + i
#         try:
#             bbox = get_bounding_box(frame)
#             results[frame_idx] = bbox
#         except Exception as e:
#             logging.error(f"Error processing frame {frame_idx}: {str(e)}")
#             results[frame_idx] = None
        
#     return results

# def get_bounding_box_coordinates(video_path: str, num_processes: int = 4, chunk_size: int = 20) -> Dict[int, Optional[Tuple[int, int, int, int]]]:
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     manager = Manager()
#     bounding_box_coordinates = manager.dict()
#     processed_frames = 0
    
#     try:
#         with Pool(processes=num_processes, initializer=init_worker) as pool:
#             while processed_frames < total_frames:
#                 # Read one chunk at a time
#                 frames = []
#                 chunk_start = processed_frames
                
#                 # Read frames for current chunk
#                 for _ in range(chunk_size):
#                     if processed_frames >= total_frames:
#                         break
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#                     frames.append(frame)
#                     processed_frames += 1
                
#                 if not frames:
#                     break
                
#                 # Process current chunk immediately
#                 chunk_results = pool.apply(process_chunk, ((chunk_start, frames),))
                
#                 # Store results
#                 for frame_idx, bbox in chunk_results.items():
#                     bounding_box_coordinates[frame_idx] = bbox
                
#                 # Update progress
#                 progress = processed_frames / total_frames
#                 progress_bar.progress(progress)
#                 status_text.text(f"Processing frame {processed_frames}/{total_frames}")
                
#                 # Clean up current chunk
#                 del frames
                
#             # Final validation
#             total_processed = len(bounding_box_coordinates)
#             if total_processed != total_frames:
#                 missing_frames = set(range(total_frames)) - set(bounding_box_coordinates.keys())
#                 logging.warning(f"Missing frames: {missing_frames}")
#                 st.warning(f"Missing {len(missing_frames)} frames out of {total_frames}")
            
#             no_detection_frames = sum(1 for v in bounding_box_coordinates.values() if v is None)
#             st.info(f"Frames with no face detection: {no_detection_frames}/{total_frames}")
            
#     except Exception as e:
#         logging.error(f"Error during processing: {str(e)}")
#         st.error(f"Error during processing: {str(e)}")
#         raise
#     finally:
#         cap.release()
    
#     return dict(bounding_box_coordinates)

