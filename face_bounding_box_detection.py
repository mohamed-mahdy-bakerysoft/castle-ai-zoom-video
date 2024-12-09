import mediapipe as mp
import cv2
import streamlit as st
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import logging

logging.basicConfig(level=logging.INFO)

HEIGHT_PADDING = 0.2
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


def get_bounding_box(frame):
    # Convert frame to RGB (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame to detect faces
    results = face_detection.process(rgb_frame)

    width = frame.shape[1]
    height = frame.shape[0]
    if results.detections:
        # Process the first detected face
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = (
                int(bboxC.xmin * width),
                int(bboxC.ymin * height),
                int(bboxC.width * width),
                int(bboxC.height * height),
            )

            # x_new, y_new, w_new, h_new = expand_bounding_box(
            #     (x, y, w, h), height, width, height_padding=HEIGHT_PADDING, width_padding=WIDTH_PADDING
            # )
            # center_x, center_y = x_new + w_new // 2, y_new + h_new // 2
            # max_up = center_y
            # max_down = height - center_y
            # max_left = center_x
            # max_right = width - center_x
            
            # scale_top = center_y / (center_y - y_new) if center_y != y_new else float('inf')
            # scale_bottom = (height - center_y) / ((y_new + h_new) - center_y) if (y_new + h_new) != center_y else float('inf')
            # scale_left = center_x / (center_x - x_new) if center_x != x_new else float('inf')
            # scale_right = (width - center_x) / ((x_new + w_new) - center_x) if (x_new + w_new) != center_x else float('inf')
            
            #min_scale = min(scale_top, scale_bottom, scale_left, scale_right)
            return (x, y, w, h)#min_scale, center_x, center_y, True
    else:
        return None


def process_bounding_boxes(frame_queue, output_queue):
    while True:
        try:
            frame_data = frame_queue.get()
            if frame_data is None:
                frame_queue.task_done()
                break  # Exit loop when sentinel is received

            frame_count, frame = frame_data
            coordinates = get_bounding_box(frame)
            # if current_scale != 1.0:
            output_queue.put(
                (
                    frame_count,
                    coordinates
                    )
            )
            frame_queue.task_done()

        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            frame_queue.task_done()  # Ensure task_done is called even in case of error


def get_bounding_box_coordinates(video_path):
    cap = cv2.VideoCapture(video_path)
    progress_bar = st.progress(0)

    output_queue = Queue(maxsize=st.session_state.total_frames)
    frame_queue = Queue(maxsize=400)
    status_text = st.empty()
    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.submit(process_bounding_boxes, frame_queue, output_queue)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_queue.put((frame_count, frame))
            frame_count += 1

            if frame_count % (st.session_state.total_frames // 20) == 0:
                progress_bar.progress(frame_count / st.session_state.total_frames)
                status_text.text(f"Processing frame {frame_count}/{st.session_state.total_frames}")

        cap.release()
        # Signal that no more frames will be added
        frame_queue.put(None)
        # Wait for the processing to complete
        frame_queue.join()


    # processed_scales = {}
    # processed_centers = {}
    # frame_face_flags = {}
    bounding_box_coordinates = {}
    while not output_queue.empty():
        frame_count, bounding_box = output_queue.get()
        bounding_box_coordinates[frame_count] = bounding_box
            
    return bounding_box_coordinates