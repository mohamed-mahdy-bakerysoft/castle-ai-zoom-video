from face_bounding_box_detection import get_bounding_box
import logging
import math

logging.basicConfig(level=logging.INFO)


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


def process_scales_centers_after_extracting_boundaries(
    out_queue, zoom_scales, zoom_effects, total_frames, fps
):

    # Get the processed scales and centers
    processed_scales = {}
    processed_centers = {}
    while not out_queue.empty():
        frame_count, processed_scale, center_x, center_y = out_queue.get()
        processed_scales[frame_count] = processed_scale
        processed_centers[frame_count] = (center_x, center_y)

    # get the minimum scale for holding in that position
    min_zoom_keys_per_effect = {}
    for i, effect in enumerate(zoom_effects):
        start_frame = int((effect.start_time + effect.zoom_in_duration)* fps)
        end_frame = min(
            total_frames,
            start_frame + int((effect.total_duration - effect.zoom_in_duration) * fps),
        )
        values = [(key, processed_scales[key]) for key in range(start_frame, end_frame)]
        min_zoom_key, min_zoom_scale = min(values, key=lambda x: x[1])
        for frame_num in range(start_frame, end_frame):
            zoom_scales[frame_num] = min_zoom_scale
            processed_centers[frame_num] = processed_centers[min_zoom_key]
        min_zoom_keys_per_effect[i] = min_zoom_key
        effect.scale = min_zoom_scale

    for i, effect in enumerate(zoom_effects):
        start_frame = int(effect.start_time * fps)
        end_frame = min(total_frames, start_frame + int(effect.zoom_in_duration * fps))
        for frame_num in range(start_frame, end_frame):
            current_time = frame_num / fps
            zoom_scales[frame_num] = effect.get_scale_at_time_with_lag(current_time)
            processed_centers[frame_num] = processed_centers[
                min_zoom_keys_per_effect[i]
            ]

    return zoom_scales, processed_centers


def get_initial_zoom_scales(total_frames, fps, zoom_effects):
    zoom_scales = [1.0] * total_frames
    for effect in zoom_effects:
        start_frame = int(effect.start_time * fps)
        end_frame = min(total_frames, start_frame + int(effect.total_duration * fps))
        for frame_num in range(start_frame, end_frame):
            current_time = frame_num / fps
            zoom_scales[frame_num] = effect.get_scale_at_time_with_lag(current_time)

    return zoom_scales


def process_zoom_scales_centers_after_extracting_boundaries(
   zoom_effects,
   total_frames, 
   fps,
   height,
   width,
   bounding_box_data
):
    
    zoom_scales = get_initial_zoom_scales(total_frames, fps  zoom_effects)





    # # Get the processed scales and centers
    # processed_scales = {}
    # processed_centers = {}
    # while not out_queue.empty():
    #     frame_count, processed_scale, center_x, center_y = out_queue.get()
    #     processed_scales[frame_count] = processed_scale
    #     processed_centers[frame_count] = (center_x, center_y)

    # # get the minimum scale for holding in that position
    # min_zoom_keys_per_effect = {}
    # for i, effect in enumerate(zoom_effects):
    #     start_frame = int((effect.start_time + effect.zoom_in_duration)* fps)
    #     end_frame = min(
    #         total_frames,
    #         start_frame + int((effect.total_duration - effect.zoom_in_duration) * fps),
    #     )
    #     values = [(key, processed_scales[key]) for key in range(start_frame, end_frame)]
    #     min_zoom_key, min_zoom_scale = min(values, key=lambda x: x[1])
    #     for frame_num in range(start_frame, end_frame):
    #         zoom_scales[frame_num] = min_zoom_scale
    #         processed_centers[frame_num] = processed_centers[min_zoom_key]
    #     min_zoom_keys_per_effect[i] = min_zoom_key
    #     effect.scale = min_zoom_scale

    # for i, effect in enumerate(zoom_effects):
    #     start_frame = int(effect.start_time * fps)
    #     end_frame = min(total_frames, start_frame + int(effect.zoom_in_duration * fps))
    #     for frame_num in range(start_frame, end_frame):
    #         current_time = frame_num / fps
    #         zoom_scales[frame_num] = effect.get_scale_at_time_with_lag(current_time)
    #         processed_centers[frame_num] = processed_centers[
    #             min_zoom_keys_per_effect[i]
    #         ]

    # return zoom_scales, processed_centers