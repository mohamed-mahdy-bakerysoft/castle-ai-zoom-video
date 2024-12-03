import os
import time
import wave
import string
import subprocess
import contextlib
import concurrent.futures
from random import choices
from typing import Union, Tuple

from numpy import arange

from span_api_splitter import literals
# from span_api_splitter.model.silero.utils import make_silero_timestamps_for_audio_part
from span_api_splitter.model.pywebrtc.utils import make_webrtc_timestamps_for_audio_part


def adjust_timestamps(audio_name: str, vad_timestamps: list, audio_duration_sec: float,
                      min_silence_len: Union[int, float], min_sentence_len: Union[int, float],
                      max_sentence_len: Union[int, float], extra_min_silence_len: float) -> list:
    """
    Adjusts vad timestamps so that the new speech splits satisfy defined constraints (splitting at
    min_silence_len, keeping the lengths of audios between min_sentence_len and max_sentence_len).
    Args:
        audio_name:
        vad_timestamps: Timestamps generated with VAD algorithm.
        audio_duration_sec: Duration of the original audio in seconds.
        min_silence_len: The audio will be cut on silence if the silence has at least min_silence_len (s) length.
        min_sentence_len: The speech part can't be cut if it's shorter than min_voiced_part_time (s).
        max_sentence_len: The speech part can't be cut if it's longer than max_voiced_part_time (s).
        extra_min_silence_len: If the speech part is longer than max_sentence_len, we will try to make the part.
            smaller by searching for smaller silences which can't be shorter than extra_min_silence_len.
    Returns:
        List containing lists of timestamps for each speech part.
    """
    if len(vad_timestamps) == 1:
        return vad_timestamps
    timestamps = []
    i = 0
    speech_start = vad_timestamps[0][0]
    speech_end = vad_timestamps[-1][1]
    speech_start_idx = 0
    need_less_silence = 0
    while i < len(vad_timestamps) - 1:
        cur_speech = vad_timestamps[i]
        silence_after_cur_speech = round(vad_timestamps[i + 1][0] - vad_timestamps[i][1], literals.PRECISION)
        silence_len = round(min_silence_len - need_less_silence, literals.PRECISION)
        if silence_after_cur_speech >= silence_len:
            speech_duration = cur_speech[1] - speech_start
            if min_sentence_len <= speech_duration <= max_sentence_len:
                speech_end = cur_speech[1]
                timestamps.append([speech_start, speech_end])
                speech_start = vad_timestamps[i + 1][0]
                speech_start_idx = i + 1
                need_less_silence = 0
            elif speech_duration < min_sentence_len:
                i += 1
                continue
            else:
                if silence_len > extra_min_silence_len:
                    need_less_silence += 0.1
                    i = speech_start_idx - 1
                else:
                    speech_end = cur_speech[1]
                    timestamps.append([speech_start, speech_end])
                    speech_start = vad_timestamps[i + 1][0]
                    speech_start_idx = i + 1
                    need_less_silence = 0
        i += 1
    if not timestamps:
        return [[0, audio_duration_sec]]
    if speech_start >= speech_end:
        if vad_timestamps[-1][1] - speech_start >= min_sentence_len:
            timestamps.append([speech_start, vad_timestamps[-1][1]])
        else:
            timestamps[-1][1] = vad_timestamps[-1][1]
    timestamps[0][0] = max(0, timestamps[0][0])
    timestamps[-1][1] = min(audio_duration_sec, timestamps[-1][1])
    return timestamps


def audio_force_split(audio_name: str, audio_end_s: float, max_sentence_len: Union[int, float],
                      audio_start_s: float = 0.) -> list:
    """
    Splitting points to split the audio into parts with max_sentence_len length.
    Args:
        audio_name:
        audio_end_s: Duration of the original audio (s).
        max_sentence_len: The speech part can't be cut if it's longer than max_voiced_part_time (s).
        audio_start_s: The starting second of the audio.
    Returns:
        List containing splitting points in seconds.
    """
    max_sentence_len = float(max_sentence_len)
    return [split_start for split_start in arange(audio_start_s + max_sentence_len, audio_end_s, max_sentence_len)]


def adjust_ends(splitting_points: list, audio_end_s: float, min_sentence_len: Union[int, float],
                max_sentence_len: Union[int, float], audio_name: str) -> list:
    """If there is a small remaining part at the end, adds it to the last timestamp."""
    if splitting_points[-1] != audio_end_s:
        if audio_end_s - splitting_points[-1] < min_sentence_len:
            if len(splitting_points) > 1:
                prev_splitting_point = splitting_points[-2]
            else:
                prev_splitting_point = 0
            if audio_end_s - prev_splitting_point > max_sentence_len:
                print(f"The size of merged chunks of {audio_name} is greater than maximum sentence length. "
                             f"{audio_end_s - prev_splitting_point} > {max_sentence_len}. Merging anyways.")
            splitting_points[-1] = audio_end_s
        else:
            splitting_points.append(audio_end_s)
    return splitting_points[:-1]


def audio_split_at_centers(audio_name: str, vad_timestamps: list, audio_duration_sec: float,
                           min_silence_len: Union[int, float], min_sentence_len: Union[int, float],
                           max_sentence_len: Union[int, float], extra_min_silence_len: float) -> Tuple[list, list]:
    """
    Makes splitting points to split at the centers of detected non-speech parts. If there is a speech part with
    duration longer than max_sentence_len splits it into two halves.
    Args:
        audio_name:
        vad_timestamps: Timestamps generated with make_vad_timestamps.
        audio_duration_sec: Duration of the original audio (s).
        min_silence_len: The audio will be cut on silence if the silence has at least min_silence_len (s) length.
        min_sentence_len: The speech part can't be cut if it's shorter than min_voiced_part_time (s).
        max_sentence_len: The speech part can't be cut if it's longer than max_voiced_part_time (s).
        extra_min_silence_len: If the speech part is longer than max_sentence_len, we will try to make the part
            smaller by searching for smaller silences which can't be shorter than extra_min_silence_len.
    Returns:
        Tuple of lists containing speech splitting points and silence durations between speech parts.
    """
    splitting_points = []
    silence_durations = []
    timestamps = adjust_timestamps(audio_name, vad_timestamps, audio_duration_sec, min_silence_len, min_sentence_len,
                                   max_sentence_len, extra_min_silence_len)
    if len(timestamps) == 1:
        if timestamps[0][1] > max_sentence_len:
            splitting_points = audio_force_split(audio_name=audio_name, audio_end_s=audio_duration_sec,
                                                 max_sentence_len=max_sentence_len)
            silence_durations = [-2] * len(splitting_points)
        else:
            splitting_points.append(audio_duration_sec)
    else:
        prev_timestamp = timestamps[0]
        for cur_timestamp in timestamps[1:]:
            center = (prev_timestamp[1] + cur_timestamp[0]) / 2
            splitting_points.append(round(center, literals.PRECISION))
            silence_durations.append(round(cur_timestamp[0] - prev_timestamp[1], literals.PRECISION))
            prev_timestamp = cur_timestamp
            # if the speech part is longer than max_sentence_len split it at the center
            if cur_timestamp[1] - cur_timestamp[0] > max_sentence_len:
                big_sentence_split_points = audio_force_split(audio_name=audio_name,
                                                              audio_end_s=round(cur_timestamp[1], literals.PRECISION),
                                                              max_sentence_len=max_sentence_len,
                                                              audio_start_s=round(cur_timestamp[0], literals.PRECISION))
                splitting_points.extend(big_sentence_split_points)
                splitting_points = adjust_ends(splitting_points, cur_timestamp[1], min_sentence_len, max_sentence_len,
                                               audio_name)
                silence_durations.extend([-1] * len(big_sentence_split_points))
                prev_timestamp = [splitting_points[-1], cur_timestamp[1]]
    splitting_points = adjust_ends(splitting_points, audio_duration_sec, min_sentence_len, max_sentence_len, audio_name)
    return splitting_points, silence_durations


def get_audio_parts_positions(audio_duration: float, part_duration: float, min_sentence_len: Union[int, float],
                              overlap_duration: Union[int, float] = 10) -> list:
    """
    Makes positions at which the audio file should be parted.
    Args:
        audio_duration: Duration of the original audio (s).
        part_duration: Parted audio duration.
        min_sentence_len: The speech part can't be cut if it's shorter than min_voiced_part_time (s).
        overlap_duration: How much overlap will audio parts have.
    Returns:
        List containing starting position for reading the file, how many frames to read and part number.
    """
    audio_nframes = int(audio_duration * literals.SAMPLE_RATE)
    part_n_frames = int(part_duration * literals.SAMPLE_RATE)
    overlap_nframes = int(overlap_duration * literals.SAMPLE_RATE)
    split_positions = []
    start = 0
    end = part_n_frames
    part_num = 0
    while end <= audio_nframes:
        split_positions.append([start, part_n_frames, part_num])
        part_num += 1
        start = end - overlap_nframes
        end = start + part_n_frames
    if not split_positions:
        return [[0, audio_nframes, 0]]
    end_position = split_positions[-1][0] + split_positions[-1][1]
    if end_position == audio_nframes:
        return split_positions
    if audio_nframes - end_position + overlap_nframes <= int(min_sentence_len * literals.SAMPLE_RATE):
        split_positions[-1][1] = audio_nframes - split_positions[-1][0]
    else:
        split_positions.append([end_position - overlap_nframes,
                                audio_nframes - end_position + overlap_nframes, part_num])
    return split_positions


def get_audio_duration(path: str) -> float:
    """Takes audio path and returns it's duration."""
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        return wf.getnframes() / wf.getframerate()


def resample_and_split_file(audio_uri: str, audio_name: str, data_dir: str, part_duration: float,
                            min_sentence_len: Union[int, float]) -> Tuple[str, float, list]:
    """Resamples the audio, makes mono and makes positions for parting the audio.
    Args:
        audio_uri: google cloud storage public or signed uri
        audio_name: Unique name.
        data_dir: Directory to save the files.
        part_duration: Parted audio duration.
        min_sentence_len: The speech part can't be cut if it's shorter than min_voiced_part_time (s).
    Returns:
        Tuple of resampled file path, audio duration and parting positions.
    """
    file_path = os.path.join(data_dir, audio_name + '.wav')
    print(f"Downloading audio {audio_name}")
    start_time = time.time()
    p = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', audio_uri, '-ac', '1', '-ar',
                          str(literals.SAMPLE_RATE), file_path])
    p.wait()
    end_time = time.time()
    duration = get_audio_duration(file_path)
    print(f"Audio {audio_name} duration: {round(duration, literals.PRECISION)}s, "
                 f"downloading time: {round(end_time - start_time, literals.PRECISION)}s")
    if duration <= min_sentence_len:
        print(f"{audio_name} Audio duration ({duration}) is less than min_sentence_len ({min_sentence_len})")
        os.remove(file_path)
        return {}, None, None
    parts_positions = get_audio_parts_positions(duration, part_duration, min_sentence_len)
    return file_path, duration, parts_positions


def get_threads_results(vad: str, request_data_json: dict, config: dict, file_path: str, audio_name: str,
                        parts_positions: list) -> list:
    """
    Collects timestamps of audio parts with multithreading.
    Args:
        vad: Which VAD algorithm to use.
        request_data_json: Parameters passed from request.
        config: Config with default parameters.
        file_path: Resampled audio file path.
        audio_name: Unique name.
        parts_positions: List containing starting position for reading the file, how many frames to read
            and part number.
    Returns:
        List of part numbers and timestamps for each audio part.
    """
    num_threads = 8
    if vad == 'webrtc':
        mode = request_data_json.get('mode', config['webrtc_params']['mode'])
        frame_duration_ms = request_data_json.get('frame_duration_ms', config['webrtc_params']['frame_duration_ms'])
        padding_duration_ms = request_data_json.get('padding_duration_ms',
                                                    config['webrtc_params']['padding_duration_ms'])
        with concurrent.futures.ThreadPoolExecutor(num_threads) as thread_exec:
            threads_results = [thread_exec.submit(make_webrtc_timestamps_for_audio_part, part_start=part_info[0],
                                                  part_n_frames=part_info[1], part_num=part_info[2], wav_path=file_path,
                                                  mode=mode, frame_duration_ms=frame_duration_ms,
                                                  padding_duration_ms=padding_duration_ms)
                               for part_info in parts_positions]
            threads_results = [it.result() for it in threads_results]

    # elif vad == 'silero':
    #     vad_params = request_data_json.get('model',
    #                                        os.path.join(config['models_dir'], config['silero_params']['model']))
    #     trig_sum = request_data_json.get('trig_sum', config['silero_params']['trig_sum'])
    #     neg_trig_sum = request_data_json.get('neg_trig_sum', config['silero_params']['neg_trig_sum'])
    #     num_steps = request_data_json.get('num_steps', config['silero_params']['num_steps'])
    #     batch_size = request_data_json.get('batch_size', config['silero_params']['batch_size'])
    #     num_samples_per_window = request_data_json.get('num_samples_per_window',
    #                                                    config['silero_params']['num_samples_per_window'])
    #     min_speech_samples = request_data_json.get('min_speech_samples', config['silero_params']['min_speech_samples'])
    #     min_silence_samples = request_data_json.get('min_silence_samples',
    #                                                 config['silero_params']['min_silence_samples'])
    #     with concurrent.futures.ThreadPoolExecutor(num_threads) as thread_exec:
    #         threads_results = [thread_exec.submit(
    #             make_silero_timestamps_for_audio_part,
    #             wav_part_path=os.path.join(config['data_dir'], f'{audio_name}_part{part_info[2]}.wav'),
    #             part_start=part_info[0], part_n_frames=part_info[1], part_num=part_info[2], file_path=file_path,
    #             vad_params=vad_params, trig_sum=trig_sum, neg_trig_sum=neg_trig_sum, num_steps=num_steps,
    #             batch_size=batch_size, num_samples_per_window=num_samples_per_window,
    #             min_speech_samples=min_speech_samples, min_silence_samples=min_silence_samples)
    #             for part_info in parts_positions]
    #         threads_results = [it.result() for it in threads_results]
    #     for part_info in parts_positions:
    #         path = os.path.join(config['data_dir'], f'{audio_name}_part{part_info[2]}.wav')
    #         if os.path.exists(path):
    #             os.remove(path)
    else:
        raise NameError("Incorrect vad name")
    return threads_results


def get_merge_idx(timestamp: list, next_timestamps: list) -> int:
    """Helping function for combining audio parts timestamps."""
    if timestamp[1] <= next_timestamps[0][0]:
        return -1
    for idx, t in enumerate(next_timestamps):
        if timestamp[1] <= t[1]:
            return idx


def get_vad_timestamps_from_threads(threads_results: list, parts_positions: list) -> list:
    """
    Collects timestamps of each audio part and merges them together.
    Args:
        threads_results: List of part numbers and timestamps for each audio part.
        parts_positions: List containing starting position for reading the file, how many frames to read
            and part number.
    Returns:
        List of timestamps for the whole audio.
    """
    # discard empty parts
    parts_timestamps_dict = {
        num_part: part_timestamps for num_part, part_timestamps
        in threads_results if part_timestamps
    }
    
    # convert number of samples to seconds
    n_parts = len(parts_positions)
    for num_part in range(n_parts):
        part_timestamps = parts_timestamps_dict.get(num_part, [])
        for i in range(len(part_timestamps)):
            parts_timestamps_dict[num_part][i][0] += parts_positions[num_part][0]/literals.SAMPLE_RATE
            parts_timestamps_dict[num_part][i][1] += parts_positions[num_part][0]/literals.SAMPLE_RATE

    # collect and merge part timestamps
    vad_timestamps = []
    for num_part, part_timestamps in parts_timestamps_dict.items():
        if not vad_timestamps:
            vad_timestamps.extend(part_timestamps)
        elif part_timestamps:
            merge_idx = get_merge_idx(vad_timestamps[-1], part_timestamps)
            if merge_idx is None:
                print(f"Merge index is None. num_part: {num_part}, timestamp: {vad_timestamps[-1]}, "
                             f"next_timestamps: {part_timestamps}")
            elif merge_idx == -1:
                vad_timestamps.extend(part_timestamps)
            else:
                vad_timestamps[-1][1] = part_timestamps[merge_idx][1]
                vad_timestamps.extend(part_timestamps[merge_idx + 1:])
                
    return vad_timestamps


def get_split_info(payload: dict, config: dict) -> dict:
    """
    Takes the request data and makes splitting points.
    Args:
        payload: Request parameters.
        config: Config containing default parameters.
    Returns:
        Dictionary of speech splitting points and silence durations between the speech parts.
    """
    audio_uri = payload[literals.AUDIO_URL]
    result_id = payload.get(literals.RESULT_ID)
    if len(result_id) > 40:
        result_id = "audio_with_long_name"
    audio_name = result_id + '_'
    audio_name += ''.join(choices(string.ascii_uppercase + string.digits, k=20))

    min_sentence_len = payload.get(literals.MIN_SENTENCE_LEN, config['min_sentence_len'])
    max_sentence_len = payload.get(literals.MAX_SENTENCE_LEN, config['max_sentence_len'])
    if max_sentence_len <= min_sentence_len:
        raise ValueError('max_sentence_len <= min_sentence_len')
    min_silence_len = payload.get(literals.MIN_SILENCE_LEN, config['min_silence_len'])
    show_silence_lengths = payload.get(literals.SHOW_SILENCE_LENGTHS, config['return_silence_lengths'])
    extra_min_silence_len = payload.get(literals.EXTRA_MIN_SILENCE_LEN, config['extra_min_silence_len'])

    vad = payload.get(literals.VAD, config['vad'])
    init_part_len = payload.get(literals.INIT_PART_LEN, config['init_part_len'])

    file_path, audio_duration, parts_positions = resample_and_split_file(audio_uri, audio_name, config['data_dir'],
                                                                         init_part_len, min_sentence_len)
    if not parts_positions:
        return {}

    threads_results = get_threads_results(vad, payload, config, file_path, audio_name, parts_positions)
    os.remove(file_path)

    vad_timestamps = get_vad_timestamps_from_threads(threads_results, parts_positions)
    if not vad_timestamps:
        splitting_points = audio_force_split(audio_name=audio_name, audio_end_s=audio_duration,
                                             max_sentence_len=max_sentence_len)
        splitting_points = adjust_ends(splitting_points, audio_duration, min_sentence_len, max_sentence_len, audio_name)
        silence_durations = [-2] * len(splitting_points)
    else:
        splitting_points, silence_durations = audio_split_at_centers(audio_name, vad_timestamps, audio_duration,
                                                                     min_silence_len, min_sentence_len,
                                                                     max_sentence_len, extra_min_silence_len)
    if not splitting_points:
        return {}

    if show_silence_lengths:
        return {'splittingPoints': splitting_points,
                'silenceDurations': silence_durations}
    else:
        return {'splittingPoints': splitting_points}