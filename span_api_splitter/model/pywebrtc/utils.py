import wave
import contextlib
import collections
from typing import Generator

import webrtcvad

from span_api_splitter import literals


def read_wave(path: str, start_pos: int, n_frames: int) -> bytes:
    """
    Reads a wav file.
    Args:
        path: Path to the wav file.
        start_pos: Starts reading the file from start_pos.
        n_frames: Number of frames to read.
    Returns:
        PCM audio bytes.
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        wf.setpos(start_pos)
        return wf.readframes(n_frames)


class Frame(object):
    """
    Represents a "frame" of audio data.
    """
    def __init__(self, data_bytes: bytes, timestamp: float) -> None:
        self.bytes = data_bytes
        self.timestamp = timestamp


def frame_generator(frame_duration_ms: int, audio: bytes) -> Generator[Frame, None, None]:
    """
    Generates audio frames from PCM audio data.
    Args:
        frame_duration_ms: The frame duration in milliseconds.
        audio: PCM audio bytes.
    Yields:
        Frames of the requested duration.
    """
    duration = frame_duration_ms / 1000.0
    n = int(literals.SAMPLE_RATE * duration * 2)
    offset = 0
    timestamp = 0.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp)
        timestamp += duration
        offset += n


def vad_collector(frame_duration_ms: int, padding_duration_ms: int, vad: webrtcvad.Vad, audio: bytes) -> list:
    """
    Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, collects only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins collecting
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Args:
        frame_duration_ms: The frame duration in milliseconds.
        padding_duration_ms: The amount to pad the window, in milliseconds.
        vad: An instance of webrtcvad.Vad.
        audio: PCM audio bytes.
    Returns:
        List of voiced parts timestamps.
    """

    # logger.debug("Running VAD")
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    frame_duration_s = frame_duration_ms / 1000.0
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    speech_start = 0
    timestamps = []
    for frame in frame_generator(frame_duration_ms, audio):
        is_speech = vad.is_speech(frame.bytes, literals.SAMPLE_RATE)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > literals.FRAME_RATIO_THRESHOLD * ring_buffer.maxlen:
                triggered = True
                speech_start = ring_buffer[0][0].timestamp
                # We want to yield all the audio we see from now until we are NOTTRIGGERED, but we have to start with
                # the audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > literals.FRAME_RATIO_THRESHOLD * ring_buffer.maxlen:
                triggered = False
                speech_end = frame.timestamp + frame_duration_s
                timestamps.append([speech_start, speech_end])
                ring_buffer.clear()
                voiced_frames = []

    if voiced_frames:
        timestamps.append([speech_start, voiced_frames[-1].timestamp + frame_duration_s])
    return timestamps


def make_webrtc_timestamps_for_audio_part(part_start: int, part_n_frames: int, part_num: int, wav_path: str, mode: int,
                                          frame_duration_ms: int, padding_duration_ms: int) -> list:
    """
    Makes timestamps for the given audio part.
    Args:
        part_start: Starts reading the file from start_pos.
        part_n_frames: Number of frames to read.
        part_num: Number of the audio part to read.
        wav_path: Path to the whole audio.
        mode: Aggressiveness of webrtc VAD from 0 to 3.
        frame_duration_ms: Frame duration in milliseconds.
        padding_duration_ms: Amount to pad the window, in milliseconds.
    Returns:
        Part number and corresponding timestamps
    """
    audio = read_wave(wav_path, part_start, part_n_frames)
    vad = webrtcvad.Vad(mode)
    vad_timestamps = vad_collector(frame_duration_ms, padding_duration_ms, vad, audio)
    return [part_num, vad_timestamps]