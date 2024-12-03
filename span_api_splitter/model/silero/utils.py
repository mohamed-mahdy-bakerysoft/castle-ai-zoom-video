import wave
import contextlib
from typing import Callable
from collections import deque

import torch
import torchaudio
import torch.nn.functional as F

from span_api_splitter import literals


def read_wave(path: str, start_pos: int, n_frames: int) -> bytes:
    """
    Reads a wav file.
    Args:
        path: Path to the wav file.
        start_pos: Starts reading the file from start_pos.
        n_frames: Number of frames to read.
    Returns:
        PCM audio data.
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        wf.setpos(start_pos)
        return wf.readframes(n_frames)


def write_wave(path: str, audio: bytes) -> None:
    """
    Writes a wav file.
    Args:
        path: Path to write to.
        audio: PCM audio data.
    Returns:
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(literals.SAMPLE_RATE)
        wf.writeframes(audio)


def read_audio(wav_path: str) -> torch.Tensor:
    """
    Reads a wav file.
    Args:
        wav_path: Path to the wav file.
    Returns:
        Float audio data.
    """
    wav, sr = torchaudio.load(wav_path)
    assert sr == literals.SAMPLE_RATE
    return wav.squeeze(0)


def validate(model, inputs: torch.Tensor) -> torch.Tensor:
    """
    Runs inference.
    """
    with torch.no_grad():
        outs = model(inputs)
    return outs


def get_speech_ts(wav: torch.Tensor, model, trig_sum: float = 0.25, neg_trig_sum: float = 0.07, num_steps: int = 8,
                  batch_size: int = 200, num_samples_per_window: int = 4000, min_speech_samples: int = 10000,  # samples
                  min_silence_samples: int = 500, run_function: Callable = validate) -> list:
    """
    This function is used for splitting long audios into speech chunks using silero VAD
    Args:
        wav: The audio waveform.
        model: Path to the silero vad model parameters.
        trig_sum: Overlapping windows are used for each audio chunk, trig sum defines average
            probability among those windows for switching into triggered state (speech state).
        neg_trig_sum: Same as trig_sum, but for switching from triggered to non-triggered state (non-speech).
        num_steps: Number of overlapping windows to split audio chunk into.
        batch_size: Batch size to feed to silero VAD.
        num_samples_per_window: Window size in samples (chunk length in samples to feed to NN).
        min_speech_samples: If speech duration is shorter than this value, do not consider it speech.
        min_silence_samples: Number of samples to wait before considering as the end of speech.
        run_function: Function to use for the model call.
    Returns:
        List containing ends and beginnings of speech chunks (in seconds).
    """

    # logger.debug("Running VAD")
    num_samples = num_samples_per_window
    assert num_samples % num_steps == 0
    step = int(num_samples / num_steps)  # stride / hop
    outs, to_concat, speeches = [], [], []
    sr = literals.SAMPLE_RATE

    for i in range(0, len(wav), step):
        chunk = wav[i: i + num_samples]
        if len(chunk) < num_samples:
            chunk = F.pad(chunk, (0, num_samples - len(chunk)))
        to_concat.append(chunk.unsqueeze(0))
        if len(to_concat) >= batch_size:
            chunks = torch.Tensor(torch.cat(to_concat, dim=0))
            out = run_function(model, chunks)
            outs.append(out)
            to_concat = []

    if to_concat:
        chunks = torch.Tensor(torch.cat(to_concat, dim=0))
        out = run_function(model, chunks)
        outs.append(out)

    outs = torch.cat(outs, dim=0)

    buffer = deque(maxlen=num_steps)  # maxlen reached => first element dropped
    triggered = False
    current_speech = dict()

    speech_probs = outs[:, 1]  # this is very misleading
    temp_end = 0
    for i, predict in enumerate(speech_probs):  # add name
        buffer.append(predict)
        smoothed_prob = (sum(buffer) / len(buffer))
        if (smoothed_prob >= trig_sum) and temp_end:
            temp_end = 0
        if (smoothed_prob >= trig_sum) and not triggered:
            triggered = True
            current_speech[0] = step * max(0, i - num_steps)
            continue
        if (smoothed_prob < neg_trig_sum) and triggered:
            if not temp_end:
                temp_end = step * i
            if step * i - temp_end < min_silence_samples:
                continue
            else:
                current_speech[1] = temp_end
                if (current_speech[1] - current_speech[0]) > min_speech_samples:
                    speeches.append([round(current_speech[0]/sr, literals.PRECISION),
                                     round(current_speech[1]/sr, literals.PRECISION)])
                temp_end = 0
                current_speech = dict()
                triggered = False
                continue
    if current_speech:
        current_speech[1] = len(wav)
        speeches.append([round(current_speech[0]/sr, literals.PRECISION),
                         round(current_speech[1]/sr, literals.PRECISION)])
    return speeches


def make_silero_timestamps_for_audio_part(wav_part_path: str, part_start: int, part_n_frames: int, part_num: int,
                                          file_path: str, vad_params: str, trig_sum: float, neg_trig_sum: float,
                                          num_steps: int, batch_size: int, num_samples_per_window: int,
                                          min_speech_samples: int, min_silence_samples: int) -> list:
    """
    Makes timestamps for audio part.
    Args:
        wav_part_path: Path of the audio part to be saved.
        part_start: Starts reading the file from start_pos.
        part_n_frames: Number of frames to read.
        part_num: Number of audio part.
        file_path: Path of the whole audio.
        vad_params: path to the silero vad model parameters.
        trig_sum: Overlapping windows are used for each audio chunk, trig sum defines average probability among
            those windows for switching into triggered state (speech state).
        neg_trig_sum: Same as trig_sum, but for switching from triggered to non-triggered state (non-speech).
        num_steps: Number of overlapping windows to split audio chunk into.
        batch_size: Batch size to feed to silero VAD.
        num_samples_per_window: Window size in samples (chunk length in samples to feed to NN).
        min_speech_samples: If speech duration is shorter than this value, do not consider it speech.
        min_silence_samples: Number of samples to wait before considering as the end of speech.
    Returns:
        List containing the audio part number and ends and beginnings of speech chunks in that part (in seconds).
    """
    model = torch.jit.load(vad_params)
    audio = read_wave(path=file_path, start_pos=part_start, n_frames=part_n_frames)
    write_wave(path=wav_part_path, audio=audio)
    audio = read_audio(wav_part_path)
    vad_timestamps = get_speech_ts(wav=audio, model=model, trig_sum=trig_sum, neg_trig_sum=neg_trig_sum,
                                   num_steps=num_steps, batch_size=batch_size,
                                   num_samples_per_window=num_samples_per_window, min_speech_samples=min_speech_samples,
                                   min_silence_samples=min_silence_samples)
    return [part_num, vad_timestamps]