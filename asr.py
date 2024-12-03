import os
import json
from concurrent import futures
from glob import glob
from typing import List, Dict

import streamlit as st
from speechmatics.models import ConnectionSettings
from speechmatics.batch_client import BatchClient
from httpx import HTTPStatusError, ReadError
import librosa

from utils.audio_text_utils import (
    extract_sentences_with_durations_with_chunks,
    logger,
    split_audio_into_chunks,
)

CONF = {
    "type": "transcription",
    "transcription_config": {
        "language": "en",
        "enable_entities": True,
    },
}


def get_audio_files(directory: str) -> List[str]:
    """
    Get audio files from a directory

    :param directory: str: Directory
    :return: list: Audio files
    """
    return sorted(
        glob(f"{directory}/*.mp3"), key=lambda x: int(x.split("/")[-1].split("_")[0])
    )


def get_video_files(directory: str) -> List[str]:
    """
    Get video files from a directory

    :param directory: str: Directory
    :return: list: Video files
    """
    return sorted(
        glob(f"{directory}/*.mp4"), key=lambda x: int(x.split("/")[-1].split("_")[0])
    )


def get_client_settings() -> ConnectionSettings:
    """
    Get client settings

    :return: ConnectionSettings: Client settings
    """
    return ConnectionSettings(
        url="https://asr.api.speechmatics.com/v2",
        auth_token=os.getenv("SPEECHMATICS_AUTH_TOKEN"),
    )


def transcribe_audios(
    audio_files: List[str], client_settings: ConnectionSettings
) -> Dict[int, List[str]]:
    """
    Transcribe audio files and save each of them.
    If the transcription already exists, it will be read.

    :param interview_id: str: Interview ID
    :param client_settings: ConnectionSettings: SpeechMatics Client settings

    :return: tuple: Transcription of audio files in splited to sentences and indices of the corresponding sentences.
    """

    # Initializing the progress bar
    progress_bar = st.progress(0, text="Transcribing recordings...")

    index_to_transcription_meta = {}

    with futures.ThreadPoolExecutor() as executor:
        future_to_index = {}

        for i, audio_file in enumerate(audio_files):
            future = executor.submit(transcribe_audio, audio_file, client_settings)
            future_to_index[future] = i

        for j, future in enumerate(futures.as_completed(future_to_index)):
            i = future_to_index[future]
            index_to_transcription_meta[i] = future.result()

            # Update the progress bar
            progress_bar.progress(
                (j + 1) / len(audio_files), text="Transcribing recordings..."
            )

    # Close the progress bar
    progress_bar.empty()
    st.write(
        f"Transcription procress completed. Number of transcriptions: {len(index_to_transcription_meta)}."
    )

    return index_to_transcription_meta


def transcribe_audio_chunk(
    audio_file: str,
    chunk_start_in_secs: float,
    client_settings: ConnectionSettings,
) -> List[Dict]:
    """
    Transcribe a single audio file if transcription does not exist.

    :param audio_file: str: Audio file
    :param chunk_start_in_secs: float: Start time of the chunk
    :param client_settings: ConnectionSettings: SpeechMatics Client settings
    """

    with BatchClient(client_settings) as client:
        try:
            job_id = client.submit_job(
                audio=audio_file,
                transcription_config=CONF,
            )
            logger.info(f"job {job_id} submitted successfully, waiting for transcript")

            # Note that in production, you should set up notifications instead of polling.
            transcript_meta = client.wait_for_completion(
                job_id, transcription_format="json"
            )

        except ReadError:
            logger.info(
                "ReadError - This is likely a connection issue. Check your internet connection and try again."
            )
        except HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.info(
                    "Invalid API key - Check your API_KEY at the top of the code!"
                )
            else:
                raise e
        else:
            transcript_with_durations, transcript_word_durations = (
                extract_sentences_with_durations_with_chunks(
                    transcript_meta, chunk_start_in_secs
                )
            )

    return transcript_with_durations, transcript_word_durations


def transcribe_audio_chunks(
    audio_chunks: List[str], client_settings: ConnectionSettings
) -> List[str]:
    """
    Transcribe audio chunks and save each of them.

    :param audio_chunks: list: Audio chunk paths
    :param client_settings: ConnectionSettings: SpeechMatics Client settings

    :return: List: Transcriptions with there start and end times
    """
    # Initializing the progress bar
    progress_bar = st.progress(0, text="Transcribing recordings...")
    index_to_chunk_transcript = {}
    index_to_chunk_transcript_words = {}
    chunk_start = 0

    with futures.ThreadPoolExecutor() as executor:
        future_to_index = {}
        for i, chunk_path in enumerate(audio_chunks):

            audio, sampling_rate = librosa.load(chunk_path, sr=None)
            chunk_duration = audio.shape[0] / sampling_rate

            future = executor.submit(
                transcribe_audio_chunk, chunk_path, chunk_start, client_settings
            )
            future_to_index[future] = i

            chunk_start += chunk_duration

        for j, future in enumerate(futures.as_completed(future_to_index)):
            i = future_to_index[future]
            res = future.result()
            index_to_chunk_transcript[i] = res[0]
            index_to_chunk_transcript_words[i] = res[1]
            # Update the progress bar
            progress_bar.progress(
                (j + 1) / len(audio_chunks), text="Transcribing recordings..."
            )

    # Close the progress bar
    progress_bar.empty()
    st.write(
        f"Transcription procress completed. Number of transcriptions: {len(index_to_chunk_transcript)}."
    )

    # Merging the chunks transcriptions
    merged_chunks_transcript = []
    for i in range(len(index_to_chunk_transcript)):
        merged_chunks_transcript.extend(
            [
                f"{chunk_info['sentence']}|{chunk_info['start_time']}|{chunk_info['end_time']}\n"
                for chunk_info in index_to_chunk_transcript[i]
            ]
        )
    merged_chunks_words = []
    for i in range(len(index_to_chunk_transcript)):
        merged_chunks_words.extend(
            [
                sentence_info_words
                for sentence_info_words in index_to_chunk_transcript_words[i]
            ]
        )

    # Deleting the temporary audio files
    os.system("rm -rf .cache/tmp/*")

    return merged_chunks_transcript, merged_chunks_words


def transcribe_audio(
    audio_file: str,
    output_file_path_sentence: str,
    output_file_path_words: str,
    client_settings: ConnectionSettings,
) -> str:
    """
    Transcribe a single audio file if transcription does not exist.

    :param audio_file: str: Audio file
    :param client_settings: ConnectionSettings: SpeechMatics Client settings
    """

    if os.path.exists(output_file_path_sentence) and os.path.exists(
        output_file_path_words
    ):
        logger.info(
            f"File {output_file_path_sentence} and {output_file_path_words} already exists"
        )

        # Reading the transcription
        with open(output_file_path_sentence, "r") as f:
            transcript_with_durations_sentence = f.readlines()

        with open(output_file_path_words) as f:
            transcript_word_durations_sentence = json.load(f)

        return transcript_with_durations_sentence, transcript_word_durations_sentence
    else:
        audio_chunks_paths = split_audio_into_chunks(audio_file)
        audio_transcript, transcript_word_durations_sentence = transcribe_audio_chunks(
            audio_chunks_paths, client_settings
        )
        if (
            audio_transcript
        ):  # If transcription is empty no need to create an empty txt file
            with open(output_file_path_sentence, "w") as f:
                f.writelines(audio_transcript)
            # Saving the transcription in txt file

            logger.info(f"Transcription sentence saved to {output_file_path_sentence}")
            with open(output_file_path_words, "w") as file:
                json.dump(transcript_word_durations_sentence, file)

        return audio_transcript, transcript_word_durations_sentence
