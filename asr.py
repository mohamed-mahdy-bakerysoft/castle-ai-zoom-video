import os
import json
from concurrent import futures
from glob import glob
from typing import List, Tuple, Dict

import streamlit as st
from speechmatics.models import ConnectionSettings
from speechmatics.batch_client import BatchClient
from httpx import HTTPStatusError, ReadError

from utils import extract_sentences_with_durations, extract_words_with_durations, logger

CONF = {
    "type": "transcription",
    "transcription_config": { 
        "language": 'en',
        "enable_entities": True,
    },
}

def get_audio_files(directory: str) -> List[str]:
    """
    Get audio files from a directory

    :param directory: str: Directory
    :return: list: Audio files
    """
    return sorted(glob(f'{directory}/*.mp3'), key=lambda x: int(x.split('/')[-1].split('_')[0]))


def get_video_files(directory: str) -> List[str]:
    """
    Get video files from a directory

    :param directory: str: Directory
    :return: list: Video files
    """
    return sorted(glob(f'{directory}/*.mp4'), key=lambda x: int(x.split('/')[-1].split('_')[0]))


def get_client_settings() -> ConnectionSettings:
    """
    Get client settings

    :return: ConnectionSettings: Client settings
    """
    return ConnectionSettings(
        url='https://asr.api.speechmatics.com/v2',
        auth_token= os.getenv('SPEECHMATICS_AUTH_TOKEN')
    )


def transcribe_audios(
        audio_files: List[str],
        client_settings: ConnectionSettings
        ) -> Dict[int, List[str]]:
    """
    Transcribe audio files and save each of them.
    If the transcription already exists, it will be read.
    
    :param interview_id: str: Interview ID
    :param client_settings: ConnectionSettings: SpeechMatics Client settings

    :return: tuple: Transcription of audio files in splited to sentences and indices of the corresponding sentences.
    """

    # Initializing the progress bar
    progress_bar = st.progress(0, text='Transcribing recordings...')

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
            progress_bar.progress((j + 1) / len(audio_files), text='Transcribing recordings...')
    
    # Close the progress bar
    progress_bar.empty()
    st.write(f'Transcription procress completed. Number of transcriptions: {len(index_to_transcription_meta)}.')

    return index_to_transcription_meta


def transcribe_audio(audio_file: str, 
                     output_file_path_sentence: str,
                     output_file_path_words: str, 
                     client_settings: ConnectionSettings) -> str:
    """
    Transcribe a single audio file if transcription does not exist.

    :param audio_file: str: Audio file
    :param client_settings: ConnectionSettings: SpeechMatics Client settings
    """
    
    # os.makedirs('./uploaded_files/transcriptions', exist_ok=True)
    # output_file_path_sentence = audio_file \
    #     .replace('recordings', 'transcriptions') \
    #     .replace('.mp3', '_trancriptions_with_align_sentence.txt')
    
    # output_file_path_words = audio_file \
    #     .replace('recordings', 'transcriptions') \
    #     .replace('.mp3', '_trancriptions_with_align_words.json')
    
    # Check if the transcription already exists
    if os.path.exists(output_file_path_sentence) and  os.path.exists(output_file_path_words) :
        logger.info(f'File {output_file_path_sentence} and {output_file_path_words} already exists')

        # Reading the transcription
        with open(output_file_path_sentence, 'r') as f:
            transcript_with_durations_sentence = f.readlines()
        
        with open(output_file_path_words) as f:
            transcript_word_durations_sentence = json.load(f)
         
        return transcript_with_durations_sentence, transcript_word_durations_sentence
    

    with BatchClient(client_settings) as client:
        try:
            job_id = client.submit_job(
                audio=audio_file,
                transcription_config=CONF,
                
            )
            logger.info(f"job {job_id} submitted successfully, waiting for transcript")

            # Note that in production, you should set up notifications instead of polling. 
            transcript_meta = client.wait_for_completion(job_id, transcription_format="json")
            
        except ReadError:
            logger.info("ReadError - This is likely a connection issue. Check your internet connection and try again.")
        except HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.info('Invalid API key - Check your API_KEY at the top of the code!')
            else:
                raise e
        else:
            transcript_with_durations_sentence, transcript_word_durations_sentence = extract_sentences_with_durations(transcript_meta)
            #Saving the transcription in txt file
            with open(output_file_path_sentence, 'w') as f:
                f.writelines(transcript_with_durations_sentence)

            logger.info(f"Transcription sentence saved to {output_file_path_sentence}")
            with open(output_file_path_words, 'w') as file:
                json.dump(transcript_word_durations_sentence, file)

    return transcript_with_durations_sentence, transcript_word_durations_sentence
