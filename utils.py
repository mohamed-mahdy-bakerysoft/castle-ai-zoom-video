from moviepy.editor import VideoFileClip
from typing import Dict, List
import subprocess
import librosa
import logging
import os
import re
from emphassess.src.emphasis_classifier.utils.infer_utils import infer_audio, Wav2Vec2ForAudioFrameClassification
import torchaudio as ta
from tqdm import tqdm
from glob import glob
from span_api_splitter.utils import get_split_info
from span_api_splitter import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("demo_logs.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

def split_words_by_duration(words: List, sentences_lengths: List):
    splitted_words = []
    for i, length in enumerate(sentences_lengths):
        splitted_words.append(words[:length])
        words = words[length:]
    return splitted_words

def save_audio_from_video(video_path: str, output_path: str):
    """
    Save audio from video

    :param video_path: str: Video path
    :param output_path: str: Output path
    """
    if os.path.exists(output_path):
        logger.info(f"Audio file already exists: {output_path}")
        return
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_path)


def clear_cache(session_state: Dict):
    """
    Clear the session state cache.
    """
    keys = list(key for key in session_state.keys() if key != 'interview_id')
    for key in keys:
        session_state.pop(key)

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        return True
    except FileNotFoundError:
        return False

# def extract_sentences_with_durations(transcript_metadata: dict) -> List[str]:
#     """
#     Extracts sentences with their start and end times from the transcript metadata.

#     Args:
#         transcript_metadata (dict): The transcript metadata.

#     Returns:
#         list: Sentence concated with it's start and end times.
#     """
#     sentences_with_durations = []
#     transcripts_with_alignments = transcript_metadata.get("results", [])
#     sentence = ""
#     sentence_with_words = []
#     for transcript in transcripts_with_alignments:
#         word = transcript["alternatives"][0]["content"]

#         if sentence == "":
#             sentence_with_words.append([])
#             start_time_sentence = transcript["start_time"]
#             start_time = transcript["start_time"]
#             end_time = transcript["end_time"]
#             sentence += word + " "
#             sentence_with_words[-1].append([word, start_time, end_time])
#         elif transcript.get("is_eos", False):  # End of sentence
#             start_time = transcript["start_time"]
#             end_time = transcript["end_time"]
#             sentence = sentence.strip() + word + " "
#             sentences_with_durations.append(f'{sentence}|{start_time_sentence}|{end_time}\n')
#             # sentence_with_words[-1].append([word, start_time, end_time])
#             sentence = ""
#         elif transcript["type"] == 'punctuation':
#             sentence = sentence.strip() + word + " "
#         else:
#             start_time = transcript["start_time"]
#             end_time = transcript["end_time"]
#             sentence_with_words[-1].append([word, start_time, end_time])
#             sentence += word + " "

#     return sentences_with_durations, sentence_with_words

def extract_sentences_with_durations_with_chunks(transcript_metadata: dict, chunk_start:float = None) -> List[str]:
    """
    Extracts sentences with their start and end times from the transcript metadata.

    Args:
        transcript_metadata (dict): The transcript metadata.

    Returns:
        list: Sentence concated with it's start and end times.
    """
    sentences_with_durations = []
    transcripts_with_alignments = transcript_metadata.get("results", [])
    sentence = ""
    sentence_with_words = []
    for transcript in transcripts_with_alignments:
        word = transcript["alternatives"][0]["content"]

        if sentence == "":
            sentence_with_words.append([])
            start_time_sentence = transcript["start_time"]
            start_time = transcript["start_time"]
            end_time = transcript["end_time"]
            sentence += word + " "
            sentence_with_words[-1].append([word, start_time + chunk_start, end_time + chunk_start])
        elif transcript.get("is_eos", False):  # End of sentence
            start_time = transcript["start_time"]
            end_time = transcript["end_time"]
            sentence = sentence.strip() + word + " "
            sentences_with_durations.append(
                {
                    "sentence": sentence,
                    "start_time": start_time_sentence + chunk_start,
                    "end_time": end_time + chunk_start
                }
            )
            # sentence_with_words[-1].append([word, start_time, end_time])
            sentence = ""
        elif transcript["type"] == 'punctuation':
            sentence = sentence.strip() + word + " "
        else:
            start_time = transcript["start_time"]
            end_time = transcript["end_time"]
            sentence_with_words[-1].append([word, start_time + chunk_start, end_time + chunk_start])
            sentence += word + " "

    return sentences_with_durations, sentence_with_words

def split_sentences_by_seconds(sentences_metadata: dict, seconds: int) -> List[str]:
    """
    Extracts sentences with their start and end times from the transcript metadata.

    Args:
        transcript_metadata (dict): The transcript metadata.

    Returns:
        list: Sentence concated with it's start and end times.
    """
    vrk = 0
    sentences = [[]]
    for sentence in sentences_metadata:
        match = re.search(r'\|(\d+\.\d+)$', sentence)
        if match:
            extracted_number = float(match.group(1))
            sentences[-1].append(sentence)
            if extracted_number//seconds > vrk:
                vrk += 1
                sentences.append([])
        else:
            print("No number found after sentence")

    return sentences

# def extract_words_with_durations(transcript_metadata: dict) -> List[str]:
#     """
#     Extracts sentences with their start and end times from the transcript metadata.

#     Args:
#         transcript_metadata (dict): The transcript metadata.

#     Returns:
#         list: Sentence concated with it's start and end times.
#     """
#     words_with_durations = []
#     transcripts_with_alignments = transcript_metadata.get("results", [])
#     index = 1
#     sentence = ""
#     for transcript in transcripts_with_alignments:
#         word = transcript["alternatives"][0]["content"]

#         if transcript["type"] == 'punctuation':
#             sentence = sentence.strip() + word + " "
#         else:
#             start_time = transcript["start_time"]
#             sentence += f"{start_time}" + '|' + word + " "
#             if start_time % 60 >index:
#                 index += 1
#                 sentence = ""
#                 words_with_durations.append(sentence)

#     return words_with_durations

def get_word_indices(full_text, target_text):
    # Clean and split texts
    full_text = re.sub(r'\[.*?\]', '', full_text)
    full_words = full_text.strip().split()
    target_words = target_text.strip().split()

    if len(target_words) <= 2:
        threshold = 1
    else:
        threshold = 0.95
    required_matches = int(len(target_words) * threshold)
    start_idx = -1
    
    for i in range(len(full_words) - len(target_words) + 1):
    # Slice of words to compare
        current_slice = full_words[i:i+len(target_words)]
        
        # Count matching words
        matches = sum(1 for x, y in zip(current_slice, target_words) if x == y)
        
        # Check if the match count meets or exceeds the required threshold
        if matches >= required_matches:
            start_idx = i
            break
            # print("Match found at position:", i)
    # Find the starting index of the sequence
    # start_idx = -1
    # for i in range(len(full_words) - len(target_words) + 1):
    #     if full_words[i:i+len(target_words)] == target_words:
            
            
    if start_idx == -1:
        print("Target text not found in full text")
        return -1, -1
        
    end_idx = start_idx + len(target_words) - 1
    return start_idx, end_idx


def add_silence_duration(word_data):
    start=None
    for i, d in enumerate(word_data):
        for j, w in enumerate(d):
            if j==0 and start==None:
                start = 0
            else:
                start = d[j-1][2]
            if w[1] - start> 0.03:
                w.append(w[1] - start)
            else:
                w.append(0)
        start = w[2]

def split_and_save_audio(audio_path, output_file_path_sentence, splitted_audio_dir):
    sr = librosa.get_samplerate(audio_path)
    if not os.listdir(splitted_audio_dir):
        with open(output_file_path_sentence, 'r') as file:
            for i, line in enumerate(tqdm(file)):
                line_stripped = line.strip()
                start = float(line_stripped.split('|')[-2])
                end = float(line_stripped.split('|')[-1])
                y, sr = ta.load(audio_path, frame_offset=int(start*sr), num_frames=int((end-start) * sr))
                ta.save(f"{splitted_audio_dir}/{audio_path.split('/')[-1].split('.')[0]}_{i}.{audio_path.split('/')[-1].split('.')[1]}", y, sr)


def save_emphasis_predictions(files, splitted_audio_txt_dir):
    
    if not os.path.exists(splitted_audio_txt_dir) or (os.path.exists(splitted_audio_txt_dir) and len(os.listdir(splitted_audio_txt_dir)) != len(files)):
        model = Wav2Vec2ForAudioFrameClassification.from_pretrained("emphassess/src/emphasis_classifier/checkpoints/")   
        os.makedirs(splitted_audio_txt_dir, exist_ok=True)
        for f in files:
            pred, emph_boundaries = infer_audio(f, model)

            print("Emphasis boundaries (in seconds): ", emph_boundaries)
            # Get the base name of the audio file
            output_filename = f.split('/')[-1].split('.')[0] + '.txt'

            with open(os.path.join(splitted_audio_txt_dir, output_filename), 'w') as f:
                for start, end in emph_boundaries:
                    # Write the interval to the file formatted to two decimal places
                    f.write(f"{start:.2f}-{end:.2f}\n")

            print(f"Emphasized intervals saved to {os.path.join(splitted_audio_txt_dir, output_filename)}")


def construct_new_sentences(files, audio_basename, word_data, sentence_info_path_updated, sentence_path):
    pattern = re.compile(rf'{audio_basename.split(".")[0]}_(\d+)\.txt$')
    # Sort the file paths based on the extracted numeric part
    files = sorted(
        files,
        key=lambda x: int(pattern.search(x).group(1))
    )
    with open(sentence_path, 'r') as file:
        sentence_content = file.read()
    sentences = sentence_content.split("\n")[:-1]

    new_sentences = []

    for i, f in tqdm(enumerate(files)):
        with open(f, 'r') as file:
            content = file.read()
        times = content.split("\n")
        for t in times[:-1]:
            st, end = t.split('-')
            start_time = float(st) + word_data[i][0][1]
            end_time = float(end) + word_data[i][0][1]
            if end_time-start_time >=0.18:
                best_match = find_overlapping_words(word_data[i], start_time, end_time )
                for ind in best_match:
                    word_data[i][ind][0] = word_data[i][ind][0].upper()
                # if new_sentence is None:
                #     new_sentence = capitalize_word_by_index(sentences[i], best_match)
                # else:
                #     new_sentence = capitalize_word_by_index(new_sentence, best_match)
        new_s = format_sentence_with_silence(word_data[i], sentences[i])
        add_sentences_to_file(new_s, sentence_info_path_updated)
        new_sentences.append(new_s + "\n")

    return new_sentences

def find_overlapping_words(word_times, start_time, end_time, overlap_threshold=0.5):
    """
    Find indices of words that overlap with a given time range by at least the specified threshold.
    
    Args:
        word_times: List of [word, start, end] lists
        start_time: Start time of the range to check
        end_time: End time of the range to check
        overlap_threshold: Minimum overlap ratio required (default: 0.5)
        
    Returns:
        List of indices where words meet the overlap threshold
    """
    overlapping_indices = []
    
    for i, (word, word_start, word_end, _) in enumerate(word_times):
        # Calculate word duration and overlap duration
        word_duration = word_end - word_start
        
        overlap_start = max(start_time, word_start)
        overlap_end = min(end_time, word_end)
        
        if overlap_end > overlap_start:  # There is some overlap
            overlap_duration = overlap_end - overlap_start
            overlap_ratio = overlap_duration / word_duration
            
            if overlap_ratio >= overlap_threshold:
                overlapping_indices.append(i)
    
    return overlapping_indices

def capitalize_word_by_index(sentence, indices):
    """
    Capitalizes words at specified indices in a sentence.
    
    Args:
        sentence: String containing the sentence and timing info
        indices: List of indices of words to capitalize
        
    Returns:
        Modified sentence with specified words capitalized
    """
    # Split the sentence and timing info
    text, start_time, end_time = sentence.rsplit('|', 2)
    
    # Split into words while preserving punctuation
    words = text.split()
    
    # Capitalize words at specified indices
    for idx in indices:
        if 0 <= idx < len(words):
            # Handle punctuation
            word = words[idx]
            punctuation = ''
            while word and word[-1] in '.,!?':
                punctuation = word[-1] + punctuation
                word = word[:-1]
            
            # Capitalize and reconstruct with punctuation
            words[idx] = word.upper() + punctuation
    
    # Reconstruct the sentence with timing info
    return f"{' '.join(words)}|{start_time}|{end_time}"

def format_sentence_with_silence(word_timings, sentence):
    """
    Format a sentence with silence indicators based on word timings.
    
    Args:
        word_timings: List of [word, start, end, silence] lists
        sentence: Original sentence with timing info
        
    Returns:
        Formatted sentence with silence indicators
    """
    # Remove timing info from original sentence
    text = sentence.split('|')[0]
    
    # Split into words while preserving punctuation
    formatted_words = []
    
    for word_info in word_timings:
        word, _, _, silence = word_info
        
        # Add silence indicator if there's silence
        if silence > 0:
            formatted_words.append(f"[{silence:.2f}]{word}")
        else:
            formatted_words.append(word)
    
    # Reconstruct the sentence
    # Get the start and end times from the original sentence
    _, start_time, end_time = sentence.rsplit('|', 2)
    
    # Join words with spaces and add timing info
    formatted_sentence = ' '.join(formatted_words) + f" |{start_time}|{end_time}"
    
    # Add punctuation
    formatted_sentence = formatted_sentence.replace(" ,", ",")
    formatted_sentence = formatted_sentence.replace(" .", ".")
    
    return formatted_sentence

def add_sentences_to_file(sentence, filename):
    """
    Appends new sentences to a file, ensuring each sentence is on a new line.

    Args:
        sentences (list): List of sentences to add.
        filename (str): Path to the text file.
    """
    #sentence = add_numbering_to_sentences(sentence, i)
    with open(filename, 'a') as file:  # Open the file in append mode
        file.write(sentence + '\n')  # Write each sentence followed by a newline


# def add_numbering_to_sentences(sentence, i):
#     """
#     Adds numbering before each sentence in a list.

#     Args:
#         sentences (list): List of sentences.

#     Returns:
#         list: Numbered sentences.
#     """
#     return f"{i+1}. {sentence}"


def split_audio_into_chunks(audio_path: str) -> List[str]:
    """
    Split audio into chunks and return chunk paths.
    """
    outputDir = '.cache/tmp/'
    os.makedirs(outputDir, exist_ok=True)
    payload = {
        'fileUrl': audio_path,
        'resultId': '3c5c778553a73441ac9d57622ed4442a'
    }

    split_info = get_split_info(payload, config)
    logger.info(f"Split info: {split_info}")

    if not split_info:  # No splitting points
        # Convert the audio to wav
        command = f"ffmpeg -i {audio_path} -ac 1 {outputDir}/part_000000000.wav"
        os.system(command)
        return [f'{outputDir}/part_000000000.wav']

    segments = ",".join([str(x) for x in split_info['splittingPoints']])

    command = "ffmpeg -i " + audio_path + " " \
        + "-f segment -segment_times " + segments \
        + " -reset_timestamps 1 -map 0:a -c:a pcm_s16le " \
        + outputDir + "/part_%09d.wav"

    os.system(command)

    splitted_audio_files = sorted(glob(outputDir + '/part*.wav'), key=lambda x: int(os.path.basename(x)[:-len('.wav')].split('_')[-1]))
    return splitted_audio_files
