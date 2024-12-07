
import csv

def timecode_to_seconds(timecode):
    # Convert "HH:MM:SS.nnn" to seconds
    h, m, s = timecode.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

def read_scenes_with_seconds(csv_path):
    scenes = []
    with open(csv_path, 'r') as f:
        next(f)  # Skip header
        next(f)  # Skip separator
        reader = csv.reader(f)
        for row in reader:
            if row:
                scenes.append({
                    'scene': row[0],
                    'start_time': timecode_to_seconds(row[2]),
                    'end_time': timecode_to_seconds(row[5])
                })
    return scenes


def find_broll_boundaries(word_data, scenes_data):
    """
    Find the sentence and word indices where each B-roll scene starts and ends.
    
    Args:
        word_data: List of sentences, where each sentence is a list of [word, start, end] lists
        scenes_data: List of scene dictionaries with keys: scene, start_time, end_time, Broll
    
    Returns:
        List of tuples (start_sent_idx, start_word_idx, end_sent_idx, end_word_idx)
        representing where to add opening and closing brackets for each B-roll scene
    """
    broll_boundaries = []
    
    # Process only B-roll scenes
    broll_scenes = [scene for scene in scenes_data if scene['Broll']]
    
    for scene in broll_scenes:
        scene_start = scene['start_time']
        scene_end = scene['end_time']
        
        start_sent_idx = end_sent_idx = None
        start_word_idx = end_word_idx = None
        
        # Find start position
        for sent_idx, sentence_words in enumerate(word_data):
            sent_start = sentence_words[0][1]
            sent_end = sentence_words[-1][2]
            
            # Skip if sentence is before scene start
            if sent_end < scene_start:
                continue
                
            # Find starting position
            if start_sent_idx is None:
                start_sent_idx = sent_idx
                # Find the first word that starts after scene_start
                for word_idx, (_, word_start, word_end, _) in enumerate(sentence_words):
                    if word_start >= scene_start:
                        start_word_idx = word_idx
                        break
                if start_word_idx is None:  # If no specific word found, start at first word
                    start_sent_idx = start_sent_idx + 1
                    start_word_idx = 0
            
            # Find ending position
            if sent_start <= scene_end and sent_end >= scene_end:
                end_sent_idx = sent_idx
                # Find the last word that ends before scene_end
                for word_idx, (_, _, word_end, _) in enumerate(sentence_words):
                    if word_end > scene_end:
                        end_word_idx = word_idx
                        break
                if word_end <= scene_end:  # If scene ends after sentence
                    end_word_idx = len(sentence_words) - 1
                        
            
            # Can stop if we've passed the scene end
            if sent_start > scene_end:
                break
        
            if start_sent_idx is not None and end_sent_idx is not None:
                broll_boundaries.append((
                    start_sent_idx,
                    start_word_idx,
                    end_sent_idx,
                    end_word_idx
                ))
    
    return broll_boundaries
