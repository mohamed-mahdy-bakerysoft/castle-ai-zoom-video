import os
from emphassess.src.emphasis_classifier.utils.infer_utils import (
    infer_audio,
    Wav2Vec2ForAudioFrameClassification,
)


def save_emphasis_predictions(files, splitted_audio_txt_dir):

    if not os.path.exists(splitted_audio_txt_dir) or (
        os.path.exists(splitted_audio_txt_dir)
        and len(os.listdir(splitted_audio_txt_dir)) != len(files)
    ):
        model = Wav2Vec2ForAudioFrameClassification.from_pretrained(
            "emphassess/src/emphasis_classifier/checkpoints/"
        )
        os.makedirs(splitted_audio_txt_dir, exist_ok=True)
        for f in files:
            pred, emph_boundaries = infer_audio(f, model)

            print("Emphasis boundaries (in seconds): ", emph_boundaries)
            # Get the base name of the audio file
            output_filename = f.split("/")[-1].split(".")[0] + ".txt"

            with open(os.path.join(splitted_audio_txt_dir, output_filename), "w") as f:
                for start, end in emph_boundaries:
                    # Write the interval to the file formatted to two decimal places
                    f.write(f"{start:.2f}-{end:.2f}\n")

            print(
                f"Emphasized intervals saved to {os.path.join(splitted_audio_txt_dir, output_filename)}"
            )