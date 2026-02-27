import argparse

from deepface_analytics.app import main

parser = argparse.ArgumentParser(description="Face Counter & Emotion Analyzer")
parser.add_argument(
    "--no-deepface",
    action="store_true",
    help="Run without DeepFace (detection only, no emotion analysis)",
)
args = parser.parse_args()

main(no_deepface=args.no_deepface)
