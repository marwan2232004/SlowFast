import os
import cv2
from slowfast.utils.parser import generate_parser

import subprocess

def prepare_for_annotation(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing {video_name}")
    # Command: ffmpeg -i input.mp4 -vf fps=1 output_%04d.jpg
    cmd = [
        'ffmpeg', 
        '-i', video_path, 
        '-vf', 'fps=1', 
        '-q:v', '2',  # High quality jpeg (2-31, lower is better)
        os.path.join(output_dir, f"%04d.jpg"),
        '-loglevel', 'error', # Quieter output
        '-y' # Overwrite output
    ]
    subprocess.run(cmd)

def prepare_videos_for_annotation(video_dir: str, output_root: str = "frames"):
    """Extracts frames from all videos in a directory and organizes them in subfolders.
    For each video:
        - Creates a folder named after the video (without extension).
        - Calls `prepare_for_annotation` to extract one frame per second.

    Args:
        video_dir (str): Path to the directory containing video files.
        output_root (str): Root directory where all frame folders will be saved.
                           Default is "frames".
    """
    os.makedirs(output_root, exist_ok=True)

    for filename in os.listdir(video_dir):
        if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv",".webm")):
            continue

        video_path = os.path.join(video_dir, filename)
        video_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(output_root, video_name)

        prepare_for_annotation(video_path, output_dir)


def main():
    DESCRIPTION = "Extract Frames from video to be annotated."
    arg_metadata = {
        "video_dir": {
            "help": "Path to the directory containing video files",
            "type": str,
        },
        "output_root": {
            "help": "Root directory where all frame folders will be saved.",
            "type": str,
            "default": "frames"
        }
    }
    parser = generate_parser(arg_metadata, DESCRIPTION)
    args = parser.parse_args()

    prepare_videos_for_annotation(args.video_dir, args.output_root)


if __name__ == "__main__":
    """
       python -m tools.prepare_for_annotation --video_dir "D:/Desktop/Action Recognition/data/videos/videos" --output_root "D:/Desktop/Action Recognition/data/videos/videos/frames"
    """
    main()