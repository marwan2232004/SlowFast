import os
import cv2
from slowfast.utils.parser import generate_parser


def prepare_for_annotation(video_path: str, output_dir: str):
    """Extracts one frame per second from a single video and saves them as images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.
                          Will be created if it does not exist.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps)

    print(f"[INFO] Processing {os.path.basename(video_path)}")
    print(f"[INFO] FPS: {fps}, Total Frames: {total_frames}, Duration: {duration}s")

    for ts in range(duration + 1):
        frame_idx = int(ts * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"[WARN] Could not grab frame at {ts}s")
            continue

        out_name = f"{ts:04d}.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, frame)

    cap.release()


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