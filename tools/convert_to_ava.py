import csv
import yaml
import os
from pathlib import Path
from tqdm import tqdm
from slowfast.utils.parser import generate_parser


def parse_label_file(lbl_path, names, video_folder, frame_second):
    """
    Parse a YOLO label file and convert annotations to AVA-like format.
    """
    rows = []
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id, x_c, y_c, w, h = parts
            cls_id = int(cls_id)
            x_c, y_c, w, h = map(float, (x_c, y_c, w, h))

            x1 = x_c - w / 2
            y1 = y_c - h / 2
            x2 = x_c + w / 2
            y2 = y_c + h / 2

            # Get label (action + person_id)
            label = names[cls_id]
            if "_" in label:
                action, pid = label.rsplit("_", 1)
                try:
                    pid = int(pid)
                except ValueError:
                    pid = -1
            else:
                action, pid = label, -1

            rows.append(
                [
                    video_folder,
                    frame_second,
                    x1,
                    y1,
                    x2,
                    y2,
                    action,
                    pid,
                ]
            )
    return rows


def video_to_ava(video_path):
    """
    Process a single video folder into AVA-like annotations.
    """
    rows = []
    video_folder = os.path.basename(video_path)
    data_yaml = Path(video_path) / "data.yaml"
    labels_folder = Path(video_path) / "train" / "labels"

    print(data_yaml)
    print(labels_folder)

    if not (data_yaml.exists() and labels_folder.is_dir()):
        print(f"⚠️  Skipping {video_folder}: no data.yaml or labels folder")
        return None

    with open(data_yaml, "r") as f:
        ydata = yaml.safe_load(f)

    names = ydata["names"]

    print(f"\n📂 Processing video folder: \033[1;34m{video_folder}\033[0m")

    for lbl_file in tqdm(sorted(labels_folder.glob("*.txt")), desc=f"   → Labels"):
        prefix = lbl_file.name.split("_jpg")[0]
        try:
            frame_second = int(prefix)
        except ValueError:
            frame_second = -1

        rows.extend(parse_label_file(lbl_file, names, video_folder, frame_second))

    print(f"   ✅ Collected {len(rows)} annotations from {video_folder}")
    return rows


def save_pbtxt(actions_dict, pbtxt_path):
    """
    Save action → id mapping into a .pbtxt file.
    """
    with open(pbtxt_path, "w") as f:
        for action, aid in sorted(actions_dict.items(), key=lambda x: x[1]):
            f.write("item {\n")
            f.write(f'  name: "{action}"\n')
            f.write(f"  id: {aid}\n")
            f.write("}\n")
    print(f"📄 Action mapping saved to {pbtxt_path}")


def yolo_dataset_to_ava(dataset_root, output_dir):
    """
    Convert YOLO dataset annotations (YOLOv11 format) to AVA-like CSV.
    """
    dataset_root = Path(dataset_root)
    data = []

    print(f"🚀 Starting conversion from dataset root: \033[1;33m{dataset_root}\033[0m")

    for video_folder in dataset_root.iterdir():
        if not video_folder.is_dir():
            continue

        rows = video_to_ava(video_folder)
        if rows:
            data.extend(rows)

    unique_actions = sorted({row[6] for row in data})
    actions_dict = {action: idx + 1 for idx, action in enumerate(unique_actions)}

    final_data = []
    for row in data:
        row_with_id = row[:6] + [actions_dict[row[6]]] + [row[7]]
        final_data.append(row_with_id)

    output_csv = os.path.join(output_dir, "output.csv")
    output_pbtxt = os.path.join(output_dir, "action_list.pbtxt")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(final_data)

    print(
        f"\n✅ \033[1;32mCSV saved to {output_csv} with {len(final_data)} rows.\033[0m"
    )

    save_pbtxt(actions_dict, output_pbtxt)


def main():
    DESCRIPTION = "Convert YOLO annotations to AVA-like CSV format."
    arg_metadata = {
        "dataset_dir": {
            "help": "Path to dataset dir containing video sub folders",
            "type": str,
        },
        "output_dir": {
            "help": "Path to output dir",
            "type": str,
        }
    }
    parser = generate_parser(arg_metadata, DESCRIPTION)
    args = parser.parse_args()

    yolo_dataset_to_ava(args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    """
    python -m tools.convert_to_ava --dataset_dir "D:/Desktop/Action Recognition/data/videos/annotations" --output_dir "D:/Desktop/Action Recognition/data" 
    """
    main()
