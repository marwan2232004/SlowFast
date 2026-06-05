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


def dataset_to_ava(dataset_path):
    """
    Process a YOLO dataset into AVA-like annotations.
    """
    rows = []

    dataset_path = Path(dataset_path)
    data_yaml = dataset_path / "data.yaml"
    labels_folder = dataset_path / "train" / "labels"

    if not (data_yaml.exists() and labels_folder.is_dir()):
        raise FileNotFoundError(f"Could not find {data_yaml} or {labels_folder}")

    with open(data_yaml, "r") as f:
        ydata = yaml.safe_load(f)

    names = ydata["names"]

    print(f"\n📂 Processing dataset: \033[1;34m{dataset_path}\033[0m")

    for lbl_file in tqdm(sorted(labels_folder.glob("*.txt")), desc="   → Labels"):
        stem = lbl_file.stem

        # Example:
        # video1_123_jpg.rf.abc123
        # -> video_name=video1
        # -> frame_second=123
        parts = stem.split("_")

        if len(parts) < 2:
            print(f"⚠️ Skipping malformed filename: {lbl_file.name}")
            continue

        video_name = parts[0]

        try:
            frame_second = int(parts[1])
        except ValueError:
            print(f"⚠️ Invalid frame number in {lbl_file.name}")
            continue

        rows.extend(
            parse_label_file(
                lbl_file,
                names,
                video_name,
                frame_second,
            )
        )

    print(f"   ✅ Collected {len(rows)} annotations")
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
    Convert YOLO dataset annotations (YOLO format) to AVA-like CSV.
    """
    dataset_root = Path(dataset_root)

    print(
        f"🚀 Starting conversion from dataset root: " f"\033[1;33m{dataset_root}\033[0m"
    )

    data = dataset_to_ava(dataset_root)

    unique_actions = sorted({row[6] for row in data})
    actions_dict = {action: idx + 1 for idx, action in enumerate(unique_actions)}

    final_data = []
    for row in data:
        final_data.append(row[:6] + [actions_dict[row[6]]] + [row[7]])

    output_csv = os.path.join(output_dir, "output.csv")
    output_pbtxt = os.path.join(output_dir, "action_list.pbtxt")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(final_data)

    print(
        f"\n✅ \033[1;32mCSV saved to {output_csv} "
        f"with {len(final_data)} rows.\033[0m"
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
        },
    }
    parser = generate_parser(arg_metadata, DESCRIPTION)
    args = parser.parse_args()

    yolo_dataset_to_ava(args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    """
    python -m tools.convert_to_ava --dataset_dir "D:/Desktop/Action Recognition/data/videos/annotations" --output_dir "D:/Desktop/Action Recognition/data"
    """
    main()
