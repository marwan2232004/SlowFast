import cv2
import json
import torch
from tqdm.auto import tqdm
from slowfast.utils import logging
from slowfast.visualization.frame_predictor import FrameActionPredictor
import numpy as np

logger = logging.get_logger(__name__)

def my_demo(cfg):
    input_video = cfg.DEMO.INPUT_VIDEO
    output_file = cfg.DEMO.OUTPUT_FILE
    class_names_file = cfg.DEMO.LABEL_FILE_PATH

    logging.setup_logging(cfg.OUTPUT_DIR)

    with open(class_names_file, "r") as f:
        label_map = json.load(f)
    id_to_label = {v: k for k, v in label_map.items()}

    fixed_colors = [
        (0, 255, 100),   # greenish
        (255, 100, 0),   # orange
        (100, 100, 255), # light red/blueish
        (255, 255, 0)    # yellow
    ]

    unique_colors = {
        cls_id: fixed_colors[i % len(fixed_colors)] for i, cls_id in enumerate(id_to_label.keys())
    }

    cap = cv2.VideoCapture(input_video)
    frames = []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if cfg.DEMO.OUTPUT_FPS > 0:
        fps = cfg.DEMO.OUTPUT_FPS

    predictor = FrameActionPredictor(cfg, img_width=width, img_height=height)
    predictor.predict_boxes(frames)

    out = cv2.VideoWriter(
        output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for i in tqdm(range(len(frames)), desc="Processing Frames", colour="green", ncols=100):
        preds, boxes = predictor.predict_frame(frames, i)
        frame = frames[i].copy()

        for box, pred in zip(boxes, preds):
            x1, y1, x2, y2 = map(int, box)
            label_id = torch.argmax(pred).item()
            conf = torch.max(pred).item()
            label_text = id_to_label.get(label_id, "Unknown")

            color = unique_colors.get(label_id, (0, 255, 100))

            # --- Draw bounding box ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

            # --- Label background (top of box) ---
            text = f"{label_text}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_x = max(x1, 0)
            text_y = max(y1 - 8, text_h + 4)
            text_x = min(text_x, width - text_w - 2)

            cv2.rectangle(frame, (text_x, text_y - text_h - 4), (text_x + text_w + 4, text_y), color, -1)
            cv2.putText(
                frame,
                text,
                (text_x + 2, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            # --- Confidence text (smaller, inside bottom of box) ---
            conf_text = f"{conf:.2f}"
            (conf_w, conf_h), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            conf_x = min(x2 - conf_w - 3, width - conf_w - 3)
            conf_y = min(y2 - 3, height - 3)

            cv2.rectangle(frame, (conf_x - 2, conf_y - conf_h - 2), (conf_x + conf_w + 2, conf_y + 2), color, -1)
            cv2.putText(
                frame,
                conf_text,
                (conf_x, conf_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        out.write(frame)

    out.release()
    logger.info(f"Saved output video to: {output_file}")
