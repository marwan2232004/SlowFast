import cv2
import json
import torch
from tqdm.auto import tqdm
from slowfast.utils import logging
from slowfast.config.defaults import get_cfg
from slowfast.visualization.frame_predictor import FrameActionPredictor

logger = logging.get_logger(__name__)

def my_demo(cfg):
    input_video = cfg.DEMO.INPUT_VIDEO
    output_file = cfg.DEMO.OUTPUT_FILE
    class_names_file = cfg.DEMO.LABEL_FILE_PATH

    logging.setup_logging(cfg.OUTPUT_DIR)

    with open(class_names_file, "r") as f:
        label_map = json.load(f)
    id_to_label = {v: k for k, v in label_map.items()}


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

            label_text = f"{id_to_label.get(label_id, 'Unknown')} ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 1, cv2.LINE_AA)

            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w + 2, y1), (0, 255, 100), -1)

            cv2.putText(
                frame,
                label_text,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        out.write(frame)

    out.release()
    logger.info(f"Saved output video to: {output_file}")
