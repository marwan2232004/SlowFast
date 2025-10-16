import cv2
import torch
from slowfast.utils import logging
from slowfast.config.defaults import get_cfg
from slowfast.visualization.frame_predictor import FrameActionPredictor
from tqdm.auto import tqdm
logger = logging.get_logger(__name__)

def my_demo(cfg):
    input_video = cfg.DEMO.INPUT_VIDEO
    output_file = cfg.DEMO.OUTPUT_FILE
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

    for i in tqdm(range(len(frames)), desc="Processing Frames"):
        preds, boxes = predictor.predict_frame(frames, i)
        frame = frames[i].copy()

        for box, pred in zip(boxes, preds):
            x1, y1, x2, y2 = map(int, box)
            label_id = torch.argmax(pred).item()
            conf = torch.max(pred).item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Action {label_id} ({conf:.2f})",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        out.write(frame)

    out.release()
    logger.info("Saved output.mp4")
