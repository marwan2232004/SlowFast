import cv2
import json
import torch
from tqdm.auto import tqdm
from collections import deque
from slowfast.utils import logging
from slowfast.visualization.frame_predictor import FrameActionPredictor

logger = logging.get_logger(__name__)

# --- CONSTANTS ---
TARGET_WORK_LABEL = "work"
FIXED_COLORS = [
    (0, 255, 100),   # greenish
    (255, 100, 0),   # orange
    (100, 100, 255), # light red/blueish
    (255, 255, 0)    # yellow
]

def load_label_map(label_path):
    """Loads the label map and creates color mapping."""
    with open(label_path, "r") as f:
        label_map = json.load(f)
    id_to_label = {v: k for k, v in label_map.items()}
    unique_colors = {
        cls_id: FIXED_COLORS[i % len(FIXED_COLORS)] 
        for i, cls_id in enumerate(id_to_label.keys())
    }
    return id_to_label, unique_colors

def draw_predictions(frame, boxes, preds, id_to_label, unique_colors):
    """Draws original bounding boxes, labels, and confidence scores."""
    work_count = 0
    total_count = len(boxes)
    width, height = frame.shape[1], frame.shape[0]

    for box, pred in zip(boxes, preds):
        x1, y1, x2, y2 = map(int, box)
        label_id = torch.argmax(pred).item()
        conf = torch.max(pred).item()
        label_text = id_to_label.get(label_id, "Unknown")
        color = unique_colors.get(label_id, (0, 255, 100))

        if label_text.lower() == TARGET_WORK_LABEL.lower():
            work_count += 1

        # 1. Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # 2. Top Label (Action Name)
        text = f"{label_text}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_x, text_y = max(x1, 0), max(y1 - 8, text_h + 4)
        cv2.rectangle(frame, (text_x, text_y - text_h - 4), (text_x + text_w + 4, text_y), color, -1)
        cv2.putText(frame, text, (text_x + 2, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)

        # 3. Bottom Label (Confidence)
        conf_text = f"{conf:.2f}"
        (conf_w, conf_h), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        conf_x, conf_y = min(x2 - conf_w - 3, width - conf_w - 3), min(y2 - 3, height - 3)
        cv2.rectangle(frame, (conf_x - 2, conf_y - conf_h - 2), (conf_x + conf_w + 2, conf_y + 2), color, -1)
        cv2.putText(frame, conf_text, (conf_x, conf_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1, cv2.LINE_AA)

    return work_count, total_count

def draw_productivity_dashboard(frame, productivity):
    """Draws the global productivity overlay in the top-left corner."""
    cv2.rectangle(frame, (0, 0), (320, 50), (0, 0, 0), -1)
    score_text = f"Utilization Rate: {productivity:.1f}%"
    cv2.putText(frame, score_text, (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

def my_demo(cfg):
    # 1. Setup & Initialization
    logging.setup_logging(cfg.OUTPUT_DIR)
    id_to_label, unique_colors = load_label_map(cfg.DEMO.LABEL_FILE_PATH)
    
    cap = cv2.VideoCapture(cfg.DEMO.INPUT_VIDEO)
    w, h = int(cap.get(3)), int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cfg.DEMO.OUTPUT_FPS if cfg.DEMO.OUTPUT_FPS > 0 else cap.get(cv2.CAP_PROP_FPS)

    predictor = FrameActionPredictor(cfg, img_width=w, img_height=h)
    out = cv2.VideoWriter(cfg.DEMO.OUTPUT_FILE, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # 2. Buffer Management
    frame_buffer = deque(maxlen=predictor.seq_length)
    ret, first_frame = cap.read()
    if not ret: return
    for _ in range(predictor.seq_length // 2 + 1): frame_buffer.append(first_frame)

    # 3. Main Processing Loop
    total_work_hits, total_all_hits = 0, 0
    readings = []
    pbar = tqdm(total=total_frames, desc="Processing", colour="green", ncols=100)

    keep_reading = True
    for i in range(total_frames):
        # Manage Buffer
        while len(frame_buffer) < predictor.seq_length and keep_reading:
            ret, frame = cap.read()
            if not ret: keep_reading = False
            else: frame_buffer.append(frame)
        if not keep_reading and len(frame_buffer) < predictor.seq_length:
            frame_buffer.append(frame_buffer[-1])

        # Inference
        current_clip = list(frame_buffer)
        vis_frame = current_clip[len(current_clip) // 2].copy()
        preds, boxes = predictor.predict_single_step(current_clip, vis_frame)

        # Update and Draw Visualization
        f_work, f_total = draw_predictions(vis_frame, boxes, preds, id_to_label, unique_colors)
        total_work_hits += f_work
        total_all_hits += f_total

        # Global Productivity Overlay
        current_prod = (total_work_hits / total_all_hits * 100) if total_all_hits > 0 else 0.0
        draw_productivity_dashboard(vis_frame, current_prod)

        if i and i % max(1, int(round(fps))) == 0:
            readings.append(current_prod)

        out.write(vis_frame)
        frame_buffer.popleft()
        pbar.update(1)

    # 4. Cleanup & Final Log
    cap.release()
    out.release()
    pbar.close()
    logger.info(f"Total Utilization: {current_prod:.2f}%")

    with open("readings.txt", "w") as f:
        for prod in readings:
            f.write(f"{prod}\n")
            
        f.write(f"Total Utilization: {current_prod}")    