#!/usr/bin/env python3
import cv2
import torch
from tqdm.auto import tqdm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from slowfast.utils import logging
from slowfast.datasets import cv2_transform
from slowfast.models import build_model
from slowfast.utils import checkpoint as cu
from slowfast.visualization.utils import process_cv2_inputs
from ultralytics import YOLO

logger = logging.get_logger(__name__)


class FrameActionPredictor:
    """
    Performs per-frame action prediction using SlowFast + Detectron2.
    Refactored for Memory Efficient Streaming.
    """

    def __init__(self, cfg, img_width, img_height, gpu_id=None):
        self.cfg = cfg
        if cfg.NUM_GPUS:
            self.gpu_id = torch.cuda.current_device() if gpu_id is None else gpu_id

        self.model = build_model(cfg, gpu_id=gpu_id)
        self.model.eval()

        if cfg.DETECTION.ENABLE:
            if cfg.DEMO.USE_YOLO:
                logger.info("Using Yolo for object detection.")
                self.object_detector = YoloPredictor(cfg)
            else:
                logger.info("Using Detectron2 for object detection.")
                self.object_detector = Detectron2Predictor(cfg, gpu_id=self.gpu_id)

        cu.load_test_checkpoint(cfg, self.model)
        logger.info("Finish loading model weights")

        logger.info(next(self.model.parameters()).device)

        self.seq_length = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE
        self.crop_size = cfg.DATA.TEST_CROP_SIZE
        self.img_height = img_height
        self.img_width = img_width

    def predict_single_step(self, clip, center_frame):
        """
        Performs detection + action prediction for a single streaming step.
        """
        boxes = self.object_detector.get_boxes(center_frame)

        if boxes is None or len(boxes) == 0:
            return [], []

        model_boxes = boxes.clone()
        model_boxes = cv2_transform.scale_boxes(
            self.crop_size,
            model_boxes,
            self.img_height,
            self.img_width,
        )

        bboxes = model_boxes.clone()
        index_pad = torch.full(
            size=(bboxes.shape[0], 1),
            fill_value=float(0),
            device=bboxes.device,
        )
        bboxes = torch.cat([index_pad, bboxes], dim=1)

        processed_clip = clip.copy()
        if self.cfg.DEMO.INPUT_FORMAT == "BGR":
            processed_clip = [
                cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in processed_clip
            ]

        processed_clip = [
            cv2_transform.scale(self.crop_size, f) for f in processed_clip
        ]
        inputs = process_cv2_inputs(processed_clip, self.cfg)

        if self.cfg.NUM_GPUS > 0:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(
                        device=torch.device(self.gpu_id), non_blocking=True
                    )
            else:
                inputs = inputs.cuda(
                    device=torch.device(self.gpu_id), non_blocking=True
                )

        with torch.no_grad():
            preds = self.model(inputs, bboxes)

        preds = preds.cpu()

        return preds, boxes.cpu()


class Detectron2Predictor:
    """Detectron2 human detector."""

    def __init__(self, cfg, gpu_id=None):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(cfg.DEMO.DETECTRON2_CFG))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.DEMO.DETECTRON2_THRESH
        self.cfg.MODEL.WEIGHTS = cfg.DEMO.DETECTRON2_WEIGHTS
        self.cfg.INPUT.FORMAT = cfg.DEMO.INPUT_FORMAT
        self.cfg.MODEL.DEVICE = f"cuda:{gpu_id}" if cfg.NUM_GPUS > 0 else "cpu"
        logger.info("Initialized Detectron2 Object Detection Model.")
        self.predictor = DefaultPredictor(self.cfg)

    def get_boxes(self, frame):
        outputs = self.predictor(frame)
        mask = outputs["instances"].pred_classes == 0  # person class only
        boxes = outputs["instances"].pred_boxes.tensor[mask]
        return boxes


class YoloPredictor:
    def __init__(self, cfg):
        choices = ["", "0", "0,1"]
        self.predictor = YOLO(cfg.DEMO.YOLO_WEIGHTS)
        self.device = choices[cfg.NUM_GPUS]
        self.conf_thresh = cfg.DEMO.YOLO_CONF_THRESH
        self.nms_thresh = cfg.DEMO.YOLO_NMS_THRESH

    def get_boxes(self, frame):
        results = self.predictor.predict(
            frame,
            device=self.device,
            conf=self.conf_thresh,
            iou=self.nms_thresh,
            verbose=False,
        )
        if len(results) == 0 or results[0].boxes is None:
            return None

        return results[0].boxes.xyxy
