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
from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform
from yolox.utils import fuse_model, postprocess

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
            if cfg.DEMO.USE_YOLOX:
                logger.info("Using YoloX for object detection.")
                self.object_detector = YoloXPredictor(cfg, gpu_id=self.gpu_id)
            else:
                logger.info("Using Detectron2 for object detection.")
                self.object_detector = Detectron2Predictor(cfg, gpu_id=self.gpu_id)

        logger.info("Start loading model weights.")
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

        # 3. Process Clip (Color conversion + Scaling)
        # Note: clip is passed as a list of numpy arrays
        processed_clip = clip.copy() # copy to avoid modifying original buffer
        if self.cfg.DEMO.INPUT_FORMAT == "BGR":
            processed_clip = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in processed_clip]

        processed_clip = [cv2_transform.scale(self.crop_size, f) for f in processed_clip]
        inputs = process_cv2_inputs(processed_clip, self.cfg)

        # 4. Prepare Bounding Box Inputs
        bboxes = model_boxes.clone()
        index_pad = torch.full(
            size=(bboxes.shape[0], 1),
            fill_value=float(0),
            device=bboxes.device,
        )
        bboxes = torch.cat([index_pad, bboxes], dim=1)

        # 5. Move to GPU
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

        # 6. Inference
        with torch.no_grad():
            preds = self.model(inputs, bboxes)

        preds = preds.cpu()
        
        # Return the predictions and the original boxes (on CPU) for drawing
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


class YoloXPredictor:
    """YoloX human detector."""
    def __init__(self, cfg, gpu_id):
        self.exp = get_exp(cfg.DEMO.YOLOX_EXP, cfg.DEMO.YOLOX_EXP_NAME)
        self.weights = cfg.DEMO.YOLOX_WEIGHTS
        self.conf_thresh = cfg.DEMO.YOLOX_CONF_THRESH
        self.nms_thresh = cfg.DEMO.YOLOX_NMS_THRESH
        self.device = f"cuda:{gpu_id}" if cfg.NUM_GPUS > 0 else "cpu"
        self.model = self.exp.get_model()
        self.model.eval()
        self.model.to(self.device)
        ckpt = torch.load(
            cfg.DEMO.YOLOX_WEIGHTS, map_location=self.device, weights_only=True
        )
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt)
        self.model = fuse_model(self.model)
        self.preproc = ValTransform(
            rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

    def get_boxes(self, frame):
        height, width = frame.shape[:2]
        img, _ = self.preproc(frame, None, self.exp.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs,
                num_classes=1,
                conf_thre=self.conf_thresh,
                nms_thre=self.nms_thresh,
            )

        if outputs[0] is not None:
            boxes = outputs[0][:, :4]
            # Scale the boxes back to the original image scale
            scale = min(
                float(height) / img.shape[2], float(width) / img.shape[1]
            )
            boxes *= scale
            return boxes

        return None