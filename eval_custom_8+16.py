from train_custom import get_celi_data
from detectron2.utils.visualizer import ColorMode
import pycocotools
import random
import cv2
import json
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import numpy as np
from detectron2.structures import BoxMode
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
import torch

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.INPUT.MASK_FORMAT = "bitmask"
cfg.SOLVER.STEPS = []
# 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
epoch = 100
cfg.SOLVER.MAX_ITER = epoch * 24

# for bachsize
#cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

# for anchor
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
#cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2.0, 3.0]]

# for classify
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

# for input
cfg.INPUT.MIN_SIZE_TRAIN = (500,)
cfg.INPUT.MAX_SIZE_TRAIN = 1333

# path to the model we just trained
cfg.MODEL.WEIGHTS = os.path.join("./output_anchor_8+16", "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
cfg.TEST.DETECTIONS_PER_IMAGE = 2000
predictor = DefaultPredictor(cfg)

dataset_dicts = get_celi_data("dataset/val")
ceil_metadata = MetadataCatalog.get("ceil_train")

for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs = predictor(im)
    print(im[:, :, ::-1])
    v = Visualizer(im[:, :, ::-1],
                   metadata=ceil_metadata,
                   scale=0.5,
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   instance_mode=ColorMode.IMAGE_BW
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('test', out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break
