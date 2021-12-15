from train_custom import get_celi_data
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
setup_logger()

dataset_dicts = get_celi_data("dataset/val")
ceil_metadata = MetadataCatalog.get("ceil_train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=ceil_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow('img', out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
