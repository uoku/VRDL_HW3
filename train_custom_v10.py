import pycocotools
import cv2
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import numpy as np
from detectron2.structures import BoxMode
import detectron2
from detectron2.config.config import CfgNode as CN
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
setup_logger()


def get_celi_data(dir_path):

    images = os.listdir(dir_path)

    dataset_dict = []
    idx = 0
    for image in images:
        print(idx)
        record = {}

        filename = os.path.join(dir_path, image, 'images', image + '.png')
        img = cv2.imread(filename)
        height, width = img.shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        masks = os.listdir(os.path.join(dir_path, image, 'masks'))
        for mask in masks:
            maskname = os.path.join(dir_path, image, 'masks', mask)
            mask_img = cv2.imread(maskname, 0)
            masknp = np.asarray(mask_img)

            pos = np.where(masknp > 0)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            seg = pycocotools.mask.encode(np.asarray(masknp, order="F"))

            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": seg,
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dict.append(record)
        idx += 1
    return dataset_dict


if __name__ == '__main__':
    for d in ["train", "val"]:
        DatasetCatalog.register("ceil_" + d, lambda d=d: get_celi_data("dataset/" + d))
        MetadataCatalog.get("ceil_" + d).set(thing_classes=["ceil"])
    ceil_metadata = MetadataCatalog.get("ceil_train")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("ceil_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.SOLVER.STEPS = []

    # for lr
    cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
    #cfg.SOLVER.STEPS = (250, 400)

    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    #epoch = 30
    #cfg.SOLVER.MAX_ITER = epoch * 24
    cfg.SOLVER.MAX_ITER = 30000
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.SOLVER.STEPS = (26000, 26666)

    # for bachsize
    #cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 128
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # for anchor
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
    #cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2.0, 3.0]]

    # RPN
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]

    # for classify
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # for NMS
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 5000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 12000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 5000

    cfg.MODEL.RPN.NMS_THRESH = 0.8

    #　random crop
    cfg.INPUT.CROP = CN({"ENABLED": True})
    cfg.INPUT.CROP.TYPE = "absolute"
    # Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
    # pixels if CROP.TYPE is "absolute"
    cfg.INPUT.CROP.SIZE = [600, 600]

    # for input
    cfg.INPUT.MIN_SIZE_TRAIN = (512, 600, 640)
    cfg.INPUT.MAX_SIZE_TRAIN = 720

    # for output
    cfg.OUTPUT_DIR = "./output_nucleus_v10"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
