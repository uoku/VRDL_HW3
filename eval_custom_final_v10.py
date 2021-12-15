import pycocotools
import cv2
import json
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import numpy as np
from detectron2.utils.logger import setup_logger

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("ceil_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.INPUT.MASK_FORMAT = "bitmask"
cfg.SOLVER.STEPS = []
# 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
# epoch = 100
# cfg.SOLVER.MAX_ITER = epoch * 24
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]

# for bachsize
# cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 64
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

# for anchor
# cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
# cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2.0, 3.0]]

# for classify
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 5000
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 60000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 55000

cfg.MODEL.RPN.NMS_THRESH = 0.9

# path to the model we just trained
cfg.MODEL.WEIGHTS = os.path.join("./output_nucleus_v10", "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # set a custom testing threshold

cfg.TEST.DETECTIONS_PER_IMAGE = 10000
predictor = DefaultPredictor(cfg)


f = open("./dataset/test_img_ids.json")
tests = json.load(f)
f.close()
answer = []
for img in tests:
    print(img['file_name'])
    image = cv2.imread(os.path.join('./dataset/test',
                       img['file_name'][:-4], "images", img['file_name']))
    outputs = predictor(image)
    out = outputs['instances']
    for idx in range(len(out)):
        dic = dict()
        dic['image_id'] = int(img['id'])
        diction = out[idx].get_fields()
        for pos in diction['pred_boxes']:
            position = (pos.to('cpu').numpy())
            new_pos = [float(position[0]), float(position[1]), float(position[2])
                       - float(position[0]), float(position[3]) - float(position[1])]
            dic['bbox'] = new_pos
        dic['score'] = diction['scores'].item()
        dic['category_id'] = 1
        mask = diction['pred_masks'].to('cpu').numpy()
        mask = pycocotools.mask.encode(np.asarray(mask[0], order="F"))
        mask['counts'] = mask['counts'].decode()
        dic['segmentation'] = mask
        answer.append(dic)

with open('answer.json', 'w') as f:
    json.dump(answer, f)
