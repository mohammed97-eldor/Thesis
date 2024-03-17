import os
from tqdm import tqdm
import torch
import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()
# import some common libraries
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
# This is a local config files for training
from Detectron2.detectron_conf import *
from detectron2.data.datasets import register_coco_instances
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import time
import datetime
import logging
from detectron2.engine import DefaultTrainer
from customtrainer import CustomTrainer

# regitser the datasets for training testing and validation
def register_datasets():
    register_coco_instances("my_dataset_train", {}, Data_cfg["Coco_labels_train_dir"], os.path.join(Data_cfg["cropped_Images_dir"], "Train"))
    register_coco_instances("my_dataset_test", {}, Data_cfg["Coco_labels_test_dir"], os.path.join(Data_cfg["cropped_Images_dir"], "Test"))
    register_coco_instances("my_dataset_val", {}, Data_cfg["Coco_labels_val_dir"], os.path.join(Data_cfg["cropped_Images_dir"], "Val"))

def change_conf_detectron(output_path, model_index = 0, momentum_index = 0):

    cfg = get_cfg()
    # layers to freeze in the feature extractor of the mask-RCNN
    cfg["MODEL"]['BACKBONE'] = CfgNode({'NAME': 'build_resnet_backbone', 'FREEZE_AT': 0})
    # path to store the output
    cfg.OUTPUT_DIR = output_path
    # refer to the personal config file and choose the backbone
    cfg.merge_from_file(model_zoo.get_config_file(MODELS_LIST[model_index]))
    #soecify the training dataset
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    # specify the testing dataset
    cfg.DATASETS.TEST = ("my_dataset_test",)
    cfg.SOLVER.MOMENTUM = Detectron2_cfg["Momentum"][momentum_index]
    cfg.SOLVER.CHECKPOINT_PERIOD =   Detectron2_cfg["Momentum"]     # The network takes a checkpoint once it finishes of every 200 iterations
    cfg.TEST.EVAL_PERIOD = 20
cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODELS_LIST[1])  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.001 # Detectron2_cfg["base_lr"]  # pick a good LR
cfg.SOLVER.MAX_ITER = 1500
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE_PER_IMAGE  # The "RoIHead batch size". 128 is faster, and good enough for this dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES  # only has one class (Track). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    