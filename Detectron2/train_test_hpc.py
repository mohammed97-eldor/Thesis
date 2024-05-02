import os
import torch
from tqdm import tqdm
import cv2
import numpy as np
from utils import evaluate_instance, FP_FN_0, TP, FP, TP_0
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from customtrainer import CustomTrainer
import argparse
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import pandas as pd
try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

NUM_WORKERS = 2
# This is the real "batch size" commonly known to deep learning people
IMS_PER_BATCH = 8
# The "RoIHead batch size". 128 is faster, and good enough for this dataset (default: 512)
BATCH_SIZE_PER_IMAGE = 512
# batch size 4 and number of regions 512 required 14.8 GB as maximum
# batch size 8 and number of regions 512 required 22 GB GB as maximum
NUM_CLASSES = 1 
MAX_ITER = 100
CHECKPOINT_PERIOD = 1000
BACKBONE = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
IMGS_VAL_VIS_SAMPLES = imgs_names = ["L6Jn0_1-1-252.png", "L1Ap5_5-1-9.png", "L1Ap5_1-1-142.png",
                                     "L1Ap5_16-1-67.png", "L1Ap5_12-1-79.png", "L1Ap5_15-1-262.png",
                                     "L4Jn0_0-1-254.png", "L5Jn0_9-1-48.png"]

Data_cfg = {
  # COCO TRAINING DATASET PATH
  "Coco_labels_train_dir": "/content/drive/MyDrive/Thesis_Organized/Data/COCO_Format/coco_train_data.json",
  # COCO TESTING DATASET PATH
  "Coco_labels_test_dir": "/content/drive/MyDrive/Thesis_Organized/Data/COCO_Format/coco_test_data.json",
  # COCO TRAINING VALIDATION PATH
  "Coco_labels_val_dir": "/content/drive/MyDrive/Thesis_Organized/Data/COCO_Format/coco_val_data.json",
  # CROPPED IMAGES PATH
  "cropped_Images_dir": "/content/drive/MyDrive/Thesis_Organized/Data/Images_cropped",
  # ORIGINAL IMAGES PATH
  "original_Images_dir": "/content/drive/MyDrive/Thesis_Organized/Data/Original Images",
  # GROUND TRUTH IMAGES PATH
  "GT_Images_dir": "/content/drive/MyDrive/Thesis_Organized/Data/Ground Truth",
  # PATH TO THE CSV FILES OF THE ANNOTATIONS CREATED IN THE DATA PROCESSING PART
  "annotations_path": "/content/drive/MyDrive/Thesis_Organized/Data/Ground Truth/annotations.csv"
}

# The augmentation will be ignored in the first testing stages
# Augmentation_cfg = {
  # """https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomRotation"""
#  "RandomBrightness": [0.8, 1.2, 0.2],   # [min_brightness, max_brightness, probability] 
#  "RandomRotation": [-90, 90, 0.2],    # [min_angle, max_angle, probability]
#  "RandomFlip": [0.2],    # [probability]
#  "RandomCrop": [0.85, 0.85, 0.2],     # [croping ratio, cropping ratio, 0.2]
#  "RandomContrast": [0.8, 1.2, 0]   # [min_contrast, max_contrast, probability] 
# }

# training and validating data will be registered so we can sbmit them to the network
def register_data():
    register_coco_instances("my_dataset_train", {}, Data_cfg["Coco_labels_train_dir"], os.path.join(Data_cfg["cropped_Images_dir"], "Train"))
    # register_coco_instances("my_dataset_test", {}, Data_cfg["Coco_labels_test_dir"], os.path.join(Data_cfg["cropped_Images_dir"], "Test"))
    register_coco_instances("my_dataset_val", {}, Data_cfg["Coco_labels_val_dir"], os.path.join(Data_cfg["cropped_Images_dir"], "Val"))

def train(momentum, learning_rate, lr_decay, backbone = BACKBONE, ims_per_batch = IMS_PER_BATCH,
          batch_size_per_image = BATCH_SIZE_PER_IMAGE, num_classes = NUM_CLASSES, num_workers = NUM_WORKERS,
          checkpoint_period = CHECKPOINT_PERIOD, num_iterations = MAX_ITER):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = f"./train_results/X101_l1smooth_{momentum}_{learning_rate}_{lr_decay}"
    # now we have some configurations that we want to fix and not tune
    # this is to unfreeze layeras in the model training, by default the first 2 layers are freezed
    cfg.merge_from_file(model_zoo.get_config_file(backbone))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(backbone)
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
    cfg.SOLVER.MAX_ITER = num_iterations
    cfg.SOLVER.MOMENTUM = momentum 
    cfg.TEST.EVAL_PERIOD = 40
    # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = learning_rate
    if lr_decay:
        cfg.SOLVER.STEPS = [3000, 6000, 9000,]
    else:
        cfg.SOLVER.STEPS = [] 
    # The "RoIHead batch size". 128 is faster, and good enough (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  
    # image normalization, they might give errors
    cfg.MODEL.PIXEL_MEAN = [26.9, 26.9, 26.9]
    cfg.MODEL.PIXEL_STD = [34.4, 34.4, 34.4]
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    # for grayscale images
    # if it gives error just remove it and make mean and std 3 values that are equal [26.9, 26.9, 26.9], [34.4, 34.4, 34.4]
    # cfg.INPUT.FORMAT = "L"
    # cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = rpn_bbox_reg_loss
    # cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = roi_bbox_reg_loss
    # cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = retinanet_bbox_reg_loss
    cfg.INPUT.RANDOM_FLIP = "none"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.SOLVER.GAMMA = 0.25
    
    
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()
    del trainer
    torch.cuda.empty_cache()
    
    # run tests
    folder_path_currenttest = os.path.join('./train_results/test_images', f"X101_l1smooth_{momentum}_{learning_rate}_{lr_decay}")
    if not os.path.exists(folder_path_currenttest):
        os.makedirs(folder_path_currenttest)
        
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    
    for im_name in tqdm(IMGS_VAL_VIS_SAMPLES):
        im_dir = os.path.join(Data_cfg["cropped_Images_dir"], "Val", im_name)
        im = cv2.imread(im_dir)
        outputs = predictor(im)
        v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        save_image = out.get_image()[:, :, ::-1]
        cv2.imwrite(os.path.join(folder_path_currenttest, im_name), save_image)
        
    df = pd.read_csv(Data_cfg["annotations_path"])
    test = df[df.folder == "Val"].copy()
    test = test.reset_index().drop(columns = ["index"])
    output_dict = {}
    
    for image in tqdm(np.unique(test.name.values)):
        temp = test[test.name == image].copy()
        im_dir = os.path.join(Data_cfg["cropped_Images_dir"], temp.folder.values[0], image)
        im = cv2.imread(im_dir)
        outputs = predictor(im)
        outputs = outputs["instances"].pred_masks.to("cpu").numpy()
        gt_lst = []
        for instance in temp.track:
          gt_dir = os.path.join(Data_cfg["GT_Images_dir"], temp.folder.values[0], temp.name.values[0][:-4], instance + ".png")
          gt = cv2.imread(gt_dir)
          gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
          gt = cv2.resize(gt, (658, 517), interpolation = cv2.INTER_AREA)
          gt = gt.astype("bool")
          gt_lst.append(gt)

        output_dict[image] = evaluate_instance(gt_lst, outputs)
        
    with open(os.path.join(cfg.OUTPUT_DIR, "test_results.p"), 'wb') as fp:
        pickle.dump(output_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
    results_dict = {
        "FP_FN_0": FP_FN_0(output_dict),
        "TP": TP(output_dict),
        "FP": FP(output_dict),
        "TP_0": TP_0(output_dict)
    }

    results = pd.DataFrame(results_dict, index = ["results"])
    results.to_csv(os.path.join(cfg.OUTPUT_DIR, "results.csv"))
    
    
def get_args():
    parser = argparse.ArgumentParser(description="Training settings")

    # Required arguments
    parser.add_argument("--momentum", required=True, type=float,
                        help="Momentum for optimizer")
    parser.add_argument("--learning_rate", required=True, type=float,
                        help="Initial learning rate")
    parser.add_argument("--lr_decay", required=True, type=bool,
                        help="Enable learning rate decay")

    # Optional arguments with default values
    parser.add_argument("--backbone", type=str, default=BACKBONE,
                        help="Backbone model for the network")
    parser.add_argument("--ims_per_batch", type=int, default=IMS_PER_BATCH,
                        help="Images per batch")
    parser.add_argument("--batch_size_per_image", type=int, default=BATCH_SIZE_PER_IMAGE,
                        help="Batch size per image")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES,
                        help="Number of classes")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS,
                        help="Number of worker threads")
    parser.add_argument("--checkpoint_period", type=int, default=CHECKPOINT_PERIOD,
                        help="Period to save checkpoints")
    parser.add_argument("--num_iterations", type=int, default=MAX_ITER,
                        help="Maximum number of iterations")

    return parser.parse_args()

if __name__ == "__main__":
    register_data()
    args = get_args()
    train_folder_path = './train_results'
    if not os.path.exists(train_folder_path):
        os.makedirs(train_folder_path)
    test_folder_path = './train_results/test_images'
    if not os.path.exists(test_folder_path):
        os.makedirs(test_folder_path)
    train(momentum=args.momentum,
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        backbone=args.backbone,
        ims_per_batch=args.ims_per_batch,
        batch_size_per_image=args.batch_size_per_image,
        num_classes=args.num_classes,
        num_workers=args.num_workers,
        checkpoint_period=args.checkpoint_period,
        num_iterations=args.num_iterations)
