import os
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg, CfgNode
from customtrainer import CustomTrainer
from tqdm import tqdm
import random
import itertools

NUM_WORKERS = 2
# This is the real "batch size" commonly known to deep learning people
IMS_PER_BATCH = 8
# The "RoIHead batch size". 128 is faster, and good enough for this dataset (default: 512)
BATCH_SIZE_PER_IMAGE = 128
NUM_CLASSES = 1 
NUMBER_OF_TRIALS = 100
MAX_ITER = 10
CHECKPOINT_PERIOD = 500

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
  "annotations_path": "Data/Ground Truth/annotations.csv"
}

Augmentation_cfg = {
  # """https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomRotation"""
 "RandomBrightness": [0.8, 1.2, 0],   # [min_brightness, max_brightness, probability] 
 "RandomRotation": [-90, 90, 0.33],    # [min_angle, max_angle, probability]
 "RandomFlip": [0.33],    # [probability]
 "RandomCrop": [0.85, 0.85, 0.2],     # [croping ratio, cropping ratio, 0.2]
 "RandomContrast": [0.8, 1.2, 0.2]   # [min_contrast, max_contrast, probability] 
}

def register_data():
    register_coco_instances("my_dataset_train", {}, Data_cfg["Coco_labels_train_dir"], os.path.join(Data_cfg["cropped_Images_dir"], "Train"))
    register_coco_instances("my_dataset_test", {}, Data_cfg["Coco_labels_test_dir"], os.path.join(Data_cfg["cropped_Images_dir"], "Test"))
    register_coco_instances("my_dataset_val", {}, Data_cfg["Coco_labels_val_dir"], os.path.join(Data_cfg["cropped_Images_dir"], "Val"))
    
def train(backbone, rpn_bbox_reg_loss, roi_bbox_reg_loss, retinanet_bbox_reg_loss, momentum,
          learning_rate, ims_per_batch = IMS_PER_BATCH, batch_size_per_image = BATCH_SIZE_PER_IMAGE,
          num_classes = NUM_CLASSES, num_workers = NUM_WORKERS):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = f"./train_results/{backbone}_{rpn_bbox_reg_loss}_{roi_bbox_reg_loss}_{retinanet_bbox_reg_loss}_{momentum}_{learning_rate}"
    # now we have some configurations that we want to fix and not tune
    # this is to unfreeze layeras in the model training, by default the first 2 layers are freezed
    cfg.merge_from_file(model_zoo.get_config_file(backbone))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_test",)
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD
    cfg.SOLVER.MAX_ITER = MAX_ITER
    cfg.SOLVER.MOMENTUM = momentum #Detectron2_cfg["Momentum"]
    # cfg.TEST.EVAL_PERIOD = 20
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = learning_rate# Detectron2_cfg["base_lr"]  # pick a good LR
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image  # The "RoIHead batch size". 128 is faster, and good enough for this dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  
    # this parameter is from detectron_conf
    cfg.DATALOADER.NUM_WORKERS = num_workers
    # image normalization, they might give errors
    cfg.MODEL.PIXEL_MEAN = [26.9]
    cfg.MODEL.PIXEL_STD = [34.4]
    # for grayscale images
    # if it gives error just remove it and make mean and std 3 values that are equal [26.9, 26.9, 26.9], [34.4, 34.4, 34.4]
    cfg.INPUT.FORMAT = "L"
    cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = rpn_bbox_reg_loss
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = roi_bbox_reg_loss
    cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = retinanet_bbox_reg_loss
    
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

hyperparameters = {
    'backbone': ["COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", 
                 "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                 "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
                 ],
    'rpn_bbox_reg_loss': ["smooth_l1", "giou", "diou", "ciou"],
    'roi_bbox_reg_loss': ["smooth_l1", "giou", "diou", "ciou"],
    'retinanet_bbox_reg_loss': ["smooth_l1", "giou", "diou", "ciou"],
    'momentum': [0.8, 0.85, 0.9, 0.95, 0.98, 0.99],
    'learning_rate': [0.00007, 0.0001, 0.00016, 0.0002, 0.0003]
}

# Function to select a random hyperparameter set
def sample_hyperparameters(hyperparameters):
    return {k: random.choice(v) for k, v in hyperparameters.items()}

def random_search():
    lst = []
    for iteration in tqdm(range(NUMBER_OF_TRIALS)):
        params = sample_hyperparameters(hyperparameters)
        while params in lst:
            params = sample_hyperparameters(hyperparameters)
        lst.append(params)
        print(params.values())
        backbone, rpn_loss, roi_loss, retinanet_loss, momentum, lr = params.values()
        # Call your training function with the sampled hyperparameters
        train(backbone, rpn_loss, roi_loss, retinanet_loss, momentum, lr)
        
def generate_combinations(hyperparameters):
    # Generate all combinations of hyperparameters
    keys, values = zip(*hyperparameters.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations

def grid_search():
    combinations = generate_combinations(hyperparameters)
    # Iterate through each combination using tqdm for a progress bar
    for params in tqdm(combinations):
        # Extract parameters from the current combination
        backbone = params['backbone']
        rpn_loss = params['rpn_bbox_reg_loss']
        roi_loss = params['roi_bbox_reg_loss']
        retinanet_loss = params['retinanet_bbox_reg_loss']
        momentum = params['momentum']
        lr = params['learning_rate']
        
        # Call your training function with the current set of hyperparameters
        train(backbone, rpn_loss, roi_loss, retinanet_loss, momentum, lr)

if __name__ == "__main__":
    folder_path = './train_results'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    register_data()
    random_search()
