# FIXED VALUES DO NOT CHANGE
MODELS_LIST = [

  "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", 
  "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
  "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
  "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml",
  "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

  ]

NUM_WORKERS = 2

IMS_PER_BATCH = 8  # This is the real "batch size" commonly known to deep learning people

BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this dataset (default: 512)

NUM_CLASSES = 1 


# DIRECTORIES OF THE DATA, PLEASE FOLLOW THE PROPER STRUCTURE IN THE
# READ ME FILE FOR THE DATA TYPES AND STRUCTURE 
Data_cfg = {

  # COCO TRAINING DATASET PATH
  "Coco_labels_train_dir": "/content/drive/MyDrive/Thesis_Organized/Data/COCO format/coco_train_data.json",

  # COCO TESTING DATASET PATH
  "Coco_labels_test_dir": "/content/drive/MyDrive/Thesis_Organized/Data/COCO format/coco_test_data.json",

  # COCO TRAINING VALIDATION PATH
  "Coco_labels_val_dir": "/content/drive/MyDrive/Thesis_Organized/Data/COCO format/coco_val_data.json",

  # CROPPED IMAGES PATH
  "cropped_Images_dir": "/content/drive/MyDrive/Thesis_Organized/Data/Cropped Images",

  # ORIGINAL IMAGES PATH
  "original_Images_dir": "/content/drive/MyDrive/Thesis_Organized/Data/Original Images",

  # GROUND TRUTH IMAGES PATH
  "GT_Images_dir": "/content/drive/MyDrive/Thesis_Organized/Data/Ground Truth",

  # PATH TO THE CSV FILES OF THE ANNOTATIONS CREATED IN THE DATA PROCESSING PART
  "annotations_path": "Data/Processed Data Frame/annotations_csv"

}

Augmentation_cfg = {

  # """https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.RandomRotation"""

 "RandomBrightness": [0.8, 1.2, 0],   # [min_brightness, max_brightness, probability] 
 "RandomRotation": [90, 90, 0.33],    # [min_angle, max_angle, probability]
 "RandomFlip": [0.33],    # [probability]
 "RandomCrop": [0.85, 0.85, 0],     # [croping ratio, cropping ratio, 0.2]
 "RandomContrast": [0.8, 1.2, 0]   # [min_contrast, max_contrast, probability] 

}


Detectron2_cfg = {

  "Momentum": 0.9,    # from source code: optimizer = torch.optim.SGD(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
  "base_lr": 0.00016,  # Learning rate
  "max_iter": 600,
  "resume": False,    # continue from last checkpoint, the checkpoint will be saved in the output folder, both the model and the last model name
  "threshold": 0.75   # confidence threshold for predictions (used for testing only)


}
