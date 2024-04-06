# Dataset Exploration and Conversion for Instance Segmentation

This folder hosts the code and resources developed during a Master's Thesis project aimed at automating the recognition of trace particles in nuclear scattering experiments. The focus is on preparing a dataset specifically for computer vision applications, emphasizing the transformation of data into the COCO (Common Objects in Context) format, which is crucial for training computer vision models for instance segmentation tasks. It also has a notebook for transforming the data into yolov5 format that was not used later on but we kept it as a backup in case we needed to use one of the yolo networks.

## Project Overview

The thesis introduces a system designed to trace the trajectories of nuclear particles captured in images, incorporating semantic segmentation via convolutional neural networks (CNNs), a novel trace division algorithm, intensity studies of traces, and a 3D reconstruction of particle trajectories. The dataset was carefully processed, with each step manually verified to ensure the highest quality. Some tracks and images were dropped due to formatting errors or inconsistencies.

## Repository Structure

- **Original Images**: Contains an `annotations.xml` file for ground truth annotations. Users must upload their images due to confidentiality. It also contains 3 folders of the original images distributed into:

	1. 800 images for training in the Train folder
	2. 100 images for validation in the Val folder
	3. 100 images for testing in the Test folder

- **Ground Truth**: Houses `annotations.csv` with bounding box and segmentation data in COCO format. It also follows the structure of the original images where we have Train, Val, and Test folders. But here each image has a folder with segmentations saved as matrices (boolean images) where each track is a atrix and has a 1 for the area covered by the track (the mask) and 0 everywhere else. Each image might have many tracks, thus each image folder might contain more than one instance of the mask matrix.

- **Images_cropped**: Hosts cropped versions of the original images to minimize noise, generated through the processing notebooks.

- **COCO_Format**: Contains JSON files of the dataset in COCO format and two notebooks, `Visualization_tools.ipynb`, and `json_structure_exploration.ipynb` for applying visualizations to COCO JSON files and some ecploration of the structure. For further details about the COCO format visit the official [site](https://cocodataset.org/#home).

### Notebooks Overview

1. **Dataset Exploration.ipynb**: Begins the dataset preparation process by cleaning, verifying the dataset, transforming data into COCO format, generating visualizations, and checking for errors.
2. **Processing the Images.ipynb**: Focuses on cropping images to produce cleaner versions for further processing.
3. **Transform Data to COCO.ipynb**: Converts CSV-formatted data into properly formatted JSON files in COCO standard, ensuring the data is ready for computer vision models.
4. **calculate_mean_std.ipynb**: This notebook calculates the channel-wise mean and standard deviation of the training dataset, this will be utilized as for the configuration (normalization) of the networks.
5. **transform_data_to_yolov5.ipynb**: as mentioned before we transformed our data into the yolov5 format as well, but we deleted it from the repository, in case you wish to redo that, please refer to the notebook.

### Final Dataset Composition and statistics

After careful processing and manual verification, the dataset composition is as follows:

| Metric                         | Train                 | Val                   | Test                  |
|--------------------------------|-----------------------|-----------------------|-----------------------|
| Number of Images               | 791                   | 97                    | 98                    |
| Number of Annotations          | 2402                  | 309                   | 286                   |
| Number of Categories           | 1                     | 1                     | 1                     |
| Number of Negative Images      | 0                     | 0                     | 0                     |
| Min Number of Annotations/Img  | 1                     | 1                     | 1                     |
| Max Number of Annotations/Img  | 10                    | 9                     | 7                     |
| Avg Number of Annotations/Img  | 3.04                  | 3.19                  | 2.92                  |
| Min Annotation Area            | 479                   | 657                   | 480                   |
| Max Annotation Area            | 12425                 | 11736                 | 11504                 |
| Avg Annotation Area            | 4854.82               | 4815.57               | 4858.85               |



## Usage Instructions

1. Clone or download this repository. Note that the original images are confidential, thus this can be only utilized by the owners.
2. Upload your images to the **Original Images** folder.
3. Execute the Jupyter Notebooks in the specified order, following the instructions within each notebook.
4. The dataset will be transformed and ready for use in computer vision tasks in COCO format.

## Requirements

- Python 3.6+
- Jupyter Notebook or JupyterLab

## Contributors

This project was developed by [MOHAMMED EL DOR], a Master's student in Data Scenince and  Engineering at TURIN POLYTECHNIC, under the guidance of Prof. LIA MORRA and Prof. FABRIZIO LAMBERTI.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
