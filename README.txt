================================================================================
CAR DETECTION WITH YOLOv11 AND YOLOv12
================================================================================

Deep Learning - Object Detection Project
Dataset: Car Detection (BMW-X5-M, Volvo-XC40, Jaguar)
Models: YOLOv11 and YOLOv12

Date: November 2025

================================================================================
PROJECT OVERVIEW
================================================================================

This project implements object detection for car models using state-of-the-art
YOLOv11 and YOLOv12 models. The implementation includes:

1. Comprehensive Exploratory Data Analysis (EDA)
2. Model training on local Linux environment with GPU acceleration
3. Performance comparison between YOLOv11 and YOLOv12
4. Detailed evaluation metrics and visualizations
5. Custom annotation tool for creating your own datasets

The dataset contains annotated images of three car models:
- BMW-X5-M
- Volvo-XC40
- Jaguar

The project demonstrates end-to-end object detection pipeline from data
preparation to model evaluation.

================================================================================
FILE STRUCTURE AND DESCRIPTIONS
================================================================================

1. train_script.ipynb
   -----------------------------------------
   Description: Main Jupyter notebook for local training
   Purpose: Complete pipeline for training and evaluating YOLOv11 and YOLOv12
   Features:
     - Complete end-to-end pipeline
     - Exploratory Data Analysis with visualizations
     - Training both YOLOv11 and YOLOv12 models
     - Performance comparison with metrics
     - Inference examples and confusion matrices
     - Results export and model saving

   How to use:
     1. Ensure dataset is in ./dataset/ directory
     2. Open in Jupyter notebook or JupyterLab
     3. Run all cells sequentially
     4. Results will be generated in the results/ directory

2. train_script_colab.ipynb
   -----------------------------------------
   Description: Google Colab version of training notebook
   Purpose: Cloud-based training for users without local GPU
   Features:
     - Same features as train_script.ipynb
     - Optimized for Google Colab environment
     - Free GPU access (T4, V100, or A100)

   How to use:
     1. Upload to Google Colab
     2. Ensure GPU runtime is selected:
        Runtime → Change runtime type → GPU
     3. Upload dataset or mount Google Drive
     4. Run all cells sequentially

3. annotate_yolo.py
   -----------------------------------------
   Description: Interactive YOLO annotation tool
   Purpose: Create custom datasets without cloud services (No Roboflow required)
   Features:
     - Local, private annotation tool
     - Draw bounding boxes with mouse
     - Support for multiple classes
     - YOLO format export
     - Load and edit existing annotations
     - Dataset organization utilities

   How to use:
     python annotate_yolo.py --images /path/to/images --labels /path/to/labels --class-list car person dog

   Arguments:
     --images       : Directory containing images to annotate (required)
     --labels       : Directory to save YOLO format labels (required)
     --classes      : Path to classes.txt file (one class per line)
     --class-list   : List of classes (e.g., --class-list BMW-X5-M Volvo-XC40 Jaguar)

   Controls:
     - Draw box: Click and drag with mouse
     - Next image: SPACE or N
     - Previous image: BACKSPACE or P
     - Save: S
     - Delete last box: D
     - Change class: C
     - Quit: Q or ESC

   Example:
     python annotate_yolo.py --images ./raw_images --labels ./annotations \
                             --class-list BMW-X5-M Volvo-XC40 Jaguar

4. requirements.txt
   -----------------------------------------
   Description: Python dependencies
   Purpose: List all required packages with version specifications

   How to use:
     # Create virtual environment
     python -m venv .venv
     source .venv/bin/activate  # On Linux/Mac

     # Install dependencies
     pip install -r requirements.txt

5. yolo11n.pt & yolo12n.pt
   -----------------------------------------
   Description: Pre-trained model weights (nano versions)
   Purpose: Starting weights for transfer learning

   - yolo11n.pt: YOLOv11 nano model (faster, smaller)
   - yolo12n.pt: YOLOv12 nano model (latest version)

6. dataset/
   -----------------------------------------
   Description: Car detection dataset directory
   Structure:
     dataset/
     ├── data.yaml              # Dataset configuration
     ├── images/
     │   ├── train/            # Training images (1197 images)
     │   └── val/              # Validation images (300 images)
     └── labels/
         ├── train/            # Training annotations
         └── val/              # Validation annotations

7. results/
   -----------------------------------------
   Description: Training results and outputs
   Contents:
     - Visualizations (PNG files)
     - Performance metrics (CSV, TXT)
     - Trained models (best.pt files)
     - Confusion matrices
     - Inference examples

8. README.txt (this file)
   -----------------------------------------
   Description: Comprehensive documentation
   Purpose: Guide users through the project setup and execution

================================================================================
SETUP INSTRUCTIONS
================================================================================

LOCAL LINUX ENVIRONMENT SETUP
------------------------------

1. System Requirements:
   - Linux Ubuntu 20.04+ (or compatible distribution)
   - Python 3.8 or higher
   - CUDA-capable GPU (recommended) or CPU
   - Minimum 8GB RAM (16GB+ recommended)
   - 20GB free disk space

2. Install Python and Dependencies:

   # Update system
   sudo apt update && sudo apt upgrade -y

   # Install Python 3 and pip
   sudo apt install python3 python3-pip python3-venv -y

   # Create virtual environment
   python3 -m venv .venv
   source .venv/bin/activate

   # Upgrade pip
   pip install --upgrade pip setuptools wheel

   # Install requirements
   pip install -r requirements.txt

3. Verify Installation:

   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
   python -c "from ultralytics import YOLO; print('Ultralytics YOLO installed successfully')"

4. Verify Dataset Structure:

   ls -la dataset/
   # Should see: data.yaml, images/, labels/

GOOGLE COLAB SETUP
------------------

1. Upload train_script_colab.ipynb to Google Colab
2. Ensure GPU runtime is selected:
   - Runtime → Change runtime type → GPU (T4, V100, or A100)
3. Upload dataset to Colab or mount Google Drive
4. Run all cells

================================================================================
EXECUTION WORKFLOW
================================================================================

COMPLETE PIPELINE (LOCAL LINUX)
--------------------------------

Option 1: Using Jupyter Notebook
   jupyter notebook train_script.ipynb
   # Run all cells sequentially

Option 2: Using JupyterLab
   jupyter lab
   # Open train_script.ipynb and run all cells

The notebook will:
1. Perform Exploratory Data Analysis
2. Train YOLOv11 model (~11 minutes on Tesla T4)
3. Train YOLOv12 model (~37 minutes on Tesla T4)
4. Compare model performance
5. Generate visualizations and reports
6. Save trained models

CREATING YOUR OWN DATASET
--------------------------

1. Collect images of objects you want to detect
2. Organize images in a directory
3. Run the annotation tool:

   python annotate_yolo.py --images ./my_images --labels ./my_labels \
                           --class-list class1 class2 class3

4. Annotate all images using the interactive tool
5. Organize into YOLO format:

   dataset/
   ├── data.yaml
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/

6. Create data.yaml:

   train: images/train
   val: images/val
   nc: 3
   names:
     0: class1
     1: class2
     2: class3

7. Update dataset path in notebook and train!

================================================================================
OUTPUT FILES AND RESULTS
================================================================================

After running the complete pipeline, you will have the following outputs:

Visualizations (results/):
-------------------------------------------
- eda_class_distribution.png         : Class distribution and objects per image
- sample_annotated_images.png        : Sample training images with annotations
- performance_comparison_local.png   : Comprehensive metrics comparison
- confusion_matrices_comparison.png  : Confusion matrices for both models
- inference_comparison_*.png         : Sample predictions from both models

Data Files (results/):
--------------------------------------------
- model_comparison_local.csv         : Comparison metrics in CSV format
- local_execution_summary.txt        : Complete execution summary
- local_environment_info.txt         : System and environment information

Models (results/models/):
--------------------------------------------
- yolov11_best.pt                    : Best YOLOv11 model weights
- yolov12_best.pt                    : Best YOLOv12 model weights

Training Outputs (current directory):
--------------------------------------------
- yolov11_car_detection/             : YOLOv11 training outputs
  └── weights/best.pt                : Best weights
  └── confusion_matrix.png           : Confusion matrix
  └── results.png                    : Training curves
- yolov12_car_detection/             : YOLOv12 training outputs
  └── weights/best.pt                : Best weights
  └── confusion_matrix.png           : Confusion matrix
  └── results.png                    : Training curves

================================================================================
PERFORMANCE METRICS EXPLAINED
================================================================================

mAP@0.5 (Mean Average Precision at IoU=0.5)
   - Primary metric for object detection
   - Measures accuracy of detections with IoU threshold of 0.5
   - Higher is better (range: 0-1)
   - Our results: YOLOv11 (0.9923), YOLOv12 (0.9913)

mAP@0.5:0.95 (Mean Average Precision at IoU=0.5 to 0.95)
   - Average mAP across IoU thresholds from 0.5 to 0.95
   - More strict metric than mAP@0.5
   - Standard metric for COCO dataset evaluation
   - Our results: YOLOv11 (0.9866), YOLOv12 (0.9847)

Precision
   - Proportion of correct positive predictions
   - Precision = TP / (TP + FP)
   - Higher is better (range: 0-1)
   - Our results: YOLOv11 (0.9892), YOLOv12 (0.9844)

Recall
   - Proportion of actual positives correctly identified
   - Recall = TP / (TP + FN)
   - Higher is better (range: 0-1)
   - Our results: YOLOv11 (0.9818), YOLOv12 (0.9898)

F1 Score
   - Harmonic mean of Precision and Recall
   - F1 = 2 × (Precision × Recall) / (Precision + Recall)
   - Balanced metric combining precision and recall
   - Our results: YOLOv11 (0.9855), YOLOv12 (0.9871)

Inference Time
   - Time taken to process a single image
   - Measured in milliseconds (ms)
   - Lower is better for real-time applications

Training Time
   - Total time to train the model
   - Our results: YOLOv11 (11.18 min), YOLOv12 (36.67 min)

================================================================================
HYPERPARAMETERS USED
================================================================================

Training Configuration:
-----------------------
- Image Size: 640x640 pixels
- Batch Size: 16 (adjustable based on GPU memory)
- Epochs: 50 (with early stopping patience=10)
- Device: CUDA (GPU) or CPU
- Workers: 2 (data loading workers)

Optimizer Settings:
-------------------
- Optimizer: AdamW
- Initial Learning Rate (lr0): 0.01
- Final Learning Rate (lrf): 0.01
- Momentum: 0.937
- Weight Decay: 0.0005

Warmup Settings:
----------------
- Warmup Epochs: 3
- Warmup Momentum: 0.8

Loss Weights:
-------------
- Box Loss: 7.5
- Class Loss: 0.5
- DFL Loss: 1.5

Model Variants:
---------------
- YOLOv11n (nano): Faster training, smaller model size
- YOLOv12n (nano): Latest version with architectural improvements

================================================================================
PROJECT RESULTS SUMMARY
================================================================================

Dataset Statistics:
-------------------
- Training Images: 1,197
- Validation Images: 300
- Total Classes: 3 (BMW-X5-M, Volvo-XC40, Jaguar)
- Annotations: YOLO format (normalized bounding boxes)

YOLOv11 Performance:
--------------------
- mAP@0.5: 0.9923
- mAP@0.5:0.95: 0.9866
- Precision: 0.9892
- Recall: 0.9818
- F1 Score: 0.9855
- Training Time: 11.18 minutes

YOLOv12 Performance:
--------------------
- mAP@0.5: 0.9913
- mAP@0.5:0.95: 0.9847
- Precision: 0.9844
- Recall: 0.9898
- F1 Score: 0.9871
- Training Time: 36.67 minutes

Key Findings:
-------------
1. Both models achieve excellent performance (>99% mAP@0.5)
2. YOLOv11 is 3.3x faster to train than YOLOv12
3. YOLOv11 has slightly higher precision (0.9892 vs 0.9844)
4. YOLOv12 has slightly higher recall (0.9898 vs 0.9818)
5. Performance difference is minimal (<0.2% in mAP)
6. Trade-off: YOLOv11 for speed, YOLOv12 for marginal accuracy gains

Computing Environment:
----------------------
- Platform: Local Linux Machine (AWS)
- GPU: Tesla T4 (16GB VRAM)
- CUDA Version: 12.8
- PyTorch Version: 2.9.1+cu128
- Total Execution Time: ~52 minutes

================================================================================
CLOUD VS LOCAL COMPARISON
================================================================================

This project supports both local and cloud execution:

Local Linux (Used in this project):
------------------------------------
Advantages:
   No time limits
   Consistent performance
   Full control over environment
   Works offline (after dataset is ready)
   Better for iterative experiments
   Private data (no upload required)

Disadvantages:
   Requires hardware investment
   Manual environment setup
   Maintenance overhead

Google Colab (Alternative):
---------------------------
Advantages:
   Free GPU access (T4, sometimes V100/A100)
   No local setup required
   Pre-installed libraries
   Easy sharing and collaboration

Disadvantages:
   Session time limits (12 hours for free tier)
   Variable GPU availability
   Internet connection required
   Limited storage
   Data upload required

Expected Observations:
----------------------
- Training time depends on GPU availability and type
- Colab GPUs (T4) have similar performance to local T4
- Local machines offer more consistent performance
- Network speed affects data transfer in Colab

================================================================================
TROUBLESHOOTING
================================================================================

Common Issues and Solutions:

1. CUDA Out of Memory Error:
   Error: RuntimeError: CUDA out of memory
   Solution: Reduce batch size in notebook (change BATCH_SIZE = 8 or 4)

2. Dataset Not Found:
   Error: FileNotFoundError: Required dataset structure not found
   Solution: Verify dataset/ directory structure and data.yaml exists

3. ImportError for ultralytics:
   Error: ModuleNotFoundError: No module named 'ultralytics'
   Solution: Ensure virtual environment is activated and requirements installed

4. Model Training Stops:
   Solution: Check disk space, GPU memory, and logs in training directories

5. Slow Training:
   Solution: Verify GPU is being used with nvidia-smi command
   If no GPU detected, training will be very slow on CPU

6. Permission Denied:
   Error: Permission denied when running scripts
   Solution: Make scripts executable: chmod +x annotate_yolo.py

7. Python Version Issues:
   Error: Syntax errors or incompatible features
   Solution: Ensure Python 3.8+ is being used (python --version)

8. OpenCV Display Issues (for annotation tool):
   Error: Can't open display
   Solution: Ensure X11 forwarding is enabled or run on local machine with display

9. Workers Error in DataLoader:
   Error: RuntimeError: DataLoader worker exited unexpectedly
   Solution: Reduce workers parameter or use workers=0

================================================================================
DATASET INFORMATION
================================================================================

Dataset: Car Detection Dataset
Classes: 3 car models
Format: YOLO (text files with normalized bounding box coordinates)

Classes:
  0: BMW-X5-M     - Luxury sports SUV
  1: Volvo-XC40   - Compact luxury SUV
  2: Jaguar       - Luxury vehicle

Dataset Split:
  Training: 1,197 images
  Validation: 300 images
  Total: 1,497 images

Structure:
  dataset/
  ├── data.yaml              # Dataset configuration
  ├── images/
  │   ├── train/            # Training images
  │   └── val/              # Validation images
  └── labels/
      ├── train/            # Training annotations
      └── val/              # Validation annotations

Annotation Format (YOLO):
  Each .txt file contains:
  <class_id> <x_center> <y_center> <width> <height>

  Where all coordinates are normalized (0-1):
  - class_id: 0, 1, or 2 (BMW-X5-M, Volvo-XC40, Jaguar)
  - x_center: center X coordinate / image width
  - y_center: center Y coordinate / image height
  - width: bounding box width / image width
  - height: bounding box height / image height

Image Properties:
  - Format: JPG/PNG
  - Dimensions: Variable (resized to 640x640 for training)
  - Average dimensions: ~1024 × 760 pixels

================================================================================
INFERENCE AND DEPLOYMENT
================================================================================

Using Trained Models:

1. Load Model:
   from ultralytics import YOLO
   model = YOLO('results/models/yolov11_best.pt')

2. Run Inference on Single Image:
   results = model.predict('path/to/image.jpg', conf=0.25)
   results[0].show()  # Display results

3. Run Inference on Multiple Images:
   results = model.predict(['img1.jpg', 'img2.jpg'], conf=0.25)

4. Run Inference on Video:
   results = model.predict('video.mp4', save=True)

5. Real-time Inference from Webcam:
   results = model.predict(source=0, show=True)  # 0 = default webcam

6. Export Model for Deployment:
   model.export(format='onnx')      # ONNX format
   model.export(format='torchscript')  # TorchScript
   model.export(format='coreml')    # CoreML (iOS)
   model.export(format='tflite')    # TensorFlow Lite (mobile)

Inference Parameters:
---------------------
- conf: Confidence threshold (default: 0.25)
- iou: IoU threshold for NMS (default: 0.7)
- imgsz: Inference image size (default: 640)
- device: cuda or cpu
- save: Save results to file
- show: Display results

Example Script:
---------------
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('results/models/yolov11_best.pt')

# Run inference
results = model.predict('test_image.jpg', conf=0.5)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = int(box.cls[0])
        print(f"Detected: {model.names[cls]} ({conf:.2f})")

================================================================================
REFERENCES AND RESOURCES
================================================================================

YOLO Documentation:
- Ultralytics YOLOv11: https://docs.ultralytics.com/models/yolo11/
- Ultralytics YOLOv12: https://docs.ultralytics.com/models/yolov12/
- Ultralytics Main: https://docs.ultralytics.com/

PyTorch:
- Documentation: https://pytorch.org/docs/
- CUDA Setup: https://pytorch.org/get-started/locally/

Computer Vision:
- OpenCV: https://docs.opencv.org/
- Object Detection: https://paperswithcode.com/task/object-detection

Dataset Tools:
- LabelImg: https://github.com/tzutalin/labelImg
- Roboflow: https://roboflow.com/ (alternative annotation tool)

Academic Resources:
- YOLO Original Paper: https://arxiv.org/abs/1506.02640
- COCO Dataset: https://cocodataset.org/


