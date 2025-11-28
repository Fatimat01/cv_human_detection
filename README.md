People Detection & Crowd Counting with YOLO
===========================================

This project fine-tunes a pretrained YOLOv8 detector to find people and compute crowd sizes. The workflow lives in `notebooks/training.ipynb` and stores artifacts inside the `cv_human_detection` folder.

Project layout
--------------
- `data/raw/`: raw dataset download (`adilshamim8/people-detection` via KaggleHub).
- `data/processed/people_detection/`: YOLO-formatted images/labels plus `data.yaml`.
- `models/`: training runs and exported prediction media.
- `notebooks/training.ipynb`: end-to-end notebook for download, conversion, training, evaluation, and inference.

Setup
-----
1) Use Python 3.10+ and create a virtual env.
2) Install dependencies: `pip install -r requirements.txt` (PyTorch/ultralytics, OpenCV, KaggleHub, scikit-learn, etc.).
3) Make sure you can access Kaggle data (set `KAGGLE_USERNAME` and `KAGGLE_KEY` if needed).

Training & evaluation
---------------------
Open and run `notebooks/training.ipynb`. The notebook:
- Downloads the `adilshamim8/people-detection` dataset into `data/raw/`.
- Converts annotations to YOLO format (XML or CSV are handled) and creates train/val splits.
- Generates `data/processed/people_detection/data.yaml`.
- Fine-tunes `yolov8n.pt` (change to `yolov8s.pt` for higher accuracy) and validates the best checkpoint.

Inference & crowd sizing
------------------------
The notebook shows how to load the trained weights from `models/people_yolov8n/weights/best.pt`, run detection on an image or video, and count the number of detected people per frame. Annotated outputs are saved under `models/predictions/`.

Tips
----
- Start with fewer epochs/batch size on CPU-only machines, then scale up when a GPU is available.
- If the dataset download changes format, adjust the conversion helpers in the notebook (they support Pascal VOC XML and CSV out of the box).
