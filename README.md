# ChessVision AI: Chessboard Corner Detection and Piece Recognition using Deep Learning

## Project Overview
This project implements **ChessVision AI**, a system for detecting chessboard squares and recognizing chess pieces from images using deep learning. The goal is to convert raw chessboard images into a **FEN (Forsyth-Edwards Notation)** representation and visualize the board with Unicode chess symbols.

The project demonstrates the integration of **object detection (YOLO)** with **image preprocessing**, **CNN-based regression**, and **geometric mapping** for accurate chessboard state extraction.

Two main Jupyter Notebooks are included:

1. **YOLO Training Notebook** (`notebooks/01_ChessPiece_YOLO_Training.ipynb`):  
   - Trains a custom YOLO model to detect 12 types of chess pieces.
   - Evaluated using precision, recall, and mAP metrics.  
   - Produces a trained model `best.pt` ready for inference.

2. **Corner Detection and FEN Generation Notebook** (`notebooks/02_Corner_Detection_to_FEN.ipynb`):  
   - Detects the four corners of the chessboard using a CNN.
   - Applies a **perspective transformation** to align the board.
   - Combines piece detection and geometric mapping to generate the **FEN notation**.
   - Saves the trained CNN as `corner_detection_model.h5`.
  
## Dataset
The dataset used for this project was obtained from **Roboflow**, containing labeled images of chessboards with annotations for all chess pieces.
Available at: [Chess Pieces Dataset](https://public.roboflow.com/object-detection/chess-full).  


> **Note:** Due to file size limitations, the full dataset is **not included**. A small sample is provided under `data/label-v03/` for demonstration purposes.


## Folder Structure
```
ChessVision-DeepLearning/
├── data/
│   ├── label-v03
│   │   ├── test
│   │   │   ├── images
│   │   │   └── labels
│   │   └── train
│   │       ├── images
│   │       └── labels
│   └── 19.JPG  # sample chessboard image for testing and visualization
├── models/
│   ├── best.PT
│   └── corner_detection_model.H5
├── notebooks/
│   ├── 01_ChessPiece_YOLO_Training.ipynb
│   └── 02_Corner_Detection_to_FEN.ipynb
├── LICENSE
├── README.md
└── requirements.txt
````

## Usage
1. Clone the repository:
```bash
git clone https://github.com/Zahra-Ghaffary/ChessVision-DeepLearning.git
cd ChessVision-DeepLearning
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open notebooks in Jupyter or VSCode to reproduce results.

## Results Highlights
* **YOLO Piece Detection:** Achieved overall mAP ~0.98, with high precision and recall across all 12 piece classes. Detected pieces are accurately assigned to board squares using IoU-based mapping.
* **Chessboard Mapping:** Perspective correction via CNN supports reliable FEN generation even with slight board distortions. Test images show MSE ~9 and R² ~0.42, sufficient for accurate corner localization.
* **Visualization:** Successfully displays all detected pieces on an 8×8 board using Unicode symbols.

## Features
* **Custom YOLO** model for detecting 12 chess piece classes
* **CNN-based** corner detection for perspective correction
* **IoU-based** square mapping to assign detected pieces
* **Automatic FEN** string generation

## References

* Original dataset: [Chess Pieces Dataset](https://public.roboflow.com/object-detection/chess-full)
* YOLO by Ultralytics: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
