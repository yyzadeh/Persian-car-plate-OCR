# Persian Car Plate OCR🚗🔍

This project detects **Persian car license plates** using **YOLOv12** and recognizes their characters using **PyTesseract**. It is an end-to-end Optical Character Recognition (OCR) pipeline for Iranian license plates.

---

## 🔌 Overview

- Car plate detection using YOLOv12 (fine-tuned on a custom dataset)
- Crop detected part
- OCR (text recognition) using Tesseract (supports Persian script)
- Dataset sourced from Kaggle and re-annotated via Roboflow

---

## 🔄 Workflow

1. Collected a dataset of Iranian car plates from Kaggle
2. augmented it using [Roboflow](https://roboflow.com/)
3. Exported the dataset in YOLO format and imported it into Google Colab
4. Installed YOLOv12 and all required dependencies
5. Fine-tuned `yolov12s` on the custom dataset
6. Ran inference on test image to detect plates
7. Cropped the detected plate region
8. Applied PyTesseract to extract Persian text

---

## 🔧 Installation

### Requirements

- Python 3.8+
- OpenCV
- ultralytics
- Tesseract OCR
- PyTesseract
- Torch torchvision

---

## 👁️ Example

| Input Image |          Cropped Plate        |    OCR Output     |
|-------------|-------------------------------|-------------------|
| car.jpg     | runs/detect/inference/car.jpg | `۲۴ ب ۴۸۳ ایران ۳۱` |

---

## 📂 Dataset

- Initial dataset downloaded from Kaggle
- Annotated and cleaned in Roboflow
- Exported in YOLOv12 format and used in Colab

---

## 📈 Results

- plate detected
- OCR accuracy depends on resolution and angle <br>
  (pytesseract doesn't work well on Persian characters)

---

## ✍️ Author

Developed by **Yousef Yousefian Zadeh**  
Electrical Engineering | Isfahan University of Technology

---

## ✨ Acknowledgements

- [YOLOv12](https://github.com/YuriSizov/YOLOv12)
- [PyTesseract](https://github.com/madmaze/pytesseract)
- [Roboflow](https://roboflow.com)
- [Kaggle](https://www.kaggle.com)
