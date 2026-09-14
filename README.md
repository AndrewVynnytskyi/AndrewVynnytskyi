# Andrii Vynnytskyi

Computer vision and machine learning — detection, tracking and segmentation in PyTorch.
Final-year Mathematics (Data Analytics) student at Gdańsk University of Technology,
looking for an ML / CV internship or junior role — Tricity (on-site or hybrid) or fully remote.

**Currently:** choosing a computer vision topic for my bachelor's thesis.

## Selected projects

- **[Coin-detection](https://github.com/AndrewVynnytskyi/Coin-detection)** — detects, tracks and
  classifies coins in handheld video. ORB + RANSAC camera-motion compensation, Kalman/SORT vs
  greedy tracking, fine-tuned YOLOv8n vs Hough Circles (F1 0.972 vs 0.996 on hand-corrected
  labels); 7 of 10 videos match the true coin count.
- **[SAR-Ship-Segmentation](https://github.com/AndrewVynnytskyi/SAR-Ship-Segmentation)** —
  DeepLabV3+ with my own ASPP and decoder on HRSID radar images. A frozen encoder matched full
  fine-tuning (0.766 vs 0.764 val mIoU, inside run-to-run noise) at about half the training time.
- **[Mot-VisDrone](https://github.com/AndrewVynnytskyi/Mot-VisDrone)** — 22 detector × tracker ×
  DINOv2-embedding runs on VisDrone2019-MOT, scored with TrackEval; best HOTA 41.88 with
  YOLOv8l + ByteTrack. Contrastive appearance embeddings did not beat motion-only tracking —
  the README explains why.
- **[Nyc-taxi-trip](https://github.com/AndrewVynnytskyi/Nyc-taxi-trip)** — ETL of 2M NYC taxi
  trips into PostgreSQL and a PyTorch fare model: RMSE $3.06 vs $10.42 for a SQL baseline.
  Installable package, mypy --strict, pytest, Docker.

## Stack

**ML / DL:** PyTorch, scikit-learn, Ultralytics YOLO, timm, DINOv2, Weights & Biases
**Computer vision:** OpenCV, object detection, multi-object tracking, semantic segmentation, Grad-CAM
**Data:** SQL, PostgreSQL, pandas, NumPy, Apache Superset
**Tools:** Python, Git, Linux, Docker, Hydra / OmegaConf, pytest

## Contact

nikoljavin@gmail.com · [LinkedIn](https://www.linkedin.com/in/andrii-vynnytskyi/)
