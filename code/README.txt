============================================================
  CO3519 Assessment 2 - Facial Emotion Recognition
  MSS Fernando | G21266967 | UCLan | April 2025
============================================================

FILES
-----
  train.py      - Trains all 6 models (Custom CNN, VGG16, ResNet50
                  on FER2013 and CK+). Saves results to results/
  crossval.py   - Runs k-fold cross-validation for all 3 models.
                  5-fold for CNN, 3-fold for VGG16 and ResNet50.
                  Saves CV charts and reports to results/cv/


DATASET FOLDER STRUCTURE REQUIRED
----------------------------------
The code expects datasets in the following layout:

  datasets/
    FER2013/
      train/   (14,400 images - 2,400 per class)
      val/     (1,800 images  - 300 per class)
      test/    (1,800 images  - 300 per class)
    CK_Plus/
      train/   (1,872 images  - augmented 6x)
      val/     (90 images     - 15 per class)
      test/    (48 images     - 8 per class)

  Each class subfolder is named:  AN  FE  HA  NE  SA  SU


ENVIRONMENT SETUP
-----------------
  Python 3.10+ with the following packages:
    tensorflow (tensorflow-macos + tensorflow-metal on Apple Silicon)
    numpy
    matplotlib
    seaborn
    scikit-learn
    Pillow

  Install via conda:
    conda create -n fer_env python=3.10
    conda activate fer_env
    pip install tensorflow-macos tensorflow-metal
    pip install numpy matplotlib seaborn scikit-learn Pillow


HOW TO RUN
----------
  Step 1 - Train all models:
    conda activate fer_env
    python train.py

  Step 2 - Run cross-validation (run after training):
    conda activate fer_env
    python crossval.py

  Note: train.py takes approximately 2-4 hours on Apple Silicon.
        crossval.py takes approximately 8-12 hours (VGG16 and
        ResNet50 CV are computationally expensive).


RESULTS PRODUCED
----------------
  After running train.py:
    results/confusion_matrices/   - 6 confusion matrix images
    results/training_curves/      - 6 training/validation curve images
    results/sample_predictions/   - sample classified face images
    results/classification_reports/ - precision/recall/F1 text reports

  After running crossval.py:
    results/cross_validation/     - 6 CV bar charts + text summaries
                                    CV_summary.txt has all results


