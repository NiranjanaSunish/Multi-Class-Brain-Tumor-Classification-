
# A Comprehensive AI Framework for MRI-Based Multi-Class Brain Tumor Classification

## Project Overview

This project aims to develop an automated brain tumor classification system using advanced AI and machine learning techniques. Brain tumors present significant challenges due to their high mortality rates, making early and accurate classification crucial for effective treatment planning. This framework integrates data preprocessing, segmentation, feature extraction, feature selection, and classification to deliver a robust and efficient solution for brain tumor diagnosis.

## Features

- **Data Preprocessing**: 
  - Skull Stripping using the Brain Extraction Tool (BET) to remove non-brain tissues.
  - Contrast Enhancement through Discrete Histogram Equalization to improve visibility of tumor regions.
  
- **Tumor Segmentation**: 
  - Model: ResNet50 U-Net architecture.
  - Combines the hierarchical feature learning capability of ResNet50 with the pixel-wise segmentation capabilities of U-Net.

- **Feature Extraction and Selection**:
  - **Feature Extraction**:
    - Discrete Wavelet Transform (DWT) for frequency-domain features.
    - Gray Level Co-occurrence Matrix (GLCM) for texture features.
  - **Feature Selection**:
    - t-SNE for dimensionality reduction while preserving relevant features.

- **Classification**:
  - Model: KNN Clustering-based Support Vector Machine (SVM).
  - Combines the clustering ability of KNN with the classification power of SVM to achieve optimal classification results.

## Datasets

1. **VASARI MRI Dataset**:
   - 60 patients, each with 4 different 3D MRI modalities: T1, T2, FLAIR, and post-contrast T1-weighted (T1-CE).
   - Classes: Astrocytoma, Oligodendroglioma, Glioblastoma Multiforme.

2. **BraTS 2019 Dataset**:
   - 335 patients, each with 4 different 3D MRI modalities: T1, T2, FLAIR, and post-contrast T1-weighted (T1-CE).
   - Classes: High-Grade Gliomas (HGG) and Low-Grade Gliomas (LGG).

## Methodology

1. **Preprocessing**:
   - Apply skull stripping and contrast enhancement to MRI images to focus the analysis on brain tissues and enhance the visibility of tumor regions.

2. **Tumor Segmentation**:
   - Use the ResNet50 U-Net architecture, pre-trained on ImageNet, for pixel-wise segmentation of brain tumors.

3. **Feature Extraction & Selection**:
   - Extract frequency-domain features using DWT and texture features using GLCM.
   - Perform dimensionality reduction using t-SNE to enhance the performance of the classifier.

4. **Classification**:
   - Implement a KNN Clustering-based SVM model to classify the tumors, with and without segmentation.

