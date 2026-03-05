# Deep-Learning-based-Phishing-Detection

This project implements a machine learning pipeline for detecting phishing websites using features extracted from both URLs and HTML content. The system collects webpage data, generates handcrafted features, and evaluates different classification models.

The pipeline begins by scraping webpage HTML using Selenium and storing the data in a MongoDB database (`scraper.py`). Invalid entries with missing HTML content are removed using a preprocessing script (`delete_null.py`). Feature engineering is performed by extracting attributes from URLs and webpage content, such as domain entropy, suspicious keywords, HTML tag counts, and link structures (`feature_extractor.py`). These features are then assembled into a structured dataset for model training (`feature_main.py`) 

Two machine learning models are implemented and evaluated:

A Random Forest classifier for phishing detection and feature importance analysis (`random_forest.py`) 

A Neural network model trained on the extracted features with SHAP-based interpretability (`neural_networks.py`) 

Model performance is evaluated using standard metrics such as accuracy, F1-score, and confusion matrices.
