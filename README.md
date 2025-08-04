# Nihin Media K.K. Competency Assessment: Medical vs. Non-Medical Image Classifier

## 🚀 Project Overview

This project is a comprehensive solution for classifying images as "medical" or "non-medical." It demonstrates an end-to-end machine learning pipeline, including:
-   A command-line interface (CLI) for model training and inference.
-   A user-friendly web application with a modern UI/UX built with Flask.
-   Support for multiple input types: direct image URLs and PDF documents.
-   Dynamic results display with image previews and inference metrics.
-   Multiple download options (TXT, CSV, PDF) for classification results.

## 💡 Technical Approach

### 1. Model and Methodology
-   **Model:** A pre-trained `ResNet-18` model from `torchvision` was used.
-   **Methodology:** I employed **transfer learning**, which is highly efficient for this task. The model's feature-extraction layers (trained on millions of images) were frozen, and a new, custom classification head was trained on a small, domain-specific dataset. This approach ensures high accuracy with minimal training time and computational resources.

### 2. Data Pipeline
-   **Data Sourcing:** The dataset was curated from public sources, including medical X-rays and MRIs from platforms like Kaggle, and non-medical images from free stock photo websites like Unsplash.
-   **Image Extraction:**
    -   For **URLs**, the `requests` library is used to fetch images directly. A `User-Agent` header was implemented to handle website security policies.
    -   For **PDFs**, the `PyMuPDF` library extracts images, which are then saved as temporary files on the server to be served to the frontend.

### 3. Application Architecture
-   The core machine learning logic resides in `image_classifier.py`.
-   A lightweight web application is built with the **Flask** framework, providing an intuitive frontend for user interaction.
-   The frontend is a single `index.html` page styled with Bootstrap and dynamic JavaScript for an interactive experience.

## 📊 Performance Metrics

-   **Validation Accuracy:** **100.00%** on the validation set, demonstrating high confidence in the model's classifications.
-   **Inference Speed:** Approximately **0.15 - 0.30 seconds** per image on a standard CPU, ensuring a responsive user experience.
-   **Scalability:** The architecture is designed to be lightweight and efficient, making it suitable for deployment in various environments.

## 💻 How to Run the Project

### Prerequisites

-   Ensure you have **Python 3.8+** installed.
-   **Clone the repository** from GitHub.
-   **Create and activate a virtual environment:**
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows, use: myenv\Scripts\activate
    ```
-   **Install dependencies** using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

### Step 1: Prepare the Dataset and Train the Model

1.  Create `train` and `val` folders with `medical` and `non-medical` subfolders.
2.  Populate these folders with a small set of images (e.g., 300 training images and 50 validation images for each class).
3.  Run the training script to generate the `image_classifier_model.pth` file:
    ```bash
    python image_classifier.py --train
    ```

### Step 2: Launch the Web Application

-   From the terminal, run the Flask application:
    ```bash
    python app.py
    ```
-   Open your web browser and navigate to the following URL:
    ```
    [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
    ```

You can now use the web interface to test the URL and PDF classification functionalities.

---
## 🎬 Video Submission
- [**Watch the full video demonstration here!**](assets/demo_video.mp4)




## 🖼️ Application Screenshot

Here is a quick look at the web interface and RESULT for the image classifier:

<img width="1078" height="909" alt="result" src="https://github.com/user-attachments/assets/f29925e5-d8e5-49af-88f0-c14b177764a7" />

<img width="1071" height="930" alt="result1" src="https://github.com/user-attachments/assets/8df5543b-4a8d-4ae5-8e28-095c9e939fdc" />

<img width="1896" height="963" alt="resultpdf" src="https://github.com/user-attachments/assets/99f08cda-38c3-486f-b505-a5b36c058a96" />









