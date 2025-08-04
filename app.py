# app.py

import os
import uuid  # Used for generating unique filenames
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from image_classifier import classify_images, get_images_from_url, get_images_from_pdf

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# This new route serves the uploaded/temporary images from the 'uploads' folder
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    """Renders the main HTML page for the user interface."""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    """
    Handles classification requests from the frontend.
    """
    predictions = []
    
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        if file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            images = get_images_from_pdf(file_path)
            
            if images:
                # Classify images and save them temporarily to the server
                # The filename is returned so the frontend can display it
                raw_predictions = classify_images(images)
                for i, img in enumerate(images):
                    # Save each extracted image with a unique ID
                    temp_filename = f"{uuid.uuid4()}.png"
                    temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
                    img.save(temp_filepath)
                    
                    # Add the temporary filename to the prediction result
                    raw_predictions[i]['imagePath'] = temp_filename
                
                predictions = raw_predictions
            
            # Clean up the original uploaded file
            os.remove(file_path)
        else:
            predictions = [{"error": "Invalid file type. Please upload a PDF."}]

    elif 'url' in request.form and request.form['url'] != '':
        url = request.form['url']
        images = get_images_from_url(url)
        
        if images:
            single_prediction = classify_images(images)[0]
            # Add the original URL to the result so the frontend can display the image
            single_prediction['imageUrl'] = url
            predictions = [single_prediction]
        else:
            predictions = [{"error": "Failed to retrieve images or the URL is not a direct image link."}]

    else:
        predictions = [{"error": "No file or URL provided."}]

    return jsonify(predictions)

if __name__ == '__main__':
    if not os.path.exists('image_classifier_model.pth'):
        print("Warning: The trained model 'image_classifier_model.pth' was not found.")
        print("Please train your model first by running: `python image_classifier.py --train`")
        
    app.run(debug=True)