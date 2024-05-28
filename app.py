import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the segmentation model
segmentation_model_path = 'unet_model.h5'
segmentation_model = load_model(segmentation_model_path, compile=False)


def load_image(image_path):
    image = Image.open(image_path).resize((256, 256))
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    if image.shape[-1] == 4:  # Check if image has alpha channel and remove it
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    return image


def predict_segmentation(image_path):
    image = load_image(image_path)
    output = segmentation_model.predict(image)
    output = output.squeeze()  # Remove batch dimension
    return output


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Predict segmentation mask
            mask = predict_segmentation(file_path)
            mask_image_path = os.path.join(
                app.config['UPLOAD_FOLDER'], 'mask_' + filename)
            Image.fromarray((mask * 255).astype(np.uint8)
                            ).save(mask_image_path)

            return render_template('result.html', original_image=filename, mask_image='mask_' + filename)
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
