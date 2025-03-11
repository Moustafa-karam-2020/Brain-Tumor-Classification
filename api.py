import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf

# Load the entire model
model = tf.keras.models.load_model('Brain_Tumor_Image_Classification_Model.pkl')

# Define class labels
class_names = ["pituitary", "notumor", "meningioma", "glioma"]
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (240, 240))
    img = img / 255.0  # Normalize pixel values
    return img

def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((240, 240))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    result01 = np.argmax(result, axis=1)
    return result01

def get_className(class_index):
    class_names = ["pituitary", "notumor", "meningioma", "glioma"]
    return class_names[class_index[0]]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)
