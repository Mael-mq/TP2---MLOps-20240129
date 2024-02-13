from flask import Flask, render_template, request, jsonify
from flask_uploads import UploadSet, configure_uploads, IMAGES
import numpy as np
import pandas as pd 
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pickle', 'rb'))
cols = ['MODELYEAR', 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']

from keras.applications.resnet50 import ResNet50
rnModel = ResNet50()
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import decode_predictions

def predictImageCategory(imagePath):
    # voir TP1 pour VGG16
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    prediction = rnModel.predict(image)
    label = decode_predictions(prediction)
    label = label[0][0]
    return (label[1], label[2]*100, "%")

photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = './static/img'
configure_uploads(app, photos)

@app.route('/')
def home():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    final = np.array(features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction=model.predict(data_unseen)
    result=prediction[0]
    return render_template('input.html',resultat=f"Les emissions de co2 sont de : {result:.2f}")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    features = [x for x in request.args.values()]
    final = np.array(features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction=model.predict(data_unseen)
    result=prediction[0]
    return jsonify(result)

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if 'photos' in request.files:
        filename = photos.save(request.files['photos'])
        prediction = predictImageCategory(f'./static/img/{filename}')
        return render_template('upload.html', filename=filename, prediction=prediction)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run()
