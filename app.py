from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from classification_models.tfkeras import Classifiers
import base64
from PIL import Image
from io import BytesIO


def create_app():
    app = Flask(__name__)
    # Carga el modelo
    with open('resnext50_model.json', 'r') as json_file:
        json_savedModel= json_file.read()
    global model
    model = tf.keras.models.model_from_json(json_savedModel)
    model.load_weights('resnext50_weights_.hdf5')
    model.compile(optimizer = 'SGD', loss = "categorical_crossentropy", metrics = ["accuracy"])
    print("Model loaded")

    @app.route('/predict', methods=['POST'])
    def predict():
        # Obtén la imagen del request
        data = request.get_json(force=True)
        base64_string = data.get('image')
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes))
        image = image.resize((224, 224))
        image_array = np.asarray(image)
        image_batch = np.expand_dims(image_array, axis=0)

        yhat = model.predict(image_batch)
        probabilities = yhat[0]

        class_index = np.argmax(probabilities)

        return jsonify({'prediction': class_index.tolist()})

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
