import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import cv2
import tensorflow as tf
import keras

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def mob():
    return render_template('index.html')

@app.route('/predictmob', methods=['POST'])
def predictmob():
    '''
    For rendering results on HTML GUI
    '''

    file = request.files['file']
    filestr = file.read()
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))

  #  img = request.files['file']
    #img = cv2.resize(img, (256, 256))
   # img = img.reshape((1, 256, 256, 3))

    # Convert image to array
    Third_Way_Test = tf.keras.utils.img_to_array(img)
    Third_Way_Test = np.expand_dims(Third_Way_Test, axis=0)
    a = model.predict(Third_Way_Test)


    # Define class labels
    class_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

    # Get the predicted class index
    predicted_class_index = np.argmax(a)

    # Get the predicted class label
    predicted_class_label = class_labels[predicted_class_index]

    # Print the predicted class label and result
    print("Predicted Class: ", predicted_class_label)
    print("Prediction Result: ", a)




    return render_template('index.html', prediction_text='PREDICTED RICE CATEGORY is {}'.format(predicted_class_label))


if __name__ == "__main__":
    app.run()


app.debug = True
