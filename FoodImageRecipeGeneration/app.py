from flask import Flask
from flask import render_template
from flask import request
import utils.SQLiteDB as dbHandler
from utils.CNNModel import Network, create_model
import tensorflow as tf
import numpy as np
import keras.utils as image

app = Flask(__name__)


def prediction(savedModel, inputImage):
    test_image = image.load_img(
        inputImage,
        target_size=(300, 300))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = savedModel.predict(test_image)
    print("Predicted result", result)
    output = {0: 'apple_pie', 1: 'cheesecake', 2: 'chicken_curry', 3: 'french_fries',
              4: 'fried_rice', 5: 'hamburger', 6: 'hot_dog', 7: 'ice_cream', 8: 'omelette',
              9: 'pizza', 10: 'sushi'}
    print("Output labels = ", output)
    print("output[np.argmax(result)] = ", output[np.argmax(result)])
    return output[np.argmax(result)]


@app.route("/", methods=['POST', 'GET'])
def home():
    savedModel = tf.keras.models.load_model('.\\pretrainedWeights\\custom_model.h5py')
    print("weight loaded")
    savedModel.summary()
    pred_label = prediction(savedModel, './/dataset//food11//test//cheesecake//305424.jpg')
    recipe = dbHandler.retrieveRecipeDataWithItemName(pred_label)
    return pred_label, recipe
    # dbHandler.createTableIfNotExist()
    # if request.method == 'POST':
    #     itemName = request.form['itemName']
    #     recipeDescription = request.form['recipeDescription']
    #     ingredients = request.form['ingredients']
    #     dbHandler.insertRecipeData(itemName, recipeDescription, ingredients)
    #     users = dbHandler.retrieveUsers()
    #     return render_template('index.html', users=users)
    # else:
    #     return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
