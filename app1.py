
# Display the Folders/Classes
# Manual garbage collection process

from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, render_template

import random
from itertools import chain
import os
import numpy as np
import gc

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array, load_img

gc.collect()



fpath = "D:\\full-stack\\dl\\images\\Images"

"""commenting """
dog_classes = os.listdir(fpath)
breeds = [breed.split('-', 1)[1]
          for breed in dog_classes]  # view some of the Labels
"""commenting"""


x = []
# y will have its breed name
y = []
fullpaths = [
    "D:\\full-stack\\dl\\images\\Images\\{}".format(dog_class) for dog_class in dog_classes]
for counter, fullpath in enumerate(fullpaths):
    for imgname in os.listdir(fullpath):
        x.append([fullpath+'\\' + imgname])
        y.append(breeds[counter])

x = list(chain.from_iterable(x))
combined = list(zip(x, y))

random.shuffle(combined)

x[:], y[:] = zip(*combined)
x = x[:1000]
y = y[:1000]
le = LabelEncoder()
le.fit(y)


# from keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template("start.html")


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        fullpaths = [
            'D:\\full-stack\\dl\\static\\uploads\\{}'.format(filename)]
        # print (fullpaths)
        img_data = np.array(
            [img_to_array(load_img(img, target_size=(299, 299)))for img in fullpaths])
        x_test1 = img_data/255.

 

        # rescale to 0-1. Divide by 255 as its the max rgb value
        from keras import models
        
        model = models.load_model('my_model.h5',compile=False)
        test_predictions = model.predict(x_test1)
        print(test_predictions)
        # from sklearn.preprocessing import LabelEncoder
        # le = LabelEncoder()
        # le.fit(y)
        predictions = le.classes_[np.argmax(test_predictions, axis=1)]
        # # print (predictions[0])
        name = predictions[0].upper().replace("_", "")
        return render_template('result.html', prediction=name, src="https://simple.wikipedia.org/wiki/" + predictions[0])


if __name__ == "__main__":
    app.run(debug=True)


app = Flask(__name__)
