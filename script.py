import tensorflow as tf
from keras.preprocessing.image  import ImageDataGenerator
from keras.utils.vis_utils import plot_model
import numpy as np
from keras.utils import load_img, img_to_array, plot_model
import sys
import visualkeras
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw

print(f"Tensorflow version : {tf.__version__}")
print(f"Keras version : {tf.keras.__version__}")
print(f"Sys version : {sys.version}")

train_datagen = ImageDataGenerator(rescale = 1./255,
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip = True)

training_set = train_datagen.flow_from_directory('Dataset/Train',
                                                 target_size = (64, 64),
                                                 batch_size = 64,
                                                 class_mode = 'categorical',
                                                 shuffle = False)

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('Dataset/Test',
                                            target_size = (64, 64),
                                            batch_size = 64,
                                            class_mode = 'categorical',
                                            shuffle = False)

training_set.samples

cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

cnn.add(tf.keras.layers.Dense(units=4, activation='softmax'))

cnn.compile(optimizer = 'Adam',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy'])

hist = cnn.fit(x = training_set, validation_data = test_set, epochs = 35)

print(test_set.class_indices)
predictions = cnn.predict(test_set)
print(predictions)
predictions = np.argmax(predictions, axis = 1)
print(predictions)

cnn.summary()

plot_model(cnn, to_file = "Models/plot_model.png", show_shapes = True, show_layer_names = True)
font = ImageFont.load_default()
visualkeras.layered_view(cnn, legend = True, font = font, to_file = "Models/layered_model.png")

plt.plot(hist.history['accuracy'])
plt.plot(hist.history["val_accuracy"])
plt.title("Model accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Epoch")
plt.legend(["train", "test"], loc = "upper left")
plt.savefig("Models/accuracy_model.png", dpi = 500)
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history["val_loss"])
plt.title("Model loss")
plt.xlabel("Loss")
plt.ylabel("Epoch")
plt.legend(["train", "test"], loc = "upper left")
plt.savefig("Models/loss_model.png", dpi = 500)
plt.show()


dictionary = {0: "Dog", 1: "Elephant", 2: "Panda", 3: "Tiger"}

for i in range(1,5):
    test_image = load_img("Predict/test" + str(i) + ".jpg", target_size = (64, 64))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image/255.0)

    prediction = np.argmax(result, axis = 1).astype(int)
    print(result)
    print("-----------")
    print(prediction)
    
    img = Image.open("Predict/test" + str(i) + ".jpg")
    img_draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Chalkduster.ttf",26)
    img_draw.text((28, 36), str(dictionary[prediction[0]]), fill=(255, 0, 0), font = font)
    img.save("Images_with_text/test" + str(i) + "_text.jpg")
    print("In the picture is", dictionary[prediction[0]])
    
results_validation = cnn.evaluate(test_set, batch_size = 32)
print("test loss, test acc: ", results_validation)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print(accuracy_score(test_set.classes, predictions))
accuracy_score(test_set.classes, predictions)
cm = confusion_matrix(test_set.classes, predictions)
print(cm)

import seaborn as sns
plot = sns.heatmap(cm, annot = True)
fig = plot.get_figure()
fig.savefig("Heatmaps/out.png") 

print(classification_report(test_set.classes, predictions))
classification_report(test_set.classes, predictions)

model_json = cnn.to_json()
with open("CNN/cnn3.json", "w") as json_file:
    json_file.write(model_json)
    
from keras.models import save_model
network_saved = save_model(cnn, "Weights/weights3.hdf5")

with open("CNN/cnn3.json", "r") as json_file:
    json_saved_model = json_file.read()
    
network_loaded = tf.keras.models.model_from_json(json_saved_model)
network_loaded.load_weights("Weights/weights3.hdf5")

network_loaded.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])
network_loaded.summary()
