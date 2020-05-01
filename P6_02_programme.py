import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import argparse

# Parameters
IMG_SIZE = 224
CLASS_NAMES = np.load(os.path.join(os.getcwd(), "data/class_names.npy"))
# Args parser
parser = argparse.ArgumentParser(description="Dog Breed recognition by Julien Martin")
parser.add_argument("--image", "-i", required=True, help="Path to input image", default=os.path.join(os.getcwd(), "test_image.jpg"))
args = parser.parse_args()
image_path = args.image

# Image preprocessing
image = np.asarray(Image.open(image_path).resize((IMG_SIZE, IMG_SIZE)))
image_preprocessed = preprocess_input(image)
image_to_predict = (np.expand_dims(image_preprocessed, 0))

# Model loading and prediction
model = load_model(os.path.join(os.getcwd(), "callbacks/checkpoints/DOG-BREED-CNN-VGG16-FCB-1588331048.hdf5"))
predictions = model.predict(image_to_predict)

# Print 5 max predictions
n_pred = 3
print(np.max(predictions[0]))
if np.max(predictions[0]) == 1:
    n_pred = 1
three_predictions = np.sort(predictions[0])[::-1][:n_pred]
three_labels = CLASS_NAMES[sorted(range(len(predictions[0])), key=lambda i: predictions[0][i])[-n_pred:]][::-1]
print("\n-----------------------------")
print("Le modèle a prédit :")
for i in range(len(three_predictions)):
    print("Race: {}, accuracy: {:.4f}%".format(three_labels[i], three_predictions[i] * 100))
print("-----------------------------")