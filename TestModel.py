from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import tensorflow as tf

import numpy as np
import cv2
import matplotlib.pyplot as plt

opciones = ['Neumonía', 'Normal']

width_shape = 224
height_shape = 224

modelt = load_model("D:/Ome/Documents/Ale/Ucamp/Módulo 7/ClaseHoy/model.h5")
print("Modelo cargado exitosamente")

imaget_path = "ImagenPrueba.jpeg"
imaget=cv2.resize(cv2.imread(imaget_path), (width_shape, height_shape), interpolation = cv2.INTER_AREA)

xt = np.asarray(imaget)
xt=preprocess_input(xt)
xt = np.expand_dims(xt,axis=0)

print("Predicción")
preds = modelt.predict(xt)

print("Predicción:", opciones[np.argmax(preds)])
plt.imshow(cv2.cvtColor(np.asarray(imaget),cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
