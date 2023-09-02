# Proyeto Final Ucamp - Ciencia de Datos e Inteligencia Artificial

*Autor: [Alejandro Ometeotl Palomares González](https://www.linkedin.com/in/alejandromepg/)*

## Introducción
Implementaremos una red neuronal usando keras-tensorflow y la ejecutaremos en un servicio web de flask
Donde clasificamos una serie de imágenes médicas utilizando redes neuronales convolucionales para determinar si un paciente padece neumonía o no.

1. Preparación del entorno
$ python -3 -m venv myvenv
$ myvenv\scripts\activate
$ pip install tensorflow
$ pip install jupyter
$ pip install keras
$ pip install pandas
$ pip install numpy scipy Pillow cython matplotlib scikit-image opencv-python h5py imgaug IPython[all]

2. Entrenar la red neuronal
Descargar el repositorio
Abrir terminal en la carpeta y correr jupyter notebook en VSC


Ejecutar Copia_de_Proyecto_M7_AOPG.ipynb

3. Probar la red neuronal
$ python TestModel.py
4. Probar el API de Flask
$ python app.py
