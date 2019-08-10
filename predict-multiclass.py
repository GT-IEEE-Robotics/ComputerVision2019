import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 150, 150
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("Label: Corner")
  elif answer == 1:
    print("Labels: Cube")
  elif answer == 2:
    print("Label: Sphere")

  return answer

pizza_t = 0
pizza_f = 0
poodle_t = 0
poodle_f = 0
rose_t = 0
rose_f = 0

for i, ret in enumerate(os.walk('./test-data/corners')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: Corner")
    result = predict(ret[0] + '/' + filename)
    if result == 0:
      pizza_t += 1
    else:
      pizza_f += 1

for i, ret in enumerate(os.walk('./test-data/cubes')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: Cube")
    result = predict(ret[0] + '/' + filename)
    if result == 1:
      poodle_t += 1
    else:
      poodle_f += 1

for i, ret in enumerate(os.walk('./test-data/spheres')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: Sphere")
    result = predict(ret[0] + '/' + filename)
    if result == 2:
      rose_t += 1
    else:
      rose_f += 1

"""
Check metrics
"""
print("True corner: ", pizza_t)
print("False corner: ", pizza_f)
print("True cube: ", poodle_t)
print("False cube: ", poodle_f)
print("True sphere: ", rose_t)
print("False sphere: ", rose_f)
