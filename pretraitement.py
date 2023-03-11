import os
import cv2
import numpy as np
from PIL import Image

def preprocessing(pathDir, pathNumpy, imgSize):
    """
    Permet de : Charger l’ensemble des images présentes dans un répertoire, 
    de les recadrer (crop ou scale) dans un format carré, de les redimensionner 
    de les fusionner dans un tableau Numpy et d’exporter ce tableau sur le disque
    """
    images = []
    labels = []
    label_to_index = {}
    index_to_label = {}
    label_index = 0
    for subdir, dirs, files in os.walk(pathDir):
        for file in files:
            img = Image.open(os.path.join(subdir, file))
            width, height = img.size
            if width > height:
                left = (width - height) / 2
                right = left + height
                top = 0
                bottom = height
            else:
                top = (height - width) / 2
                bottom = top + width
                left = 0
                right = width
            img = img.crop((left, top, right, bottom))
            img = img.resize((imgSize, imgSize))
            img = np.array(img)
            if img.shape[0] != imgSize or img.shape[1] != imgSize:
                continue
            images.append(img)
            label = subdir.split(os.path.sep)[-1]
            if label not in label_to_index:
                label_to_index[label] = label_index
                index_to_label[label_index] = label
                label_index += 1
            labels.append(label_to_index[label])
    target_shape = (imgSize, imgSize, 3)
    for i, img in enumerate(images):
        if img.shape != target_shape:
            images[i] = np.resize(img, target_shape)
    images = np.stack(images)
    labels = np.array(labels)

    np.save(os.path.join(pathNumpy, 'images.npy'), images)
    np.save(os.path.join(pathNumpy, 'labels.npy'), labels)
    np.save(os.path.join(pathNumpy, 'label_to_index.npy'), label_to_index)
    np.save(os.path.join(pathNumpy, 'index_to_label.npy'), index_to_label)

# def preprocessing(pathDir, pathNumpy, resizeImg, imgSize):
#     """
#     Permet de : Charger l’ensemble des images présentes dans un répertoire, 
#     de les recadrer (crop ou scale) dans un format carré, de les redimensionner 
#     de les fusionner dans un tableau Numpy et d’exporter ce tableau sur le disque
#     """

#     for felin_label in os.listdir(pathDir):
#         path_label = pathDir + '\\' + felin_label
#         imgs = []

#         for img_felin in os.listdir(path_label):
#             path_img_felin = path_label + '\\' + img_felin
#             img = cv2.imread(path_img_felin)
#             if resizeImg == True:
#                 if img is None:
#                     continue
#                 if img.size == 0:
#                     continue
#                 img = cv2.resize(img, imgSize)
#             imgs.append(img)
#         imgs = np.array(imgs) / 255.
#         if not os.path.exists(pathNumpy):
#             os.makedirs(pathNumpy)
#         np.save(os.path.join(pathNumpy, felin_label + '.npy'), imgs)
#         # np.save(pathNumpy + '\\' + felin_label + '.npy', imgs)

if __name__ == '__main__':
    # pathNumpy = '.\\numpy_felin_label'
    # pathNumpy2 = '.\\numpy_felin_label_light'
    # pathDir = '.\\felin_label_light'
    # resizeImg = True
    # imgSize = (224, 224)
    # preprocessing(pathDir, pathNumpy2, resizeImg, imgSize)

    pathDir = './felin_label'
    pathNumpy = './np_felin'
    imgSize = 224
    preprocessing(pathDir, pathNumpy, imgSize)

