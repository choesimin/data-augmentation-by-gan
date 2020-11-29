import cv2
import os
import numpy as np
from PIL import Image
 
path = 'results_gray\\bs512_lrg0.0002_lrd0.0002_ep300'
imagePaths = [os.path.join(path,file_name) for file_name in os.listdir(path)]
for imagePath in imagePaths:
    img = Image.open(imagePath).convert('L')
    img_numpy = np.array(img, 'uint8')
    cv2.imwrite("results_gray\\bs512_lrg0.0002_lrd0.0002_ep300\\" + imagePath.split("\\")[-1], img_numpy)
print("All Done")
