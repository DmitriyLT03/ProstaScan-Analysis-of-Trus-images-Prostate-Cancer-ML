import cv2
import numpy as np
from PIL import Image


class Visual:
    def __init__(self):
        pass
    
    def preproc_predict(image):
        predict = (image[0][0]*255).astype(np.uint8)
        img  = np.array(Image.fromarray(predict))
        return img

    def green_chanel(image):
        image = cv2.resize(
            image,
            (1280, 872),
            interpolation = cv2.INTER_AREA
        )
        _, green, _ = cv2.split(image)
        zeros = np.zeros(green.shape, np.uint8)
        greenBGR = cv2.merge([zeros,green,zeros])
        return greenBGR

    def blue_chanel_predict(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,
                           (1280, 872), 
                           interpolation = cv2.INTER_AREA
                           )
        blue, _, _ = cv2.split(image)
        zeros = np.zeros(blue.shape, np.uint8)
        blueBGR = cv2.merge([blue,zeros,zeros])
        return blueBGR

    def green_chanel_predict(image):
        image = cv2.cvtColor(
            image, 
            cv2.COLOR_BGR2RGB
        )
        image = cv2.resize(
            image, 
            (1280, 872), 
            interpolation = cv2.INTER_AREA
        )
        _, green, _ = cv2.split(image)
        zeros = np.zeros(green.shape, np.uint8)
        green = cv2.merge([zeros,green,zeros])
        return green

    def calc_IoU(image_first, image_second):
        bitwiseOR = cv2.bitwise_or(
            image_first, 
            image_second
        )
        bitwiseOR = bitwiseOR / 255
        bitwiseOR = (np.rint(bitwiseOR)).astype(int)
        
        summary = bitwiseOR.sum()
        bitwiseAnd = cv2.bitwise_and(
            image_first, 
            image_second
        )
        bitwiseAnd = bitwiseAnd / 255
        bitwiseAnd = (np.rint(bitwiseAnd)).astype(int)
        peres = bitwiseAnd.sum()
        
        return peres / summary