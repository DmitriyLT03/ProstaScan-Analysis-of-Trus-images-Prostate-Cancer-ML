import cv2
import numpy as np
import albumentations as albu
from PIL import Image


class Visual:
    def __init__(self):
        pass
    
    def preproc_predict(image):
        predict = (image[0][0]*255).astype(np.uint8)
        predict = cv2.resize(
            predict,
            (980, 615), 
            interpolation = cv2.INTER_AREA
        )
        img  = np.array(Image.fromarray(predict))
        return img

    def green_chanel(image):
        image = cv2.resize(
            image,
            (980, 615),
            interpolation = cv2.INTER_AREA
        )
        _, green, _ = cv2.split(image)
        zeros = np.zeros(green.shape, np.uint8)
        greenBGR = cv2.merge([zeros,green,zeros])
        return greenBGR

    def blue_chanel_predict(image):
        image = cv2.cvtColor(
            image, 
            cv2.COLOR_BGR2RGB
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
    
    def Normalize(self):
        return albu.Compose([
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                always_apply=False,
                p=1.0
            )
        ])