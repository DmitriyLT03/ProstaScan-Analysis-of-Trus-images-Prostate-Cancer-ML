import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


class Segmentator:
    def __init__(self, path_model: str):
        self.model = ort.InferenceSession(path_model)
        self.input_resize = (768, 768)
        self.output_resize = (980, 615)

    def predict(self, image):
        image = cv2.resize(
            image,
            (1280, 872),
            interpolation=cv2.INTER_AREA
        )
        image = image[135:750, 145:1125]
        preprocessed_image = self.preproc_input(image)
        outputs = self.model.run(
            None,
            {"actual_input_1": preprocessed_image},
        )
        return cv2.addWeighted(
            image,
            1,
            self.preproc_output(outputs),
            0.6,
            0
        )

    def preproc_input(self, image):
        b, g, r = cv2.split(image)
        image = cv2.merge([b, g, r])
        image = cv2.resize(
            image,
            self.input_resize,
            interpolation=cv2.INTER_AREA
        ).astype('float32')
        image = np.moveaxis(image, -1, 0)
        return np.array([image])

    def preproc_output(self, output):
        predict = (output[0][0][0]*255).astype(np.uint8)
        predict = cv2.resize(
            predict,
            (980, 615),
            interpolation=cv2.INTER_AREA
        )
        img = np.array(Image.fromarray(predict))
        rgb_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        _, _, red = cv2.split(rgb_image)
        zeros = np.zeros(red.shape, np.uint8)
        redBGR = cv2.merge([zeros, zeros, red])
        return redBGR
