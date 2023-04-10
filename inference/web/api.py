import os
import sys
import cv2
from flask import Flask, make_response, request, send_file
from pathlib import Path
from inference import Segmentator
# try:
#     from ..inference import Segmentator
# except ImportError:
#     q = Path(sys.path[0])
#     sys.path.append(q.parents[0])
#     print(sys.path)
#     from ..inference import Segmentator


s = Segmentator("../model/UNET++_trained.onnx")

def process_image(nn: Segmentator, input_image: Path, output_image: Path):
    image = cv2.imread(input_image.absolute().as_posix())
    preds = nn.predict(image)
    s_ok = cv2.imwrite(output_image.absolute().as_posix(), preds)
    if not s_ok:
        print(f"ERROR: Unable to save image to {output_image}")
        sys.exit(1)

def process_image_web():
    r = None
    if "image" in request.files:
        f = request.files["image"]
        f.save(f.filename)
        fp = Path(f.filename)
        process_image(s, fp, fp)
        r = send_file(fp.as_posix())
    else:
        r = make_response("No image provided", 422)
    return r
    

if __name__ == "__main__":
    app = Flask("Segmentator")
    app.add_url_rule(
        rule="/api/process_image", 
        endpoint="process_image_web",
        view_func=process_image_web,
        methods=["POST"]
    )
    app.run("", 3000)
