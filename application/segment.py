from pathlib import Path
import cv2
import argparse
import os
import sys

from inference import Segmentator


def process_image(nn: Segmentator, input_image: Path, output_image: Path):
    image = cv2.imread(input_image.absolute().as_posix())
    preds = nn.predict(image)
    s_ok = cv2.imwrite(output_image.absolute().as_posix(), preds)
    if not s_ok:
        print(f"ERROR: Unable to save image to {output_image}")
        sys.exit(1)


def main():
    m = argparse.ArgumentParser()
    m.add_argument("--model", "-m", type=Path,
                   default='../model/best_model_3fold.pth',
                   help="Путь до модели ONNX"
                   )
    m.add_argument("--input-image", "-ii", type=Path,
                   help="Входное изображение",
                   required=False
                   )
    m.add_argument("--output-image", "-oi", type=Path,
                   help="Выходное изображение или выходная папка",
                   required=False
                   )

    m.add_argument("--input-dir", "-id", type=Path,
                   help="Входная папка",
                   required=False
                   )
    m.add_argument("--output-dir", "-od", type=Path,
                   help="Выходная папка",
                   required=False
                   )
    try:
        options = m.parse_args().__dict__
    except:
        sys.exit(1)

    NeuralNetwork = Segmentator(options['model'].absolute().as_posix())

    input_image = options.get("input_image")
    output_image = options.get("output_image")
    input_dir = options.get("input_dir")
    output_dir = options.get("output_dir")
    if input_image is not None and output_image is not None:
        input_image = Path(input_image)
        output_image = Path(output_image)
        if output_image.is_dir():
            output_image = output_image.joinpath(input_image.name)
        else:
            output_image.parents[0].mkdir(parents=True, exist_ok=True)

        process_image(NeuralNetwork, input_image, output_image)
    elif input_dir is not None and output_dir is not None:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        if not input_dir.is_dir() or not output_dir.is_dir():
            print("ERROR: Given input and output args are not directories")
            sys.exit(1)
        output_dir.mkdir(parents=True, exist_ok=True)
        files = os.listdir(input_dir)
        for i, file_str in enumerate(files):
            print("Progress {:6.2f}%".format((i + 1) / len(files) * 100), end="\r")
            input_image = input_dir.joinpath(file_str)
            output_image = output_dir.joinpath(input_image.name)
            process_image(NeuralNetwork, input_image, output_image)
        print()
    else:
        raise RuntimeError("Error")


if __name__ == "__main__":
    main()
