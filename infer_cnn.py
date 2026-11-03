"""加载公开的 150×150 CNN 权重并输出类别索引，不猜测类别名称。"""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='运行普通 CNN 基线推理')
    parser.add_argument('--model', type=Path,
                        default=Path('weights/chinese_medicine_classifier_simple.h5'),
                        help='CNN 模型路径，默认为 weights/chinese_medicine_classifier_simple.h5')
    parser.add_argument('--image', type=Path, required=True, help='输入图片路径')
    return parser.parse_args()


def predict(model_path, image_path):
    """按 150×150、RGB 和 0–1 缩放演示；原 CNN 预处理尚未核实。"""
    import numpy as np
    import tensorflow as tf
    from PIL import Image

    model = tf.keras.models.load_model(model_path, compile=False)
    with Image.open(image_path) as image:
        pixels = np.asarray(image.convert('RGB').resize((150, 150)), dtype=np.float32) / 255.0
    scores = model(np.expand_dims(pixels, axis=0), training=False).numpy()[0]
    indices = np.argsort(scores)[::-1][:5]
    return {
        'tensorflow': tf.__version__,
        'input_shape': list(model.input_shape[1:]),
        'output_dimensions': int(scores.shape[0]),
        'top5': [{'index': int(i), 'score': float(scores[i])} for i in indices],
    }


def main():
    args = parse_args()
    import json
    print(json.dumps(predict(args.model, args.image), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
