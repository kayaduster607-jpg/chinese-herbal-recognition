"""可复现的 MobileNetV2 训练入口，保留原实验的 0–1 图像缩放方式。"""
import argparse
import json
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='中草药图像分类：MobileNetV2 迁移学习')
    parser.add_argument('--data-dir', type=Path, required=True, help='类别文件夹的父目录')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'), help='输出目录；split 子目录必须不存在')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def split_dataset(source, target, seed):
    """每类随机划分 70/15/15；拒绝复用已有 split，防止混入旧数据。"""
    if target.exists():
        raise FileExistsError('输出划分目录已存在，请使用新的 --output-dir: {}'.format(target))
    classes = sorted(p for p in source.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if not classes:
        raise ValueError('未找到类别文件夹: {}'.format(source))
    rng = random.Random(seed)
    for folder in classes:
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        rng.shuffle(files)
        train_end, val_end = int(len(files) * 0.7), int(len(files) * 0.85)
        for split, chosen in [('train', files[:train_end]), ('val', files[train_end:val_end]), ('test', files[val_end:])]:
            destination = target / split / folder.name
            destination.mkdir(parents=True, exist_ok=True)
            for image in chosen:
                shutil.copy2(image, destination / image.name)


def generators(split, batch_size):
    """训练集增强；验证与测试只做与原实验相同的 rescale。"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    augmented = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.2,
        height_shift_range=0.2, horizontal_flip=True, zoom_range=0.2, brightness_range=[0.8, 1.2], fill_mode='nearest')
    standard = ImageDataGenerator(rescale=1./255)
    result = []
    for name in ('train', 'val', 'test'):
        generator = augmented if name == 'train' else standard
        result.append(generator.flow_from_directory(str(split / name), target_size=(224, 224),
            batch_size=batch_size, class_mode='categorical', shuffle=name == 'train'))
    return result


def build_model(number_of_classes):
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.optimizers import Adam
    backbone = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    backbone.trainable = False
    model = models.Sequential([backbone, layers.GlobalAveragePooling2D(), layers.Dense(128, activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.3), layers.Dense(number_of_classes, activation='softmax')])
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError('epochs 和 batch-size 必须为正整数')
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras import callbacks
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    split = args.output_dir / 'split'
    split_dataset(args.data_dir, split, args.seed)
    train, val, test = generators(split, args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'class_indices.json').write_text(json.dumps(train.class_indices, ensure_ascii=False, indent=2), encoding='utf-8')
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(train.classes), y=train.classes)
    model = build_model(train.num_classes)
    controls = [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)]
    history = model.fit(train, epochs=args.epochs, validation_data=val,
        class_weight=dict(enumerate(weights)), callbacks=controls)
    loss, accuracy = model.evaluate(test)
    model.save(args.output_dir / 'herbal_mobilenetv2.keras')
    pd.DataFrame(history.history).to_csv(args.output_dir / 'training_history.csv', index=False)
    (args.output_dir / 'evaluation.json').write_text(json.dumps({'test_loss': float(loss),
        'test_accuracy': float(accuracy), 'seed': args.seed, 'test_samples': test.samples}, indent=2), encoding='utf-8')
    print('测试集准确率: {:.4f}'.format(accuracy))


if __name__ == '__main__':
    main()
