"""
Train the EffNet model
"""
import os
import argparse
import pandas as pd
import csv
from normalizer import Normalizer

from mobilenetv2 import MobileNetV2

import tensorflow as tf


def generate(args):
    """Data generation and augmentation

    # Arguments
        args: Dictionary, command line arguments
        
    # Returns
        train_generator: train set generator
        validation_generator: validation set generator
        num_training: Integer, number of images in the train split.
        num_validation: Integer, number of images in the validation split.
    """

    #  Using the data Augmentation in traning data

    normalizer = Normalizer()

    train_aug = tf.keras.preprocessing.image.ImageDataGenerator(
        #rescale=1. / 255.,
        shear_range=args.shear_range,
        zoom_range=args.zoom_range,
        rotation_range=args.rotation_range,
        width_shift_range=args.width_shift_range,
        height_shift_range=args.height_shift_range,
        horizontal_flip=args.horizontal_flip,
        vertical_flip=args.vertical_flip,
        preprocessing_function=normalizer)


    validation_aug = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=normalizer)

    train_generator = train_aug.flow_from_directory(
        args.train_dir,
        target_size=(args.input_size, args.input_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=True)

    mean, std = [], []
    if args.mean is None or args.std is None:
        mean, std = normalizer.get_stats(args.train_dir, train_generator.filenames, (args.input_size, args.input_size))
    else:
        mean = [float(m.strip()) for m in args.mean.split(',')]
        std = [float(s.strip()) for s in args.std.split(',')]
        normalizer.set_stats(mean, std)

    if not os.path.exists('model'):
        os.makedirs('model')
    with open('model/stats.txt', 'w') as stats:
        stats.write("Dataset mean [r, g, b] = {}\n".format(mean))


    label_map = (train_generator.class_indices)
    label_map = dict((v,k) for k,v in label_map.items())

    with open('model/labels.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, lineterminator='\n')
        csv_writer.writerows(label_map.items())

    validation_generator = validation_aug.flow_from_directory(
        args.validation_dir,
        target_size=(args.input_size, args.input_size),
        batch_size=args.batch_size,
        class_mode='categorical')

    return train_generator, validation_generator, train_generator.samples, validation_generator.samples, len(label_map)



def train(args):
    """Train the model.

    # Arguments
        args: Dictionary, command line arguments."""
        

    train_generator, validation_generator, num_training, num_validation, num_classes = generate(args)
    print("{} classes found".format(num_classes))

    model = MobileNetV2((args.input_size, args.input_size, 3), num_classes, args.plot_model)

    opt = tf.keras.optimizers.Adam()
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=30, verbose=1, mode='auto')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

    hist = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=num_training // args.batch_size,
        validation_steps=num_validation // args.batch_size,
        epochs=args.epochs,
        callbacks=[earlystop])

    if not os.path.exists('model'):
        os.makedirs('model')

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/hist.csv', encoding='utf-8', index=False)
    if not os.path.exists('model/output'):
        os.makedirs('model/output')
    model.save('model/output')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required arguments.
    parser.add_argument(
        "-t",
        "--train_dir",
        required=True,
        help="Path to directory containing training images")
    parser.add_argument(
            "-v",
            "--validation_dir",
            required=True,
            help="Path to directory containing validation images")
    # Optional arguments.
    parser.add_argument(
        "-s",
        "--input_size",
        type=int,
        default=224,
        help="Input image size.")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="Number of images in a training batch.")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.")
    parser.add_argument(
            "-p",
            "--plot_model",
            type=bool,
            default=False)
    parser.add_argument(
        "--shear_range",
        type=float,
        default=0.2,
        help="Shear range value for data augmentation.")
    parser.add_argument(
        "--zoom_range",
        type=float,
        default=0.2,
        help="Zoom range value for data augmentation.")
    parser.add_argument(
        "--rotation_range",
        type=int,
        default=90,
        help="Rotation range value for data augmentation.")
    parser.add_argument(
        "--width_shift_range",
        type=float,
        default=0.2,
        help="Width shift range value for data augmentation.")
    parser.add_argument(
        "--height_shift_range",
        type=float,
        default=0.2,
        help="Height shift range value for data augmentation.")
    parser.add_argument(
        "--horizontal_flip",
        type=bool,
        default=True,
        help="Whether or not to flip horizontally for data augmentation.")
    parser.add_argument(
        "--vertical_flip",
        type=bool,
        default=False,
        help="Whether or not to flip vertically for data augmentation.")
    parser.add_argument(
            "--mean",
            default=None,
            help="Dataset mean values for r, g, b values separates by commas")
    parser.add_argument(
            "--std",
            default=None,
            help="Dataset std values for r, g, b values separates by commas")
        
    args = parser.parse_args()
    train(args)
