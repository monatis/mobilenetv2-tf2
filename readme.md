# mobilenetv2-tf2
A TensorFlow 2.0 implementation of [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381), aka MobileNetV2.

## Motivation

MobileNetV2  is still one of the most efficient architectures for image classification. Considering that TensorFlow 2.0 has already hit version beta1, I think that a flexible and reusable implementation of MobileNetV2 in TF 2.0 might be useful for practitioners.

## Implementation

I  implemented a running mean and standard deviation calculation with [Welford algorithm](https://www.johndcook.com/blog/standard_deviation/), which eliminates the problem of loading the whole dataset into the memory. `Normalizer` class, calculating the mean and standard deviation, is also used as a `preprocessing_function` argument to `tf.keras.preprocessing.image.ImageDataGenerator`.

## Install

1. `conda create -n mobilenetv2 python=3.6.8`
2. `conda activate mobilenetv2`
3. `git clone https://github.com/monatis/mobilenetv2-tf2.git`
4. `cd mobilenetv2-tf2`
5. `python -m pip install -r requirements.gpu.txt` # Change to `requirements.cpu.txt` if you're not using GPU.

## Usage

`train_dir` and `validation_dir` directories should contain a subdirectory for each class in the dataset. Then run:

- `python train.py --train_dir /path/to/training/images --validation_dir /path/to/validation/images`
- See `model/` directory for training output.

run `python train.py --help` to see all the options.

## Roadmap

- [x] Share model architecture and a training script.
- [x] Implement export to saved model.
- [x] Implement command line arguments to configure data augmentation.
- [ ] Share an inference script.
- [x] Implement mean and STD normalization.
- [ ] Implement confusion matrix.
- [ ] Implement export to TFLite for model inference.
- [ ] Share an example Android app using the exported TFLite model.

## License

MIT