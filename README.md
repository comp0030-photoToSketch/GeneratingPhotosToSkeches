# Sketch to sketch implementation

## Goal

Encoder[Sketch] -> Learned features vector -> Sketch-RNN Decoder[Sketch] -> reproduced sketch

1. Find dataset for encoder
2. Experiment on different pytorch encoders
3. Find a way to pass to decoder
4. Check produced sketches look like what's expected

## Dataset

- datasets/quickdraw/train - 50 images
- datasets/quickdraw/test - 25 images
- datasets/quickdraw/val - 23 images
- npz file containg quickdraw sketch based representations

## Architectures:

1. Muna: Handmade AE and VAE, Pytorch's VGG-16
2. Jingze:
3. Lischen:
4. Duncheng: Here is a test

