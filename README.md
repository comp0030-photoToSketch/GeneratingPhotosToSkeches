Incomplete - 19:21 25/1/21

# Sketch to sketch implementation

Install pytorch from requirements .txt (not sure if we need any other packages yet) 

- pip install -r requirements.txt

If you have problems check out this thread

- https://stackoverflow.com/questions/60912744/install-pytorch-from-requirements-txt


#Goal

Encoder[Sketch] -> Learned features vector -> Sketch-RNN Decoder[Sketch] -> reproduced sketch

1. Find dataset for encoder
2. Experiment on different pytorch encoders
3. Find a way to pass to decoder
4. Check produced sketches look like what's expected

# Dataset

- datasets/quickdraw/train - 50 images
- datasets/quickdraw/test - 25 images
- datasets/quickdraw/val - 23 images

# Encoder Architectures:

1. Muna:
2. Jingze:
3, Lischen:
4, Duncheng: Here is a test

# Decoder Architecture:

SketchRNN, takes input in stroke-5 format, need to convert this to stroke-5
