# Repository for COMP0031 Group Research Project

## Generating Sketches from Photos

### Abstract
We present a simple photo to stroke-sequence model, that can be constructed to form a sketch using conditional generation. The model has an Image Encoder that has been trained using cat and chair stroke-sequence data from the QuickDraw! dataset. The Image Encoder uses a convulutional neural network (CNN) and supervised learning with paired stroke-sequence data and the corresponding sketch images of that data. The decoder uses the existing sketch-rnn Decoder RNN architecture, but was trained alongside the new Image Encoder model. We test this model using the Sketchy dataset to show it works relatively well in the photo to sketch domain. Using this model we are able to prove that it is possible to generate sketches that can mislead a human into thinking they were machine generated.
 
### Important Files + Descriptions:

- photoToSketch_cat_sketchy_dataset.ipynb: Contains trained photo to sketch model on sketchy dataset cats

- photoToSketch_chair_sketchy_dataset.ipynb: Contains trained photo to sketch model on sketchy dataset chairs

- sketchToSketch_ImageEncoder_cat.ipynb: Contains code for supervised CNN Image Encoder.

- sketchToSketch_chair.ipynb: Contains code for supervised CNN Image Encoder, with file paths for chair dataset.

- sketchrnn_cat_interpolation.ipynb: Contains sketchRNN model and examples from Ha&Eck paper.

- sketchrnn_chair_trained.ipynb: Contains sketchRNN model trained on chairs

- sketchrnn_reimplementation.ipynb : Attempted sketchRNN implementation from scratch.

### Setup

1. create a virtual environment, see 'setup_ve' file for instructions
2. type conda activate env_pytorch into terminal to run your virtual environment
3. type jupyter notebook into terminal to open all notebooks
