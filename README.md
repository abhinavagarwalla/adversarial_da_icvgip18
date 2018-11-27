Source Code for the paper "Unsupervised Domain Adaptation for Learning Eye Gaze from a Million Synthetic Images: An Adversarial Approach"

# Data Generation

Use the scripts in the folder eye_gaze/dataio for processing data and generating TFRecords

Synthetic Images
----------------
1. Generate the synthetic images from the UnityEyes simulator
2. Run python gaze_cropper.py with applicable paths to process the raw images
3. Run python gaze_converter.py with applicable paths for generating TFRecords from processed images

Real Images
-----------
1. Download MPIIGaze dataset into a folder.
2. Run python real_gaze_converter.py with applicable path to generator TFRecords for real images


# Training Steps

Use the scripts in the folder eye_gaze/src for training and evaluating the models.

For training source model [pretraining the generator]
-----------------------------------------------------

1. Run python src/gaze_train_regressor.py from eye_gaze folder, with the input arguments set likewise.
2. gaze_train_regressor.py essentially calls gaze_model_regressor with the supplied arguments and starts the training process


For training target model [generator] and discriminator
-------------------------------------------

1. Run python src/gaze_da_train_regressor.py from eye_gaze folder, with the input arguments set likewise.
2. gaze_da_train_regressor.py essentially calls gaze_da_model_regressor with the supplied arguments and starts the training process

Evaluating the model
--------------------

1. Run python src/gaze_train_regressor.py with evaluate set to True, checkpoint_dir and checkpoint_file to point to the trained model