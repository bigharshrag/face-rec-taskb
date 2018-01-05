The webapp is available at https://face-rec-taskb.herokuapp.com

The face detection is done using openCV library. For face recognition I have fine tuned a CNN pretrained on VGG Face. 

The saved model is saved_model.h5 file and the file used for training is keras_ft.py.
The dataset used was scrapped from google images and stored in dataset directory. For training I cropped out the faces from the images and are stored in the cropped_data directory. This is split into a test and train set.
