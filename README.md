# CS239_GAN-Based-Testing-of-Face-Detection-Systems

dataloader.py loads the male and female images. <br/>
model.py has the model architecture for generators and discriminators. <br/>
train.py can be used to train the model by passing commandline arguments. <br/>
test.py can be used to test the model by passing commandline arguments. <br/>
solver.py is called while training and it runs the training epochs. 

cycleGAN_train_test.ipynb uses the above files to train and test the cycleGAN.<br/>
Run the file by setting appropriate commandline arguments. <br/>


face_detection.py trains the face detection model on male images. <br/>
face_detection_test.py can be used to test the face detection model on original male images and GAN-generated female images. <br/>
