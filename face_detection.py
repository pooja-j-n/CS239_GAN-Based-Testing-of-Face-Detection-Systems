import dlib
import glob
import sys
import os

path = "/Applications/CS 239/semi-supervised-CycleGAN-master/data/train/0"
options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
options.C = 2
options.be_verbose = True
training_xml_path = "/Applications/CS 239/semi-supervised-CycleGAN-master/target.xml"
dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)


testing_xml_path = "/Applications/CS 239/semi-supervised-CycleGAN-master/testing_target.xml"

print("Training accuracy: {}".format(dlib.test_simple_object_detector(training_xml_path, "detector.svm")))

print("Testing accuracy: {}".format(dlib.test_simple_object_detector(testing_xml_path, "detector.svm")))

detector = dlib.simple_object_detector("detector.svm")