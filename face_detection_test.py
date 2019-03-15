import dlib
import glob
import sys
import os
import pdb

detector = dlib.simple_object_detector("/Applications/dlib-master/tools/imglab/build/detector.svm")

f1 = "/Applications/CS 239/semi-supervised-CycleGAN-master/original/120_original.png"
img = dlib.load_rgb_image(f1)
dets = detector(img)
win = dlib.image_window()
win.clear_overlay()
win.set_image(img)
win.add_overlay(dets)

pdb.set_trace()
f1 = "/Applications/CS 239/semi-supervised-CycleGAN-master/transformed/121_transformed.png"
img = dlib.load_rgb_image(f1)
dets = detector(img)
win = dlib.image_window()
win.clear_overlay()
win.set_image(img)
win.add_overlay(dets)

pdb.set_trace()
original = "/Applications/CS 239/semi-supervised-CycleGAN-master/original/"
transformed = "/Applications/CS 239/semi-supervised-CycleGAN-master/transformed/"
o_list = os.listdir(original)
t_list = os.listdir(transformed)

o_list.sort()
t_list.sort()
inconsistent_behaviors = 0
for i in range(len(o_list)):
    f1 = o_list[i]
    img = dlib.load_rgb_image(original + f1)
    dets1 = detector(img)
    f2 = t_list[i]
    img = dlib.load_rgb_image(transformed + f2)
    dets2 = detector(img)
    if(len(dets1)!=len(dets2)):
        inconsistent_behaviors = inconsistent_behaviors + 1

print("Number of inconsistent behaviors: " + str(inconsistent_behaviors))