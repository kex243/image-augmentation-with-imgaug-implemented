import os, glob
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps
import random

#Number_of_augmentation_to_use = 2
Number_of_augmentation_to_use = random.randrange(1,3,1)
addmax = -150
addmin = 150

#creates folders if dont exist
try:
    os.mkdir(str(os.getcwd()) + "\\edited")
except OSError:
    print ("Creation of the directory failed")
else:
    print ("Successfully created the directory")
    
try:
    os.mkdir(str(os.getcwd()) + "\\images")
except OSError:
    print ("Creation of the directory failed")
else:
    print ("Successfully created the directory")
	
cwd = os.getcwd()
path, dirs, files = next( os.walk( cwd + "\images" ) )
for count in range(50) :
    for i in files :
        path_to_image = cwd + '\images' + "\\" + i
        output_path = cwd + '\edited' + "\\" + i
        print (path_to_image)
        A = Image.open(path_to_image)
        seq = iaa.SomeOf(Number_of_augmentation_to_use,[
            
                        #You need to change only this part of the code- comment something or change parameters. Full docks are here https://imgaug.readthedocs.io/en/latest/source/overview_of_augmenters.html
            
            iaa.WithChannels(0, iaa.Add((addmin, addmax))),#this 3 blocks add some color to exact channel (0,1 or 2)
            iaa.WithChannels(1, iaa.Add((addmin, addmax))),
            iaa.WithChannels(2, iaa.Add((addmin, addmax))),
            
            iaa.Add((-40, 40)), #for all the channels
            
            iaa.Multiply((0.5, 1.5), per_channel=0.5), #similar to previous but with bigger effect
            
            iaa.Dropout2d(p=0.5), #big dropping out information from channels, even bigger effect than previous
            iaa.ChannelShuffle(0.35, channels=[0, 1, 2]), #this one has a chance to shuffle channels of image
            
            iaa.Invert(0.25, per_channel=0.5), #inverts colors
            
            iaa.BlendAlphaElementwise(0.5, iaa.Grayscale(1.0)), #very interesting effect for making planets less colorous with light inclusions
            
            #iaa.BlendAlphaHorizontalLinearGradient(iaa.AddToHue((-100, 100))), #similar one but from the bottom only. This one and one below can breack symmetry of the left and right parts of images.
            #iaa.BlendAlphaVerticalLinearGradient(iaa.AddToHue((-100, 100))),
            
            iaa.GaussianBlur(sigma=(0.0, 3.0)), #I don't think that blur is needed.
            
            iaa.WithBrightnessChannels(iaa.Add((-50, 50))), #brightness
            iaa.RemoveSaturation(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.ChangeColorTemperature((1100, 10000)), #temperature
            iaa.GammaContrast((0.5, 2.0)), #basick gamma
            iaa.GammaContrast((0.5, 2.0), per_channel=True), #contrast
            iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True),
            iaa.LogContrast(gain=(0.6, 1.4)), #additional contrast
            iaa.LogContrast(gain=(0.6, 1.4), per_channel=True), #and more- per channel
            iaa.LinearContrast((0.4, 1.6)), #more falttening
            
            iaa.Fliplr(0.5), #flips image
            iaa.Flipud(0.5), #flips image
            

        ])
        open_cvA = cv2.cvtColor(np.array(A), cv2.COLOR_RGB2BGR)
        open_cvA = seq(image=open_cvA)
        open_cvA = cv2.cvtColor(open_cvA, cv2.COLOR_BGR2RGB)
        A = Image.fromarray(open_cvA)
        A.save(output_path[:-4] + str(count) + ".png")
    # hope no one will ever see this code...
