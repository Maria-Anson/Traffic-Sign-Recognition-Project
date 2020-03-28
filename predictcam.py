import cv2
import os
os.system('python camera.py')
# USAGE
# python predict.py --model output/trafficsignnet.model --images gtsrb-german-traffic-sign/Test --examples examples

# import the necessary packages
from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from imutils import paths
from skimage import io
import numpy as np
import argparse
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg///
'''
import imutils
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained traffic sign recognizer")
args = vars(ap.parse_args())

# load the traffic sign recognizer model
print("[INFO] loading model...")
model = load_model(args["model"])

# load the label names
labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

# grab the paths to the input images, shuffle them, and grab a sample
print("[INFO] predicting...")

# loop over the image paths

	# load the image, resize it to 32x32 pixels, and then apply
	# Contrast Limited Adaptive Histogram Equalization (CLAHE),
	# just like we did during training
image = cv2.imread(r'D:\main\My works\Project\Traffic\traffic-sign-recognition\Cam\saved_img.jpg')
#plt.imshow(mpimg.imread(r'D:\main\My works\Project\Traffic\traffic-sign-recognition\Cam\saved_img.jpg'))
a=image.shape
b=image.shape[0]
image = cv2.resize(image, (32, 32))
image = exposure.equalize_adapthist(image, clip_limit=0.1)

	# preprocess the image by scaling it to the range [0, 1]
image = image.astype("float32") / 255.0
image = np.expand_dims(image, axis=0)

	# make predictions using the traffic sign recognizer CNN
preds = model.predict(image)
j = preds.argmax(axis=1)[0]
label = labelNames[j]

	# load the image using OpenCV, resize it, and draw the label
	# on it
dim=(b,128)    
image = cv2.imread(r'D:\main\My works\Project\Traffic\traffic-sign-recognition\Cam\saved_img.jpg')

cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,0.45, (0, 0, 255), 2)

	# save the image to disk
cv2.imwrite(r'D:\main\My works\Project\Traffic\traffic-sign-recognition\Cam\Output\final.jpg',image)
image=cv2.imread(r"D:\main\My works\Project\Traffic\traffic-sign-recognition\Cam\Output\final.jpg")
cv2.namedWindow("Output Window",cv2.WINDOW_NORMAL)
cv2.imshow("Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
