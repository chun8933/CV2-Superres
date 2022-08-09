'''
python cam.py
'''
# import the necessary packages
import argparse
import time
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default='./models/ESPCN_x4.pb',
                help="path to super resolution model")
args = vars(ap.parse_args())

modelName = args["model"].split("models/")[-1].split("_")[0].lower()
modelScale = args["model"].split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])

# initialize OpenCV's super resolution DNN object, load the super
# resolution model from disk, and set the model name and scale
print("[INFO] loading super resolution model: {}".format(
    args["model"]))
print("[INFO] model name: {}".format(modelName))
print("[INFO] model scale: {}".format(modelScale))
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(args["model"])
sr.setModel(modelName, modelScale)

cap = cv2.VideoCapture(0)
time.sleep(1.0)

# ready for process
success, image = cap.read()  # get first frame b4 loop start
print("[INFO] w: {}, h: {}".format(image.shape[1], image.shape[0]))
upscaled = sr.upsample(image)
cv2.imshow("Original", image)
cv2.imshow("Super Resolution", upscaled)

while success:
    upscaled = sr.upsample(image)
    cv2.imshow("Original", image)
    cv2.imshow("Super Resolution", upscaled)
    success, image = cap.read()  # Check Next and get the image

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cv2.waitKey(0)
