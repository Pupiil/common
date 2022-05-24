# USAGE
# When encoding on laptop, desktop, or GPU (slower, more accurate):
# python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method cnn
# When encoding on Raspberry Pi (faster, more accurate):
# python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method hog

# import the necessary packages
from imutils import paths
import face_recognition
import pathlib
import pickle
import cv2
import os

def encode():

    config = {
        "dataset": f"{pathlib.Path(__file__).parent.absolute()}/dataset/",
        "encodings": f"{pathlib.Path(__file__).parent.absolute()}/data/PREncodings.pkl",
        "detection_method": "cnn",
    }

    # grab the paths to the input images in our dataset
    print("[RECOGNITION::ENCODE_FACES::INFO] quantifying faces...")
    imagePaths = list(paths.list_images(config["dataset"]))

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[RECOGNITION::ENCODE_FACES::INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, model=config["detection_method"])

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    # dump the facial encodings + names to disk
    print("[RECOGNITION::ENCODE_FACES::INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(config["encodings"], "wb")
    f.write(pickle.dumps(data))
    f.close()

encode()
