from exif import Image
from datetime import datetime
from picamera import PiCamera
import cv2
import math
import os


dir_path = os.path.dirname(os.path.realpath(__file__))

class speedImage:
    def __init__(self):
        self.data = {
            "image1": {
                "time": 0,
                "cv": None,
                "keypoints": None,
                "descriptors": None,
                "coordinates": None
            },
            "image2": {
                "time": 0,
                "cv": None,
                "keypoints": None,
                "descriptors": None,
                "coordinates": None
            },

            "timeDifference": 0,
            "matches": None,
            "speed": 0,
        }

    def speed(self, image1, image2, feature=1000, GSD=12648):
        def get_time(image):
            with open(image, 'rb') as imageFile:
                img = Image(imageFile)
                timeStr = img.get("datetime_original")
                time = datetime.strptime(timeStr, '%Y:%m:%d %H:%M:%S')
            return time

        # get time difference
        self.data["image1"]["time"] = get_time(image1)
        self.data["image2"]["time"] = get_time(image2)
        self.data["timeDifference"] = self.data["image2"]["time"] - self.data["image1"]["time"]
        self.data["timeDifference"] =  self.data["timeDifference"].seconds

        # convert to cv
        self.data["image1"]["cv"] = cv2.imread(image1, 0)
        self.data["image2"]["cv"] = cv2.imread(image2, 0)

        # calculate features
        orb = cv2.ORB_create(nfeatures = feature)
        self.data["image1"]["keypoints"], self.data["image1"]["descriptors"] = orb.detectAndCompute(self.data["image1"]["cv"], None)
        self.data["image2"]["keypoints"], self.data["image2"]["descriptors"] = orb.detectAndCompute(self.data["image2"]["cv"], None)

        # calculate matches
        bruteForce = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.data["matches"] = bruteForce.match(self.data["image1"]["descriptors"], self.data["image2"]["descriptors"])
        self.data["matches"] = sorted(self.data["matches"], key=lambda x: x.distance)

        # find matching coordinates
        self.data["image1"]["coordinates"] = []
        self.data["image2"]["coordinates"] = []
        for match in self.data["matches"]:
            image_1_idx = match.queryIdx
            image_2_idx = match.trainIdx
            (x1,y1) = self.data["image1"]["keypoints"][image_1_idx].pt
            (x2,y2) = self.data["image2"]["keypoints"][image_2_idx].pt
            self.data["image1"]["coordinates"].append((x1,y1))
            self.data["image2"]["coordinates"].append((x2,y2))

        # calculate mean distance
        allDistances = 0
        mergedCoordinates = list(zip(self.data["image1"]["coordinates"], self.data["image2"]["coordinates"]))
        for coordinate in mergedCoordinates:
            x_difference = coordinate[0][0] - coordinate[1][0]
            y_difference = coordinate[0][1] - coordinate[1][1]
            distance = math.hypot(x_difference, y_difference)
            allDistances = allDistances + distance
        featureDistance = allDistances / len(mergedCoordinates)

        # calculate speed in kmps
        self.data["speed"] = (featureDistance * GSD / 100000 ) / self.data["timeDifference"]
        return self.data["speed"] 

    def display_matches(self):
        # Utilisez self.data pour obtenir les données nécessaires
        match_img = cv2.drawMatches(self.data["image1"]["cv"], 
                                    self.data["image1"]["keypoints"], 
                                    self.data["image2"]["cv"], 
                                    self.data["image2"]["keypoints"], 
                                    self.data["matches"][:100], 
                                    None)
        
        resize = cv2.resize(match_img, (1600, 600), interpolation=cv2.INTER_AREA)
        cv2.imshow('matche', resize)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class pictureCamera:
    def __init__(self, resolutionHeight = 2592, resolutionLength = 1944):
        self.resolutionHeight = resolutionHeight
        self.resolutionLength = resolutionLength
        self.pictureNumber = 0

    def take(self, number):
        camera = PiCamera()
        camera.resolution = (self.resolutionHeight, self.resolutionLength)

        for _ in range(number):
            self.pictureNumber += 1
            camera.capture(f'{dir_path}/Picture/picture{self.pictureNumber:03d}.jpg')

        return self.pictureNumber

 

pictureCamera = pictureCamera()

for i in range(1):
    pictureNumber = pictureCamera.take(2)

    speedImageInstance = speedImage()
    print(speedImageInstance.speed(f"{dir_path}/Picture/picture{pictureNumber - 1:03d}.jpg", f"{dir_path}/Picture/picture{pictureNumber:03d}.jpg"))
    