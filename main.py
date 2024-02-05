from exif import Image
from datetime import datetime
from picamera import PiCamera
from orbit import ISS
from logzero import logger
from pathlib import Path
import logzero
import cv2
import math
import numpy as np


paths = Path(__file__).parent.resolve()
logzero.logfile(paths / "logFile.log")

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
        try:
            def get_time(image):

                try:
                    with open(image, 'rb') as imageFile:
                        img = Image(imageFile)
                        timeStr = img.get("datetime_original")
                        time = datetime.strptime(timeStr, '%Y:%m:%d %H:%M:%S')
                    return time
                
                except Exception as error:
                    logger.exception(f"An error occurs when collecting the time from the image: {error}")
                    return None


            # get time difference
            self.data["image1"]["time"] = get_time(image1)
            self.data["image2"]["time"] = get_time(image2)
            self.data["timeDifference"] = self.data["image2"]["time"] - self.data["image1"]["time"]
            self.data["timeDifference"] =  self.data["timeDifference"].seconds
        
            # convert to cv
            self.data["image1"]["cv"] = cv2.imread(str(image1), 0)
            self.data["image2"]["cv"] = cv2.imread(str(image2), 0)

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

            logger.info(f"The calculation was made, speed: {self.data['speed']:.4f}") 
            return self.data["speed"]
        
        except Exception as error:
            logger.exception(f"An error occurs when calculating speed with images: {error}")
            return 0

    def displayMatches(self):
        try:
            match_img = cv2.drawMatches(self.data["image1"]["cv"], 
                                        self.data["image1"]["keypoints"], 
                                        self.data["image2"]["cv"], 
                                        self.data["image2"]["keypoints"], 
                                        self.data["matches"][:100], 
                                        None)
            
            resize = cv2.resize(match_img, (1600, 600), interpolation=cv2.INTER_AREA)
            cv2.imshow('matche', resize)
            logger.info("open display matches")

            cv2.waitKey(0)
            cv2.destroyAllWindows()
            logger.info("close display matches")

        except Exception as error:
            logger.exception(f"An error occurs when displaying the match: {error}")


class pictureCamera:
    def __init__(self, resolutionHeight = 4056, resolutionLength = 3040):
        self.pictureNumber = 0
        self.camera = PiCamera()
        self.camera.resolution = (resolutionHeight, resolutionLength)

    def take(self, number):
        try:
            def convert(angle):
                sign, degrees, minutes, seconds = angle.signed_dms()
                exif_angle = f'{degrees:.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
                
                return sign < 0, exif_angle


            for _ in range(number):
                self.pictureNumber += 1

                iss = ISS()
                point = iss.coordinates()

                south, exif_latitude = convert(point.latitude)
                west, exif_longitude = convert(point.longitude)

                self.camera.exif_tags['GPS.GPSLatitude'] = exif_latitude
                self.camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
                self.camera.exif_tags['GPS.GPSLongitude'] = exif_longitude
                self.camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"

                self.camera.capture(f'{paths}/Picture/picture{self.pictureNumber:03d}.jpg')

                logger.info(f"Taking the picture {self.pictureNumber:03d}")

        except Exception as error:
            logger.exception(f"An error occurred when taking the picture: {error}")
            return None
        
        return self.pictureNumber
    

class dataStorage:
    def __init__(self, file):
        self.file = file

    def speedData(self, data):
        try:
            with open(self.file, 'w') as file:
                file.write(data)

            logger.info("Speed data saved in the file")
            return True
        
        except Exception as error:
            logger.exception(f"an error occurs when saving data: {error}")
            return False




if __name__ == "__main__":
    try:
        pictureCamera = pictureCamera()
        speedImage = speedImage()
        speedDataStorage = dataStorage(paths / 'result.txt')
        speed = list(range(5))

        for i in range(5): 
            pictureNumber = pictureCamera.take(2)

            if pictureNumber != None:
                speed[i] = speedImage.speed(paths / 'Picture' / f'picture{pictureNumber - 1:03d}.jpg', paths / 'Picture' / f'picture{pictureNumber:03d}.jpg')
                speedDataStorage.speedData("{:.4f}".format(np.mean(speed)))
            

        speedDataStorage.speedData("{:.4f}".format(np.mean(speed)))

    except Exception as error:
        logger.exception(f"an error occurs when the main function: {error}")

