from exif import Image as exifImage
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from datetime import datetime
from picamera import PiCamera
from logzero import logger
from pathlib import Path
from orbit import ISS
import numpy as np
import logzero
import math
import cv2


paths = Path(__file__).parent.resolve()
logzero.logfile(paths / "logFile.log")

class speed:
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

    def speedImage(self, image1, image2, feature=1000, GSD=12648):
        try:
            def getData(image):
                try:
                    with open(image, 'rb') as imageFile:
                        img = exifImage(imageFile)
                        timeStr = img.get("datetime_original")
                        time = datetime.strptime(timeStr, '%Y:%m:%d %H:%M:%S')
                    return time
                
                except Exception as error:
                    logger.exception(f"An error occurs when collecting the data from the image: {error}")
                    return None


            # get time difference
            self.data["image1"]["time"] = getData(image1)
            self.data["image2"]["time"] = getData(image2)
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

    def speedCoordinated(self, image1, image2):
        try:
            def getData(image):
                try:
                    with open(image, 'rb') as imageFile:
                        img = exifImage(imageFile)

                        timeStr = img.get("datetime_original")
                        time = datetime.strptime(timeStr, '%Y:%m:%d %H:%M:%S')

                        lat, lon = img.get("gps_latitude"), img.get("gps_longitude")
                        lat_ref, lon_ref = img.get("gps_latitude_ref"), img.get("gps_longitude_ref")

                        lat = lat[0] + (lat[1] / 60) + (lat[2] / 3600)
                        lon = lon[0] + (lon[1] / 60) + (lon[2] / 3600)

                        lat *= -1 if lat_ref != "N" else 1
                        lon *= -1 if lon_ref != "E" else 1

                    return time, lat, lon
                
                except Exception as error:
                    logger.exception(f"An error occurs when collecting the data from the image: {error}")
                    return None

            time1, lat1, lon1 = getData(image1)
            time2, lat2, lon2 = getData(image2)
            time = time2 - time1
            
            # formule de haversine
            a = math.sin((math.radians(lat2) - math.radians(lat1)) / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin((math.radians(lon2) - math.radians(lon1)) / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = 6371.0097714 * c

            # return speed
        
        except Exception as error:
            logger.exception(f"An error occurs when calculating speed with images: {error}")
            return 0


class pictureCamera:
    def __init__(self, resolutionHeight = 4056, resolutionLength = 3040):
        self.pictureNumber = 0
        self.camera = PiCamera()
        self.camera.resolution = (resolutionHeight, resolutionLength)

    def take(self, number):
        try:
            def convertDms(angle):
                sign, degrees, minutes, seconds = angle.signed_dms()
                return sign < 0, {'degrees': degrees, 'minutes': minutes, 'seconds': seconds}
            
            def convertDec(angle, direction):
                degrees, minutes, seconds = angle['degrees'], angle['minutes'], angle['seconds']
                Decimal = degrees + (minutes / 60) + (seconds / 3600)
                Decimal *= -1 if direction else 1
                return Decimal
            
            coordinated = []
            for _ in range(number):
                self.pictureNumber += 1

                iss = ISS()
                point = iss.coordinates()

                lat = point.latitude
                lon = point.longitude
                south, exifLatitude = convertDms(lat)
                west, exifLongitude = convertDms(lon)

                self.camera.exif_tags['GPS.GPSLatitude'] = "{:.0f}/1,{:.0f}/1,{:.0f}/10".format(exifLatitude['degrees'], exifLatitude['minutes'], exifLatitude['seconds']*10)
                self.camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
                self.camera.exif_tags['GPS.GPSLongitude'] = "{:.0f}/1,{:.0f}/1,{:.0f}/10".format(exifLongitude['degrees'], exifLongitude['minutes'], exifLongitude['seconds']*10)
                self.camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"

                self.camera.capture(f'{paths}/Picture/picture{self.pictureNumber:03d}.jpg')

                latitude = convertDec(exifLatitude, south)
                longitude = convertDec(exifLongitude, west)
                coordinated.append((latitude, longitude))

                logger.info(f"Taking the picture {self.pictureNumber:03d}")

        except Exception as error:
            logger.exception(f"An error occurred when taking the picture: {error}")
            return None
        
        return self.pictureNumber, coordinated

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


class statistic:
    def __init__(self, output):
        self.output = output

    def drawPointMap(self, coordinated):
        input = paths / "Resources" / "map.png"
        output = self.output / "stationsTracking.png"

        worldMap = Image.open(input)

        for ligne in coordinated:
            for paire in ligne:
                latitude, longitude = paire

                x = ((longitude * math.pi * 6378137 / 180) + (math.pi * 6378137)) / ((2 * math.pi * 6378137 / 256) / 8)
                y = (((math.log(math.tan((90 - latitude) * math.pi / 360)) / (math.pi / 180)) * math.pi * 6378137 / 180) + (math.pi * 6378137)) / ((2 * math.pi * 6378137 / 256) / 8)
                
                draw = ImageDraw.Draw(worldMap)
                pointSize = 3
                draw.ellipse([x - pointSize, y - pointSize, x + pointSize, y + pointSize], fill="#b52b10", outline="#760000")

        worldMap.save(output)

    def graphicSpeedPicture(self, data):
        plt.plot(data)
        plt.legend(['Speed with picture'], loc='lower right')
        plt.savefig(self.output / 'graphic_SpeedPicture.png')



if __name__ == "__main__":
    try:
        pictureCamera = pictureCamera()
        speed = speed()
        speedDataStorage = dataStorage(paths / 'result.txt')
        statistic = statistic(paths / "Statistic")
        
        speedPicture, speedCoordinated = [], []
        coordinated = []

        for i in range(15): 
            pictureNumber, pictureCoordinated = pictureCamera.take(2)
            # coordinated.append(pictureCoordinated)

            if pictureNumber != None:
                # speedPicture.append(speed.speedImage(paths / 'Picture' / f'picture{pictureNumber - 1:03d}.jpg', paths / 'Picture' / f'picture{pictureNumber:03d}.jpg'))
                speed.speedCoordinated(paths / 'Picture' / f'picture{pictureNumber - 1:03d}.jpg', paths / 'Picture' / f'picture{pictureNumber:03d}.jpg')
                # speedDataStorage.speedData("{:.4f}".format(np.mean(speedPicture)))
            
        # speedDataStorage.speedData("{:.4f}".format(np.mean(speedPicture)))
        # statistic.drawPointMap(coordinated)
        # statistic.graphicSpeedPicture(speedPicture)

    except Exception as error:
        logger.exception(f"an error occurs when the main function: {error}")

