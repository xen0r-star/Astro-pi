from datetime import datetime, timedelta
from exif import Image as exifImage
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
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


class checking:
    """
    Verifie l'existence des dossiers et des fichiers necessaires.
    """
    def folder(self):
        pictureFolder = Path(paths / "Picture")
        statisticFolder = Path(paths / "Statistic")

        if not pictureFolder.is_dir():
            pictureFolder.mkdir(parents=True, exist_ok=True)
        
        if not statisticFolder.is_dir():
            statisticFolder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Folder check is complete") 
    
    def file(self):
        logFile = Path(paths / "logFile.log")
        resultFile = Path(paths / "result.txt")

        if not logFile.is_file():
            logFile.touch()

        if not resultFile.is_file():
            resultFile.touch()

        logger.info(f"File check is complete") 
    
    def mapFile(self):
        mapFile = Path(paths / "Resources" / "map.png")

        if not mapFile.is_file():
            mapFile = False
        else:
            mapFile = True

        logger.info(f"Map file check is complete") 

        return mapFile


class speed:
    """
    Effectue des calculs de vitesse a partir d'images ou de coordonnees GPS.
    """
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


            # obtenir la difference d'heure
            time1 = getData(image1)
            time2 = getData(image2)

            if None in [time1, time2]:
                logger.error("Time data is not available in one or both images")
                return None
            
            timeDifference = (time2 - time1).seconds
        
            # convertir en cv
            imageCv1 = cv2.imread(str(image1), 0)
            imageCv2 = cv2.imread(str(image2), 0)

            # calculer les caracteristiques
            orb = cv2.ORB_create(nfeatures = feature)
            keypoints1, descriptors1 = orb.detectAndCompute(imageCv1, None)
            keypoints2, descriptors2 = orb.detectAndCompute(imageCv2, None)

            # calculer les correspondances
            bruteForce = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bruteForce.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)

            # trouver les coordonnees correspondantes
            coordinates1 = []
            coordinates2 = []
            for match in matches:
                image1Idx = match.queryIdx
                image2Idx = match.trainIdx
                (x1,y1) = keypoints1[image1Idx].pt
                (x2,y2) = keypoints2[image2Idx].pt
                coordinates1.append((x1,y1))
                coordinates2.append((x2,y2))

            # calculer la distance moyenne
            allDistances = 0
            mergedCoordinates = list(zip(coordinates1, coordinates2))
            for coordinate in mergedCoordinates:
                x_difference = coordinate[0][0] - coordinate[1][0]
                y_difference = coordinate[0][1] - coordinate[1][1]
                distance = math.hypot(x_difference, y_difference)
                allDistances = allDistances + distance
            featureDistance = allDistances / len(mergedCoordinates)

            # calculer la vitesse en kmps
            speed = (featureDistance * GSD / 100000 ) / timeDifference

            logger.info(f"The calculation with the images has finished, speed: {speed:.4f}") 
            return speed
        
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

            if None in [time1, lat1, lon1, time2, lat2, lon2]:
                logger.error("GPS data is not available in one or both images")
                return None

            timeDifference = (time2 - time1).seconds
            
            # formule de haversine
            a = math.sin((math.radians(lat2) - math.radians(lat1)) / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin((math.radians(lon2) - math.radians(lon1)) / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = 6371.0097714 * c

            speed = ((distance * 1000) / timeDifference) / 1000

            logger.info(f"The calculation with the coordinated values has finished, speed: {speed:.4f}") 
            return speed
        
        except Exception as error:
            logger.exception(f"An error occurs when calculating speed with coordinated: {error}")
            return 0



class pictureCamera:
    """
    Capture des images avec la camera et enregistre les coordonnees GPS dans les donnees Exif.
    """
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
    """
    Gere le stockage des donnees dans un fichier.
    """
    def __init__(self, file):
        self.file = file

    def speedData(self, data):
        try:
            with open(self.file, 'w') as file:
                file.write(data)
                file.close()

            logger.info("Speed data saved in the file")
            return True
        
        except Exception as error:
            logger.exception(f"an error occurs when saving data: {error}")
            return False



class statistic:
    """
    Genere des statistiques et des graphiques a partir des donnees.
    """
    def __init__(self, output):
        self.output = output

    def drawPointMap(self, coordinated):
        try:
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
            logger.info(f"Map tracking is complete") 

        except Exception as error:
            logger.exception(f"an error occurs when drawing the tracking: {error}")

    def graphicSpeedPicture(self, data1_1, data1_2, data2, data3):
        try:
            plt.plot(data1_1)
            plt.plot(data1_2)
            plt.plot(data2)
            plt.plot(data3)

            plt.legend(['Speed with picture', 'Speed with picture cleaned', 'Speed with coordinated', 'Average speed'], loc='upper left')
            plt.savefig(self.output / 'graphic_SpeedPicture.png')

            logger.info(f"The speed graph is complete") 

        except Exception as error:
            logger.exception(f"an error occurs when creating the speed graph: {error}")

    def outlier(self, data):
        try:
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)

            IQR = Q3 - Q1

            lowLimit = Q1 - 1.5 * IQR
            upperLimit = Q3 + 1.5 * IQR

            outliersIndices = np.where((data < lowLimit) | (data > upperLimit))
            print(lowLimit, upperLimit)

            # dataCleaned = [value for value in data]
            # for index in outliersIndices[0]:
            #     dataCleaned[index] = None

            # print(dataCleaned)

            return dataCleaned
                
        except Exception as error:
            logger.exception(f"an error occurs with outliers: {error}")



if __name__ == "__main__":
    try:
        # Debut du chronometre
        startTime = datetime.now()

        # Verification des dossiers et fichiers necessaires
        checking = checking()

        checking.folder()
        checking.file()
        mapFile = checking.mapFile()

        # Initialisation des objets pour la capture d'images, le calcul de vitesse, le stockage de donnees et les statistiques
        pictureCamera = pictureCamera()
        speed = speed()
        speedDataStorage = dataStorage(paths / 'result.txt')
        statistic = statistic(paths / "Statistic")
        
        # Listes pour stocker les donnees de vitesse et les coordonnees GPS
        speedPicture, speedCoordinated, speedAverage = [], [], []
        coordinated = []

        # Temps actuel
        nowTime = datetime.now()

        # Boucle pour capturer les images et calculer la vitesse
        while (nowTime < startTime + (timedelta(minutes=10) - timedelta(seconds=45))): 
            pictureNumber, pictureCoordinated = pictureCamera.take(2)
            coordinated.append(pictureCoordinated)

            if pictureNumber != None:
                speedPicture.append(speed.speedImage(paths / 'Picture' / f'picture{pictureNumber - 1:03d}.jpg', paths / 'Picture' / f'picture{pictureNumber:03d}.jpg'))
                speedCoordinated.append(speed.speedCoordinated(paths / 'Picture' / f'picture{pictureNumber - 1:03d}.jpg', paths / 'Picture' / f'picture{pictureNumber:03d}.jpg'))

            # speedPictureCleaned = statistic.outlier(speedPicture)
            # speedAverage.append(speedPictureCleaned[i])
            speedAverage.append((speedPicture[-1] + speedCoordinated[-1]) / 2)

            # Sauvegarde des valeurs
            speedDataStorage.speedData("{:.4f}".format(np.mean(speedAverage)))

            # Temps actuel
            nowTime = datetime.now()
        
        # Enregistrement des donnees de vitesse moyenne
        speedDataStorage.speedData("{:.4f}".format(np.mean(speedAverage)))

        # Creation de la carte des points et du graphique de vitesse
        statistic.graphicSpeedPicture(speedPicture, speedPicture, speedCoordinated, speedAverage)
        if mapFile == True:
            statistic.drawPointMap(coordinated)

        endTime = datetime.now()
        logger.info(f"running time {endTime - startTime}")

    except Exception as error:
        logger.exception(f"an error occurs when the main function: {error}")

