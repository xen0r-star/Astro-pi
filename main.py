# Equipe: Astro Elite
# Professeur : Laila Bouteglifine
# Etudiants : Florian Berte, Thibaut Dudart, Rafaël Ravry
# Ecole : Institut Saint-François de Sales
# Ville : Ath
# Pays : Belgique

import cv2
from datetime import datetime, timedelta
from exif import Image as exifImage
from logzero import logger
import logzero
import math
import matplotlib.pyplot as plt
import numpy as np
from orbit import ISS
import pandas as pd
from pathlib import Path
from picamera import PiCamera
from PIL import Image, ImageDraw
from sense_hat import SenseHat


paths = Path(__file__).parent.resolve()
logzero.logfile(paths / "logFile.log")


class checking:
    """
    Verifie l'existence des dossiers et des fichiers necessaires.
    """
    def folder(self):
        pictureFolder = Path(paths / "Picture")
        statisticFolder = Path(paths / "Statistic")
        dataFolder = Path(paths / "Data")

        if not pictureFolder.is_dir():
            pictureFolder.mkdir(parents=True, exist_ok=True)
        
        if not statisticFolder.is_dir():
            statisticFolder.mkdir(parents=True, exist_ok=True)

        if not dataFolder.is_dir():
            dataFolder.mkdir(parents=True, exist_ok=True)
        
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
    def speedPicture(self, image1, image2, feature=1000, GSD=12648):
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
                logger.exception("Time data is not available in one or both images")
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
                logger.exception("GPS data is not available in one or both images")
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

            return self.pictureNumber, coordinated
        
        except Exception as error:
            logger.exception(f"An error occurred when taking the picture: {error}")
            return None



class dataStorage:
    """
    Gere le stockage des donnees dans un fichier.
    """
    def dataFile(self, data, file):
        try:
            with open(file, 'w') as file:
                file.write(data)
                file.close()

            logger.info("Data saved in the file")
            return True
        
        except Exception as error:
            logger.exception(f"An error occurs when saving data in the file: {error}")
            return False
        
    def speedDataFrame(self, speedPicture, speedPictureCleaned, speedCoordinated, speedAverage, loopTime, file):
        try:
            if Path(file).is_file():
                df = pd.read_csv(file)
            else:
                df = pd.DataFrame(columns=["Speed Picture", "Speed Picture Cleaned", "Speed Coordinated", "Speed Average", "Loop Time"])

            Data = [
                [speedPicture, speedPictureCleaned, speedCoordinated, speedAverage, loopTime]
            ]

            newData = pd.DataFrame(Data, columns=["Speed Picture", "Speed Picture Cleaned", "Speed Coordinated", "Speed Average", "Loop Time"])
            df = pd.concat([df, newData], ignore_index=True)

            df.to_csv(file, index=False)

            logger.info("Speed data saved in the csv")
            return True
        
        except Exception as error:
            logger.exception(f"An error occurs when saving speed data in the csv: {error}")
            return False
        
    def coordinatedDataFrame(self, coordinated, file):
        try:
            if Path(file).is_file():
                df = pd.read_csv(file)
            else:
                df = pd.DataFrame(columns=["Latitude", "longitude"])

            Data = [
                [coordinated[0][0], coordinated[0][1]],
                [coordinated[1][0], coordinated[1][1]]
            ]

            newData = pd.DataFrame(Data, columns=["Latitude", "longitude"])
            df = pd.concat([df, newData], ignore_index=True)

            df.to_csv(file, index=False)

            logger.info("Coordinated data saved in the csv")
            return True
        
        except Exception as error:
            logger.exception(f"An error occurs when saving coordinated data in the csv: {error}")
            return False
        
    def environmentDataFrame(self, humidity, temperature, pressure, file):
        try:
            if Path(file).is_file():
                df = pd.read_csv(file)
            else:
                df = pd.DataFrame(columns=["Humidity", "Temperature", "Pressure"])

            Data = [
                [humidity, temperature, pressure]
            ]

            newData = pd.DataFrame(Data, columns=["Humidity", "Temperature", "Pressure"])
            df = pd.concat([df, newData], ignore_index=True)

            df.to_csv(file, index=False)

            logger.info("Environment data saved in the csv")
            return True
        
        except Exception as error:
            logger.exception(f"An error occurs when saving Environment data in the csv: {error}")
            return False
        
    def IMUDataFrame(self, gyroscope, accelerometer, magnetometer, file):
        try:
            if Path(file).is_file():
                df = pd.read_csv(file)
            else:
                df = pd.DataFrame(columns=["Gyroscope X", "Gyroscope Y", "Gyroscope Z", "Accelerometer X", "Accelerometer Y", "Accelerometer Z", "Magnetometer X", "Magnetometer Y", "Magnetometer Z"])

            Data = [
                [
                    gyroscope["x"], gyroscope["y"], gyroscope["z"],
                    accelerometer["x"], accelerometer["y"], accelerometer["z"],
                    magnetometer["x"], magnetometer["y"], magnetometer["z"]
                ]
            ]

            newData = pd.DataFrame(Data, columns=["Gyroscope X", "Gyroscope Y", "Gyroscope Z", "Accelerometer Roll", "Accelerometer Pitch", "Accelerometer Yaw", "Magnetometer X", "Magnetometer Y", "Magnetometer Z"])
            df = pd.concat([df, newData], ignore_index=True)

            df.to_csv(file, index=False)

            logger.info("IMU data saved in the csv")
            return True
        
        except Exception as error:
            logger.exception(f"An error occurs when saving IMU data in the csv: {error}")
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
            logger.exception(f"An error occurs when drawing the tracking: {error}")

    def graphicSpeedPicture(self, speedPicture, speedPictureCleaned, speedCoordinated, speedAverage):
        try:
            plt.clf()

            x = list(range(max(len(speedPicture), len(speedPictureCleaned), len(speedCoordinated), len(speedAverage))))

            plt.plot(x, speedPicture, marker='.', label='Speed with picture')
            plt.plot(x, speedPictureCleaned, marker='.', label='Speed with picture cleaned')
            plt.plot(x, speedCoordinated, marker='.', label='Speed with coordinated')
            plt.plot(x, speedAverage, marker='.', label='Average speed')

            plt.legend(loc='upper left')
            plt.savefig(self.output / 'graphic_SpeedPicture.png')

            logger.info(f"The speed graph is complete") 

        except Exception as error:
            logger.exception(f"An error occurs when creating the speed graph: {error}")

    def graphicTime(self, time):
        try:
            plt.clf()

            x = list(range(len(time)))
            plt.plot(x, time, marker='.', label='Time per iteration')

            plt.legend(loc='upper left')
            plt.savefig(self.output / 'graphic_Time.png')

            logger.info(f"The time graph is complete") 

        except Exception as error:
            logger.exception(f"An error occurs when creating the time graph: {error}")    
    
    def graphicHumidity(self, humidity):
        try:
            plt.clf()

            x = list(range(len(humidity)))
            plt.plot(x, humidity, marker='.', label='Humidity')

            plt.legend(loc='upper left')
            plt.savefig(self.output / 'graphic_Humidity.png')

            logger.info(f"The humidity graph is complete") 

        except Exception as error:
            logger.exception(f"An error occurs when creating the humidity graph: {error}")

    def graphicTemperature(self, temperature):
        try:
            plt.clf()

            x = list(range(len(temperature)))
            plt.plot(x, temperature, marker='.', label='Temperature')

            plt.legend(loc='upper left')
            plt.savefig(self.output / 'graphic_Temperature.png')

            logger.info(f"The temperature graph is complete") 

        except Exception as error:
            logger.exception(f"An error occurs when creating the temperature graph: {error}")

    def graphicPressure(self, pressure):
        try:
            plt.clf()

            x = list(range(len(pressure)))
            plt.plot(x, pressure, marker='.', label='Pressure')

            plt.legend(loc='upper left')
            plt.savefig(self.output / 'graphic_Pressure.png')

            logger.info(f"The pressure graph is complete") 

        except Exception as error:
            logger.exception(f"An error occurs when creating the pressure graph: {error}")

    def outlier(self, data, dataCleaned = []):
        try:
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)

            IQR = Q3 - Q1

            lowLimit = Q1 - 1.5 * IQR
            upperLimit = Q3 + 1.5 * IQR

            outliersIndices = np.where((data < lowLimit) | (data > upperLimit))

            dataCopy = [value for value in data]
            for index in outliersIndices[0]:
                dataCopy[index] = None

            dataCleaned.append(dataCopy[-1])

            return dataCleaned
                
        except Exception as error:
            logger.exception(f"An error occurs with outliers: {error}")



class SenseHatSensor:
    """
    Utilise les capteurs de Sense HAT
    """
    def __init__(self):
        self.sense = SenseHat()

    def gyroscope(self):
        try:
            gyroData = self.sense.get_gyroscope_raw()
            return {
                "x": gyroData["x"],
                "y": gyroData["y"],
                "z": gyroData["z"]
            }
        
        except Exception as error:
            print(f"An error occurs with gyroscope: {error}")
            return None

    def accelerometer(self):
        try:
            accelData = self.sense.get_accelerometer_raw()
            return {
                "x": accelData["x"],
                "y": accelData["y"],
                "z": accelData["z"]
            }
    
        except Exception as error:
            print(f"An error occurs with accelerometer: {error}")
            return None

    def magnetometer(self):
        try:
            magData = self.sense.get_compass_raw()
            return {
                "x": magData["x"],
                "y": magData["y"],
                "z": magData["z"]
            }
    
        except Exception as error:
            print(f"An error occurs with magnetometer: {error}")
            return None
    
    def humidity(self):
        try:
            humidity = self.sense.get_humidity()
            return humidity
    
        except Exception as error:
            print(f"An error occurs with humidity sensor: {error}")
            return None
    
    def temperature(self):
        try:
            temp = self.sense.get_temperature()
            return temp
    
        except Exception as error:
            print(f"An error occurs with temperature sensor: {error}")
            return None
    
    def pressure(self):
        try:
            pressure = self.sense.get_pressure()
            return pressure
    
        except Exception as error:
            print(f"An error occurs with pressure sensor: {error}")
            return None




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
        dataStorage = dataStorage()
        statistic = statistic(paths / "Statistic")
        SenseHatSensor = SenseHatSensor()
        
        # Listes pour stocker les informations
        speedPicture, speedPictureCleaned, speedCoordinated, speedAverage = [], [], [], []
        humidity, temperature, pressure = [], [], [] 
        gyroscope, accelerometer, magnetometer = [], [], []
        coordinated = []
        loopTime = []
        pictureNumber = 0

        logger.info(f"Start of loop")

        # Temps actuel
        nowTime = datetime.now()

        # Boucle pour capturer les images et calculer la vitesse
        while ((nowTime < startTime + timedelta(minutes=9)) and (pictureNumber < 42)):
            startLoopTime = datetime.now()

            pictureNumber, pictureCoordinated = pictureCamera.take(2)
            coordinated.append(pictureCoordinated)

            if pictureNumber != None:
                speedPicture.append(speed.speedPicture(paths / 'Picture' / f'picture{pictureNumber - 1:03d}.jpg', paths / 'Picture' / f'picture{pictureNumber:03d}.jpg'))
                speedCoordinated.append(speed.speedCoordinated(paths / 'Picture' / f'picture{pictureNumber - 1:03d}.jpg', paths / 'Picture' / f'picture{pictureNumber:03d}.jpg'))

            speedPictureCleaned = statistic.outlier(speedPicture, speedPictureCleaned)
            speedAverage.append(speedCoordinated[-1] if speedPictureCleaned[-1] is None else (speedPictureCleaned[-1] + speedCoordinated[-1]) / 2)

            # Sense Hat capteur
            gyroscope.append(SenseHatSensor.gyroscope())
            accelerometer.append(SenseHatSensor.accelerometer())
            magnetometer.append(SenseHatSensor.magnetometer())

            humidity.append(SenseHatSensor.humidity()) 
            temperature.append(SenseHatSensor.temperature()) 
            pressure.append(SenseHatSensor.pressure())

            # Temps d'iteration
            endLoopTime = datetime.now()
            loopTime.append((endLoopTime - startLoopTime).total_seconds())

            # Sauvegarde des valeurs
            dataStorage.dataFile("{:.4f}".format(np.mean(speedAverage)), paths / 'result.txt')
            dataStorage.speedDataFrame(speedPicture[-1], speedPictureCleaned[-1], speedCoordinated[-1], speedAverage[-1], loopTime[-1], paths / 'Data' / 'dataSpeed.csv')
            dataStorage.coordinatedDataFrame(coordinated[-1], paths / 'Data' / 'dataCoordinated.csv')
            dataStorage.environmentDataFrame(humidity[-1], temperature[-1], pressure[-1], paths / 'Data' / 'dataEnvironment.csv')
            dataStorage.IMUDataFrame(gyroscope[-1], accelerometer[-1], magnetometer[-1], paths / 'Data' / 'dataIMU.csv')

            # Temps actuel
            nowTime = datetime.now()
        
        logger.info(f"End of loop")

        # Enregistrement des donnees de vitesse moyenne
        dataStorage.dataFile("{:.4f}".format(np.mean(speedAverage)), paths / 'result.txt')

        # Creation de la carte des points et des graphiques
        if mapFile == True:
            statistic.drawPointMap(coordinated)
        statistic.graphicSpeedPicture(speedPicture, speedPictureCleaned, speedCoordinated, speedAverage)
        statistic.graphicTime(loopTime)
        statistic.graphicHumidity(humidity)
        statistic.graphicTemperature(temperature)
        statistic.graphicPressure(pressure)

        endTime = datetime.now()
        logger.info(f"Running time {endTime - startTime}")

    except Exception as error:
        logger.exception(f"An error occurs when the main function: {error}")