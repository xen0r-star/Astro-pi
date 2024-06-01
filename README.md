# Astro-Pi / Astro Elite
- Nom de l'équipe : Astro Elite
- Étudiants : [Florian Berte](https://github.com/xen0r-star), [Thibaut Dudart](https://github.com/thibautddrt), [Rafaël Ravry](https://github.com/xansterrr)
- Professeur : Laila Bouteglifine
- Pays : Belgique
- Ville : [Ath](https://maps.app.goo.gl/BtFSd77azyfDAs5f6)
- École : [Institut Saint-François de Sales](https://maps.app.goo.gl/fj6R5pSYGHteDu2t7)

<p align="left">
    <img src="logo.jpg" alt="Logo du projet" class="logo" height=200 width=200 style="border-radius: 20px; margin: 0 10px;">
</p>


## 📝 Table des matières

- [❓ Explication du projet](#Projet)
- [💻 Explications du Code](#Code)
- [🚀 Utilisation](#Utilisation)


## ❓ Explication du projet <a name = "Projet"></a>

### Introduction
Ce projet a pour objectif de déterminer la vitesse de la Station Spatiale Internationale (ISS) en exploitant une méthode qui repose sur la capture d'images de la Terre à l'aide d'une caméra Raspberry Pi, combinée aux coordonnées spatiales extraites grâce à la bibliothèque Orbit. Des données et des graphiques sont générés au cours du processus pour analyser les résultats obtenus.

### Méthodologie
1. **Capturer des Images et calculer la Vitesse**<br>
Utilisation d'une caméra Raspberry Pi pour capturer des images de la Terre depuis l'ISS et calculer sa vitesse.

2. **Acquisition de Coordonnées et calculer la Vitesse**<br>
Utilisation de la bibliothèque Orbit pour récupérer les coordonnées de longitude et de latitude de l'ISS et calculer sa vitesse.

3. **Collecte de données Sense HAT**<br>
Collecte de données du Sense HAT. (Gyroscope, Accéléromètre, Magnétomètre, Humidité, Température et Pression)

4. **Collecte et affinement des données**<br>
Répétition des mesures pour la collecte et l'affinement des données pendants 10 minutes.

5. **Création de graphiques et enregistrement de la vitesse**<br>
Création de graphiques et de cartes pour toutes les données collectées, avec enregistrement de la vitesse dans le fichier.

### Résultats Attendus
- Détermination précise de la vitesse de la Station Spatiale Internationale (ISS) à partir des images capturées et des coordonnées spatiales extraites.
- Graphiques illustrant la variation de la vitesse de l'ISS au fil du temps.
- Cartes montrant la trajectoire de l'ISS par rapport à la Terre.
- Analyse des données collectées à partir du Sense HAT, y compris les mesures de gyroscope, d'accéléromètre, de magnétomètre, d'humidité, de température et de pression.
- Enregistrement des données collecté dans des fichiers.

<br>

## 💻 Explications du Code <a name = "Code"></a>
Ce code est conçu pour capturer des images avec une caméra et calculer la vitesse à partir de ces images et des coordonnées GPS. Il gère également le stockage des données et la génération de statistiques et de graphiques.

### Bibliothéque utiliser
- **Bibliothèques Principales**<br>
    - cv2 (OpenCV)
    - exif
    - numpy
- **Bibliothèques Utilitaires**<br>
    - datetime
    - logzero
    - matplotlib
    - pandas
    - pathlib
    - PIL (Pillow)
- **Bibliothèques Matérielles**<br>
    - picamera
    - sense_hat


### Classe checking
- **Fonction folder**<br>
Vérifie l'existence des dossiers nécessaires ("Picture", "Statistic" et "Data") et les crée s'ils n'existent pas.

    ```py
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
    ```


- **Fonction file**<br>
Vérifie l'existence des fichiers nécessaires ("logFile.log" et "result.txt") et les crée s'ils n'existent pas.

    ```py
    def file(self):
        logFile = Path(paths / "logFile.log")
        resultFile = Path(paths / "result.txt")

        if not logFile.is_file():
            logFile.touch()

        if not resultFile.is_file():
            resultFile.touch()
    ```


- **Fonction mapFile**<br>
Vérifie si le fichier de carte ("map.png") existe dans le dossier "Resources" et retourne un booléen en conséquence.

    ```py
    def mapFile(self):
        mapFile = Path(paths / "Resources" / "map.png")

        if not mapFile.is_file():
            mapFile = False
        else:
            mapFile = True

        return mapFile
    ```


### Classe speed
- **Fonction speedPicture**<br>
Calcule la vitesse à partir de deux images en utilisant les caractéristiques ORB pour détecter les points d'intérêt et les correspondances entre les images.

    ```py
    def speedPicture(self, image1, image2, feature=1000, GSD=12648):
        def getData(image):
            with open(image, 'rb') as imageFile:
                img = exifImage(imageFile)
                timeStr = img.get("datetime_original")
                time = datetime.strptime(timeStr, '%Y:%m:%d %H:%M:%S')
            return time

        # obtenir la difference d'heure
        time1 = getData(image1)
        time2 = getData(image2)
        
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

        return speed

    ```


- **Fonction speedCoordinated**<br>
Calcule la vitesse à partir de deux images en utilisant les coordonnées GPS stockées dans les données Exif.

    ```py
    def speedCoordinated(self, image1, image2):
        def getData(image):
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
        
        # obtenir les donnees d'image
        time1, lat1, lon1 = getData(image1)
        time2, lat2, lon2 = getData(image2)

        timeDifference = (time2 - time1).seconds
        
        # formule de haversine
        a = math.sin((math.radians(lat2) - math.radians(lat1)) / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin((math.radians(lon2) - math.radians(lon1)) / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = 6371.0097714 * c

        speed = ((distance * 1000) / timeDifference) / 1000

        return speed
    ```


### Classe pictureCamera
- **Fonction take**<br>
Capture un certain nombre d'images avec la caméra, enregistre les coordonnées GPS dans les données Exif et stocke les images dans le dossier "Picture".

    ```py
    def take(self, number):
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

        return self.pictureNumber, coordinated
    ```


### Classe dataStorage
- **Fonction dataFile**<br>
Stocke les données dans un fichier texte.

    ```py
    def dataFile(self, data, file):
        with open(file, 'w') as file:
            file.write(data)
            file.close()

            return True
    ```


- **Fonction speedDataFrame**<br>
Stocke les données de vitesse dans un fichier CSV.

    ```py
    def speedDataFrame(self, speedPicture, speedPictureCleaned, speedCoordinated, speedAverage, loopTime, file):
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

        return True
    ```


- **Fonction coordinatedDataFrame**<br>
Stocke les données de coordonné dans un fichier CSV.

    ```py
    def coordinatedDataFrame(self, coordinated, file):
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

        return True
    ```


- **Fonction environmentDataFrame**<br>
Stocke les données d'environment dans un fichier CSV.

    ```py
    def environmentDataFrame(self, humidity, temperature, pressure, file):
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
        
        return True
    ```


- **Fonction IMUDataFrame**<br>
Stocke les données IMU dans un fichier CSV.

    ```py
    def IMUDataFrame(self, gyroscope, accelerometer, magnetometer, file):
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

        return True
    ```


### Classe statistic
- **Fonction drawPointMap**<br>
Dessine les points de suivi des stations sur une carte à partir des coordonnées GPS.

    ```py
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
    ```


- **Fonction graphicSpeedPicture**<br>
Génère un graphique de la vitesse à partir des différentes méthodes de calcul.

    ```py
    def graphicSpeedPicture(self, speedPicture, speedPictureCleaned, speedCoordinated, speedAverage):
        plt.clf()

        x = list(range(max(len(speedPicture), len(speedPictureCleaned), len(speedCoordinated), len(speedAverage))))

        plt.plot(x, speedPicture, marker='.', label='Speed with picture')
        plt.plot(x, speedPictureCleaned, marker='.', label='Speed with picture cleaned')
        plt.plot(x, speedCoordinated, marker='.', label='Speed with coordinated')
        plt.plot(x, speedAverage, marker='.', label='Average speed')

        plt.legend(loc='upper left')
        plt.savefig(self.output / 'graphic_SpeedPicture.png')
    ```


- **Fonction graphicTime**<br>
Génère un graphique du temps d'itération.

    ```py
    def graphicTime(self, time):
        plt.clf()

        x = list(range(len(time)))
        plt.plot(x, time, marker='.', label='Time per iteration')

        plt.legend(loc='upper left')
        plt.savefig(self.output / 'graphic_Time.png')
    ```


- **Fonction graphicHumidity**<br>
Génère un graphique des valeur d'humidité.

    ```py
    def graphicHumidity(self, humidity):
        plt.clf()

        x = list(range(len(humidity)))
        plt.plot(x, humidity, marker='.', label='Humidity')

        plt.legend(loc='upper left')
        plt.savefig(self.output / 'graphic_Humidity.png')
    ```


- **Fonction graphicTemperature**<br>
Génère un graphique des valeur de température.

    ```py
    def graphicTemperature(self, temperature):
        plt.clf()

        x = list(range(len(temperature)))
        plt.plot(x, temperature, marker='.', label='Temperature')

        plt.legend(loc='upper left')
        plt.savefig(self.output / 'graphic_Temperature.png')
    ```


- **Fonction graphicPressure**<br>
Génère un graphique des valeur de pression.

    ```py
    def graphicPressure(self, pressure):
        plt.clf()

        x = list(range(len(pressure)))
        plt.plot(x, pressure, marker='.', label='Pressure')

        plt.legend(loc='upper left')
        plt.savefig(self.output / 'graphic_Pressure.png')
    ```


- **Fonction outlier**<br>
Identifie et gère les valeurs aberrantes des données.

    ```py
    def outlier(self, data, dataCleaned = []):
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
    ```


### Classe SenseHatSensor
- **Fonction gyroscope**<br>
Fournit les données du gyroscope, comprenant les valeurs des axes x, y et z.

    ```py
    def gyroscope(self):
        gyroData = self.sense.get_gyroscope_raw()
        return {
            "x": gyroData["x"],
            "y": gyroData["y"],
            "z": gyroData["z"]
        }
    ```


- **Fonction accelerometer**<br>
Fournit les données du accéléromètre, comprenant les valeurs des axes x, y et z.

    ```py
    def accelerometer(self):
        accelData = self.sense.get_accelerometer_raw()
        return {
            "x": accelData["x"],
            "y": accelData["y"],
            "z": accelData["z"]
        }
    ```


- **Fonction magnetometer**<br>
Fournit les données du magnétomètre, comprenant les valeurs des axes x, y et z.

    ```py
    def magnetometer(self):
        magData = self.sense.get_compass_raw()
        return {
            "x": magData["x"],
            "y": magData["y"],
            "z": magData["z"]
        }
    ```


- **Fonction humidity**<br>
Fournit les données d'humidité.

    ```py
    def humidity(self):
        humidity = self.sense.get_humidity()
        return humidity
    ```


- **Fonction temperature**<br>
Fournit les données de température.

    ```py
    def temperature(self):
        temp = self.sense.get_temperature()
        return temp
    ```


- **Fonction pressure**<br>
Fournit les données de pression.

    ```py
    def pressure(self):
        pressure = self.sense.get_pressure()
        return pressure
    ```


### Partie Principale
1. **Initialisation**<br>
Le chronomètre est enclenché pour mesurer la durée totale d'exécution. Ensuite, la classe de vérification est utilisée pour vérifier l'existence des dossiers et des fichiers nécessaires, et les classes et variables sont initialisées.

    ```py
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

    # Temps actuel
    nowTime = datetime.now()
    ```


2. **Capture d'Images et Calcul de Vitesse**<br>
La boucle itérative capture des images à intervalles réguliers pendant 9 minutes ou jusqu'à ce qu'un maximum de 42 images soit atteint. La classe pictureCamera prend les images et enregistre les coordonnées GPS dans les données Exif. Des calculs de valeurs aberrantes sont effectués sur les mesures de vitesse à partir d'images en raison de leur variabilité importante. Les données des capteurs Sense Hat sont collectées et à chaque itération, toutes les données sont sauvegardées.

    ```py
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
    ```


3. **Génération de Statistiques et de Graphiques**<br>
Une fois toutes les images capturées et la vitesse calculée, des statistiques sont générées à partir des données collectées. La classe statistic est utilisée pour créer des graphiques de vitesse, de temps, d'humidité, de température et de pression. Si une carte est disponible, les points de suivi des stations sont dessinés sur la carte.

    ```py
    # Enregistrement des donnees de vitesse moyenne
    dataStorage.dataFile("{:.4f}".format(np.mean(speedAverage)), paths / 'result.txt')

    # Creation de la carte des points et du graphique de vitesse
    if mapFile == True:
        statistic.drawPointMap(coordinated)
    statistic.graphicSpeedPicture(speedPicture, speedPictureCleaned, speedCoordinated, speedAverage)
    statistic.graphicTime(loopTime)
    statistic.graphicHumidity(humidity)
    statistic.graphicTemperature(temperature)
    statistic.graphicPressure(pressure)
    ```

<br>

## 🚀 Utilisation <a name = "Utilisation"></a>
Pour utiliser ce code, assurez-vous d'avoir installé les dépendances nécessaires et de disposer des autorisations appropriées pour accéder à la caméra et aux fichiers du système. Ensuite, exécutez le script principal (main.py) pour capturer les images, calculer la vitesse et générer les statistiques. Assurez-vous que les dossiers et fichiers nécessaires sont présents avant d'exécuter le code.

Les instructions pour lancer une simulation du programme sont disponibles sur le site [rasperry.org](https://projects.raspberrypi.org/en/projects/mission-space-lab-creator-guide/2).