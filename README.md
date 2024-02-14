# Projet de Calcul de la Vitesse de l'ISS
- Nom de l'équipe : FloThiRaf
- Professeur : Laila Bouteglifine
- Étudiants : [Florian Berte](https://github.com/xen0r-star), [Thibaut Dudart](https://github.com/thibautddrt), [Rafaël Ravry](https://github.com/xansterrr)
- École : [Institut Saint-François de Sales](https://maps.app.goo.gl/fj6R5pSYGHteDu2t7)
- Ville : [Ath](https://maps.app.goo.gl/BtFSd77azyfDAs5f6)
- Pays : Belgique

## Introduction
Le projet vise à calculer la vitesse de la Station Spatiale Internationale (ISS) en utilisant une approche basée sur la capture d'images de la Terre à l'aide d'une caméra Raspberry Pi. L'objectif est d'obtenir des images simultanées de la Terre, de les analyser à l'aide de Google Coral pour distinguer les zones avec nuages de celles sans nuages, puis de calculer la vitesse de l'ISS en mesurant la distance entre des éléments repérés sur ces images.

## Méthodologie
1. **Acquisition des Images**<br>
Utilisation d'une caméra Raspberry Pi pour capturer des images de la Terre depuis la Station Spatiale Internationale.

2. **Analyse d'Images**<br>
Traitement des images à l'aide de Google Coral pour différencier les zones avec nuages de celles sans nuages.

3. **Sélection des Paires d'Images Simultanées**<br>
Répétition du processus jusqu'à l'obtention de deux images ayant les mêmes caractéristiques (nuageuses ou sans nuages) et prises simultanément.

4. **Calcul de la Distance**<br>
Mesure de la distance entre deux éléments identifiables sur les images en utilisant des coordonnées x et y.

5. **Calcul de la Vitesse**<br>
Utilisation des coordonnées récupérées pour calculer la vitesse de l'ISS en analysant le déplacement relatif entre les deux images.

6. **Répétition du Processus**<br>
Répétition du processus plusieurs fois pour obtenir une vitesse moyenne de l'ISS.


## Résultats Attendus
Le projet devrait fournir une méthode permettant de calculer la vitesse de l'ISS en utilisant des images de la Terre capturées à partir de la station spatiale. Les résultats seront basés sur une analyse précise des images et une mesure correcte de la distance entre des éléments repérés.

## Explications du Code
Ce code est conçu pour capturer des images avec une caméra et calculer la vitesse à partir de ces images et des coordonnées GPS. Il gère également le stockage des données et la génération de statistiques et de graphiques.

### Classe checking
- **Fonction folder**<br>
Vérifie l'existence des dossiers nécessaires ("Picture" et "Statistic") et les crée s'ils n'existent pas.
    ```py
    def folder(self):
        pictureFolder = Path(paths / "Picture")
        statisticFolder = Path(paths / "Statistic")

        if not pictureFolder.is_dir():
            pictureFolder.mkdir(parents=True, exist_ok=True)
        
        if not statisticFolder.is_dir():
            statisticFolder.mkdir(parents=True, exist_ok=True)
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
    def graphicSpeedPicture(self, data1_1, data1_2, data2, data3):
        plt.clf()

        x = list(range(max(len(data1_1), len(data1_2), len(data2), len(data3))))

        plt.plot(x, data1_1, marker='.', label='Speed with picture')
        plt.plot(x, data1_2, marker='.', label='Speed with picture cleaned')
        plt.plot(x, data2, marker='.', label='Speed with coordinated')
        plt.plot(x, data3, marker='.', label='Average speed')

        plt.legend(loc='upper left')
        plt.savefig(self.output / 'graphic_SpeedPicture.png')
    ```

- **Fonction graphicTime**<br>
Génère un graphique du temps d'itération.
    ```py
    def graphicTime(self, data1):
        plt.clf()

        x = list(range(len(data1)))
        plt.plot(x, data1, marker='.', label='Time per iteration')

        plt.legend(loc='upper left')
        plt.savefig(self.output / 'graphic_Time.png')
    ```

- **Fonction outlier**<br>
Identifie et gère les valeurs aberrantes dans les données de vitesse.
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

        print(dataCleaned)
        print(dataCopy[-1])
        dataCleaned.append(dataCopy[-1])
        print(dataCleaned)

        return dataCleaned
    ```

### Partie Principale
1. **Initialisation**<br>
Le chronomètre est lancé pour mesurer le temps d'exécution total. Ensuite, la vérification des dossiers et des fichiers nécessaires est effectuée à l'aide de la classe checking.
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

    # Listes pour stocker les informations
    speedPicture, speedPictureCleaned, speedCoordinated, speedAverage = [], [], [], []
    coordinated = []
    loopTime = []
    pictureNumber = 0

    # Temps actuel
    nowTime = datetime.now()
    ```

2. **Capture d'Images et Calcul de Vitesse**<br>
Une boucle est utilisée pour capturer les images à intervalles réguliers pendant une période de temps définie ou jusqu'à un nombre d'images spécifié. La classe pictureCamera est utilisée pour prendre les images et enregistrer les coordonnées GPS dans les données Exif. La vitesse est calculée à partir des images prises à l'aide des méthodes de la classe speed. Les données de vitesse sont stockées à l'aide de la classe dataStorage.
    ```py
    while ((nowTime < startTime + timedelta(minutes=9)) and (pictureNumber < 42)):
        startLoopTime = datetime.now()

        pictureNumber, pictureCoordinated = pictureCamera.take(2)
        coordinated.append(pictureCoordinated)

        if pictureNumber != None:
            speedPicture.append(speed.speedPicture(paths / 'Picture' / f'picture{pictureNumber - 1:03d}.jpg', paths / 'Picture' / f'picture{pictureNumber:03d}.jpg'))
            speedCoordinated.append(speed.speedCoordinated(paths / 'Picture' / f'picture{pictureNumber - 1:03d}.jpg', paths / 'Picture' / f'picture{pictureNumber:03d}.jpg'))

        speedPictureCleaned = statistic.outlier(speedPicture, speedPictureCleaned)
        if speedPictureCleaned[-1] == None:
            speedAverage.append(speedCoordinated[-1])
        else:
            speedAverage.append((speedPictureCleaned[-1] + speedCoordinated[-1]) / 2)

        # Temps d'iteration
        endLoopTime = datetime.now()
        loopTime.append((endLoopTime - startLoopTime).total_seconds())

        # Sauvegarde des valeurs
        dataStorage.dataFile("{:.4f}".format(np.mean(speedAverage)), paths / 'result.txt')
        dataStorage.speedDataFrame(speedPicture[-1], speedPictureCleaned[-1], speedCoordinated[-1], speedAverage[-1], loopTime[-1], paths / 'dataSpeed.csv')

        # Temps actuel
        nowTime = datetime.now()
    ```

3. **Génération de Statistiques et de Graphiques**<br>
Une fois toutes les images capturées et la vitesse calculée, des statistiques sont générées à partir des données collectées. La classe statistic est utilisée pour créer des graphiques de vitesse et de temps. Si une carte est disponible, les points de suivi des stations sont dessinés sur la carte.
    ```py
    # Enregistrement des donnees de vitesse moyenne
    dataStorage.dataFile("{:.4f}".format(np.mean(speedAverage)), paths / 'result.txt')

    # Creation de la carte des points et du graphique de vitesse
    statistic.graphicSpeedPicture(speedPicture, speedPictureCleaned, speedCoordinated, speedAverage)
    statistic.graphicTime(loopTime)
    if mapFile == True:
        statistic.drawPointMap(coordinated)
    ```

### Utilisation
Pour utiliser ce code, assurez-vous d'avoir installé les dépendances nécessaires et de disposer des autorisations appropriées pour accéder à la caméra et aux fichiers du système. Ensuite, exécutez le script principal (main.py) pour capturer les images, calculer la vitesse et générer les statistiques. Assurez-vous que les dossiers et fichiers nécessaires sont présents avant d'exécuter le code.