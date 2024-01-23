from exif import Image
from datetime import datetime
import cv2
import math


image_1 = './Image/Cloud/img_005.jpg'
image_2 = './Image/Cloud/img_006.jpg'


def speedImage(image1, image2, feature=1000, GSD=12648):
    def get_time(image):
        with open(image, 'rb') as image_file:
            img = Image(image_file)
            time_str = img.get("datetime_original")
            time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
        return time


    # get time difference
    time_1 = get_time(image1)
    time_2 = get_time(image2)
    time_difference = time_2 - time_1
    time_difference =  time_difference.seconds

    # convert to cv
    image_1_cv = cv2.imread(image1, 0)
    image_2_cv = cv2.imread(image2, 0)

    # calculate features
    orb = cv2.ORB_create(nfeatures = feature)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)

    # calculate matches
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    # find matching coordinates
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1,y1))
        coordinates_2.append((x2,y2))

    # calculate mean distance
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances = all_distances + distance
    average_feature_distance = all_distances / len(merged_coordinates)

    # calculate speed in kmps
    distance = average_feature_distance * GSD / 100000
    return distance / time_difference


print(speedImage(image_1, image_2))


    

def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
    resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', resize)
    cv2.waitKey(0)
    cv2.destroyWindow('matches')

# display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches) # Display matches