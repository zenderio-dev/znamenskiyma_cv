import numpy as np
from skimage.measure import regionprops
import cv2
from skimage.morphology import label

lower_bound = np.array([5, 120, 110])
upper_bound = np.array([120, 260, 220])

pencil_count = 0

for image_number in range(1, 13):
    image_path = f"images/img ({image_number}).jpg"
    color_image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    dilated_mask = cv2.dilate(color_mask, np.ones((9, 9), np.uint8))
    processed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=3)

    labeled_mask = label(processed_mask)
    region_properties = regionprops(labeled_mask)

    potential_pencils = [region for region in region_properties if (region.area > 85000 and 1 - region.eccentricity < 0.02)]
    number_of_pencils = len(potential_pencils)
    pencil_count += number_of_pencils
    print(f"Image {image_number} Pencils:", number_of_pencils)

print("Total pencils in the pictures:", pencil_count)