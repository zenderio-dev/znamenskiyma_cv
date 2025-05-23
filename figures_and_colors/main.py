import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import label
from skimage.measure import regionprops

loaded_image = plt.imread("balls_and_rects.png")

distinct_colors = np.unique(loaded_image.reshape(-1, loaded_image.shape[-1]), axis=0)

tolerance = 10 / 255.0

circle_counts = {}
rectangle_counts = {}

temp_image = loaded_image.copy()

for color in distinct_colors[1:]:
    color_key = ", ".join([str(val) for val in color])

    color_mask = np.all(np.abs(temp_image - color) < tolerance, axis=-1)

    labeled_mask = label(color_mask)

    regions = regionprops(labeled_mask)

    circle_counts[color_key] = 0
    rectangle_counts[color_key] = 0

    for region in regions:
        if np.isclose(region.eccentricity, 0):
            circle_counts[color_key] += 1
        else:
            rectangle_counts[color_key] += 1

print("Круги:\n", circle_counts)
total_circles = sum(circle_counts.values())

print("Прямоугольники:\n", rectangle_counts)
total_rectangles = sum(rectangle_counts.values())

print(f"Всего кругов: {total_circles}, всего прямоугольников: {total_rectangles}, всего объектов: {total_circles + total_rectangles}")
