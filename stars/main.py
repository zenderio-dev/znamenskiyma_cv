import numpy as np
from skimage.measure import label
from skimage.morphology import binary_opening

image = np.load("stars.npy")

cross = np.zeros((5, 5), dtype=int)
indices = np.arange(5)
cross[indices, indices] = 1        
cross[indices, 4 - indices] = 1       


plus = np.zeros((5, 5), dtype=int)
plus[2, :] = 1 
plus[:, 2] = 1 

opened_cross = binary_opening(image, cross)
opened_plus = binary_opening(image, plus)

num_cross = np.max(label(opened_cross))
num_plus = np.max(label(opened_plus))
total_stars = num_cross + num_plus

print("Количество звёзд:", total_stars)