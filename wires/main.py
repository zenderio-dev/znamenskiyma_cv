import numpy as np
from skimage.measure import label
from skimage.morphology import (binary_erosion)


for i in range(1, 7):
    print(f'файл: {i}')
    filename = f"wires{i}npy.txt"
    data = np.load(filename)
    labeled = label(data)
    struct_elem = np.ones((3, 1))
        
    for k in range(1, np.max(labeled)+1):
        
        wires = (labeled==k)
        erosionWires = binary_erosion(wires, struct_elem)
        newWires = label(erosionWires)

        if np.max(newWires) == np.max(wires):
            print(f'провод {k} не разрезан')
        elif np.max(newWires) < np.max(wires):
            print('провода не существует')
        else:
            print( f"у провода {k} {np.max(newWires)-np.max(wires)} разрывов")