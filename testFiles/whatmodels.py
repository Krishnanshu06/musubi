import os
import pybullet_data

data_path = pybullet_data.getDataPath()

for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".urdf"):
            # Prints the relative path ready for p.loadURDF()
            print(os.path.relpath(os.path.join(root, file), data_path))
