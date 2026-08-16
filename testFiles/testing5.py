import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

planeId = p.loadURDF("plane.urdf")

colShape= p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=[0.5, 0.5, 0.5]
)

visShape = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=[0.5, 0.5, 0.5],
    rgbaColor=[1, 0, 0, 1] 
)

cube = p.createMultiBody(
    baseMass=100,
    baseCollisionShapeIndex=colShape,
    baseVisualShapeIndex=visShape,
    basePosition=[0.5, 0, 0.5]
)

while True:
    p.stepSimulation()
    time.sleep(1/240)

