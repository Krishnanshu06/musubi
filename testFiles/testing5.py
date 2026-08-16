import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

planeId = p.loadURDF("plane.urdf")

halfExtents = [1, 1, 0.5]
colShapeCube= p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=halfExtents
)

visShapeCube = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=halfExtents,
    rgbaColor=[1, 0, 0, 1] 
)

cube = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=colShapeCube,
    baseVisualShapeIndex=visShapeCube,
    basePosition=[0, 0, 5]
)

colShapeSphere= p.createCollisionShape(
    p.GEOM_SPHERE,
    radius=0.5)

visShapeSphere = p.createVisualShape(
    p.GEOM_SPHERE,
    radius=0.5,
    rgbaColor=[1, 1, 0, 0.4] 
)

Sphere = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=colShapeSphere,
    baseVisualShapeIndex=visShapeSphere,
    basePosition=[0, 1, 2]
)

colShapeCylinder= p.createCollisionShape(
    p.GEOM_CYLINDER,
    radius=0.5,
    height=1)

visShapeCylinder = p.createVisualShape(
    p.GEOM_CYLINDER,
    radius=0.5,
    length=1,
    rgbaColor=[1, 0, 0.5, 0.2] 
)

Cylinder = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=colShapeCylinder,
    baseVisualShapeIndex=visShapeCylinder,
    basePosition=[2, 1, 2]
)

while True:
    p.stepSimulation()
    time.sleep(1/240)

