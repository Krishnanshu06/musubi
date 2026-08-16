import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

planeId = p.loadURDF("plane.urdf")

#########################################################

halfExtents = [0.5, 0.5, 1]
colShapeCube= p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=halfExtents
)

visShapeCube = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=halfExtents,
    rgbaColor=[1, 0, 0, 1] 
)

##########################################################

halfExtentsLink1 = [0.25, 0.25, 0.5]
colShapeCubeLink1= p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=halfExtentsLink1
)

visShapeCubeLink1 = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=halfExtentsLink1,
    rgbaColor=[1, 0, 0, 1] 
)

##########################################################

halfExtentsLink0 = [0.4, 0.4, 0.75]
colShapeCubeLink0= p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=halfExtentsLink0
)

visShapeCubeLink0 = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=halfExtentsLink1,
    rgbaColor=[1, 0, 0, 1] 
)

##########################################################

colShapeSphereLink= p.createCollisionShape(
    p.GEOM_SPHERE,
    radius=0.5)

visShapeSphereLink = p.createVisualShape(
    p.GEOM_SPHERE,
    radius=0.5,
    rgbaColor=[1, 1, 0, 1] 
)

shape = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=colShapeCube,
    baseVisualShapeIndex=visShapeCube,
    basePosition=[0, 0, 2],

    linkMasses=[1,1],
    linkCollisionShapeIndices=[colShapeCubeLink0, colShapeCubeLink1],
    linkVisualShapeIndices=[visShapeCubeLink0, visShapeCubeLink1],
    linkPositions=[
        [0, 0, 1.5],
        [0, 0, 1]
    ],
    
    linkOrientations=[
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ],

    linkInertialFramePositions=[
        [0, 0, 0],
        [0, 0, 0]
    ],

    linkInertialFrameOrientations=[
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ],

    linkParentIndices=[
        0,
        1
    ],

    linkJointTypes=[
        p.JOINT_REVOLUTE,
        p.JOINT_REVOLUTE
    ],

    linkJointAxis=[
        [0, 1, 0],
        [0, 1, 0]
    ]
)

while True:
#     p.setJointMotorControl2(
#     shape,
#     0,
#     p.VELOCITY_CONTROL,
#     targetVelocity=20,
#     force=10
# )
    p.stepSimulation()
    time.sleep(1/240)

