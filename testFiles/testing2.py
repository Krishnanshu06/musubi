import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf")
robot = p.loadURDF("kuka_iiwa/model.urdf", [0,0,0])

p.setGravity(0,0,-9.81)

# Create one slider for every movable joint
sliders = []

for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)

    name = info[1].decode()

    lower = info[8]
    upper = info[9]

    slider = p.addUserDebugParameter(name, lower, upper, 0)

    sliders.append(slider)

while p.isConnected():

    for i in range(p.getNumJoints(robot)):

        target = p.readUserDebugParameter(sliders[i])
    
        p.setJointMotorControl2(
            bodyUniqueId=robot,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target
        )

    p.stepSimulation()
    time.sleep(1/240)