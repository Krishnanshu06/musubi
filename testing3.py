import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

planeId = p.loadURDF("plane.urdf")

startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
bot = p.loadURDF("kuka_iiwa/model.urdf", startPos, startOrientation)

n = p.getNumJoints(bot)
endEffectorIndex = 6

slider1 = p.addUserDebugParameter("X", -3.14, 3.14, 0)
slider2 = p.addUserDebugParameter("Y", -3.14, 3.14, 0)
slider3 = p.addUserDebugParameter("Z", -3.14, 3.14, 0)

while p.isConnected():
    targetX = p.readUserDebugParameter(slider1)
    targetY = p.readUserDebugParameter(slider2)
    targetZ = p.readUserDebugParameter(slider3)

    targetPos = [targetX, targetY, targetZ]
    jointPoses = p.calculateInverseKinematics(bot, endEffectorIndex, targetPos)

    for i in range(n):
        p.setJointMotorControl2(
            bodyUniqueId=bot,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=jointPoses[i]
        )

    p.stepSimulation()
    time.sleep(1/240)