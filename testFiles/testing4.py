import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

planeId = p.loadURDF("plane.urdf")
scale = 1
box = p.loadURDF("cube_small.urdf", [0.5, 0, 0.5], globalScaling=scale)
p.changeDynamics(
    box,
    -1,
    mass=100
)

startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
bot = p.loadURDF("kuka_iiwa/model.urdf", startPos, startOrientation)

n = p.getNumJoints(bot)
endEffectorIndex = 6

slider1 = p.addUserDebugParameter("X", -3, 3, 0)
slider2 = p.addUserDebugParameter("Y", -3, 3, 0)
slider3 = p.addUserDebugParameter("Z", 0, 6, 0)

collidingAlready = False
while p.isConnected():
    targetX = p.readUserDebugParameter(slider1)
    targetY = p.readUserDebugParameter(slider2)
    targetZ = p.readUserDebugParameter(slider3)
    targetPos = [targetX, targetY, targetZ]

    jointPoses = p.calculateInverseKinematics(bot, endEffectorIndex, targetPos)

    contacts = p.getContactPoints(
    bodyA=bot,
    bodyB=box
    )

    pos, orn = p.getBasePositionAndOrientation(box)


    if contacts and not collidingAlready:
        print("COLLISION!")
        scale = scale * 1.5
        p.removeBody(box)
        box = p.loadURDF("cube_small.urdf",pos,orn, globalScaling=scale)
        collidingAlready = True
    elif not contacts:
        collidingAlready = False



    for i in range(n):
        p.setJointMotorControl2(
            bodyUniqueId=bot,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=jointPoses[i]
        )

    p.stepSimulation()
    time.sleep(1/240)