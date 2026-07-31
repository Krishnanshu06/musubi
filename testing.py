import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,1]
startPos2 = [2,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
boxId2 = p.loadURDF("racecar/racecar.urdf",startPos2, startOrientation)
n = p.getNumJoints(boxId2)
for i in range(n):
    info = p.getJointInfo(boxId2, i)
    print(f"""
Joint Index : {info[0]}
Joint Name  : {info[1].decode()}
Joint Type  : {info[2]}
Lower Limit : {info[8]}
Upper Limit : {info[9]}
Max Force   : {info[10]}
Max Velocity: {info[11]}
Link Name   : {info[12].decode()}
-------------------------------
""")

maxForce = 5
for i in range (100000):
    p.stepSimulation()
    time.sleep(1./240.)
    if (i%1000==0):
        p.setJointMotorControl2(bodyUniqueId=boxId2, 
        jointIndex=2, 
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity = 50,
        force = maxForce)
        p.setJointMotorControl2(bodyUniqueId=boxId2, 
        jointIndex=3, 
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity = 50,
        force = maxForce)
        p.setJointMotorControl2(bodyUniqueId=boxId2, 
        jointIndex=5, 
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity = 50,
        force = maxForce)
        p.setJointMotorControl2(bodyUniqueId=boxId2, 
        jointIndex=7, 
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity = 50,
        force = maxForce)

p.disconnect()
