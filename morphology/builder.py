"""
Creature builder and morphology engine for procedural multi-body generation in PyBullet.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
import pybullet as p


@dataclass
class JointInfo:
    """Holds parsed metadata about a creature's joint."""
    index: int
    name: str
    joint_type: int
    lower_limit: float
    upper_limit: float
    max_force: float
    max_velocity: float
    link_name: str


class Creature:
    """
    Represents a multi-body robot/creature in PyBullet with actuation and sensor utilities.
    """
    def __init__(self, body_id: int, client_id: int):
        self.body_id = body_id
        self.client_id = client_id
        self.joints: List[JointInfo] = []
        self.actuated_joint_indices: List[int] = []
        self._parse_joints()

    def _parse_joints(self) -> None:
        """Inspect all joints and extract controllable (revolute/prismatic) joints."""
        self.joints.clear()
        self.actuated_joint_indices.clear()
        num_joints = p.getNumJoints(self.body_id, physicsClientId=self.client_id)

        for i in range(num_joints):
            info = p.getJointInfo(self.body_id, i, physicsClientId=self.client_id)
            joint = JointInfo(
                index=info[0],
                name=info[1].decode("utf-8"),
                joint_type=info[2],
                lower_limit=info[8],
                upper_limit=info[9],
                max_force=info[10] if info[10] > 0 else 30.0,
                max_velocity=info[11] if info[11] > 0 else 10.0,
                link_name=info[12].decode("utf-8"),
            )
            self.joints.append(joint)
            # Controllable if revolute or prismatic
            if joint.joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.actuated_joint_indices.append(i)

    @property
    def num_actuators(self) -> int:
        return len(self.actuated_joint_indices)

    def apply_action_positions(
        self,
        target_positions: np.ndarray,
        forces: Optional[List[float]] = None
    ) -> None:
        """
        Apply normalized actions [-1, 1] scaled to joint lower/upper angle limits.
        """
        assert len(target_positions) == self.num_actuators, (
            f"Expected {self.num_actuators} actions, got {len(target_positions)}"
        )

        for idx, joint_idx in enumerate(self.actuated_joint_indices):
            joint = self.joints[joint_idx]
            low, high = joint.lower_limit, joint.upper_limit
            # If limits not properly set, default to [-pi/2, pi/2]
            if low >= high:
                low, high = -np.pi / 2, np.pi / 2
            
            # Map [-1, 1] normalized action to [low, high]
            target_angle = low + (target_positions[idx] + 1.0) * 0.5 * (high - low)
            force = forces[idx] if forces is not None else joint.max_force

            p.setJointMotorControl2(
                bodyUniqueId=self.body_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(target_angle),
                force=force,
                physicsClientId=self.client_id,
            )

    def get_base_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (position, orientation_quaternion, linear_velocity, angular_velocity).
        """
        pos, orn = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.body_id, physicsClientId=self.client_id)
        return (
            np.array(pos, dtype=np.float32),
            np.array(orn, dtype=np.float32),
            np.array(lin_vel, dtype=np.float32),
            np.array(ang_vel, dtype=np.float32),
        )

    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (joint_positions, joint_velocities) for all actuated joints.
        """
        states = p.getJointStates(
            self.body_id,
            self.actuated_joint_indices,
            physicsClientId=self.client_id,
        )
        positions = [s[0] for s in states]
        velocities = [s[1] for s in states]
        return np.array(positions, dtype=np.float32), np.array(velocities, dtype=np.float32)

    def get_ground_contacts(self, plane_id: int) -> int:
        """Returns number of contact points between creature and ground plane."""
        contacts = p.getContactPoints(
            bodyA=self.body_id,
            bodyB=plane_id,
            physicsClientId=self.client_id,
        )
        return len(contacts)


class MorphologyBuilder:
    """
    Builder for procedural or preset robot morphologies using PyBullet primitives.
    """
    @staticmethod
    def create_quadruped(
        client_id: int,
        start_pos: Tuple[float, float, float] = (0.0, 0.0, 0.5),
        torso_size: Tuple[float, float, float] = (0.4, 0.3, 0.1),
        leg_length: float = 0.25,
        leg_radius: float = 0.04,
    ) -> Creature:
        """
        Builds a symmetric 4-legged robot (8 actuated revolute joints: 4 hip, 4 knee).
        """
        hx, hy, hz = torso_size[0] / 2, torso_size[1] / 2, torso_size[2] / 2
        
        # 1. Torso Shapes
        col_torso = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=client_id
        )
        vis_torso = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=[0.2, 0.6, 0.86, 1.0], physicsClientId=client_id
        )

        # 2. Leg Link Primitives
        col_upper = p.createCollisionShape(
            p.GEOM_CAPSULE, radius=leg_radius, height=leg_length, physicsClientId=client_id
        )
        vis_upper = p.createVisualShape(
            p.GEOM_CAPSULE, radius=leg_radius, length=leg_length, rgbaColor=[0.9, 0.4, 0.2, 1.0], physicsClientId=client_id
        )

        col_lower = p.createCollisionShape(
            p.GEOM_CAPSULE, radius=leg_radius * 0.8, height=leg_length, physicsClientId=client_id
        )
        vis_lower = p.createVisualShape(
            p.GEOM_CAPSULE, radius=leg_radius * 0.8, length=leg_length, rgbaColor=[0.95, 0.8, 0.2, 1.0], physicsClientId=client_id
        )

        # 4 Legs: Front-Left, Front-Right, Back-Left, Back-Right
        # Each leg: Upper Link (connected to torso) -> Lower Link (connected to Upper Link)
        link_masses = []
        link_col_indices = []
        link_vis_indices = []
        link_positions = []
        link_orientations = []
        link_parent_indices = []
        link_joint_types = []
        link_joint_axes = []

        leg_offsets = [
            (hx, hy),    # Front-Left
            (hx, -hy),   # Front-Right
            (-hx, hy),   # Back-Left
            (-hx, -hy),  # Back-Right
        ]

        for i, (ox, oy) in enumerate(leg_offsets):
            upper_idx = i * 2
            lower_idx = i * 2 + 1

            # Upper Leg (Hip pitch joint)
            link_masses.append(0.5)
            link_col_indices.append(col_upper)
            link_vis_indices.append(vis_upper)
            link_positions.append([ox, oy, 0.0])
            link_orientations.append([0, 0, 0, 1])
            link_parent_indices.append(0)  # Parent is base torso
            link_joint_types.append(p.JOINT_REVOLUTE)
            link_joint_axes.append([0, 1, 0])

            # Lower Leg (Knee pitch joint)
            link_masses.append(0.3)
            link_col_indices.append(col_lower)
            link_vis_indices.append(vis_lower)
            link_positions.append([0, 0, -leg_length])
            link_orientations.append([0, 0, 0, 1])
            link_parent_indices.append(upper_idx + 1)  # Parent is upper leg (1-indexed in multi-body link list)
            link_joint_types.append(p.JOINT_REVOLUTE)
            link_joint_axes.append([0, 1, 0])

        body_id = p.createMultiBody(
            baseMass=2.5,
            baseCollisionShapeIndex=col_torso,
            baseVisualShapeIndex=vis_torso,
            basePosition=list(start_pos),
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_col_indices,
            linkVisualShapeIndices=link_vis_indices,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=[[0, 0, -leg_length / 2] for _ in range(8)],
            linkInertialFrameOrientations=[[0, 0, 0, 1] for _ in range(8)],
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes,
            physicsClientId=client_id,
        )

        # Set joint limits and lateral friction on feet
        for j_idx in range(8):
            p.changeDynamics(body_id, j_idx, lateralFriction=1.0, physicsClientId=client_id)

        return Creature(body_id=body_id, client_id=client_id)
