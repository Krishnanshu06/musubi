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
    def __init__(
        self,
        body_id: int,
        client_id: int,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
        max_forces: Optional[List[float]] = None,
    ):
        self.body_id = body_id
        self.client_id = client_id
        self.joints: List[JointInfo] = []
        self.actuated_joint_indices: List[int] = []
        self._parse_joints(joint_limits=joint_limits, max_forces=max_forces)

    def _parse_joints(
        self,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
        max_forces: Optional[List[float]] = None,
    ) -> None:
        """Inspect all joints and extract controllable joints with explicit angle limits."""
        self.joints.clear()
        self.actuated_joint_indices.clear()
        num_joints = p.getNumJoints(self.body_id, physicsClientId=self.client_id)

        actuator_idx = 0
        for i in range(num_joints):
            info = p.getJointInfo(self.body_id, i, physicsClientId=self.client_id)
            joint_type = info[2]

            # Default limits from PyBullet info or fallback
            low_lim = info[8]
            high_lim = info[9]
            force = info[10] if info[10] > 0 else 35.0

            # Override with explicit DNA limits if provided
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                if joint_limits is not None and actuator_idx < len(joint_limits):
                    low_lim, high_lim = joint_limits[actuator_idx]
                elif low_lim >= high_lim:
                    low_lim, high_lim = -np.pi / 4, np.pi / 4

                if max_forces is not None and actuator_idx < len(max_forces):
                    force = max_forces[actuator_idx]

                self.actuated_joint_indices.append(i)
                actuator_idx += 1

            joint = JointInfo(
                index=info[0],
                name=info[1].decode("utf-8"),
                joint_type=joint_type,
                lower_limit=float(low_lim),
                upper_limit=float(high_lim),
                max_force=float(force),
                max_velocity=info[11] if info[11] > 0 else 10.0,
                link_name=info[12].decode("utf-8"),
            )
            self.joints.append(joint)

    @property
    def num_actuators(self) -> int:
        return len(self.actuated_joint_indices)

    def apply_action_positions(
        self,
        target_positions: np.ndarray,
        forces: Optional[List[float]] = None
    ) -> None:
        """
        Apply normalized actions [-1, 1] mapped strictly to joint lower/upper limits.
        """
        assert len(target_positions) == self.num_actuators, (
            f"Expected {self.num_actuators} actions, got {len(target_positions)}"
        )

        for idx, joint_idx in enumerate(self.actuated_joint_indices):
            joint = self.joints[joint_idx]
            low, high = joint.lower_limit, joint.upper_limit
            
            # Map [-1, 1] to [low, high]
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
        """Returns (position, orientation_quaternion, linear_velocity, angular_velocity)."""
        pos, orn = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.body_id, physicsClientId=self.client_id)
        return (
            np.array(pos, dtype=np.float32),
            np.array(orn, dtype=np.float32),
            np.array(lin_vel, dtype=np.float32),
            np.array(ang_vel, dtype=np.float32),
        )

    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (joint_positions, joint_velocities) for all actuated joints."""
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
    """Builder for baseline creatures."""
    @staticmethod
    def create_quadruped(
        client_id: int,
        start_pos: Tuple[float, float, float] = (0.0, 0.0, 0.5),
        torso_size: Tuple[float, float, float] = (0.45, 0.28, 0.12),
        leg_length: float = 0.24,
        leg_radius: float = 0.035,
    ) -> Creature:
        from morphology.genome import CreatureGenome
        return CreatureGenome.preset_quadruped().build(client_id=client_id, start_pos=start_pos)
