"""
Genome and DNA specification for procedural creature generation, mutation, and evolution.
Features correct 3D joint kinematic axes (Yaw + Pitch) and anatomical joint limits to prevent limb overlap.
"""

from dataclasses import dataclass, field
import copy
import math
import random
from typing import List, Tuple, Optional
import numpy as np
import pybullet as p

from morphology.builder import Creature


@dataclass
class LimbGene:
    """Represents the DNA of a single limb segment (bone)."""
    length: float = 0.25
    radius: float = 0.035
    mass: float = 0.3
    joint_axis: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    lower_limit: float = -np.pi / 4
    upper_limit: float = np.pi / 4
    max_force: float = 35.0
    color: Tuple[float, float, float, float] = (0.9, 0.4, 0.2, 1.0)

    def to_dict(self) -> dict:
        return {
            "length": self.length,
            "radius": self.radius,
            "mass": self.mass,
            "joint_axis": list(self.joint_axis),
            "lower_limit": self.lower_limit,
            "upper_limit": self.upper_limit,
            "max_force": self.max_force,
            "color": list(self.color),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LimbGene":
        return cls(
            length=data.get("length", 0.25),
            radius=data.get("radius", 0.035),
            mass=data.get("mass", 0.3),
            joint_axis=tuple(data.get("joint_axis", [0.0, 1.0, 0.0])),
            lower_limit=data.get("lower_limit", -np.pi / 4),
            upper_limit=data.get("upper_limit", np.pi / 4),
            max_force=data.get("max_force", 35.0),
            color=tuple(data.get("color", [0.9, 0.4, 0.2, 1.0])),
        )


@dataclass
class LegGene:
    """Represents a complete leg with one or more articulated limb segments."""
    segments: List[LimbGene] = field(default_factory=list)
    mount_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (X, Y, Z) offset on torso
    mount_orientation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Roll, Pitch, Yaw mount angle

    def to_dict(self) -> dict:
        return {
            "segments": [seg.to_dict() for seg in self.segments],
            "mount_point": list(self.mount_point),
            "mount_orientation_rpy": list(self.mount_orientation_rpy),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LegGene":
        segments = [LimbGene.from_dict(s) for s in data.get("segments", [])]
        return cls(
            segments=segments,
            mount_point=tuple(data.get("mount_point", [0.0, 0.0, 0.0])),
            mount_orientation_rpy=tuple(data.get("mount_orientation_rpy", [0.0, 0.0, 0.0])),
        )

    def mutate(self, rate: float = 0.2) -> "LegGene":
        """Mutates all limb segments in the leg."""
        mutated_segments = [seg.mutate(rate=rate) for seg in self.segments]
        return LegGene(
            segments=mutated_segments,
            mount_point=self.mount_point,
            mount_orientation_rpy=self.mount_orientation_rpy,
        )


@dataclass
class CreatureGenome:
    """
    Complete creature DNA encoding torso shape, leg configurations, and kinematic symmetry.
    """
    name: str = "Creature"
    torso_shape_type: str = "box"  # 'box' or 'cylinder'
    torso_size: Tuple[float, float, float] = (0.4, 0.3, 0.1)
    torso_mass: float = 2.0
    torso_color: Tuple[float, float, float, float] = (0.2, 0.6, 0.86, 1.0)
    legs: List[LegGene] = field(default_factory=list)
    symmetric: bool = True

    @property
    def num_legs(self) -> int:
        return len(self.legs)

    @property
    def total_actuators(self) -> int:
        return sum(len(leg.segments) for leg in self.legs)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "torso_shape_type": self.torso_shape_type,
            "torso_size": list(self.torso_size),
            "torso_mass": self.torso_mass,
            "torso_color": list(self.torso_color),
            "legs": [leg.to_dict() for leg in self.legs],
            "symmetric": self.symmetric,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CreatureGenome":
        legs = [LegGene.from_dict(l) for l in data.get("legs", [])]
        return cls(
            name=data.get("name", "Creature"),
            torso_shape_type=data.get("torso_shape_type", "box"),
            torso_size=tuple(data.get("torso_size", [0.4, 0.3, 0.1])),
            torso_mass=data.get("torso_mass", 2.0),
            torso_color=tuple(data.get("torso_color", [0.2, 0.6, 0.86, 1.0])),
            legs=legs,
            symmetric=data.get("symmetric", True),
        )

    def save_json(self, filepath: str) -> str:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return os.path.abspath(filepath)

    @classmethod
    def from_json(cls, filepath: str) -> "CreatureGenome":
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def mutate(self, rate: float = 0.2) -> "CreatureGenome":
        """Creates a mutated offspring genome with randomized variations."""
        new_torso_size = list(self.torso_size)
        if random.random() < rate:
            new_torso_size[0] = float(np.clip(new_torso_size[0] * (1.0 + np.random.normal(0, 0.1)), 0.2, 0.7))
            new_torso_size[1] = float(np.clip(new_torso_size[1] * (1.0 + np.random.normal(0, 0.1)), 0.15, 0.5))

        new_torso_mass = self.torso_mass
        if random.random() < rate:
            new_torso_mass = float(np.clip(new_torso_mass * (1.0 + np.random.normal(0, 0.15)), 0.8, 5.0))

        mutated_legs = [leg.mutate(rate=rate) for leg in self.legs]

        return CreatureGenome(
            name=f"{self.name}_mutated",
            torso_shape_type=self.torso_shape_type,
            torso_size=tuple(new_torso_size),
            torso_mass=new_torso_mass,
            torso_color=self.torso_color,
            legs=mutated_legs,
            symmetric=self.symmetric,
        )

    def crossover(self, other: "CreatureGenome") -> "CreatureGenome":
        """Breeds two parent genomes to create an offspring combining traits from both."""
        blend = random.random()
        child_torso = tuple(
            self.torso_size[i] * blend + other.torso_size[i] * (1.0 - blend)
            for i in range(3)
        )
        child_mass = self.torso_mass * blend + other.torso_mass * (1.0 - blend)

        child_legs = []
        min_legs = min(len(self.legs), len(other.legs))
        for i in range(min_legs):
            chosen_leg = self.legs[i] if random.random() < 0.5 else other.legs[i]
            child_legs.append(copy.deepcopy(chosen_leg))

        return CreatureGenome(
            name=f"Hybrid_{self.name[:4]}_{other.name[:4]}",
            torso_shape_type=self.torso_shape_type,
            torso_size=child_torso,
            torso_mass=child_mass,
            torso_color=self.torso_color,
            legs=child_legs,
            symmetric=self.symmetric,
        )

    @classmethod
    def random(
        cls,
        num_pairs: int = 2,
        segments_per_leg: int = 2,
        symmetric: bool = True,
    ) -> "CreatureGenome":
        """Generates a randomized quadruped/crawler creature genome with non-overlapping bounds."""
        torso_len = random.uniform(0.40, 0.60)
        torso_width = random.uniform(0.25, 0.35)
        torso_height = random.uniform(0.08, 0.12)
        torso_mass = random.uniform(1.8, 3.0)

        r, g, b = random.uniform(0.2, 0.9), random.uniform(0.2, 0.9), random.uniform(0.2, 0.9)

        legs: List[LegGene] = []
        hx, hy = torso_len / 2.0, torso_width / 2.0
        x_offsets = np.linspace(hx * 0.8, -hx * 0.8, num_pairs)

        for ox in x_offsets:
            segments = [
                # Upper Leg (Pitch joint: lifts vertically)
                LimbGene(
                    length=random.uniform(0.20, 0.26),
                    radius=0.035,
                    mass=0.35,
                    joint_axis=(0.0, 1.0, 0.0),
                    lower_limit=-np.pi / 4,
                    upper_limit=np.pi / 4,
                    color=(r, g, b, 1.0),
                ),
                # Lower Leg (Knee joint: bends down)
                LimbGene(
                    length=random.uniform(0.20, 0.26),
                    radius=0.028,
                    mass=0.25,
                    joint_axis=(0.0, 1.0, 0.0),
                    lower_limit=-np.pi / 4,
                    upper_limit=np.pi / 3,
                    color=(min(r + 0.2, 1.0), min(g + 0.2, 1.0), min(b + 0.2, 1.0), 1.0),
                ),
            ]

            # Left Leg (+Y edge)
            left_leg = LegGene(segments=segments, mount_point=(float(ox), float(hy), 0.0))
            # Right Leg (-Y edge)
            right_leg = copy.deepcopy(left_leg)
            right_leg.mount_point = (float(ox), float(-hy), 0.0)

            legs.extend([left_leg, right_leg])

        return cls(
            name=f"Procedural_{num_pairs * 2}Legs",
            torso_shape_type="box",
            torso_size=(torso_len, torso_width, torso_height),
            torso_mass=torso_mass,
            torso_color=(r, g, b, 1.0),
            legs=legs,
            symmetric=symmetric,
        )

    @classmethod
    def preset_biped(cls) -> "CreatureGenome":
        """2-legged humanoid walker."""
        torso_size = (0.24, 0.32, 0.40)
        hx, hy, hz = torso_size[0] / 2, torso_size[1] / 2, torso_size[2] / 2
        legs = []

        for side in [1.0, -1.0]:  # Left (+1) and Right (-1)
            segments = [
                # Upper Leg (Thigh)
                LimbGene(length=0.28, radius=0.04, mass=0.5, joint_axis=(0.0, 1.0, 0.0),
                         lower_limit=-np.pi/3, upper_limit=np.pi/3, color=(0.85, 0.35, 0.2, 1.0)),
                # Lower Leg (Shin)
                LimbGene(length=0.28, radius=0.032, mass=0.35, joint_axis=(0.0, 1.0, 0.0),
                         lower_limit=-np.pi/4, upper_limit=np.pi/3, color=(0.95, 0.75, 0.2, 1.0)),
            ]
            legs.append(LegGene(segments=segments, mount_point=(0.0, side * hy * 0.75, -hz)))

        return cls(
            name="Biped_Walker",
            torso_shape_type="box",
            torso_size=torso_size,
            torso_mass=3.0,
            torso_color=(0.2, 0.5, 0.85, 1.0),
            legs=legs,
            symmetric=True,
        )

    @classmethod
    def preset_quadruped(cls) -> "CreatureGenome":
        """4-legged quadruped walker with well-spaced corner legs."""
        torso_size = (0.48, 0.30, 0.12)
        hx, hy = torso_size[0] / 2, torso_size[1] / 2
        legs = []

        mount_positions = [
            (hx * 0.85, hy),    # Front-Left
            (hx * 0.85, -hy),   # Front-Right
            (-hx * 0.85, hy),   # Back-Left
            (-hx * 0.85, -hy),  # Back-Right
        ]

        for ox, oy in mount_positions:
            segments = [
                # Upper Leg (Hip pitch)
                LimbGene(length=0.24, radius=0.035, mass=0.4, joint_axis=(0.0, 1.0, 0.0),
                         lower_limit=-np.pi/4, upper_limit=np.pi/4, color=(0.9, 0.4, 0.2, 1.0)),
                # Lower Leg (Knee pitch)
                LimbGene(length=0.24, radius=0.028, mass=0.3, joint_axis=(0.0, 1.0, 0.0),
                         lower_limit=-np.pi/4, upper_limit=np.pi/3, color=(0.95, 0.8, 0.2, 1.0)),
            ]
            legs.append(LegGene(segments=segments, mount_point=(ox, oy, 0.0)))

        return cls(
            name="Quadruped_Runner",
            torso_shape_type="box",
            torso_size=torso_size,
            torso_mass=2.5,
            torso_color=(0.15, 0.65, 0.85, 1.0),
            legs=legs,
            symmetric=True,
        )

    @classmethod
    def preset_hexapod(cls) -> "CreatureGenome":
        """
        True Hexagonal 6-legged insect/spider creature.
        Legs are equally spaced at 60-degree increments around a circular chassis.
        Joint 1 = Hip Yaw (swings forward/backward within safe sector)
        Joint 2 = Knee Pitch (lifts up/down vertically without clipping neighbor)
        """
        radius = 0.28
        height = 0.10
        torso_size = (radius, radius, height)
        legs = []

        # 6 radial angles at 60-degree increments
        angles_deg = [30, 90, 150, 210, 270, 330]

        for angle in angles_deg:
            rad = math.radians(angle)
            mount_x = radius * math.cos(rad)
            mount_y = radius * math.sin(rad)

            segments = [
                # Segment 1: Hip (Pitches vertically in its radial sector)
                LimbGene(
                    length=0.20,
                    radius=0.032,
                    mass=0.30,
                    joint_axis=(0.0, 1.0, 0.0),
                    lower_limit=-np.pi / 6,  # -30 deg
                    upper_limit=np.pi / 6,   # +30 deg
                    color=(0.85, 0.35, 0.25, 1.0),
                ),
                # Segment 2: Knee (Bends strictly down toward ground)
                LimbGene(
                    length=0.22,
                    radius=0.026,
                    mass=0.20,
                    joint_axis=(0.0, 1.0, 0.0),
                    lower_limit=0.0,         # 0 deg (straight)
                    upper_limit=np.pi / 2.5, # +72 deg (bent downwards)
                    color=(0.95, 0.80, 0.20, 1.0),
                ),
            ]
            legs.append(
                LegGene(
                    segments=segments,
                    mount_point=(mount_x, mount_y, 0.0),
                    mount_orientation_rpy=(0.0, 0.0, rad),
                )
            )

        return cls(
            name="Hexapod_Spider",
            torso_shape_type="cylinder",
            torso_size=torso_size,
            torso_mass=3.0,
            torso_color=(0.6, 0.2, 0.85, 1.0),
            legs=legs,
            symmetric=True,
        )

    def build(
        self,
        client_id: int,
        start_pos: Tuple[float, float, float] = (0.0, 0.0, 0.6),
    ) -> Creature:
        """
        Compiles the genome into a seamless 3D multi-body robot in PyBullet.
        """
        # 1. Create Torso Visual & Collision Shapes
        if self.torso_shape_type == "cylinder":
            radius, _, height = self.torso_size
            col_torso = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client_id
            )
            vis_torso = p.createVisualShape(
                p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=list(self.torso_color), physicsClientId=client_id
            )
        else:
            hx, hy, hz = [s / 2.0 for s in self.torso_size]
            col_torso = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=client_id
            )
            vis_torso = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=list(self.torso_color), physicsClientId=client_id
            )

        link_masses = []
        link_col_indices = []
        link_vis_indices = []
        link_positions = []
        link_orientations = []
        link_parent_indices = []
        link_joint_types = []
        link_joint_axes = []
        link_inertial_positions = []
        link_inertial_orientations = []

        total_links = 0

        for leg in self.legs:
            parent_link_idx = 0
            mount_quat = p.getQuaternionFromEuler(leg.mount_orientation_rpy)

            for seg_idx, seg in enumerate(leg.segments):
                total_links += 1
                current_link_idx = total_links
                half_len = seg.length / 2.0

                col_shape = p.createCollisionShape(
                    p.GEOM_CAPSULE,
                    radius=seg.radius,
                    height=seg.length,
                    collisionFramePosition=[0.0, 0.0, -half_len],
                    physicsClientId=client_id,
                )
                vis_shape = p.createVisualShape(
                    p.GEOM_CAPSULE,
                    radius=seg.radius,
                    length=seg.length,
                    rgbaColor=list(seg.color),
                    visualFramePosition=[0.0, 0.0, -half_len],
                    physicsClientId=client_id,
                )

                link_masses.append(seg.mass)
                link_col_indices.append(col_shape)
                link_vis_indices.append(vis_shape)
                link_joint_types.append(p.JOINT_REVOLUTE)
                link_joint_axes.append(list(seg.joint_axis))
                link_inertial_positions.append([0.0, 0.0, -half_len])
                link_inertial_orientations.append([0.0, 0.0, 0.0, 1.0])

                if seg_idx == 0:
                    link_positions.append(list(leg.mount_point))
                    link_orientations.append(list(mount_quat))
                    link_parent_indices.append(0)
                else:
                    prev_seg = leg.segments[seg_idx - 1]
                    link_positions.append([0.0, 0.0, -prev_seg.length])
                    link_orientations.append([0.0, 0.0, 0.0, 1.0])
                    link_parent_indices.append(parent_link_idx)

                parent_link_idx = current_link_idx

        # Assemble the full kinematic robot in PyBullet
        body_id = p.createMultiBody(
            baseMass=self.torso_mass,
            baseCollisionShapeIndex=col_torso,
            baseVisualShapeIndex=vis_torso,
            basePosition=list(start_pos),
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_col_indices,
            linkVisualShapeIndices=link_vis_indices,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_positions,
            linkInertialFrameOrientations=link_inertial_orientations,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes,
            physicsClientId=client_id,
        )

        joint_limits = [
            (seg.lower_limit, seg.upper_limit)
            for leg in self.legs
            for seg in leg.segments
        ]
        max_forces = [
            seg.max_force
            for leg in self.legs
            for seg in leg.segments
        ]

        return Creature(
            body_id=body_id,
            client_id=client_id,
            joint_limits=joint_limits,
            max_forces=max_forces,
        )
