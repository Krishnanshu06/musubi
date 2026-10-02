"""
URDF Exporter module for Musubi.
Translates CreatureGenome instances and procedural morphologies into standard XML URDF files.
"""

import math
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Optional, Tuple
import numpy as np

from morphology.genome import CreatureGenome, LegGene, LimbGene


class URDFExporter:
    """
    Exports CreatureGenome structures into valid, standard URDF XML format for simulation and robotics.
    """

    @staticmethod
    def _compute_box_inertia(mass: float, sx: float, sy: float, sz: float) -> Tuple[float, float, float]:
        """Calculates principal moments of inertia for a rectangular box."""
        ixx = (1.0 / 12.0) * mass * (sy**2 + sz**2)
        iyy = (1.0 / 12.0) * mass * (sx**2 + sz**2)
        izz = (1.0 / 12.0) * mass * (sx**2 + sy**2)
        return ixx, iyy, izz

    @staticmethod
    def _compute_cylinder_inertia(mass: float, radius: float, length: float) -> Tuple[float, float, float]:
        """Calculates principal moments of inertia for a cylinder/capsule along Z-axis."""
        ixx = (1.0 / 12.0) * mass * (3 * radius**2 + length**2)
        iyy = ixx
        izz = 0.5 * mass * radius**2
        return ixx, iyy, izz

    @classmethod
    def export_genome(
        cls,
        genome: CreatureGenome,
        output_filepath: str,
        robot_name: Optional[str] = None,
    ) -> str:
        """
        Converts a CreatureGenome into a complete .urdf file.
        """
        robot_name = robot_name or genome.name.replace(" ", "_")
        robot = ET.Element("robot", name=robot_name)

        # ----------------------------------------------------
        # 1. Base / Torso Link
        # ----------------------------------------------------
        torso_link = ET.SubElement(robot, "link", name="torso")

        # Torso Inertial
        inertial = ET.SubElement(torso_link, "inertial")
        ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{genome.torso_mass:.3f}")

        if genome.torso_shape_type == "cylinder":
            radius, _, height = genome.torso_size
            ixx, iyy, izz = cls._compute_cylinder_inertia(genome.torso_mass, radius, height)
        else:
            sx, sy, sz = genome.torso_size
            ixx, iyy, izz = cls._compute_box_inertia(genome.torso_mass, sx, sy, sz)

        ET.SubElement(inertial, "inertia", ixx=f"{ixx:.6f}", ixy="0", ixz="0", iyy=f"{iyy:.6f}", iyz="0", izz=f"{izz:.6f}")

        # Torso Visual
        vis = ET.SubElement(torso_link, "visual")
        ET.SubElement(vis, "origin", xyz="0 0 0", rpy="0 0 0")
        vis_geom = ET.SubElement(vis, "geometry")
        if genome.torso_shape_type == "cylinder":
            ET.SubElement(vis_geom, "cylinder", radius=f"{radius:.3f}", length=f"{height:.3f}")
        else:
            ET.SubElement(vis_geom, "box", size=f"{sx:.3f} {sy:.3f} {sz:.3f}")

        mat = ET.SubElement(vis, "material", name="torso_mat")
        r, g, b, a = genome.torso_color
        ET.SubElement(mat, "color", rgba=f"{r:.2f} {g:.2f} {b:.2f} {a:.2f}")

        # Torso Collision
        col = ET.SubElement(torso_link, "collision")
        ET.SubElement(col, "origin", xyz="0 0 0", rpy="0 0 0")
        col_geom = ET.SubElement(col, "geometry")
        if genome.torso_shape_type == "cylinder":
            ET.SubElement(col_geom, "cylinder", radius=f"{radius:.3f}", length=f"{height:.3f}")
        else:
            ET.SubElement(col_geom, "box", size=f"{sx:.3f} {sy:.3f} {sz:.3f}")

        # ----------------------------------------------------
        # 2. Legs and Joints
        # ----------------------------------------------------
        for leg_i, leg in enumerate(genome.legs):
            parent_name = "torso"

            for seg_j, seg in enumerate(leg.segments):
                link_name = f"leg_{leg_i}_seg_{seg_j}"
                joint_name = f"joint_{leg_i}_{seg_j}"
                half_len = seg.length / 2.0

                # --- Create Link ---
                link_el = ET.SubElement(robot, "link", name=link_name)

                # Link Inertial
                l_inertial = ET.SubElement(link_el, "inertial")
                ET.SubElement(l_inertial, "origin", xyz=f"0 0 {-half_len:.4f}", rpy="0 0 0")
                ET.SubElement(l_inertial, "mass", value=f"{seg.mass:.3f}")
                l_ixx, l_iyy, l_izz = cls._compute_cylinder_inertia(seg.mass, seg.radius, seg.length)
                ET.SubElement(l_inertial, "inertia", ixx=f"{l_ixx:.6f}", ixy="0", ixz="0", iyy=f"{l_iyy:.6f}", iyz="0", izz=f"{l_izz:.6f}")

                # Link Visual
                l_vis = ET.SubElement(link_el, "visual")
                ET.SubElement(l_vis, "origin", xyz=f"0 0 {-half_len:.4f}", rpy="0 0 0")
                l_vis_geom = ET.SubElement(l_vis, "geometry")
                ET.SubElement(l_vis_geom, "cylinder", radius=f"{seg.radius:.3f}", length=f"{seg.length:.3f}")
                l_mat = ET.SubElement(l_vis, "material", name=f"{link_name}_mat")
                lr, lg, lb, la = seg.color
                ET.SubElement(l_mat, "color", rgba=f"{lr:.2f} {lg:.2f} {lb:.2f} {la:.2f}")

                # Link Collision
                l_col = ET.SubElement(link_el, "collision")
                ET.SubElement(l_col, "origin", xyz=f"0 0 {-half_len:.4f}", rpy="0 0 0")
                l_col_geom = ET.SubElement(l_col, "geometry")
                ET.SubElement(l_col_geom, "cylinder", radius=f"{seg.radius:.3f}", length=f"{seg.length:.3f}")

                # --- Create Joint ---
                joint_el = ET.SubElement(robot, "joint", name=joint_name, type="revolute")
                ET.SubElement(joint_el, "parent", link=parent_name)
                ET.SubElement(joint_el, "child", link=link_name)

                if seg_j == 0:
                    # Mount joint to torso
                    mx, my, mz = leg.mount_point
                    mr, mp, myaw = leg.mount_orientation_rpy
                    ET.SubElement(joint_el, "origin", xyz=f"{mx:.4f} {my:.4f} {mz:.4f}", rpy=f"{mr:.4f} {mp:.4f} {myaw:.4f}")
                else:
                    # Mount joint to previous segment tip
                    prev_len = leg.segments[seg_j - 1].length
                    ET.SubElement(joint_el, "origin", xyz=f"0 0 {-prev_len:.4f}", rpy="0 0 0")

                jx, jy, jz = seg.joint_axis
                ET.SubElement(joint_el, "axis", xyz=f"{jx} {jy} {jz}")
                ET.SubElement(
                    joint_el,
                    "limit",
                    lower=f"{seg.lower_limit:.4f}",
                    upper=f"{seg.upper_limit:.4f}",
                    effort=f"{seg.max_force:.1f}",
                    velocity="10.0",
                )
                ET.SubElement(joint_el, "dynamics", damping="0.05", friction="0.1")

                parent_name = link_name

        # ----------------------------------------------------
        # 3. Format & Write to File
        # ----------------------------------------------------
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        raw_xml = ET.tostring(robot, encoding="utf-8")
        pretty_xml = minidom.parseString(raw_xml).toprettyxml(indent="  ")

        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

        return os.path.abspath(output_filepath)
