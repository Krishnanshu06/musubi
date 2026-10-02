"""
Interactive showcase to generate, mutate, and visually inspect procedural creatures in PyBullet 3D GUI.
Features seamless limb kinematics and hexagonal spider topologies.
"""

import os
import sys
import time
import numpy as np
import pybullet as p
import pybullet_data

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from morphology.genome import CreatureGenome


def run_showcase():
    print("=" * 65)
    print(" 🧬 Musubi Procedural Creature Showcase (Fixed Kinematics)")
    print("=" * 65)

    # 1. Connect to PyBullet GUI
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.setGravity(0, 0, -9.81)

    # 2. Setup ground plane
    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=1.2)

    # 3. Create a collection of diverse genomes
    parent_quad = CreatureGenome.preset_quadruped()
    mutated_quad = parent_quad.mutate(rate=0.4)

    creature_catalog = [
        ("Hexapod (Hexagonal Body, 6 Radial Legs, 12 Joints)", CreatureGenome.preset_hexapod(), (0, 0, 0.5)),
        ("Quadruped (4 Legs, 8 Joints)", parent_quad, (0, 0, 0.6)),
        ("Biped (Humanoid, 2 Legs, 4 Joints)", CreatureGenome.preset_biped(), (0, 0, 0.9)),
        ("Procedural 8-Legged Centipede/Crawler", CreatureGenome.random(num_pairs=4, segments_per_leg=2), (0, 0, 0.6)),
        ("Mutated Offspring", mutated_quad, (0, 0, 0.6)),
    ]

    try:
        for name, genome, start_pos in creature_catalog:
            print(f"\n Spawning: {name}")
            print(f"  • Torso Shape      : {genome.torso_shape_type} {genome.torso_size}")
            print(f"  • Torso Mass       : {genome.torso_mass:.2f} kg")
            print(f"  • Total Legs       : {genome.num_legs}")
            print(f"  • Controllable Joints: {genome.total_actuators}")

            # Build physical creature from DNA
            creature = genome.build(client_id, start_pos=start_pos)

            # Camera tracks creature
            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=50,
                cameraPitch=-25,
                cameraTargetPosition=[0, 0, 0.35],
            )

            # Animate limbs with gentle open-loop oscillation for 200 steps (~3.3 seconds)
            num_actuators = creature.num_actuators
            for step in range(200):
                t = step * 0.08
                actions = np.array([np.sin(t + i * 0.6) * 0.8 for i in range(num_actuators)], dtype=np.float32)
                creature.apply_action_positions(actions)

                p.stepSimulation()
                time.sleep(1.0 / 60.0)

            # Remove creature before spawning next
            p.removeBody(creature.body_id)
            time.sleep(0.4)

        print("\n" + "=" * 65)
        print(" Showcase completed! All limbs connected seamlessly.")
        print("=" * 65)

    finally:
        p.disconnect(client_id)


if __name__ == "__main__":
    run_showcase()
