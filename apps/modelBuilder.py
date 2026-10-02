"""
Musubi URDF Studio & Creature Builder Desktop Application.
Interactive PySide6 GUI to design custom creatures, tweak advanced robotics parameters,
preview in 3D PyBullet, export standard XML URDFs, and launch RL training.
"""

import copy
import math
import os
import subprocess
import sys
import threading
import time
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pybullet as p
import pybullet_data
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from morphology.genome import CreatureGenome, LegGene, LimbGene
from morphology.urdf_exporter import URDFExporter


class URDFBuilderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Musubi 🍙 — URDF Studio & Creature Designer")
        self.resize(1100, 750)

        # Simulation state
        self.pybullet_client_id: int = -1
        self.current_creature = None
        self.animating = False
        self.anim_thread = None

        self._init_ui()
        self._apply_dark_theme()
        self._load_preset("Quadruped")

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # ==========================================
        # LEFT PANEL: Parameters & Controls (Scrollable)
        # ==========================================
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(460)

        panel_widget = QWidget()
        panel_layout = QVBoxLayout(panel_widget)
        panel_layout.setSpacing(12)

        # 1. Preset Selector
        preset_box = QGroupBox("🧬 Morphology Presets")
        preset_layout = QHBoxLayout(preset_box)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Quadruped (4 Legs)", "Hexapod (6 Legs)", "Biped (2 Legs)", "Centipede (8 Legs)", "Custom"])
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(QLabel("Preset:"))
        preset_layout.addWidget(self.preset_combo)
        panel_layout.addWidget(preset_box)

        # 2. Torso Configuration
        torso_box = QGroupBox("📦 Torso / Chassis Settings")
        torso_form = QFormLayout(torso_box)

        self.torso_shape_combo = QComboBox()
        self.torso_shape_combo.addItems(["box", "cylinder"])
        torso_form.addRow("Shape:", self.torso_shape_combo)

        self.torso_len_spin = self._create_double_spin(0.1, 2.0, 0.45, 0.05)
        self.torso_width_spin = self._create_double_spin(0.1, 2.0, 0.28, 0.05)
        self.torso_height_spin = self._create_double_spin(0.05, 1.0, 0.12, 0.02)
        torso_form.addRow("Length / Radius (m):", self.torso_len_spin)
        torso_form.addRow("Width (m):", self.torso_width_spin)
        torso_form.addRow("Height (m):", self.torso_height_spin)

        self.torso_mass_spin = self._create_double_spin(0.5, 50.0, 2.5, 0.5)
        torso_form.addRow("Mass (kg):", self.torso_mass_spin)
        panel_layout.addWidget(torso_box)

        # 3. Leg Topology Configuration
        leg_box = QGroupBox("🦵 Leg Topology & Anatomy")
        leg_form = QFormLayout(leg_box)

        self.num_legs_spin = QSpinBox()
        self.num_legs_spin.setRange(2, 8)
        self.num_legs_spin.setSingleStep(2)
        self.num_legs_spin.setValue(4)
        leg_form.addRow("Total Legs:", self.num_legs_spin)

        self.upper_len_spin = self._create_double_spin(0.1, 1.0, 0.24, 0.02)
        self.upper_rad_spin = self._create_double_spin(0.01, 0.1, 0.035, 0.005)
        self.upper_mass_spin = self._create_double_spin(0.1, 10.0, 0.4, 0.1)
        leg_form.addRow("Upper Leg Length (m):", self.upper_len_spin)
        leg_form.addRow("Upper Leg Radius (m):", self.upper_rad_spin)
        leg_form.addRow("Upper Leg Mass (kg):", self.upper_mass_spin)

        self.lower_len_spin = self._create_double_spin(0.1, 1.0, 0.24, 0.02)
        self.lower_rad_spin = self._create_double_spin(0.01, 0.1, 0.028, 0.005)
        self.lower_mass_spin = self._create_double_spin(0.1, 10.0, 0.3, 0.1)
        leg_form.addRow("Lower Leg Length (m):", self.lower_len_spin)
        leg_form.addRow("Lower Leg Radius (m):", self.lower_rad_spin)
        leg_form.addRow("Lower Leg Mass (kg):", self.lower_mass_spin)

        self.motor_force_spin = self._create_double_spin(5.0, 500.0, 35.0, 5.0)
        leg_form.addRow("Motor Max Force (N):", self.motor_force_spin)

        panel_layout.addWidget(leg_box)
        panel_layout.addStretch()

        scroll.setWidget(panel_widget)
        main_layout.addWidget(scroll, 1)

        # ==========================================
        # RIGHT PANEL: 3D Viewport Controls & Actions
        # ==========================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(12)

        # 1. Info & Telemetry Box
        info_box = QGroupBox("📊 Robot Telemetry & Specifications")
        info_layout = QVBoxLayout(info_box)
        self.telemetry_label = QLabel("Click 'Spawn / Preview in 3D' to inspect.")
        self.telemetry_label.setStyleSheet("font-size: 13px; line-height: 1.4;")
        info_layout.addWidget(self.telemetry_label)
        right_layout.addWidget(info_box)

        # 2. Action Buttons
        btn_box = QGroupBox("🚀 Actions & Simulation")
        btn_layout = QVBoxLayout(btn_box)
        btn_layout.setSpacing(10)

        self.btn_preview = QPushButton("🌐 1. Spawn / Preview in 3D (PyBullet GUI)")
        self.btn_preview.setStyleSheet("background-color: #007acc; font-weight: bold; padding: 10px;")
        self.btn_preview.clicked.connect(self._preview_in_pybullet)
        btn_layout.addWidget(self.btn_preview)

        self.btn_animate = QPushButton("🎮 2. Test Joint Flex Motion")
        self.btn_animate.setStyleSheet("background-color: #2ea44f; font-weight: bold; padding: 10px;")
        self.btn_animate.clicked.connect(self._toggle_animation)
        btn_layout.addWidget(self.btn_animate)

        self.btn_export = QPushButton("💾 3. Export as Standard .URDF")
        self.btn_export.setStyleSheet("background-color: #6f42c1; font-weight: bold; padding: 10px;")
        self.btn_export.clicked.connect(self._export_urdf)
        btn_layout.addWidget(self.btn_export)

        self.btn_train = QPushButton("⚡ 4. Launch RL Locomotion Training")
        self.btn_train.setStyleSheet("background-color: #d93f0b; font-weight: bold; padding: 10px;")
        self.btn_train.clicked.connect(self._launch_training)
        btn_layout.addWidget(self.btn_train)

        right_layout.addWidget(btn_box)

        # 3. Log Console
        log_box = QGroupBox("📝 Console Output")
        log_layout = QVBoxLayout(log_box)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #1e1e1e; font-family: Consolas, monospace; font-size: 12px;")
        log_layout.addWidget(self.console)
        right_layout.addWidget(log_box)

        main_layout.addWidget(right_panel, 1)

    def _create_double_spin(self, min_v, max_v, def_v, step):
        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setValue(def_v)
        spin.setSingleStep(step)
        spin.setDecimals(3)
        return spin

    def _on_preset_changed(self, text):
        if "Quadruped" in text:
            self._load_preset("Quadruped")
        elif "Hexapod" in text:
            self._load_preset("Hexapod")
        elif "Biped" in text:
            self._load_preset("Biped")
        elif "Centipede" in text:
            self._load_preset("Centipede")

    def _load_preset(self, preset_name):
        if preset_name == "Quadruped":
            self.torso_shape_combo.setCurrentText("box")
            self.torso_len_spin.setValue(0.48)
            self.torso_width_spin.setValue(0.30)
            self.torso_height_spin.setValue(0.12)
            self.torso_mass_spin.setValue(2.5)
            self.num_legs_spin.setValue(4)
            self.upper_len_spin.setValue(0.24)
            self.lower_len_spin.setValue(0.24)
        elif preset_name == "Hexapod":
            self.torso_shape_combo.setCurrentText("cylinder")
            self.torso_len_spin.setValue(0.28)  # Radius
            self.torso_width_spin.setValue(0.28)
            self.torso_height_spin.setValue(0.10)
            self.torso_mass_spin.setValue(3.0)
            self.num_legs_spin.setValue(6)
            self.upper_len_spin.setValue(0.20)
            self.lower_len_spin.setValue(0.22)
        elif preset_name == "Biped":
            self.torso_shape_combo.setCurrentText("box")
            self.torso_len_spin.setValue(0.24)
            self.torso_width_spin.setValue(0.32)
            self.torso_height_spin.setValue(0.40)
            self.torso_mass_spin.setValue(3.0)
            self.num_legs_spin.setValue(2)
            self.upper_len_spin.setValue(0.28)
            self.lower_len_spin.setValue(0.28)
        elif preset_name == "Centipede":
            self.torso_shape_combo.setCurrentText("box")
            self.torso_len_spin.setValue(0.70)
            self.torso_width_spin.setValue(0.26)
            self.torso_height_spin.setValue(0.10)
            self.torso_mass_spin.setValue(3.5)
            self.num_legs_spin.setValue(8)
            self.upper_len_spin.setValue(0.22)
            self.lower_len_spin.setValue(0.22)

    def _build_genome_from_gui(self) -> CreatureGenome:
        """Constructs a CreatureGenome instance directly from GUI spinbox values."""
        shape = self.torso_shape_combo.currentText()
        torso_size = (
            self.torso_len_spin.value(),
            self.torso_width_spin.value(),
            self.torso_height_spin.value(),
        )
        torso_mass = self.torso_mass_spin.value()
        num_legs = self.num_legs_spin.value()

        u_len = self.upper_len_spin.value()
        u_rad = self.upper_rad_spin.value()
        u_mass = self.upper_mass_spin.value()

        l_len = self.lower_len_spin.value()
        l_rad = self.lower_rad_spin.value()
        l_mass = self.lower_mass_spin.value()

        force = self.motor_force_spin.value()

        legs = []

        if shape == "cylinder" and num_legs == 6:
            # Radial Hexagonal Spacing
            radius = torso_size[0]
            angles_deg = [30, 90, 150, 210, 270, 330]
            for angle in angles_deg:
                rad = math.radians(angle)
                mx, my = radius * math.cos(rad), radius * math.sin(rad)
                segs = [
                    LimbGene(length=u_len, radius=u_rad, mass=u_mass, max_force=force,
                             lower_limit=-np.pi/6, upper_limit=np.pi/6, color=(0.85, 0.35, 0.25, 1.0)),
                    LimbGene(length=l_len, radius=l_rad, mass=l_mass, max_force=force,
                             lower_limit=0.0, upper_limit=np.pi/2.5, color=(0.95, 0.80, 0.20, 1.0)),
                ]
                legs.append(LegGene(segments=segs, mount_point=(mx, my, 0.0), mount_orientation_rpy=(0.0, 0.0, rad)))
        elif num_legs == 2:
            # Biped
            hy, hz = torso_size[1] / 2, torso_size[2] / 2
            for side in [1.0, -1.0]:
                segs = [
                    LimbGene(length=u_len, radius=u_rad, mass=u_mass, max_force=force,
                             lower_limit=-np.pi/3, upper_limit=np.pi/3, color=(0.85, 0.35, 0.2, 1.0)),
                    LimbGene(length=l_len, radius=l_rad, mass=l_mass, max_force=force,
                             lower_limit=-np.pi/4, upper_limit=np.pi/3, color=(0.95, 0.75, 0.2, 1.0)),
                ]
                legs.append(LegGene(segments=segs, mount_point=(0.0, side * hy * 0.75, -hz)))
        else:
            # Pair-based Quadruped / Centipede
            num_pairs = num_legs // 2
            hx, hy = torso_size[0] / 2, torso_size[1] / 2
            x_offsets = np.linspace(hx * 0.8, -hx * 0.8, num_pairs)

            for ox in x_offsets:
                segs = [
                    LimbGene(length=u_len, radius=u_rad, mass=u_mass, max_force=force,
                             lower_limit=-np.pi/4, upper_limit=np.pi/4, color=(0.9, 0.4, 0.2, 1.0)),
                    LimbGene(length=l_len, radius=l_rad, mass=l_mass, max_force=force,
                             lower_limit=-np.pi/4, upper_limit=np.pi/3, color=(0.95, 0.8, 0.2, 1.0)),
                ]
                left_leg = LegGene(segments=segs, mount_point=(float(ox), float(hy), 0.0))
                right_leg = LegGene(segments=copy.deepcopy(segs), mount_point=(float(ox), float(-hy), 0.0))
                legs.extend([left_leg, right_leg])

        return CreatureGenome(
            name=f"Custom_{num_legs}Legs",
            torso_shape_type=shape,
            torso_size=torso_size,
            torso_mass=torso_mass,
            torso_color=(0.2, 0.6, 0.86, 1.0),
            legs=legs,
            symmetric=True,
        )

    def _preview_in_pybullet(self):
        """Spawns the designed robot inside a live PyBullet 3D GUI window."""
        self._ensure_pybullet_connected()
        genome = self._build_genome_from_gui()

        p.resetSimulation(physicsClientId=self.pybullet_client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.pybullet_client_id)
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.pybullet_client_id)
        p.changeDynamics(plane_id, -1, lateralFriction=1.2, physicsClientId=self.pybullet_client_id)

        start_z = 0.6 if genome.num_legs != 2 else 0.9
        self.current_creature = genome.build(self.pybullet_client_id, start_pos=(0, 0, start_z))

        # Reset camera
        p.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=[0, 0, 0.35],
            physicsClientId=self.pybullet_client_id,
        )

        # Update telemetry
        total_mass = genome.torso_mass + sum(seg.mass for leg in genome.legs for seg in leg.segments)
        self.telemetry_label.setText(
            f"<b>Model Name:</b> {genome.name}<br>"
            f"<b>Total Controllable Joints:</b> {genome.total_actuators}<br>"
            f"<b>Total Multi-Body Links:</b> {genome.total_actuators + 1}<br>"
            f"<b>Total Assembly Mass:</b> {total_mass:.2f} kg<br>"
            f"<b>Torso Dimensions:</b> {genome.torso_size}<br>"
            f"<b>Status:</b> <span style='color: #4cd137;'>Loaded in PyBullet 3D</span>"
        )
        self.console.append(f"✅ Spawned {genome.name} ({genome.total_actuators} joints, {total_mass:.2f} kg) in 3D Viewport.")

    def _ensure_pybullet_connected(self):
        if self.pybullet_client_id < 0 or not p.isConnected(physicsClientId=self.pybullet_client_id):
            self.pybullet_client_id = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.pybullet_client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.pybullet_client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.pybullet_client_id)

    def _toggle_animation(self):
        if self.animating:
            self.animating = False
            self.btn_animate.setText("🎮 2. Test Joint Flex Motion")
            self.console.append("⏸ Joint flex motion paused.")
        else:
            if not self.current_creature or not p.isConnected(physicsClientId=self.pybullet_client_id):
                self._preview_in_pybullet()

            self.animating = True
            self.btn_animate.setText("❚❚ Pause Motion")
            self.console.append("▶ Playing joint flex motion test...")
            threading.Thread(target=self._run_joint_animation, daemon=True).start()

    def _run_joint_animation(self):
        step = 0
        while self.animating and p.isConnected(physicsClientId=self.pybullet_client_id):
            t = step * 0.06
            num = self.current_creature.num_actuators
            actions = np.array([np.sin(t + i * 0.5) * 0.75 for i in range(num)], dtype=np.float32)
            self.current_creature.apply_action_positions(actions)
            p.stepSimulation(physicsClientId=self.pybullet_client_id)
            time.sleep(1.0 / 60.0)
            step += 1

    def _export_urdf(self):
        genome = self._build_genome_from_gui()
        default_name = f"{genome.name.lower()}.urdf"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Robot as URDF",
            os.path.join(os.getcwd(), "models_exported", default_name),
            "URDF Files (*.urdf);;All Files (*)",
        )

        if file_path:
            saved_path = URDFExporter.export_genome(genome, file_path)
            self.console.append(f"💾 Successfully exported URDF to:\n   {saved_path}")
            QMessageBox.information(self, "Export Successful", f"URDF file generated successfully:\n\n{saved_path}")

    def _launch_training(self):
        genome = self._build_genome_from_gui()
        reply = QMessageBox.question(
            self,
            "Launch RL Training",
            f"Do you want to train an RL locomotion policy for this {genome.num_legs}-legged creature ({genome.total_actuators} joints)?\n\n"
            "This will start headless multi-core PPO training.",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.console.append(f"🚀 Initializing PPO training pipeline for {genome.name}...")
            # Save custom creature DNA JSON
            dna_path = os.path.join(os.getcwd(), "models_exported", f"{genome.name.lower()}_dna.json")
            genome.save_json(dna_path)

            cmd = [
                sys.executable,
                os.path.join(os.getcwd(), "agents", "train_sb3.py"),
                "--creature", dna_path,
                "--timesteps", "300000",
                "--n-envs", "4",
            ]
            self.console.append(f"⚡ Running command: {' '.join(cmd)}")
            subprocess.Popen(cmd)
            QMessageBox.information(
                self,
                "Training Started",
                f"Training process launched in background!\n\nCreature: {genome.name} ({genome.num_legs} legs, {genome.total_actuators} joints)\nTarget Timesteps: 300,000\nWorkers: 4 CPU Cores\nLogs: runs/\nModel: models/{genome.name.lower()}/",
            )

    def _apply_dark_theme(self):
        dark_style = """
        QMainWindow, QWidget {
            background-color: #181824;
            color: #f1f2f6;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QGroupBox {
            border: 1px solid #2f3542;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 15px;
            font-weight: bold;
            color: #70a1ff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            border: none;
            border-radius: 6px;
            color: white;
            font-size: 13px;
            padding: 8px 12px;
        }
        QPushButton:hover {
            opacity: 0.9;
        }
        QComboBox, QDoubleSpinBox, QSpinBox, QTextEdit {
            background-color: #2f3542;
            border: 1px solid #57606f;
            border-radius: 5px;
            padding: 5px;
            color: white;
        }
        QScrollArea {
            border: none;
        }
        """
        self.setStyleSheet(dark_style)

    def closeEvent(self, event):
        self.animating = False
        if self.pybullet_client_id >= 0 and p.isConnected(physicsClientId=self.pybullet_client_id):
            p.disconnect(physicsClientId=self.pybullet_client_id)
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = URDFBuilderApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
