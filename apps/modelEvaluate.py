"""
Musubi Model Player & Evaluation Dashboard.
Interactive PySide6 GUI to discover all trained RL models, inspect their creature DNA,
and launch live 3D policy rollouts in PyBullet with real-time performance telemetry.
"""

import json
import os
import sys
import threading
import time
from typing import Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pybullet as p
import pybullet_data
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from stable_baselines3 import PPO

from envs.locomotion_env import LocomotionEnv
from morphology.genome import CreatureGenome


class TelemetrySignals(QObject):
    log_signal = Signal(str)
    stats_signal = Signal(str)
    finished_signal = Signal()


class EvalPlayerApp(QMainWindow):
    def __init__(self, models_dir: str = "models"):
        super().__init__()
        self.models_dir = os.path.abspath(models_dir)
        self.setWindowTitle("Musubi 🍙 — Trained Policy Player & Evaluation Dashboard")
        self.resize(1050, 700)

        self.discovered_models: List[Dict] = []
        self.selected_model_info: Optional[Dict] = None
        self.signals = TelemetrySignals()
        self.signals.log_signal.connect(self._append_log)
        self.signals.stats_signal.connect(self._update_stats_display)
        self.signals.finished_signal.connect(self._on_eval_finished)

        self.is_running_eval = False

        self._init_ui()
        self._apply_dark_theme()
        self._refresh_models_list()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # ==========================================
        # LEFT PANEL: Discovered Models Table
        # ==========================================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)

        header_layout = QHBoxLayout()
        header_label = QLabel("📂 Trained Models Catalog")
        header_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #70a1ff;")
        header_layout.addWidget(header_label)

        btn_refresh = QPushButton("🔄 Refresh")
        btn_refresh.setStyleSheet("background-color: #2f3542; padding: 5px 10px;")
        btn_refresh.clicked.connect(self._refresh_models_list)
        header_layout.addWidget(btn_refresh)
        left_layout.addLayout(header_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Experiment / Model", "Creature", "Legs", "Joints"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_model_selected)
        left_layout.addWidget(self.table)

        main_layout.addWidget(left_panel, 3)

        # ==========================================
        # RIGHT PANEL: Specs, Controls & Live Playback
        # ==========================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(12)

        # 1. Creature & Checkpoint Info Box
        info_box = QGroupBox("🧬 Selected Model Specifications")
        info_layout = QVBoxLayout(info_box)
        self.info_label = QLabel("Select a model from the list on the left to inspect.")
        self.info_label.setStyleSheet("font-size: 12px; line-height: 1.5;")
        info_layout.addWidget(self.info_label)
        right_layout.addWidget(info_box)

        # 2. Evaluation Settings Box
        settings_box = QGroupBox("⚙️ Playback Settings")
        settings_form = QFormLayout(settings_box)

        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(1, 20)
        self.episodes_spin.setValue(3)
        settings_form.addRow("Test Episodes:", self.episodes_spin)

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(100, 3000)
        self.max_steps_spin.setValue(1000)
        self.max_steps_spin.setSingleStep(100)
        settings_form.addRow("Max Steps / Episode:", self.max_steps_spin)

        self.chk_deterministic = QCheckBox("Deterministic Policy (Cleanest walking gait)")
        self.chk_deterministic.setChecked(True)
        settings_form.addRow(self.chk_deterministic)

        right_layout.addWidget(settings_box)

        # 3. Launch Button
        self.btn_run = QPushButton("▶ Launch Live 3D Evaluation in PyBullet")
        self.btn_run.setStyleSheet("background-color: #2ea44f; font-weight: bold; font-size: 14px; padding: 12px;")
        self.btn_run.clicked.connect(self._start_evaluation)
        right_layout.addWidget(self.btn_run)

        # 4. Live Telemetry & Console
        log_box = QGroupBox("📊 Live Evaluation Telemetry")
        log_layout = QVBoxLayout(log_box)
        self.stats_label = QLabel("Ready.")
        self.stats_label.setStyleSheet("font-weight: bold; color: #a4b0be; font-size: 13px;")
        log_layout.addWidget(self.stats_label)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #1e1e1e; font-family: Consolas, monospace; font-size: 12px;")
        log_layout.addWidget(self.console)
        right_layout.addWidget(log_box)

        main_layout.addWidget(right_panel, 2)

    def _refresh_models_list(self):
        """Scans the models directory for all .zip model files and their metadata."""
        self.discovered_models.clear()
        self.table.setRowCount(0)

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)

        for root, dirs, files in os.walk(self.models_dir):
            for file in files:
                if file.endswith(".zip"):
                    zip_path = os.path.join(root, file)
                    rel_dir = os.path.relpath(root, self.models_dir)
                    exp_name = rel_dir if rel_dir != "." else file

                    # Look for genome.json in the same folder
                    genome_path = os.path.join(root, "genome.json")
                    genome = None
                    if os.path.exists(genome_path):
                        try:
                            genome = CreatureGenome.from_json(genome_path)
                        except Exception:
                            pass

                    # Look for training_info.json
                    info_path = os.path.join(root, "training_info.json")
                    train_info = {}
                    if os.path.exists(info_path):
                        try:
                            with open(info_path, "r") as f:
                                train_info = json.load(f)
                        except Exception:
                            pass

                    model_data = {
                        "exp_name": exp_name,
                        "file_name": file,
                        "zip_path": zip_path,
                        "genome": genome,
                        "train_info": train_info,
                    }
                    self.discovered_models.append(model_data)

        # Populate table
        self.table.setRowCount(len(self.discovered_models))
        for row, m in enumerate(self.discovered_models):
            display_name = f"{m['exp_name']} ({m['file_name']})"
            creature_name = m["genome"].name if m["genome"] else "Quadruped (Default)"
            num_legs = str(m["genome"].num_legs) if m["genome"] else "4"
            num_joints = str(m["genome"].total_actuators) if m["genome"] else "8"

            self.table.setItem(row, 0, QTableWidgetItem(display_name))
            self.table.setItem(row, 1, QTableWidgetItem(creature_name))
            self.table.setItem(row, 2, QTableWidgetItem(num_legs))
            self.table.setItem(row, 3, QTableWidgetItem(num_joints))

        if self.discovered_models:
            self.table.selectRow(0)

    def _on_model_selected(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        if row < len(self.discovered_models):
            self.selected_model_info = self.discovered_models[row]
            m = self.selected_model_info
            g = m["genome"]

            g_text = f"<b>Creature Name:</b> {g.name if g else 'Quadruped'}<br>"
            g_text += f"<b>Total Legs:</b> {g.num_legs if g else 4}<br>"
            g_text += f"<b>Actuated Joints:</b> {g.total_actuators if g else 8}<br>"
            g_text += f"<b>Model Weights File:</b> <code>{os.path.basename(m['zip_path'])}</code><br>"
            if m["train_info"]:
                g_text += f"<b>Trained Timesteps:</b> {m['train_info'].get('total_timesteps', 'N/A'):,}<br>"
                g_text += f"<b>Date Trained:</b> {m['train_info'].get('timestamp', 'N/A')}<br>"

            self.info_label.setText(g_text)

    def _start_evaluation(self):
        if self.is_running_eval:
            return

        if not self.selected_model_info:
            QMessageBox.warning(self, "No Model Selected", "Please select a model from the list.")
            return

        self.is_running_eval = True
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ Evaluation in Progress...")

        episodes = self.episodes_spin.value()
        max_steps = self.max_steps_spin.value()
        deterministic = self.chk_deterministic.isChecked()

        # Run evaluation in background thread so GUI remains responsive
        threading.Thread(
            target=self._run_eval_worker,
            args=(self.selected_model_info, episodes, max_steps, deterministic),
            daemon=True,
        ).start()

    def _run_eval_worker(self, model_info: Dict, episodes: int, max_steps: int, deterministic: bool):
        zip_path = model_info["zip_path"]
        genome = model_info["genome"] or CreatureGenome.preset_quadruped()

        self.signals.log_signal.emit(f"🚀 Loading model: {zip_path}")
        self.signals.log_signal.emit(f"🧬 Spawning matching creature: {genome.name} ({genome.total_actuators} joints)")

        try:
            model = PPO.load(zip_path)
            env = LocomotionEnv(
                render_mode="human",
                max_episode_steps=max_steps,
                creature_builder_fn=genome.build,
            )

            all_rewards = []
            all_distances = []
            all_speeds = []

            for ep in range(episodes):
                obs, info = env.reset(seed=ep * 42)
                total_reward = 0.0
                start_x = info.get("base_position", [0, 0, 0])[0]

                self.signals.log_signal.emit(f"\n--- Episode {ep + 1}/{episodes} ---")

                for step in range(max_steps):
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward

                    # Real-time visual pace
                    time.sleep(1.0 / 60.0)

                    if step % 30 == 0:
                        fwd_vel = info.get("forward_velocity", 0.0)
                        self.signals.stats_signal.emit(
                            f"Episode {ep + 1}/{episodes} | Step {step} | Reward: {total_reward:+.1f} | Speed: {fwd_vel:+.2f} m/s"
                        )

                    if terminated or truncated:
                        final_pos, _, _, _ = env.creature.get_base_state()
                        dist = final_pos[0] - start_x
                        avg_spd = dist / (step * env.frame_skip * env.time_step + 1e-6)

                        all_rewards.append(total_reward)
                        all_distances.append(dist)
                        all_speeds.append(avg_spd)

                        reason = "Fell / Tilted" if terminated else "Completed"
                        self.signals.log_signal.emit(f"Episode {ep + 1} ({reason}):")
                        self.signals.log_signal.emit(f"  • Distance: {dist:+.2f} m | Speed: {avg_spd:+.2f} m/s | Reward: {total_reward:+.2f}")
                        break

            env.close()

            summary = (
                f"🏆 Evaluation Summary:\n"
                f"  Mean Distance : {np.mean(all_distances):.2f} m\n"
                f"  Mean Speed    : {np.mean(all_speeds):.2f} m/s\n"
                f"  Mean Reward   : {np.mean(all_rewards):.2f}"
            )
            self.signals.log_signal.emit(f"\n{summary}")
            self.signals.stats_signal.emit(f"Finished. Mean Speed: {np.mean(all_speeds):.2f} m/s | Distance: {np.mean(all_distances):.2f} m")

        except Exception as e:
            self.signals.log_signal.emit(f"❌ Error during evaluation: {e}")
        finally:
            self.signals.finished_signal.emit()

    def _append_log(self, text: str):
        self.console.append(text)

    def _update_stats_display(self, text: str):
        self.stats_label.setText(text)

    def _on_eval_finished(self):
        self.is_running_eval = False
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶ Launch Live 3D Evaluation in PyBullet")

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
        QTableWidget {
            background-color: #2f3542;
            border: 1px solid #57606f;
            border-radius: 6px;
            gridline-color: #57606f;
            color: white;
            selection-background-color: #007acc;
        }
        QHeaderView::section {
            background-color: #1e1e2e;
            color: #70a1ff;
            font-weight: bold;
            border: 1px solid #2f3542;
            padding: 4px;
        }
        QSpinBox, QTextEdit {
            background-color: #2f3542;
            border: 1px solid #57606f;
            border-radius: 5px;
            padding: 5px;
            color: white;
        }
        """
        self.setStyleSheet(dark_style)


def main():
    app = QApplication(sys.argv)
    window = EvalPlayerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
