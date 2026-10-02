"""
Base PyBullet-Gymnasium environment for Musubi.
Handles physics client lifecycles, GUI/DIRECT rendering, frame skipping, and world setup.
"""

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class MusubiBaseEnv(gym.Env):
    """
    Base environment wrapping PyBullet physics engine for Gymnasium compatibility.
    
    Attributes:
        render_mode: None (headless/fast), 'human' (PyBullet GUI), or 'rgb_array' (synthetic camera images).
        time_step: Physics simulation sub-step delta time (default 1/240 s).
        frame_skip: Number of physics steps per RL environment step (action repetition).
        gravity: 3D vector for gravity acceleration (default [0, 0, -9.81]).
        ground_friction: Lateral friction coefficient for ground plane.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        time_step: float = 1.0 / 240.0,
        frame_skip: int = 4,
        gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81),
        ground_friction: float = 1.2,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.time_step = time_step
        self.frame_skip = frame_skip
        self.gravity = gravity
        self.ground_friction = ground_friction

        self.client_id: int = -1
        self.plane_id: int = -1
        self._camera_distance: float = 2.0
        self._camera_yaw: float = 50.0
        self._camera_pitch: float = -30.0

        # Action and observation spaces should be defined in subclass
        self.action_space: spaces.Space = spaces.Box(
            low=-1.0, high=1.0, shape=(0,), dtype=np.float32
        )
        self.observation_space: spaces.Space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32
        )

        self._initialize_simulation()

    def _initialize_simulation(self) -> None:
        """Connect to PyBullet in GUI or DIRECT mode."""
        if self.client_id >= 0 and p.isConnected(physicsClientId=self.client_id):
            return

        if self.render_mode == "human":
            self.client_id = p.connect(p.GUI)
            # Disable unneeded default UI panels for a cleaner view
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client_id)
        else:
            # Headless mode for fast parallel RL rollouts
            self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)
        p.setGravity(*self.gravity, physicsClientId=self.client_id)

    def _setup_world(self) -> None:
        """Reset simulation and recreate the ground plane with realistic friction."""
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(*self.gravity, physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        p.changeDynamics(
            self.plane_id,
            -1,
            lateralFriction=self.ground_friction,
            physicsClientId=self.client_id,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to an initial state.
        Must be overridden or extended by subclasses to spawn creatures.
        """
        super().reset(seed=seed)
        self._setup_world()
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Applies action, steps the simulation for `frame_skip` iterations, and returns (obs, reward, terminated, truncated, info).
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses of MusubiBaseEnv must implement `step()`.")

    def step_simulation(self) -> None:
        """Step the physics engine `frame_skip` times per policy step."""
        for _ in range(self.frame_skip):
            p.stepSimulation(physicsClientId=self.client_id)

    def render(self) -> Optional[np.ndarray]:
        """
        Renders the current frame.
        If render_mode is 'rgb_array', returns an RGB numpy array of shape (H, W, 3).
        """
        if self.render_mode == "rgb_array":
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0],
                distance=self._camera_distance,
                yaw=self._camera_yaw,
                pitch=self._camera_pitch,
                roll=0,
                upAxisIndex=2,
                physicsClientId=self.client_id,
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=4.0 / 3.0,
                nearVal=0.1,
                farVal=100.0,
                physicsClientId=self.client_id,
            )
            _, _, rgba, _, _ = p.getCameraImage(
                width=640,
                height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self.client_id,
            )
            return rgba[:, :, :3]
        return None

    def update_camera_tracking(self, target_pos: Tuple[float, float, float]) -> None:
        """Smoothly track the robot with the camera in GUI mode."""
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=self._camera_distance,
                cameraYaw=self._camera_yaw,
                cameraPitch=self._camera_pitch,
                cameraTargetPosition=target_pos,
                physicsClientId=self.client_id,
            )

    def close(self) -> None:
        """Clean up physics server connection."""
        if self.client_id >= 0:
            if p.isConnected(physicsClientId=self.client_id):
                p.disconnect(physicsClientId=self.client_id)
            self.client_id = -1
