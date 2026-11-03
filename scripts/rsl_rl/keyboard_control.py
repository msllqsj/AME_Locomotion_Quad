# SpeedScaledKeyboard adapted from robot_lab (Copyright 2024-2026 Ziqi Fan).
# SPDX-License-Identifier: Apache-2.0

"""Keyboard velocity commands for interactive playback."""

import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand


class SpeedScaledKeyboard:
    """Add independently adjustable x, y, and yaw speed scales to a keyboard controller."""

    def __init__(
        self,
        controller,
        vx_speed_scale: float = 1.0,
        vy_speed_scale: float = 1.0,
        wz_speed_scale: float = 1.0,
        speed_scale_step: float = 0.1,
        min_speed_scale: float = 0.1,
        max_speed_scale: float = 2.0,
    ):
        if not 0.0 < min_speed_scale <= max_speed_scale:
            raise ValueError("Speed scale limits must be positive and ordered.")
        for name, speed_scale in (("vx", vx_speed_scale), ("vy", vy_speed_scale), ("wz", wz_speed_scale)):
            if not min_speed_scale <= speed_scale <= max_speed_scale:
                raise ValueError(f"{name} speed scale must be within the configured limits.")
        if speed_scale_step <= 0.0:
            raise ValueError("Speed scale step must be positive.")

        self._controller = controller
        self._default_speed_scales = {"vx": vx_speed_scale, "vy": vy_speed_scale, "wz": wz_speed_scale}
        self._speed_scales = self._default_speed_scales.copy()
        self._speed_scale_step = speed_scale_step
        self._min_speed_scale = min_speed_scale
        self._max_speed_scale = max_speed_scale
        self._last_command = None

        self._controller.add_callback("W", lambda: self._increase_speed("vx"))
        self._controller.add_callback("S", lambda: self._decrease_speed("vx"))
        self._controller.add_callback("A", lambda: self._increase_speed("vy"))
        self._controller.add_callback("D", lambda: self._decrease_speed("vy"))
        self._controller.add_callback("E", lambda: self._increase_speed("wz"))
        self._controller.add_callback("Q", lambda: self._decrease_speed("wz"))
        self._controller.add_callback("R", self.reset_speed_scales)

        print("[INFO] Keyboard speed controls: W/S vx, A/D vy, Q/E wz, R reset")
        print("[INFO] Hold arrow Up/Down: forward/backward; Left/Right: strafe; Z/X: turn; release: stop")
        print("[INFO] Speed-scale keys change multipliers only; press movement keys to send a nonzero command.")
        self._print_speed_scales()

    @property
    def vx_speed_scale(self) -> float:
        """Current x-axis velocity-command multiplier."""
        return self._speed_scales["vx"]

    @property
    def vy_speed_scale(self) -> float:
        """Current y-axis velocity-command multiplier."""
        return self._speed_scales["vy"]

    @property
    def wz_speed_scale(self) -> float:
        """Current yaw velocity-command multiplier."""
        return self._speed_scales["wz"]

    def advance(self):
        """Return the keyboard command with separate x, y, and yaw speed scales."""
        command = self._controller.advance().clone()
        command[0] *= self._speed_scales["vx"]
        command[1] *= self._speed_scales["vy"]
        command[2] *= self._speed_scales["wz"]
        if self._last_command is None or not torch.allclose(command, self._last_command):
            vx, vy, wz = command.detach().cpu().tolist()
            print(f"[INFO] Keyboard target velocity: vx={vx:.3f} m/s, vy={vy:.3f} m/s, wz={wz:.3f} rad/s")
            self._last_command = command.clone()
        return command

    def reset_speed_scales(self):
        """Restore all initial speed scales."""
        self._speed_scales = self._default_speed_scales.copy()
        self._print_speed_scales()

    def _increase_speed(self, axis: str):
        self._set_speed_scale(axis, self._speed_scales[axis] + self._speed_scale_step)

    def _decrease_speed(self, axis: str):
        self._set_speed_scale(axis, self._speed_scales[axis] - self._speed_scale_step)

    def _set_speed_scale(self, axis: str, speed_scale: float):
        self._speed_scales[axis] = self._clamp_speed_scale(speed_scale)
        self._print_speed_scales()

    def _clamp_speed_scale(self, speed_scale: float) -> float:
        return round(min(max(speed_scale, self._min_speed_scale), self._max_speed_scale), 10)

    def _print_speed_scales(self):
        print(
            f"[INFO] Keyboard speed scales: vx={self._speed_scales['vx']:.1f}x, "
            f"vy={self._speed_scales['vy']:.1f}x, wz={self._speed_scales['wz']:.1f}x"
        )


class KeyboardVelocityCommand(UniformVelocityCommand):
    """Use the same keyboard command for every environment, including after reset."""

    def __init__(self, cfg, env):
        from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

        super().__init__(cfg, env)
        self.controller = SpeedScaledKeyboard(
            Se2Keyboard(
                Se2KeyboardCfg(
                    v_x_sensitivity=1.0,
                    v_y_sensitivity=1.0,
                    omega_z_sensitivity=1.57,
                    sim_device=self.device,
                )
            )
        )

    def _resample_command(self, env_ids):
        self.vel_command_b[env_ids] = self.controller.advance()

    def _update_command(self):
        self.vel_command_b[:] = self.controller.advance()
