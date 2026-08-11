"""Isolated Gymnasium renderer for safe coexistence with Tk on macOS."""

from __future__ import annotations

import multiprocessing
import os
from multiprocessing.connection import Connection
from typing import Optional

import numpy as np


def _render_worker(connection: Connection) -> None:
    """Own Pygame/SDL in a process that never imports or initializes Tk."""
    # This process only produces RGB arrays. A headless SDL driver prevents a
    # second macOS Dock application while keeping SDL isolated from Tk.
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["SDL_JOYSTICK_HIDAPI"] = "0"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    import gymnasium

    environment = gymnasium.make("CartPole-v1", render_mode="rgb_array")
    try:
        while True:
            command, payload = connection.recv()
            if command == "reset":
                observation, info = environment.reset(seed=payload)
                connection.send((observation, info, np.asarray(environment.render())))
            elif command == "step":
                observation, reward, terminated, truncated, info = environment.step(int(payload))
                connection.send(
                    (
                        observation,
                        float(reward),
                        bool(terminated),
                        bool(truncated),
                        info,
                        np.asarray(environment.render()),
                    )
                )
            elif command == "close":
                break
            else:
                raise ValueError(f"Unbekannter Render-Befehl: {command}")
    except (EOFError, BrokenPipeError):
        pass
    finally:
        environment.close()
        connection.close()


class CartPoleRenderer:
    """Small request/response facade around the isolated render process."""

    def __init__(self) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(target=_render_worker, args=(child,), daemon=True)
        self._process.start()
        child.close()
        self._closed = False

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict, np.ndarray]:
        self._ensure_open()
        self._connection.send(("reset", seed))
        observation, info, frame = self._connection.recv()
        return np.asarray(observation), dict(info), np.asarray(frame)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict, np.ndarray]:
        self._ensure_open()
        self._connection.send(("step", int(action)))
        observation, reward, terminated, truncated, info, frame = self._connection.recv()
        return (
            np.asarray(observation), float(reward), bool(terminated),
            bool(truncated), dict(info), np.asarray(frame),
        )

    def _ensure_open(self) -> None:
        if self._closed or not self._process.is_alive():
            raise RuntimeError("Der isolierte CartPole-Renderer ist nicht verfügbar.")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.is_alive():
                self._connection.send(("close", None))
                self._process.join(timeout=2)
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1)
            self._connection.close()
