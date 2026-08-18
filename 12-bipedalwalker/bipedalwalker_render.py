"""Isolated official Gymnasium renderer for macOS-safe Tk integration."""

from __future__ import annotations

import multiprocessing
import os
from multiprocessing.connection import Connection
from typing import Optional

import numpy as np


def _worker(connection: Connection) -> None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["SDL_JOYSTICK_HIDAPI"] = "0"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    import gymnasium

    env = gymnasium.make("BipedalWalker-v3", hardcore=False, render_mode="rgb_array")
    try:
        while True:
            command, payload = connection.recv()
            if command == "reset":
                observation, info = env.reset(seed=payload)
                connection.send((observation, info, np.asarray(env.render())))
            elif command == "step":
                action = np.asarray(payload, dtype=np.float32).reshape(-1)
                observation, reward, terminated, truncated, info = env.step(action)
                connection.send((observation, reward, terminated, truncated, info, np.asarray(env.render())))
            elif command == "close":
                break
    except (EOFError, BrokenPipeError):
        pass
    finally:
        env.close()
        connection.close()


class BipedalWalkerRenderer:
    """Rendert im eigenen Prozess, damit SDL und Tk sich unter macOS nicht
    im selben Prozess initialisieren."""

    def __init__(self) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self.connection = parent
        self.process = context.Process(target=_worker, args=(child,), daemon=True)
        self.process.start()
        child.close()
        self.closed = False

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict, np.ndarray]:
        self._check()
        self.connection.send(("reset", seed))
        observation, info, frame = self.connection.recv()
        return np.asarray(observation), dict(info), np.asarray(frame)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict, np.ndarray]:
        self._check()
        self.connection.send(("step", np.asarray(action, dtype=np.float32)))
        observation, reward, terminated, truncated, info, frame = self.connection.recv()
        return (np.asarray(observation), float(reward), bool(terminated), bool(truncated),
                dict(info), np.asarray(frame))

    def _check(self) -> None:
        if self.closed or not self.process.is_alive():
            raise RuntimeError("Der isolierte BipedalWalker-Renderer ist nicht verfügbar.")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.is_alive():
                self.connection.send(("close", None))
                self.process.join(timeout=2)
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
            self.connection.close()
