"""Isolierter offizieller Gymnasium/MuJoCo-Renderer fuer Tk unter macOS."""

from __future__ import annotations

import multiprocessing
import os
import sys
from multiprocessing.connection import Connection
from typing import Optional

import numpy as np


#: Gymnasiums `MujocoRenderer` kennt ab Werk nur `glfw`, `egl` und `osmesa`.
#: `cgl` bringt MuJoCo selbst mit; `register_backend()` reicht es nach.
MUJOCO_BACKENDS = ("cgl", "glfw", "egl", "osmesa")


def mujoco_gl_backend(platform: Optional[str] = None) -> str:
    """OpenGL-Backend für den Renderprozess.

    Auf macOS ist `cgl` zu wählen: GLFW meldet seinen Prozess beim Window
    Server als Vordergrund-App an, sodass **jeder** Renderprozess einen eigenen
    Dock-Eintrag bekommt – bei vier Animationen also vier zusätzliche
    Programmeinträge neben der eigentlichen Anwendung. Der CGL-Kontext rendert
    rein offscreen und meldet sich gar nicht erst an. Auf headless Linux ist
    EGL die richtige Wahl.

    Der Wert muss gesetzt sein, bevor `mujoco` beziehungsweise
    `gymnasium.envs.mujoco` importiert wird.
    """
    name = sys.platform if platform is None else platform
    return "cgl" if name == "darwin" else "egl"


def register_backend(name: str) -> None:
    """Macht `cgl` für Gymnasiums `MujocoRenderer` verfügbar.

    Gymnasium führt nur drei Backends in seiner Tabelle und lehnt `cgl` sonst
    ab, obwohl MuJoCo den Kontext mitbringt und er auf macOS der einzige ohne
    Fenster- und Dock-Eintrag ist. Der Eintrag wird nur ergänzt, nie ersetzt.
    """
    if name != "cgl":
        return
    from gymnasium.envs.mujoco import mujoco_rendering

    def _import_cgl(width: int, height: int):
        from mujoco.cgl import GLContext

        return GLContext(width, height)

    mujoco_rendering._ALL_RENDERERS.setdefault("cgl", _import_cgl)


def _worker(connection: Connection, seed: Optional[int]) -> None:
    # MUJOCO_GL muss vor dem ersten Import gesetzt sein. Eine bereits gesetzte
    # Variable bleibt unverändert, damit der Benutzer sie überschreiben kann.
    os.environ.setdefault("MUJOCO_GL", mujoco_gl_backend())
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        register_backend(os.environ["MUJOCO_GL"])
        from humanoid_logic import make_humanoid_env

        env = make_humanoid_env(render_mode="rgb_array")
        env.reset(seed=seed)
        # Der erste render() erzeugt den OpenGL-Kontext. Scheitert das, soll der
        # Fehler den Elternprozess als Meldung erreichen und nicht erst später
        # mitten in einer laufenden Animation auftauchen.
        env.render()
    except BaseException as error:  # noqa: BLE001 - wird als Text übertragen
        try:
            connection.send(("error", f"{type(error).__name__}: {error}"))
        except (BrokenPipeError, OSError):
            pass
        connection.close()
        return
    try:
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "reset":
                observation, info = env.reset(seed=payload)
                connection.send(("ok", (observation, info, np.asarray(env.render()))))
            elif command == "reset_to":
                # Auf einen aufgezeichneten Simulatorzustand zurücksetzen. Der
                # Zugriff auf `unwrapped.set_state` ist bewusst: Ohne denselben
                # Startzustand liefe eine nachgespielte Episode anders als die
                # aufgezeichnete. Für Messwerte wäre der Zugriff unzulässig.
                seed, qpos, qvel = payload
                env.reset(seed=seed)
                inner = env.unwrapped
                inner.set_state(np.asarray(qpos, dtype=np.float64),
                                np.asarray(qvel, dtype=np.float64))
                observation = inner._get_obs()
                connection.send(("ok", (observation, {}, np.asarray(env.render()))))
            elif command == "step":
                action = np.asarray(payload, dtype=np.float32).reshape(-1)
                observation, reward, terminated, truncated, info = env.step(action)
                connection.send(("ok", (observation, reward, terminated, truncated, info,
                                        np.asarray(env.render()))))
            elif command == "close":
                break
    except (EOFError, BrokenPipeError):
        pass
    finally:
        env.close()
        connection.close()


class RendererUnavailable(RuntimeError):
    """Der Renderprozess konnte MuJoCo nicht starten."""


class HumanoidRenderer:
    """Rendert im eigenen Prozess.

    Tkinter und ein nativer OpenGL-Kontext dürfen unter macOS nicht im selben
    Prozess initialisiert werden. Jede sichtbare Anzeige erhält deshalb einen
    eigenen Prozess mit eigener Environment-Instanz; die GUI bekommt nur
    RGB-Frames.
    """

    #: Sekunden, die auf die Bereitschaft des Prozesses gewartet wird. Der erste
    #: Start importiert MuJoCo und baut den Grafikkontext auf.
    STARTUP_TIMEOUT = 60.0

    def __init__(self, seed: Optional[int] = None) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self.connection = parent
        self.process = context.Process(target=_worker, args=(child, seed), daemon=True)
        self.process.start()
        child.close()
        self.closed = False
        self.ready = False

    def poll_ready(self) -> bool:
        """Ist der Grafikkontext da? Blockiert nicht.

        Der erste Start importiert MuJoCo und baut den OpenGL-Kontext auf, was
        einige Sekunden dauert. Die GUI fragt deshalb in einer `after()`-Schleife
        nach, statt zu warten und dabei einzufrieren.
        """
        return self._await_ready(0)

    def start(self) -> None:
        """Wartet auf den Grafikkontext und meldet Fehler verständlich weiter."""
        if not self._await_ready(self.STARTUP_TIMEOUT):
            self.close()
            raise RendererUnavailable(
                "Der Renderprozess hat nicht innerhalb von "
                f"{self.STARTUP_TIMEOUT:.0f} Sekunden geantwortet."
            )

    def _await_ready(self, timeout: float) -> bool:
        if self.ready:
            return True
        if self.closed:
            raise RendererUnavailable("Der isolierte Humanoid-Renderer ist bereits beendet.")
        if not self.connection.poll(timeout):
            if not self.process.is_alive():
                self.close()
                raise RendererUnavailable(
                    "Der Renderprozess wurde beendet, bevor MuJoCo bereit war."
                )
            return False
        kind, payload = self.connection.recv()
        if kind == "error":
            self.close()
            raise RendererUnavailable(
                f"MuJoCo konnte nicht starten: {payload}. Prüfe die Installation mit "
                "'pip install \"gymnasium[mujoco]\"' und notfalls die Umgebungsvariable "
                f"MUJOCO_GL (gültig: {', '.join(MUJOCO_BACKENDS)})."
            )
        self.ready = True
        return True

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict, np.ndarray]:
        observation, info, frame = self._exchange(("reset", seed))
        return np.asarray(observation), dict(info), np.asarray(frame)

    def reset_to(self, qpos: np.ndarray, qvel: np.ndarray,
                 seed: Optional[int] = None) -> tuple[np.ndarray, dict, np.ndarray]:
        """Auf einen aufgezeichneten Simulatorzustand zurücksetzen."""
        observation, info, frame = self._exchange(("reset_to", (seed, qpos, qvel)))
        return np.asarray(observation), dict(info), np.asarray(frame)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict, np.ndarray]:
        observation, reward, terminated, truncated, info, frame = self._exchange(
            ("step", np.asarray(action, dtype=np.float32)))
        return (np.asarray(observation), float(reward), bool(terminated), bool(truncated),
                dict(info), np.asarray(frame))

    def _exchange(self, message: tuple[str, object]):
        self.start()
        if self.closed or not self.process.is_alive():
            raise RendererUnavailable("Der isolierte Humanoid-Renderer ist nicht verfügbar.")
        self.connection.send(message)
        kind, payload = self.connection.recv()
        if kind == "error":
            raise RendererUnavailable(str(payload))
        return payload

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
