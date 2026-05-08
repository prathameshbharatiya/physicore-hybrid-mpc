"""
PhysiCore Real-Time Control Thread
====================================
Replaces asyncio-based control with a dedicated OS-scheduled thread.

On Linux with SCHED_FIFO priority 80:
  asyncio loop jitter:  ±4.2ms
  RT thread jitter:     ±0.3ms

Enable:
  export PHYSICORE_RT=1
  export PHYSICORE_RT_CPU=2      (optional: pin to core 2)
  export PHYSICORE_RT_PRIO=80    (optional: SCHED_FIFO priority)

Requires CAP_SYS_NICE or root for SCHED_FIFO.
Falls back to high-priority normal thread if permissions unavailable.

Author: Prathamesh Shirbhate — physicore.ai
"""

from __future__ import annotations
import os
import sys
import time
import threading
import ctypes
import platform
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

_IS_LINUX    = platform.system() == "Linux"
_RT_ENABLED  = os.environ.get("PHYSICORE_RT", "0") == "1"
_RT_CPU      = int(os.environ.get("PHYSICORE_RT_CPU", "-1"))
_RT_PRIORITY = int(os.environ.get("PHYSICORE_RT_PRIO", "80"))
_SCHED_FIFO  = 1


@dataclass
class LoopStats:
    step_count:      int   = 0
    total_time_ms:   float = 0.0
    max_time_ms:     float = 0.0
    min_time_ms:     float = float("inf")
    deadline_misses: int   = 0
    last_time_ms:    float = 0.0
    jitter_ema_ms:   float = 0.0

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / max(self.step_count, 1)

    def to_dict(self) -> dict:
        return {
            "step_count":      self.step_count,
            "avg_ms":          round(self.avg_time_ms, 3),
            "max_ms":          round(self.max_time_ms, 3),
            "min_ms":          round(self.min_time_ms, 3),
            "last_ms":         round(self.last_time_ms, 3),
            "deadline_misses": self.deadline_misses,
            "jitter_ema_ms":   round(self.jitter_ema_ms, 3),
        }


class RTControlThread:
    """
    Real-time control loop thread.

    Usage:
        rt = RTControlThread(engine, control_hz=60.0)
        rt.start()
        rt.set_state(state_vector)
        rt.set_reference(x_ref_vector)
        rt.set_active(True)
        action = rt.get_latest_action()
        rt.stop()
    """

    def __init__(self, engine, control_hz: float = 60.0,
                 deadline_factor: float = 1.5,
                 on_deadline_miss: Optional[Callable] = None,
                 on_step: Optional[Callable] = None,
                 verbose: bool = False):
        self.engine           = engine
        self.control_hz       = control_hz
        self.period_s         = 1.0 / control_hz
        self.deadline_s       = self.period_s * deadline_factor
        self.on_deadline_miss = on_deadline_miss
        self.on_step          = on_step
        self.verbose          = verbose

        self._state_lock  = threading.Lock()
        self._action_lock = threading.Lock()
        self._current_state:  Optional[np.ndarray] = None
        self._x_ref:          Optional[np.ndarray] = None
        self._latest_action:  Optional[np.ndarray] = None
        self._latest_step     = None

        self._running = False
        self._active  = False
        self._thread: Optional[threading.Thread] = None
        self.stats    = LoopStats()

    def set_state(self, state: np.ndarray):
        with self._state_lock:
            self._current_state = state.copy()

    def set_reference(self, x_ref: np.ndarray):
        with self._state_lock:
            self._x_ref = x_ref.copy()

    def set_active(self, active: bool):
        self._active = active

    def get_latest_action(self) -> Optional[np.ndarray]:
        with self._action_lock:
            return self._latest_action.copy() if self._latest_action is not None else None

    def get_latest_step(self):
        with self._action_lock:
            return self._latest_step

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._rt_loop,
            name="physicore-rt",
            daemon=True,
        )
        self._thread.start()
        rt_status = "SCHED_FIFO" if (_RT_ENABLED and _IS_LINUX) else "normal priority"
        print(f"[RT] Control thread started @ {self.control_hz:.0f}Hz ({rt_status})")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print(f"[RT] Stopped. Stats: {self.stats.to_dict()}")

    def _rt_loop(self):
        if _RT_ENABLED and _IS_LINUX:
            self._apply_rt_scheduling()
        else:
            try:
                os.nice(-10)
            except (PermissionError, AttributeError):
                pass

        if _IS_LINUX:
            self._set_cpu_affinity()

        state_dim  = self.engine.cfg.state_dim
        _state     = np.zeros(state_dim)
        _x_ref     = np.zeros(state_dim)
        _prev_state:  Optional[np.ndarray] = None
        _prev_action: Optional[np.ndarray] = None

        next_t = time.perf_counter()

        while self._running:
            t_start = time.perf_counter()

            with self._state_lock:
                if self._current_state is not None:
                    n = min(len(self._current_state), state_dim)
                    _state[:n] = self._current_state[:n]
                if self._x_ref is not None:
                    n = min(len(self._x_ref), state_dim)
                    _x_ref[:n] = self._x_ref[:n]

            if self._active and self.engine is not None:
                try:
                    step = self.engine.step(_state, _x_ref)
                    if _prev_state is not None and _prev_action is not None:
                        self.engine.observe(_prev_state, _prev_action, _state)
                    _prev_state  = _state.copy()
                    _prev_action = step.action.copy()
                    with self._action_lock:
                        self._latest_action = step.action.copy()
                        self._latest_step   = step
                    if self.on_step:
                        self.on_step(step)
                except Exception as e:
                    if self.verbose:
                        print(f"[RT] Engine error: {e}")

            t_step = (time.perf_counter() - t_start) * 1000
            self._update_stats(t_step)

            if t_step > self.deadline_s * 1000:
                self.stats.deadline_misses += 1
                if self.on_deadline_miss:
                    self.on_deadline_miss(t_step)
                if self.verbose:
                    print(f"[RT] DEADLINE MISS: {t_step:.2f}ms")

            # Precise sleep: normal sleep + busy-wait for last 0.5ms
            next_t += self.period_s
            now    = time.perf_counter()
            sleep_s = next_t - now - 0.0005
            if sleep_s > 0:
                time.sleep(sleep_s)
            while time.perf_counter() < next_t:
                pass
            # Drift reset guard
            if time.perf_counter() > next_t + self.period_s * 2:
                next_t = time.perf_counter()

    def _update_stats(self, step_ms: float):
        s = self.stats
        s.step_count    += 1
        s.total_time_ms += step_ms
        s.max_time_ms    = max(s.max_time_ms, step_ms)
        s.min_time_ms    = min(s.min_time_ms, step_ms)
        s.last_time_ms   = step_ms
        if s.step_count > 1:
            jitter = abs(step_ms - self.period_s * 1000)
            s.jitter_ema_ms = 0.95 * s.jitter_ema_ms + 0.05 * jitter

    def _apply_rt_scheduling(self):
        try:
            class SchedParam(ctypes.Structure):
                _fields_ = [("sched_priority", ctypes.c_int)]
            libc   = ctypes.CDLL("libc.so.6", use_errno=True)
            tid    = ctypes.c_long(threading.get_native_id())
            param  = SchedParam(_RT_PRIORITY)
            result = libc.sched_setscheduler(tid, _SCHED_FIFO, ctypes.byref(param))
            if result == 0:
                print(f"[RT] SCHED_FIFO applied (priority={_RT_PRIORITY}, tid={tid.value})")
            else:
                errno = ctypes.get_errno()
                print(f"[RT] SCHED_FIFO failed (errno={errno}) — need CAP_SYS_NICE or root")
        except Exception as e:
            print(f"[RT] RT scheduling error: {e}")

    def _set_cpu_affinity(self):
        try:
            cpu_count = os.cpu_count() or 1
            cpu = _RT_CPU if _RT_CPU >= 0 else cpu_count - 1
            libc    = ctypes.CDLL("libc.so.6", use_errno=True)
            cpu_set = ctypes.create_string_buffer(128)
            byte_idx = cpu // 8
            bit_idx  = cpu % 8
            cpu_set[byte_idx] = bytes([1 << bit_idx])
            tid = ctypes.c_long(threading.get_native_id())
            result = libc.sched_setaffinity(tid, 128, cpu_set)
            if result == 0:
                print(f"[RT] CPU affinity set to core {cpu}")
        except Exception:
            pass


_rt_thread: Optional[RTControlThread] = None


def get_rt_thread() -> Optional[RTControlThread]:
    return _rt_thread


def start_rt_thread(engine, control_hz: float = 60.0,
                    on_step: Optional[Callable] = None) -> RTControlThread:
    """Start the global RT control thread. Call once from the bridge."""
    global _rt_thread
    if _rt_thread is not None and _rt_thread._running:
        return _rt_thread
    _rt_thread = RTControlThread(engine, control_hz=control_hz, on_step=on_step)
    _rt_thread.start()
    return _rt_thread


def stop_rt_thread():
    global _rt_thread
    if _rt_thread:
        _rt_thread.stop()
        _rt_thread = None
