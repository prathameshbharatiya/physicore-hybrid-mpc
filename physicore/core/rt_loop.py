from __future__ import annotations

import threading
import time
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable, List
import numpy as np


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class ControlOutput:
    action:            np.ndarray
    state_estimate:    np.ndarray
    safety_violations: list          # List[str] — human-readable violation strings
    loop_time_us:      float         # wall-clock time for one tick (microseconds)
    deadline_missed:   bool          # True when loop_time_us > period * 1.05


# ── Rolling statistics ────────────────────────────────────────────────────────

class _LoopStats:
    """Rolling statistics over the last *window* ticks."""

    def __init__(self, window: int = 1000) -> None:
        self._times: deque = deque(maxlen=window)
        self.missed: int = 0

    def record(self, elapsed_us: float, missed: bool) -> None:
        self._times.append(elapsed_us)
        if missed:
            self.missed += 1

    def to_dict(self) -> dict:
        if not self._times:
            return {
                'mean_ms':         0.0,
                'max_ms':          0.0,
                'jitter_ms':       0.0,
                'missed_deadlines': 0,
            }
        arr = np.array(self._times) / 1000.0  # us -> ms
        return {
            'mean_ms':         round(float(arr.mean()), 3),
            'max_ms':          round(float(arr.max()),  3),
            'jitter_ms':       round(float(arr.std()),  3),
            'missed_deadlines': self.missed,
        }


# ── Real-time loop ────────────────────────────────────────────────────────────

class RTLoop:
    """
    Real-time control loop manager.

    Runs the MPC engine, optional state estimator, and optional hardware safety
    interlock at a fixed rate (``hz`` Hz) in a dedicated background thread.

    Platform-specific priority elevation:
      * Linux with CAP_SYS_NICE: SCHED_FIFO at priority 10
      * Windows: THREAD_PRIORITY_HIGHEST via Win32 API
      * Fallback: runs at the OS default priority (silently)

    The busy-wait / sleep hybrid ensures low jitter without burning 100 % of a
    core for the entire period.  The loop sleeps for (period - 0.5 ms) and then
    busy-waits the final 0.5 ms to hit the deadline precisely.

    Usage (threaded)::

        loop = RTLoop(engine, hz=200, safety=interlock, estimator=ekf)
        loop.start()
        # ... application runs ...
        loop.stop()

    Usage (single-threaded / external tick)::

        loop = RTLoop(engine, hz=200)
        out  = loop.tick(raw_sensors, x_ref)
    """

    def __init__(
        self,
        engine,
        hz:        float = 1000.0,
        safety=None,     # HardwareSafetyInterlock | None
        estimator=None,  # StateEstimator | None
    ) -> None:
        self.engine     = engine
        self.hz         = float(hz)
        self.period_s   = 1.0 / max(self.hz, 1.0)
        self._safety    = safety
        self._estimator = estimator

        self._running: bool = False
        self._thread:  Optional[threading.Thread] = None
        self._stats    = _LoopStats()

        self._state_cb:  Optional[Callable] = None
        self._action_cb: Optional[Callable] = None

        # Latest output, protected by a lock for safe cross-thread reads
        self._last_output: Optional[ControlOutput] = None
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the RT loop in a dedicated daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name="physicore-rt"
        )
        self._thread.start()
        self._set_thread_priority()

    def stop(self) -> None:
        """Signal the loop to stop and block until the thread exits (timeout 2 s)."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def tick(self, raw_sensors: dict, x_ref: np.ndarray) -> ControlOutput:
        """
        Execute one control cycle and return a :class:`ControlOutput`.

        Steps
        -----
        1. Extract raw state from *raw_sensors* (key ``'state'``).
        2. If an estimator is attached:
           a. ``predict`` using ``'last_action'`` from sensors (if available).
           b. ``update`` using ``'observation'`` from sensors (if available).
        3. Run one MPC step via ``engine.step(state_est, x_ref)``.
        4. Pass the action through the safety interlock (if armed).
        5. Fire optional callbacks and record statistics.
        """
        t0_us = time.perf_counter() * 1e6

        # ── 1. Build raw state ────────────────────────────────────────────────
        state_dim = self.engine.cfg.state_dim
        state_raw = np.asarray(
            raw_sensors.get('state', np.zeros(state_dim)), dtype=float
        )

        # ── 2. State estimation ───────────────────────────────────────────────
        if self._estimator is not None:
            # Propagate dynamics with last applied action (if provided)
            action_last = raw_sensors.get('last_action', None)
            if action_last is not None:
                self._estimator.predict(
                    state_raw,
                    np.asarray(action_last, dtype=float),
                    self.engine.physics.dynamics_fn,
                    self.engine.physics.params,
                    self.period_s,
                )
            # Fuse sensor observation (if provided this tick)
            obs = raw_sensors.get('observation', None)
            if obs is not None:
                self._estimator.update(np.asarray(obs, dtype=float))
            state_est = self._estimator.estimate
        else:
            state_est = state_raw

        # ── 3. MPC step ───────────────────────────────────────────────────────
        ctrl   = self.engine.step(state_est, x_ref)
        action = ctrl.action.copy()

        # ── 4. Safety interlock ───────────────────────────────────────────────
        violations: list = []
        if self._safety is not None and self._safety.is_armed:
            if self._safety.is_estopped:
                action = np.zeros_like(action)
            else:
                action, viol_list = self._safety.check_and_clip(
                    action, state_est, self.engine.physics.params
                )
                violations = [str(v) for v in viol_list]

        # ── 5. Timing ─────────────────────────────────────────────────────────
        t1_us      = time.perf_counter() * 1e6
        elapsed_us = t1_us - t0_us
        missed     = elapsed_us > (self.period_s * 1e6 * 1.05)  # 5 % slack

        out = ControlOutput(
            action            = action,
            state_estimate    = state_est,
            safety_violations = violations,
            loop_time_us      = elapsed_us,
            deadline_missed   = missed,
        )

        with self._lock:
            self._last_output = out

        self._stats.record(elapsed_us, missed)

        # ── Optional callbacks (swallow exceptions to never break the loop) ───
        if self._state_cb is not None:
            try:
                self._state_cb(state_est)
            except Exception:
                pass
        if self._action_cb is not None:
            try:
                self._action_cb(action)
            except Exception:
                pass

        return out

    def set_state_callback(self, fn: Callable) -> None:
        """Register a callback invoked with the estimated state on every tick."""
        self._state_cb = fn

    def set_action_callback(self, fn: Callable) -> None:
        """Register a callback invoked with the safe action on every tick."""
        self._action_cb = fn

    @property
    def stats(self) -> dict:
        """Rolling statistics over the last 1000 ticks."""
        return self._stats.to_dict()

    @property
    def missed_deadline_count(self) -> int:
        """Total number of ticks that exceeded the period deadline."""
        return self._stats.missed

    @property
    def last_output(self) -> Optional[ControlOutput]:
        """Most recent :class:`ControlOutput` (thread-safe read)."""
        with self._lock:
            return self._last_output

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        """
        Main loop body — executes in the dedicated ``physicore-rt`` thread.

        Timing strategy
        ---------------
        * Compute the absolute deadline for the *next* tick before sleeping.
        * Sleep for (remaining - 0.5 ms) to avoid OS scheduling overhead.
        * Busy-wait the final 0.5 ms to hit the deadline with sub-millisecond
          precision.
        * Advance the absolute deadline by one period (no drift accumulation).
        """
        state_dim = self.engine.cfg.state_dim
        ref       = np.zeros(state_dim)

        # Anchor the first deadline to *now* so tick 0 starts immediately
        next_tick = time.perf_counter()

        while self._running:
            now = time.perf_counter()

            if now < next_tick:
                # Sleep for the bulk of the remaining time, busy-wait the tail
                sleep_s = next_tick - now - 0.0005  # 0.5 ms busy-wait margin
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                # Busy-wait until the exact deadline
                while time.perf_counter() < next_tick:
                    pass

            # Advance deadline *before* the tick so drift cannot accumulate
            next_tick += self.period_s

            sensors = {'state': np.zeros(state_dim)}
            try:
                self.tick(sensors, ref)
            except Exception:
                # Never let a transient exception kill the RT thread
                pass

    def _set_thread_priority(self) -> None:
        """
        Attempt to elevate the RT thread's scheduling priority.

        * Linux: tries SCHED_FIFO priority 10 via ``libpthread``.
        * Windows: tries THREAD_PRIORITY_HIGHEST via ``kernel32``.
        * All errors are silently ignored; the loop continues at normal priority.
        """
        if self._thread is None or self._thread.ident is None:
            return

        # ── Linux: real-time SCHED_FIFO ───────────────────────────────────────
        if sys.platform.startswith('linux'):
            try:
                import ctypes
                import ctypes.util

                lib_name = ctypes.util.find_library('pthread') or 'libpthread.so.0'
                libpthread = ctypes.CDLL(lib_name)

                SCHED_FIFO = 1

                class sched_param(ctypes.Structure):
                    _fields_ = [('sched_priority', ctypes.c_int)]

                param = sched_param(10)  # priority 10 (valid range 1-99 for FIFO)
                ret   = libpthread.pthread_setschedparam(
                    ctypes.c_ulong(self._thread.ident),
                    ctypes.c_int(SCHED_FIFO),
                    ctypes.byref(param),
                )
                # ret == 0 -> success; non-zero -> EPERM or similar (ignored)
            except Exception:
                pass
            return  # On Linux we're done regardless of outcome

        # ── Windows: THREAD_PRIORITY_HIGHEST ─────────────────────────────────
        if sys.platform == 'win32':
            try:
                import ctypes

                THREAD_SET_INFORMATION  = 0x0020
                THREAD_PRIORITY_HIGHEST = 2

                kernel32 = ctypes.windll.kernel32
                handle   = kernel32.OpenThread(
                    THREAD_SET_INFORMATION, False, self._thread.ident
                )
                if handle:
                    kernel32.SetThreadPriority(handle, THREAD_PRIORITY_HIGHEST)
                    kernel32.CloseHandle(handle)
            except Exception:
                pass
