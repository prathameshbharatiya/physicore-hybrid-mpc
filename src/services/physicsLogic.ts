import { StateVector, ControlInput, PhysicalParams } from "../types";

const DT = 1 / 60;

// Generic N-DOF joint dynamics
export const jointSpaceDynamics = (
  x: StateVector,
  u: ControlInput,
  p: PhysicalParams
): StateVector => {
  const n = Math.floor(x.length / 2);
  const q = x.slice(0, n);
  const dq = x.slice(n, n * 2);
  const m = Math.max(p.mass, 0.001);
  const f = Math.max(p.friction, 0.0);

  const M = Array.from({ length: n }, (_, i) => Math.max(m * 0.1 * Math.pow(0.85, i), 0.0005));
  const gc = Array.from({ length: n }, (_, i) =>
    i < 3 ? m * 9.81 * 0.3 * Math.cos(q[i]) * Math.pow(0.6, i) : 0.0
  );

  const tau = u.slice(0, n);
  const ddq = tau.map((t, i) => (t + gc[i] - f * dq[i]) / M[i]);

  return [...dq, ...ddq];
};

// 2D planar dynamics (backward compat for original frontend sim)
export const dynamicsDerivative = (
  x: StateVector,
  u: ControlInput,
  p: PhysicalParams
): StateVector => {
  const vx = x[2] ?? 0;
  const vy = x[3] ?? 0;
  const omega = x[5] ?? 0;
  const fx = u[0] ?? 0;
  const fy = u[1] ?? 0;
  const ax = (fx / p.mass) - (p.friction * vx);
  const ay = (fy / p.mass) - (p.friction * vy) + p.gravity;
  return [vx, vy, ax, ay, omega, 0];
};

export const universalDynamics = (
  x: StateVector,
  u: ControlInput,
  p: PhysicalParams,
  dof?: number
): StateVector => {
  const n = dof ?? Math.floor(x.length / 2);
  if (x.length === 6 && u.length <= 2 && !dof) {
    return dynamicsDerivative(x, u, p);
  }
  return jointSpaceDynamics(x, u, p);
};

// 4th-Order Runge-Kutta
export const stepDynamicsRK4 = (
  x: StateVector,
  u: ControlInput,
  p: PhysicalParams,
  dt: number = DT,
  dof?: number
): StateVector => {
  const f = (s: StateVector) => universalDynamics(s, u, p, dof);
  const k1 = f(x);
  const x2 = x.map((v, i) => v + k1[i] * dt / 2);
  const k2 = f(x2);
  const x3 = x.map((v, i) => v + k2[i] * dt / 2);
  const k3 = f(x3);
  const x4 = x.map((v, i) => v + k3[i] * dt);
  const k4 = f(x4);
  return x.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
};

// Numerical Jacobian
export const computeJacobian = (
  x: StateVector,
  u: ControlInput,
  p: PhysicalParams,
  dof?: number
): number[][] => {
  const eps = 1e-4;
  return u.map((_, i) => {
    const uP = [...u]; uP[i] += eps;
    const uM = [...u]; uM[i] -= eps;
    const xP = stepDynamicsRK4(x, uP, p, DT, dof);
    const xM = stepDynamicsRK4(x, uM, p, DT, dof);
    return xP.map((v, j) => (v - xM[j]) / (2 * eps));
  });
};

// Zero state/action factories
export const zeroState = (dof: number, mode: 'joint' | 'base' = 'joint'): StateVector => {
  if (mode === 'joint') return new Array(dof * 2).fill(0);
  return new Array(12).fill(0);
};

export const zeroAction = (dof: number): ControlInput => new Array(dof).fill(0);

// Joint limit clamping
export const clampToJointLimits = (
  action: ControlInput,
  limitsLo?: number[],
  limitsHi?: number[]
): ControlInput => {
  if (!limitsLo || !limitsHi) return action;
  return action.map((v, i) => Math.max(limitsLo[i] ?? -Infinity, Math.min(limitsHi[i] ?? Infinity, v)));
};

// ── WebSocket with exponential backoff reconnect ─────────────────────────────

export interface ManagedWebSocketOptions {
  onMessage: (data: unknown) => void;
  onStatusChange?: (status: 'connected' | 'reconnecting' | 'disconnected') => void;
  onOpen?: () => void;
  maxRetryMs?: number; // default 30000
}

export class ManagedWebSocket {
  // Manages a WebSocket with automatic exponential backoff reconnect.
  // Delays: 1s, 2s, 4s, 8s, 16s, 30s (capped at maxRetryMs).
  // Call connect() to start, close() to stop permanently.
  private ws: WebSocket | null = null;
  private retryDelay = 1000;
  private readonly maxRetry: number;
  private stopped = false;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly url: string;
  private readonly opts: ManagedWebSocketOptions;

  constructor(url: string, opts: ManagedWebSocketOptions) {
    this.url = url;
    this.opts = opts;
    this.maxRetry = opts.maxRetryMs ?? 30000;
  }

  connect(): void {
    if (this.stopped) return;

    // Clear any pending retry timer
    if (this.retryTimer !== null) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }

    // Close existing socket if open
    if (this.ws !== null) {
      // Remove handlers to avoid double-triggering reconnect logic
      this.ws.onopen = null;
      this.ws.onmessage = null;
      this.ws.onclose = null;
      this.ws.onerror = null;
      this.ws.close();
      this.ws = null;
    }

    let ws: WebSocket;
    try {
      ws = new WebSocket(this.url);
    } catch {
      // Construction failed (e.g. invalid URL), treat as a connection error
      this.opts.onStatusChange?.('reconnecting');
      this.scheduleReconnect();
      return;
    }

    this.ws = ws;

    ws.onopen = () => {
      if (this.stopped) {
        ws.close();
        return;
      }
      // Reset backoff on successful connection
      this.retryDelay = 1000;
      this.opts.onStatusChange?.('connected');
      this.opts.onOpen?.();
    };

    ws.onmessage = (event: MessageEvent) => {
      if (this.stopped) return;
      let parsed: unknown = event.data;
      if (typeof event.data === 'string') {
        try {
          parsed = JSON.parse(event.data) as unknown;
        } catch {
          parsed = event.data;
        }
      }
      this.opts.onMessage(parsed);
    };

    ws.onclose = () => {
      if (this.stopped) {
        this.opts.onStatusChange?.('disconnected');
        return;
      }
      this.opts.onStatusChange?.('reconnecting');
      this.scheduleReconnect();
    };

    ws.onerror = () => {
      // onerror is always followed by onclose; let onclose handle reconnect.
      // We notify status here so the UI reflects the problem immediately.
      if (!this.stopped) {
        this.opts.onStatusChange?.('reconnecting');
      }
    };
  }

  private scheduleReconnect(): void {
    if (this.stopped) return;
    const delay = this.retryDelay;
    // Exponential backoff, capped at maxRetry
    this.retryDelay = Math.min(this.retryDelay * 2, this.maxRetry);
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      this.connect();
    }, delay);
  }

  close(): void {
    this.stopped = true;
    if (this.retryTimer !== null) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    if (this.ws !== null) {
      this.ws.onopen = null;
      this.ws.onmessage = null;
      this.ws.onclose = null;
      this.ws.onerror = null;
      this.ws.close();
      this.ws = null;
    }
    this.opts.onStatusChange?.('disconnected');
  }

  get readyState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED;
  }
}

// Factory for simple one-off managed WebSocket
export const createManagedWebSocket = (
  url: string,
  opts: ManagedWebSocketOptions
): ManagedWebSocket => {
  const mws = new ManagedWebSocket(url, opts);
  mws.connect();
  return mws;
};
