
import { StateVector, ControlInput, PhysicalParams } from "../types";

const DT = 1/60; 

/**
 * Returns dx/dt given state x, control u, and parameters p
 */
export const dynamicsDerivative = (
  x: StateVector,
  u: ControlInput,
  p: PhysicalParams
): StateVector => {
  const [, , vx, vy, , omega] = x;
  const [fx, fy] = u;

  // Accel = (Force / Mass) - Friction * Velocity + Gravity
  const ax = (fx / p.mass) - (p.friction * vx);
  const ay = (fy / p.mass) - (p.friction * vy) + p.gravity;

  // [px, py, vx, vy, theta, omega] -> [vx, vy, ax, ay, omega, 0]
  return [vx, vy, ax, ay, omega, 0];
};

/**
 * 4th-Order Runge-Kutta Integration
 * x_t+1 = x_t + DT/6 * (k1 + 2k2 + 2k3 + k4)
 */
export const stepDynamicsRK4 = (
  x: StateVector,
  u: ControlInput,
  p: PhysicalParams,
  dt: number = DT
): StateVector => {
  const k1 = dynamicsDerivative(x, u, p);
  
  const x2 = x.map((v, i) => v + k1[i] * dt / 2) as StateVector;
  const k2 = dynamicsDerivative(x2, u, p);
  
  const x3 = x.map((v, i) => v + k2[i] * dt / 2) as StateVector;
  const k3 = dynamicsDerivative(x3, u, p);
  
  const x4 = x.map((v, i) => v + k3[i] * dt) as StateVector;
  const k4 = dynamicsDerivative(x4, u, p);

  return x.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])) as StateVector;
};

/**
 * Numerical Jacobian computation via finite differences using RK4
 */
export const computeJacobian = (
  x: StateVector,
  u: ControlInput,
  p: PhysicalParams
): number[][] => {
  const eps = 1e-4;
  const jacobian: number[][] = [];

  for (let i = 0; i < u.length; i++) {
    const uPlus = [...u] as ControlInput;
    uPlus[i] += eps;
    const xPlus = stepDynamicsRK4(x, uPlus, p);
    
    const uMinus = [...u] as ControlInput;
    uMinus[i] -= eps;
    const xMinus = stepDynamicsRK4(x, uMinus, p);

    jacobian.push(xPlus.map((v, idx) => (v - xMinus[idx]) / (2 * eps)));
  }

  return jacobian;
};
