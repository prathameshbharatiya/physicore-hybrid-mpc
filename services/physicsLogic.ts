
import { StateVector, ControlInput, PhysicalParams } from "../types";

const DT = 1/60; 

export const dynamicsDerivative = (
  x: StateVector,
  u: ControlInput,
  p: PhysicalParams
): StateVector => {
  const [, , vx, vy, , omega] = x;
  const [fx, fy] = u;
  const ax = (fx / p.mass) - (p.friction * vx);
  const ay = (fy / p.mass) - (p.friction * vy) + p.gravity;
  return [vx, vy, ax, ay, omega, 0];
};

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
