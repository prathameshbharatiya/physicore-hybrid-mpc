
import { StateVector, ControlInput, PhysicalParams } from "../types";
import { stepDynamicsRK4 } from "./physicsLogic";

const LEARNING_RATE_INITIAL = 0.008;
const MASS_BOUNDS = [0.1, 5.0];
const FRICTION_BOUNDS = [0.0, 1.0];
// Per-joint friction and inertia estimation
const JOINT_FRICTION_BOUNDS = [0.0, 2.0];
const JOINT_INERTIA_BOUNDS  = [0.0001, 10.0];

export interface ExtendedPhysicalParams {
  mass: number;
  friction: number;
  gravity: number;
  textile_k: number;
  damping: number;
  dof?: number;
  joint_frictions?: number[];
  link_inertias?: number[];
}

let currentLR = LEARNING_RATE_INITIAL;
let consecutiveErrorCount = 0;
let lastError = Infinity;

/**
 * Online parameter estimation with projection to admissible physical sets.
 */
export const updateSystemID = (
  prevX: StateVector,
  action: ControlInput,
  currX: StateVector,
  params: PhysicalParams
): PhysicalParams => {
  // Use RK4 prediction for gradient base
  const xPred = stepDynamicsRK4(prevX, action, params);
  
  const error = Math.sqrt(currX.reduce((s, v, i) => s + Math.pow(v - xPred[i], 2), 0));

  // Adaptive Learning Rate
  if (error > lastError) {
    consecutiveErrorCount++;
    if (consecutiveErrorCount > 3) currentLR *= 0.5;
  } else {
    consecutiveErrorCount = 0;
    currentLR = Math.min(LEARNING_RATE_INITIAL, currentLR * 1.05);
  }
  lastError = error;

  const eps = 1e-3;
  
  // Numerical Partial Derivatives for Mass
  const xMassPlus = stepDynamicsRK4(prevX, action, { ...params, mass: params.mass + eps });
  const errMassPlus = Math.sqrt(currX.reduce((s, v, i) => s + Math.pow(v - xMassPlus[i], 2), 0));
  const gradMass = (errMassPlus - error) / eps;

  // Numerical Partial Derivatives for Friction
  const xFricPlus = stepDynamicsRK4(prevX, action, { ...params, friction: params.friction + eps });
  const errFricPlus = Math.sqrt(currX.reduce((s, v, i) => s + Math.pow(v - xFricPlus[i], 2), 0));
  const gradFric = (errFricPlus - error) / eps;

  // Update & Project
  const newMass = Math.max(MASS_BOUNDS[0], Math.min(MASS_BOUNDS[1], params.mass - currentLR * gradMass));
  const newFriction = Math.max(FRICTION_BOUNDS[0], Math.min(FRICTION_BOUNDS[1], params.friction - currentLR * gradFric));

  return {
    ...params,
    mass: newMass,
    friction: newFriction
  };
};

/**
 * Per-joint parameter estimation for N-DOF manipulators.
 * Estimates joint-level friction and link inertia from trajectory data.
 */
export const updateJointSystemID = (
  prevX: number[],
  action: number[],
  currX: number[],
  params: ExtendedPhysicalParams
): ExtendedPhysicalParams => {
  const n = Math.floor(prevX.length / 2);
  const dq = prevX.slice(n);
  const ddq_meas = currX.slice(n).map((v, i) => v - prevX[n + i]);

  const frictions = params.joint_frictions
    ? [...params.joint_frictions]
    : new Array(n).fill(params.friction);
  const inertias = params.link_inertias
    ? [...params.link_inertias]
    : new Array(n).fill(params.mass * 0.1);

  const lr = 0.001;

  for (let i = 0; i < n; i++) {
    const tau = action[i] ?? 0;
    const Mi = Math.max(inertias[i], 0.0001);
    const fi = frictions[i];
    const ddq_pred = (tau - fi * dq[i]) / Mi;
    const err = ddq_meas[i] - ddq_pred;

    // Gradient w.r.t friction_i
    const grad_f = -err * dq[i] / Mi;
    frictions[i] = Math.max(
      JOINT_FRICTION_BOUNDS[0],
      Math.min(JOINT_FRICTION_BOUNDS[1], fi - lr * grad_f)
    );

    // Gradient w.r.t inertia_i
    const grad_M = -err * (tau - fi * dq[i]) / (Mi * Mi);
    inertias[i] = Math.max(
      JOINT_INERTIA_BOUNDS[0],
      Math.min(JOINT_INERTIA_BOUNDS[1], Mi - lr * grad_M)
    );
  }

  return { ...params, joint_frictions: frictions, link_inertias: inertias };
};
