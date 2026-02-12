
import { StateVector, ControlInput, PhysicalParams } from "../types";
import { stepDynamicsRK4 } from "./physicsLogic";

const LEARNING_RATE_INITIAL = 0.008;
const MASS_BOUNDS = [0.1, 5.0];
const FRICTION_BOUNDS = [0.0, 1.0];

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
