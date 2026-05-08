
import { StateVector, ControlInput, PhysicalParams } from "../types";
import { stepDynamicsRK4 } from "./physicsLogic";
import { ensembleDynamics } from "./learnedDynamics";

const HORIZON = 12;
const SAMPLES = 12;
const ELITES = 4;
const ITERATIONS = 1;
const UNCERTAINTY_LAMBDA = 3.0; 

let previousOptimalSequence: ControlInput[] = Array(HORIZON).fill([] as ControlInput);
let _cachedDof = 2;

/**
 * Model Predictive Control (MPC) via Sampling-based Optimization (CEM)
 * Combined with learned residuals and uncertainty awareness.
 */
export const computeMPCAction = (
  x0: StateVector,
  target: number[],
  p: PhysicalParams,
  weights: { q: number; r: number } = { q: 1.0, r: 0.1 },
  dof?: number
): { action: ControlInput, ensembleUncertainty: number } => {
  const actionDim = dof ?? (target.length > 2 ? target.length : 2);

  // Reset warm start if DOF changed
  if (actionDim !== _cachedDof) {
    previousOptimalSequence = Array(HORIZON).fill(new Array(actionDim).fill(0));
    _cachedDof = actionDim;
  }

  const last = previousOptimalSequence.slice(1);
  let currentMean: ControlInput[] = [
    ...last.map((a) => (a.length === actionDim ? [...a] : new Array(actionDim).fill(0))),
    new Array(actionDim).fill(0)
  ];
  let currentStd: ControlInput[] = Array.from({ length: HORIZON }, () => new Array(actionDim).fill(0.15));

  let totalUncertaintyAtStep0 = 0;

  for (let iter = 0; iter < ITERATIONS; iter++) {
    const trajectories: { cost: number; sequence: ControlInput[] }[] = [];

    for (let s = 0; s < SAMPLES; s++) {
      const sequence: ControlInput[] = [];
      let cost = 0;
      let xt = [...x0] as StateVector;

      for (let h = 0; h < HORIZON; h++) {
        const a: ControlInput = Array.from({ length: actionDim }, (_, j) =>
          (currentMean[h][j] ?? 0) + (Math.random() - 0.5) * 2 * (currentStd[h][j] ?? 0.15)
        );
        sequence.push(a);

        // Perform physics step using RK4
        const xPhys = stepDynamicsRK4(xt, a, p);
        
        // Query the learned residual model ensemble for correction
        const { mean: residual, variance } = ensembleDynamics.predict(xt, a);
        
        // Track uncertainty for telemetry feedback
        if (h === 0) totalUncertaintyAtStep0 += variance / SAMPLES;

        // Apply hybrid core correction: next_state = physics_prediction + learned_residual
        xt = xPhys.map((v, i) => v + residual[i]) as StateVector;
        
        // Compute quadratic cost: State deviation + Control effort + Epistemic uncertainty penalty
        const distSq = target.reduce((sum, t, i) => {
          const xv = xt[i] ?? 0;
          return sum + Math.pow(xv - t, 2);
        }, 0);
        const effortSq = a.reduce((s, v) => s + v * v, 0);
        
        cost += distSq * weights.q + effortSq * weights.r + variance * UNCERTAINTY_LAMBDA;
      }
      trajectories.push({ cost, sequence });
    }

    // Cross-Entropy Method (CEM) Update: Rank trajectories by cost and select elites
    trajectories.sort((a, b) => a.cost - b.cost);
    const elites = trajectories.slice(0, ELITES);
    
    // Refine the action distribution parameters (mean and standard deviation) from elite samples
    for (let h = 0; h < HORIZON; h++) {
      const meanVec = Array.from({ length: actionDim }, (_, j) =>
        elites.reduce((acc, e) => acc + (e.sequence[h][j] ?? 0), 0) / ELITES
      );
      currentMean[h] = meanVec;

      const varVec = Array.from({ length: actionDim }, (_, j) =>
        elites.reduce((acc, e) => acc + Math.pow((e.sequence[h][j] ?? 0) - meanVec[j], 2), 0) / ELITES
      );
      currentStd[h] = varVec.map((v) => Math.sqrt(v) + 0.01);
    }
  }

  // Update warm-start buffer for the next control cycle to maintain temporal consistency
  previousOptimalSequence = currentMean;

  return { 
    action: currentMean[0], 
    ensembleUncertainty: totalUncertaintyAtStep0 
  };
};
