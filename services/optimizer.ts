
import { StateVector, ControlInput, PhysicalParams } from "../types";
import { stepDynamicsRK4 } from "./physicsLogic";
import { ensembleDynamics } from "./learnedDynamics";

const HORIZON = 12;
const SAMPLES = 12;
const ELITES = 4;
const ITERATIONS = 1;
const UNCERTAINTY_LAMBDA = 3.0; 

let previousOptimalSequence: ControlInput[] = Array(HORIZON).fill([0, 0] as ControlInput);

// Fix: Completed the implementation of computeMPCAction to satisfy the return type requirements and finish the optimization logic.
export const computeMPCAction = (
  x0: StateVector,
  target: [number, number],
  p: PhysicalParams,
  weights: { q: number; r: number } = { q: 1.0, r: 0.1 }
): { action: ControlInput, ensembleUncertainty: number } => {
  
  let currentMean: ControlInput[] = [...previousOptimalSequence.slice(1), [0, 0] as ControlInput];
  let currentStd: ControlInput[] = Array(HORIZON).fill([0.15, 0.15] as ControlInput);

  let totalUncertaintyAtStep0 = 0;

  for (let iter = 0; iter < ITERATIONS; iter++) {
    const trajectories: { cost: number; sequence: ControlInput[] }[] = [];

    for (let s = 0; s < SAMPLES; s++) {
      const sequence: ControlInput[] = [];
      let cost = 0;
      let xt = [...x0] as StateVector;

      for (let h = 0; h < HORIZON; h++) {
        const a: ControlInput = [
          currentMean[h][0] + (Math.random() - 0.5) * 2 * currentStd[h][0],
          currentMean[h][1] + (Math.random() - 0.5) * 2 * currentStd[h][1]
        ];
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
        const distSq = Math.pow(xt[0] - target[0], 2) + Math.pow(xt[1] - target[1], 2);
        const effortSq = Math.pow(a[0], 2) + Math.pow(a[1], 2);
        
        cost += distSq * weights.q + effortSq * weights.r + variance * UNCERTAINTY_LAMBDA;
      }
      trajectories.push({ cost, sequence });
    }

    // Cross-Entropy Method (CEM) Update: Rank trajectories by cost and select elites
    trajectories.sort((a, b) => a.cost - b.cost);
    const elites = trajectories.slice(0, ELITES);
    
    // Refine the action distribution parameters (mean and standard deviation) from elite samples
    for (let h = 0; h < HORIZON; h++) {
      const meanX = elites.reduce((acc, e) => acc + e.sequence[h][0], 0) / ELITES;
      const meanY = elites.reduce((acc, e) => acc + e.sequence[h][1], 0) / ELITES;
      currentMean[h] = [meanX, meanY];
      
      const varX = elites.reduce((acc, e) => acc + Math.pow(e.sequence[h][0] - meanX, 2), 0) / ELITES;
      const varY = elites.reduce((acc, e) => acc + Math.pow(e.sequence[h][1] - meanY, 2), 0) / ELITES;
      currentStd[h] = [Math.sqrt(varX) + 0.01, Math.sqrt(varY) + 0.01];
    }
  }

  // Update warm-start buffer for the next control cycle to maintain temporal consistency
  previousOptimalSequence = currentMean;

  return { 
    action: currentMean[0], 
    ensembleUncertainty: totalUncertaintyAtStep0 
  };
};
