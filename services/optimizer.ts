
import { StateVector, ControlInput, PhysicalParams } from "../types";
import { stepDynamicsRK4 } from "./physicsLogic";
import { ensembleDynamics } from "./learnedDynamics";

const HORIZON = 15;
const SAMPLES = 30;
const ELITES = 6;
const ITERATIONS = 2;
const UNCERTAINTY_LAMBDA = 5.0; // Penalty for epistemic uncertainty

// State for Warm-Starting
let previousOptimalSequence: ControlInput[] = Array(HORIZON).fill([0, 0]);

export const computeMPCAction = (
  x0: StateVector,
  target: [number, number],
  p: PhysicalParams,
  weights: { q: number; r: number } = { q: 1.0, r: 0.1 }
): { action: ControlInput, ensembleUncertainty: number } => {
  
  // 1. Warm-Start: Shift previous sequence left
  // Fix: Explicitly type currentMean and currentStd as ControlInput[] to avoid inference as number[][]
  let currentMean: ControlInput[] = [...previousOptimalSequence.slice(1), [0, 0]];
  let currentStd: ControlInput[] = Array(HORIZON).fill([0.2, 0.2]);

  let totalUncertaintyAtStep0 = 0;

  for (let iter = 0; iter < ITERATIONS; iter++) {
    const trajectories: { cost: number; sequence: ControlInput[] }[] = [];

    for (let s = 0; s < SAMPLES; s++) {
      const sequence: ControlInput[] = [];
      let cost = 0;
      let xt = [...x0] as StateVector;

      for (let h = 0; h < HORIZON; h++) {
        // Sample action for this step
        const a: ControlInput = [
          currentMean[h][0] + (Math.random() - 0.5) * 2 * currentStd[h][0],
          currentMean[h][1] + (Math.random() - 0.5) * 2 * currentStd[h][1]
        ];
        sequence.push(a);

        // Dynamics Step (Hybrid: RK4 Physics + Neural Residual)
        const xPhys = stepDynamicsRK4(xt, a, p);
        const { mean: xRes, variance } = ensembleDynamics.predict(xt, a);
        
        xt = xPhys.map((v, i) => v + xRes[i]) as StateVector;

        // Cost Calculation
        const distSq = Math.pow(xt[0] - target[0], 2) + Math.pow(xt[1] - target[1], 2);
        const effort = Math.pow(a[0], 2) + Math.pow(a[1], 2);
        
        // Add Uncertainty Penalty to Cost
        cost += (distSq * weights.q) + (effort * weights.r) + (variance * UNCERTAINTY_LAMBDA);
        
        if (h === 0 && s === 0) totalUncertaintyAtStep0 = variance;
      }
      trajectories.push({ cost, sequence });
    }

    trajectories.sort((a, b) => a.cost - b.cost);
    const elites = trajectories.slice(0, ELITES);

    // Update Sampling Distribution
    for (let h = 0; h < HORIZON; h++) {
      const hMeanX = elites.reduce((s, e) => s + e.sequence[h][0], 0) / ELITES;
      const hMeanY = elites.reduce((s, e) => s + e.sequence[h][1], 0) / ELITES;
      currentMean[h] = [hMeanX, hMeanY];
      
      const hStdX = Math.sqrt(elites.reduce((s, e) => s + Math.pow(e.sequence[h][0] - hMeanX, 2), 0) / ELITES) + 0.05;
      const hStdY = Math.sqrt(elites.reduce((s, e) => s + Math.pow(e.sequence[h][1] - hMeanY, 2), 0) / ELITES) + 0.05;
      currentStd[h] = [hStdX, hStdY];
    }
  }

  previousOptimalSequence = currentMean;
  return { 
    action: currentMean[0], 
    ensembleUncertainty: totalUncertaintyAtStep0 
  };
};
