
import { StateVector, PhysicalParams } from "../types";

export interface ResidualReport {
  pattern: string;
  forceError: number;
  confidence: number;
}

export const calculateResiduals = (predicted: StateVector, actual: StateVector) => {
  return {
    posError: Math.sqrt(Math.pow(actual[0] - predicted[0], 2) + Math.pow(actual[1] - predicted[1], 2)),
    velError: Math.sqrt(Math.pow(actual[2] - predicted[2], 2) + Math.pow(actual[3] - predicted[3], 2))
  };
};

export const classifyPattern = (history: number[]): string => {
  if (history.length < 5) return "nominal";
  const recent = history.slice(-10);
  const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
  
  if (mean > 2.0) return "systematic_overprediction";
  if (mean < -2.0) return "systematic_underprediction";
  
  // Simplified oscillation check
  let signChanges = 0;
  for (let i = 1; i < recent.length; i++) {
    if ((recent[i] >= 0 && recent[i-1] < 0) || (recent[i] < 0 && recent[i-1] >= 0)) {
      signChanges++;
    }
  }
  if (signChanges > 4) return "oscillation";
  
  return "nominal";
};
