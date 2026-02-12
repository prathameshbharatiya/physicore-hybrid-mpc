
export enum SimMode {
  MPC_STABILIZATION = 'MPC_STABILIZATION',
  SYSTEM_IDENTIFICATION = 'SYSTEM_IDENTIFICATION',
  BENCHMARK_SHIFT = 'BENCHMARK_SHIFT'
}

export type StateVector = [number, number, number, number, number, number];
export type ControlInput = [number, number];

export interface PhysicalParams {
  mass: number;
  friction: number;
  gravity: number;
}

export interface SimState {
  current: StateVector;
  target: [number, number];
  estimatedParams: PhysicalParams;
  predictionError: number;
  controlEffort: number;
  stability: number;
  time: number;
  controlAction: ControlInput;
  uncertainty: number; // Epistemic uncertainty from ensemble
  isBenchmarking: boolean;
}

export interface TelemetryPoint {
  time: number;
  value: number;
  label: string;
}

export interface MetaAnalysisResponse {
  insight: string;
  diagnostics: string[];
  suggestedCostTweaks: {
    q_weight: number;
    r_weight: number;
  };
}
