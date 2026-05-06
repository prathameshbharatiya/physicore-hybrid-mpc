
export enum SimMode {
  MPC_STABILIZATION = 'MPC_STABILIZATION',
  SYSTEM_IDENTIFICATION = 'SYSTEM_IDENTIFICATION',
  BENCHMARK_SHIFT = 'BENCHMARK_SHIFT'
}

/** 
 * State Vector [x, y, vx, vy, theta, omega]
 */
export type StateVector = [number, number, number, number, number, number];

/**
 * Control Input [Fx, Fy]
 */
export type ControlInput = [number, number];

export interface PhysicalParams {
  mass: number;
  friction: number;
  gravity: number;
  textile_k: number;
  damping: number;
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
  uncertainty: number;
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
  recommendations?: {
    parameter: string;
    value: number;
    rationale: string;
  }[];
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
}

export interface GeneratedFile {
  filename: string;
  content: string;
}

export interface CustomExtension {
  id: string;
  name: string;
  description: string;
  code: string;
  createdAt: string;
}

export interface Project {
  id: string;
  name: string;
  description: string;
  hardware: string;
  platform: string;
  answers: Record<string, string>;
  generatedFiles: GeneratedFile[];
  customExtensions: CustomExtension[];
  createdAt: string;
  updatedAt: string;
  lastBridgeSession?: string;
  registryPlatformKey: string;
  connectionMode: 'ros2_websocket' | 'hil' | 'digital_twin' | 'mavlink_bridge';
  endpoint: string;
  notes: string;
}

export interface FailureLog {
  id: string;
  timestamp: number;
  task: string;
  failure_type: string;
  sim_params: PhysicalParams;
  diagnosis?: string;
  fix_applied?: boolean;
}
