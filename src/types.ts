
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
  features?: FeatureManifest[];
  ieProgress?: {
    phase: string;
    hw: string;
    qIndex: number;
    answers: Record<string, string>;
  };
}

// Feature Manifest — the contract that ties a custom feature to the whole system
export interface FeatureManifest {
  id: string;
  name: string;
  description: string;
  // Telemetry keys this feature produces — dashboard subscribes to these
  telemetry_keys: string[];
  // New fault types this feature can produce — debugger learns these
  fault_types: string[];
  // Which PhysiCore hooks this feature uses
  hooks: ('pre_step' | 'post_step' | 'on_fault' | 'on_telemetry')[];
  // Files this feature modified (relative paths)
  files_modified: string[];
  // The conversation that generated this feature
  conversation: { role: 'user' | 'assistant'; text: string }[];
  // Generated file contents keyed by filename
  generated_files: Record<string, string>;
  createdAt: string;
}

// Extend Project with feature manifests and IE progress
export interface ProjectV2 extends Project {
  features: FeatureManifest[];
  ieProgress?: {
    phase: string;
    hw: string;
    qIndex: number;
    answers: Record<string, string>;
  };
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
