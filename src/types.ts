
export enum SimMode {
  MPC_STABILIZATION = 'MPC_STABILIZATION',
  SYSTEM_IDENTIFICATION = 'SYSTEM_IDENTIFICATION',
  BENCHMARK_SHIFT = 'BENCHMARK_SHIFT'
}

/**
 * State Vector — variable length depending on platform and DOF.
 * Examples:
 *   balancing_bot:   length 4   [pitch, pitch_rate, x, v]
 *   ground_rover:    length 6   [x, y, theta, vx, vy, omega]
 *   quadrotor:       length 13  [x,y,z, vx,vy,vz, qw,qx,qy,qz, p,q,r]
 *   manipulator(6):  length 12  [q0..q5, dq0..dq5]
 *   manipulator(7):  length 14  [q0..q6, dq0..dq6]
 *   legged(12 DOF):  length 30  [base(6), q0..q11, dq0..dq11]
 *   humanoid(36DOF): length 78  [base(6), q0..q35, dq0..dq35]
 */
export type StateVector = number[];

/**
 * Control Input — variable length depending on platform and DOF.
 * Examples:
 *   balancing_bot:  length 1   [torque]
 *   ground_rover:   length 2   [v_left, v_right]
 *   quadrotor:      length 4   [thrust, roll_cmd, pitch_cmd, yaw_cmd]
 *   manipulator(6): length 6   [tau0..tau5]
 *   manipulator(7): length 7   [tau0..tau6]
 *   dual_arm(7+7):  length 14  [tau_L0..tau_L6, tau_R0..tau_R6]
 */
export type ControlInput = number[];

export interface PhysicalParams {
  mass: number;
  friction: number;
  gravity: number;
  textile_k: number;
  damping: number;
  // High-DOF extensions
  dof?: number;
  link_masses?: number[];
  link_lengths?: number[];
  joint_frictions?: number[];
  link_inertias?: number[];
}

export interface PlatformDOFConfig {
  platform: string;
  dof: number;
  state_dim: number;
  action_dim: number;
  joint_names?: string[];
  joint_limits_lo?: number[];
  joint_limits_hi?: number[];
}

export interface JointState {
  positions: number[];
  velocities: number[];
  efforts: number[];
  names: string[];
  n_joints: number;
}

export interface SimState {
  current: StateVector;
  target: number[];
  estimatedParams: PhysicalParams;
  predictionError: number;
  controlEffort: number;
  stability: number;
  time: number;
  controlAction: ControlInput;
  uncertainty: number;
  isBenchmarking: boolean;
  // High-DOF additions
  jointState?: JointState;
  platformDOF?: PlatformDOFConfig;
  dof?: number;
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
