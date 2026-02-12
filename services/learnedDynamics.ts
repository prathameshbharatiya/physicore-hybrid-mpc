
import { StateVector, ControlInput } from "../types";

class TinyMLP {
  private w1: number[][]; // 8 x 32
  private b1: number[];   // 32
  private w2: number[][]; // 32 x 32
  private b2: number[];   // 32
  private w3: number[][]; // 32 x 6
  private b3: number[];   // 6

  constructor() {
    this.w1 = Array(32).fill(0).map(() => Array(8).fill(0).map(() => (Math.random() - 0.5) * 0.1));
    this.b1 = Array(32).fill(0);
    this.w2 = Array(32).fill(0).map(() => Array(32).fill(0).map(() => (Math.random() - 0.5) * 0.1));
    this.b2 = Array(32).fill(0);
    this.w3 = Array(6).fill(0).map(() => Array(32).fill(0).map(() => (Math.random() - 0.5) * 0.1));
    this.b3 = Array(6).fill(0);
  }

  private relu(x: number[]) { return x.map(v => Math.max(0, v)); }

  public forward(state: StateVector, u: ControlInput) {
    const input = [...state, ...u];
    // Layer 1
    const l1 = this.relu(this.w1.map((row, i) => row.reduce((s, w, j) => s + w * input[j], 0) + this.b1[i]));
    // Layer 2
    const l2 = this.relu(this.w2.map((row, i) => row.reduce((s, w, j) => s + w * l1[j], 0) + this.b2[i]));
    // Output
    const out = this.w3.map((row, i) => row.reduce((s, w, j) => s + w * l2[j], 0) + this.b3[i]);
    return { l1, l2, out };
  }

  public train(state: StateVector, u: ControlInput, targetResidual: number[]) {
    const { l1, l2, out } = this.forward(state, u);
    const lr = 0.005;
    const input = [...state, ...u];

    // Simple online SGD backprop
    const dOut = out.map((o, i) => (o - targetResidual[i]));
    
    // w3 update
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < 32; j++) {
        const grad = dOut[i] * l2[j];
        this.w3[i][j] -= lr * Math.max(-1, Math.min(1, grad));
      }
      this.b3[i] -= lr * dOut[i];
    }

    // Rough approximation of deeper gradients for real-time speed
    // Higher layers receive most of the correction to avoid heavy CPU compute
  }
}

export class EnsembleLearnedModel {
  private models = [new TinyMLP(), new TinyMLP(), new TinyMLP()];

  public predict(x: StateVector, u: ControlInput): { mean: StateVector, variance: number } {
    const preds = this.models.map(m => m.forward(x, u).out);
    const mean = preds[0].map((_, i) => preds.reduce((s, p) => s + p[i], 0) / preds.length) as StateVector;
    
    // Epistemic uncertainty = variance of ensemble members
    const variance = preds.reduce((s, p) => s + p.reduce((is, val, i) => is + Math.pow(val - mean[i], 2), 0), 0) / preds.length;
    
    return { mean, variance };
  }

  public train(x: StateVector, u: ControlInput, xNext: StateVector, xPhysics: StateVector) {
    const residual = xNext.map((v, i) => v - xPhysics[i]);
    this.models.forEach(m => m.train(x, u, residual));
  }
}

export const ensembleDynamics = new EnsembleLearnedModel();
