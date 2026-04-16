
import { GoogleGenerativeAI } from "@google/generative-ai";
import { SimState, MetaAnalysisResponse } from "../types";

export const performMetaAnalysis = async (
  state: SimState,
  history: SimState[]
): Promise<MetaAnalysisResponse> => {
  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");
  const model = genAI.getGenerativeModel({ 
    model: "gemini-1.5-flash",
    generationConfig: {
      responseMimeType: "application/json",
    }
  });
  
  const prompt = `
    System: You are the "PhysiCore Meta-Analyst".
    Role: Interpret telemetry and diagnostics from a model-based predictive control (MPC) system.
    
    Diagnostics:
    - Target: [${state.target[0]}, ${state.target[1]}]
    - Estimated Mass: ${state.estimatedParams.mass.toFixed(3)}
    - Estimated Friction: ${state.estimatedParams.friction.toFixed(3)}
    - Prediction L2 Loss: ${state.predictionError.toFixed(5)}
    - Control Effort: ${state.controlEffort.toFixed(3)}
    - Mode: ${state.stability > 80 ? 'Stable Convergence' : 'Oscillatory/Unstable'}
    
    Analyze the trajectory history and parameter drift. Identify if the model is over-damped or if the system ID is diverging.
    
    Return strict JSON with specific suggested Q/R weights for the MPC cost function.
    
    Required JSON structure:
    {
      "insight": "high-level summary",
      "diagnostics": ["point 1", "point 2"],
      "suggestedCostTweaks": {
        "q_weight": 1.0,
        "r_weight": 0.1
      }
    }
  `;

  try {
    const result = await model.generateContent(prompt);
    const text = result.response.text();
    if (!text) throw new Error("Empty response from meta-analyst");
    return JSON.parse(text.trim()) as MetaAnalysisResponse;
  } catch (e: any) {
    if (e.message?.includes('429')) {
      console.warn("Meta-analyst rate limit exceeded (429).");
      throw new Error("QUOTA_EXHAUSTED");
    }
    console.error("Meta-analyst uplink failed", e);
    throw new Error("ANALYSIS_FAILED");
  }
};
