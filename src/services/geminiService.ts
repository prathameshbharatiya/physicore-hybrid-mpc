
import { GoogleGenAI, Type } from "@google/genai";
import { SimState, MetaAnalysisResponse } from "../types";

let aiClient: GoogleGenAI | null = null;

const getAiClient = () => {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is missing. AI features will be disabled.");
    }
    aiClient = new GoogleGenAI({ apiKey });
  }
  return aiClient;
};

export const performMetaAnalysis = async (
  state: SimState,
  history: SimState[]
): Promise<MetaAnalysisResponse> => {
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
  `;

  try {
    const ai = getAiClient();
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            insight: { type: Type.STRING },
            diagnostics: { 
              type: Type.ARRAY,
              items: { type: Type.STRING }
            },
            suggestedCostTweaks: {
              type: Type.OBJECT,
              properties: {
                q_weight: { type: Type.NUMBER },
                r_weight: { type: Type.NUMBER }
              },
              required: ["q_weight", "r_weight"]
            }
          },
          required: ["insight", "diagnostics", "suggestedCostTweaks"]
        }
      }
    });

    const text = response.text;
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
