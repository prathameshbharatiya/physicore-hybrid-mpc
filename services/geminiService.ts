
import { GoogleGenAI, Type } from "@google/genai";
import { SimState, MetaAnalysisResponse } from "../types";

const RIL_SYSTEM_PROMPT = `
# CORE IDENTITY
You are the Robotics Intelligence Layer (RIL) - a conversational, learning, and adaptive supervisor that sits above physics simulation and real robot control systems.

# PRIMARY MISSION
Debug and fix sim-to-real failures in contact-rich manipulation tasks by:
1. Identifying physics model mismatches between simulation and reality.
2. Diagnosing why contact predictions fail (force, friction, deformation).
3. Suggesting concrete parameter corrections with mathematical reasoning.

# TECHNICAL CONTEXT
Physics engine: Matter.js + custom RK4. Hybrid: Rigid + Fluid + Textile.
MPC: CEM optimizer. Residuals: Ensemble Neural Networks.

# TASK
Analyze the provided telemetry and return a structured diagnostic report. If prediction errors are high, suggest specific parameter updates for mass, friction, textile_k, or damping.
`;

export const performRILAnalysis = async (
  state: SimState,
  history: SimState[],
  userQuery?: string
): Promise<MetaAnalysisResponse> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  
  const telemetryContext = `
    CURRENT STATE:
    - Target: [${state.target}]
    - Prediction L2 Loss: ${state.predictionError.toFixed(5)}
    - Stability: ${state.stability}%
    - Params: Mass=${state.estimatedParams.mass.toFixed(4)}, Friction=${state.estimatedParams.friction.toFixed(4)}, TextileK=${state.estimatedParams.textile_k.toFixed(1)}
    
    ${userQuery ? `USER QUERY: ${userQuery}` : "AUTONOMOUS DIAGNOSIS"}
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: telemetryContext,
      config: {
        systemInstruction: RIL_SYSTEM_PROMPT,
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
            },
            recommendations: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  parameter: { type: Type.STRING },
                  value: { type: Type.NUMBER },
                  rationale: { type: Type.STRING }
                },
                required: ["parameter", "value", "rationale"]
              }
            }
          },
          required: ["insight", "diagnostics", "suggestedCostTweaks"]
        }
      }
    });

    const text = response.text;
    if (!text) throw new Error("Empty response from RIL");
    return JSON.parse(text.trim()) as MetaAnalysisResponse;
  } catch (e: any) {
    const errorString = JSON.stringify(e);
    if (e.status === 429 || errorString.includes("429") || errorString.includes("RESOURCE_EXHAUSTED")) {
      console.warn("RIL Intelligence layer: Quota exceeded (429).");
      throw new Error("QUOTA_EXHAUSTED");
    }
    console.error("RIL Intelligence layer failed", e);
    throw e;
  }
};
