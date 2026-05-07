import { GoogleGenAI, Type } from "@google/genai";
import { SimState, MetaAnalysisResponse } from "../types";

// PhysiCore Meta-Analyst — Gemini direct from browser
// Key source: VITE_GEMINI_API_KEY in Vercel environment variables
// No localhost dependency. Degrades gracefully if key missing.

function getKey(): string {
  return (import.meta as any).env?.VITE_GEMINI_API_KEY || "";
}

export const performMetaAnalysis = async (
  state: SimState,
  history: SimState[]
): Promise<MetaAnalysisResponse> => {
  const key = getKey();
  if (!key) {
    // No key — return local deterministic analysis
    const err  = state.predictionError;
    const mass = state.estimatedParams.mass;
    const stab = state.stability;
    return {
      insight: err > 0.01
        ? `Prediction error elevated (${err.toFixed(5)}) — SysID adapting. Mass: ${mass.toFixed(3)}kg.`
        : `Model nominal. Mass: ${mass.toFixed(3)}kg, stability: ${stab.toFixed(0)}%.`,
      diagnostics: [
        `Prediction error: ${err.toFixed(5)}`,
        `Estimated mass: ${mass.toFixed(3)}kg`,
        `Stability: ${stab.toFixed(0)}%`,
      ],
      suggestedCostTweaks: {
        q_weight: stab < 50 ? 2.0 : 1.0,
        r_weight: err > 0.05 ? 0.05 : 0.1,
      },
    };
  }

  const ai = new GoogleGenAI({ apiKey: key });

  const prompt = `PhysiCore Meta-Analyst session data:
Estimated mass: ${state.estimatedParams.mass.toFixed(3)}kg
Estimated friction: ${state.estimatedParams.friction.toFixed(3)}
Prediction L2 error: ${state.predictionError.toFixed(5)}
Control effort: ${state.controlEffort.toFixed(3)}
Stability: ${state.stability.toFixed(0)}%
Uncertainty: ${state.uncertainty.toFixed(4)}

Return JSON: { insight: string, diagnostics: string[], suggestedCostTweaks: { q_weight: number, r_weight: number } }`;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            insight: { type: Type.STRING },
            diagnostics: { type: Type.ARRAY, items: { type: Type.STRING } },
            suggestedCostTweaks: {
              type: Type.OBJECT,
              properties: {
                q_weight: { type: Type.NUMBER },
                r_weight: { type: Type.NUMBER },
              },
              required: ["q_weight", "r_weight"],
            },
          },
          required: ["insight", "diagnostics", "suggestedCostTweaks"],
        },
      },
    });

    const text = response.text;
    if (!text) throw new Error("Empty response");
    return JSON.parse(text.trim()) as MetaAnalysisResponse;
  } catch (e: any) {
    if (e.message?.includes("429")) throw new Error("QUOTA_EXHAUSTED");
    // Graceful fallback on any error
    return {
      insight: `Analysis unavailable (${e.message?.slice(0,40) || "error"}). Local: mass=${state.estimatedParams.mass.toFixed(3)}kg, error=${state.predictionError.toFixed(5)}.`,
      diagnostics: [`Prediction error: ${state.predictionError.toFixed(5)}`, `Mass: ${state.estimatedParams.mass.toFixed(3)}kg`],
      suggestedCostTweaks: { q_weight: 1.0, r_weight: 0.1 },
    };
  }
};