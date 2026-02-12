
import React, { useState, useEffect, useCallback, useRef } from 'react';
import SimulationCanvas from './components/SimulationCanvas';
import Dashboard from './components/Dashboard';
import { SimMode, SimState, TelemetryPoint, StateVector, ControlInput, MetaAnalysisResponse } from './types';
import { computeMPCAction } from './services/optimizer';
import { updateSystemID } from './services/systemID';
import { ensembleDynamics } from './services/learnedDynamics';
import { performMetaAnalysis } from './services/geminiService';
import { stepDynamicsRK4 } from './services/physicsLogic';

const App: React.FC = () => {
  const [mode, setMode] = useState<SimMode>(SimMode.MPC_STABILIZATION);
  const [simState, setSimState] = useState<SimState | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetryPoint[]>([]);
  const [metaAnalysis, setMetaAnalysis] = useState<MetaAnalysisResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  const [target, setTarget] = useState<[number, number]>([400, 300]);
  const [controlAction, setControlAction] = useState<ControlInput>([0, 0]);
  const [physicsPriors, setPhysicsPriors] = useState({ mass: 1.0, friction: 0.1, gravity: 0.5 });
  const [costWeights, setCostWeights] = useState({ q: 1.5, r: 0.05 });
  const [uncertainty, setUncertainty] = useState(0);

  const historyRef = useRef<SimState[]>([]);
  const lastStateRef = useRef<StateVector | null>(null);
  const benchmarkRef = useRef<{ active: boolean, startTime: number }>({ active: false, startTime: 0 });

  const handleStateUpdate = useCallback((state: SimState) => {
    if (lastStateRef.current) {
      const xPhys = stepDynamicsRK4(lastStateRef.current, controlAction, physicsPriors);
      ensembleDynamics.train(lastStateRef.current, controlAction, state.current, xPhys);

      const updatedParams = updateSystemID(
        lastStateRef.current,
        controlAction,
        state.current,
        physicsPriors
      );
      setPhysicsPriors(updatedParams);
    }

    const { action, ensembleUncertainty } = computeMPCAction(state.current, target, physicsPriors, costWeights);
    setControlAction(action);
    setUncertainty(ensembleUncertainty);

    const predError = lastStateRef.current ? Math.sqrt(state.current.reduce((s, v, i) => s + Math.pow(v - stepDynamicsRK4(lastStateRef.current!, controlAction, physicsPriors)[i], 2), 0)) : 0;
    
    lastStateRef.current = state.current;
    const enrichedState = { 
      ...state, 
      predictionError: predError, 
      uncertainty: ensembleUncertainty,
      isBenchmarking: benchmarkRef.current.active
    };
    
    setSimState(enrichedState);
    historyRef.current = [...historyRef.current.slice(-60), enrichedState];
    
    setTelemetry(prev => [...prev.slice(-40), {
      time: state.time,
      value: predError,
      label: 'L2 Residual'
    }]);
  }, [target, physicsPriors, controlAction, costWeights]);

  const triggerFrictionShift = () => {
    benchmarkRef.current = { active: true, startTime: Date.now() };
    setPhysicsPriors(prev => ({ ...prev, friction: prev.friction > 0.4 ? 0.05 : 0.7 }));
    setTimeout(() => { benchmarkRef.current.active = false; }, 8000);
  };

  // Fixed MetaAnalysis Effect: Use an interval that doesn't clear on every simState update
  useEffect(() => {
    const triggerAnalysis = async () => {
      // Use the latest history value from the ref to avoid dependency on simState
      if (historyRef.current.length < 10 || isAnalyzing) return;
      
      setIsAnalyzing(true);
      try {
        const latestState = historyRef.current[historyRef.current.length - 1];
        const result = await performMetaAnalysis(latestState, historyRef.current);
        setMetaAnalysis(result);
        if (result.suggestedCostTweaks) {
          setCostWeights({
            q: result.suggestedCostTweaks.q_weight,
            r: result.suggestedCostTweaks.r_weight
          });
        }
      } catch (err) {
        console.error("Meta-analyst failed:", err);
      } finally {
        setIsAnalyzing(false);
      }
    };

    const interval = setInterval(triggerAnalysis, 15000);
    return () => clearInterval(interval);
  }, []); // Only run once on mount

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-400 overflow-hidden font-mono">
      <aside className="w-80 bg-slate-900/50 border-r border-white/5 flex flex-col p-6 shadow-2xl z-20">
        <header className="mb-8">
          <div className="flex items-center gap-2 text-white font-black text-xl italic tracking-tighter">
            <div className="w-8 h-8 bg-indigo-700 rounded-sm flex items-center justify-center not-italic text-sm shadow-[0_0_15px_rgba(79,70,229,0.4)]">Φ</div>
            PHYSICORE <span className="text-indigo-400 font-light">HYBRID</span>
          </div>
          <p className="text-[9px] text-slate-600 mt-2 uppercase tracking-widest">Ensemble Predictive Control</p>
        </header>

        <div className="flex-1 space-y-6 overflow-y-auto pr-2 custom-scroll">
          <section className="bg-slate-900/80 p-4 rounded border border-white/5 space-y-4 shadow-inner">
             <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Robust System ID</h3>
             <div className="space-y-3">
               <div>
                 <div className="flex justify-between text-[9px] mb-1"><span>Mass Estimate</span><span className="text-white">{physicsPriors.mass.toFixed(4)}kg</span></div>
                 <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden"><div className="bg-indigo-500 h-full transition-all duration-300" style={{ width: `${(physicsPriors.mass / 5) * 100}%` }}></div></div>
               </div>
               <div>
                 <div className="flex justify-between text-[9px] mb-1"><span>Friction (μ)</span><span className="text-white">{physicsPriors.friction.toFixed(4)}</span></div>
                 <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden"><div className="bg-indigo-500 h-full transition-all duration-300" style={{ width: `${physicsPriors.friction * 100}%` }}></div></div>
               </div>
             </div>
          </section>

          <section className="bg-indigo-950/20 border border-indigo-500/20 p-4 rounded shadow-sm">
             <h3 className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest mb-3">Epistemic Uncertainty</h3>
             <div className="flex items-end gap-1 h-12 bg-black/20 rounded p-1">
               {telemetry.slice(-24).map((t, i) => (
                 <div key={i} className="flex-1 bg-indigo-600/40 rounded-t-sm" style={{ height: `${Math.min(100, (uncertainty * 500) + (i * 2))}%` }}></div>
               ))}
             </div>
             <p className="text-[9px] text-indigo-300/70 mt-2 flex justify-between">
               <span>Variance Threshold</span>
               <span className="font-bold text-indigo-400">{uncertainty.toExponential(2)}</span>
             </p>
          </section>

          <section>
             <h3 className="text-[10px] font-bold text-slate-700 uppercase tracking-widest mb-4 px-1">Neural Residuals</h3>
             <div className="grid grid-cols-2 gap-2 text-[10px]">
               <div className="p-3 bg-slate-900/50 border border-white/5 rounded">
                 <span className="text-slate-600 block mb-1">State Gain (Q)</span>
                 <span className="text-white font-bold">{costWeights.q.toFixed(2)}</span>
               </div>
               <div className="p-3 bg-slate-900/50 border border-white/5 rounded">
                 <span className="text-slate-600 block mb-1">Actuation (R)</span>
                 <span className="text-white font-bold">{costWeights.r.toFixed(2)}</span>
               </div>
             </div>
          </section>
        </div>

        <footer className="mt-8 pt-6 border-t border-white/5 space-y-3">
           <button 
             onClick={triggerFrictionShift}
             className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white text-[10px] font-bold uppercase transition-all flex items-center justify-center gap-2 rounded shadow-lg shadow-indigo-900/20 active:scale-95"
           >
             🔥 Shift Environment
           </button>
           <button 
             onClick={() => {
                setPhysicsPriors({ mass: 1, friction: 0.1, gravity: 0.5 });
                historyRef.current = [];
                setTelemetry([]);
             }}
             className="w-full py-2 border border-slate-800 text-slate-600 text-[9px] font-bold uppercase hover:border-slate-700 hover:text-slate-400 transition-colors rounded"
           >
             Reset Beliefs
           </button>
        </footer>
      </aside>

      <main className="flex-1 flex flex-col p-8 relative bg-black/40 backdrop-blur-3xl">
        <div className="flex-1 relative mb-8 group rounded shadow-2xl overflow-hidden border border-white/5">
          <SimulationCanvas 
            mode={mode} 
            onStateUpdate={handleStateUpdate} 
            target={target}
            controlAction={controlAction}
            physicsPriors={physicsPriors}
          />
          
          <div className="absolute inset-0 cursor-crosshair"
            onMouseMove={(e) => { if (e.buttons === 1) {
              const rect = e.currentTarget.getBoundingClientRect();
              setTarget([e.clientX - rect.left, e.clientY - rect.top]);
            }}}
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              setTarget([e.clientX - rect.left, e.clientY - rect.top]);
            }}
          />

          {simState?.isBenchmarking && (
            <div className="absolute inset-0 bg-indigo-900/10 pointer-events-none flex items-center justify-center backdrop-blur-[1px]">
              <div className="bg-slate-950/80 border-2 border-indigo-500 px-10 py-6 text-indigo-400 font-black text-2xl animate-pulse shadow-2xl flex flex-col items-center">
                <span>BENCHMARKING ADAPTATION</span>
                <span className="text-[10px] mt-2 tracking-[0.3em] font-light opacity-60 italic">Shift Detected in System Flux</span>
              </div>
            </div>
          )}

          <div className="absolute top-6 left-6 pointer-events-none space-y-2">
            <div className="bg-slate-950/90 border border-white/10 p-4 rounded-sm backdrop-blur-xl shadow-2xl">
              <div className="text-[10px] text-indigo-400 font-bold mb-1 tracking-widest uppercase flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-ping"></span>
                Hybrid Core v2.4
              </div>
              <div className="text-[9px] text-slate-500 space-y-1 font-mono">
                <div>SOLVER: RK4 (4th Order)</div>
                <div>RESIDUAL: Ensemble MLP</div>
                <div>UNCERTAINTY: Epistemic</div>
              </div>
            </div>
          </div>
        </div>

        <div className="h-64 grid grid-cols-12 gap-6 shrink-0">
           <div className="col-span-7 h-full">
             <Dashboard telemetry={telemetry} avgVelocity={simState?.current[2] || 0} stability={simState?.stability || 0} />
           </div>

           <div className="col-span-5 bg-slate-900/30 border border-white/5 rounded-sm p-5 flex flex-col overflow-hidden shadow-inner">
              <div className="flex justify-between items-center mb-4 border-b border-white/5 pb-2">
                <h3 className="text-[10px] font-black text-slate-400 tracking-widest uppercase italic">Meta-Analyst Intelligence</h3>
                {isAnalyzing && <div className="flex gap-1"><div className="w-1 h-3 bg-indigo-600/50 animate-pulse"></div><div className="w-1 h-3 bg-indigo-600/50 animate-pulse delay-100"></div></div>}
              </div>

              <div className="flex-1 overflow-y-auto space-y-4 pr-2 custom-scroll">
                {metaAnalysis ? (
                  <>
                    <p className="text-[11px] text-indigo-200 leading-relaxed italic border-l-2 border-indigo-600/50 pl-4 py-1">
                      "{metaAnalysis.insight}"
                    </p>
                    <div className="space-y-2.5">
                       {metaAnalysis.diagnostics.map((d, i) => (
                         <div key={i} className="flex gap-2 text-[10px] text-slate-500 items-start">
                           <span className="text-indigo-700 font-bold mt-0.5">»</span> 
                           <span className="flex-1">{d}</span>
                         </div>
                       ))}
                    </div>
                  </>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center opacity-20">
                    <div className="w-12 h-12 border border-indigo-500/30 rounded-full animate-ping mb-4"></div>
                    <span className="text-[9px] uppercase tracking-widest font-bold">Synchronizing Diagnostics...</span>
                  </div>
                )}
              </div>
           </div>
        </div>
      </main>
      
      <style>{`
        .custom-scroll::-webkit-scrollbar { width: 4px; }
        .custom-scroll::-webkit-scrollbar-track { background: transparent; }
        .custom-scroll::-webkit-scrollbar-thumb { background: #312e81; border-radius: 2px; }
      `}</style>
    </div>
  );
};

export default App;
