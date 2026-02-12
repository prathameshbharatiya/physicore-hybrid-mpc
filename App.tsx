
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
  
  // Physics & Control State
  const [target, setTarget] = useState<[number, number]>([400, 300]);
  const [controlAction, setControlAction] = useState<ControlInput>([0, 0]);
  const [physicsPriors, setPhysicsPriors] = useState({ mass: 1.0, friction: 0.1, gravity: 0.5 });
  const [costWeights, setCostWeights] = useState({ q: 1.5, r: 0.05 });
  const [uncertainty, setUncertainty] = useState(0);

  const historyRef = useRef<SimState[]>([]);
  const lastStateRef = useRef<StateVector | null>(null);
  const benchmarkRef = useRef<{ active: boolean, startTime: number }>({ active: false, startTime: 0 });

  // Main Numerical Control Loop (100Hz)
  const handleStateUpdate = useCallback((state: SimState) => {
    if (lastStateRef.current) {
      // 1. Train Ensemble Residual Model (Physics Ground Truth vs Actual)
      const xPhys = stepDynamicsRK4(lastStateRef.current, controlAction, physicsPriors);
      ensembleDynamics.train(lastStateRef.current, controlAction, state.current, xPhys);

      // 2. Perform Online System Identification
      const updatedParams = updateSystemID(
        lastStateRef.current,
        controlAction,
        state.current,
        physicsPriors
      );
      setPhysicsPriors(updatedParams);
    }

    // 3. Compute MPC action with Uncertainty Awareness
    const { action, ensembleUncertainty } = computeMPCAction(state.current, target, physicsPriors, costWeights);
    setControlAction(action);
    setUncertainty(ensembleUncertainty);

    // 4. Update state tracking
    const predError = lastStateRef.current ? Math.sqrt(state.current.reduce((s, v, i) => s + Math.pow(v - stepDynamicsRK4(lastStateRef.current!, controlAction, physicsPriors)[i], 2), 0)) : 0;
    
    lastStateRef.current = state.current;
    const enrichedState = { 
      ...state, 
      predictionError: predError, 
      uncertainty: ensembleUncertainty,
      isBenchmarking: benchmarkRef.current.active
    };
    
    setSimState(enrichedState);
    historyRef.current = [...historyRef.current.slice(-100), enrichedState];
    
    setTelemetry(prev => [...prev.slice(-60), {
      time: state.time,
      value: predError,
      label: 'L2 Residual'
    }]);
  }, [target, physicsPriors, controlAction, costWeights]);

  // Environmental Shift Benchmark
  const triggerFrictionShift = () => {
    benchmarkRef.current = { active: true, startTime: Date.now() };
    setPhysicsPriors(prev => ({ ...prev, friction: prev.friction > 0.5 ? 0.05 : 0.8 }));
    setTimeout(() => { benchmarkRef.current.active = false; }, 10000);
  };

  useEffect(() => {
    const interval = setInterval(async () => {
      if (!simState || isAnalyzing) return;
      setIsAnalyzing(true);
      try {
        const result = await performMetaAnalysis(simState, historyRef.current);
        setMetaAnalysis(result);
        if (result.suggestedCostTweaks) {
          setCostWeights({
            q: result.suggestedCostTweaks.q_weight,
            r: result.suggestedCostTweaks.r_weight
          });
        }
      } catch (err) { console.error(err); } 
      finally { setIsAnalyzing(false); }
    }, 10000);
    return () => clearInterval(interval);
  }, [simState, isAnalyzing]);

  return (
    <div className="flex h-screen w-screen bg-black text-slate-400 overflow-hidden font-mono selection:bg-blue-900/50">
      <aside className="w-80 bg-slate-950 border-r border-white/5 flex flex-col p-6 shadow-2xl z-20">
        <header className="mb-8">
          <div className="flex items-center gap-2 text-white font-black text-xl italic tracking-tighter">
            <div className="w-8 h-8 bg-indigo-700 rounded-sm flex items-center justify-center not-italic text-sm">Φ</div>
            PHYSICORE <span className="text-indigo-500 font-light">HYBRID</span>
          </div>
          <p className="text-[9px] text-slate-600 mt-2 uppercase tracking-widest">Ensemble Predictive Control</p>
        </header>

        <div className="flex-1 space-y-6 overflow-y-auto pr-2 custom-scroll">
          <section className="bg-slate-900/40 p-4 rounded border border-white/5 space-y-4">
             <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Robust System ID</h3>
             <div className="space-y-3">
               <div>
                 <div className="flex justify-between text-[9px]"><span>Mass Estimate</span><span className="text-white">{physicsPriors.mass.toFixed(4)}kg</span></div>
                 <div className="w-full bg-slate-800 h-1 mt-1"><div className="bg-indigo-500 h-full transition-all duration-300" style={{ width: `${(physicsPriors.mass / 5) * 100}%` }}></div></div>
               </div>
               <div>
                 <div className="flex justify-between text-[9px]"><span>Friction (μ)</span><span className="text-white">{physicsPriors.friction.toFixed(4)}</span></div>
                 <div className="w-full bg-slate-800 h-1 mt-1"><div className="bg-indigo-500 h-full transition-all duration-300" style={{ width: `${physicsPriors.friction * 100}%` }}></div></div>
               </div>
             </div>
          </section>

          <section className="bg-indigo-900/10 border border-indigo-500/20 p-4 rounded">
             <h3 className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest mb-3">Epistemic Uncertainty</h3>
             <div className="flex items-end gap-1 h-12">
               {telemetry.slice(-20).map((t, i) => (
                 <div key={i} className="flex-1 bg-indigo-600/30" style={{ height: `${Math.min(100, (uncertainty * 1000) * (i/20))}%` }}></div>
               ))}
             </div>
             <p className="text-[9px] text-indigo-300 mt-2">Ensemble Variance: <span className="font-bold">{uncertainty.toExponential(3)}</span></p>
          </section>

          <section>
             <h3 className="text-[10px] font-bold text-slate-700 uppercase tracking-widest mb-4">Neural Residuals</h3>
             <div className="grid grid-cols-2 gap-2 text-[10px]">
               <div className="p-3 bg-slate-900/50 border border-white/5">
                 <span className="text-slate-600 block mb-1">State Gain (Q)</span>
                 <span className="text-white">{costWeights.q.toFixed(2)}</span>
               </div>
               <div className="p-3 bg-slate-900/50 border border-white/5">
                 <span className="text-slate-600 block mb-1">Actuation (R)</span>
                 <span className="text-white">{costWeights.r.toFixed(2)}</span>
               </div>
             </div>
          </section>
        </div>

        <footer className="mt-8 pt-6 border-t border-white/5 space-y-3">
           <button 
             onClick={triggerFrictionShift}
             className="w-full py-3 bg-indigo-700 text-white text-[10px] font-bold uppercase hover:bg-indigo-600 transition-all flex items-center justify-center gap-2"
           >
             🔥 Shift Environmental Friction
           </button>
           <button 
             onClick={() => setPhysicsPriors({ mass: 1, friction: 0.1, gravity: 0.5 })}
             className="w-full py-2 border border-slate-800 text-slate-500 text-[9px] font-bold uppercase hover:border-slate-700"
           >
             Reset Beliefs
           </button>
        </footer>
      </aside>

      <main className="flex-1 flex flex-col p-8 relative">
        <div className="flex-1 relative mb-8 group">
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
            <div className="absolute inset-0 bg-indigo-500/5 pointer-events-none flex items-center justify-center">
              <div className="border-4 border-indigo-500 p-8 text-indigo-500 font-black text-4xl animate-pulse">
                BENCHMARKING ADAPTATION...
              </div>
            </div>
          )}

          <div className="absolute top-6 left-6 pointer-events-none space-y-2">
            <div className="bg-black/80 border border-white/10 p-4 rounded-sm backdrop-blur-md">
              <div className="text-[10px] text-indigo-400 font-bold mb-1 tracking-widest uppercase">Hybrid Dynamics Core</div>
              <div className="text-[9px] text-slate-500 space-y-1">
                <div>SOLVER: RK4 (4th Order)</div>
                <div>RESIDUAL: Ensemble MLP (32x32)</div>
                <div>UNCERTAINTY: Epistemic Variance</div>
              </div>
            </div>
          </div>
        </div>

        <div className="h-64 grid grid-cols-12 gap-6 shrink-0">
           <div className="col-span-7 h-full">
             <Dashboard telemetry={telemetry} avgVelocity={simState?.current[2] || 0} stability={simState?.stability || 0} />
           </div>

           <div className="col-span-5 bg-slate-950 border border-white/5 rounded-sm p-5 flex flex-col overflow-hidden">
              <div className="flex justify-between items-center mb-4 border-b border-white/5 pb-2">
                <h3 className="text-[10px] font-black text-slate-300 tracking-widest uppercase italic">Meta-Analyst Diagnostic</h3>
                {isAnalyzing && <div className="flex gap-1"><div className="w-1 h-1 bg-indigo-600 animate-bounce"></div><div className="w-1 h-1 bg-indigo-600 animate-bounce delay-75"></div></div>}
              </div>

              <div className="flex-1 overflow-y-auto space-y-4 pr-2 custom-scroll">
                {metaAnalysis ? (
                  <>
                    <p className="text-[11px] text-indigo-200 leading-relaxed italic border-l-2 border-indigo-700 pl-3">
                      "{metaAnalysis.insight}"
                    </p>
                    <div className="space-y-2">
                       {metaAnalysis.diagnostics.map((d, i) => (
                         <div key={i} className="flex gap-2 text-[10px] text-slate-600">
                           <span className="text-indigo-900 font-bold">#</span> {d}
                         </div>
                       ))}
                    </div>
                  </>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center opacity-10">
                    <span className="text-[9px] uppercase tracking-widest font-bold">Observing Numerical Flux...</span>
                  </div>
                )}
              </div>
           </div>
        </div>
      </main>
      
      <style>{`
        .custom-scroll::-webkit-scrollbar { width: 4px; }
        .custom-scroll::-webkit-scrollbar-track { background: transparent; }
        .custom-scroll::-webkit-scrollbar-thumb { background: #312e81; }
      `}</style>
    </div>
  );
};

export default App;
