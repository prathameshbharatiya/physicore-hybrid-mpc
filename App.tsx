
import React, { useState, useEffect, useCallback, useRef } from 'react';
import SimulationCanvas from './components/SimulationCanvas';
import Dashboard from './components/Dashboard';
import ChatInterface from './components/ChatInterface';
import ParameterEditor from './components/ParameterEditor';
import { SimMode, SimState, TelemetryPoint, StateVector, ControlInput, MetaAnalysisResponse, ChatMessage, FailureLog, PhysicalParams } from './types';
import { computeMPCAction } from './services/optimizer';
import { updateSystemID } from './services/systemID';
import { ensembleDynamics } from './services/learnedDynamics';
import { performRILAnalysis } from './services/geminiService';
import { stepDynamicsRK4 } from './services/physicsLogic';
import { logFailure, getFailures } from './services/failureDB';

const App: React.FC = () => {
  const [mode, setMode] = useState<SimMode>(SimMode.MPC_STABILIZATION);
  const [simState, setSimState] = useState<SimState | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetryPoint[]>([]);
  const [metaAnalysis, setMetaAnalysis] = useState<MetaAnalysisResponse | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isMatterLoaded, setIsMatterLoaded] = useState(false);
  const [quotaExceeded, setQuotaExceeded] = useState(false);
  const [showDocs, setShowDocs] = useState(false);
  
  const [target, setTarget] = useState<[number, number]>([400, 300]);
  const [controlAction, setControlAction] = useState<ControlInput>([0, 0]);
  const [physicsPriors, setPhysicsPriors] = useState<PhysicalParams>({ mass: 1.0, friction: 0.1, gravity: 0.5, textile_k: 400, damping: 0.15 });
  const [costWeights, setCostWeights] = useState({ q: 1.5, r: 0.05 });
  const [uncertainty, setUncertainty] = useState(0);

  const historyRef = useRef<SimState[]>([]);
  const lastStateRef = useRef<StateVector | null>(null);
  const benchmarkRef = useRef<{ active: boolean, startTime: number }>({ active: false, startTime: 0 });

  useEffect(() => {
    const checkMatter = () => {
      if ((window as any).Matter) {
        setIsMatterLoaded(true);
      } else {
        setTimeout(checkMatter, 100);
      }
    };
    checkMatter();
    
    getFailures().then(logs => {
      console.log(`Loaded ${logs.length} historical failure entries.`);
    });
  }, []);

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
      
      setPhysicsPriors(prev => ({ ...prev, mass: updatedParams.mass, friction: updatedParams.friction }));

      const predError = Math.sqrt(state.current.reduce((s, v, i) => s + Math.pow(v - xPhys[i], 2), 0));
      if (predError > 15.0 && !benchmarkRef.current.active) {
        const failure: FailureLog = {
          id: crypto.randomUUID(),
          timestamp: Date.now(),
          task: "manipulation_fold",
          failure_type: predError > 30 ? "CRITICAL_OVERSHOOT" : "L2_RESIDUAL_HIGH",
          sim_params: { ...physicsPriors }
        };
        logFailure(failure);
      }
    }

    const { action, ensembleUncertainty } = computeMPCAction(state.current, target, physicsPriors, costWeights);
    setControlAction(action);
    setUncertainty(ensembleUncertainty);

    const xPhysFinal = lastStateRef.current ? stepDynamicsRK4(lastStateRef.current, controlAction, physicsPriors) : state.current;
    const predError = lastStateRef.current ? Math.sqrt(state.current.reduce((s, v, i) => s + Math.pow(v - xPhysFinal[i], 2), 0)) : 0;
    
    lastStateRef.current = state.current;
    const enrichedState = { 
      ...state, 
      predictionError: predError, 
      uncertainty: ensembleUncertainty,
      isBenchmarking: benchmarkRef.current.active
    };
    
    setSimState(enrichedState);
    historyRef.current = [...historyRef.current.slice(-120), enrichedState];
    
    setTelemetry(prev => [...prev.slice(-60), {
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

  const handleQuery = async (query: string) => {
    if (isAnalyzing || quotaExceeded) return;
    
    const userMsg: ChatMessage = { role: 'user', content: query, timestamp: Date.now() };
    setMessages(prev => [...prev, userMsg]);
    setIsAnalyzing(true);
    
    try {
      const latestState = historyRef.current[historyRef.current.length - 1];
      const result = await performRILAnalysis(latestState, historyRef.current, query);
      
      setMetaAnalysis(result);
      
      if (result.suggestedCostTweaks) {
        setCostWeights({
          q: result.suggestedCostTweaks.q_weight,
          r: result.suggestedCostTweaks.r_weight
        });
      }

      const assistantMsg: ChatMessage = { 
        role: 'assistant', 
        content: result.insight + (result.recommendations?.length ? "\n\nApplying parameter corrections based on mathematical reasoning." : ""), 
        timestamp: Date.now() 
      };
      setMessages(prev => [...prev, assistantMsg]);
      
    } catch (err: any) {
      if (err.message === "QUOTA_EXHAUSTED") {
        setQuotaExceeded(true);
        setTimeout(() => setQuotaExceeded(false), 60000);
        setMessages(prev => [...prev, { role: 'assistant', content: "⚠️ API Quota exhausted. Intelligence layer suspended for 60s.", timestamp: Date.now() }]);
      } else {
        setMessages(prev => [...prev, { role: 'assistant', content: "Intelligence uplink interrupted. Local physics kernels still active.", timestamp: Date.now() }]);
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  useEffect(() => {
    const triggerAnalysis = async () => {
      const latestState = historyRef.current[historyRef.current.length - 1];
      const needsAnalysis = latestState?.predictionError > 5.0;

      if (historyRef.current.length < 40 || isAnalyzing || quotaExceeded || !needsAnalysis) return;
      
      setIsAnalyzing(true);
      try {
        const result = await performRILAnalysis(latestState, historyRef.current);
        setMetaAnalysis(result);
        if (result.suggestedCostTweaks) {
          setCostWeights({
            q: result.suggestedCostTweaks.q_weight,
            r: result.suggestedCostTweaks.r_weight
          });
        }
      } catch (err: any) {
        if (err.message === "QUOTA_EXHAUSTED") {
          setQuotaExceeded(true);
          setTimeout(() => setQuotaExceeded(false), 60000);
        }
      } finally {
        setIsAnalyzing(false);
      }
    };

    const interval = setInterval(triggerAnalysis, 60000);
    return () => clearInterval(interval);
  }, [isAnalyzing, quotaExceeded]);

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-400 overflow-hidden font-mono selection:bg-indigo-500/30">
      <aside className="w-80 bg-slate-900/50 border-r border-white/5 flex flex-col p-6 shadow-2xl z-20 shrink-0">
        <header className="mb-8">
          <div className="flex items-center gap-2 text-white font-black text-xl italic tracking-tighter">
            <div className="w-8 h-8 bg-indigo-700 rounded-sm flex items-center justify-center not-italic text-sm shadow-[0_0_20px_rgba(79,70,229,0.3)]">RIL</div>
            PHYSICORE <span className="text-indigo-400 font-light italic">INTELLIGENCE</span>
          </div>
          <p className="text-[9px] text-slate-600 mt-2 uppercase tracking-widest">Robotics Intelligence Layer v1.0</p>
        </header>

        <div className="flex-1 space-y-6 overflow-y-auto pr-2 custom-scroll">
          <ParameterEditor 
            params={physicsPriors} 
            onChange={(u) => setPhysicsPriors(p => ({ ...p, ...u }))}
            onReset={() => {
              setPhysicsPriors({ mass: 1, friction: 0.1, gravity: 0.5, textile_k: 400, damping: 0.15 });
              historyRef.current = [];
              setTelemetry([]);
              setQuotaExceeded(false);
              setMessages([]);
            }}
          />

          <section className="bg-indigo-950/20 border border-indigo-500/20 p-4 rounded shadow-sm">
             <div className="flex justify-between items-center mb-3">
               <h3 className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest">Epistemic Uncertainty</h3>
               {quotaExceeded && <span className="text-[8px] text-amber-500 animate-pulse font-bold">API_OFFLINE</span>}
             </div>
             <div className="flex items-end gap-1 h-12 bg-black/20 rounded p-1">
               {telemetry.slice(-24).map((t, i) => (
                 <div key={i} className="flex-1 bg-indigo-600/40 rounded-t-sm" style={{ height: `${Math.min(100, (uncertainty * 500) + (i * 2))}%` }}></div>
               ))}
             </div>
             <p className="text-[9px] text-indigo-300/70 mt-2 flex justify-between">
               <span>Ensemble Variance</span>
               <span className="font-bold text-indigo-400">{uncertainty.toExponential(2)}</span>
             </p>
          </section>

          <section className="bg-slate-900/30 border border-white/5 p-4 rounded">
             <button 
               onClick={() => setShowDocs(!showDocs)}
               className="text-[10px] font-bold text-slate-500 uppercase tracking-widest hover:text-indigo-400 flex items-center justify-between w-full"
             >
               Architecture Info
               <span>{showDocs ? '−' : '+'}</span>
             </button>
             {showDocs && (
               <div className="mt-4 text-[9px] leading-relaxed space-y-3 normal-case text-slate-400 border-t border-white/5 pt-3">
                 <p><strong className="text-white">Multiphysics Core:</strong> Matter.js handles rigid bodies. Custom Verlet integration drives the 96-node cloth mesh. Fluids use Lagrangian particles with repulsive potentials.</p>
                 <p><strong className="text-white">Brain Architecture:</strong> MPC + CEM optimization selects optimal actions. Ensemble Neural Networks learn unmodeled residuals. RIL (Gemini) monitors drift and stabilizes hyperparameters.</p>
               </div>
             )}
          </section>
        </div>

        <footer className="mt-8 pt-6 border-t border-white/5">
           <button 
             onClick={triggerFrictionShift}
             className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white text-[10px] font-bold uppercase transition-all flex items-center justify-center gap-2 rounded shadow-lg shadow-indigo-900/20 active:scale-95"
           >
             🔥 Shift Environment
           </button>
        </footer>
      </aside>

      <main className="flex-1 flex flex-col p-8 relative bg-black/20 backdrop-blur-3xl min-w-0">
        <div className="flex-1 relative mb-8 group rounded shadow-2xl overflow-hidden border border-white/5 bg-slate-950">
          {isMatterLoaded ? (
            <SimulationCanvas 
              mode={mode} 
              onStateUpdate={handleStateUpdate} 
              target={target}
              controlAction={controlAction}
              physicsPriors={physicsPriors}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-slate-950 text-indigo-500 flex-col gap-4">
              <div className="w-12 h-12 border-4 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin"></div>
              <span className="text-[10px] uppercase tracking-widest animate-pulse">Initializing Multiphysics Core...</span>
            </div>
          )}
          
          <div className="absolute inset-0 cursor-crosshair z-10"
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
            <div className="absolute inset-0 bg-indigo-900/10 pointer-events-none flex items-center justify-center backdrop-blur-[1px] z-20">
              <div className="bg-slate-950/90 border-2 border-indigo-500 px-10 py-6 text-indigo-400 font-black text-2xl animate-pulse shadow-2xl flex flex-col items-center">
                <span>BENCHMARKING ADAPTATION</span>
                <span className="text-[10px] mt-2 tracking-[0.3em] font-light opacity-60 italic">Contact Mismatch Detected</span>
              </div>
            </div>
          )}

          <div className="absolute top-6 left-6 pointer-events-none space-y-2 z-20">
            <div className="bg-slate-950/90 border border-white/10 p-4 rounded-sm backdrop-blur-xl shadow-2xl">
              <div className="text-[10px] text-indigo-400 font-bold mb-1 tracking-widest uppercase flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-ping"></span>
                RIL Hybrid Core
              </div>
              <div className="text-[9px] text-slate-500 space-y-1 font-mono">
                <div>SOLVER: RK4</div>
                <div>SIM_TIME: {(historyRef.current.length * 0.032).toFixed(2)}s</div>
                <div>L2_ERROR: {simState?.predictionError.toFixed(4) || '0.0000'}</div>
              </div>
            </div>
          </div>
        </div>

        <div className="h-64 grid grid-cols-12 gap-6 shrink-0 min-h-0">
           <div className="col-span-7 h-full">
             <Dashboard telemetry={telemetry} avgVelocity={simState?.current[2] || 0} stability={simState?.stability || 0} />
           </div>

           <div className="col-span-5 h-full">
             <ChatInterface onQuery={handleQuery} messages={messages} isAnalyzing={isAnalyzing} />
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
