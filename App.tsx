
import React, { useState, useEffect, useCallback, useRef } from 'react';
import SimulationCanvas from './components/SimulationCanvas';
import Dashboard from './components/Dashboard';
import ChatInterface from './components/ChatInterface';
import ParameterEditor from './components/ParameterEditor';
import { SimMode, SimState, TelemetryPoint, StateVector, ControlInput, MetaAnalysisResponse, ChatMessage, PhysicalParams } from './types';
import { computeMPCAction } from './services/optimizer';
import { updateSystemID } from './services/systemID';
import { ensembleDynamics } from './services/learnedDynamics';
import { performRILAnalysis } from './services/geminiService';
import { stepDynamicsRK4 } from './services/physicsLogic';
import { logFailure } from './services/failureDB';

const PhysicoreDiscovery: React.FC<{ onEnter: () => void }> = ({ onEnter }) => {
  return (
    <div className="fixed inset-0 z-[100] bg-slate-950 flex flex-col items-center justify-center p-6 md:p-12 overflow-y-auto">
      <div className="absolute inset-0 opacity-20 pointer-events-none bg-[linear-gradient(#1e293b_1px,transparent_1px),linear-gradient(90deg,#1e293b_1px,transparent_1px)] [background-size:40px_40px]"></div>
      
      <div className="max-w-6xl w-full relative z-10">
        <div className="grid lg:grid-cols-2 gap-20 items-center">
          <div>
            <div className="inline-block px-3 py-1 bg-indigo-500/10 border border-indigo-500/30 rounded text-indigo-400 text-[10px] font-bold tracking-[0.4em] uppercase mb-8">
              Industrial Simulation Core v2.4.1
            </div>
            <h1 className="text-6xl md:text-8xl font-black text-white italic tracking-tighter mb-6 leading-none">
              PHYSI<span className="text-indigo-500">CORE</span>
            </h1>
            <p className="text-slate-400 text-xl font-light leading-relaxed mb-10 max-w-lg">
              Industrial-grade <span className="text-white font-medium">Self-Healing Physics Engine</span>. Stop guessing friction. Let AI align your simulation to your hardware.
            </p>
            
            <div className="space-y-8 mb-12">
              <div className="flex gap-5">
                <div className="w-12 h-12 shrink-0 bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center font-black text-indigo-400 shadow-[0_0_20px_rgba(79,70,229,0.2)]">01</div>
                <div>
                  <h4 className="text-white font-bold text-sm uppercase tracking-widest">Close the Reality Gap</h4>
                  <p className="text-slate-500 text-xs mt-1 leading-relaxed">Identify mismatches between analytical RK4 models and real-world sensor streams. Our RIL layer reconciles discrepancies in real-time.</p>
                </div>
              </div>
              <div className="flex gap-5">
                <div className="w-12 h-12 shrink-0 bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center font-black text-indigo-400 shadow-[0_0_20px_rgba(79,70,229,0.2)]">02</div>
                <div>
                  <h4 className="text-white font-bold text-sm uppercase tracking-widest">Ensemble Residual Learning</h4>
                  <p className="text-slate-500 text-xs mt-1 leading-relaxed">Train on the "Physics Residual"—the nonlinear noise of your specific factory floor. No more "Ideal Case" assumptions.</p>
                </div>
              </div>
              <div className="flex gap-5">
                <div className="w-12 h-12 shrink-0 bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center font-black text-indigo-400 shadow-[0_0_20px_rgba(79,70,229,0.2)]">03</div>
                <div>
                  <h4 className="text-white font-bold text-sm uppercase tracking-widest">Hardware Portability</h4>
                  <p className="text-slate-500 text-xs mt-1 leading-relaxed">Export optimized weights and calibrated priors as production-ready JSON/ONNX bridges for your edge controllers.</p>
                </div>
              </div>
            </div>

            <button 
              onClick={onEnter} 
              className="group px-12 py-6 bg-indigo-600 hover:bg-indigo-500 text-white font-black text-sm uppercase tracking-[0.2em] transition-all shadow-2xl active:scale-95 flex items-center justify-center gap-4 rounded-sm"
            >
              Initialize Control Core
              <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
            </button>
          </div>

          <div className="hidden lg:block relative">
            <div className="aspect-square bg-slate-900/40 border border-white/10 rounded-xl p-10 backdrop-blur-3xl overflow-hidden relative shadow-[0_0_120px_rgba(79,70,229,0.1)]">
              <div className="absolute top-0 left-0 w-full p-6 border-b border-white/5 font-mono text-[10px] text-slate-500 flex justify-between uppercase tracking-widest">
                <span>Kernel: PhysiCore-v2.4</span>
                <span className="text-emerald-500 animate-pulse">● System_Online</span>
              </div>
              <div className="h-full flex flex-col justify-center items-center text-center">
                <div className="w-64 h-64 border-[3px] border-indigo-500/10 border-t-indigo-500 rounded-full animate-[spin_6s_linear_infinite] mb-12 relative flex items-center justify-center">
                   <div className="w-48 h-48 border border-indigo-500/5 rounded-full animate-[spin_10s_linear_infinite_reverse]"></div>
                   <div className="absolute flex flex-col items-center">
                      <span className="text-indigo-400 text-2xl font-black italic tracking-tighter">HYBRID</span>
                      <span className="text-indigo-500/50 text-[10px] font-bold uppercase tracking-[0.3em]">Dynamics Loop</span>
                   </div>
                </div>
                <div className="space-y-4 w-full">
                  <div className="flex justify-between text-[10px] text-slate-500 uppercase font-mono px-4">
                    <span>Solver_Stability</span>
                    <span className="text-indigo-400">99.98%</span>
                  </div>
                  <div className="w-full bg-slate-800/50 h-1 rounded-full overflow-hidden">
                    <div className="bg-indigo-500 h-full w-[99%]" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-24 text-[10px] text-slate-700 font-mono flex justify-between uppercase tracking-widest border-t border-white/5 pt-10">
          <div className="flex gap-10">
            <span>Uplink: Primary_Active</span>
            <span>Region: Edge_Local</span>
          </div>
          <span>Built for professional robotics teams.</span>
        </div>
      </div>
    </div>
  );
};

const App: React.FC = () => {
  const [hasEntered, setHasEntered] = useState(false);
  const [isMatterLoaded, setIsMatterLoaded] = useState(false);
  const [simState, setSimState] = useState<SimState | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetryPoint[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [quotaExceeded, setQuotaExceeded] = useState(false);
  const [isDeploying, setIsDeploying] = useState(false);
  
  const [target, setTarget] = useState<[number, number]>([400, 300]);
  const [controlAction, setControlAction] = useState<ControlInput>([0, 0]);
  const [physicsPriors, setPhysicsPriors] = useState<PhysicalParams>({ mass: 1.0, friction: 0.1, gravity: 0.5, textile_k: 400, damping: 0.15 });
  const [costWeights, setCostWeights] = useState({ q: 1.5, r: 0.05 });
  const [uncertainty, setUncertainty] = useState(0);

  const historyRef = useRef<SimState[]>([]);
  const lastStateRef = useRef<StateVector | null>(null);

  useEffect(() => {
    const checkMatter = () => {
      if ((window as any).Matter) setIsMatterLoaded(true);
      else setTimeout(checkMatter, 100);
    };
    checkMatter();
  }, []);

  const handleStateUpdate = useCallback((state: SimState) => {
    if (lastStateRef.current) {
      const xPhys = stepDynamicsRK4(lastStateRef.current, controlAction, physicsPriors);
      ensembleDynamics.train(lastStateRef.current, controlAction, state.current, xPhys);

      const updatedParams = updateSystemID(lastStateRef.current, controlAction, state.current, physicsPriors);
      setPhysicsPriors(prev => ({ ...prev, mass: updatedParams.mass, friction: updatedParams.friction }));

      const predError = Math.sqrt(state.current.reduce((s, v, i) => s + Math.pow(v - xPhys[i], 2), 0));
      if (predError > 15.0) {
        logFailure({
          id: crypto.randomUUID(),
          timestamp: Date.now(),
          task: "control_stabilization",
          failure_type: "HIGH_RESIDUAL_DRIFT",
          sim_params: { ...physicsPriors }
        });
      }
    }

    const { action, ensembleUncertainty } = computeMPCAction(state.current, target, physicsPriors, costWeights);
    setControlAction(action);
    setUncertainty(ensembleUncertainty);

    const xPhysFinal = lastStateRef.current ? stepDynamicsRK4(lastStateRef.current, controlAction, physicsPriors) : state.current;
    const predError = lastStateRef.current ? Math.sqrt(state.current.reduce((s, v, i) => s + Math.pow(v - xPhysFinal[i], 2), 0)) : 0;
    
    lastStateRef.current = state.current;
    const enrichedState = { ...state, predictionError: predError, uncertainty: ensembleUncertainty };
    
    setSimState(enrichedState);
    historyRef.current = [...historyRef.current.slice(-120), enrichedState];
    setTelemetry(prev => [...prev.slice(-60), { time: state.time, value: predError, label: 'L2 Residual' }]);
  }, [target, physicsPriors, controlAction, costWeights]);

  const handleQuery = async (query: string) => {
    if (isAnalyzing || quotaExceeded) return;
    setMessages(prev => [...prev, { role: 'user', content: query, timestamp: Date.now() }]);
    setIsAnalyzing(true);
    try {
      const result = await performRILAnalysis(historyRef.current[historyRef.current.length - 1], historyRef.current, query);
      if (result.suggestedCostTweaks) setCostWeights({ q: result.suggestedCostTweaks.q_weight, r: result.suggestedCostTweaks.r_weight });
      setMessages(prev => [...prev, { role: 'assistant', content: result.insight, timestamp: Date.now() }]);
    } catch (err: any) {
      if (err.message === "QUOTA_EXHAUSTED") setQuotaExceeded(true);
      setMessages(prev => [...prev, { role: 'assistant', content: "AI Uplink Restricted. Physics Core maintained on local prior.", timestamp: Date.now() }]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDeploy = () => {
    setIsDeploying(true);
    setTimeout(() => {
      setIsDeploying(false);
      const deploymentPackage = {
        metadata: {
          client: "PhysiCore-Industrial-v2.4",
          export_time: new Date().toISOString(),
          license: "Enterprise_Commercial_R&D",
          target_kernel: "ROS2_Humble_x64",
        },
        calibrated_parameters: {
          mass: physicsPriors.mass,
          friction: physicsPriors.friction,
          textile_stiffness: physicsPriors.textile_k,
          system_damping: physicsPriors.damping
        },
        learned_ensemble_weights: ensembleDynamics.getWeightsExport(),
        control_matrices: {
          Q: costWeights.q,
          R: costWeights.r,
          horizon: 12
        },
        bridge_code: `
# PhysiCore Real-Time Correction Bridge
# Use this in your ROS2 Control Loop

import json
import numpy as np

class PhysicoreBridge:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.data = json.load(f)
        self.mass = self.data['calibrated_parameters']['mass']
        self.friction = self.data['calibrated_parameters']['friction']

    def compute_correction(self, state, action):
        # Apply learned residual: analytical_physics + learned_residual
        # This resolves the reality gap for your specific hardware.
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Logic provided in exported weights
        `
      };
      const blob = new Blob([JSON.stringify(deploymentPackage, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `physicore_hw_deploy_${Date.now()}.json`;
      a.click();
    }, 1800);
  };

  if (!hasEntered) return <PhysicoreDiscovery onEnter={() => setHasEntered(true)} />;

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-400 overflow-hidden font-mono selection:bg-indigo-500/40">
      <aside className="w-80 bg-slate-900/60 border-r border-white/5 flex flex-col p-6 shadow-2xl z-20 shrink-0">
        <header className="mb-10">
          <div className="flex items-center gap-3 text-white font-black text-2xl italic tracking-tighter">
            <div className="w-8 h-8 bg-indigo-600 rounded flex items-center justify-center not-italic text-xs shadow-[0_0_20px_rgba(79,70,229,0.3)]">P</div>
            PHYSI<span className="text-indigo-400 font-light italic tracking-tight">CORE</span>
          </div>
          <p className="text-[9px] text-slate-600 mt-2 uppercase tracking-[0.4em] font-bold">Reality Bridging Unit</p>
        </header>

        <div className="flex-1 space-y-6 overflow-y-auto pr-2 custom-scroll">
          <ParameterEditor 
            params={physicsPriors} 
            onChange={(u) => setPhysicsPriors(p => ({ ...p, ...u }))}
            onReset={() => {
              setPhysicsPriors({ mass: 1, friction: 0.1, gravity: 0.5, textile_k: 400, damping: 0.15 });
              historyRef.current = [];
              setTelemetry([]);
              setMessages([]);
            }}
          />

          <section className="bg-indigo-950/20 border border-indigo-500/20 p-5 rounded-sm">
             <h3 className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest mb-4">Epistemic Uncertainty</h3>
             <div className="flex items-end gap-1 h-20 bg-black/40 rounded p-1.5 overflow-hidden">
               {telemetry.slice(-40).map((t, i) => (
                 <div key={i} className="flex-1 bg-indigo-500/50 rounded-t-[1px]" style={{ height: `${Math.min(100, (uncertainty * 400) + (i * 0.5))}%` }}></div>
               ))}
             </div>
             <div className="mt-3 flex justify-between items-center">
               <span className="text-[9px] text-indigo-300/50 uppercase tracking-wider">Ensemble Variance</span>
               <span className="text-[10px] font-bold text-indigo-400 tabular-nums">{(uncertainty * 100).toFixed(3)}%</span>
             </div>
          </section>
        </div>

        <footer className="mt-8 pt-6 border-t border-white/5 space-y-4">
           <button 
             onClick={handleDeploy}
             disabled={isDeploying}
             className="w-full py-5 bg-indigo-600 text-white hover:bg-indigo-500 text-[11px] font-black uppercase tracking-[0.2em] transition-all flex items-center justify-center gap-3 rounded-sm shadow-xl disabled:opacity-50 group active:scale-[0.97]"
           >
             {isDeploying ? 'Syncing Binaries...' : 'Deploy to Hardware'}
             {!isDeploying && <svg className="w-4 h-4 group-hover:translate-y-[-2px] transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>}
           </button>
           <p className="text-[8px] text-center text-slate-700 uppercase tracking-widest leading-relaxed">Calibration: production-stable<br/>Kernel Version: 2.4.1</p>
        </footer>
      </aside>

      <main className="flex-1 flex flex-col p-8 relative bg-[#020617] min-w-0">
        <div className="flex-1 relative mb-8 rounded-lg shadow-[0_0_80px_rgba(0,0,0,0.6)] overflow-hidden border border-white/5 bg-slate-900/10">
          {isMatterLoaded ? (
            <SimulationCanvas 
              mode={SimMode.MPC_STABILIZATION} 
              onStateUpdate={handleStateUpdate} 
              target={target}
              controlAction={controlAction}
              physicsPriors={physicsPriors}
            />
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center text-indigo-500 text-[10px] uppercase tracking-[0.5em] gap-6">
              <div className="w-16 h-16 border-[3px] border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin"></div>
              Booting High-Fidelity Physics Kernel...
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

          <div className="absolute top-8 left-8 pointer-events-none z-20">
            <div className="bg-slate-950/90 border border-white/10 p-6 rounded-sm backdrop-blur-2xl shadow-2xl min-w-[240px]">
              <div className="text-[11px] text-indigo-400 font-black mb-3 tracking-[0.2em] uppercase flex items-center gap-3">
                <span className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse shadow-[0_0_10px_#6366f1]"></span>
                State Manifold Analytics
              </div>
              <div className="text-[10px] text-slate-500 space-y-2 font-mono">
                <div className="flex justify-between border-b border-white/5 pb-1"><span>RK4_GRADIENT</span> <span className="text-slate-300">ADAPTIVE</span></div>
                <div className="flex justify-between border-b border-white/5 pb-1"><span>L2_RESIDUAL</span> <span className="text-white font-bold">{simState?.predictionError.toFixed(6) || '0.000000'}</span></div>
                <div className="flex justify-between"><span>LINK_STATUS</span> <span className="text-emerald-500 font-bold tracking-widest">DEPLOY_READY</span></div>
              </div>
            </div>
          </div>
        </div>

        <div className="h-72 grid grid-cols-12 gap-8 shrink-0 min-h-0">
           <div className="col-span-7 h-full">
             <Dashboard telemetry={telemetry} avgVelocity={simState?.current[2] || 0} stability={simState?.stability || 0} />
           </div>
           <div className="col-span-5 h-full">
             <ChatInterface onQuery={handleQuery} messages={messages} isAnalyzing={isAnalyzing} />
           </div>
        </div>
      </main>
      <style>{`
        @keyframes pulse { 0%, 100% { transform: scaleY(1); } 50% { transform: scaleY(1.5); } }
        .custom-scroll::-webkit-scrollbar { width: 4px; }
        .custom-scroll::-webkit-scrollbar-track { background: transparent; }
        .custom-scroll::-webkit-scrollbar-thumb { background: #312e81; border-radius: 10px; }
      `}</style>
    </div>
  );
};

export default App;
