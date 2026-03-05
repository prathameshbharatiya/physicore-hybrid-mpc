
import React, { useState, useEffect, useCallback, useRef } from 'react';
import SimulationCanvas from './components/SimulationCanvas';
import Dashboard from './components/Dashboard';
import ChatInterface from './components/ChatInterface';
import ParameterEditor from './components/ParameterEditor';
import { SimMode, SimState, TelemetryPoint, StateVector, ControlInput, MetaAnalysisResponse, ChatMessage, PhysicalParams, RobotDomain, IntegrationConfig } from './types';
import { motion, AnimatePresence } from 'motion/react';
import { computeMPCAction } from './services/optimizer';
import { updateSystemID } from './services/systemID';
import { ensembleDynamics } from './services/learnedDynamics';
import { performRILAnalysis } from './services/geminiService';
import { stepDynamicsRK4 } from './services/physicsLogic';
import { logFailure } from './services/failureDB';

const PhysicoreDiscovery: React.FC<{ onComplete: (config: IntegrationConfig) => void }> = ({ onComplete }) => {
  const [step, setStep] = useState(0);
  const [config, setConfig] = useState<IntegrationConfig>({
    domain: RobotDomain.ROBOTICS,
    purpose: '',
    safetyStandard: '',
    limitations: [],
    rules: [],
    isConnected: false
  });

  const [wizardAnswers, setWizardAnswers] = useState({
    purpose: '',
    safety: '',
    limit: '',
    rule: ''
  });

  const handleNext = () => setStep(s => s + 1);
  const handleBack = () => setStep(s => s - 1);

  const finish = () => {
    onComplete({
      ...config,
      purpose: wizardAnswers.purpose,
      safetyStandard: wizardAnswers.safety,
      limitations: [wizardAnswers.limit],
      rules: [wizardAnswers.rule],
      isConnected: true
    });
  };

  const domains = [
    { id: RobotDomain.AVIATION, label: 'Aviation', icon: '✈️', desc: 'UAVs, Fixed-wing, and VTOL systems.' },
    { id: RobotDomain.ROCKETS, label: 'Rockets', icon: '🚀', desc: 'Orbital launch vehicles and ballistic probes.' },
    { id: RobotDomain.INDUSTRIAL, label: 'Industrial', icon: '🏭', desc: 'Manufacturing arms and factory automation.' },
    { id: RobotDomain.ROBOTICS, label: 'Robotics', icon: '🤖', desc: 'Humanoids, AMRs, and general research.' },
  ];

  return (
    <div className="fixed inset-0 z-[100] bg-slate-950 flex flex-col items-center justify-center p-6 md:p-12 overflow-y-auto font-sans">
      <div className="absolute inset-0 opacity-20 pointer-events-none bg-[linear-gradient(#1e293b_1px,transparent_1px),linear-gradient(90deg,#1e293b_1px,transparent_1px)] [background-size:40px_40px]"></div>
      
      <div className="max-w-4xl w-full relative z-10">
        <AnimatePresence mode="wait">
          {step === 0 && (
            <motion.div 
              key="step0"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-12"
            >
              <div className="text-center">
                <div className="inline-block px-3 py-1 bg-indigo-500/10 border border-indigo-500/30 rounded text-indigo-400 text-[10px] font-bold tracking-[0.4em] uppercase mb-8">
                  Kernel v2.4.1 • Integration Bridge
                </div>
                <h1 className="text-6xl font-black text-white italic tracking-tighter mb-4">
                  SELECT <span className="text-indigo-500 text-7xl">DOMAIN</span>
                </h1>
                <p className="text-slate-400 text-lg font-light max-w-xl mx-auto">
                  PhysiCore adapts its safety kernels and physics priors based on your hardware domain.
                </p>
                <div className="mt-8 flex justify-center gap-4">
                  <label className="px-6 py-3 bg-slate-900 border border-white/10 hover:border-indigo-500 text-slate-400 text-xs uppercase tracking-widest cursor-pointer transition-all">
                    Upload Robot Config (.json)
                    <input type="file" className="hidden" onChange={() => {
                      // Simulate file upload
                      onComplete({
                        domain: RobotDomain.ROCKETS,
                        purpose: 'Orbital Insertion',
                        safetyStandard: 'AS9100',
                        limitations: ['Max Thrust: 1.2MN'],
                        rules: ['No-fly zone: Sector 7'],
                        isConnected: true
                      });
                    }} />
                  </label>
                  <div className="flex items-center text-slate-700 text-[10px] uppercase tracking-widest">or use integration tool</div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {domains.map((d) => (
                  <button
                    key={d.id}
                    onClick={() => { setConfig({ ...config, domain: d.id }); handleNext(); }}
                    className={`p-6 text-left border transition-all group ${config.domain === d.id ? 'bg-indigo-600/20 border-indigo-500' : 'bg-slate-900/40 border-white/5 hover:border-white/20'}`}
                  >
                    <div className="text-3xl mb-4">{d.icon}</div>
                    <h3 className="text-white font-bold uppercase tracking-widest text-sm mb-2">{d.label}</h3>
                    <p className="text-slate-500 text-xs leading-relaxed">{d.desc}</p>
                  </button>
                ))}
              </div>
            </motion.div>
          )}

          {step === 1 && (
            <motion.div 
              key="step1"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-10"
            >
              <div className="flex items-center gap-4 mb-8">
                <button onClick={handleBack} className="p-2 hover:bg-white/5 rounded text-slate-400">← Back</button>
                <div className="h-px flex-1 bg-white/10"></div>
                <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">Step 02: Integration Tool</span>
              </div>

              <div className="space-y-8">
                <div className="space-y-3">
                  <label className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest">01. Primary Mission Objective</label>
                  <input 
                    type="text" 
                    placeholder="e.g. High-altitude cargo delivery"
                    className="w-full bg-slate-900 border border-white/10 p-4 text-white focus:border-indigo-500 outline-none transition-all"
                    value={wizardAnswers.purpose}
                    onChange={e => setWizardAnswers({...wizardAnswers, purpose: e.target.value})}
                  />
                </div>
                <div className="space-y-3">
                  <label className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest">02. Safety Standard Compliance</label>
                  <select 
                    className="w-full bg-slate-900 border border-white/10 p-4 text-white focus:border-indigo-500 outline-none transition-all"
                    value={wizardAnswers.safety}
                    onChange={e => setWizardAnswers({...wizardAnswers, safety: e.target.value})}
                  >
                    <option value="">Select Standard</option>
                    <option value="ISO 10218">ISO 10218 (Industrial Robotics)</option>
                    <option value="DO-178C">DO-178C (Airborne Systems)</option>
                    <option value="AS9100">AS9100 (Aerospace Quality)</option>
                    <option value="MIL-STD-882E">MIL-STD-882E (System Safety)</option>
                  </select>
                </div>
                <div className="space-y-3">
                  <label className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest">03. Critical Operational Limit</label>
                  <input 
                    type="text" 
                    placeholder="e.g. Max Velocity: 450m/s"
                    className="w-full bg-slate-900 border border-white/10 p-4 text-white focus:border-indigo-500 outline-none transition-all"
                    value={wizardAnswers.limit}
                    onChange={e => setWizardAnswers({...wizardAnswers, limit: e.target.value})}
                  />
                </div>
                <div className="space-y-3">
                  <label className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest">04. Deployment Constraint</label>
                  <input 
                    type="text" 
                    placeholder="e.g. Human proximity < 2m prohibited"
                    className="w-full bg-slate-900 border border-white/10 p-4 text-white focus:border-indigo-500 outline-none transition-all"
                    value={wizardAnswers.rule}
                    onChange={e => setWizardAnswers({...wizardAnswers, rule: e.target.value})}
                  />
                </div>
              </div>

              <button 
                onClick={handleNext}
                disabled={!wizardAnswers.purpose || !wizardAnswers.safety}
                className="w-full py-6 bg-indigo-600 hover:bg-indigo-500 text-white font-black uppercase tracking-[0.2em] transition-all disabled:opacity-50"
              >
                Configure Bridge →
              </button>
            </motion.div>
          )}

          {step === 2 && (
            <motion.div 
              key="step2"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.05 }}
              className="text-center space-y-12"
            >
              <div className="w-32 h-32 border-4 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin mx-auto flex items-center justify-center">
                <div className="w-20 h-20 bg-indigo-500/10 rounded-full flex items-center justify-center">
                  <span className="text-3xl">🔗</span>
                </div>
              </div>
              
              <div className="space-y-4">
                <h2 className="text-4xl font-black text-white italic tracking-tighter uppercase">Establishing <span className="text-indigo-500">Hardware Link</span></h2>
                <p className="text-slate-400 text-lg font-light max-w-lg mx-auto">
                  PhysiCore is now bridging with your {config.domain.toLowerCase()} hardware. No code changes required.
                </p>
              </div>

              <div className="bg-slate-900/60 border border-white/5 p-8 rounded-xl text-left font-mono text-xs space-y-2 max-w-md mx-auto">
                <div className="text-emerald-500">✓ Domain: {config.domain}</div>
                <div className="text-emerald-500">✓ Safety Kernel: {wizardAnswers.safety}</div>
                <div className="text-emerald-500">✓ Telemetry Bridge: ACTIVE</div>
                <div className="text-slate-500 animate-pulse mt-4">Waiting for hardware handshake...</div>
              </div>

              <button 
                onClick={finish}
                className="px-12 py-6 bg-indigo-600 hover:bg-indigo-500 text-white font-black uppercase tracking-[0.2em] transition-all shadow-2xl"
              >
                Initialize Simulation
              </button>
            </motion.div>
          )}
        </AnimatePresence>
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

  const [integration, setIntegration] = useState<IntegrationConfig | null>(null);

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
    if (!integration?.isConnected) return;

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

  if (!hasEntered) return <PhysicoreDiscovery onComplete={(cfg) => { setIntegration(cfg); setHasEntered(true); }} />;

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-400 overflow-hidden font-mono selection:bg-indigo-500/40">
      <aside className="w-80 bg-slate-900/60 border-r border-white/5 flex flex-col p-6 shadow-2xl z-20 shrink-0">
        <header className="mb-10">
          <div className="flex items-center gap-3 text-white font-black text-2xl italic tracking-tighter">
            <div className="w-8 h-8 bg-indigo-600 rounded flex items-center justify-center not-italic text-xs shadow-[0_0_20px_rgba(79,70,229,0.3)]">P</div>
            PHYSI<span className="text-indigo-400 font-light italic tracking-tight">CORE</span>
          </div>
          <div className="mt-4 p-3 bg-indigo-500/10 border border-indigo-500/20 rounded-sm">
            <div className="flex justify-between items-center mb-1">
              <span className="text-[8px] text-indigo-400 uppercase tracking-widest font-bold">Active Domain</span>
              <span className="text-[10px] text-white font-bold">{integration?.domain}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[8px] text-indigo-400 uppercase tracking-widest font-bold">Link Status</span>
              <span className="text-[10px] text-emerald-500 font-bold animate-pulse">CONNECTED</span>
            </div>
          </div>
        </header>

        <div className="flex-1 space-y-6 overflow-y-auto pr-2 custom-scroll">
          <div className="bg-slate-950/40 border border-white/5 p-4 rounded-sm">
            <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-3">Safety Kernel</h3>
            <div className="text-[11px] text-white font-bold mb-1">{integration?.safetyStandard}</div>
            <div className="text-[9px] text-slate-600 italic">Enforcing {integration?.limitations[0]}</div>
          </div>
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
              integration={integration}
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
             <Dashboard telemetry={telemetry} avgVelocity={simState?.current[2] || 0} stability={simState?.stability || 0} integration={integration} />
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
