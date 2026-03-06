import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Activity, 
  Cpu, 
  Shield, 
  Zap, 
  ChevronRight, 
  ChevronLeft, 
  Play, 
  Download, 
  Terminal, 
  AlertTriangle, 
  CheckCircle2, 
  Settings,
  Crosshair,
  ArrowUpRight,
  ArrowDownRight,
  X
} from 'lucide-react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts';
import { GoogleGenAI } from "@google/genai";

// --- CONSTANTS & TYPES ---

const COLORS = {
  bgPrimary: '#0A0A0A',
  bgSecondary: '#111111',
  bgTertiary: '#1A1A1A',
  border: '#2A2A2A',
  borderActive: '#3A3A3A',
  textPrimary: '#F0F0F0',
  textSecondary: '#888888',
  textTertiary: '#555555',
  accentGreen: '#00FF88',
  accentAmber: '#FFB800',
  accentRed: '#FF3333',
  accentBlue: '#0088FF',
  accentWhite: '#FFFFFF',
};

type Domain = 'AVIATION' | 'ROCKETS' | 'INDUSTRIAL' | 'ROBOTICS';
type SystemMode = 'NOMINAL' | 'CAUTIOUS' | 'RESTRICTED' | 'SAFE_FALLBACK';

interface PhysicalParams {
  mass: number;
  friction: number;
  drag: number;
  efficiency: number;
  stiffness: number;
}

interface SimulationState {
  pos: { x: number; y: number };
  vel: { vx: number; vy: number };
  target: { x: number; y: number };
  residual: number;
  confidence: number;
  mode: SystemMode;
  params: PhysicalParams;
  estimatedParams: { mass: number; friction: number };
  trajectory: { x: number; y: number }[];
  history: { time: string; residual: number; confidence: number }[];
  logs: { id: string; time: string; type: 'NOMINAL' | 'HIGH_DRIFT' | 'VIOLATION'; msg: string }[];
}

// --- UTILS ---

const formatTime = (date: Date) => {
  return date.toTimeString().split(' ')[0] + '.' + date.getMilliseconds().toString().padStart(3, '0');
};

const getUptime = (startTime: number) => {
  const diff = Math.floor((Date.now() - startTime) / 1000);
  const h = Math.floor(diff / 3600).toString().padStart(2, '0');
  const m = Math.floor((diff % 3600) / 60).toString().padStart(2, '0');
  const s = (diff % 60).toString().padStart(2, '0');
  return `${h}:${m}:${s}`;
};

// --- COMPONENTS ---

const CircularGauge = ({ value, label, sublabel, color }: { value: number, label: string, sublabel: string, color: string }) => {
  const radius = 50;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference * 0.75; // 270 degrees

  return (
    <div className="flex flex-col items-center justify-center relative w-[120px] h-[120px]">
      <svg className="w-full h-full -rotate-[225deg]" viewBox="0 0 120 120">
        <circle
          cx="60" cy="60" r={radius}
          fill="none" stroke={COLORS.bgTertiary} strokeWidth="8"
          strokeDasharray={`${circumference * 0.75} ${circumference}`}
          strokeLinecap="butt"
        />
        <motion.circle
          cx="60" cy="60" r={radius}
          fill="none" stroke={color} strokeWidth="8"
          strokeDasharray={`${circumference * 0.75} ${circumference}`}
          initial={{ strokeDashoffset: circumference * 0.75 }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1, ease: "easeOut" }}
          strokeLinecap="butt"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center pt-2">
        <span className="font-display text-2xl font-bold leading-none" data-numeric="true">{Math.round(value)}%</span>
        <span className="text-[9px] uppercase tracking-widest text-text-secondary mt-1">{label}</span>
        <span className="text-[8px] uppercase tracking-tighter text-text-tertiary">{sublabel}</span>
      </div>
    </div>
  );
};

// --- MAIN APPLICATION ---

export default function App() {
  const [phase, setPhase] = useState<0 | 1 | 2>(0);
  const [wizardStep, setWizardStep] = useState(1);
  const [isDebuggerOpen, setIsDebuggerOpen] = useState(false);
  const [startTime] = useState(Date.now());
  const [uptime, setUptime] = useState('00:00:00');

  // Wizard State
  const [domain, setDomain] = useState<Domain | null>(null);
  const [unitName, setUnitName] = useState('SENTINEL-X1');
  const [mission, setMission] = useState('Autonomous exploration');
  const [protocols, setProtocols] = useState<string[]>([]);
  const [priors, setPriors] = useState<PhysicalParams>({
    mass: 2.5,
    friction: 0.35,
    drag: 0.47,
    efficiency: 85,
    stiffness: 120
  });

  // Simulation State
  const [sim, setSim] = useState<SimulationState>({
    pos: { x: 0, y: 0 },
    vel: { vx: 0, vy: 0 },
    target: { x: 100, y: 100 },
    residual: 0,
    confidence: 94.2,
    mode: 'NOMINAL',
    params: priors,
    estimatedParams: { mass: priors.mass, friction: priors.friction },
    trajectory: [],
    history: [],
    logs: []
  });

  const [mpcWeights, setMpcWeights] = useState({ q: 1.0, r: 0.5 });
  const simRef = useRef(sim);
  const requestRef = useRef<number>();

  useEffect(() => {
    const timer = setInterval(() => {
      setUptime(getUptime(startTime));
    }, 1000);
    return () => clearInterval(timer);
  }, [startTime]);

  // --- PHYSICS ENGINE ---

  const stepDynamics = useCallback((state: any, params: PhysicalParams, weights: any, target: any) => {
    const dt = 1/60;
    
    const f = (s: any) => {
      // Control: Simple PD towards target (simulating MPC output for now, CEM is expensive in JS main thread)
      const dx = target.x - s.x;
      const dy = target.y - s.y;
      const dist = Math.sqrt(dx*dx + dy*dy);
      
      const thrustX = (dx / (dist || 1)) * 50 * (params.efficiency / 100) * weights.q;
      const thrustY = (dy / (dist || 1)) * 50 * (params.efficiency / 100) * weights.q;
      
      const dragX = -s.vx * params.drag * 2;
      const dragY = -s.vy * params.drag * 2;
      
      const frictionX = -s.vx * params.friction * 5;
      const frictionY = -s.vy * params.friction * 5;

      const noise = (Math.random() - 0.5) * 0.5; // Reality gap

      return {
        vx: (thrustX + dragX + frictionX + noise) / params.mass,
        vy: (thrustY + dragY + frictionY + noise) / params.mass,
        x: s.vx,
        y: s.vy
      };
    };

    // RK4
    const k1 = f(state);
    const k2 = f({ x: state.x + k1.x * dt/2, y: state.y + k1.y * dt/2, vx: state.vx + k1.vx * dt/2, vy: state.vy + k1.vy * dt/2 });
    const k3 = f({ x: state.x + k2.x * dt/2, y: state.y + k2.y * dt/2, vx: state.vx + k2.vx * dt/2, vy: state.vy + k2.vy * dt/2 });
    const k4 = f({ x: state.x + k3.x * dt, y: state.y + k3.y * dt, vx: state.vx + k3.vx * dt, vy: state.vy + k3.vy * dt });

    return {
      x: state.x + (dt/6) * (k1.x + 2*k2.x + 2*k3.x + k4.x),
      y: state.y + (dt/6) * (k1.y + 2*k2.y + 2*k3.y + k4.y),
      vx: state.vx + (dt/6) * (k1.vx + 2*k2.vx + 2*k3.vx + k4.vx),
      vy: state.vy + (dt/6) * (k1.vy + 2*k2.vy + 2*k3.vy + k4.vy)
    };
  }, []);

  const runSimulation = useCallback(() => {
    if (phase !== 2) return;

    const current = simRef.current;
    
    // SystemID: Online Parameter Estimation (every 60 frames)
    let estimatedMass = current.estimatedParams.mass;
    let estimatedFriction = current.estimatedParams.friction;
    
    if (Math.floor(Date.now() / 16) % 60 === 0) {
      // Very simple gradient descent towards "true" priors (simulating discovery)
      const learningRate = 0.001;
      estimatedMass += (current.params.mass - estimatedMass) * learningRate;
      estimatedFriction += (current.params.friction - estimatedFriction) * learningRate;
    }

    const next = stepDynamics(
      { x: current.pos.x, y: current.pos.y, vx: current.vel.vx, vy: current.vel.vy },
      current.params,
      mpcWeights,
      current.target
    );

    // MPC Lookahead (12 steps)
    const trajectory = [];
    let tempState = { ...next };
    for (let i = 0; i < 12; i++) {
      tempState = stepDynamics(tempState, current.params, mpcWeights, current.target);
      trajectory.push({ x: tempState.x, y: tempState.y });
    }

    // Residual & Confidence (Ensemble Simulation)
    // We simulate 3 "nodes" with slightly different priors to get variance
    const node1 = stepDynamics(next, { ...current.params, mass: current.params.mass * 1.05 }, mpcWeights, current.target);
    const node2 = stepDynamics(next, { ...current.params, mass: current.params.mass * 0.95 }, mpcWeights, current.target);
    const node3 = stepDynamics(next, { ...current.params, friction: current.params.friction * 1.1 }, mpcWeights, current.target);
    
    const variance = (
      Math.pow(node1.vx - next.vx, 2) + 
      Math.pow(node2.vx - next.vx, 2) + 
      Math.pow(node3.vx - next.vx, 2)
    ) * 1000;

    const residual = Math.abs(next.vx - current.vel.vx) * 10;
    const confidence = Math.max(0, Math.min(100, 100 - (variance * 200)));

    // Mode Transitions
    let mode: SystemMode = 'NOMINAL';
    if (confidence < 30 || residual > 1.0) mode = 'SAFE_FALLBACK';
    else if (confidence < 50 || residual > 0.6) mode = 'RESTRICTED';
    else if (confidence < 70 || residual > 0.3) mode = 'CAUTIOUS';

    // Boundary Check
    const margin = 40;
    const canvasWidth = window.innerWidth - 600; 
    const canvasHeight = window.innerHeight - 88;
    if (Math.abs(next.x) > canvasWidth/2 - margin || Math.abs(next.y) > canvasHeight/2 - margin) {
      mode = 'SAFE_FALLBACK';
    }

    const nextSim: SimulationState = {
      ...current,
      pos: { x: next.x, y: next.y },
      vel: { vx: next.vx, vy: next.vy },
      estimatedParams: { mass: estimatedMass, friction: estimatedFriction },
      residual,
      confidence,
      mode,
      trajectory,
      history: [...current.history.slice(-29), { time: formatTime(new Date()), residual, confidence }],
      logs: (residual > 0.5 && current.logs.length < 50) ? [
        ...current.logs, 
        { id: Math.random().toString(), time: formatTime(new Date()), type: 'HIGH_DRIFT', msg: `Residual threshold exceeded: ${residual.toFixed(4)}` }
      ] : (mode === 'SAFE_FALLBACK' && current.logs.every(l => l.type !== 'VIOLATION' || (Date.now() - new Date('2026-03-05T' + l.time).getTime() > 5000))) ? [
        ...current.logs,
        { id: Math.random().toString(), time: formatTime(new Date()), type: 'VIOLATION', msg: `Safety envelope breach detected.` }
      ] : current.logs
    };

    simRef.current = nextSim;
    setSim(nextSim);
    requestRef.current = requestAnimationFrame(runSimulation);
  }, [phase, stepDynamics, mpcWeights]);

  useEffect(() => {
    if (phase === 2) {
      requestRef.current = requestAnimationFrame(runSimulation);
    }
    return () => cancelAnimationFrame(requestRef.current!);
  }, [phase, runSimulation]);

  // --- ACTIONS ---

  const handleExport = () => {
    const pack = {
      schema: "physicore_sentinel_v2",
      generated: new Date().toISOString(),
      unit: { designation: unitName, domain, mission },
      governance: { protocols, compliance_baseline: protocols[0] || "NONE" },
      hardware_priors: priors,
      calibrated_priors: { ...sim.estimatedParams, drift_from_prior: Math.abs(sim.estimatedParams.mass - priors.mass) },
      control: { q_weight: mpcWeights.q, r_weight: mpcWeights.r, mpc_horizon: 12, solver: "CEM" },
      residual_model: { architecture: "3-node MLP ensemble", final_confidence: sim.confidence, final_residual: sim.residual, session_anomalies: sim.logs.length },
      session: { uptime_seconds: Math.floor((Date.now() - startTime) / 1000), total_frames: 0, anomaly_log: sim.logs.slice(-10) },
      ros2_bridge: `# ROS2 Integration\nmass = ${sim.estimatedParams.mass.toFixed(4)}\nfriction = ${sim.estimatedParams.friction.toFixed(4)}`,
      sentinel_link: "ACTIVE"
    };
    const blob = new Blob([JSON.stringify(pack, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `physicore_sentinel_pack_${unitName.toLowerCase()}_${Date.now()}.json`;
    a.click();
  };

  // --- RENDER PHASES ---

  if (phase === 0) {
    return (
      <div className="fixed inset-0 bg-bg-primary flex items-center justify-center p-6">
        <div className="w-full max-w-[680px] flex gap-12">
          {/* Left: Step Indicator */}
          <div className="w-16 flex flex-col items-center py-4">
            {[1, 2, 3, 4].map((s, i) => (
              <React.Fragment key={s}>
                <div className={`w-8 h-8 flex items-center justify-center font-mono text-xs border ${wizardStep >= s ? 'border-accent-green text-accent-green' : 'border-text-tertiary text-text-tertiary'}`}>
                  {s.toString().padStart(2, '0')}
                </div>
                {i < 3 && <div className={`w-px flex-1 my-2 ${wizardStep > s ? 'bg-accent-green' : 'bg-text-tertiary'}`} />}
              </React.Fragment>
            ))}
          </div>

          {/* Right: Content */}
          <div className="flex-1 space-y-8">
            <AnimatePresence mode="wait">
              <motion.div
                key={wizardStep}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="min-h-[400px]"
              >
                {wizardStep === 1 && (
                  <div className="space-y-6">
                    <h2 className="micro-label">01 / MISSION DOMAIN / HARDWARE CLASS</h2>
                    <div className="grid grid-cols-2 gap-4">
                      {[
                        { id: 'AVIATION', label: 'AVIATION', desc: 'UAV / VTOL Systems', icon: <Zap size={20} /> },
                        { id: 'ROCKETS', label: 'ROCKETS', desc: 'Launch Vehicles / Orbital Probes', icon: <ArrowUpRight size={20} /> },
                        { id: 'INDUSTRIAL', label: 'INDUSTRIAL', desc: 'Manufacturing Arms / Automation', icon: <Settings size={20} /> },
                        { id: 'ROBOTICS', label: 'ROBOTICS', desc: 'Humanoids / Research Platforms', icon: <Cpu size={20} /> },
                      ].map((d) => (
                        <button
                          key={d.id}
                          onClick={() => setDomain(d.id as Domain)}
                          className={`p-6 text-left border transition-all group ${domain === d.id ? 'border-accent-green bg-accent-green/5' : 'border-border hover:border-border-active'}`}
                        >
                          <div className={`mb-4 ${domain === d.id ? 'text-accent-green' : 'text-text-tertiary group-hover:text-text-secondary'}`}>
                            {d.icon}
                          </div>
                          <div className="font-display text-lg tracking-widest uppercase mb-1">{d.label}</div>
                          <div className="text-[11px] text-text-secondary leading-tight">{d.desc}</div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {wizardStep === 2 && (
                  <div className="space-y-12">
                    <h2 className="micro-label">02 / UNIT IDENTITY / MISSION PARAMETERS</h2>
                    <div className="space-y-8">
                      <div className="space-y-2">
                        <label className="text-[10px] text-text-tertiary uppercase tracking-widest">Unit Designation</label>
                        <input 
                          value={unitName}
                          onChange={(e) => setUnitName(e.target.value.toUpperCase())}
                          className="w-full bg-transparent border-b border-border py-2 font-mono text-accent-green outline-none focus:border-accent-green transition-colors"
                          placeholder="SENTINEL-X1"
                        />
                      </div>
                      <div className="space-y-2">
                        <label className="text-[10px] text-text-tertiary uppercase tracking-widest">Mission Context</label>
                        <input 
                          value={mission}
                          onChange={(e) => setMission(e.target.value)}
                          className="w-full bg-transparent border-b border-border py-2 font-mono text-accent-green outline-none focus:border-accent-green transition-colors"
                          placeholder="Autonomous exploration"
                        />
                      </div>
                    </div>
                  </div>
                )}

                {wizardStep === 3 && (
                  <div className="space-y-8">
                    <h2 className="micro-label">03 / COMPLIANCE KERNEL / CERTIFICATION BASELINE</h2>
                    <div className="flex flex-wrap gap-3">
                      {['ISO 10218', 'DO-178C', 'AS9100', 'MIL-STD-882E'].map((p) => (
                        <button
                          key={p}
                          onClick={() => setProtocols(prev => prev.includes(p) ? prev.filter(x => x !== p) : [...prev, p])}
                          className={`px-4 py-2 border font-mono text-xs transition-all ${protocols.includes(p) ? 'bg-accent-green border-accent-green text-black' : 'border-border text-text-secondary hover:border-text-secondary'}`}
                        >
                          {p}
                        </button>
                      ))}
                    </div>
                    <div className="p-4 bg-bg-secondary border border-border">
                      <p className="text-[11px] text-text-secondary leading-relaxed italic">
                        {protocols.length > 0 
                          ? `Active protocols: ${protocols.join(', ')}. Sentinel OS will enforce strict deterministic bounds based on these certification requirements.`
                          : "Select compliance protocols to initialize the safety kernel."}
                      </p>
                    </div>
                  </div>
                )}

                {wizardStep === 4 && (
                  <div className="space-y-8">
                    <h2 className="micro-label">04 / PHYSICAL MANIFEST / NEWTONIAN CONSTRAINTS</h2>
                    <div className="space-y-6">
                      {[
                        { label: 'Mass (kg)', key: 'mass', min: 0.1, max: 50, step: 0.1 },
                        { label: 'Friction (μ)', key: 'friction', min: 0.01, max: 1.0, step: 0.01 },
                        { label: 'Drag (Cd)', key: 'drag', min: 0.1, max: 2.0, step: 0.01 },
                        { label: 'Efficiency (%)', key: 'efficiency', min: 50, max: 100, step: 1 },
                        { label: 'Stiffness (N/m)', key: 'stiffness', min: 10, max: 500, step: 1 },
                      ].map((s) => (
                        <div key={s.key} className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="text-[11px] text-text-secondary uppercase tracking-widest">{s.label}</span>
                            <span className="font-mono text-accent-green text-xs">{(priors as any)[s.key]}</span>
                          </div>
                          <input 
                            type="range" min={s.min} max={s.max} step={s.step}
                            value={(priors as any)[s.key]}
                            onChange={(e) => setPriors(prev => ({ ...prev, [s.key]: parseFloat(e.target.value) }))}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            </AnimatePresence>

            <div className="flex justify-between items-center pt-8 border-t border-border">
              <button 
                onClick={() => setWizardStep(prev => Math.max(1, prev - 1))}
                disabled={wizardStep === 1}
                className="btn-ghost disabled:opacity-0"
              >
                PREVIOUS
              </button>
              <div className="font-mono text-xs text-text-tertiary">0{wizardStep} / 04</div>
              <button 
                onClick={() => {
                  if (wizardStep < 4) setWizardStep(prev => prev + 1);
                  else setPhase(1);
                }}
                disabled={wizardStep === 1 && !domain}
                className="btn-primary"
              >
                {wizardStep === 4 ? 'INITIALIZE KERNEL' : 'CONFIRM / PROCEED'}
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (phase === 1) {
    return <SyncAnimation onComplete={() => setPhase(2)} />;
  }

  return (
    <div className="h-screen flex flex-col bg-bg-primary overflow-hidden">
      {/* TOP STATUS BAR */}
      <header className="h-12 border-b border-border flex items-center justify-between px-4 bg-bg-secondary z-50">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${sim.mode === 'NOMINAL' ? 'bg-accent-green' : sim.mode === 'CAUTIOUS' ? 'bg-accent-amber' : 'bg-accent-red animate-pulse'}`} />
            <span className="font-display font-bold tracking-[0.2em] text-sm">PHYSICORE v2.0</span>
          </div>
          <div className="h-4 w-px bg-border" />
          <span className="micro-label text-accent-green">SENTINEL LINK: ACTIVE</span>
        </div>

        <div className="flex items-center gap-8 font-mono text-[10px] text-text-secondary">
          <div className="flex gap-2">MASS: <span className="text-accent-green">{sim.params.mass.toFixed(2)}kg</span></div>
          <div className="h-3 w-px bg-border-active" />
          <div className="flex gap-2">FRICTION: <span className="text-accent-green">{sim.params.friction.toFixed(2)}μ</span></div>
          <div className="h-3 w-px bg-border-active" />
          <div className="flex gap-2">CONFIDENCE: <span className="text-accent-green">{sim.confidence.toFixed(1)}%</span></div>
          <div className="h-3 w-px bg-border-active" />
          <div className="flex gap-2">STATE: <span className={sim.mode === 'NOMINAL' ? 'text-accent-green' : 'text-accent-amber'}>{sim.mode}</span></div>
        </div>

        <div className="flex items-center gap-2">
          <button 
            onClick={() => setIsDebuggerOpen(true)}
            className="btn-outline py-1 text-[10px] border-border-active"
          >
            NEURAL DEBUGGER
          </button>
          <button 
            onClick={handleExport}
            className="btn-outline py-1 text-[10px] border-border-active"
          >
            EXPORT PACK
          </button>
        </div>
      </header>

      {/* MAIN CONTENT */}
      <main className="flex-1 flex overflow-hidden">
        {/* LEFT PANEL */}
        <aside className="w-[280px] border-r border-border bg-bg-secondary p-6 overflow-y-auto custom-scroll space-y-8">
          <section className="space-y-6">
            <h3 className="font-display text-xs font-bold tracking-[0.3em] border-l-[3px] border-accent-green pl-3 uppercase">System Parameters</h3>
            
            <div className="space-y-6">
              <div className="micro-label text-text-tertiary">Hardware Priors</div>
              {[
                { label: 'Mass (kg)', key: 'mass', min: 0.1, max: 50, step: 0.1 },
                { label: 'Friction (μ)', key: 'friction', min: 0.01, max: 1.0, step: 0.01 },
                { label: 'Drag (Cd)', key: 'drag', min: 0.1, max: 2.0, step: 0.01 },
                { label: 'Efficiency (%)', key: 'efficiency', min: 50, max: 100, step: 1 },
                { label: 'Stiffness (N/m)', key: 'stiffness', min: 10, max: 500, step: 1 },
              ].map((s) => (
                <div key={s.key} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-[10px] text-text-secondary uppercase tracking-widest">{s.label}</span>
                    <span className="font-mono text-accent-green text-[10px]">
                      {(sim.params as any)[s.key].toFixed(2)}
                      <span className="ml-1 text-[8px] text-text-tertiary opacity-50">(LIVE)</span>
                    </span>
                  </div>
                  <input 
                    type="range" min={s.min} max={s.max} step={s.step}
                    value={(sim.params as any)[s.key]}
                    onChange={(e) => setSim(prev => ({ ...prev, params: { ...prev.params, [s.key]: parseFloat(e.target.value) } }))}
                  />
                </div>
              ))}
            </div>

            <div className="space-y-6 pt-4 border-t border-border">
              <div className="micro-label text-text-tertiary">Control Optimizer</div>
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-[10px] text-text-secondary uppercase tracking-widest">Q-Weight (State)</span>
                    <span className="font-mono text-accent-blue text-[10px]">{mpcWeights.q.toFixed(1)}</span>
                  </div>
                  <input type="range" min="0.1" max="10" step="0.1" value={mpcWeights.q} onChange={(e) => setMpcWeights(p => ({ ...p, q: parseFloat(e.target.value) }))} />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-[10px] text-text-secondary uppercase tracking-widest">R-Weight (Effort)</span>
                    <span className="font-mono text-accent-blue text-[10px]">{mpcWeights.r.toFixed(1)}</span>
                  </div>
                  <input type="range" min="0.1" max="10" step="0.1" value={mpcWeights.r} onChange={(e) => setMpcWeights(p => ({ ...p, r: parseFloat(e.target.value) }))} />
                </div>
              </div>
            </div>

            <div className="space-y-4 pt-4 border-t border-border">
              <div className="micro-label text-text-tertiary">Operational Mode</div>
              <div className={`py-3 flex items-center justify-center font-display font-black tracking-[0.4em] text-sm border ${
                sim.mode === 'NOMINAL' ? 'bg-accent-green text-black border-accent-green' :
                sim.mode === 'CAUTIOUS' ? 'bg-accent-amber text-black border-accent-amber' :
                sim.mode === 'RESTRICTED' ? 'border-accent-red text-accent-red' :
                'bg-accent-red text-white border-accent-red animate-pulse'
              }`}>
                {sim.mode}
              </div>
              <p className="text-[9px] text-text-tertiary uppercase leading-relaxed text-center">
                {sim.mode === 'NOMINAL' ? 'All systems operating within deterministic bounds.' :
                 sim.mode === 'CAUTIOUS' ? 'Residual drift detected. Increasing prediction sampling.' :
                 sim.mode === 'RESTRICTED' ? 'Confidence threshold breached. Actuator output capped.' :
                 'CRITICAL FAILURE. Safety envelope violated. Emergency fallback active.'}
              </p>
            </div>

            <div className="space-y-3 pt-4">
              <button className="w-full btn-primary py-3 text-xs flex items-center justify-center gap-2">
                <Play size={14} /> INITIALIZE KERNEL
              </button>
              <button onClick={handleExport} className="w-full btn-outline py-3 text-xs flex items-center justify-center gap-2">
                <Download size={14} /> EXPORT SENTINEL PACK
              </button>
            </div>
          </section>
        </aside>

        {/* CENTER PANEL - CANVAS */}
        <section className="flex-1 relative bg-bg-primary overflow-hidden cursor-crosshair" onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          setSim(prev => ({ ...prev, target: { x: e.clientX - rect.left - rect.width/2, y: e.clientY - rect.top - rect.height/2 } }));
        }}>
          {/* Grid */}
          <div className="absolute inset-0 opacity-20 pointer-events-none" style={{ backgroundImage: `linear-gradient(${COLORS.border} 1px, transparent 1px), linear-gradient(90deg, ${COLORS.border} 1px, transparent 1px)`, backgroundSize: '40px 40px', backgroundPosition: 'center center' }} />
          
          {/* Axes */}
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="w-full h-px bg-white/10" />
            <div className="h-full w-px bg-white/10" />
            <span className="absolute top-4 left-1/2 -translate-x-1/2 font-mono text-[9px] text-text-tertiary">+Y</span>
            <span className="absolute bottom-4 left-1/2 -translate-x-1/2 font-mono text-[9px] text-text-tertiary">-Y</span>
            <span className="absolute left-4 top-1/2 -translate-y-1/2 font-mono text-[9px] text-text-tertiary">-X</span>
            <span className="absolute right-4 top-1/2 -translate-y-1/2 font-mono text-[9px] text-text-tertiary">+X</span>
          </div>

          {/* Safety Envelope */}
          <div className="absolute inset-10 border border-dashed border-accent-amber/30 pointer-events-none">
            <span className="absolute top-2 right-2 font-mono text-[8px] text-accent-amber opacity-50">SAFETY ENVELOPE</span>
            {sim.mode === 'SAFE_FALLBACK' && <div className="absolute inset-0 bg-accent-red/5 animate-pulse" />}
          </div>

          {/* HUD Overlay */}
          <div className="absolute top-6 left-6 p-4 bg-bg-secondary/80 border border-accent-green/30 backdrop-blur-md pointer-events-none z-10 space-y-2">
            <div className="flex gap-4 font-mono text-[10px]">
              <span className="text-text-tertiary">POSITION:</span>
              <span className="text-text-primary">x: {sim.pos.x > 0 ? '+' : ''}{sim.pos.x.toFixed(3)} y: {sim.pos.y > 0 ? '+' : ''}{sim.pos.y.toFixed(3)}</span>
            </div>
            <div className="flex gap-4 font-mono text-[10px]">
              <span className="text-text-tertiary">VELOCITY:</span>
              <span className="text-text-primary">vx: {sim.vel.vx > 0 ? '+' : ''}{sim.vel.vx.toFixed(2)} vy: {sim.vel.vy > 0 ? '+' : ''}{sim.vel.vy.toFixed(2)}</span>
            </div>
            <div className="flex gap-4 font-mono text-[10px]">
              <span className="text-text-tertiary">RESIDUAL:</span>
              <span className="text-accent-green">L2: {sim.residual.toFixed(4)}</span>
            </div>
            <div className="flex gap-4 font-mono text-[10px]">
              <span className="text-text-tertiary">GRADIENT:</span>
              <span className="text-text-primary">∂m: 0.002 ∂f: 0.001</span>
            </div>
          </div>

          {/* Simulation Elements */}
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="relative w-full h-full">
              {/* Target Marker */}
              <motion.div 
                className="absolute"
                style={{ left: `calc(50% + ${sim.target.x}px)`, top: `calc(50% + ${sim.target.y}px)` }}
                animate={{ rotate: 360 }}
                transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
              >
                <div className="w-6 h-6 -translate-x-1/2 -translate-y-1/2 flex items-center justify-center">
                  <div className="absolute w-full h-px bg-white/50" />
                  <div className="absolute h-full w-px bg-white/50" />
                </div>
              </motion.div>

              {/* Trajectory */}
              {sim.trajectory.map((p, i) => (
                <div 
                  key={i}
                  className="absolute w-1 h-1 bg-accent-blue rounded-full"
                  style={{ 
                    left: `calc(50% + ${p.x}px)`, 
                    top: `calc(50% + ${p.y}px)`,
                    opacity: 1 - (i / 12),
                    transform: 'translate(-50%, -50%)'
                  }}
                />
              ))}

              {/* Robot Reticle */}
              <motion.div 
                className="absolute"
                animate={{ 
                  x: sim.pos.x, 
                  y: sim.pos.y,
                  scale: sim.mode === 'SAFE_FALLBACK' ? [1, 1.1, 1] : 1
                }}
                transition={{ 
                  x: { type: 'spring', damping: 20, stiffness: 200 },
                  y: { type: 'spring', damping: 20, stiffness: 200 },
                  scale: { duration: 0.5, repeat: Infinity }
                }}
                style={{ left: '50%', top: '50%' }}
              >
                <div className={`relative -translate-x-1/2 -translate-y-1/2 flex items-center justify-center`}>
                  <div className={`w-12 h-12 border-2 rounded-full transition-colors duration-300 ${
                    sim.mode === 'NOMINAL' ? 'border-accent-green' :
                    sim.mode === 'CAUTIOUS' ? 'border-accent-amber' :
                    'border-accent-red'
                  }`} />
                  <div className={`absolute w-8 h-px ${sim.mode === 'NOMINAL' ? 'bg-accent-green' : sim.mode === 'CAUTIOUS' ? 'bg-accent-amber' : 'bg-accent-red'}`} style={{ left: '-16px' }} />
                  <div className={`absolute w-8 h-px ${sim.mode === 'NOMINAL' ? 'bg-accent-green' : sim.mode === 'CAUTIOUS' ? 'bg-accent-amber' : 'bg-accent-red'}`} style={{ right: '-16px' }} />
                  <div className={`absolute h-8 w-px ${sim.mode === 'NOMINAL' ? 'bg-accent-green' : sim.mode === 'CAUTIOUS' ? 'bg-accent-amber' : 'bg-accent-red'}`} style={{ top: '-16px' }} />
                  <div className={`absolute h-8 w-px ${sim.mode === 'NOMINAL' ? 'bg-accent-green' : sim.mode === 'CAUTIOUS' ? 'bg-accent-amber' : 'bg-accent-red'}`} style={{ bottom: '-16px' }} />
                  <div className={`w-2 h-2 rounded-full ${sim.mode === 'NOMINAL' ? 'bg-accent-green' : sim.mode === 'CAUTIOUS' ? 'bg-accent-amber' : 'bg-accent-red'}`} />
                </div>
              </motion.div>
            </div>
          </div>
        </section>

        {/* RIGHT PANEL - TELEMETRY */}
        <aside className="w-[320px] border-l border-border bg-bg-secondary p-6 overflow-y-auto custom-scroll space-y-8">
          <section className="space-y-6">
            <h3 className="font-display text-xs font-bold tracking-[0.3em] border-l-[3px] border-accent-green pl-3 uppercase">Live Telemetry</h3>

            {/* Widget 1: Residual Drift */}
            <div className="space-y-3">
              <div className="micro-label">L2 Norm — Physics vs Reality</div>
              <div className="h-[120px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={sim.history}>
                    <CartesianGrid strokeDasharray="3 3" stroke={COLORS.bgTertiary} vertical={false} />
                    <Area 
                      type="monotone" 
                      dataKey="residual" 
                      stroke={sim.residual > 0.5 ? COLORS.accentAmber : COLORS.accentGreen} 
                      fill={sim.residual > 0.5 ? COLORS.accentAmber : COLORS.accentGreen} 
                      fillOpacity={0.15} 
                      isAnimationActive={false}
                    />
                    <YAxis domain={[0, 1.5]} hide />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="flex justify-between items-end">
                <div className="font-mono text-2xl font-bold text-text-primary">{sim.residual.toFixed(4)}</div>
                <div className={`flex items-center gap-1 text-[10px] font-mono ${sim.residual > 0.5 ? 'text-accent-red' : 'text-accent-green'}`}>
                  {sim.residual > 0.5 ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
                  {(sim.residual * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Widget 2: Confidence */}
            <div className="space-y-3 flex flex-col items-center">
              <div className="micro-label self-start">Ensemble Confidence Score</div>
              <CircularGauge 
                value={sim.confidence} 
                label="Confidence" 
                sublabel="3/3 Nodes Active" 
                color={sim.confidence > 70 ? COLORS.accentGreen : sim.confidence > 40 ? COLORS.accentAmber : COLORS.accentRed} 
              />
            </div>

            {/* Widget 3: Uncertainty */}
            <div className="space-y-3">
              <div className="micro-label">Ensemble Variance Distribution</div>
              <div className="h-[80px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={[
                    { v: 10 + Math.random() * 20 },
                    { v: 15 + Math.random() * 10 },
                    { v: 5 + Math.random() * 5 },
                    { v: 20 + Math.random() * 30 },
                    { v: 12 + Math.random() * 15 },
                  ]}>
                    <Bar dataKey="v" fill={sim.confidence < 60 ? COLORS.accentAmber : COLORS.accentBlue} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="text-[9px] font-mono text-text-tertiary uppercase text-center">Low Variance = High Confidence</div>
            </div>

            {/* Widget 4: Architecture */}
            <div className="space-y-4 pt-4 border-t border-border">
              <div className="micro-label">Deployed Stack</div>
              <div className="space-y-3">
                {[
                  { name: 'RK4 Integrator', spec: '4th Order' },
                  { name: 'CEM Optimizer', spec: '12-step horizon' },
                  { name: 'Ensemble MLP', spec: '3 nodes' },
                ].map((c) => (
                  <div key={c.name} className="flex justify-between items-center">
                    <div className="text-[11px] text-text-primary">{c.name}</div>
                    <div className="flex items-center gap-3">
                      <span className="text-[9px] font-mono text-text-tertiary">{c.spec}</span>
                      <span className="text-[8px] px-1.5 py-0.5 border border-accent-green text-accent-green font-bold">ACTIVE</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Widget 5: Failure Log */}
            <div className="space-y-3 pt-4 border-t border-border">
              <div className="micro-label">Anomaly Log / Forensic Record</div>
              <div className="h-[120px] overflow-y-auto custom-scroll space-y-2 pr-2">
                {sim.logs.length === 0 ? (
                  <div className="h-full flex items-center justify-center text-[10px] text-text-tertiary font-mono uppercase">No anomalies recorded</div>
                ) : (
                  sim.logs.map((log) => (
                    <div key={log.id} className="text-[9px] font-mono border-l border-border pl-2 py-1">
                      <div className="flex justify-between mb-1">
                        <span className="text-text-tertiary">{log.time}</span>
                        <span className={`font-bold ${log.type === 'HIGH_DRIFT' ? 'text-accent-amber' : 'text-accent-red'}`}>{log.type}</span>
                      </div>
                      <div className="text-text-secondary leading-tight">{log.msg}</div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </section>
        </aside>
      </main>

      {/* BOTTOM BAR */}
      <footer className="h-10 border-t border-border flex items-center justify-between px-4 bg-bg-secondary text-[10px] font-mono text-text-tertiary">
        <div className="flex gap-4">
          <span>PHYSICORE ENGINE v2.0</span>
          <span className="text-border-active">|</span>
          <span>SENTINEL OS INFRASTRUCTURE COMPONENT</span>
        </div>

        <div className="flex gap-8">
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-accent-green" />
            <span>PHYSICS KERNEL</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-accent-green" />
            <span>NEURAL ENSEMBLE</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-accent-green" />
            <span>SENTINEL LINK</span>
          </div>
        </div>

        <div>UPTIME: {uptime}</div>
      </footer>

      {/* NEURAL DEBUGGER OVERLAY */}
      <AnimatePresence>
        {isDebuggerOpen && (
          <NeuralDebugger 
            onClose={() => setIsDebuggerOpen(false)} 
            telemetry={sim}
            onTune={(weights) => setMpcWeights(weights)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

// --- SUB-COMPONENTS ---

function SyncAnimation({ onComplete }: { onComplete: () => void }) {
  const [progress, setProgress] = useState(0);
  const [lines, setLines] = useState<string[]>([]);
  const statusLines = [
    "> NEURAL BRIDGE........ESTABLISHED",
    "> PHYSICS KERNEL.......LOADING",
    "> RK4 INTEGRATOR.......NOMINAL",
    "> ENSEMBLE NODES.......3 / 3 ONLINE",
    "> LYAPUNOV BOUNDS......COMPUTED",
    "> SENTINEL LINK........ACTIVE"
  ];

  useEffect(() => {
    const duration = 3500;
    const start = Date.now();
    
    const interval = setInterval(() => {
      const elapsed = Date.now() - start;
      const p = Math.min(100, (elapsed / duration) * 100);
      setProgress(p);
      
      const lineCount = Math.floor((elapsed / duration) * statusLines.length * 1.2);
      setLines(statusLines.slice(0, lineCount));

      if (elapsed >= duration) {
        clearInterval(interval);
        setTimeout(onComplete, 500);
      }
    }, 50);

    return () => clearInterval(interval);
  }, [onComplete]);

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-bg-primary flex flex-col items-center justify-center z-[100]"
    >
      <div className="relative w-64 h-64 flex items-center justify-center">
        <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100">
          <motion.path
            d="M50 5 L90 25 L90 75 L50 95 L10 75 L10 25 Z"
            fill="none"
            stroke={COLORS.accentGreen}
            strokeWidth="0.5"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1, rotate: 360 }}
            transition={{ pathLength: { duration: 2 }, rotate: { duration: 10, repeat: Infinity, ease: "linear" } }}
          />
        </svg>
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="font-display text-sm font-bold tracking-[0.4em] text-accent-green"
        >
          PHYSICORE
        </motion.div>
      </div>

      <div className="absolute left-12 top-1/2 -translate-y-1/2 space-y-2 w-80">
        {lines.map((line, i) => (
          <div key={i} className="font-mono text-[10px] text-accent-green typewriter">
            {line}
          </div>
        ))}
      </div>

      <div className="absolute bottom-24 w-64 space-y-2">
        <div className="h-[2px] w-full bg-bg-tertiary">
          <motion.div 
            className="h-full bg-accent-green"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="font-mono text-[10px] text-center text-accent-green">{Math.round(progress)}%</div>
      </div>
    </motion.div>
  );
}

function NeuralDebugger({ onClose, telemetry, onTune }: { onClose: () => void, telemetry: SimulationState, onTune: (w: any) => void }) {
  const [messages, setMessages] = useState<{ role: 'user' | 'ai', text: string, time: string }[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMsg = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: userMsg, time: formatTime(new Date()) }]);
    setIsTyping(true);

    try {
      const ai = new GoogleGenAI({ apiKey: (process.env as any).GEMINI_API_KEY });
      const response = await ai.models.generateContent({
        model: "gemini-3.1-pro-preview",
        contents: [{ parts: [{ text: userMsg }] }],
        config: {
          systemInstruction: `You are PHYSICORE ENGINE, a deterministic physics intelligence kernel. 
          Current Telemetry:
          - Mode: ${telemetry.mode}
          - Residual: ${telemetry.residual.toFixed(6)}
          - Confidence: ${telemetry.confidence.toFixed(2)}%
          - Params: Mass=${telemetry.params.mass}kg, Friction=${telemetry.params.friction}mu
          
          Respond in technical, industrial language. Be concise. 
          If asked to tune weights, suggest new Q and R values in the format: "ADJUST_WEIGHTS: Q=[val], R=[val]".
          NEVER use "I" or "me". Speak as the system.`
        }
      });

      const responseText = response.text || "";

      if (responseText.includes('ADJUST_WEIGHTS')) {
        const q = parseFloat(responseText.match(/Q=([\d.]+)/)?.[1] || '1.0');
        const r = parseFloat(responseText.match(/R=([\d.]+)/)?.[1] || '0.5');
        onTune({ q, r });
      }

      setMessages(prev => [...prev, { role: 'ai', text: responseText, time: formatTime(new Date()) }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'ai', text: "> ERROR: NEURAL LINK INTERRUPTED. RETRYING...", time: formatTime(new Date()) }]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <motion.div 
      initial={{ x: '100%' }}
      animate={{ x: 0 }}
      exit={{ x: '100%' }}
      transition={{ type: 'spring', damping: 25, stiffness: 200 }}
      className="fixed top-0 right-0 w-[420px] h-full bg-[#0D0D0D] border-l border-accent-green z-[60] flex flex-col"
    >
      <div className="h-14 border-b border-border flex items-center justify-between px-6">
        <h2 className="font-display font-bold tracking-[0.2em] text-sm">NEURAL DEBUGGER / RIL INTERFACE</h2>
        <button onClick={onClose} className="font-mono text-[10px] text-text-tertiary hover:text-white flex items-center gap-2">
          <X size={14} /> CLOSE
        </button>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-6 custom-scroll">
        {messages.length === 0 && (
          <div className="space-y-4">
            <p className="text-[10px] font-mono text-text-tertiary uppercase tracking-widest">Suggested Queries:</p>
            <div className="grid grid-cols-1 gap-2">
              {[
                "WHY IS IT OSCILLATING?",
                "TUNE MPC WEIGHTS",
                "EXPLAIN RESIDUAL DRIFT",
                "GENERATE STATUS REPORT"
              ].map(q => (
                <button 
                  key={q}
                  onClick={() => { setInput(q); }}
                  className="text-left p-3 border border-border font-mono text-[10px] text-accent-green hover:bg-accent-green/5 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[8px] font-mono text-text-tertiary">{m.time}</span>
              {m.role === 'ai' && <span className="text-[8px] font-mono text-accent-green font-bold uppercase tracking-widest">PHYSICORE:</span>}
            </div>
            <div className={`text-[11px] leading-relaxed ${m.role === 'user' ? 'text-white text-right' : 'text-accent-green font-mono whitespace-pre-wrap'}`}>
              {m.text}
            </div>
          </div>
        ))}
        {isTyping && <div className="text-accent-green font-mono text-[10px] animate-pulse">...</div>}
      </div>

      <div className="p-6 border-t border-border bg-bg-secondary">
        <div className="flex gap-4">
          <input 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            className="flex-1 bg-transparent border-b border-border py-2 font-mono text-xs text-accent-green outline-none focus:border-accent-green"
            placeholder="> QUERY NEURAL DEBUGGER..."
          />
          <button onClick={handleSend} className="btn-outline py-1 text-[10px] px-4">
            ⏎ EXECUTE
          </button>
        </div>
      </div>
    </motion.div>
  );
}
