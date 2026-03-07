import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Activity, Cpu, Shield, Zap, ChevronRight, ChevronLeft, 
  Play, Download, Terminal, AlertTriangle, CheckCircle2, 
  Settings, Crosshair, ArrowUpRight, ArrowDownRight, X,
  Maximize2, Activity as FrequencyIcon, RefreshCw, Cpu as Chip,
  Globe, Link, Wifi, Radio, HardDrive, FileJson, Copy, Check
} from 'lucide-react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, 
  ResponsiveContainer, BarChart, Bar, ReferenceLine
} from 'recharts';
import { GoogleGenAI } from "@google/genai";

// --- CONSTANTS ---
const COLORS = {
  bgPrimary: '#0A0A0A', bgSecondary: '#111111', bgTertiary: '#1A1A1A',
  border: '#2A2A2A', borderActive: '#3A3A3A',
  textPrimary: '#F0F0F0', textSecondary: '#888888', textTertiary: '#555555',
  accentGreen: '#00FF88', accentAmber: '#FFB800', accentRed: '#FF3333',
  accentBlue: '#0088FF', accentWhite: '#FFFFFF',
};

// --- TYPES ---
type Phase = 'wizard' | 'sync' | 'connect' | 'dashboard';
type Domain = 'AVIATION' | 'ROCKETS' | 'INDUSTRIAL' | 'ROBOTICS';
type SystemMode = 'NOMINAL' | 'CAUTIOUS' | 'RESTRICTED' | 'SAFE_FALLBACK';
type ConnectionType = 'live' | 'hil' | 'twin';

interface PhysicalParams {
  mass: number; friction: number; drag: number; efficiency: number; stiffness: number;
}

interface LogEntry { time: string; type: string; desc: string; }

// --- UTILS ---
const formatTime = (date: Date) => {
  const h = date.getHours().toString().padStart(2, '0');
  const m = date.getMinutes().toString().padStart(2, '0');
  const s = date.getSeconds().toString().padStart(2, '0');
  const ms = date.getMilliseconds().toString().padStart(3, '0');
  return `${h}:${m}:${s}.${ms}`;
};

const mag = (v: { x: number; y: number }) => Math.sqrt(v.x * v.x + v.y * v.y);
const norm = (v: { x: number; y: number }) => {
  const m = mag(v);
  return m === 0 ? { x: 0, y: 0 } : { x: v.x / m, y: v.y / m };
};

// --- COMPONENTS ---

const CircularGauge = ({ value, color }: { value: number; color: string }) => {
  const radius = 45;
  const circ = 2 * Math.PI * radius;
  const offset = circ - (value / 100) * circ * (240/360);
  return (
    <div className="relative w-[120px] h-[120px] flex items-center justify-center">
      <svg className="w-full h-full -rotate-[210deg]" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r={radius} fill="none" stroke="#1A1A1A" strokeWidth="8" strokeDasharray={`${circ * (240/360)} ${circ}`} strokeLinecap="butt" />
        <circle cx="60" cy="60" r={radius} fill="none" stroke={color} strokeWidth="8" strokeDasharray={`${circ * (240/360)} ${circ}`} strokeDashoffset={offset} strokeLinecap="butt" className="transition-all duration-500" />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center pt-2">
        <span className="font-display text-2xl font-bold text-text-primary">{Math.round(value)}</span>
        <span className="text-[8px] font-mono text-text-tertiary uppercase">Confidence</span>
      </div>
    </div>
  );
};

// --- MAIN APP ---

export default function App() {
  const [phase, setPhase] = useState<Phase>('wizard');
  const [wizardStep, setWizardStep] = useState(1);
  const [wizardData, setWizardData] = useState({
    domain: 'AVIATION' as Domain, unitName: 'SENTINEL-X1', mission: 'Autonomous exploration unit',
    protocols: [] as string[], priors: { mass: 2.5, friction: 0.35, drag: 0.47, efficiency: 85, stiffness: 120 }
  });

  const [syncDisplay, setSyncDisplay] = useState(0);
  const [debuggerOpen, setDebuggerOpen] = useState(false);
  const [terminalOpen, setTerminalOpen] = useState(false);
  const [anomalyLog, setAnomalyLog] = useState<LogEntry[]>([]);
  const [uiParams, setUiParams] = useState({ mass: 2.5, friction: 0.35, confidence: 100, stability: 100, mode: 'NOMINAL' as SystemMode });
  
  const hardwareConnected = useRef(false);
  const connectionMode = useRef({ type: 'twin' as ConnectionType, profile: 'Digital Twin', endpoint: '', noiseIntensity: 0.05, latencyMs: 0 });
  const syncStarted = useRef(false);
  const requestRef = useRef<number>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sessionStart = useRef(Date.now());
  const metStart = useRef<number | null>(null);

  // Simulation State Refs
  const sim = useRef({
    pos: { x: 0, y: 0 }, vel: { vx: 0, vy: 0 }, target: { x: 0, y: 0 },
    mass: 2.5, friction: 0.35, drag: 0.47, efficiency: 85,
    residualL2: 0, confidence: 100, stabilityScore: 100,
    mode: 'NOMINAL' as SystemMode, frameCount: 0,
    paramHistory: { mass: [] as number[], friction: [] as number[] },
    trajectoryPts: [] as { x: number; y: number }[],
    velocityHistory: [] as number[], residualHistory: [] as number[],
    varianceBuckets: new Array(8).fill(0),
    anomalyLog: [] as LogEntry[],
    qWeight: 1.0, rWeight: 0.5,
    lastResidualAbove05: false, boundaryViolatedFrames: 0,
    phasePortrait: [] as { x: number; velMag: number }[],
    showPhasePortrait: true
  });

  // Sync Animation
  useEffect(() => {
    if (phase !== 'sync' || syncStarted.current) return;
    syncStarted.current = true;
    const start = performance.now();
    const duration = 4000;
    const tick = (now: number) => {
      const p = Math.min((now - start) / duration, 1);
      setSyncDisplay(Math.floor(p * 100));
      if (p < 1) requestAnimationFrame(tick);
      else setTimeout(() => setPhase('connect'), 400);
    };
    requestAnimationFrame(tick);
  }, [phase]);

  // Simulation Logic
  const runSim = useCallback(() => {
    if (phase !== 'dashboard' || !hardwareConnected.current) return;
    const s = sim.current;
    const dt = 1/60;
    s.frameCount++;

    const deriv = (p: {x:number, y:number}, v: {vx:number, vy:number}, mass: number, friction: number, noiseScale = 1.0) => {
      const d = { x: s.target.x - p.x, y: s.target.y - p.y };
      const dist = Math.min(mag(d), 180);
      const dir = norm(d);
      const thrust = { x: dir.x * dist * 0.9 * (s.efficiency/100), y: dir.y * dist * 0.9 * (s.efficiency/100) };
      const vM = mag({x: v.vx, y: v.vy});
      const drag_f = { x: -v.vx * s.drag * 0.4 * vM, y: -v.vy * s.drag * 0.4 * vM };
      const fric_f = { x: -v.vx * friction * 2.2, y: -v.vy * friction * 2.2 };
      const nS = connectionMode.current.noiseIntensity * 1.2 * noiseScale;
      const noise = { x: (Math.random()-0.5)*nS, y: (Math.random()-0.5)*nS };
      return {
        dp: { x: v.vx, y: v.vy },
        dv: { x: (thrust.x + drag_f.x + fric_f.x + noise.x) / mass, y: (thrust.y + drag_f.y + fric_f.y + noise.y) / mass }
      };
    };

    // RK4
    const k1 = deriv(s.pos, s.vel, s.mass, s.friction);
    const k2 = deriv({x: s.pos.x + k1.dp.x*dt*0.5, y: s.pos.y + k1.dp.y*dt*0.5}, {vx: s.vel.vx + k1.dv.x*dt*0.5, vy: s.vel.vy + k1.dv.y*dt*0.5}, s.mass, s.friction);
    const k3 = deriv({x: s.pos.x + k2.dp.x*dt*0.5, y: s.pos.y + k2.dp.y*dt*0.5}, {vx: s.vel.vx + k2.dv.x*dt*0.5, vy: s.vel.vy + k2.dv.y*dt*0.5}, s.mass, s.friction);
    const k4 = deriv({x: s.pos.x + k3.dp.x*dt, y: s.pos.y + k3.dp.y*dt}, {vx: s.vel.vx + k3.dv.x*dt, vy: s.vel.vy + k3.dv.y*dt}, s.mass, s.friction);

    const newPos = { x: s.pos.x + (dt/6)*(k1.dp.x + 2*k2.dp.x + 2*k3.dp.x + k4.dp.x), y: s.pos.y + (dt/6)*(k1.dp.y + 2*k2.dp.y + 2*k3.dp.y + k4.dp.y) };
    const newVel = { vx: s.vel.vx + (dt/6)*(k1.dv.x + 2*k2.dv.x + 2*k3.dv.x + k4.dv.x), vy: s.vel.vy + (dt/6)*(k1.dv.y + 2*k2.dv.y + 2*k3.dv.y + k4.dv.y) };
    const vM = mag({x: newVel.vx, y: newVel.vy});
    if (vM > 220) { newVel.vx *= 220/vM; newVel.vy *= 220/vM; }

    // Ensemble & Residual
    const scales = [0.7, 1.0, 1.3];
    const preds = scales.map(sc => {
      const d = deriv(s.pos, s.vel, s.mass, s.friction, sc);
      return { x: s.pos.x + d.dp.x*dt, y: s.pos.y + d.dp.y*dt };
    });
    const avgP = { x: (preds[0].x+preds[1].x+preds[2].x)/3, y: (preds[0].y+preds[1].y+preds[2].y)/3 };
    const rawRes = Math.sqrt(Math.pow(avgP.x - newPos.x, 2) + Math.pow(avgP.y - newPos.y, 2)) * 10;
    s.residualL2 = 0.94 * s.residualL2 + 0.06 * rawRes;
    const varRaw = (Math.pow(preds[0].x-avgP.x,2) + Math.pow(preds[1].x-avgP.x,2) + Math.pow(preds[2].x-avgP.x,2))/3;
    const confRaw = Math.max(0, Math.min(100, 100 - varRaw * 180));
    s.confidence = 0.96 * s.confidence + 0.04 * confRaw;
    s.stabilityScore = Math.max(0, Math.min(100, s.confidence * 0.6 + (1 - s.residualL2) * 40));

    // SystemID
    if (s.frameCount % 50 === 0) {
      const pM = 0.025, pF = 0.008;
      const getErr = (m:number, f:number) => {
        const d = deriv(s.pos, s.vel, m, f, 0);
        const p = { x: s.pos.x + d.dp.x*dt, y: s.pos.y + d.dp.y*dt };
        return Math.sqrt(Math.pow(p.x - newPos.x, 2) + Math.pow(p.y - newPos.y, 2));
      };
      const gM = (getErr(s.mass + pM, s.friction) - getErr(s.mass - pM, s.friction)) / 0.05;
      const gF = (getErr(s.mass, s.friction + pF) - getErr(s.mass, s.friction - pF)) / 0.016;
      s.mass = Math.max(0.3, Math.min(25, s.mass - 0.015 * gM));
      s.friction = Math.max(0.02, Math.min(0.95, s.friction - 0.006 * gF));
      s.paramHistory.mass.push(s.mass); s.paramHistory.friction.push(s.friction);
      if (s.paramHistory.mass.length > 30) { s.paramHistory.mass.shift(); s.paramHistory.friction.shift(); }
    }

    // Freq Analysis
    s.velocityHistory.push(vM);
    if (s.velocityHistory.length > 120) s.velocityHistory.shift();
    if (s.frameCount % 30 === 0 && s.velocityHistory.length === 120) {
      for (let i = 0; i < 8; i++) {
        const chunk = s.velocityHistory.slice(i*15, (i+1)*15);
        s.varianceBuckets[i] = chunk.reduce((a,b) => a+b, 0) / 15;
      }
    }

    // Mode & Boundary
    const margin = 55;
    const canvas = canvasRef.current;
    if (canvas) {
      const w = canvas.width, h = canvas.height;
      if (Math.abs(newPos.x) > w/2 - margin || Math.abs(newPos.y) > h/2 - margin) {
        s.boundaryViolatedFrames = 120;
        newVel.vx *= 0.35; newVel.vy *= 0.35;
        if (s.frameCount % 60 === 0) addLog('VIOLATION', 'Safety envelope breach detected');
      } else {
        s.boundaryViolatedFrames = Math.max(0, s.boundaryViolatedFrames - 1);
      }
    }

    const prevMode = s.mode;
    if (s.confidence < 22 || s.residualL2 > 1.3 || s.boundaryViolatedFrames > 0) s.mode = 'SAFE_FALLBACK';
    else if (s.confidence < 42 || s.residualL2 > 0.85) s.mode = 'RESTRICTED';
    else if (s.confidence < 62 || s.residualL2 > 0.42) s.mode = 'CAUTIOUS';
    else s.mode = 'NOMINAL';
    if (s.mode !== prevMode) addLog('MODE_CHANGE', `System transitioned to ${s.mode}`);

    if (s.residualL2 > 0.5 && !s.lastResidualAbove05) { addLog('HIGH_DRIFT', 'Residual threshold exceeded 0.5'); s.lastResidualAbove05 = true; }
    if (s.residualL2 < 0.2 && s.lastResidualAbove05) { addLog('RECOVERY', 'Residual normalized below 0.2'); s.lastResidualAbove05 = false; }

    s.pos = newPos; s.vel = newVel;
    s.residualHistory.push(s.residualL2);
    if (s.residualHistory.length > 120) s.residualHistory.shift();
    s.phasePortrait.push({ x: s.pos.x, velMag: vM });
    if (s.phasePortrait.length > 200) s.phasePortrait.shift();

    // MPC
    s.trajectoryPts = [];
    let tP = { ...newPos }, tV = { ...newVel };
    for (let i = 0; i < 12; i++) {
      const d = deriv(tP, tV, s.mass, s.friction, 0);
      tP = { x: tP.x + d.dp.x*dt, y: tP.y + d.dp.y*dt };
      tV = { vx: tV.vx + d.dv.x*dt, vy: tV.vy + d.dv.y*dt };
      s.trajectoryPts.push({ ...tP });
    }

    if (s.frameCount % 6 === 0) setUiParams({ mass: s.mass, friction: s.friction, confidence: s.confidence, stability: s.stabilityScore, mode: s.mode });
    
    draw();
    requestRef.current = requestAnimationFrame(runSim);
  }, [phase]);

  const addLog = (type: string, desc: string) => {
    const entry = { time: formatTime(new Date()), type, desc };
    sim.current.anomalyLog.unshift(entry);
    if (sim.current.anomalyLog.length > 50) sim.current.anomalyLog.pop();
    setAnomalyLog([...sim.current.anomalyLog]);
  };

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const s = sim.current;
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = '#080808'; ctx.fillRect(0,0,w,h);
    ctx.save(); ctx.translate(w/2, h/2);
    
    // Grid
    ctx.strokeStyle = '#131313'; ctx.lineWidth = 0.5;
    for (let x = -w/2; x < w/2; x += 40) { ctx.beginPath(); ctx.moveTo(x, -h/2); ctx.lineTo(x, h/2); ctx.stroke(); }
    for (let y = -h/2; y < h/2; y += 40) { ctx.beginPath(); ctx.moveTo(-w/2, y); ctx.lineTo(w/2, y); ctx.stroke(); }
    ctx.strokeStyle = '#222222'; ctx.beginPath(); ctx.moveTo(-w/2, 0); ctx.lineTo(w/2, 0); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0, -h/2); ctx.lineTo(0, h/2); ctx.stroke();
    ctx.fillStyle = '#333333'; ctx.font = '9px JetBrains Mono'; ctx.fillText('+Y', 5, -h/2+15); ctx.fillText('-Y', 5, h/2-5); ctx.fillText('-X', -w/2+5, -5); ctx.fillText('+X', w/2-20, -5);

    // Envelope
    const m = 55;
    ctx.strokeStyle = s.boundaryViolatedFrames > 0 ? COLORS.accentRed : 'rgba(255, 184, 0, 0.2)';
    if (s.boundaryViolatedFrames > 0) { ctx.setLineDash([]); ctx.fillStyle = 'rgba(255, 51, 51, 0.03)'; ctx.fillRect(-w/2+m, -h/2+m, w-2*m, h-2*m); } else ctx.setLineDash([6, 4]);
    ctx.strokeRect(-w/2+m, -h/2+m, w-2*m, h-2*m); ctx.setLineDash([]);

    // Trajectory
    ctx.strokeStyle = 'rgba(0, 153, 255, 0.25)'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(s.pos.x, s.pos.y); s.trajectoryPts.forEach(p => ctx.lineTo(p.x, p.y)); ctx.stroke(); ctx.setLineDash([]);
    s.trajectoryPts.forEach((p, i) => { ctx.fillStyle = `rgba(0, 153, 255, ${0.9 * (1 - i/12)})`; ctx.beginPath(); ctx.arc(p.x, p.y, 2, 0, Math.PI*2); ctx.fill(); });

    // Target
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)'; ctx.beginPath(); ctx.moveTo(s.target.x-10, s.target.y); ctx.lineTo(s.target.x+10, s.target.y); ctx.moveTo(s.target.x, s.target.y-10); ctx.lineTo(s.target.x, s.target.y+10); ctx.stroke();
    ctx.save(); ctx.translate(s.target.x, s.target.y); ctx.rotate(s.frameCount * 0.01); ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'; ctx.strokeRect(-8, -8, 16, 16); ctx.restore();

    // Robot
    const mC = s.mode === 'NOMINAL' ? COLORS.accentGreen : s.mode === 'CAUTIOUS' ? COLORS.accentAmber : COLORS.accentRed;
    ctx.save(); ctx.translate(s.pos.x, s.pos.y); ctx.rotate(s.frameCount * 0.0035);
    ctx.strokeStyle = mC; ctx.lineWidth = 1.5; if (s.mode === 'SAFE_FALLBACK') ctx.globalAlpha = 0.5 + 0.5*Math.sin(s.frameCount*0.15);
    ctx.beginPath(); ctx.arc(0, 0, 20, 0, Math.PI*2); ctx.stroke();
    for (let i = 0; i < 4; i++) { ctx.rotate(Math.PI/2); ctx.beginPath(); ctx.moveTo(20, 0); ctx.lineTo(28, 0); ctx.stroke(); }
    for (let i = 0; i < 4; i++) { ctx.rotate(Math.PI/2); ctx.beginPath(); ctx.moveTo(15, 15); ctx.lineTo(18, 18); ctx.stroke(); }
    ctx.fillStyle = mC; ctx.beginPath(); ctx.arc(0, 0, 3.5, 0, Math.PI*2); ctx.fill(); ctx.restore();
    ctx.strokeStyle = 'rgba(0, 153, 255, 0.7)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(s.pos.x, s.pos.y);
    const vD = norm(s.vel); const vLen = Math.min(mag(s.vel)*0.3, 28); ctx.lineTo(s.pos.x + vD.x*vLen, s.pos.y + vD.y*vLen); ctx.stroke();

    // HUD
    ctx.restore(); ctx.fillStyle = 'rgba(13, 13, 13, 0.85)'; ctx.strokeStyle = '#2A2A2A'; ctx.fillRect(10, 10, 160, 95); ctx.strokeRect(10, 10, 160, 95);
    ctx.fillStyle = COLORS.textSecondary; ctx.font = '10px JetBrains Mono';
    const lines = [
      `POS  x:${s.pos.x >= 0 ? '+' : ''}${s.pos.x.toFixed(1).padStart(5, '0')} y:${s.pos.y >= 0 ? '+' : ''}${s.pos.y.toFixed(1).padStart(5, '0')}`,
      `VEL  vx:${s.vel.vx >= 0 ? '+' : ''}${s.vel.vx.toFixed(2)} vy:${s.vel.vy >= 0 ? '+' : ''}${s.vel.vy.toFixed(2)}`,
      `RESID L2:${s.residualL2.toFixed(4)}`,
      `STAB  ${s.stabilityScore.toFixed(1)}%`,
      `FRM   ${s.frameCount.toString().padStart(6, '0')}`
    ];
    lines.forEach((l, i) => ctx.fillText(l, 20, 25 + i*15));

    if (s.showPhasePortrait) {
      ctx.fillStyle = 'rgba(13, 13, 13, 0.85)'; ctx.fillRect(w-130, h-130, 120, 120); ctx.strokeRect(w-130, h-130, 120, 120);
      ctx.strokeStyle = '#00FFFF'; ctx.lineWidth = 1; ctx.beginPath();
      s.phasePortrait.forEach((p, i) => {
        const px = w-130 + 60 + (p.x / (w/2)) * 60; const py = h-130 + 120 - (p.velMag / 220) * 120;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }); ctx.stroke();
      ctx.fillStyle = '#00FFFF'; const last = s.phasePortrait[s.phasePortrait.length-1];
      if (last) { ctx.beginPath(); ctx.arc(w-130 + 60 + (last.x / (w/2)) * 60, h-130 + 120 - (last.velMag / 220) * 120, 2, 0, Math.PI*2); ctx.fill(); }
    }
  };

  useEffect(() => {
    if (phase === 'dashboard' && hardwareConnected.current) {
      const handleResize = () => { if (canvasRef.current) { canvasRef.current.width = canvasRef.current.parentElement?.clientWidth || 800; canvasRef.current.height = canvasRef.current.parentElement?.clientHeight || 600; } };
      window.addEventListener('resize', handleResize); handleResize();
      requestRef.current = requestAnimationFrame(runSim);
      metStart.current = Date.now();
    }
    return () => cancelAnimationFrame(requestRef.current!);
  }, [phase, runSim]);

  // --- RENDER HELPERS ---
  const renderWizard = () => (
    <div className="fixed inset-0 bg-bg-primary flex">
      <div className="w-[200px] border-r border-border flex flex-col items-center py-20 gap-12">
        {[1, 2, 3, 4].map(s => (
          <div key={s} className="flex flex-col items-center gap-2">
            <div className={`w-8 h-8 flex items-center justify-center font-mono text-xs border ${wizardStep === s ? 'border-accent-green text-accent-green' : wizardStep > s ? 'border-accent-green/40 text-accent-green/40' : 'border-text-tertiary text-text-tertiary'}`}>
              {wizardStep > s ? <CheckCircle2 size={14} /> : `0${s}`}
            </div>
            {s < 4 && <div className={`w-px h-12 ${wizardStep > s ? 'bg-accent-green/40' : 'bg-border'}`} />}
          </div>
        ))}
      </div>
      <div className="flex-1 flex flex-col justify-center px-20 max-w-[800px]">
        <AnimatePresence mode="wait">
          <motion.div key={wizardStep} initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} className="space-y-10">
            {wizardStep === 1 && (
              <div className="space-y-8">
                <h2 className="micro-label">01 / MISSION DOMAIN / HARDWARE CLASS</h2>
                <div className="grid grid-cols-2 gap-4">
                  {['AVIATION', 'ROCKETS', 'INDUSTRIAL', 'ROBOTICS'].map(d => (
                    <button key={d} onClick={() => setWizardData(p => ({ ...p, domain: d as Domain }))} className={`p-6 text-left border transition-all ${wizardData.domain === d ? 'border-accent-green bg-[#0D1A12]' : 'border-border hover:border-border-active'}`}>
                      <div className="font-display text-lg tracking-widest uppercase mb-1">{d}</div>
                      <div className="text-[11px] text-text-secondary leading-tight">{d === 'AVIATION' ? 'UAV / VTOL Systems' : d === 'ROCKETS' ? 'Launch Vehicles / Orbital' : d === 'INDUSTRIAL' ? 'Manufacturing Arms' : 'Humanoids / Research'}</div>
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
                    <label className="micro-label text-text-tertiary">Unit Designation</label>
                    <input value={wizardData.unitName} onChange={e => setWizardData(p => ({ ...p, unitName: e.target.value.toUpperCase() }))} className="w-full bg-transparent border-b border-border py-2 font-mono text-lg text-accent-green outline-none focus:border-accent-green" placeholder="SENTINEL-X1" />
                  </div>
                  <div className="space-y-2">
                    <label className="micro-label text-text-tertiary">Mission Context</label>
                    <input value={wizardData.mission} onChange={e => setWizardData(p => ({ ...p, mission: e.target.value }))} className="w-full bg-transparent border-b border-border py-2 font-body text-sm text-text-primary outline-none focus:border-accent-green" placeholder="Autonomous exploration unit" />
                  </div>
                </div>
              </div>
            )}
            {wizardStep === 3 && (
              <div className="space-y-8">
                <h2 className="micro-label">03 / COMPLIANCE KERNEL / CERTIFICATION BASELINE</h2>
                <div className="flex gap-4">
                  {['ISO 10218', 'DO-178C', 'AS9100', 'MIL-STD-882E'].map(p => (
                    <button key={p} onClick={() => setWizardData(prev => ({ ...prev, protocols: prev.protocols.includes(p) ? prev.protocols.filter(x => x !== p) : [...prev.protocols, p] }))} className={`px-4 py-2 border font-body text-xs transition-all ${wizardData.protocols.includes(p) ? 'bg-accent-green text-black border-accent-green' : 'border-border text-text-secondary hover:border-text-secondary'}`}>{p}</button>
                  ))}
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
                  ].map(s => (
                    <div key={s.key} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-[10px] text-text-secondary uppercase tracking-widest">{s.label}</span>
                        <span className="font-mono text-accent-green text-xs">{(wizardData.priors as any)[s.key]}</span>
                      </div>
                      <input type="range" min={s.min} max={s.max} step={s.step} value={(wizardData.priors as any)[s.key]} onChange={e => setWizardData(p => ({ ...p, priors: { ...p.priors, [s.key]: parseFloat(e.target.value) } }))} />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
        <div className="mt-20 flex items-center justify-between border-t border-border pt-8">
          <button onClick={() => setWizardStep(s => s - 1)} className={`btn-ghost ${wizardStep === 1 ? 'invisible' : ''}`}>← PREVIOUS</button>
          <div className="font-mono text-xs text-text-tertiary">0{wizardStep} / 04</div>
          <button onClick={() => wizardStep === 4 ? setPhase('sync') : setWizardStep(s => s + 1)} className="btn-primary">CONFIRM →</button>
        </div>
      </div>
    </div>
  );

  const renderSync = () => (
    <div className="fixed inset-0 bg-bg-primary flex flex-col items-center justify-center z-[100]">
      <div className="flex gap-20 items-center">
        <div className="w-80 space-y-2">
          {[
            "NEURAL BRIDGE........ESTABLISHED", "PHYSICS KERNEL.......INITIALIZING", "RK4 INTEGRATOR.......4TH ORDER — NOMINAL",
            "MATTER ENGINE........LINKED", "ENSEMBLE NODES.......3 / 3 ONLINE", "MPC HORIZON..........12-STEP LOOKAHEAD",
            "LYAPUNOV BOUNDS......COMPUTED — VALID", "SYSTEMID ENGINE......GRADIENT DESCENT — ARMED", "SENTINEL LINK........ACTIVE — AWAITING HARDWARE"
          ].map((line, i) => (
            <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: syncDisplay > (i*11) ? 1 : 0, x: syncDisplay > (i*11) ? 0 : -10 }} className="font-mono text-[10px] flex justify-between">
              <span className="text-text-secondary">{line.split('..')[0]}</span>
              <span className="text-text-tertiary">........</span>
              <span className="text-accent-green">{line.split('..')[line.split('..').length - 1]}</span>
            </motion.div>
          ))}
        </div>
        <div className="relative w-48 h-48 flex items-center justify-center">
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100">
            <motion.path d="M50 5 L90 25 L90 75 L50 95 L10 75 L10 25 Z" fill="none" stroke={COLORS.accentGreen} strokeWidth="2" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.5 }} />
            <motion.path d="M50 20 L75 32 L75 68 L50 80 L25 68 L25 32 Z" fill="none" stroke={COLORS.accentGreen} strokeWidth="1" animate={{ rotate: 360 }} transition={{ duration: 10, repeat: Infinity, ease: "linear" }} />
          </svg>
          <div className="font-display text-xs font-bold tracking-[0.4em] text-accent-green">PHYSICORE</div>
        </div>
      </div>
      <div className="absolute bottom-20 w-[400px] space-y-3">
        <div className="h-0.5 w-full bg-bg-tertiary overflow-hidden">
          <div className="h-full bg-accent-green transition-all duration-100" style={{ width: `${syncDisplay}%` }} />
        </div>
        <div className="flex justify-between items-center font-mono text-[10px] text-text-tertiary">
          <span>{syncDisplay === 100 ? 'KERNEL ARMED — AWAITING HARDWARE HANDSHAKE' : 'INITIALIZING PHYSICS KERNEL...'}</span>
          <span>{syncDisplay}%</span>
        </div>
      </div>
    </div>
  );

  const renderConnect = () => (
    <div className="fixed inset-0 bg-bg-primary flex flex-col items-center justify-center p-20">
      <h1 className="font-display text-3xl font-bold tracking-[0.3em] mb-12">HARDWARE CONNECTION GATE</h1>
      <div className="grid grid-cols-3 gap-8 w-full max-w-[1000px]">
        {[
          { id: 'live', label: 'LIVE HARDWARE', icon: <Radio size={32} />, desc: 'ROS2 / Serial / WebSocket' },
          { id: 'hil', label: 'HIL SIMULATION', icon: <Chip size={32} />, desc: 'Hardware-in-the-Loop' },
          { id: 'twin', label: 'DIGITAL TWIN ONLY', icon: <Globe size={32} />, desc: 'Software-only validation' },
        ].map(m => (
          <button key={m.id} onClick={() => { connectionMode.current.type = m.id as ConnectionType; hardwareConnected.current = true; setPhase('dashboard'); }} className="p-8 border border-border hover:border-accent-green group transition-all text-left">
            <div className="text-text-tertiary group-hover:text-accent-green mb-6">{m.icon}</div>
            <div className="font-display text-xl tracking-widest mb-2">{m.label}</div>
            <div className="text-xs text-text-secondary">{m.desc}</div>
          </button>
        ))}
      </div>
    </div>
  );

  const renderDashboard = () => (
    <div className="h-screen flex flex-col bg-bg-primary overflow-hidden">
      <header className="h-11 border-b border-border flex items-center justify-between px-4 bg-bg-secondary">
        <div className="flex items-center gap-4">
          <div className={`w-2 h-2 rounded-full ${uiParams.mode === 'NOMINAL' ? 'bg-accent-green' : uiParams.mode === 'CAUTIOUS' ? 'bg-accent-amber' : 'bg-accent-red animate-pulse'}`} />
          <div className="font-display font-bold tracking-[0.2em] text-sm">PHYSICORE v3.0</div>
          <div className="h-4 w-px bg-border" />
          <div className="font-mono text-[10px] text-text-tertiary uppercase">{connectionMode.current.type} LINK: <span className="text-accent-green">ACTIVE</span></div>
        </div>
        <div className="flex items-center gap-6 font-mono text-[11px] text-text-secondary">
          <div>MASS: <span className="text-accent-green">{uiParams.mass.toFixed(3)}kg</span></div>
          <div>FRICTION: <span className="text-accent-green">{uiParams.friction.toFixed(4)}μ</span></div>
          <div>CONFIDENCE: <span className="text-accent-green">{uiParams.confidence.toFixed(1)}%</span></div>
          <div>STABILITY: <span className="text-accent-green">{uiParams.stability.toFixed(1)}%</span></div>
          <div>MODE: <span className={uiParams.mode === 'NOMINAL' ? 'text-accent-green' : 'text-accent-amber'}>{uiParams.mode}</span></div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => setTerminalOpen(true)} className="btn-outline py-1 text-[10px]">INTEGRATION</button>
          <button onClick={() => setDebuggerOpen(true)} className="btn-outline py-1 text-[10px]">DEBUGGER</button>
          <button className="btn-outline py-1 text-[10px] border-accent-green text-accent-green">EXPORT</button>
        </div>
      </header>

      <main className="flex-1 flex overflow-hidden">
        <aside className="w-[250px] border-r border-border bg-bg-secondary p-4 overflow-y-auto custom-scroll space-y-8">
          <section>
            <div className="micro-label border-l-2 border-accent-green pl-3 mb-4">Hardware Priors</div>
            <div className="space-y-4">
              {['mass', 'friction', 'drag', 'efficiency', 'stiffness'].map(k => (
                <div key={k} className="space-y-1.5">
                  <div className="flex justify-between items-center"><span className="text-[10px] text-text-secondary uppercase">{k}</span><span className="font-mono text-accent-green text-[10px]">{(sim.current as any)[k].toFixed(2)}</span></div>
                  <input type="range" min="0.1" max="50" step="0.1" value={(sim.current as any)[k]} onChange={e => (sim.current as any)[k] = parseFloat(e.target.value)} />
                </div>
              ))}
            </div>
          </section>
          <section>
            <div className="micro-label border-l-2 border-accent-green pl-3 mb-4">Operational Mode</div>
            <div className={`h-12 flex items-center justify-center font-display font-black tracking-[0.4em] text-sm border ${uiParams.mode === 'NOMINAL' ? 'bg-[#001A0D] border-accent-green text-accent-green' : uiParams.mode === 'CAUTIOUS' ? 'bg-[#1A1200] border-accent-amber text-accent-amber' : 'bg-accent-red text-black animate-pulse'}`}>{uiParams.mode}</div>
          </section>
        </aside>

        <section className="flex-1 relative bg-bg-primary" onClick={e => { const r = e.currentTarget.getBoundingClientRect(); sim.current.target = { x: e.clientX - r.left - r.width/2, y: e.clientY - r.top - r.height/2 }; }}>
          <canvas ref={canvasRef} className="w-full h-full" />
          <button onClick={() => sim.current.showPhasePortrait = !sim.current.showPhasePortrait} className="absolute top-3 right-3 btn-outline py-1 text-[9px] px-2">◈ PHASE</button>
        </section>

        <aside className="w-[290px] border-l border-border bg-bg-secondary p-4 overflow-y-auto custom-scroll space-y-8">
          <section className="space-y-4">
            <div className="micro-label">Residual Drift Monitor</div>
            <div className="h-[100px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={sim.current.residualHistory.map(v => ({v}))}>
                  <CartesianGrid stroke="#1A1A1A" strokeDasharray="3 3" vertical={false} />
                  <Area type="monotone" dataKey="v" stroke={sim.current.residualL2 > 0.5 ? COLORS.accentAmber : COLORS.accentGreen} fill={sim.current.residualL2 > 0.5 ? COLORS.accentAmber : COLORS.accentGreen} fillOpacity={0.12} isAnimationActive={false} />
                  <ReferenceLine y={0.5} stroke={COLORS.accentAmber} strokeDasharray="4 2" />
                  <YAxis hide domain={[0, 1.5]} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </section>
          <section className="flex flex-col items-center gap-4">
            <div className="micro-label self-start">Policy Confidence Gauge</div>
            <CircularGauge value={uiParams.confidence} color={uiParams.confidence > 70 ? COLORS.accentGreen : uiParams.confidence > 40 ? COLORS.accentAmber : COLORS.accentRed} />
          </section>
          <section>
            <div className="micro-label mb-4">Forensic Record / Failure Log</div>
            <div className="h-[150px] overflow-y-auto custom-scroll space-y-2 pr-2">
              {anomalyLog.length === 0 ? (
                <div className="h-full flex items-center justify-center text-text-tertiary font-mono text-[10px]">NO ANOMALIES RECORDED</div>
              ) : (
                anomalyLog.map((l, i) => (
                  <div key={i} className="flex flex-col gap-1 border-b border-bg-tertiary pb-2">
                    <div className="flex justify-between items-center"><span className="font-mono text-[9px] text-text-tertiary">{l.time}</span><span className={`text-[8px] px-1 border font-bold ${l.type === 'VIOLATION' ? 'border-accent-red text-accent-red' : 'border-accent-amber text-accent-amber'}`}>{l.type}</span></div>
                    <div className="text-[10px] text-text-secondary">{l.desc}</div>
                  </div>
                ))
              )}
            </div>
          </section>
        </aside>
      </main>

      <footer className="h-9 border-t border-border flex items-center justify-between px-4 bg-bg-secondary text-[10px] font-mono text-text-tertiary">
        <div className="flex gap-4"><span>PHYSICORE ENGINE v3.0</span><span className="text-border-active">|</span><span>SENTINEL OS</span></div>
        <div className="flex gap-6">
          <div className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 rounded-full bg-accent-green" /> PHYSICS</div>
          <div className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 rounded-full bg-accent-green" /> ENSEMBLE</div>
          <div className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 rounded-full bg-accent-green" /> SENTINEL</div>
        </div>
        <div className="flex gap-6">
          <div>UPTIME: <span className="text-text-secondary">{getUptime(sessionStart.current)}</span></div>
        </div>
      </footer>

      <AnimatePresence>
        {debuggerOpen && <NeuralDebugger onClose={() => setDebuggerOpen(false)} sim={sim.current} />}
        {terminalOpen && <IntegrationTerminal onClose={() => setTerminalOpen(false)} data={wizardData} sim={sim.current} />}
      </AnimatePresence>
    </div>
  );

  const getUptime = (start: number) => {
    const diff = Math.floor((Date.now() - start) / 1000);
    const h = Math.floor(diff / 3600).toString().padStart(2, '0');
    const m = Math.floor((diff % 3600) / 60).toString().padStart(2, '0');
    const s = (diff % 60).toString().padStart(2, '0');
    return `${h}:${m}:${s}`;
  };

  return (
    <div className="w-full h-full">
      {phase === 'wizard' && renderWizard()}
      {phase === 'sync' && renderSync()}
      {phase === 'connect' && renderConnect()}
      {phase === 'dashboard' && renderDashboard()}
    </div>
  );
}

function NeuralDebugger({ onClose, sim }: { onClose: () => void; sim: any }) {
  const [messages, setMessages] = useState<{ role: 'user' | 'ai'; text: string; time: string }[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = async (txt?: string) => {
    const msg = txt || input; if (!msg.trim()) return;
    setMessages(p => [...p, { role: 'user', text: msg, time: formatTime(new Date()) }]); setInput(''); setLoading(true);
    try {
      const ai = new GoogleGenAI({ apiKey: (process.env as any).GEMINI_API_KEY });
      const response = await ai.models.generateContent({
        model: "gemini-3.1-pro-preview",
        contents: [{ parts: [{ text: msg }] }],
        config: {
          systemInstruction: `You are the PhysiCore Neural Debugger. Speak as a system. Always prefix reasoning with >.
          TELEMETRY: Mode=${sim.mode}, Confidence=${sim.confidence.toFixed(1)}%, Residual=${sim.residualL2.toFixed(4)}.
          If asked to tune, respond with: TUNE: Q=value R=value. Keep under 120 words.`
        }
      });
      setMessages(p => [...p, { role: 'ai', text: response.text || "NO RESPONSE", time: formatTime(new Date()) }]);
    } catch (e) { setMessages(p => [...p, { role: 'ai', text: "> ERROR: NEURAL LINK INTERRUPTED", time: formatTime(new Date()) }]); } finally { setLoading(false); }
  };

  return (
    <motion.div initial={{ x: 400 }} animate={{ x: 0 }} exit={{ x: 400 }} className="fixed top-0 right-0 w-[380px] h-full bg-[#0D0D0D] border-l border-accent-green z-[200] flex flex-col">
      <div className="h-12 border-b border-border flex items-center justify-between px-4">
        <div className="flex flex-col"><span className="font-display text-sm font-bold text-accent-green">NEURAL DEBUGGER</span></div>
        <button onClick={onClose} className="text-text-tertiary hover:text-accent-red font-mono text-sm">✕</button>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-6 custom-scroll">
        {messages.length === 0 && (
          <div className="flex flex-col gap-2 items-center justify-center h-full">
            {["WHY IS IT OSCILLATING?", "TUNE MPC WEIGHTS AUTOMATICALLY", "EXPLAIN RESIDUAL DRIFT"].map(q => (
              <button key={q} onClick={() => handleSend(q)} className="w-full p-2 border border-border font-mono text-[10px] text-text-secondary hover:border-accent-green hover:text-accent-green transition-all">{q}</button>
            ))}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
            {m.role === 'ai' && <span className="text-[9px] font-mono text-accent-green mb-1">{"> PHYSICORE ENGINE:"}</span>}
            <div className={`text-[11px] leading-relaxed ${m.role === 'user' ? 'text-text-primary' : 'text-accent-green font-mono'}`}>{m.text}</div>
            <span className="text-[8px] font-mono text-text-tertiary mt-1">{m.time}</span>
          </div>
        ))}
        {loading && <div className="text-accent-green font-mono text-[10px] animate-pulse">{"> PHYSICORE ENGINE: ANALYZING TELEMETRY..."}</div>}
      </div>
      <div className="h-14 border-t border-border p-3 flex gap-2 bg-bg-secondary">
        <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSend()} className="flex-1 bg-transparent border-b border-border font-mono text-xs text-accent-green outline-none" placeholder="> QUERY NEURAL DEBUGGER..." />
        <button onClick={() => handleSend()} className="font-display text-[10px] font-bold text-accent-green px-2 border border-accent-green">⏎ EXECUTE</button>
      </div>
    </motion.div>
  );
}

function IntegrationTerminal({ onClose, data, sim }: { onClose: () => void; data: any; sim: any }) {
  const [platform, setPlatform] = useState('ROS2');
  const code = `import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String

class PhysiCoreBridge(Node):
    def __init__(self):
        super().__init__('physicore_bridge')
        self.config = {
            'mass': ${sim.mass.toFixed(3)},
            'friction': ${sim.friction.toFixed(4)},
            'mode': '${sim.mode}'
        }
        self.pub = self.create_publisher(String, '/physicore/state', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = f"MODE: {self.config['mode']} | MASS: {self.config['mass']}"
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PhysiCoreBridge()
    rclpy.spin(node)
    rclpy.shutdown()`;

  return (
    <motion.div initial={{ x: -520 }} animate={{ x: 0 }} exit={{ x: -520 }} className="fixed top-0 left-0 w-[520px] h-full bg-[#0D0D0D] border-r border-accent-blue z-[200] flex flex-col">
      <div className="h-12 border-b border-border flex items-center justify-between px-4">
        <span className="font-display text-sm font-bold text-accent-blue">INTEGRATION ENGINEER TERMINAL</span>
        <button onClick={onClose} className="text-text-tertiary hover:text-accent-red font-mono text-sm">✕</button>
      </div>
      <div className="flex gap-2 p-4 border-b border-border">
        {['ROS2', 'ARDUPILOT', 'PX4', 'MATLAB', 'CUSTOM'].map(p => (
          <button key={p} onClick={() => setPlatform(p)} className={`px-3 py-1 text-[10px] border ${platform === p ? 'border-accent-blue text-accent-blue bg-accent-blue/10' : 'border-border text-text-tertiary'}`}>{p}</button>
        ))}
      </div>
      <div className="flex-1 p-4 overflow-y-auto custom-scroll bg-black">
        <pre className="font-mono text-[11px] leading-relaxed text-text-primary whitespace-pre-wrap">
          {code}
        </pre>
      </div>
      <div className="p-4 border-t border-border bg-bg-secondary flex justify-between">
        <button className="btn-outline py-1 text-[10px] border-accent-blue text-accent-blue">⬇ DOWNLOAD {platform}_BRIDGE.PY</button>
        <button className="btn-outline py-1 text-[10px] border-accent-blue text-accent-blue flex items-center gap-2"><Copy size={12} /> COPY</button>
      </div>
    </motion.div>
  );
}
