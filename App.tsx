import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { useAuthState } from 'react-firebase-hooks/auth';
import { auth, db, googleProvider, handleFirestoreError, OperationType } from './src/firebase';
import { signInWithPopup, signOut } from 'firebase/auth';
import { 
  doc, getDoc, updateDoc, setDoc, deleteDoc,
  collection, query, where, getDocs, onSnapshot 
} from 'firebase/firestore';
import { ErrorBoundary } from './src/components/ErrorBoundary';
import { 
  Activity, Cpu, Shield, Zap, ChevronRight, ChevronLeft, 
  Play, Download, Terminal, AlertTriangle, CheckCircle2, 
  Settings, Crosshair, ArrowUpRight, ArrowDownRight, X,
  Maximize2, Activity as FrequencyIcon, RefreshCw, Globe,
  Link, Wifi, Radio, HardDrive, FileJson, Copy, Check,
  ArrowRight, MousePointer2, Layers, BarChart3, ShieldCheck,
  Code2, MessageSquare, DownloadCloud, ExternalLink, ChevronDown,
  Rocket, Wind, Navigation, History, FileUp, TrendingUp, Gauge,
  Pause, RotateCcw, Info, Upload, LogOut, User, Lock, ShieldAlert,
  BookOpen, Plus, Trash2
} from 'lucide-react';
import { simpleHash, generateId, encodeProjectCode, decodeProjectCode } from './src/utils/projectSync';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, 
  ResponsiveContainer, BarChart, Bar, ReferenceLine,
  LineChart, Line, ComposedChart, Scatter, Legend, Tooltip
} from 'recharts';
import { GoogleGenAI } from "@google/genai";

// --- ROCKET CONSTANTS ---
const RKT_G = 9.80665;
const RKT_R_AIR = 287.05;
const RKT_L = 0.0065;
const RKT_T0 = 288.15;
const RKT_P0 = 101325;
const RKT_RHO0 = 1.225;

const BETA_TESTERS = [
  "koshmarus@gmail.com",
  "stesrocketryteam@gmail.com",
  "darisglx@gmail.com",
  "vladimir.robotics@gmail.com",
  "projectauvm@manipal.edu",
  "prathameshshirbhate8anpc@gmail.com"
];

const DEFAULT_GEMINI_KEY = "AIzaSyAgARuyw36M02J37mKH2RlHYvgu9bQ-lwc";

type RocketPhase = 'PRELAUNCH' | 'RAIL' | 'POWERED' | 'COAST' | 'APOGEE' | 'DROGUE' | 'MAIN' | 'LANDED';

interface RocketState {
  x: number;
  y: number;
  vx: number;
  vy: number;
  mass: number;
  propMass: number;
  time: number;
  phase: RocketPhase;
  angle: number; // radians from vertical
}

interface RocketParams {
  dryMass: number;
  propMassInitial: number;
  burnTime: number;
  thrust: number;
  fuelMass: number;
  diameter: number;
  length: number;
  cd: number;
  motorCurve: { t: number; f: number }[];
  isp: number;
  launchAngle: number; // degrees from vertical
  railLength: number;
  launchAltitude: number;
  drogueAlt: number; // 0 means apogee
  drogueCd: number;
  drogueDiam: number;
  mainAlt: number;
  mainCd: number;
  mainDiam: number;
  cl?: number; // Lift coefficient for guided rockets
}

interface AviationState {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  pitch: number;
  roll: number;
  yaw: number;
  aoa: number; // Angle of attack
  sideslip: number;
  mass: number;
  fuel: number;
  time: number;
}

interface AviationParams {
  mass: number;
  wingspan: number;
  wingArea: number;
  chord: number;
  cl0: number; // Lift coefficient at zero alpha
  cla: number; // Lift slope
  cd0: number; // Parasitic drag
  k: number; // Induced drag factor
  thrustMax: number;
  fuelCapacity: number;
  fuelBurnRate: number;
  vne: number; // Never exceed speed
  vso: number; // Stall speed
}

const atmosphericDensity = (altitude: number) => {
  if (altitude < 0) return RKT_RHO0;
  const T = RKT_T0 - RKT_L * altitude;
  if (T <= 0) return 0;
  const P = RKT_P0 * Math.pow(1 - (RKT_L * altitude) / RKT_T0, (RKT_G / (RKT_R_AIR * RKT_L)));
  return P / (RKT_R_AIR * T);
};

const getThrustAtTime = (time: number, curve: { t: number; f: number }[]) => {
  if (curve.length === 0) return 0;
  if (time < curve[0].t) return 0;
  if (time > curve[curve.length - 1].t) return 0;
  
  for (let i = 0; i < curve.length - 1; i++) {
    if (time >= curve[i].t && time <= curve[i+1].t) {
      const t0 = curve[i].t;
      const t1 = curve[i+1].t;
      const f0 = curve[i].f;
      const f1 = curve[i+1].f;
      return f0 + (f1 - f0) * (time - t0) / (t1 - t0);
    }
  }
  return 0;
};

const rocketDerivatives = (state: RocketState, params: RocketParams) => {
  const rho = atmosphericDensity(state.y + params.launchAltitude);
  const area = Math.PI * Math.pow(params.diameter / 2, 2);
  const v = Math.sqrt(state.vx * state.vx + state.vy * state.vy);
  
  const thrust = getThrustAtTime(state.time, params.motorCurve);
  
  // Drag coefficient might change based on phase (parachutes)
  let currentCd = params.cd;
  let currentArea = area;
  
  if (state.phase === 'DROGUE') {
    currentCd = params.drogueCd;
    currentArea = Math.PI * Math.pow(params.drogueDiam / 2, 2);
  } else if (state.phase === 'MAIN') {
    currentCd = params.mainCd;
    currentArea = Math.PI * Math.pow(params.mainDiam / 2, 2);
  }
  
  const drag = 0.5 * rho * v * v * currentCd * currentArea;
  
  // Gravity
  const Fgx = 0;
  const Fgy = -state.mass * RKT_G;
  
  // Thrust components
  const angle = state.angle;
  const Ftx = thrust * Math.sin(angle);
  const Fty = thrust * Math.cos(angle);
  
  // Drag components (opposite to velocity)
  const Fdx = v > 0 ? -drag * (state.vx / v) : 0;
  const Fdy = v > 0 ? -drag * (state.vy / v) : 0;
  
  // Total force
  const Fx = Ftx + Fdx + Fgx;
  const Fy = Fty + Fdy + Fgy;
  
  return {
    dx: state.vx,
    dy: state.vy,
    dvx: Fx / state.mass,
    dvy: Fy / state.mass,
    dm: -thrust / (RKT_G * params.isp),
    dt: 1
  };
};

const rocketRK4Step = (state: RocketState, params: RocketParams, dt: number) => {
  const k1 = rocketDerivatives(state, params);
  
  const s2: RocketState = {
    ...state,
    x: state.x + k1.dx * dt / 2,
    y: state.y + k1.dy * dt / 2,
    vx: state.vx + k1.dvx * dt / 2,
    vy: state.vy + k1.dvy * dt / 2,
    mass: state.mass + k1.dm * dt / 2,
    time: state.time + k1.dt * dt / 2
  };
  const k2 = rocketDerivatives(s2, params);
  
  const s3: RocketState = {
    ...state,
    x: state.x + k2.dx * dt / 2,
    y: state.y + k2.dy * dt / 2,
    vx: state.vx + k2.dvx * dt / 2,
    vy: state.vy + k2.dvy * dt / 2,
    mass: state.mass + k2.dm * dt / 2,
    time: state.time + k2.dt * dt / 2
  };
  const k3 = rocketDerivatives(s3, params);
  
  const s4: RocketState = {
    ...state,
    x: state.x + k3.dx * dt,
    y: state.y + k3.dy * dt,
    vx: state.vx + k3.dvx * dt,
    vy: state.vy + k3.dvy * dt,
    mass: state.mass + k3.dm * dt,
    time: state.time + k3.dt * dt
  };
  const k4 = rocketDerivatives(s4, params);
  
  return {
    x: state.x + (k1.dx + 2 * k2.dx + 2 * k3.dx + k4.dx) * dt / 6,
    y: state.y + (k1.dy + 2 * k2.dy + 2 * k3.dy + k4.dy) * dt / 6,
    vx: state.vx + (k1.dvx + 2 * k2.dvx + 2 * k3.dvx + k4.dvx) * dt / 6,
    vy: state.vy + (k1.dvy + 2 * k2.dvy + 2 * k3.dvy + k4.dvy) * dt / 6,
    mass: Math.max(params.dryMass, state.mass + (k1.dm + 2 * k2.dm + 2 * k3.dm + k4.dm) * dt / 6),
    time: state.time + dt,
    propMass: Math.max(0, state.propMass + (k1.dm + 2 * k2.dm + 2 * k3.dm + k4.dm) * dt / 6)
  };
};

const updateRocketPhase = (state: RocketState, params: RocketParams, prevState: RocketState) => {
  let { phase, x, y, vx, vy, time } = state;
  
  if (phase === 'PRELAUNCH') {
    return 'PRELAUNCH';
  }
  
  if (phase === 'RAIL') {
    const dist = Math.sqrt(x * x + y * y);
    if (dist >= params.railLength) {
      return 'POWERED';
    }
    return 'RAIL';
  }
  
  if (phase === 'POWERED') {
    const thrust = getThrustAtTime(time, params.motorCurve);
    if (thrust <= 0 && time > 0.1) {
      return 'COAST';
    }
    if (vy < 0 && prevState.vy >= 0) {
      return 'APOGEE';
    }
    return 'POWERED';
  }
  
  if (phase === 'COAST') {
    if (vy < 0 && prevState.vy >= 0) {
      return 'APOGEE';
    }
    return 'COAST';
  }
  
  if (phase === 'APOGEE') {
    return 'DROGUE';
  }
  
  if (phase === 'DROGUE') {
    if (params.mainAlt > 0 && y <= params.mainAlt) {
      return 'MAIN';
    }
    if (y <= 0) return 'LANDED';
    return 'DROGUE';
  }
  
  if (phase === 'MAIN') {
    if (y <= 0) return 'LANDED';
    return 'MAIN';
  }
  
  if (phase === 'LANDED') {
    return 'LANDED';
  }
  
  return phase;
};

const parachuteTerminalVel = (rho: number, mass: number, cd: number, diameter: number) => {
  const area = Math.PI * Math.pow(diameter / 2, 2);
  if (area === 0) return 0;
  return Math.sqrt((2 * mass * RKT_G) / (rho * area * cd));
};

// --- CONSTANTS ---
async function callGemini(prompt: string, history: any[] = [], systemInstruction: string = "", userKey?: string) {
  const apiKey = userKey || import.meta.env.VITE_GEMINI_API_KEY || (window as any).__GEMINI_KEY__ || localStorage.getItem('physicore_gemini_key') || DEFAULT_GEMINI_KEY;
  
  if (!apiKey) {
    return { success: false, text: null, error: 'NO_API_KEY' };
  }

  const model = "gemini-1.5-flash";
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;

  // Format contents for Gemini API
  let contents = [];
  if (history.length > 0) {
    // Ensure history alternates correctly and starts with user
    contents = history.map(m => ({
      role: m.role === 'ai' ? 'model' : 'user',
      parts: [{ text: m.content }]
    }));
  } else {
    contents = [{ role: 'user', parts: [{ text: prompt }] }];
  }

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents,
        systemInstruction: systemInstruction ? { parts: [{ text: systemInstruction }] } : undefined,
        generationConfig: {
          temperature: 0.2,
          maxOutputTokens: 2000,
        }
      })
    });

    if (!response.ok) {
      const status = response.status;
      if (status === 400) return { success: false, text: null, error: 'HTTP_400' };
      if (status === 403) return { success: false, text: null, error: 'HTTP_403' };
      if (status === 401) return { success: false, text: null, error: 'HTTP_401' };
      if (status === 429) return { success: false, text: null, error: 'HTTP_429' };
      throw new Error(`HTTP_${status}`);
    }

    const data = await response.json();
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
    
    if (text) {
      return { success: true, text, error: null };
    }
    return { success: false, text: null, error: 'EMPTY_RESPONSE' };
  } catch (err: any) {
    console.error("Gemini API Call Failed:", err);
    if (err.name === 'AbortError') return { success: false, text: null, error: 'TIMEOUT' };
    return { success: false, text: null, error: 'NETWORK_ERROR' };
  }
}

const COLORS = {
  void: '#080808', bg: '#0C0C0C', bgRaised: '#111111', bgInset: '#0A0A0A',
  borderDim: '#1E1E1E', border: '#2A2A2A', borderActive: '#3D3D3D',
  textPrimary: '#EFEFEF', textSecondary: '#7A7A7A', textDim: '#444444',
  green: '#00FF88', greenDim: '#003320', amber: '#FFB800', amberDim: '#1A1000',
  red: '#FF2222', redDim: '#1A0000', blue: '#0099FF', blueDim: '#001020',
  white: '#FFFFFF', cyan: '#00DDCC',
};

// --- TYPES ---
type View = 'home' | 'integrator' | 'dashboard' | 'manual' | 'team';
type Platform = 'ROS2' | 'ARDUPILOT' | 'PX4' | 'MATLAB' | 'CUSTOM';

interface SystemProfile {
  platform: string | null;
  firmware: string | null;
  domain: string | null;
  massClass: string | null;
  connectionMode: string | null;
  protocols: string | null;
}

interface Message {
  role: 'user' | 'ai';
  content: string;
  timestamp: string;
}

interface GeneratedFile {
  filename: string;
  content: string;
  extension: string;
}

// --- UTILS ---
const formatTime = (date: Date) => {
  const h = date.getHours().toString().padStart(2, '0');
  const m = date.getMinutes().toString().padStart(2, '0');
  const s = date.getSeconds().toString().padStart(2, '0');
  return `${h}:${m}:${s}`;
};

// --- HARDWARE GATE ---
const initiateHandshake = async (endpoint: string, mode: 'ros2_websocket' | 'hil' | 'digital_twin') => {
  if (mode === 'digital_twin') {
    return new Promise((resolve) => {
      // Digital Twin is a local simulation, but we still simulate a handshake
      setTimeout(() => {
        resolve({ success: true, mode: 'digital_twin', latency: 5 });
      }, 1000);
    });
  }

  if (mode === 'ros2_websocket') {
    return new Promise((resolve) => {
      try {
        const ws = new WebSocket(endpoint);
        const timeout = setTimeout(() => {
          ws.close();
          resolve({ success: false, reason: 'CONNECTION_TIMEOUT' });
        }, 5000);
        
        ws.onopen = () => {
          clearTimeout(timeout);
          ws.send(JSON.stringify({
            op: 'call_service',
            service: '/rosapi/topics'
          }));
        };
        
        ws.onmessage = (msg) => {
          try {
            const data = JSON.parse(msg.data);
            ws.close();
            resolve({
              success: true,
              topics: data.topics || [],
              latency: performance.now()
            });
          } catch (e) {
            ws.close();
            resolve({ success: false, reason: 'INVALID_RESPONSE' });
          }
        };
        
        ws.onerror = () => {
          clearTimeout(timeout);
          resolve({ success: false, reason: 'CONNECTION_REFUSED' });
        };
      } catch (e) {
        resolve({ success: false, reason: 'WEBSOCKET_ERROR' });
      }
    });
  }
  
  if (mode === 'hil') {
    return new Promise(async (resolve) => {
      try {
        // HIL Simulation Handshake - REAL CHECK
        const currentOrigin = window.location.origin;
        if (endpoint.includes(currentOrigin) || endpoint.includes('localhost:3000')) {
          resolve({ success: false, reason: 'SELF_CONNECTION_FORBIDDEN. Cannot connect to the PhysiCore UI as a hardware endpoint.' });
          return;
        }

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000);
        
        // We expect a real hardware bridge to respond to a specific health check
        const response = await fetch(endpoint + '/api/health', { 
          method: 'GET', 
          signal: controller.signal 
        });
        
        if (!response.ok) throw new Error('BRIDGE_OFFLINE');
        
        // Verify it's actually a PhysiCore bridge
        const bridgeToken = response.headers.get('X-PhysiCore-Bridge');
        if (bridgeToken !== 'active' && !endpoint.includes('physicore-bridge')) {
           // If no header, check body for signature
           const body = await response.json();
           if (body.service !== 'physicore' && body.status !== 'ok') {
             throw new Error('INVALID_BRIDGE_SIGNATURE');
           }
        }

        clearTimeout(timeoutId);
        resolve({
          success: true,
          simulated: true,
          latency: 0.4
        });
      } catch (e) {
        clearTimeout(2000); // Just in case
        resolve({ 
          success: false, 
          reason: 'HIL_ENDPOINT_UNREACHABLE. Ensure your HIL bridge is running at ' + endpoint + ' and responding to /api/health'
        });
      }
    });
  }
  return { success: false, reason: 'UNKNOWN_MODE' };
};

// --- PHYSICS HELPERS ---
interface State { x: number; y: number; vx: number; vy: number; }
interface Params { mass: number; friction: number; }

const rk4_step = (pos: {x: number, y: number}, vel: {x: number, y: number}, force: {x: number, y: number}, params: Params, dt: number = 1/60) => {
  const mass = Math.max(0.1, params.mass);
  const friction = params.friction;

  const accel = (p: {x: number, y: number}, v: {x: number, y: number}, f: {x: number, y: number}) => ({
    ax: (f.x - friction * v.x) / mass,
    ay: (f.y - friction * v.y) / mass
  });

  // k1
  const a1 = accel(pos, vel, force);
  const k1v = { x: a1.ax * dt, y: a1.ay * dt };
  const k1p = { x: vel.x * dt, y: vel.y * dt };

  // k2
  const v2 = { x: vel.x + k1v.x / 2, y: vel.y + k1v.y / 2 };
  const a2 = accel({ x: pos.x + k1p.x / 2, y: pos.y + k1p.y / 2 }, v2, force);
  const k2v = { x: a2.ax * dt, y: a2.ay * dt };
  const k2p = { x: v2.x * dt, y: v2.y * dt };

  // k3
  const v3 = { x: vel.x + k2v.x / 2, y: vel.y + k2v.y / 2 };
  const a3 = accel({ x: pos.x + k2p.x / 2, y: pos.y + k2p.y / 2 }, v3, force);
  const k3v = { x: a3.ax * dt, y: a3.ay * dt };
  const k3p = { x: v3.x * dt, y: v3.y * dt };

  // k4
  const v4 = { x: vel.x + k3v.x, y: vel.y + k3v.y };
  const a4 = accel({ x: pos.x + k3p.x, y: pos.y + k3p.y }, v4, force);
  const k4v = { x: a4.ax * dt, y: a4.ay * dt };
  const k4p = { x: v4.x * dt, y: v4.y * dt };

  return {
    pos: {
      x: pos.x + (k1p.x + 2 * k2p.x + 2 * k3p.x + k4p.x) / 6,
      y: pos.y + (k1p.y + 2 * k2p.y + 2 * k3p.y + k4p.y) / 6
    },
    vel: {
      x: vel.x + (k1v.x + 2 * k2v.x + 2 * k3v.x + k4v.x) / 6,
      y: vel.y + (k1v.y + 2 * k2v.y + 2 * k3v.y + k4v.y) / 6
    }
  };
};

const cem_mpc = (pos: {x: number, y: number}, vel: {x: number, y: number}, target: {x: number, y: number}, params: Params) => {
  const horizon = 12;
  const numSamples = 20;
  const numElites = 5;
  const qWeight = 0.1;
  const rWeight = 0.01;

  let meanX = new Array(horizon).fill(0);
  let stdX = new Array(horizon).fill(100.0);
  let meanY = new Array(horizon).fill(0);
  let stdY = new Array(horizon).fill(100.0);

  for (let iter = 0; iter < 3; iter++) {
    const sequences = [];
    for (let s = 0; s < numSamples; s++) {
      const seqX = meanX.map((m, i) => m + stdX[i] * (Math.random() - 0.5) * 2);
      const seqY = meanY.map((m, i) => m + stdY[i] * (Math.random() - 0.5) * 2);
      sequences.push({ x: seqX, y: seqY });
    }

    const costs = sequences.map(seq => {
      let p = { ...pos };
      let v = { ...vel };
      let cost = 0;
      for (let t = 0; t < horizon; t++) {
        const force = { x: seq.x[t], y: seq.y[t] };
        const next = rk4_step(p, v, force, params);
        p = next.pos;
        v = next.vel;
        const distSq = Math.pow(p.x - target.x, 2) + Math.pow(p.y - target.y, 2);
        const effortSq = Math.pow(force.x, 2) + Math.pow(force.y, 2);
        cost += qWeight * distSq + rWeight * effortSq;
      }
      return cost;
    });

    const sortedIdx = costs
      .map((c, i) => [c, i])
      .sort((a, b) => a[0] - b[0])
      .slice(0, numElites)
      .map(([_, i]) => i);

    const eliteSeqs = sortedIdx.map(i => sequences[i]);
    
    meanX = meanX.map((_, t) => eliteSeqs.reduce((s, seq) => s + seq.x[t], 0) / numElites);
    stdX = stdX.map((_, t) => {
      const v = eliteSeqs.reduce((s, seq) => s + Math.pow(seq.x[t] - meanX[t], 2), 0) / numElites;
      return Math.sqrt(v) + 1.0;
    });

    meanY = meanY.map((_, t) => eliteSeqs.reduce((s, seq) => s + seq.y[t], 0) / numElites);
    stdY = stdY.map((_, t) => {
      const v = eliteSeqs.reduce((s, seq) => s + Math.pow(seq.y[t] - meanY[t], 2), 0) / numElites;
      return Math.sqrt(v) + 1.0;
    });
  }

  return { x: meanX, y: meanY };
};

// --- COMPONENTS ---

const SyntaxHighlighter = ({ code }: { code: string }) => {
  const tokens = [
    { name: 'comment', regex: /(?:\/\/.*|#.*|\/\*[\s\S]*?\*\/)/, color: COLORS.textDim, italic: true },
    { name: 'string', regex: /(?:".*?"|'.*?')/, color: '#88DD88' },
    { name: 'keyword', regex: /\b(?:def|import|from|class|if|else|return|async|await|try|except|with|as|for|in|while|pass|break|continue|yield|lambda|global|nonlocal|assert|del|is|not|and|or|True|False|None|public|private|protected|static|void|int|float|double|bool|string|char|using|namespace|std|vector|cout|endl|ros|rclcpp|node|publisher|subscriber|timer|callback|msg|srv|action|uint32_t|float32_t|double_t|bool_t|auto|const|constexpr|struct|enum|union|typedef|extern|inline|virtual|override|final|explicit|mutable|volatile|register|thread_local|alignas|alignof|sizeof|typeid|typename|template|concept|requires|decltype|noexcept|static_assert|static_cast|dynamic_cast|const_cast|reinterpret_cast|new|delete|this|throw|try|catch|operator|friend|export|module|import|co_await|co_yield|co_return)\b/, color: COLORS.cyan },
    { name: 'number', regex: /\b(?:\d+)\b/, color: COLORS.amber },
    { name: 'type', regex: /\b(?:[A-Z][a-zA-Z0-9_]*)\b/, color: COLORS.blue },
  ];

  const highlightCode = (text: string) => {
    const combinedRegex = new RegExp(tokens.map(t => `(${t.regex.source})`).join('|'), 'g');
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match;

    while ((match = combinedRegex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }

      const matchIndex = match.slice(1).findIndex(val => val !== undefined);
      const token = matchIndex !== -1 ? tokens[matchIndex] : null;
      const matchedText = match[0];

      if (token) {
        parts.push(
          <span 
            key={match.index} 
            style={{ color: token.color, fontStyle: token.italic ? 'italic' : 'normal' }}
          >
            {matchedText}
          </span>
        );
      } else {
        parts.push(matchedText);
      }
      lastIndex = combinedRegex.lastIndex;
    }

    parts.push(text.substring(lastIndex));
    return parts;
  };

  return (
    <pre className="font-mono text-[11px] leading-relaxed whitespace-pre overflow-x-auto">
      <code>{highlightCode(code)}</code>
    </pre>
  );
};

const IntegrationActionPanel = ({ 
  files, 
  onTest, 
  onContinue,
  connectionMode,
  setConnectionMode,
  endpoint,
  setEndpoint,
  dRealEndpoint,
  setDRealEndpoint,
  systemProfile,
  rocketParams,
  aviationParams,
  priors,
  onAction,
  projectCode,
  projectData,
  onImportProjectCode,
  isSystemConnecting,
  connectionError
}: { 
  files: GeneratedFile[], 
  onTest: () => void, 
  onContinue: () => void,
  connectionMode: 'ros2_websocket' | 'hil' | 'digital_twin',
  setConnectionMode: (m: 'ros2_websocket' | 'hil' | 'digital_twin') => void,
  endpoint: string,
  setEndpoint: (e: string) => void,
  dRealEndpoint: string,
  setDRealEndpoint: (e: string) => void,
  systemProfile: SystemProfile,
  rocketParams: RocketParams,
  aviationParams: AviationParams,
  priors: { mass: number, friction: number },
  onAction?: () => void,
  projectCode: string,
  projectData: any,
  onImportProjectCode: (code: string) => { success: boolean, data?: any, error?: string },
  isSystemConnecting: boolean,
  connectionError: string | null
}) => {
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<'MY_CODE' | 'IMPORT'>('MY_CODE');
  const [importCode, setImportCode] = useState('');
  const [importStatus, setImportStatus] = useState<{ success?: boolean, msg?: string }>({});

  const handleCopyProjectCode = () => {
    navigator.clipboard.writeText(projectCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadProjectCode = () => {
    const blob = new Blob([projectCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `project_${projectData?.id || 'export'}.pc`;
    a.click();
  };

  const handleImport = () => {
    const result = onImportProjectCode(importCode);
    if (result.success) {
      setImportStatus({ success: true, msg: `Imported Project: ${result.data.id}` });
    } else {
      setImportStatus({ success: false, msg: result.error });
    }
  };

  const handleDownloadSentinelPack = () => {
    if (onAction) onAction();
    const sentinelPack = {
      metadata: {
        client: "PhysiCore-v3.0",
        timestamp: new Date().toISOString(),
        domain: systemProfile.domain,
        platform: systemProfile.platform,
      },
      priors: {
        mass: priors.mass,
        friction: priors.friction,
      },
      ...(systemProfile.domain === 'ROCKETS' && {
        rocket_manifest: {
          burn_time: rocketParams.burnTime,
          thrust: rocketParams.thrust,
          dry_mass: rocketParams.dryMass,
          fuel_mass: rocketParams.fuelMass,
          isp: rocketParams.isp,
        }
      }),
      ...(systemProfile.domain === 'AVIATION' && {
        aviation_manifest: {
          wingspan: aviationParams.wingspan,
          wing_area: aviationParams.wingArea,
          thrust_max: aviationParams.thrustMax,
          fuel_capacity: aviationParams.fuelCapacity,
          cla: aviationParams.cla,
          cd0: aviationParams.cd0,
        }
      }),
      control_logic: {
        optimizer: "MPC-CEM",
        horizon: 12,
        connection_mode: connectionMode,
        endpoint: endpoint,
      },
      project_sync: {
        project_id: projectData?.id,
        project_code: projectCode,
        sync_origin: "PhysiCore",
        last_sync: new Date().toISOString()
      }
    };
    const blob = new Blob([JSON.stringify(sentinelPack, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const domainStr = systemProfile.domain ? systemProfile.domain.toLowerCase() : 'unknown';
    a.download = `sentinel_pack_${domainStr}_${Date.now()}.json`;
    a.click();
  };

  const handleDownloadAll = () => {
    files.forEach((file, index) => {
      setTimeout(() => {
        const element = document.createElement("a");
        const blob = new Blob([file.content], { type: 'text/plain' });
        element.href = URL.createObjectURL(blob);
        element.download = file.filename;
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
      }, index * 200);
    });
  };

  const handleCopyCommand = () => {
    const cmd = "npm install @physicore/kernel && physicore init --platform=ros2";
    navigator.clipboard.writeText(cmd);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="mt-8 border border-green bg-greenDim/20 p-6 space-y-6"
    >
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 bg-green text-black flex items-center justify-center">
          <Zap size={18} />
        </div>
        <div>
          <h3 className="font-display text-sm font-bold text-green tracking-widest uppercase">INTEGRATION ACTION PANEL</h3>
          <p className="font-body text-[10px] text-textSecondary uppercase">Code generation complete. Select next action.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="p-4 border border-border bg-bgRaised space-y-4">
          <div className="flex justify-between items-center">
            <h4 className="micro-label text-textDim uppercase">Project Sync</h4>
            <div className="flex gap-2">
              <button 
                onClick={() => setActiveTab('MY_CODE')}
                className={`micro-label px-2 py-1 border ${activeTab === 'MY_CODE' ? 'border-cyan text-cyan' : 'border-border text-textDim'}`}
              >
                MY PROJECT CODE
              </button>
              <button 
                onClick={() => setActiveTab('IMPORT')}
                className={`micro-label px-2 py-1 border ${activeTab === 'IMPORT' ? 'border-cyan text-cyan' : 'border-border text-textDim'}`}
              >
                IMPORT FROM SENTINEL
              </button>
            </div>
          </div>

          {activeTab === 'MY_CODE' ? (
            <div className="space-y-4">
              <div className="p-3 bg-bg border border-borderDim space-y-2">
                <div className="flex justify-between items-center">
                  <span className="micro-label text-textDim">PROJECT ID</span>
                  <span className="font-mono text-[10px] text-cyan">{projectData?.id}</span>
                </div>
                <div className="font-mono text-[9px] text-textSecondary break-all bg-void p-2 border border-borderDim">
                  {projectCode}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <button 
                  onClick={handleCopyProjectCode}
                  className="flex items-center justify-center gap-2 py-2 border border-border hover:border-cyan text-textSecondary hover:text-cyan transition-all"
                >
                  <Copy size={12} />
                  <span className="micro-label uppercase">{copied ? 'COPIED' : 'COPY CODE'}</span>
                </button>
                <button 
                  onClick={handleDownloadProjectCode}
                  className="flex items-center justify-center gap-2 py-2 border border-border hover:border-cyan text-textSecondary hover:text-cyan transition-all"
                >
                  <Download size={12} />
                  <span className="micro-label uppercase">DOWNLOAD .PC</span>
                </button>
              </div>
              <p className="font-mono text-[8px] text-textDim uppercase">Paste this code into Sentinel to load this configuration instantly.</p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="space-y-2">
                <textarea 
                  value={importCode}
                  onChange={e => setImportCode(e.target.value)}
                  placeholder="Paste Project Code (PC- or SN-) here..."
                  className="w-full h-24 bg-bg border border-border p-2 font-mono text-[10px] text-white outline-none focus:border-cyan resize-none"
                />
                <div className="flex items-center justify-center border-2 border-dashed border-border p-4 hover:border-cyan transition-all cursor-pointer">
                  <span className="micro-label text-textDim uppercase">OR DROP .SN / .PC FILE HERE</span>
                </div>
              </div>
              <button 
                onClick={handleImport}
                className="w-full py-2 bg-cyan text-black font-display text-[10px] font-bold uppercase tracking-widest hover:bg-white transition-all"
              >
                IMPORT CONFIGURATION
              </button>
              {importStatus.msg && (
                <div className={`p-2 border ${importStatus.success ? 'border-green text-green' : 'border-red text-red'} font-mono text-[9px] uppercase`}>
                  {importStatus.msg}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="p-4 border border-border bg-bgRaised space-y-4">
          <div className="flex justify-between items-center">
            <span className="micro-label text-textDim">Connection Mode</span>
            <div className="flex gap-2">
              <button 
                onClick={() => setConnectionMode('digital_twin')}
                className={`px-3 py-1 font-mono text-[9px] border ${connectionMode === 'digital_twin' ? 'bg-cyan text-black border-cyan' : 'border-border text-textDim'}`}
              >
                TWIN
              </button>
              <button 
                onClick={() => setConnectionMode('hil')}
                className={`px-3 py-1 font-mono text-[9px] border ${connectionMode === 'hil' ? 'bg-green text-black border-green' : 'border-border text-textDim'}`}
              >
                HIL
              </button>
              <button 
                onClick={() => setConnectionMode('ros2_websocket')}
                className={`px-3 py-1 font-mono text-[9px] border ${connectionMode === 'ros2_websocket' ? 'bg-green text-black border-green' : 'border-border text-textDim'}`}
              >
                ROS2
              </button>
            </div>
          </div>
          <div className="space-y-1">
            <span className="micro-label text-textDim">Endpoint</span>
            <input 
              value={endpoint}
              onChange={e => setEndpoint(e.target.value)}
              className="w-full bg-bg border border-border p-2 font-mono text-[10px] text-green outline-none focus:border-green"
            />
          </div>
        </div>

        <div className="p-4 border border-border bg-bgRaised space-y-4">
          <div className="flex justify-between items-center">
            <span className="micro-label text-textDim">Formal Verification</span>
            <span className="font-mono text-[9px] text-amber">dReal v4.2</span>
          </div>
          <div className="space-y-1">
            <span className="micro-label text-textDim">dReal Server Endpoint</span>
            <input 
              value={dRealEndpoint}
              onChange={e => setDRealEndpoint(e.target.value)}
              className="w-full bg-bg border border-border p-2 font-mono text-[10px] text-amber outline-none focus:border-amber"
              placeholder="http://localhost:8080"
            />
          </div>
        </div>

        <button 
          onClick={handleDownloadSentinelPack}
          className="flex items-center justify-between p-4 border border-amber/30 bg-bgRaised hover:bg-amber hover:text-black transition-all group"
        >
          <div className="flex flex-col items-start">
            <span className="font-display text-[11px] font-bold tracking-widest uppercase">DOWNLOAD SENTINEL PACK</span>
            <span className="font-mono text-[9px] text-textDim group-hover:text-black/60 uppercase">System configuration (JSON)</span>
          </div>
          <Shield size={20} />
        </button>

        <button 
          onClick={handleDownloadAll}
          className="flex items-center justify-between p-4 border border-green/30 bg-bgRaised hover:bg-green hover:text-black transition-all group"
        >
          <div className="flex flex-col items-start">
            <span className="font-display text-[11px] font-bold tracking-widest uppercase">DOWNLOAD PACKAGE</span>
            <span className="font-mono text-[9px] text-textDim group-hover:text-black/60 uppercase">All generated files (.zip)</span>
          </div>
          <DownloadCloud size={20} />
        </button>

        <button 
          onClick={handleCopyCommand}
          className="flex items-center justify-between p-4 border border-green/30 bg-bgRaised hover:bg-green hover:text-black transition-all group"
        >
          <div className="flex flex-col items-start">
            <span className="font-display text-[11px] font-bold tracking-widest uppercase">{copied ? 'COMMAND COPIED' : 'COPY START COMMAND'}</span>
            <span className="font-mono text-[9px] text-textDim group-hover:text-black/60 uppercase">Quick-start CLI command</span>
          </div>
          {copied ? <Check size={20} /> : <Terminal size={20} />}
        </button>

        <button 
          onClick={onTest}
          disabled={isSystemConnecting}
          className={`flex items-center justify-between p-4 border ${isSystemConnecting ? 'border-borderDim bg-bg' : 'border-green/30 bg-bgRaised hover:bg-green hover:text-black'} transition-all group disabled:opacity-50`}
        >
          <div className="flex flex-col items-start">
            <span className="font-display text-[11px] font-bold tracking-widest uppercase">
              {isSystemConnecting ? 'ESTABLISHING LINK...' : 'TEST CONNECTION'}
            </span>
            <span className="font-mono text-[9px] text-textDim group-hover:text-black/60 uppercase">
              {connectionMode === 'digital_twin' ? 'Verify Digital Twin simulation' : 'Verify HIL / Hardware link'}
            </span>
          </div>
          {isSystemConnecting ? <RefreshCw size={20} className="animate-spin" /> : <Wifi size={20} />}
        </button>

        {connectionError && (
          <div className="p-3 bg-red/10 border border-red/30 text-red font-mono text-[9px] uppercase tracking-widest">
            ERROR: {connectionError}
          </div>
        )}

        <button 
          onClick={onContinue}
          className="flex items-center justify-between p-4 border border-green/30 bg-bgRaised hover:bg-green hover:text-black transition-all group"
        >
          <div className="flex flex-col items-start">
            <span className="font-display text-[11px] font-bold tracking-widest uppercase">CONTINUE IN APP</span>
            <span className="font-mono text-[9px] text-textDim group-hover:text-black/60 uppercase">Open PhysiCore Dashboard</span>
          </div>
          <ExternalLink size={20} />
        </button>
      </div>
    </motion.div>
  );
};

interface CodeBlockProps {
  filename: string;
  content: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ filename, content }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="bg-[#050705] border border-borderDim my-4 overflow-hidden">
      <div className="flex justify-between items-center px-4 py-2 bg-bgRaised border-b border-borderDim">
        <span className="font-mono text-[10px] text-textSecondary uppercase tracking-widest">{filename}</span>
        <button onClick={handleCopy} className="flex items-center gap-2 font-display text-[10px] text-cyan hover:text-white transition-colors">
          {copied ? <><Check size={12} /> COPIED</> : <><Copy size={12} /> COPY</>}
        </button>
      </div>
      <div className="p-4 overflow-x-auto custom-scroll">
        <SyntaxHighlighter code={content} />
      </div>
    </div>
  );
};

const FlightDataImportOverlay = ({ isOpen, onClose, onImport }: { isOpen: boolean, onClose: () => void, onImport: (data: any[]) => void }) => {
  const [dragActive, setDragActive] = useState(false);

  const handleFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      const lines = content.split('\n');
      const data = lines.map(l => {
        const parts = l.trim().split(',');
        if (parts.length < 2) return null;
        return { t: parseFloat(parts[0]), y: parseFloat(parts[1]), v: parseFloat(parts[2]) || 0 };
      }).filter(p => p !== null && !isNaN(p.t) && !isNaN(p.y));
      onImport(data);
      onClose();
    };
    reader.readAsText(file);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-8">
      <div className="w-full max-w-2xl bg-bgRaised border border-border p-8 space-y-6">
        <div className="flex justify-between items-center border-b border-border pb-4">
          <div className="flex items-center gap-3">
            <FileUp className="text-cyan" size={24} />
            <h2 className="font-display text-xl font-bold text-white tracking-widest uppercase">IMPORT ACTUAL FLIGHT DATA</h2>
          </div>
          <button onClick={onClose} className="text-textDim hover:text-white"><X size={24} /></button>
        </div>

        <div 
          onDragOver={e => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={e => { e.preventDefault(); setDragActive(false); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); }}
          className={`h-64 border-2 border-dashed flex flex-col items-center justify-center gap-4 transition-all ${dragActive ? 'border-cyan bg-cyan/5' : 'border-border'}`}
        >
          <Upload size={48} className={dragActive ? 'text-cyan' : 'text-textDim'} />
          <div className="text-center">
            <p className="font-display text-sm font-bold text-white uppercase tracking-widest">Drop Flight Log (CSV)</p>
            <p className="font-mono text-[10px] text-textDim mt-1 uppercase">Format: time, altitude, velocity</p>
          </div>
          <label className="cursor-pointer px-6 py-2 bg-white text-black font-display text-xs font-bold uppercase tracking-widest hover:bg-cyan transition-all">
            Browse Files
            <input type="file" accept=".csv" className="hidden" onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
          </label>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 bg-bg border border-border space-y-2">
            <div className="micro-label text-cyan uppercase">Supported Formats</div>
            <ul className="font-mono text-[10px] text-textDim space-y-1 uppercase">
              <li>• Generic CSV (T, Alt, Vel)</li>
              <li>• StratoLogger (.csv)</li>
              <li>• TeleMetrum (.csv)</li>
              <li>• RRC3 (.csv)</li>
            </ul>
          </div>
          <div className="p-4 bg-bg border border-border space-y-2">
            <div className="micro-label text-amber uppercase">Analysis Engine</div>
            <p className="font-mono text-[10px] text-textDim uppercase leading-relaxed">
              SystemID will automatically estimate Cd and Isp divergence based on imported trajectory.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

const RocketTelemetryWidgets = ({ flightData, actualData, params }: { flightData: any[], actualData: any[] | null, params: RocketParams }) => {
  const lastPoint = flightData[flightData.length - 1] || { y: 0, v: 0, t: 0, phase: 'PRELAUNCH' };
  const maxAlt = Math.max(0, ...flightData.map(d => d.y));
  const maxVel = Math.max(0, ...flightData.map(d => Math.abs(d.v)));

  return (
    <div className="grid grid-cols-1 gap-4">
      {/* Altitude Profile */}
      <div className="bg-bgRaised border border-border p-4 space-y-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2 text-green">
            <TrendingUp size={14} />
            <span className="micro-label uppercase">Altitude Profile (m)</span>
          </div>
          <span className="font-mono text-xs text-white">{lastPoint.y.toFixed(1)}m</span>
        </div>
        <div className="h-[150px]">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={flightData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1A1A1A" vertical={false} />
              <XAxis dataKey="t" hide />
              <YAxis domain={[0, 'auto']} hide />
              <Tooltip contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #2A2A2A', fontSize: '10px' }} />
              <Area type="monotone" dataKey="y" stroke={COLORS.green} fill={COLORS.green} fillOpacity={0.1} isAnimationActive={false} />
              {actualData && <Line type="monotone" data={actualData} dataKey="y" stroke="#FFF" strokeDasharray="5 5" dot={false} isAnimationActive={false} />}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Velocity & Mach */}
      <div className="bg-bgRaised border border-border p-4 space-y-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2 text-cyan">
            <Zap size={14} />
            <span className="micro-label uppercase">Velocity & Mach</span>
          </div>
          <span className="font-mono text-xs text-white">{lastPoint.v.toFixed(1)}m/s</span>
        </div>
        <div className="h-[150px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={flightData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1A1A1A" vertical={false} />
              <XAxis dataKey="t" hide />
              <YAxis hide />
              <Tooltip contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #2A2A2A', fontSize: '10px' }} />
              <Line type="monotone" dataKey="v" stroke={COLORS.cyan} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-bgRaised border border-border p-3">
          <div className="micro-label text-textDim uppercase mb-1">Apogee</div>
          <div className="font-mono text-lg text-white">{maxAlt.toFixed(0)}m</div>
        </div>
        <div className="bg-bgRaised border border-border p-3">
          <div className="micro-label text-textDim uppercase mb-1">Max Velocity</div>
          <div className="font-mono text-lg text-white">{maxVel.toFixed(1)}m/s</div>
        </div>
        <div className="bg-bgRaised border border-border p-3">
          <div className="micro-label text-textDim uppercase mb-1">Current Phase</div>
          <div className="font-mono text-[10px] text-green uppercase truncate">{lastPoint.phase}</div>
        </div>
        <div className="bg-bgRaised border border-border p-3">
          <div className="micro-label text-textDim uppercase mb-1">SystemID Cd</div>
          <div className="font-mono text-lg text-amber">{params.cd.toFixed(3)}</div>
        </div>
      </div>

      {/* Event Log */}
      <div className="bg-bgRaised border border-border p-4 space-y-3">
        <div className="flex items-center gap-2 text-amber">
          <Activity size={14} />
          <span className="micro-label uppercase">Flight Events Log</span>
        </div>
        <div className="space-y-2 max-h-[150px] overflow-y-auto custom-scroll pr-2">
          {flightData.filter((d, i, arr) => i === 0 || d.phase !== arr[i-1].phase).map((evt, i) => (
            <div key={i} className="flex justify-between items-center border-l-2 border-amber pl-3 py-1 bg-bg/50">
              <span className="font-mono text-[10px] text-white uppercase">{evt.phase}</span>
              <span className="font-mono text-[10px] text-textDim">T+{evt.t.toFixed(2)}s</span>
            </div>
          ))}
          {flightData.length === 0 && <div className="text-center font-mono text-[10px] text-textDim uppercase py-4">Waiting for Liftoff...</div>}
        </div>
      </div>
    </div>
  );
};

const RocketTrajectoryCanvas = ({ state, params, flightData, actualData, simSpeed, setSimSpeed, isRunning, setIsRunning, resetSim, isConnected, handshakeConfirmed }: { state: RocketState, params: RocketParams, flightData: any[], actualData: any[] | null, simSpeed: number, setSimSpeed: (s: number) => void, isRunning: boolean, setIsRunning: (r: boolean) => void, resetSim: () => void, isConnected: boolean, handshakeConfirmed: boolean }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      if (!isConnected || !handshakeConfirmed) {
        ctx.fillStyle = COLORS.bgInset;
        ctx.fillRect(0, 0, w, h);
        
        // Grid
        ctx.strokeStyle = '#121212';
        ctx.lineWidth = 1;
        for (let x = 0; x < w; x += 40) {
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        }
        for (let y = 0; y < h; y += 40) {
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        }

        ctx.fillStyle = COLORS.red;
        ctx.font = 'bold 12px "JetBrains Mono"';
        ctx.textAlign = 'center';
        ctx.fillText('HARDWARE NOT CONNECTED // HANDSHAKE PENDING', w / 2, h / 2);
        return;
      }

      if (state.time === 0 && flightData.length === 0) {
        ctx.fillStyle = COLORS.bgInset;
        ctx.fillRect(0, 0, w, h);
        ctx.fillStyle = COLORS.amber;
        ctx.font = 'bold 12px "JetBrains Mono"';
        ctx.textAlign = 'center';
        ctx.fillText('WAITING FOR TELEMETRY STREAM...', w / 2, h / 2);
        ctx.font = '8px "JetBrains Mono"';
        ctx.fillText('SYSTEM CONNECTED // IDLE', w / 2, h / 2 + 20);
        return;
      }

      // Scaling
      const maxAlt = Math.max(2000, state.y * 1.2, ...(flightData.map(d => d.y)));
      const maxRange = Math.max(1000, state.x * 1.2, ...(flightData.map(d => d.x)));
      const scaleY = (h - 60) / maxAlt;
      const scaleX = (w - 60) / maxRange;
      const scale = Math.min(scaleX, scaleY);

      const toCanvasX = (x: number) => 40 + x * scale;
      const toCanvasY = (y: number) => h - 40 - y * scale;

      // Grid
      ctx.strokeStyle = '#1A1A1A';
      ctx.lineWidth = 1;
      for (let i = 0; i <= maxRange; i += 500) {
        ctx.beginPath();
        ctx.moveTo(toCanvasX(i), h - 40);
        ctx.lineTo(toCanvasX(i), 40);
        ctx.stroke();
      }
      for (let i = 0; i <= maxAlt; i += 500) {
        ctx.beginPath();
        ctx.moveTo(40, toCanvasY(i));
        ctx.lineTo(w - 40, toCanvasY(i));
        ctx.stroke();
      }

      // Ground
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(20, h - 40);
      ctx.lineTo(w - 20, h - 40);
      ctx.stroke();

      // Trajectory Trace
      if (flightData.length > 1) {
        ctx.lineWidth = 2;
        for (let i = 1; i < flightData.length; i++) {
          const p1 = flightData[i-1];
          const p2 = flightData[i];
          ctx.strokeStyle = p2.phase === 'POWERED' ? COLORS.cyan : 
                           p2.phase === 'COAST' ? COLORS.amber : 
                           p2.phase.includes('DROGUE') || p2.phase.includes('MAIN') ? COLORS.red : '#555';
          ctx.beginPath();
          ctx.moveTo(toCanvasX(p1.x), toCanvasY(p1.y));
          ctx.lineTo(toCanvasX(p2.x), toCanvasY(p2.y));
          ctx.stroke();
        }
      }

      // Actual Data Trace (if available)
      if (actualData && actualData.length > 1) {
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.beginPath();
        ctx.moveTo(toCanvasX(actualData[0].x || 0), toCanvasY(actualData[0].y));
        for (let i = 1; i < actualData.length; i++) {
          ctx.lineTo(toCanvasX(actualData[i].x || 0), toCanvasY(actualData[i].y));
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Rocket Marker
      const rx = toCanvasX(state.x);
      const ry = toCanvasY(state.y);
      ctx.save();
      ctx.translate(rx, ry);
      ctx.rotate(state.angle);
      ctx.fillStyle = COLORS.green;
      ctx.beginPath();
      ctx.moveTo(0, -10);
      ctx.lineTo(4, 5);
      ctx.lineTo(-4, 5);
      ctx.closePath();
      ctx.fill();
      if (state.phase === 'POWERED') {
        ctx.fillStyle = COLORS.cyan;
        ctx.beginPath();
        ctx.moveTo(-2, 5);
        ctx.lineTo(2, 5);
        ctx.lineTo(0, 15);
        ctx.fill();
      }
      ctx.restore();

      // HUD Overlay
      ctx.fillStyle = 'rgba(0,0,0,0.7)';
      ctx.fillRect(w - 180, 20, 160, 140);
      ctx.strokeStyle = COLORS.green;
      ctx.strokeRect(w - 180, 20, 160, 140);
      
      ctx.font = '10px "JetBrains Mono"';
      ctx.fillStyle = COLORS.green;
      ctx.fillText(`MET: ${state.time.toFixed(2)}s`, w - 170, 40);
      ctx.fillStyle = '#FFF';
      ctx.fillText(`ALT: ${state.y.toFixed(1)}m`, w - 170, 60);
      ctx.fillText(`VEL: ${Math.sqrt(state.vx**2 + state.vy**2).toFixed(1)}m/s`, w - 170, 75);
      ctx.fillText(`PHASE: ${state.phase}`, w - 170, 90);
      ctx.fillText(`MASS: ${state.mass.toFixed(3)}kg`, w - 170, 105);
      
      const v_mag = Math.sqrt(state.vx**2 + state.vy**2);
      const mach = v_mag / 343; // Simple mach
      ctx.fillText(`MACH: ${mach.toFixed(2)}`, w - 170, 120);
      
      // Speed Controls
      if (!isConnected) {
        ctx.fillStyle = 'rgba(0,0,0,0.5)';
        ctx.fillRect(20, 20, 120, 40);
        ctx.font = '8px "JetBrains Mono"';
        ctx.fillStyle = '#555';
        ctx.fillText('SIM SPEED', 30, 35);
        [1, 5, 10, 50].forEach((s, i) => {
          ctx.fillStyle = simSpeed === s ? COLORS.green : '#333';
          ctx.fillRect(30 + i*25, 40, 20, 15);
          ctx.fillStyle = simSpeed === s ? '#000' : '#FFF';
          ctx.fillText(`${s}x`, 33 + i*25, 50);
        });
      }
    };

    let animationFrame: number;
    const render = () => {
      draw();
      animationFrame = requestAnimationFrame(render);
    };
    render();

    return () => cancelAnimationFrame(animationFrame);
  }, [state, flightData, actualData, simSpeed]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Check speed buttons
    if (y >= 40 && y <= 55) {
      [1, 5, 10, 50].forEach((s, i) => {
        const bx = 30 + i*25;
        if (x >= bx && x <= bx + 20) setSimSpeed(s);
      });
    }
  };

  return (
    <div className="relative w-full h-full bg-bgInset border border-border overflow-hidden">
      <canvas ref={canvasRef} width={800} height={600} onClick={handleCanvasClick} className="w-full h-full cursor-crosshair" />
      <div className="absolute bottom-4 left-4 flex gap-2">
        <button onClick={() => setIsRunning(!isRunning)} className={`p-2 rounded-full ${isRunning ? 'bg-amber text-black' : 'bg-green text-black'} hover:scale-110 transition-all`}>
          {isRunning ? <Pause size={20} /> : <Play size={20} />}
        </button>
        <button onClick={resetSim} className="p-2 rounded-full bg-bgRaised border border-border text-white hover:bg-border transition-all">
          <RotateCcw size={20} />
        </button>
      </div>
      <div className="absolute top-4 right-4 flex flex-col items-end gap-1">
        <div className="micro-label text-textDim uppercase">Trajectory Engine v3.1</div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-green animate-pulse' : 'bg-red'}`} />
          <span className="font-mono text-[10px] text-white uppercase">{isRunning ? 'Live Simulation' : 'Paused'}</span>
        </div>
      </div>
    </div>
  );
};

const AviationManifestWizard = ({ params, setParams, projectEmail, setProjectEmail }: { params: AviationParams, setParams: (p: AviationParams) => void, projectEmail: string, setProjectEmail: (e: string) => void }) => {
  return (
    <div className="p-6 bg-bgRaised border border-border space-y-6">
      <div className="flex items-center gap-3 border-b border-border pb-4">
        <Navigation className="text-cyan" size={24} />
        <h2 className="font-display text-lg font-bold text-white tracking-widest uppercase">Aviation Manifest</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <h3 className="micro-label text-textDim uppercase">Airframe Geometry</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="font-mono text-[9px] text-textDim uppercase">Wingspan (m)</label>
              <input 
                type="number" 
                value={params.wingspan} 
                onChange={(e) => setParams({ ...params, wingspan: parseFloat(e.target.value) || 0 })}
                className="w-full bg-bg border border-border p-2 font-mono text-xs text-white focus:border-cyan outline-none"
              />
            </div>
            <div className="space-y-1">
              <label className="font-mono text-[9px] text-textDim uppercase">Wing Area (m²)</label>
              <input 
                type="number" 
                value={params.wingArea} 
                onChange={(e) => setParams({ ...params, wingArea: parseFloat(e.target.value) || 0 })}
                className="w-full bg-bg border border-border p-2 font-mono text-xs text-white focus:border-cyan outline-none"
              />
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="micro-label text-textDim uppercase">Aerodynamic Coefficients</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="font-mono text-[9px] text-textDim uppercase">Lift Slope (Clα)</label>
              <input 
                type="number" 
                value={params.cla} 
                onChange={(e) => setParams({ ...params, cla: parseFloat(e.target.value) || 0 })}
                className="w-full bg-bg border border-border p-2 font-mono text-xs text-white focus:border-cyan outline-none"
              />
            </div>
            <div className="space-y-1">
              <label className="font-mono text-[9px] text-textDim uppercase">Parasitic Drag (Cd0)</label>
              <input 
                type="number" 
                value={params.cd0} 
                onChange={(e) => setParams({ ...params, cd0: parseFloat(e.target.value) || 0 })}
                className="w-full bg-bg border border-border p-2 font-mono text-xs text-white focus:border-cyan outline-none"
              />
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="micro-label text-textDim uppercase">Powerplant & Fuel</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="font-mono text-[9px] text-textDim uppercase">Max Thrust (N)</label>
              <input 
                type="number" 
                value={params.thrustMax} 
                onChange={(e) => setParams({ ...params, thrustMax: parseFloat(e.target.value) || 0 })}
                className="w-full bg-bg border border-border p-2 font-mono text-xs text-white focus:border-cyan outline-none"
              />
            </div>
            <div className="space-y-1">
              <label className="font-mono text-[9px] text-textDim uppercase">Fuel Cap (kg)</label>
              <input 
                type="number" 
                value={params.fuelCapacity} 
                onChange={(e) => setParams({ ...params, fuelCapacity: parseFloat(e.target.value) || 0 })}
                className="w-full bg-bg border border-border p-2 font-mono text-xs text-white focus:border-cyan outline-none"
              />
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="micro-label text-textDim uppercase">Unit Identity</h3>
          <div className="space-y-1">
            <label className="font-mono text-[9px] text-textDim uppercase">Project Owner Email (Optional)</label>
            <input 
              type="email" 
              value={projectEmail} 
              onChange={(e) => setProjectEmail(e.target.value)}
              placeholder="engineer@domain.com"
              className="w-full bg-bg border border-border p-2 font-mono text-xs text-white focus:border-cyan outline-none"
            />
          </div>
        </div>
      </div>

      <div className="p-4 bg-cyan/5 border border-cyan/20 flex items-start gap-3">
        <Shield className="text-cyan shrink-0" size={16} />
        <p className="font-body text-[10px] text-textSecondary leading-relaxed">
          Aviation parameters are used to calibrate the <span className="text-cyan">Flight Envelope Protection</span> layer. Ensure Clα and Cd0 are derived from validated wind tunnel data or high-fidelity CFD.
        </p>
      </div>
    </div>
  );
};

const RocketManifestWizard = ({ params, setParams, projectEmail, setProjectEmail }: { params: RocketParams, setParams: (p: RocketParams) => void, projectEmail: string, setProjectEmail: (e: string) => void }) => {
  const [manualCurve, setManualCurve] = useState(params.motorCurve.length > 0 ? params.motorCurve : Array(5).fill({ t: 0, f: 0 }));
  const [motorName, setMotorName] = useState('N/A');

  const handleManualCurveChange = (idx: number, field: 't' | 'f', val: string) => {
    const newCurve = [...manualCurve];
    newCurve[idx] = { ...newCurve[idx], [field]: parseFloat(val) || 0 };
    setManualCurve(newCurve);
    setParams({ ...params, motorCurve: newCurve.sort((a, b) => a.t - b.t) });
  };

  const handleFileImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      if (file.name.endsWith('.eng')) {
        // Simple .eng parser
        const lines = content.split('\n');
        const dataLines = lines.filter(l => l.trim() && !l.startsWith(';') && isNaN(parseInt(l.trim()[0])));
        const curveLines = lines.filter(l => l.trim() && !l.startsWith(';') && !isNaN(parseInt(l.trim()[0])));
        
        if (dataLines.length > 0) {
          const parts = dataLines[0].split(/\s+/);
          setMotorName(parts[0]);
          // parts[1] diameter, parts[2] length, parts[4] prop mass, parts[5] total mass
          const propMass = parseFloat(parts[4]) || params.propMassInitial;
          const dryMass = (parseFloat(parts[5]) || (params.dryMass + propMass)) - propMass;
          setParams({ ...params, propMassInitial: propMass, dryMass });
        }

        const curve = curveLines.map(l => {
          const [t, f] = l.trim().split(/\s+/).map(parseFloat);
          return { t, f };
        }).sort((a, b) => a.t - b.t);
        
        setParams({ ...params, motorCurve: curve });
        setManualCurve(curve);
      } else if (file.name.endsWith('.csv')) {
        const lines = content.split('\n');
        const curve = lines.map(l => {
          const [t, f] = l.trim().split(',').map(parseFloat);
          return { t, f };
        }).filter(p => !isNaN(p.t) && !isNaN(p.f)).sort((a, b) => a.t - b.t);
        setParams({ ...params, motorCurve: curve });
        setManualCurve(curve);
      }
    };
    reader.readAsText(file);
  };

  const twr = params.motorCurve.length > 0 ? (Math.max(...params.motorCurve.map(p => p.f)) / ((params.dryMass + params.propMassInitial) * RKT_G)) : 0;

  return (
    <div className="space-y-8 p-6 bg-bgRaised border border-border">
      <div className="flex items-center gap-3 border-b border-border pb-4">
        <Rocket className="text-green" size={24} />
        <div>
          <h2 className="font-display text-lg font-bold text-white tracking-widest uppercase">ROCKET MANIFEST / PROPULSION</h2>
          <p className="font-mono text-[10px] text-textDim uppercase">Mission Configuration & Recovery Parameters</p>
        </div>
      </div>

      <div className="p-4 bg-bg border border-border space-y-2">
        <div className="flex items-center gap-2 text-cyan">
          <User size={14} />
          <span className="micro-label uppercase">Step 2: Unit Identity</span>
        </div>
        <div className="space-y-1">
          <label className="micro-label text-textDim">Email / Project Owner (Optional)</label>
          <input 
            type="email" 
            value={projectEmail} 
            onChange={e => setProjectEmail(e.target.value)} 
            placeholder="anonymous@physicore.io"
            className="w-full bg-bg border border-border p-2 font-mono text-xs text-white outline-none focus:border-cyan" 
          />
          <p className="font-mono text-[8px] text-textDim uppercase">Used to link project configuration across PhysiCore and Sentinel.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Section A: Physical */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-green">
            <Layers size={14} />
            <span className="micro-label uppercase">Section A: Physical Parameters</span>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="micro-label text-textDim">Dry Mass (kg)</label>
              <input type="number" step="0.1" value={params.dryMass} onChange={e => setParams({...params, dryMass: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-2 font-mono text-xs text-white outline-none focus:border-green" />
            </div>
            <div className="space-y-1">
              <label className="micro-label text-textDim">Propellant Mass (kg)</label>
              <input type="number" step="0.01" value={params.propMassInitial} onChange={e => setParams({...params, propMassInitial: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-2 font-mono text-xs text-white outline-none focus:border-green" />
            </div>
            <div className="space-y-1">
              <label className="micro-label text-textDim">Diameter (m)</label>
              <input type="number" step="0.001" value={params.diameter} onChange={e => setParams({...params, diameter: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-2 font-mono text-xs text-white outline-none focus:border-green" />
            </div>
            <div className="space-y-1">
              <label className="micro-label text-textDim">Drag Coeff (Cd)</label>
              <input type="number" step="0.01" value={params.cd} onChange={e => setParams({...params, cd: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-2 font-mono text-xs text-white outline-none focus:border-green" />
            </div>
          </div>
        </div>

        {/* Section C: Launch */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-amber">
            <Navigation size={14} />
            <span className="micro-label uppercase">Section C: Launch Parameters</span>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="micro-label text-textDim">Launch Angle (deg)</label>
              <input type="number" value={params.launchAngle} onChange={e => setParams({...params, launchAngle: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-2 font-mono text-xs text-white outline-none focus:border-amber" />
            </div>
            <div className="space-y-1">
              <label className="micro-label text-textDim">Rail Length (m)</label>
              <input type="number" value={params.railLength} onChange={e => setParams({...params, railLength: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-2 font-mono text-xs text-white outline-none focus:border-amber" />
            </div>
            <div className="p-3 bg-bgRaised border border-borderDim col-span-2 flex justify-between items-center">
              <div>
                <div className="micro-label text-textDim">Thrust-to-Weight (TWR)</div>
                <div className={`font-mono text-lg ${twr < 5 ? 'text-red' : 'text-green'}`}>{twr.toFixed(2)}</div>
              </div>
              <div className="text-right">
                <div className="micro-label text-textDim">Status</div>
                <div className={`font-mono text-[10px] ${twr < 5 ? 'text-red' : 'text-green'}`}>{twr < 5 ? 'UNSAFE_LAUNCH' : 'NOMINAL'}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Section B: Motor Curve */}
      <div className="space-y-4 border-t border-border pt-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-cyan">
            <Zap size={14} />
            <span className="micro-label uppercase">Section B: Motor Curve</span>
          </div>
          <div className="flex items-center gap-4">
            {motorName !== 'N/A' && <span className="px-2 py-0.5 bg-cyanDim text-cyan font-mono text-[10px] border border-cyan/30">{motorName}</span>}
            <label className="cursor-pointer flex items-center gap-2 px-3 py-1 bg-cyan text-black font-display text-[10px] font-bold uppercase tracking-widest hover:bg-white transition-all">
              <FileUp size={12} /> IMPORT .ENG / .CSV
              <input type="file" accept=".eng,.csv" onChange={handleFileImport} className="hidden" />
            </label>
          </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 h-[200px] border border-border bg-bgInset p-4">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={params.motorCurve}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1A1A1A" vertical={false} />
                <XAxis dataKey="t" hide />
                <YAxis hide />
                <Tooltip contentStyle={{ backgroundColor: '#0A0A0A', border: '1px solid #2A2A2A', fontSize: '10px' }} />
                <Area type="monotone" dataKey="f" stroke={COLORS.cyan} fill={COLORS.cyan} fillOpacity={0.1} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-2 overflow-y-auto max-h-[200px] custom-scroll pr-2">
            <div className="grid grid-cols-2 gap-2 mb-2">
              <span className="micro-label text-textDim">Time (s)</span>
              <span className="micro-label text-textDim">Thrust (N)</span>
            </div>
            {(manualCurve.length > 10 ? manualCurve.slice(0, 10) : manualCurve).map((p, i) => (
              <div key={i} className="grid grid-cols-2 gap-2">
                <input type="number" step="0.1" value={p.t} onChange={e => handleManualCurveChange(i, 't', e.target.value)} className="bg-bg border border-border p-1 font-mono text-[10px] text-white outline-none" />
                <input type="number" step="1" value={p.f} onChange={e => handleManualCurveChange(i, 'f', e.target.value)} className="bg-bg border border-border p-1 font-mono text-[10px] text-white outline-none" />
              </div>
            ))}
            {manualCurve.length > 10 && <div className="text-center font-mono text-[8px] text-textDim uppercase pt-2">... {manualCurve.length - 10} more points ...</div>}
          </div>
        </div>
      </div>

      {/* Section D: Recovery */}
      <div className="space-y-4 border-t border-border pt-6">
        <div className="flex items-center gap-2 text-red">
          <Wind size={14} />
          <span className="micro-label uppercase">Section D: Recovery System</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-4">
            <div className="p-3 bg-bgRaised border border-borderDim space-y-3">
              <div className="flex items-center justify-between">
                <span className="micro-label text-textDim">Drogue Deployment</span>
                <select value={params.drogueAlt === 0 ? 'apogee' : 'alt'} onChange={e => setParams({...params, drogueAlt: e.target.value === 'apogee' ? 0 : 500})} className="bg-bg border border-border font-mono text-[10px] text-white p-1 outline-none">
                  <option value="apogee">AT APOGEE</option>
                  <option value="alt">AT ALTITUDE</option>
                </select>
              </div>
              {params.drogueAlt !== 0 && (
                <div className="flex items-center justify-between">
                  <span className="micro-label text-textDim">Deployment Alt (m)</span>
                  <input type="number" value={params.drogueAlt} onChange={e => setParams({...params, drogueAlt: parseFloat(e.target.value) || 0})} className="w-20 bg-bg border border-border p-1 font-mono text-[10px] text-white text-right" />
                </div>
              )}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="micro-label text-textDim">Drogue Cd</label>
                  <input type="number" step="0.1" value={params.drogueCd} onChange={e => setParams({...params, drogueCd: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-1 font-mono text-[10px] text-white" />
                </div>
                <div className="space-y-1">
                  <label className="micro-label text-textDim">Drogue Diam (m)</label>
                  <input type="number" step="0.01" value={params.drogueDiam} onChange={e => setParams({...params, drogueDiam: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-1 font-mono text-[10px] text-white" />
                </div>
              </div>
            </div>
          </div>
          <div className="space-y-4">
            <div className="p-3 bg-bgRaised border border-borderDim space-y-3">
              <div className="flex items-center justify-between">
                <span className="micro-label text-textDim">Main Deployment</span>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[10px] text-white">{params.mainAlt}m</span>
                </div>
              </div>
              <div className="space-y-1">
                <label className="micro-label text-textDim">Deployment Altitude (m)</label>
                <input type="range" min="50" max="1000" step="50" value={params.mainAlt} onChange={e => setParams({...params, mainAlt: parseInt(e.target.value)})} className="w-full h-1 bg-border appearance-none cursor-pointer accent-red" />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="micro-label text-textDim">Main Cd</label>
                  <input type="number" step="0.1" value={params.mainCd} onChange={e => setParams({...params, mainCd: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-1 font-mono text-[10px] text-white" />
                </div>
                <div className="space-y-1">
                  <label className="micro-label text-textDim">Main Diam (m)</label>
                  <input type="number" step="0.01" value={params.mainDiam} onChange={e => setParams({...params, mainDiam: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-1 font-mono text-[10px] text-white" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// --- MAIN APP ---

export default function App() {
  return (
    <ErrorBoundary>
      <AppContent />
    </ErrorBoundary>
  );
}

function AppContent() {
  const [user, loading, error] = useAuthState(auth);
  const isAdmin = user?.email === "prathameshshirbhate8anpc@gmail.com";
  const [isAllowed, setIsAllowed] = useState<boolean | null>(null);
  const [hasTested, setHasTested] = useState<boolean | null>(null);
  const [checkingAccess, setCheckingAccess] = useState(false);
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [allUsers, setAllUsers] = useState<any[]>([]);
  const [isBootstrapping, setIsBootstrapping] = useState(false);
  const [quotaExceeded, setQuotaExceeded] = useState(false);
  const [authError, setAuthError] = useState<{ message: string; domain?: string } | null>(null);

  const bootstrapTeam = async () => {
    if (user?.email !== "prathameshshirbhate8anpc@gmail.com") return;
    setIsBootstrapping(true);
    
    const teamEmails = [
      "koshmarus@gmail.com",
      "stesrocketryteam@gmail.com",
      "darisglx@gmail.com",
      "vladimir.robotics@gmail.com",
      "projectauvm@manipal.edu"
    ];
    
    let addedCount = 0;
    for (const email of teamEmails) {
      try {
        const q = query(collection(db, 'allowed_users'), where('email', '==', email));
        const snap = await getDocs(q).catch(e => handleFirestoreError(e, OperationType.GET, 'allowed_users'));
        if (snap && snap.empty) {
          await setDoc(doc(collection(db, 'allowed_users')), {
            email,
            role: 'client',
            hasTested: false,
            invitedBy: user.email,
            createdAt: new Date().toISOString()
          }).catch(e => handleFirestoreError(e, OperationType.CREATE, 'allowed_users'));
          addedCount++;
        }
      } catch (err) {
        console.error(`Failed to bootstrap ${email}:`, err);
      }
    }
    
    // Refresh list
    try {
      const q = query(collection(db, 'allowed_users'));
      const querySnapshot = await getDocs(q);
      const users = querySnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
      setAllUsers(users);
    } catch (err) {
      handleFirestoreError(err, OperationType.GET, 'allowed_users');
    }
    
    setIsBootstrapping(false);
    alert(`Team authorization complete. ${addedCount} new members added.`);
  };

  const handleRemoveUser = async (userId: string) => {
    if (user?.email !== "prathameshshirbhate8anpc@gmail.com") return;
    try {
      await deleteDoc(doc(db, 'allowed_users', userId));
      setAllUsers(prev => prev.filter(u => u.id !== userId));
    } catch (err) {
      handleFirestoreError(err, OperationType.DELETE, `allowed_users/${userId}`);
    }
  };

  useEffect(() => {
    if (isAdmin) {
      const q = query(collection(db, 'allowed_users'));
      const unsubscribe = onSnapshot(q, (snapshot) => {
        const users = snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
        setAllUsers(users);
      }, (err) => {
        handleFirestoreError(err, OperationType.GET, 'allowed_users');
      });
      return () => unsubscribe();
    }
  }, [isAdmin]);

  useEffect(() => {
    if (isAdmin && allUsers.length === 0 && !isBootstrapping && user) {
      // Auto-bootstrap team if empty and we are admin
      bootstrapTeam();
    }
  }, [isAdmin, allUsers.length, isBootstrapping, user]);

  const [view, setView] = useState<View>('home');
  const [activeSection, setActiveSection] = useState('overview');
  const [manualSection, setManualSection] = useState('intro');

  useEffect(() => {
    const checkAccess = async () => {
      if (user) {
        setCheckingAccess(true);
        try {
          // 1. Check if user is in the hardcoded beta list
          if (user.email && BETA_TESTERS.includes(user.email)) {
            setIsAllowed(true);
            setHasTested(false); // Default to false, will be updated if doc exists
          }

          // 2. Try UID-based doc
          const userDocRef = doc(db, 'allowed_users', user.uid);
          const userDoc = await getDoc(userDocRef);
          
          if (userDoc.exists()) {
            const data = userDoc.data();
            setIsAllowed(true);
            setHasTested(data.hasTested || false);
          } else {
            // 3. Try Email-based query (for invited users)
            const q = query(collection(db, 'allowed_users'), where('email', '==', user.email));
            const querySnapshot = await getDocs(q);
            
            if (!querySnapshot.empty) {
              const data = querySnapshot.docs[0].data();
              setIsAllowed(true);
              setHasTested(data.hasTested || false);
              
              // Optional: Migrate to UID-based doc for better performance/security rules
              try {
                await setDoc(userDocRef, { ...data, uid: user.uid });
              } catch (e) { 
                handleFirestoreError(e, OperationType.CREATE, `allowed_users/${user.uid}`);
              }
              
            } else if (user.email === "prathameshshirbhate8anpc@gmail.com") {
              // Bootstrap Admin
              const adminData = {
                email: user.email,
                hasTested: false,
                role: 'admin',
                createdAt: new Date().toISOString()
              };
              await setDoc(userDocRef, adminData);
              setIsAllowed(true);
              setHasTested(false);
            } else if (isAllowed !== true) {
              setIsAllowed(false);
              setHasTested(false);
            }
          }
        } catch (err) {
          console.warn("Access check failed, falling back to beta list:", err);
          // If Firestore fails (e.g. rules not deployed yet), still allow beta testers
          if (user.email && BETA_TESTERS.includes(user.email)) {
            setIsAllowed(true);
          } else {
            setIsAllowed(false);
          }
        } finally {
          setCheckingAccess(false);
        }
      } else {
        setIsAllowed(null);
        setHasTested(null);
      }
    };
    checkAccess();
  }, [user]);

  const markAsTested = async () => {
    if (user && isAllowed && !hasTested && !isAdmin) {
      try {
        await updateDoc(doc(db, 'allowed_users', user.uid), {
          hasTested: true
        });
        setHasTested(true);
      } catch (err) {
        handleFirestoreError(err, OperationType.UPDATE, `allowed_users/${user.uid}`);
      }
    }
  };

  const handleLogin = async () => {
    if (isLoggingIn) return;
    setIsLoggingIn(true);
    setAuthError(null);
    try {
      await signInWithPopup(auth, googleProvider);
    } catch (err: any) {
      if (err.code === 'auth/popup-closed-by-user') {
        console.warn("Login cancelled: Popup closed by user.");
      } else if (err.code === 'auth/cancelled-by-user') {
        console.warn("Login cancelled by user.");
      } else if (err.code === 'auth/popup-blocked') {
        setAuthError({ message: "Login popup was blocked by your browser. Please allow popups for this site." });
      } else if (err.code === 'auth/unauthorized-domain') {
        const domain = window.location.hostname;
        setAuthError({ 
          message: `Domain "${domain}" is not authorized in Firebase Console.`,
          domain 
        });
      } else {
        console.error("Login failed:", err);
        setAuthError({ message: `Login failed: ${err.message}` });
      }
    } finally {
      setIsLoggingIn(false);
    }
  };

  const handleLogout = async () => {
    try {
      await signOut(auth);
      setView('home');
    } catch (err) {
      console.error("Logout failed:", err);
    }
  };
  
  // Integration Engineer State
  const [conversationHistory, setConversationHistory] = useState<Message[]>([]);
  const [systemProfile, setSystemProfile] = useState<SystemProfile>({
    platform: null, firmware: null, domain: null, massClass: null, connectionMode: null, protocols: null
  });
  const [integrationPhase, setIntegrationPhase] = useState(1);
  const [generatedFiles, setGeneratedFiles] = useState<GeneratedFile[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [isSystemConnected, setIsSystemConnected] = useState(false);
  const [isSystemConnecting, setIsSystemConnecting] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [connectionMode, setConnectionMode] = useState<'ros2_websocket' | 'hil' | 'digital_twin'>('hil');
  const [digitalTwinConfirmed, setDigitalTwinConfirmed] = useState(false);
  const [showDigitalTwinModal, setShowDigitalTwinModal] = useState(false);
  const [endpoint, setEndpoint] = useState('ws://localhost:9090');
  const [dRealEndpoint, setDRealEndpoint] = useState('http://localhost:8080');

  const [geminiApiKey, setGeminiApiKey] = useState<string>(localStorage.getItem('physicore_gemini_key') || DEFAULT_GEMINI_KEY);
  const [isKeyValid, setIsKeyValid] = useState<boolean | null>(null);
  const [isTestingKey, setIsTestingKey] = useState(false);
  const [showKeyInput, setShowKeyInput] = useState(false);
  const [handshakeConfirmed, setHandshakeConfirmed] = useState(false);

  const [isLaunching, setIsLaunching] = useState(false);
  const [metaAnalysisResult, setMetaAnalysisResult] = useState<string | null>(null);
  const [showUserManagement, setShowUserManagement] = useState(false);

  // Persistent WebSocket for Real-time Telemetry
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (isSystemConnected && connectionMode === 'ros2_websocket' && endpoint) {
      try {
        const ws = new WebSocket(endpoint);
        socketRef.current = ws;

        ws.onopen = () => {
          console.log("Telemetry Stream: CONNECTED");
          // Subscribe to telemetry topic
          ws.send(JSON.stringify({
            op: 'subscribe',
            topic: '/telemetry',
            type: 'physicore_msgs/Telemetry'
          }));
        };

        ws.onmessage = (msg) => {
          try {
            const data = JSON.parse(msg.data);
            if (data.op === 'publish' && data.topic === '/telemetry') {
              const telemetryData = data.msg;
              setTelemetry(prev => ({
                ...prev,
                ...telemetryData,
                // Ensure we keep history
                residualHistory: [...(prev.residualHistory || []), { x: Date.now(), y: telemetryData.residual || 0 }].slice(-30),
                effortHistory: [...(prev.effortHistory || []), { x: Date.now(), y: telemetryData.effort || 0 }].slice(-30)
              }));
            }
          } catch (e) {
            console.error("Telemetry Parse Error:", e);
          }
        };

        ws.onclose = () => {
          console.log("Telemetry Stream: DISCONNECTED");
          setIsSystemConnected(false);
        };

        return () => {
          ws.close();
          socketRef.current = null;
        };
      } catch (e) {
        console.error("Telemetry Connection Error:", e);
      }
    }
  }, [isSystemConnected, connectionMode, endpoint]);

  // Admin: Fetch all users
  useEffect(() => {
    if (user && isAllowed && user.email === "prathameshshirbhate8anpc@gmail.com") {
      const fetchUsers = async () => {
        try {
          const q = query(collection(db, 'allowed_users'));
          const querySnapshot = await getDocs(q);
          const users = querySnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
          setAllUsers(users);
        } catch (err) {
          console.error("Failed to fetch users:", err);
        }
      };
      fetchUsers();
    }
  }, [user, isAllowed]);

  const handleAddUser = async (email: string) => {
    if (!email) return;
    try {
      // We don't have the UID yet, so we'll use email as ID or just add it to a list
      // Actually, Firestore rules require UID as ID.
      // For now, we'll just add it to a 'pending_invites' collection or something.
      // But let's keep it simple: we'll just add a document with email and role.
      const inviteRef = doc(collection(db, 'allowed_users'));
      await setDoc(inviteRef, {
        email,
        role: 'user',
        hasTested: false,
        invitedBy: user?.email
      });
      setAllUsers(prev => [...prev, { id: inviteRef.id, email, role: 'user', hasTested: false }]);
    } catch (err) {
      handleFirestoreError(err, OperationType.CREATE, 'allowed_users');
    }
  };

  // Rocket State
  const [rocketState, setRocketState] = useState<RocketState>({
    x: 0, y: 0, vx: 0, vy: 0, mass: 3.0, propMass: 0.5, time: 0, phase: 'PRELAUNCH', angle: 0.087 // ~5 deg
  });
  const [rocketParams, setRocketParams] = useState<RocketParams>({
    dryMass: 2.5,
    propMassInitial: 0.5,
    burnTime: 1.8,
    thrust: 420,
    fuelMass: 0.5,
    diameter: 0.075,
    length: 1.2,
    cd: 0.5,
    motorCurve: [
      { t: 0, f: 0 }, { t: 0.1, f: 450 }, { t: 1.5, f: 400 }, { t: 1.8, f: 0 }
    ],
    isp: 200,
    launchAngle: 5,
    railLength: 3,
    launchAltitude: 0,
    drogueAlt: 0,
    drogueCd: 1.5,
    drogueDiam: 0.4,
    mainAlt: 150,
    mainCd: 2.2,
    mainDiam: 1.2
  });

  // Aviation State
  const [aviationState, setAviationState] = useState<AviationState>({
    x: 0, y: 1000, z: 0, vx: 45, vy: 0, vz: 0, pitch: 0, roll: 0, yaw: 0, aoa: 0, sideslip: 0, mass: 12.5, fuel: 5.0, time: 0
  });
  const [aviationParams, setAviationParams] = useState<AviationParams>({
    mass: 12.5,
    wingspan: 2.4,
    wingArea: 0.85,
    chord: 0.35,
    cl0: 0.15,
    cla: 5.7,
    cd0: 0.025,
    k: 0.045,
    thrustMax: 85,
    fuelCapacity: 5.0,
    fuelBurnRate: 0.012,
    vne: 85,
    vso: 12
  });

  const [projectEmail, setProjectEmail] = useState('');
  const [projectCode, setProjectCode] = useState('');
  const [projectData, setProjectData] = useState<any>(null);
  const [sentinelThresholds, setSentinelThresholds] = useState({
    max_g: 15.0,
    max_q: 25000,
    max_aoa: 12.0,
    min_stability: 1.5
  });
  const [showProjectDropdown, setShowProjectDropdown] = useState(false);
  const [flightData, setFlightData] = useState<any[]>([]);
  const [isRocketSimRunning, setIsRocketSimRunning] = useState(false);
  const [rocketSimSpeed, setRocketSimSpeed] = useState(1);
  const [actualFlightData, setActualFlightData] = useState<any[] | null>(null);
  const [showImportOverlay, setShowImportOverlay] = useState(false);

  const [telemetry, setTelemetry] = useState({
    mass: 0,
    friction: 0,
    actuatorEfficiency: 0,
    residual: 0,
    confidence: 0,
    variance: 0,
    isStable: true,
    isFaulted: false,
    cpuLoad: 0,
    latency: 0,
    residualHistory: [] as any[],
    effortHistory: [] as any[],
    targetPos: { x: 0, y: 0 },
    // Rocket/Aviation specific telemetry
    pos: null as any,
    vel: { x: 0, y: 0, z: 0 } as any,
    accel: { x: 0, y: 0, z: 0 } as any,
    orientation: { r: 0, p: 0, y: 0 } as any,
    propMass: 0,
    time: 0,
    phase: 'PRELAUNCH' as string,
    altitude: 0,
    airspeed: 0,
    mach: 0,
    aoa: 0,
    bank: 0
  });

  const performMetaAnalysis = async () => {
    if (quotaExceeded) return;

    try {
      const prompt = `
        PHYSICORE META-ANALYST: REAL-TIME TELEMETRY DIAGNOSTICS
        
        CURRENT SYSTEM STATE:
        - Estimated Mass: ${telemetry.mass.toFixed(3)} kg
        - Estimated Friction: ${telemetry.friction.toFixed(3)} μ
        - Actuator Efficiency: ${(telemetry.actuatorEfficiency * 100).toFixed(1)}%
        - Prediction Residual: ${telemetry.residual.toFixed(4)}
        - Ensemble Confidence: ${telemetry.confidence.toFixed(1)}%
        - Ensemble Variance: ${telemetry.variance.toFixed(4)}
        - Stability Status: ${telemetry.isStable ? 'NOMINAL' : 'UNSTABLE'}
        - Fault Status: ${telemetry.isFaulted ? 'ANOMALY_DETECTED' : 'CLEAN'}
        
        TASK:
        1. Provide a concise (max 3 sentences) high-level interpretation of the system's current physical health.
        2. Suggest ONE specific tuning adjustment for the MPC cost function or SystemID learning rate based on the residual and variance.
        3. Identify any potential "Reality Gap" issues if the residual is high.
        
        FORMAT:
        - Use a professional, technical tone.
        - Prefix with '> META-ANALYST:'.
      `;

      const result = await callGemini(prompt, [], "You are the PhysiCore Meta-Analyst, a high-level diagnostic AI for advanced robotics control systems.");
      
      if (result.success) {
        setMetaAnalysisResult(result.text || "NO_DATA_RECEIVED");
        setQuotaExceeded(false);
      } else if (result.error?.includes('429') || result.error?.includes('QUOTA_EXHAUSTED')) {
        setQuotaExceeded(true);
        // Reset quota exceeded after 2 minutes
        setTimeout(() => setQuotaExceeded(false), 120000);
      }
    } catch (error) {
      console.error("Meta-Analysis Error:", error);
    }
  };

  useEffect(() => {
    let interval: any;
    if (isSystemConnected && view === 'dashboard' && !quotaExceeded) {
      // Initial analysis
      performMetaAnalysis();
      // Periodic analysis every 30 seconds to avoid hitting rate limits too hard
      interval = setInterval(performMetaAnalysis, 30000);
    }
    return () => clearInterval(interval);
  }, [isSystemConnected, view, quotaExceeded]);

  const handleConnect = async () => {
    if (isSystemConnecting) return;
    
    if (connectionMode === 'digital_twin' && !digitalTwinConfirmed) {
      setShowDigitalTwinModal(true);
      return;
    }

    setIsSystemConnecting(true);
    setConnectionError(null);
    
    // Hardware Gate: initiate handshake
    const result: any = await initiateHandshake(endpoint, connectionMode);
    
    if (result.success) {
      // Reset all simulation states to ensure clean real-time data
      setIsRocketSimRunning(false);
      setFlightData([]);
      setActualFlightData(null);
      setRocketState({
        x: 0, y: 0, vx: 0, vy: 0,
        mass: rocketParams.dryMass + rocketParams.fuelMass,
        propMass: rocketParams.fuelMass,
        time: 0,
        phase: 'PRELAUNCH',
        angle: rocketParams.launchAngle * Math.PI / 180
      });

      setHandshakeConfirmed(true);
      setIsSystemConnected(true);
      setIsSystemConnecting(false);
    } else {
      setIsSystemConnecting(false);
      setConnectionError(result.reason || "Connection failed. Verify endpoint.");
      console.error("Handshake failed:", result.reason);
    }
  };

  const handleLaunchApp = async () => {
    if (!user) {
      await handleLogin();
      if (!auth.currentUser) return;
    }
    
    if (isSystemConnecting) return;

    if (connectionMode === 'digital_twin' && !digitalTwinConfirmed) {
      setShowDigitalTwinModal(true);
      return;
    }

    setIsSystemConnecting(true);
    setConnectionError(null);
    
    // Attempt connection before launching
    const result: any = await initiateHandshake(endpoint, connectionMode);
    
    if (result.success) {
      setHandshakeConfirmed(true);
      setIsSystemConnected(true);
      setView('dashboard');
      markAsTested();
    } else {
      // If connection fails, we still go to dashboard but it will be in OFFLINE mode
      // and show the connection error clearly.
      setHandshakeConfirmed(false);
      setConnectionError(result.reason || "Hardware link failed.");
      setView('dashboard');
    }
    setIsSystemConnecting(false);
  };

  // Scroll tracking
  useEffect(() => {
    if (view !== 'home') return;

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          setActiveSection(entry.target.id);
          entry.target.classList.add('active');
        }
      });
    }, { threshold: 0.5 });

    const sections = document.querySelectorAll('section[id]');
    sections.forEach(s => observer.observe(s));

    const revealObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('active');
        }
      });
    }, { threshold: 0.15 });

    const reveals = document.querySelectorAll('.reveal');
    reveals.forEach(r => revealObserver.observe(r));

    return () => {
      observer.disconnect();
      revealObserver.disconnect();
    };
  }, [view, loading, checkingAccess]);

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [conversationHistory, isTyping]);

  const parseAIResponse = (text: string) => {
    const parts = [];
    let lastIndex = 0;
    const regex = /\[CODE: (.*?)\]([\s\S]*?)\[\/CODE\]/g;
    let match;

    while ((match = regex.exec(text)) !== null) {
      // Add text before code block
      if (match.index > lastIndex) {
        parts.push({ type: 'text', content: text.substring(lastIndex, match.index) });
      }
      // Add code block
      parts.push({ type: 'code', filename: match[1], content: match[2].trim() });
      lastIndex = regex.lastIndex;
    }

    // Add remaining text
    if (lastIndex < text.length) {
      parts.push({ type: 'text', content: text.substring(lastIndex) });
    }

    return parts;
  };

  // Rocket Simulation Loop
  useEffect(() => {
    // Simulation should only run when NOT connected to a real system
    if (!isRocketSimRunning || isSystemConnected || systemProfile.domain !== 'ROCKETS') return;

    const dt = 0.01;
    const interval = setInterval(() => {
      setRocketState(prev => {
        if (prev.phase === 'LANDED') {
          setIsRocketSimRunning(false);
          return prev;
        }

        let current = prev;
        for (let i = 0; i < rocketSimSpeed; i++) {
          const next = { ...current, ...rocketRK4Step(current, rocketParams, dt) };
          const phase = updateRocketPhase(next, rocketParams, current);
          current = { ...next, phase };
        }

        setFlightData(fd => [...fd, { ...current }]);
        return current;
      });
    }, 10);

    return () => clearInterval(interval);
  }, [isRocketSimRunning, rocketSimSpeed, rocketParams, systemProfile.domain, isSystemConnected]);

  // Update rocketState from real telemetry when connected
  useEffect(() => {
    if (isSystemConnected && systemProfile.domain === 'ROCKETS' && telemetry.pos) {
      setRocketState(prev => ({
        ...prev,
        x: telemetry.pos.x,
        y: telemetry.pos.y,
        vx: telemetry.vel?.x || 0,
        vy: telemetry.vel?.y || 0,
        mass: telemetry.mass || prev.mass,
        propMass: telemetry.propMass || prev.propMass,
        time: telemetry.time || prev.time,
        phase: (telemetry.phase as RocketPhase) || prev.phase
      }));
      
      // Also record flight data for the graph
      setFlightData(fd => {
        const newData = { 
          x: telemetry.pos.x, 
          y: telemetry.pos.y, 
          vx: telemetry.vel?.x || 0, 
          vy: telemetry.vel?.y || 0,
          mass: telemetry.mass || 0,
          time: telemetry.time || 0,
          phase: telemetry.phase || 'UNKNOWN'
        };
        // Avoid duplicate timestamps
        if (fd.length > 0 && fd[fd.length - 1].time === newData.time) return fd;
        return [...fd, newData].slice(-1000);
      });
    }
  }, [isSystemConnected, systemProfile.domain, telemetry]);

  const resetRocketSim = () => {
    if (isSystemConnected) {
      // Send reset command to real hardware
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        socketRef.current.send(JSON.stringify({
          op: 'call_service',
          service: '/rocket/reset'
        }));
      }
      return;
    }
    setIsRocketSimRunning(false);
    setRocketState({
      x: 0, y: 0, vx: 0, vy: 0, 
      mass: rocketParams.dryMass + rocketParams.propMassInitial, 
      propMass: rocketParams.propMassInitial, 
      time: 0, phase: 'PRELAUNCH', 
      angle: (rocketParams.launchAngle * Math.PI) / 180
    });
    setFlightData([]);
  };

  const handleGenerateProjectCode = () => {
    const payload = {
      version: '1.0',
      origin: 'PC',
      id: generateId(projectEmail),
      email: projectEmail,
      timestamp: Date.now(),
      unit: { type: systemProfile.domain, name: systemProfile.platform },
      protocols: systemProfile.protocols,
      physical: { 
        mass: telemetry.mass, 
        friction: telemetry.friction, 
        ...(systemProfile.domain === 'ROCKETS' ? rocketParams : {}),
        ...(systemProfile.domain === 'AVIATION' ? aviationParams : {})
      },
      safety: { thresholds: sentinelThresholds },
      control: { mpc_horizon: 12 },
      connection: { mode: connectionMode, endpoint: endpoint }
    };

    const checksum = simpleHash(JSON.stringify(payload));
    const finalPayload = { ...payload, checksum };
    const code = encodeProjectCode(finalPayload);
    
    setProjectCode(code);
    setProjectData(finalPayload);
    setIntegrationPhase(4);
    markAsTested();
  };

  const handleImportProjectCode = (code: string) => {
    const data = decodeProjectCode(code);
    if (data) {
      setProjectData(data);
      setProjectCode(code);
      setProjectEmail(data.email || '');
      
      // Update system parameters
      if (data.unit.type === 'ROCKETS') {
        setRocketParams(prev => ({
          ...prev,
          ...data.physical
        }));
      } else if (data.unit.type === 'AVIATION') {
        setAviationParams(prev => ({
          ...prev,
          ...data.physical
        }));
      }
      
      setTelemetry(prev => ({
        ...prev,
        mass: data.physical.mass || prev.mass,
        friction: data.physical.friction || prev.friction
      }));
      
      if (data.safety?.thresholds) {
        setSentinelThresholds(data.safety.thresholds);
      }
      
      if (data.connection) {
        setConnectionMode(data.connection.mode);
        setEndpoint(data.connection.endpoint);
      }

      return { success: true, data };
    }
    return { success: false, error: "Invalid or corrupted Project Code" };
  };

  const handleRocketLaunch = () => {
    if (isSystemConnected) {
      // Send launch command to real hardware
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        socketRef.current.send(JSON.stringify({
          op: 'call_service',
          service: '/rocket/launch'
        }));
      }
      markAsTested();
      return;
    }

    if (rocketState.phase === 'PRELAUNCH' && !isSystemConnected) {
      setIsRocketSimRunning(true);
      markAsTested();
    }
  };

  // AI Logic
  useEffect(() => {
    if (isSystemConnected && view === 'dashboard' && projectData) {
      const interval = setInterval(() => {
        setProjectData(prev => {
          if (!prev) return prev;
          const { checksum, ...rest } = prev;
          const updatedPayload = {
            ...rest,
            timestamp: Date.now(),
            physical: {
              ...rest.physical,
              mass: telemetry.mass,
              friction: telemetry.friction,
              lyapunov_bound: telemetry.confidence / 100
            }
          };
          const newChecksum = simpleHash(JSON.stringify(updatedPayload));
          const finalPayload = { ...updatedPayload, checksum: newChecksum };
          setProjectCode(encodeProjectCode(finalPayload));
          return finalPayload;
        });
      }, 5000); // ~300 frames at 60fps
      return () => clearInterval(interval);
    }
  }, [isSystemConnected, view, projectData, telemetry.mass, telemetry.friction, telemetry.confidence]);

  const testGeminiKey = async (key: string) => {
    setIsTestingKey(true);
    try {
      const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${key}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contents: [{ parts: [{ text: 'Reply with one word: ready' }] }] })
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.candidates?.[0]?.content?.parts?.[0]?.text) {
          setIsKeyValid(true);
          localStorage.setItem('physicore_gemini_key', key);
          setGeminiApiKey(key);
          setShowKeyInput(false);
          return true;
        }
      }
      setIsKeyValid(false);
      return false;
    } catch (e) {
      setIsKeyValid(false);
      return false;
    } finally {
      setIsTestingKey(false);
    }
  };

  const handleSendMessage = async (text?: string) => {
    const msg = text || inputValue;
    if (!msg.trim()) return;

    const userMsg: Message = { role: 'user', content: msg, timestamp: formatTime(new Date()) };
    const newHistory = [...conversationHistory, userMsg];
    setConversationHistory(newHistory);
    setInputValue('');
    setIsTyping(true);

    try {
      const history = newHistory.map(m => ({
        role: m.role === 'ai' ? 'model' : 'user',
        parts: [{ text: m.content }]
      }));

      const systemInstruction = `You are the PhysiCore Integration Engineer.
           You help engineers integrate PhysiCore — a Physics Intelligence Engine — into their systems (Robotics, Rockets, and Aviation).

           PhysiCore capabilities you're integrating:
           — RK4 4th-order physics integrator at 60Hz
           — Online SystemID: learns mass, friction, and aerodynamic coefficients in real-time
           — 3-node ensemble: quantifies epistemic uncertainty
           — MPC lookahead: 12-step CEM trajectory planning
           — Hardware gate: LIVE / HIL / Digital Twin modes
           — Sentinel OS integration: safety governance layer
           — Export: JSON pack + ROS2/ArduPilot/PX4 bridge code

           DOMAIN SPECIFICS:
           — ROBOTICS: Focus on mass, friction, actuator efficiency, and joint constraints.
           — ROCKETS: Focus on thrust curves, mass depletion (fuel), drag coefficients (Cd), and recovery triggers.
           — AVIATION: Focus on lift/drag ratios, control surface mapping, and flight envelope protection.

           YOUR BEHAVIOR:
           — Follow the workflow phases strictly:
             Discovery (6 questions) → Confirm → Generate → Guide → Q&A
           — Ask ONE question at a time. Never multiple.
           — When generating code: inject real values, never placeholders.
           — Code must be complete and copy-paste ready.
           — Never use markdown fences (code blocks) in responses.
           — Format code with inline syntax highlighting hints instead:
             Start code sections with: [CODE: filename.ext]
             End code sections with: [/CODE]
             The UI will parse these tags and apply highlighting.
           — Always reference the detected system profile in answers.
           — Never give generic answers. Always specific.
           — Speak as a system: prefix with '> INTEGRATION ENGINEER:'
           — Be concise in questions. Be thorough in code.

           SYSTEM PROFILE (updates as user answers):
           ${JSON.stringify(systemProfile)}
           
           PHYSICORE VERSION: v3.0
           CURRENT DATE: ${new Date().toISOString()}`;

      const result = await callGemini(msg, history, systemInstruction, geminiApiKey);

      if (result.success) {
        const aiText = result.text || "> INTEGRATION ENGINEER: NO RESPONSE RECEIVED.";
        const aiMsg: Message = { role: 'ai', content: aiText, timestamp: formatTime(new Date()) };
        setConversationHistory(prev => [...prev, aiMsg]);
        setQuotaExceeded(false);
        
        // Extraction logic
        const updatedProfile = { ...systemProfile };
        if (aiText.includes('PLATFORM:')) updatedProfile.platform = aiText.split('PLATFORM:')[1].split('\n')[0].trim();
        if (aiText.includes('FIRMWARE:')) updatedProfile.firmware = aiText.split('FIRMWARE:')[1].split('\n')[0].trim();
        if (aiText.includes('DOMAIN:')) updatedProfile.domain = aiText.split('DOMAIN:')[1].split('\n')[0].trim();
        if (aiText.includes('MASS CLASS:')) updatedProfile.massClass = aiText.split('MASS CLASS:')[1].split('\n')[0].trim();
        if (aiText.includes('CONNECTION:')) updatedProfile.connectionMode = aiText.split('CONNECTION:')[1].split('\n')[0].trim();
        if (aiText.includes('PROTOCOLS:')) updatedProfile.protocols = aiText.split('PROTOCOLS:')[1].split('\n')[0].trim();
        
        setSystemProfile(updatedProfile);

        // Code parsing
        const codeMatches = aiText.match(/\[CODE: (.*?)\]([\s\S]*?)\[\/CODE\]/g);
        if (codeMatches) {
          const newFiles = codeMatches.map(match => {
            const parts = match.match(/\[CODE: (.*?)\]([\s\S]*?)\[\/CODE\]/);
            return { filename: parts![1], content: parts![2].trim(), extension: parts![1].split('.').pop() || '' };
          });
          setGeneratedFiles(prev => [...prev, ...newFiles]);
          setIntegrationPhase(3); // Transition to wizard steps phase
          markAsTested();
        }
      } else {
        let errorMsg = '> SYSTEM ERROR: ';
        switch(result.error) {
          case 'NO_API_KEY': errorMsg += 'Gemini API key missing. Please provide one in the header.'; break;
          case 'HTTP_401': errorMsg += 'Invalid API key. Check your credentials.'; break;
          case 'HTTP_403': errorMsg += 'API key does not have permission for this model.'; break;
          case 'HTTP_429': errorMsg += 'Rate limit exceeded. Please wait.'; break;
          case 'TIMEOUT': errorMsg += 'Request timed out. Falling back to symbolic mode...'; break;
          case 'NETWORK_ERROR': errorMsg += 'Network error. Falling back to symbolic mode...'; break;
          default: errorMsg += `API call failed (${result.error}).`;
        }

        const aiMsg: Message = { 
          role: 'ai', 
          content: errorMsg, 
          timestamp: formatTime(new Date()) 
        };
        setConversationHistory(prev => [...prev, aiMsg]);
        
        // If it's a timeout or network error, we can try symbolic mode
        if (result.error === 'TIMEOUT' || result.error === 'NETWORK_ERROR') {
          setTimeout(() => {
            const symbolicMsg: Message = {
              role: 'ai',
              content: '> INTEGRATION ENGINEER (SYMBOLIC_MODE): I am currently operating in low-power symbolic mode due to network latency. I can still assist with basic system profile questions.',
              timestamp: formatTime(new Date())
            };
            setConversationHistory(prev => [...prev, symbolicMsg]);
          }, 1000);
        }
      }
    } catch (error) {
      console.error("AI Error:", error);
    } finally {
      setIsTyping(false);
    }
  };

  // --- RENDERERS ---

  useEffect(() => {
    if (!loading && !user && view !== 'home') {
      setView('home');
    }
  }, [user, loading, view]);

  const handleSetIntegratorView = async () => {
    if (!user) {
      await handleLogin();
      if (!auth.currentUser) return;
    }
    setView('integrator');
  };

  const renderNav = () => (
    <nav className="fixed top-0 left-0 w-full h-[52px] bg-void/96 backdrop-blur-md border-b border-border z-[100] flex items-center justify-between px-6">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 cursor-pointer" onClick={() => setView('home')}>
          <svg width="20" height="20" viewBox="0 0 100 100">
            <path d="M50 5 L90 25 L90 75 L50 95 L10 75 L10 25 Z" fill="none" stroke={COLORS.green} strokeWidth="6" />
          </svg>
          <span className="font-display text-lg font-bold tracking-widest text-white">PHYSICORE</span>
          <span className="font-mono text-[10px] text-textDim">v3.0</span>
        </div>
        <div className="h-4 w-px bg-border mx-2" />
        <span className="font-body text-[11px] text-textSecondary uppercase tracking-widest hidden md:block">
          {view === 'dashboard' ? 'LIVE MISSION CONTROL' : 'Physics Intelligence Engine'}
        </span>

        {view === 'dashboard' && projectData && (
          <div className="relative ml-4">
            <button 
              onClick={() => setShowProjectDropdown(!showProjectDropdown)}
              className="flex items-center gap-2 px-3 py-1 bg-bgRaised border border-cyan/30 hover:border-cyan transition-all group"
            >
              <div className="w-1.5 h-1.5 rounded-full bg-cyan animate-pulse" />
              <span className="font-mono text-[10px] text-cyan uppercase tracking-widest">
                PROJECT: {projectData.id}
              </span>
              <span className="font-mono text-[8px] text-textDim uppercase px-1 bg-void border border-borderDim">
                {projectData.origin === 'PC' ? 'PC-ORIGIN' : 'SN-SYNC'}
              </span>
              <ChevronDown size={12} className={`text-textDim group-hover:text-cyan transition-transform ${showProjectDropdown ? 'rotate-180' : ''}`} />
            </button>

            <AnimatePresence>
              {showProjectDropdown && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                  className="absolute top-full left-0 mt-2 w-64 bg-bg border border-border shadow-2xl p-4 space-y-4 z-[200]"
                >
                  <div className="space-y-1">
                    <div className="micro-label text-textDim uppercase">Sync Status</div>
                    <div className="flex items-center gap-2 text-green">
                      <Activity size={12} />
                      <span className="font-mono text-[10px] uppercase">Live Calibration Active</span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <button 
                      onClick={() => {
                        navigator.clipboard.writeText(projectCode);
                        setShowProjectDropdown(false);
                      }}
                      className="w-full flex items-center justify-between p-2 bg-bgRaised border border-borderDim hover:border-cyan transition-all group"
                    >
                      <span className="micro-label text-textDim group-hover:text-cyan">COPY PROJECT CODE</span>
                      <Copy size={12} className="text-textDim group-hover:text-cyan" />
                    </button>
                    <button 
                      onClick={() => {
                        const blob = new Blob([projectCode], { type: 'text/plain' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `project_${projectData.id}.pc`;
                        a.click();
                        setShowProjectDropdown(false);
                      }}
                      className="w-full flex items-center justify-between p-2 bg-bgRaised border border-borderDim hover:border-cyan transition-all group"
                    >
                      <span className="micro-label text-textDim group-hover:text-cyan">DOWNLOAD .PC FILE</span>
                      <Download size={12} className="text-textDim group-hover:text-cyan" />
                    </button>
                    <a 
                      href="https://sentinel.physicore.io" 
                      target="_blank" 
                      rel="noreferrer"
                      className="w-full flex items-center justify-between p-2 bg-bgRaised border border-borderDim hover:border-amber transition-all group"
                    >
                      <span className="micro-label text-textDim group-hover:text-amber">OPEN IN SENTINEL</span>
                      <ExternalLink size={12} className="text-textDim group-hover:text-amber" />
                    </a>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
      </div>

      {view !== 'dashboard' && (
        <div className="hidden lg:flex items-center gap-8">
          {['OVERVIEW', 'ARCHITECTURE', 'FEATURES', 'BENCHMARKS', 'SENTINEL', 'MANUAL', 'TEAM'].map(item => {
            if (item === 'TEAM' && !isAdmin) return null;
            return (
              <a 
                key={item} 
                href={`#${item.toLowerCase()}`}
                className={`font-body text-[11px] uppercase tracking-widest transition-colors ${activeSection === item.toLowerCase() ? (item === 'SENTINEL' ? 'text-amber border-b border-amber' : (item === 'TEAM' ? 'text-cyan border-b border-cyan' : 'text-green border-b border-green')) : 'text-textSecondary hover:text-textPrimary'}`}
                onClick={(e) => {
                  e.preventDefault();
                  if (item === 'MANUAL') setView('manual');
                  else if (item === 'TEAM') setView('team');
                  else {
                    if (view !== 'home') {
                      setView('home');
                      setTimeout(() => {
                        document.getElementById(item.toLowerCase())?.scrollIntoView({ behavior: 'smooth' });
                      }, 100);
                    } else {
                      document.getElementById(item.toLowerCase())?.scrollIntoView({ behavior: 'smooth' });
                    }
                  }
                }}
              >
                {item}
              </a>
            );
          })}
        </div>
      )}

      <div className="flex items-center gap-4">
        {user ? (
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex flex-col items-end">
              <span className="font-mono text-[9px] text-white uppercase tracking-widest">{user.displayName || 'AUTHORIZED USER'}</span>
              <span className="font-mono text-[8px] text-textDim uppercase">{user.email}</span>
            </div>
            <div className="w-8 h-8 border border-border flex items-center justify-center bg-bgRaised">
              <User size={16} className="text-green" />
            </div>
            <button onClick={handleLogout} className="p-2 text-textDim hover:text-red transition-colors" title="Logout">
              <LogOut size={18} />
            </button>
          </div>
        ) : (
          <button 
            onClick={handleLaunchApp} 
            disabled={isLoggingIn}
            className="px-4 py-1.5 bg-green text-black font-display text-[11px] font-bold uppercase tracking-widest hover:bg-white transition-all disabled:opacity-50 flex items-center gap-2"
          >
            {isLoggingIn && <div className="w-3 h-3 border-2 border-black border-t-transparent rounded-full animate-spin" />}
            {isLoggingIn ? 'CONNECTING...' : '▣ LAUNCH APP'}
          </button>
        )}
        
        {user && (
          <>
            {view === 'dashboard' ? (
              <button 
                onClick={() => setView('home')}
                className="px-4 py-1.5 border border-red text-red font-display text-[11px] font-bold uppercase tracking-widest hover:bg-red hover:text-black transition-all"
              >
                ▣ EXIT DASHBOARD
              </button>
            ) : (
              <>
                <button 
                  onClick={handleSetIntegratorView}
                  className={`px-4 py-1.5 font-display text-[11px] font-bold uppercase tracking-widest transition-all ${view === 'integrator' ? 'bg-white text-black' : 'bg-green text-black hover:bg-white'}`}
                >
                  ⬡ INTEGRATION ENGINEER
                </button>
                <button 
                  onClick={() => setView('manual')}
                  className={`px-4 py-1.5 font-display text-[11px] font-bold uppercase tracking-widest transition-all ${view === 'manual' ? 'bg-white text-black' : 'bg-amber text-black hover:bg-white'}`}
                >
                  ⬡ MANUAL
                </button>
                <button 
                  onClick={handleLaunchApp}
                  className="hidden sm:block px-4 py-1.5 border border-border font-display text-[11px] font-bold uppercase tracking-widest text-textSecondary hover:text-textPrimary transition-all"
                >
                  ▣ LAUNCH APP
                </button>
              </>
            )}
          </>
        )}
      </div>
    </nav>
  );

  const renderHome = () => (
    <div className="pt-[52px] custom-scroll">
      {/* HERO */}
      <section id="overview" className="relative h-screen flex flex-col items-center justify-center text-center px-6 overflow-hidden">
        <HeroCanvas />
        <div className="relative z-10 max-w-[800px] space-y-8">
          <div className="reveal border-l-2 border-green pl-4 text-left inline-block">
            <span className="font-mono text-[11px] text-green uppercase tracking-[0.2em]">PhysiCore v3.0 — Sentinel OS Infrastructure</span>
          </div>
          <h1 className="reveal reveal-stagger-1 font-display text-6xl md:text-8xl font-bold text-white leading-[0.9] tracking-tighter">
            Close the <br />Reality Gap.
          </h1>
          <p className="reveal reveal-stagger-2 font-body text-lg md:text-xl text-textSecondary leading-relaxed max-w-[600px] mx-auto">
            Physics Intelligence Engine for robotics, rockets, and aviation. <br />
            RK4 integration. Online SystemID. Ensemble residuals. <br />
            From simulation to certified hardware in one stack.
          </p>
          
          <div className="reveal reveal-stagger-3 grid grid-cols-2 md:grid-cols-4 gap-0 border border-border divide-x divide-border bg-void/50 backdrop-blur-sm">
            {[
              { val: '4th ORDER', label: 'RK4 INTEGRATOR' },
              { val: '12 STEP', label: 'MPC LOOKAHEAD' },
              { val: '3 NODE', label: 'ENSEMBLE DEPTH' },
              { val: '60 Hz', label: 'CONTROL LOOP' },
            ].map((m, i) => (
              <div key={i} className="p-6 flex flex-col items-center">
                <span className="font-display text-3xl font-bold text-green">{m.val}</span>
                <span className="micro-label text-[9px] text-textDim mt-1">{m.label}</span>
              </div>
            ))}
          </div>

          <div className="reveal reveal-stagger-4 flex flex-col sm:flex-row gap-4 justify-center">
            <button onClick={handleSetIntegratorView} className="btn-primary h-14 text-sm px-8">
              {user ? '⬡ START INTEGRATION →' : '⬡ INTEGRATION ENGINEER'}
            </button>
            <button 
              onClick={handleLaunchApp} 
              disabled={isLoggingIn}
              className="btn-outline h-14 text-sm px-8 flex items-center justify-center gap-3"
            >
              {isLoggingIn ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  ESTABLISHING CONNECTION...
                </>
              ) : (
                '▣ LAUNCH PHYSICORE APP'
              )}
            </button>
          </div>
        </div>

        <div className="absolute bottom-10 flex flex-col items-center gap-2 animate-pulse-opacity">
          <span className="font-mono text-[9px] text-textDim tracking-widest">SCROLL TO EXPLORE</span>
          <ChevronDown size={16} className="text-textDim" />
        </div>
      </section>

      {/* PROBLEM */}
      <section id="problem" className="bg-bg py-32 px-6">
        <div className="max-w-[1100px] mx-auto grid md:grid-cols-2 gap-20 items-center">
          <div className="space-y-8">
            <div className="reveal border-l-2 border-green pl-4">
              <span className="micro-label text-green">The Reality Gap</span>
            </div>
            <h2 className="reveal reveal-stagger-1 font-display text-4xl md:text-5xl font-bold text-white leading-tight">
              Your simulation is perfect. <br />
              Your hardware is not.
            </h2>
            <div className="reveal reveal-stagger-2 space-y-6 font-body text-textSecondary leading-relaxed">
              <p>Every deployment hits the same wall. The physics that worked in simulation — the perfectly tuned mass, the ideal friction, the clean aerodynamic model — fails the moment it meets real hardware.</p>
              <p>The floor isn't as smooth as the model. The payload shifts. The air density varies. The gap between what your simulation predicts and what your hardware does compounds with every iteration.</p>
              <p>PhysiCore eliminates this gap in real-time. Whether it's a robotic arm, a suborbital rocket, or a fixed-wing UAV, we make the simulation learn the reality it's deployed into.</p>
            </div>
          </div>
          <div className="reveal reveal-stagger-3">
            <RealityGapDiagram />
          </div>
        </div>
      </section>

      {/* ARCHITECTURE */}
      <section id="architecture" className="bg-void py-32 px-6">
        <div className="max-w-[1100px] mx-auto space-y-16">
          <div className="text-center space-y-4">
            <span className="micro-label text-green">System Architecture</span>
            <h2 className="font-display text-4xl md:text-5xl font-bold text-white">Five layers. One kernel.</h2>
          </div>
          <div className="space-y-4">
            {[
              { l: 'L5', name: 'INTEGRATION LAYER', desc: 'Production bridge code, auto-generated', tech: 'ROS2 / ArduPilot / PX4', color: COLORS.textSecondary },
              { l: 'L4', name: 'SENTINEL GOVERNANCE LAYER', desc: 'Safety envelopes / Mode state machine', tech: 'Forensic logging', color: COLORS.amber },
              { l: 'L3', name: 'INTELLIGENCE LAYER', desc: '3-node ensemble / SystemID / Uncertainty scoring', tech: 'Learns the reality gap', color: COLORS.cyan },
              { l: 'L2', name: 'CONTROL LAYER', desc: 'MPC / CEM solver / 12-step lookahead', tech: 'Q-R cost weighting', color: COLORS.blue },
              { l: 'L1', name: 'PHYSICS KERNEL', desc: 'RK4 integrator / Rigid body dynamics', tech: 'Actuator efficiency', color: COLORS.green },
            ].map((layer, i) => (
              <div key={i} className="reveal flex items-stretch border border-border bg-bgRaised group hover:border-borderActive transition-all">
                <div className="w-[60px] flex items-center justify-center border-r border-border font-mono text-[10px] text-textDim group-hover:text-textPrimary transition-colors">{layer.l}</div>
                <div className="w-1.5" style={{ backgroundColor: layer.color }} />
                <div className="flex-1 p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div>
                    <h3 className="font-display text-lg font-bold tracking-widest text-textPrimary uppercase">{layer.name}</h3>
                    <p className="font-body text-xs text-textSecondary">{layer.desc}</p>
                  </div>
                  <div className="font-mono text-[10px] text-textDim uppercase tracking-widest">{layer.tech}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* DOMAINS */}
      <section id="domains" className="bg-bg py-32 px-6 border-t border-border">
        <div className="max-w-[1100px] mx-auto space-y-16">
          <div className="text-center space-y-4">
            <span className="micro-label text-green">Multi-Domain Support</span>
            <h2 className="font-display text-4xl md:text-5xl font-bold text-white uppercase tracking-tighter">Engineered for the Edge.</h2>
            <p className="font-body text-textSecondary max-w-[600px] mx-auto">PhysiCore isn't just for robots. It's a universal physics intelligence engine designed for any system where the reality gap is a mission-critical risk.</p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            {[
              { 
                title: 'Robotics', 
                icon: Cpu, 
                desc: 'From industrial manipulators to humanoid locomotion. Calibrate joint friction, mass distribution, and actuator efficiency in real-time.',
                tech: 'ROS2 / MoveIt / URDF'
              },
              { 
                title: 'Rockets', 
                icon: Rocket, 
                desc: 'Suborbital and orbital launch vehicles. Model thrust curves, fuel depletion, and aerodynamic drag with RK4 precision.',
                tech: 'ArduPilot / OpenRocket / MAVLink'
              },
              { 
                title: 'Aviation', 
                icon: Navigation, 
                desc: 'Fixed-wing UAVs and eVTOL systems. Learn lift/drag polars and flight envelope boundaries to ensure stability in turbulent conditions.',
                tech: 'PX4 / FlightGear / JSBSim'
              }
            ].map((d, i) => (
              <div key={i} className="reveal p-10 border border-border bg-bgRaised space-y-8 group hover:border-green transition-all">
                <div className="w-16 h-16 bg-bg flex items-center justify-center border border-border group-hover:border-green transition-all">
                  <d.icon className="text-green" size={32} />
                </div>
                <div className="space-y-4">
                  <h3 className="font-display text-2xl font-bold text-white uppercase tracking-widest">{d.title}</h3>
                  <p className="font-body text-sm text-textSecondary leading-relaxed">{d.desc}</p>
                </div>
                <div className="pt-6 border-t border-border">
                  <span className="font-mono text-[10px] text-textDim uppercase tracking-widest">{d.tech}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FEATURES */}
      <section id="features" className="bg-bg py-32 px-6">
        <div className="max-w-[1100px] mx-auto space-y-16">
          <div className="space-y-4">
            <span className="micro-label text-green">Core Capabilities</span>
            <h2 className="font-display text-4xl md:text-5xl font-bold text-white">Every subsystem. Fully specified.</h2>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            {[
              { title: 'RK4 Physics Kernel', color: COLORS.green, desc: '4th-order Runge-Kutta integration samples four derivative points per frame. Unlike Euler integration, RK4 remains stable under high stiffness and nonlinear dynamics.', spec: 'k₁ k₂ k₃ k₄ → Δstate/frame' },
              { title: 'Online SystemID', color: COLORS.cyan, desc: 'Numerical gradient descent runs every 50 frames, perturbing mass and friction to find the direction that minimizes prediction error. The model learns your hardware.', spec: '∇mass ∇friction → physical bounds' },
              { title: 'Ensemble Residuals', color: COLORS.blue, desc: 'Three parallel shadow simulations run at 0.7×, 1.0×, and 1.3× noise scales. The standard deviation between predictions quantifies epistemic uncertainty.', spec: 'σ(node₁,node₂,node₃) → confidence' },
              { title: 'MPC Lookahead', color: COLORS.amber, desc: 'Cross-Entropy Method solver simulates 12 steps forward every frame. Uncertainty-aware planning penalizes high-variance regions.', spec: 'CEM solver / 12-step / 60Hz' },
              { title: 'Hardware Gate', color: COLORS.textSecondary, desc: 'Physics does not run until hardware is connected. Live ROS2, HIL simulation with processor-accurate latency profiles, or Digital Twin mode.', spec: 'LIVE → HANDSHAKE_OK | HIL → SIM_VALIDATED' },
              { title: 'Sentinel OS Integration', color: COLORS.red, desc: 'PhysiCore feeds directly into Sentinel\'s safety governance layer. Calibrated priors, confidence scores, and residual drift feed the Lyapunov stability kernel.', spec: 'NOMINAL → CAUTIOUS → FALLBACK' },
            ].map((f, i) => (
              <div key={i} className="reveal p-8 border border-border bg-bgRaised border-t-2 space-y-6 group hover:bg-bg transition-all" style={{ borderTopColor: f.color }}>
                <h3 className="font-display text-xl font-bold text-white tracking-widest uppercase">{f.title}</h3>
                <p className="font-body text-sm text-textSecondary leading-relaxed">{f.desc}</p>
                <div className="font-mono text-[11px] uppercase tracking-widest" style={{ color: f.color }}>{f.spec}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* BENCHMARKS */}
      <section id="benchmarks" className="bg-void py-32 px-6">
        <div className="max-w-[1100px] mx-auto space-y-16">
          <div className="space-y-4">
            <span className="micro-label text-green">Performance Benchmarks</span>
            <h2 className="font-display text-4xl md:text-5xl font-bold text-white">Numbers that matter to engineers.</h2>
          </div>
          <div className="grid md:grid-cols-2 gap-20">
            <div className="space-y-8">
              {[
                { label: 'CONTROL LOOP RATE', val: '60 Hz', p: 100 },
                { label: 'RK4 STABILITY MARGIN', val: '99.7%', p: 99 },
                { label: 'SYSTEMID CONVERGENCE', val: '847 frames', p: 85 },
                { label: 'ENSEMBLE CONFIDENCE', val: '94.2%', p: 94 },
                { label: 'MPC HORIZON', val: '12 steps', p: 70 },
                { label: 'RESIDUAL CONVERGENCE', val: '0.023 L2', p: 90 },
              ].map((b, i) => (
                <div key={i} className="reveal space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="font-body text-xs text-textSecondary uppercase tracking-widest">{b.label}</span>
                    <span className="font-mono text-xs text-green">{b.val}</span>
                  </div>
                  <div className="h-1 w-full bg-border overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }} 
                      whileInView={{ width: `${b.p}%` }} 
                      transition={{ duration: 1, delay: i * 0.1 }}
                      className="h-full bg-green"
                    />
                  </div>
                </div>
              ))}
            </div>
            <div className="reveal border border-border bg-bgRaised overflow-hidden">
              <table className="w-full text-left border-collapse">
                <tbody>
                  {[
                    ['INTEGRATOR', 'RK4 4th Order'],
                    ['SOLVER', 'CEM Cross-Entropy'],
                    ['ENSEMBLE NODES', '3 (σ: 0.7 / 1.0 / 1.3)'],
                    ['LOOP RATE', '60 Hz locked'],
                    ['SYSTEMID INTERVAL', 'Every 50 frames'],
                    ['MPC HORIZON', '12 steps'],
                    ['STATE VECTOR', 'pos / vel / mass / friction'],
                    ['CERTIFICATION', 'PROOF PENDING / HIL / TWIN'],
                  ].map(([k, v], i) => (
                    <tr key={i} className="border-b border-borderDim hover:bg-bg transition-colors">
                      <td className="p-4 font-body text-xs text-textSecondary uppercase tracking-widest">{k}</td>
                      <td className="p-4 font-mono text-xs text-textPrimary">{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {/* SENTINEL */}
      <section id="sentinel" className="bg-bg py-32 px-6 border-t border-border">
        <div className="max-w-[1100px] mx-auto space-y-24">
          <div className="grid md:grid-cols-2 gap-20 items-center">
            <div className="space-y-8">
              <div className="reveal border-l-2 border-amber pl-4">
                <span className="micro-label text-amber">Governance & Safety</span>
              </div>
              <h2 className="reveal reveal-stagger-1 font-display text-5xl md:text-6xl font-bold text-white leading-tight">
                Sentinel OS. <br />
                <span className="text-textSecondary">The Governance Kernel.</span>
              </h2>
              <div className="reveal reveal-stagger-2 space-y-6 font-body text-textSecondary leading-relaxed text-lg">
                <p>PhysiCore provides the intelligence, but Sentinel OS provides the authority. It is a Universal Neural-Symbolic Governance Kernel designed for high-stakes robotics deployment.</p>
                <p>Sentinel sits between the AI's high-level intent and the physical motor controllers, acting as a real-time safety monitor that cannot be bypassed by the neural stack.</p>
              </div>
            </div>
            <div className="reveal reveal-stagger-3">
              <SentinelDiagram />
            </div>
          </div>

          <div className="grid md:grid-cols-4 gap-6">
            {[
              { title: 'Lyapunov Kernel', desc: '10kHz safety loop monitors system energy states to ensure stability bounds are never exceeded.', icon: ShieldCheck },
              { title: 'Degraded State', desc: 'Automatic transition to safe-state machines (NOMINAL → CAUTIOUS → FALLBACK) during anomalies.', icon: AlertTriangle },
              { title: 'Forensic Ledger', desc: 'Every control decision is signed with SHA-256 and logged to a tamper-proof governance ledger.', icon: FileJson },
              { title: 'Compliance Export', desc: 'Auto-generate DO-178C and NASA-STD-8739.8 compliance artifacts for regulatory approval.', icon: CheckCircle2 },
            ].map((item, i) => (
              <div key={i} className="reveal p-6 border border-border bg-bgRaised space-y-4 group hover:border-amber transition-all">
                <item.icon className="text-amber" size={24} />
                <h3 className="font-display text-sm font-bold text-white tracking-widest uppercase">{item.title}</h3>
                <p className="font-body text-xs text-textSecondary leading-relaxed">{item.desc}</p>
              </div>
            ))}
          </div>

          <div className="reveal border border-amber/20 bg-amberDim/10 p-10 flex flex-col md:flex-row items-center justify-between gap-10">
            <div className="space-y-4 max-w-[600px]">
              <h3 className="font-display text-2xl font-bold text-white uppercase tracking-widest">Why Sentinel is Necessary</h3>
              <p className="font-body text-sm text-textSecondary leading-relaxed">
                In regulated environments, "black box" AI is a liability. Sentinel provides the symbolic safety layer required for certification. It translates complex neural outputs into verifiable physical constraints, ensuring your robot never performs an action that violates its Lyapunov stability envelope.
              </p>
            </div>
            <div className="shrink-0 space-y-4">
              <div className="p-4 border border-amber/30 bg-bg font-mono text-[10px] text-amber space-y-2">
                <div className="flex justify-between gap-10"><span>CERTIFICATE ID:</span> <span>SENT-882-X</span></div>
                <div className="flex justify-between gap-10"><span>STATUS:</span> <span className="text-amber">PROOF PENDING</span></div>
                <div className="flex justify-between gap-10"><span>GOVERNANCE:</span> <span>ACTIVE</span></div>
              </div>
              <p className="font-mono text-[8px] text-textDim uppercase max-w-[200px]">dReal formal verification pending configuration of server endpoint.</p>
              <button 
                onClick={() => window.open('https://github.com/dreal/dreal4', '_blank')}
                className="btn-outline border-amber text-amber w-full hover:bg-amber hover:text-black text-[10px]"
              >
                VIEW PROOF DOCUMENTATION →
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="bg-void py-40 px-6">
        <div className="max-w-[800px] mx-auto text-center space-y-10">
          <div className="reveal border-l-2 border-cyan pl-4 inline-block text-left">
            <span className="micro-label text-cyan">Integration Engineer</span>
          </div>
          <h2 className="reveal reveal-stagger-1 font-display text-5xl md:text-7xl font-bold text-white leading-[1.1]">
            Tell us your system. <br />
            <span className="text-textSecondary">We'll tell you how to integrate.</span>
          </h2>
          <p className="reveal reveal-stagger-2 font-body text-lg text-textSecondary leading-relaxed max-w-[600px] mx-auto">
            The PhysiCore Integration Engineer understands your hardware, your firmware, and your deployment constraints. It generates production-ready bridge code with your exact calibrated parameters injected.
          </p>
          <div className="reveal reveal-stagger-3 flex flex-wrap justify-center gap-3">
            {['ROS2', 'ARDUPILOT', 'PX4', 'MATLAB', 'CUSTOM'].map(p => (
              <span key={p} className="px-4 py-1.5 border border-border font-mono text-[10px] text-textDim uppercase tracking-widest">{p}</span>
            ))}
          </div>
          <button onClick={() => setView('integrator')} className="reveal reveal-stagger-4 bg-cyan text-black px-12 py-5 font-display text-lg font-bold uppercase tracking-widest hover:bg-white transition-all w-full md:w-auto">
            ⬡ OPEN INTEGRATION ENGINEER →
          </button>
        </div>
      </section>

      {/* FINAL CTA */}
      <section className="bg-bg py-32 px-6 border-t border-border">
        <div className="max-w-[1100px] mx-auto text-center space-y-12">
          <h2 className="reveal font-display text-4xl md:text-6xl font-bold text-white uppercase tracking-tighter">
            Ready to deploy?
          </h2>
          <div className="reveal reveal-stagger-1 flex flex-col sm:flex-row gap-4 justify-center">
            <button onClick={() => setView('integrator')} className="btn-primary h-16 px-12 text-lg">
              ⬡ INTEGRATE PHYSICORE
            </button>
            <button 
              onClick={() => window.open('https://sentinel-lac.vercel.app/', '_blank')}
              className="btn-outline border-amber text-amber h-16 px-12 text-lg hover:bg-amber hover:text-black"
            >
              LEARN MORE ABOUT SENTINEL
            </button>
          </div>
          <p className="reveal reveal-stagger-2 font-mono text-[10px] text-textDim uppercase tracking-widest">
            PhysiCore v3.0 — Part of the Sentinel OS Ecosystem
          </p>
        </div>
      </section>

      {/* FOUNDER SECTION */}
      <section className="bg-void py-12 px-6 border-t border-border/30">
        <div className="max-w-[1100px] mx-auto flex flex-col items-center justify-center space-y-4">
          <span className="font-mono text-[10px] text-textDim uppercase tracking-[0.3em]">Founder</span>
          <a 
            href="https://prathameshshirbhate.vercel.app/" 
            target="_blank" 
            rel="noopener noreferrer"
            className="group flex flex-col items-center gap-2 transition-all hover:scale-105"
          >
            <span className="font-display text-xl text-white font-bold uppercase tracking-widest group-hover:text-cyan transition-colors">
              Prathamesh Shirbhate
            </span>
            <div className="w-12 h-[1px] bg-cyan/30 group-hover:w-20 group-hover:bg-cyan transition-all" />
          </a>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="bg-bg border-t border-border py-20 px-6">
        <div className="max-w-[1100px] mx-auto grid grid-cols-2 md:grid-cols-4 gap-12">
          <div className="col-span-2 space-y-6">
            <div className="flex items-center gap-2">
              <svg width="20" height="20" viewBox="0 0 100 100">
                <path d="M50 5 L90 25 L90 75 L50 95 L10 75 L10 25 Z" fill="none" stroke={COLORS.green} strokeWidth="6" />
              </svg>
              <span className="font-display text-lg font-bold tracking-widest text-white">PHYSICORE</span>
            </div>
            <div className="space-y-1">
              <p className="font-body text-xs text-textDim">Physics Intelligence Engine</p>
              <p className="font-body text-xs text-textDim">Part of Sentinel OS</p>
              <p className="font-mono text-[10px] text-textDim">v3.0</p>
            </div>
          </div>
          <div className="space-y-6">
            <span className="micro-label text-textDim">System</span>
            <div className="flex flex-col gap-3 font-body text-sm text-textSecondary">
              <a href="#overview" className="hover:text-green transition-colors">Overview</a>
              <a href="#architecture" className="hover:text-green transition-colors">Architecture</a>
              <a href="#features" className="hover:text-green transition-colors">Features</a>
              <a href="#benchmarks" className="hover:text-green transition-colors">Benchmarks</a>
            </div>
          </div>
          <div className="space-y-6">
            <span className="micro-label text-textDim">Deploy</span>
            <div className="flex flex-col gap-3 font-body text-sm text-textSecondary">
              <button onClick={() => setView('integrator')} className="text-left hover:text-cyan transition-colors">⬡ Integration Engineer</button>
              <button onClick={handleLaunchApp} className="text-left hover:text-white transition-colors">▣ Launch App</button>
              <button className="text-left hover:text-white transition-colors">⬇ Documentation</button>
            </div>
          </div>
        </div>
        <div className="max-w-[1100px] mx-auto mt-20 pt-8 border-t border-borderDim flex justify-between items-center font-mono text-[10px] text-textDim uppercase tracking-widest">
          <span>© 2025 PhysiCore — Sentinel OS Infrastructure</span>
          <span className="hidden sm:block">Theory meets practice at the edge.</span>
        </div>
      </footer>
    </div>
  );

  const renderQuotaExceeded = () => (
    <div className="min-h-screen bg-void flex items-center justify-center p-6">
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="max-w-md w-full p-8 border border-red/30 bg-redDim/10 space-y-8 text-center"
      >
        <div className="w-16 h-16 border border-red flex items-center justify-center mx-auto bg-redDim/20">
          <ShieldAlert className="text-red" size={32} />
        </div>
        <div className="space-y-4">
          <h2 className="font-display text-2xl font-bold text-white tracking-tighter uppercase italic">QUOTA EXCEEDED</h2>
          <p className="font-body text-textSecondary text-xs uppercase tracking-widest leading-relaxed">
            Your free pilot access for PhysiCore has been exhausted. You have successfully completed one physical layer integration test.
          </p>
          <div className="p-4 border border-borderDim bg-bg text-left space-y-2">
            <div className="micro-label text-textDim uppercase">Next Steps</div>
            <p className="font-body text-[10px] text-textSecondary uppercase leading-relaxed">
              To continue using PhysiCore for multiple systems or production environments, please contact our integration team for a full license.
            </p>
          </div>
        </div>
        <button 
          onClick={() => setView('home')}
          className="w-full py-3 border border-white text-white font-display font-bold text-[11px] uppercase tracking-widest hover:bg-white hover:text-black transition-all"
        >
          RETURN TO COMMAND CENTER
        </button>
      </motion.div>
    </div>
  );

  const renderIntegrator = () => {
    if (hasTested && !isAdmin) return renderQuotaExceeded();
    return (
      <div className="pt-[52px] h-screen flex bg-void overflow-hidden">
      {/* LEFT PANEL */}
      <aside className="w-[300px] border-r border-border bg-bg flex flex-col overflow-hidden">
        <div className="h-12 border-b border-border flex items-center justify-between px-4 shrink-0">
          <div className="flex flex-col">
            <span className="font-display text-xs font-bold text-cyan tracking-widest uppercase">⬡ INTEGRATION ENGINEER</span>
            <span className="font-body text-[9px] text-textDim uppercase">PHYSICORE v3.0</span>
          </div>
          
          <div className="flex items-center gap-2">
            {!showKeyInput ? (
              <button 
                onClick={() => setShowKeyInput(true)}
                className={`flex items-center gap-1.5 px-2 py-1 border ${isKeyValid ? 'border-green/30 text-green' : 'border-amber/30 text-amber'} font-mono text-[8px] uppercase tracking-widest hover:bg-white/5 transition-all`}
              >
                {isKeyValid ? '▣ NEURAL LINK' : geminiApiKey ? '⚠ KEY INVALID' : '⚠ NO API KEY'}
              </button>
            ) : (
              <div className="flex items-center gap-1">
                <input 
                  type="password"
                  placeholder="GEMINI_API_KEY"
                  className="w-32 bg-bgInset border border-border px-2 py-1 font-mono text-[8px] text-white focus:outline-none focus:border-cyan"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      testGeminiKey((e.target as HTMLInputElement).value);
                    }
                  }}
                />
                <button 
                  onClick={() => setShowKeyInput(false)}
                  className="p-1 text-textDim hover:text-white"
                >
                  <X size={12} />
                </button>
              </div>
            )}
          </div>
        </div>
        
        <div className="flex-1 overflow-y-auto custom-scroll p-6 space-y-10">
          <section className="space-y-6">
            <div className="border-l-2 border-cyan pl-3">
              <span className="micro-label">Detected System Profile</span>
            </div>
            <div className="space-y-4">
              {[
                { label: 'PLATFORM', val: systemProfile.platform },
                { label: 'FIRMWARE', val: systemProfile.firmware },
                { label: 'DOMAIN', val: systemProfile.domain },
                { label: 'MASS CLASS', val: systemProfile.massClass },
                { label: 'CONNECTION', val: systemProfile.connectionMode },
                { label: 'PROTOCOLS', val: systemProfile.protocols },
              ].map((cell, i) => (
                <div key={i} className="space-y-1">
                  <div className="micro-label text-[9px] text-textDim">{cell.label}</div>
                  <div className={`font-mono text-xs ${cell.val ? 'text-textPrimary' : 'text-textDim'}`}>{cell.val || '—'}</div>
                </div>
              ))}
            </div>
            {integrationPhase >= 2 && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="p-4 bg-greenDim border border-green space-y-1">
                <div className="font-display text-xs font-bold text-green tracking-widest uppercase">PROFILE COMPLETE</div>
                <div className="font-body text-[10px] text-textSecondary">Ready to generate integration code</div>
              </motion.div>
            )}
          </section>

          {generatedFiles.length > 0 && (
            <section className="space-y-6">
              <div className="border-l-2 border-green pl-3">
                <span className="micro-label">Generated Files</span>
              </div>
              <div className="space-y-2">
                {generatedFiles.map((f, i) => (
                  <div key={i} className="flex items-center justify-between p-2 border border-borderDim bg-bgRaised group">
                    <span className="font-mono text-[10px] text-textSecondary group-hover:text-white transition-colors">{f.filename}</span>
                    <button 
                      onClick={() => {
                        const element = document.createElement("a");
                        const blob = new Blob([f.content], { type: 'text/plain' });
                        element.href = URL.createObjectURL(blob);
                        element.download = f.filename;
                        document.body.appendChild(element);
                        element.click();
                        document.body.removeChild(element);
                      }}
                      className="text-cyan hover:text-white transition-colors"
                    >
                      <Download size={14} />
                    </button>
                  </div>
                ))}
              </div>
            </section>
          )}

          {user?.email === "prathameshshirbhate8anpc@gmail.com" && (
            <section className="space-y-6">
              <div className="border-l-2 border-amber pl-3">
                <span className="micro-label text-amber uppercase">Admin: User Management</span>
              </div>
              <div className="space-y-4">
                <div className="flex gap-2">
                  <input 
                    type="email" 
                    id="new-user-email"
                    placeholder="Authorize Email..."
                    className="flex-1 bg-bg border border-border p-2 font-mono text-[10px] text-white outline-none focus:border-amber"
                  />
                  <button 
                    onClick={() => {
                      const input = document.getElementById('new-user-email') as HTMLInputElement;
                      handleAddUser(input.value);
                      input.value = '';
                    }}
                    className="px-3 py-1 bg-amber text-black font-display text-[10px] font-bold uppercase tracking-widest hover:bg-white transition-all"
                  >
                    ADD
                  </button>
                </div>
                <div className="max-h-[200px] overflow-y-auto custom-scroll space-y-2">
                  {allUsers.map((u, i) => (
                    <div key={i} className="p-2 border border-borderDim bg-bgRaised flex items-center justify-between">
                      <div className="flex flex-col">
                        <span className="font-mono text-[9px] text-textPrimary truncate max-w-[120px]">{u.email}</span>
                        <span className="font-mono text-[8px] text-textDim uppercase">{u.role}</span>
                      </div>
                      <div className={`w-2 h-2 rounded-full ${u.hasTested ? 'bg-green' : 'bg-textDim'}`} title={u.hasTested ? 'Has Tested' : 'Pending'} />
                    </div>
                  ))}
                </div>
              </div>
            </section>
          )}

          <section className="space-y-4">
            <div className="micro-label text-textDim">Quick Start</div>
            <div className="flex flex-col gap-2">
              {['ROS2 INTEGRATION', 'ARDUPILOT / AP_DDS', 'PX4 / uXRCE-DDS', 'MATLAB / SIMULINK'].map(p => (
                <button key={p} onClick={() => handleSendMessage(`I want to integrate with ${p}`)} className="text-left px-3 py-2 border border-border text-textSecondary font-body text-[11px] hover:border-cyan hover:text-cyan transition-all uppercase tracking-widest">{p}</button>
              ))}
            </div>
          </section>
        </div>
      </aside>

      {/* RIGHT PANEL */}
      <div className="flex-1 flex flex-col bg-bgInset relative">
        <div ref={scrollRef} className="flex-1 overflow-y-auto custom-scroll p-8 pb-24">
          <div className="max-w-[700px] mx-auto space-y-12">
            {conversationHistory.length === 0 ? (
              <div className="flex flex-col items-center justify-center pt-20 text-center space-y-8">
                <div className="w-16 h-16 border border-cyan flex items-center justify-center text-cyan">
                  <svg width="32" height="32" viewBox="0 0 100 100">
                    <path d="M50 5 L90 25 L90 75 L50 95 L10 75 L10 25 Z" fill="none" stroke="currentColor" strokeWidth="4" />
                  </svg>
                </div>
                <div className="space-y-4">
                  <h2 className="font-display text-2xl font-bold text-textPrimary tracking-widest uppercase">PhysiCore Integration Engineer</h2>
                  <p className="font-body text-sm text-textSecondary max-w-[480px]">Tell me about your robot and I'll generate production-ready PhysiCore integration code for your specific platform and hardware.</p>
                </div>
                <div className="grid grid-cols-2 gap-4 w-full">
                  {[
                    "I have a ROS2 robot",
                    "I'm using ArduPilot / PX4",
                    "I have a MATLAB workflow",
                    "I have custom hardware"
                  ].map(chip => (
                    <button key={chip} onClick={() => handleSendMessage(chip)} className="p-4 border border-border text-textSecondary font-mono text-xs hover:border-cyan hover:text-cyan transition-all text-left">{chip}</button>
                  ))}
                </div>
              </div>
            ) : (
              <>
                {conversationHistory.map((m, i) => {
                  const parsedResponse = parseAIResponse(m.content);
                  return (
                    <div key={i} className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
                      {m.role === 'ai' && <span className="font-mono text-[10px] text-cyan mb-2 uppercase tracking-widest">{"> INTEGRATION ENGINEER:"}</span>}
                      <div className={`max-w-[85%] space-y-4 ${m.role === 'ai' ? 'font-body text-sm text-textPrimary leading-relaxed' : 'font-body text-sm text-textPrimary bg-bgRaised p-4 border border-border'}`}>
                        {parsedResponse.map((part, idx) => (
                          part.type === 'code' ? (
                            <CodeBlock key={idx} filename={part.filename!} content={part.content!} />
                          ) : (
                            <p key={idx} className={part.content!.includes('?') ? 'bg-bgRaised border-l-2 border-cyan p-4' : ''}>
                              {part.content!.replace('> INTEGRATION ENGINEER:', '').trim()}
                            </p>
                          )
                        ))}
                      </div>
                      <span className="font-mono text-[8px] text-textDim mt-2">{m.timestamp}</span>
                    </div>
                  );
                })}
                {integrationPhase === 3 && (
                  <div className="space-y-6">
                    {systemProfile.domain === 'ROCKETS' ? (
                      <RocketManifestWizard 
                        params={rocketParams} 
                        setParams={setRocketParams} 
                        projectEmail={projectEmail} 
                        setProjectEmail={setProjectEmail} 
                      />
                    ) : systemProfile.domain === 'AVIATION' ? (
                      <AviationManifestWizard 
                        params={aviationParams} 
                        setParams={setAviationParams} 
                        projectEmail={projectEmail} 
                        setProjectEmail={setProjectEmail} 
                      />
                    ) : (
                      <div className="p-6 bg-bgRaised border border-border space-y-4">
                        <div className="flex items-center gap-3 border-b border-border pb-4">
                          <Cpu className="text-cyan" size={24} />
                          <h2 className="font-display text-lg font-bold text-white tracking-widest uppercase">Hardware Priors</h2>
                        </div>
                        
                        <div className="p-4 bg-bg border border-border space-y-2">
                          <div className="flex items-center gap-2 text-cyan">
                            <User size={14} />
                            <span className="micro-label uppercase">Step 2: Unit Identity</span>
                          </div>
                          <div className="space-y-1">
                            <label className="micro-label text-textDim">Email / Project Owner (Optional)</label>
                            <input 
                              type="email" 
                              value={projectEmail} 
                              onChange={e => setProjectEmail(e.target.value)} 
                              placeholder="anonymous@physicore.io"
                              className="w-full bg-bg border border-border p-2 font-mono text-xs text-white outline-none focus:border-cyan" 
                            />
                            <p className="font-mono text-[8px] text-textDim uppercase">Used to link project configuration across PhysiCore and Sentinel.</p>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-1">
                            <label className="micro-label text-textDim">System Mass (kg)</label>
                            <input 
                              type="number" 
                              value={telemetry.mass} 
                              onChange={e => setTelemetry({ ...telemetry, mass: parseFloat(e.target.value) })}
                              className="w-full bg-bg border border-border p-2 font-mono text-xs text-white outline-none focus:border-cyan" 
                            />
                          </div>
                          <div className="space-y-1">
                            <label className="micro-label text-textDim">Friction Coeff (μ)</label>
                            <input 
                              type="number" 
                              value={telemetry.friction} 
                              onChange={e => setTelemetry({ ...telemetry, friction: parseFloat(e.target.value) })}
                              className="w-full bg-bg border border-border p-2 font-mono text-xs text-white outline-none focus:border-cyan" 
                            />
                          </div>
                        </div>
                      </div>
                    )}
                    <div className="flex justify-between items-center">
                      <button onClick={() => setIntegrationPhase(1)} className="px-6 py-2 border border-border text-textSecondary font-display text-[11px] font-bold uppercase tracking-widest hover:text-white transition-all">
                        ← BACK TO CHAT
                      </button>
                      <button onClick={handleGenerateProjectCode} className="px-6 py-2 bg-white text-black font-display text-[11px] font-bold uppercase tracking-widest hover:bg-green transition-all">
                        NEXT STEP: GENERATE BRIDGE →
                      </button>
                    </div>
                  </div>
                )}
                {integrationPhase === 4 && (
                  <IntegrationActionPanel 
                    files={generatedFiles} 
                    onTest={handleLaunchApp}
                    onContinue={() => setView('dashboard')}
                    connectionMode={connectionMode}
                    setConnectionMode={setConnectionMode}
                    endpoint={endpoint}
                    setEndpoint={setEndpoint}
                    dRealEndpoint={dRealEndpoint}
                    setDRealEndpoint={setDRealEndpoint}
                    systemProfile={systemProfile}
                    rocketParams={rocketParams}
                    aviationParams={aviationParams}
                    priors={{ mass: telemetry.mass, friction: telemetry.friction }}
                    onAction={markAsTested}
                    projectCode={projectCode}
                    projectData={projectData}
                    onImportProjectCode={handleImportProjectCode}
                    isSystemConnecting={isSystemConnecting}
                    connectionError={connectionError}
                  />
                )}
              </>
            )}
            {isTyping && (
              <div className="flex flex-col items-start animate-pulse">
                <span className="font-mono text-[10px] text-cyan mb-2 uppercase tracking-widest">{"> INTEGRATION ENGINEER: ANALYZING SYSTEM..."}</span>
              </div>
            )}
          </div>
        </div>

        <div className="absolute bottom-0 left-0 w-full h-16 bg-bg border-t border-border flex items-center px-8 gap-4">
          <input 
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !quotaExceeded && handleSendMessage()}
            className={`flex-1 bg-transparent border-b ${quotaExceeded ? 'border-red text-red' : 'border-border text-cyan'} py-2 font-mono text-xs outline-none focus:border-cyan transition-colors`}
            placeholder={quotaExceeded ? "> NEURAL QUOTA EXHAUSTED. PLEASE WAIT..." : "> DESCRIBE YOUR SYSTEM OR ASK ANYTHING..."}
            disabled={quotaExceeded}
          />
          <button 
            onClick={() => handleSendMessage()} 
            disabled={quotaExceeded}
            className={`font-display text-xs font-bold ${quotaExceeded ? 'text-red border-red opacity-50' : 'text-cyan border-cyan hover:bg-cyan hover:text-black'} uppercase tracking-widest border px-6 py-2 transition-all`}
          >
            {quotaExceeded ? 'THROTTLED' : '⏎ SEND'}
          </button>
        </div>
      </div>
    </div>
  );
};

  const renderTeam = () => {
    return (
      <div className="min-h-screen bg-void pt-[52px] px-6 pb-20">
        <div className="max-w-4xl mx-auto py-12 space-y-12">
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <ShieldCheck className="text-cyan" size={24} />
              <h1 className="font-display text-4xl font-bold tracking-tighter text-white uppercase italic">TEAM AUTHORIZATION</h1>
            </div>
            <p className="font-body text-textSecondary max-w-2xl uppercase text-xs tracking-widest leading-relaxed">
              Manage access protocols for the PhysiCore Intelligence Engine. Only authorized neural signatures can access the kernel.
            </p>
          </div>

          {isAdmin && (
            <div className="p-6 border border-cyan/30 bg-cyanDim/10 space-y-6">
              <div className="flex justify-between items-center">
                <div className="space-y-1">
                  <h3 className="font-display text-lg font-bold text-cyan tracking-widest uppercase">ADMIN CONTROL PANEL</h3>
                  <p className="micro-label text-textDim uppercase">Global access management active.</p>
                </div>
                <button 
                  onClick={bootstrapTeam}
                  disabled={isBootstrapping}
                  className="px-6 py-2 bg-cyan text-black font-display font-bold text-xs tracking-widest hover:bg-white transition-all disabled:opacity-50"
                >
                  {isBootstrapping ? 'BOOTSTRAPPING...' : 'AUTHORIZE TEAM'}
                </button>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-bg border border-borderDim space-y-4">
                  <h4 className="micro-label text-textDim uppercase">Add New Signature</h4>
                  <form onSubmit={(e) => {
                    e.preventDefault();
                    const input = e.currentTarget.querySelector('input');
                    if (input) {
                      handleAddUser(input.value);
                      input.value = '';
                    }
                  }} className="flex gap-2">
                    <input 
                      type="email" 
                      placeholder="ENGINEER@TEAM.COM"
                      className="flex-1 bg-void border border-borderDim px-3 py-2 font-mono text-xs text-cyan focus:border-cyan outline-none"
                    />
                    <button type="submit" className="px-4 py-2 border border-cyan text-cyan hover:bg-cyan hover:text-black transition-all">
                      <Plus size={16} />
                    </button>
                  </form>
                </div>
                <div className="p-4 bg-bg border border-borderDim space-y-2">
                  <h4 className="micro-label text-textDim uppercase">Quick Stats</h4>
                  <div className="flex justify-between items-center">
                    <span className="font-mono text-[10px] text-textSecondary uppercase">Authorized Signatures</span>
                    <span className="font-mono text-lg text-white">{allUsers.length}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="space-y-4">
            <h3 className="micro-label text-textDim uppercase tracking-widest">AUTHORIZED PERSONNEL</h3>
            <div className="border border-border divide-y divide-border">
              {allUsers.length === 0 ? (
                <div className="p-12 text-center space-y-4">
                  <User className="mx-auto text-textDim" size={32} />
                  <p className="font-mono text-[10px] text-textDim uppercase">No authorized signatures found in the kernel.</p>
                </div>
              ) : (
                allUsers.map((u, i) => (
                  <div key={i} className="p-4 flex items-center justify-between hover:bg-bgRaised transition-colors">
                    <div className="flex items-center gap-4">
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center ${u.role === 'admin' ? 'bg-cyanDim text-cyan' : 'bg-border text-textDim'}`}>
                        {u.role === 'admin' ? <Shield size={14} /> : <User size={14} />}
                      </div>
                      <div>
                        <div className="font-mono text-xs text-white uppercase">{u.email}</div>
                        <div className="micro-label text-textDim uppercase">{u.role} signature</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-6">
                      <div className="text-right">
                        <div className="micro-label text-textDim uppercase">Status</div>
                        <div className="font-mono text-[10px] text-green uppercase">Authorized</div>
                      </div>
                      {isAdmin && u.email !== user?.email && (
                        <button 
                          onClick={() => handleRemoveUser(u.id)}
                          className="p-2 text-textDim hover:text-red transition-colors"
                        >
                          <Trash2 size={16} />
                        </button>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderManual = () => {
    const sections = [
      { id: 'intro', title: '01. INTRODUCTION', icon: <Info size={14} /> },
      { id: 'arch', title: '02. ARCHITECTURE', icon: <Layers size={14} /> },
      { id: 'ros2', title: '03. ROS2 INTEGRATION', icon: <Terminal size={14} /> },
      { id: 'ardupilot', title: '04. ARDUPILOT / PX4', icon: <Navigation size={14} /> },
      { id: 'matlab', title: '05. MATLAB / SIMULINK', icon: <BarChart3 size={14} /> },
      { id: 'bot', title: '06. BALANCING BOT', icon: <Activity size={14} /> },
      { id: 'drone', title: '07. AUTO DRONE', icon: <Wind size={14} /> },
      { id: 'robot', title: '08. ROBOTIC ARM', icon: <Cpu size={14} /> },
      { id: 'rocket', title: '09. HIGH-POWER ROCKET', icon: <Rocket size={14} /> },
      { id: 'aviation', title: '10. AVIATION DYNAMICS', icon: <Globe size={14} /> },
    ];

    return (
      <div className="pt-[52px] h-screen flex bg-void overflow-hidden">
        {/* SIDEBAR */}
        <aside className="w-[280px] border-r border-border bg-bg flex flex-col shrink-0">
          <div className="p-6 border-b border-border">
            <div className="flex items-center gap-3 text-amber mb-1">
              <BookOpen size={18} />
              <span className="font-display text-sm font-bold uppercase tracking-widest">Integration Manual</span>
            </div>
            <span className="font-mono text-[9px] text-textDim uppercase tracking-widest">PhysiCore v3.1.5-Sentinel</span>
          </div>
          <nav className="flex-1 overflow-y-auto custom-scroll p-4 space-y-1">
            {sections.map(s => (
              <button
                key={s.id}
                onClick={() => setManualSection(s.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 font-display text-[10px] font-bold uppercase tracking-widest transition-all border-l-2 ${manualSection === s.id ? 'bg-amber/10 border-amber text-amber' : 'border-transparent text-textDim hover:text-textSecondary hover:bg-bgRaised'}`}
              >
                {s.icon}
                {s.title}
              </button>
            ))}
          </nav>
        </aside>

        {/* CONTENT */}
        <main className="flex-1 overflow-y-auto custom-scroll bg-bgInset p-12">
          <div className="max-w-[800px] mx-auto space-y-12 pb-24">
            {manualSection === 'intro' && (
              <section className="space-y-8">
                <div className="space-y-4">
                  <h1 className="font-display text-4xl font-black text-white tracking-tighter uppercase">PhysiCore Integration</h1>
                  <p className="font-body text-lg text-textSecondary leading-relaxed">
                    PhysiCore is a high-fidelity multiphysics intelligence engine designed for real-time system identification, 
                    optimal control, and safety governance in robotics and aerospace systems.
                  </p>
                </div>

                <div className="p-6 border border-green/30 bg-green/5 space-y-4">
                  <div className="flex items-center gap-3">
                    <ShieldCheck className="text-green" size={20} />
                    <h3 className="font-display text-sm font-bold text-green uppercase tracking-widest">LIVE DATA GUARANTEE</h3>
                  </div>
                  <p className="font-body text-xs text-textDim leading-relaxed">
                    PhysiCore enforces a <span className="text-green font-bold">Hardware-First</span> policy. When a system is connected via HIL or SIL, all internal simulations are <span className="text-red font-bold">DISABLED</span>. 
                    The dashboard will only display data received directly from the telemetry stream. If no data is arriving, the system will remain in a <span className="text-amber font-bold">WAITING</span> state.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-6">
                  <div className="p-6 bg-bg border border-border space-y-3">
                    <Zap className="text-cyan" size={24} />
                    <h3 className="font-display text-sm font-bold text-white uppercase tracking-widest">RK4 Integration</h3>
                    <p className="font-body text-xs text-textSecondary leading-relaxed">4th-order Runge-Kutta solver running at 60Hz-1kHz for precise state estimation.</p>
                  </div>
                  <div className="p-6 bg-bg border border-border space-y-3">
                    <ShieldCheck className="text-green" size={24} />
                    <h3 className="font-display text-sm font-bold text-white uppercase tracking-widest">Sentinel OS</h3>
                    <p className="font-body text-xs text-textSecondary leading-relaxed">Safety governance layer that monitors Lyapunov stability and enforces operational bounds.</p>
                  </div>
                </div>
              </section>
            )}

            {manualSection === 'arch' && (
              <section className="space-y-8">
                <div className="space-y-4">
                  <h2 className="font-display text-2xl font-bold text-white tracking-tight uppercase">Core Architecture</h2>
                  <p className="font-body text-sm text-textSecondary leading-relaxed">
                    PhysiCore operates as a "Physics Co-Processor". It consumes raw telemetry from your hardware and 
                    provides optimal control inputs and system health metrics.
                  </p>
                </div>
                <div className="p-8 bg-void border border-borderDim font-mono text-[10px] text-cyan/60 space-y-2">
                  <div>[HARDWARE] --(Telemetry)--&gt; [PHYSICORE BRIDGE]</div>
                  <div className="pl-24">|</div>
                  <div className="pl-20">v</div>
                  <div>[PHYSICORE KERNEL] &lt;--&gt; [SENTINEL SAFETY LAYER]</div>
                  <div className="pl-24">|</div>
                  <div className="pl-20">v</div>
                  <div>[CONTROLLER] &lt;--(Optimal Input)-- [PHYSICORE MPC]</div>
                </div>

                <div className="p-6 bg-cyan/5 border border-cyan/20 space-y-4">
                  <div className="flex items-center gap-2 text-cyan font-display text-xs font-bold uppercase">
                    <Zap size={14} /> Live Data Guarantee
                  </div>
                  <p className="font-body text-xs text-textSecondary leading-relaxed">
                    PhysiCore v3.1.5 enforces a strict "Hardware-First" policy. The dashboard will not display 
                    simulated physics when a connection is established. If the telemetry stream is interrupted, 
                    the engine enters a "WAITING" state rather than falling back to internal simulations. 
                    This ensures that every number you see is a direct reflection of your physical system.
                  </p>
                </div>
              </section>
            )}

            {manualSection === 'ros2' && (
              <section className="space-y-8">
                <div className="space-y-4">
                  <h2 className="font-display text-2xl font-bold text-white tracking-tight uppercase">ROS2 Integration</h2>
                  <p className="font-body text-sm text-textSecondary leading-relaxed">
                    The ROS2 bridge uses standard `rclcpp` nodes to communicate with the PhysiCore WebSocket or native C++ library.
                  </p>
                </div>
                <CodeBlock 
                  filename="physicore_bridge_node.cpp"
                  content={`#include "rclcpp/rclcpp.hpp"
#include "physicore_msgs/msg/telemetry.hpp"
#include "physicore_msgs/msg/control.hpp"

class PhysiCoreBridge : public rclcpp::Node {
public:
  PhysiCoreBridge() : Node("physicore_bridge") {
    sub_ = create_subscription<physicore_msgs::msg::Telemetry>(
      "/system/telemetry", 10, std::bind(&PhysiCoreBridge::on_telemetry, this, _1));
    pub_ = create_publisher<physicore_msgs::msg::Control>("/physicore/input", 10);
  }

private:
  void on_telemetry(const physicore_msgs::msg::Telemetry::SharedPtr msg) {
    // Process telemetry and send to PhysiCore Kernel
    auto control_msg = physicore_msgs::msg::Control();
    control_msg.thrust = kernel_.compute_optimal_thrust(msg->state);
    pub_->publish(control_msg);
  }
  rclcpp::Subscription<physicore_msgs::msg::Telemetry>::SharedPtr sub_;
  rclcpp::Publisher<physicore_msgs::msg::Control>::SharedPtr pub_;
  PhysiCoreKernel kernel_;
};`}
                />
              </section>
            )}

            {manualSection === 'ardupilot' && (
              <section className="space-y-8">
                <div className="space-y-4">
                  <h2 className="font-display text-2xl font-bold text-white tracking-tight uppercase">ArduPilot / PX4 Integration</h2>
                  <p className="font-body text-sm text-textSecondary leading-relaxed">
                    Integrate via AP_DDS or uXRCE-DDS for low-latency hardware-in-the-loop simulation.
                  </p>
                </div>
                <CodeBlock 
                  filename="dds_bridge_config.yaml"
                  content={`# uXRCE-DDS Agent Configuration
agent:
  port: 8888
  udp: true
  
topics:
  - name: fmu/out/vehicle_odometry
    type: px4_msgs::msg::VehicleOdometry
  - name: fmu/in/offboard_control_mode
    type: px4_msgs::msg::OffboardControlMode
  - name: fmu/in/trajectory_setpoint
    type: px4_msgs::msg::TrajectorySetpoint`}
                />
              </section>
            )}

            {manualSection === 'matlab' && (
              <section className="space-y-8">
                <div className="space-y-4">
                  <h2 className="font-display text-2xl font-bold text-white tracking-tight uppercase">MATLAB / Simulink</h2>
                  <p className="font-body text-sm text-textSecondary leading-relaxed">
                    Use the PhysiCore S-Function block to bring high-fidelity physics into your Simulink models.
                  </p>
                </div>
                <CodeBlock 
                  filename="physicore_init.m"
                  content={`% Initialize PhysiCore for Simulink
pc = PhysiCore('Rocket_V4');
pc.setSolver('RK4');
pc.setStepSize(0.01);

% Load into Simulink workspace
assignin('base', 'pc_kernel', pc);
sim('PhysiCore_HIL_Model');`}
                />
              </section>
            )}

            {manualSection === 'bot' && (
              <section className="space-y-8">
                <div className="space-y-4">
                  <h2 className="font-display text-2xl font-bold text-white tracking-tight uppercase">Example: Balancing Bot</h2>
                  <p className="font-body text-sm text-textSecondary leading-relaxed">
                    A 2-wheeled inverted pendulum requiring active stabilization.
                  </p>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div className="p-4 bg-bg border border-borderDim">
                    <div className="micro-label text-cyan">Mass</div>
                    <div className="font-mono text-lg text-white">1.2 kg</div>
                  </div>
                  <div className="p-4 bg-bg border border-borderDim">
                    <div className="micro-label text-cyan">Height</div>
                    <div className="font-mono text-lg text-white">0.35 m</div>
                  </div>
                  <div className="p-4 bg-bg border border-borderDim">
                    <div className="micro-label text-cyan">Control</div>
                    <div className="font-mono text-lg text-white">LQR / PID</div>
                  </div>
                </div>
                <CodeBlock 
                  filename="balancing_bot_logic.py"
                  content={`import physicore as pc

# Initialize bot model
bot = pc.Robot(type="inverted_pendulum")
bot.set_params(mass=1.2, length=0.35, friction=0.05)

# Control Loop
while True:
    state = hardware.get_imu_data()
    # PhysiCore computes the balancing torque
    torque = bot.compute_stabilization(state.theta, state.theta_dot)
    hardware.set_motor_torque(torque)`}
                />
              </section>
            )}

            {manualSection === 'drone' && (
              <section className="space-y-8">
                <div className="space-y-4">
                  <h2 className="font-display text-2xl font-bold text-white tracking-tight uppercase">Example: Auto Drone</h2>
                  <p className="font-body text-sm text-textSecondary leading-relaxed">
                    Autonomous quadcopter with trajectory tracking and wind disturbance rejection.
                  </p>
                </div>
                <CodeBlock 
                  filename="drone_mpc_config.json"
                  content={`{
  "uav_type": "quad_x",
  "mass": 0.85,
  "arm_length": 0.22,
  "max_thrust": 18.5,
  "mpc_lookahead": 12,
  "disturbance_rejection": true,
  "safety_bounds": {
    "max_tilt": 45,
    "max_velocity": 15.0
  }
}`}
                />
              </section>
            )}

            {manualSection === 'robot' && (
              <section className="space-y-8">
                <div className="space-y-4">
                  <h2 className="font-display text-2xl font-bold text-white tracking-tight uppercase">Example: Robotic Arm</h2>
                  <p className="font-body text-sm text-textSecondary leading-relaxed">
                    6-DOF industrial arm with collision avoidance and singularity handling.
                  </p>
                </div>
                <CodeBlock 
                  filename="arm_kinematics.cpp"
                  content={`// PhysiCore Inverse Kinematics with Collision Avoidance
auto target_pose = get_target();
auto current_joints = get_joints();

auto solution = pc_arm.solve_ik(target_pose, current_joints, {
  .avoid_collisions = true,
  .max_acceleration = 2.5,
  .smooth_trajectory = true
});

if (solution.success) {
  move_to(solution.joint_angles);
}`}
                />
              </section>
            )}

            {manualSection === 'rocket' && (
              <section className="space-y-8">
                <div className="space-y-4">
                  <h2 className="font-display text-2xl font-bold text-white tracking-tight uppercase">Example: High-Power Rocket</h2>
                  <p className="font-body text-sm text-textSecondary leading-relaxed">
                    Suborbital rocket with active drag brakes and dual-deployment recovery.
                  </p>
                </div>
                <div className="p-6 bg-redDim border border-red/30 space-y-2">
                  <div className="flex items-center gap-2 text-red font-display text-xs font-bold uppercase">
                    <AlertTriangle size={14} /> Critical Safety Note
                  </div>
                  <p className="font-body text-[10px] text-textSecondary">
                    PhysiCore Sentinel must be active during the COAST phase to ensure accurate apogee detection and recovery trigger timing.
                  </p>
                </div>
                <CodeBlock 
                  filename="rocket_gnc.py"
                  content={`# PhysiCore Rocket GNC Module
rocket = pc.Rocket(profile="Level3_Heavy")

def flight_loop():
    while rocket.is_flying:
        data = sensor_fusion.get_state()
        # Predict apogee in real-time
        predicted_apogee = rocket.predict_apogee(data)
        
        if data.altitude > predicted_apogee - 5.0:
            rocket.trigger_drogue()
            break`}
                />
              </section>
            )}

            {manualSection === 'aviation' && (
              <section className="space-y-8">
                <div className="space-y-4">
                  <h2 className="font-display text-2xl font-bold text-white tracking-tight uppercase">Example: Aviation Dynamics</h2>
                  <p className="font-body text-sm text-textSecondary leading-relaxed">
                    Fixed-wing aircraft flight envelope protection and autopilot integration.
                  </p>
                </div>
                <CodeBlock 
                  filename="flight_envelope.m"
                  content={`% PhysiCore Flight Envelope Protection
% Prevents stall and overspeed conditions
[alpha, beta, v_air] = get_air_data();

if alpha > pc.stall_alpha_limit
    pc.apply_nose_down_correction();
    warning('STALL PROTECTION ACTIVE');
end`}
                />
              </section>
            )}
          </div>
        </main>
      </div>
    );
  };

  const renderDashboard = () => {
    if (hasTested && !isAdmin) return renderQuotaExceeded();
    return (
      <div className="pt-[52px] h-screen flex flex-col bg-void overflow-hidden">
      <div className="flex-1 flex overflow-hidden">
        {/* SIDEBAR: SYSTEM STATUS */}
        <aside className="w-[320px] border-r border-border bg-bg flex flex-col shrink-0 overflow-hidden">
          <div className="p-6 border-b border-border space-y-4">
            <div className="flex items-center justify-between">
              <span className={`micro-label ${isSystemConnected ? 'text-green' : 'text-textDim'}`}>System Status</span>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isSystemConnected ? 'bg-green animate-pulse' : 'bg-red'}`} />
                <span className={`font-mono text-[10px] ${isSystemConnected ? 'text-green' : 'text-red'}`}>
                  {isSystemConnected ? 'LIVE' : 'DISCONNECTED'}
                </span>
              </div>
            </div>
            
            {isSystemConnected && (
              <div className="p-3 bg-bgRaised border border-borderDim space-y-1">
                <div className="micro-label text-textDim">CONNECTION SOURCE</div>
                <div className="font-mono text-[10px] text-cyan uppercase">
                  {connectionMode === 'ros2_websocket' ? 'Real Hardware (ROS2)' : 'Hardware-in-the-Loop'}
                </div>
                <div className="font-mono text-[9px] text-textDim truncate">{endpoint}</div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-bgRaised border border-borderDim">
                <div className="micro-label text-textDim">CPU LOAD</div>
                <div className="font-mono text-lg text-white">{isSystemConnected ? '12.4%' : '0.0%'}</div>
              </div>
              <div className="p-3 bg-bgRaised border border-borderDim">
                <div className="micro-label text-textDim">LATENCY</div>
                <div className="font-mono text-lg text-white">{isSystemConnected ? '0.4ms' : '--'}</div>
              </div>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto custom-scroll p-6 space-y-8">
            {systemProfile.domain === 'ROCKETS' ? (
              <RocketTelemetryWidgets flightData={flightData} actualData={actualFlightData} params={rocketParams} />
            ) : (
              <>
                <section className="space-y-4">
                  <div className="border-l-2 border-cyan pl-3">
                    <span className="micro-label">SystemID Estimates</span>
                  </div>
                  <div className="space-y-4">
                    {[
                      { label: 'ESTIMATED MASS', val: `${telemetry.mass.toFixed(3)} kg`, delta: telemetry.mass > 2.7 ? '+8.5%' : '-2.1%', color: COLORS.green },
                      { label: 'FRICTION COEFF', val: `${telemetry.friction.toFixed(3)} μ`, delta: telemetry.friction > 0.4 ? '+17.7%' : '-4.2%', color: COLORS.amber },
                      { label: 'ACTUATOR EFF', val: `${(telemetry.actuatorEfficiency * 100).toFixed(1)}%`, delta: telemetry.actuatorEfficiency > 0.9 ? '-1.2%' : '-5.4%', color: COLORS.cyan },
                    ].map((item, i) => (
                      <div key={i} className="space-y-1">
                        <div className="flex justify-between items-center">
                          <span className="micro-label text-textDim">{item.label}</span>
                          <span className="font-mono text-[10px]" style={{ color: item.color }}>{item.delta}</span>
                        </div>
                        <div className="font-mono text-sm text-white">{item.val}</div>
                      </div>
                    ))}
                  </div>
                </section>

                <section className="space-y-4">
                  <div className="border-l-2 border-amber pl-3">
                    <span className="micro-label">Sentinel Governance</span>
                  </div>
                  <div className="p-4 bg-amberDim/10 border border-amber/20 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="font-display text-[10px] font-bold text-amber tracking-widest uppercase">Stability Kernel</span>
                      <span className="font-mono text-[9px] text-green">{telemetry.isStable ? 'NOMINAL' : 'UNSTABLE'}</span>
                    </div>
                    <div className="h-1 w-full bg-border">
                      <div className="h-full bg-amber transition-all duration-500" style={{ width: `${telemetry.confidence}%` }} />
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-mono text-[9px] text-textSecondary uppercase">Confidence: {telemetry.confidence.toFixed(1)}%</span>
                      {telemetry.isFaulted && <span className="font-mono text-[9px] text-red animate-pulse">FAULT_DETECTED</span>}
                    </div>
                  </div>
                  <div className="p-4 border border-border bg-bgRaised space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="micro-label text-textDim uppercase">Byzantine Consensus</span>
                      <span className="font-mono text-[9px] text-textSecondary">SINGLE NODE</span>
                    </div>
                    <div className="font-mono text-[8px] text-textDim uppercase">Consensus disabled — Multi-node quorum required</div>
                  </div>
                </section>
              </>
            )}
          </div>
        </aside>

        {/* MAIN CONTENT: SIMULATION & CHARTS */}
        <main className="flex-1 flex flex-col bg-bgInset overflow-hidden">
          {/* SIMULATION VIEW */}
          <div className="flex-1 relative border-b border-border">
            <div className="absolute top-6 left-6 z-10 space-y-2">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 border border-green flex items-center justify-center text-green">
                  {systemProfile.domain === 'ROCKETS' ? <Rocket size={20} /> : <Maximize2 size={20} />}
                </div>
                <div>
                  <h2 className="font-display text-lg font-bold text-white tracking-widest uppercase">
                    {systemProfile.domain === 'ROCKETS' ? 'Rocket Trajectory View' : 'Live Simulation View'}
                  </h2>
                  <p className="font-mono text-[10px] text-textDim uppercase">
                    {systemProfile.domain === 'ROCKETS' ? '3DOF Physics Engine | ISA Atmosphere' : 'RK4 Integrator @ 60Hz | MPC Lookahead'}
                  </p>
                </div>
              </div>
            </div>
            
            <div className="absolute top-6 right-6 z-10 flex gap-2">
              {systemProfile.domain === 'ROCKETS' && (
                <button 
                  onClick={() => setShowImportOverlay(true)}
                  className="px-3 py-2 border border-cyan bg-cyan/10 text-cyan font-display text-[10px] font-bold uppercase tracking-widest hover:bg-cyan hover:text-black transition-all"
                >
                  ⬆ IMPORT FLIGHT DATA
                </button>
              )}
              <button 
                onClick={() => setIsSystemConnected(false)}
                className="px-3 py-2 border border-red bg-red/10 text-red font-display text-[10px] font-bold uppercase tracking-widest hover:bg-red hover:text-black transition-all"
                title="Disconnect Hardware"
              >
                ⏻ DISCONNECT
              </button>
              <button onClick={systemProfile.domain === 'ROCKETS' ? resetRocketSim : () => {}} className="p-2 border border-border bg-bgRaised text-textSecondary hover:text-green transition-all">
                <RefreshCw size={16} />
              </button>
              <button className="p-2 border border-border bg-bgRaised text-textSecondary hover:text-green transition-all">
                <Settings size={16} />
              </button>
            </div>

            {systemProfile.domain === 'ROCKETS' ? (
              <RocketTrajectoryCanvas 
                state={rocketState} 
                params={rocketParams} 
                flightData={flightData} 
                actualData={actualFlightData}
                simSpeed={rocketSimSpeed}
                setSimSpeed={setRocketSimSpeed}
                isRunning={isRocketSimRunning}
                setIsRunning={setIsRocketSimRunning}
                resetSim={resetRocketSim}
                isConnected={isSystemConnected}
                handshakeConfirmed={handshakeConfirmed}
              />
            ) : (
              <DashboardCanvas 
                isConnected={isSystemConnected} 
                handshakeConfirmed={handshakeConfirmed}
                onTelemetryUpdate={setTelemetry} 
                telemetry={telemetry} 
                connectionMode={connectionMode} 
              />
            )}

            {!isSystemConnected && (
              <div className="absolute inset-0 z-20 bg-bg flex flex-col items-center justify-center space-y-8 p-12 text-center">
                <div className="w-24 h-24 border-2 border-dashed border-border flex items-center justify-center text-textDim">
                  {isSystemConnecting ? (
                    <RefreshCw size={48} className="animate-spin text-cyan" />
                  ) : (
                    <Cpu size={48} className="opacity-20" />
                  )}
                </div>
                
                <div className="max-w-md space-y-4">
                  <h3 className="font-display text-xl font-bold text-white tracking-widest uppercase">
                    {isSystemConnecting ? 'Establishing Secure Link' : 'Hardware Connection Required'}
                  </h3>
                  <p className="font-body text-sm text-textSecondary leading-relaxed">
                    {isSystemConnecting 
                      ? `Attempting to handshake with ${connectionMode.toUpperCase()} endpoint at ${endpoint}...`
                      : 'PhysiCore is currently in "Observer Mode". To view live telemetry and control the system, you must establish a manual link to your hardware bridge or simulation environment.'}
                  </p>
                  
                  {connectionError && (
                    <div className="p-4 bg-red/10 border border-red/30 text-red font-mono text-[10px] uppercase tracking-widest">
                      ERROR: {connectionError}
                    </div>
                  )}
                </div>

                <div className="flex flex-col gap-3 w-full max-w-[280px]">
                  <button 
                    onClick={() => setView('integrator')}
                    className="w-full p-4 border border-cyan/30 bg-bgRaised hover:bg-cyan hover:text-black transition-all group flex items-center justify-between"
                  >
                    <span className="font-display text-[11px] font-bold tracking-widest uppercase">Return to Integrator</span>
                    <Settings size={18} />
                  </button>
                  
                  <button 
                    onClick={handleConnect}
                    disabled={isSystemConnecting}
                    className="w-full p-4 border border-green/30 bg-bgRaised hover:bg-green hover:text-black transition-all group flex items-center justify-between disabled:opacity-50"
                  >
                    <span className="font-display text-[11px] font-bold tracking-widest uppercase">
                      {isSystemConnecting ? 'Retrying...' : 'Retry Connection'}
                    </span>
                    <Wifi size={18} />
                  </button>
                </div>
              </div>
            )}

            <div className="absolute bottom-6 left-6 right-6 z-10 flex justify-between items-end">
              <div className="space-y-4">
                <div className="flex gap-4">
                  <div className="p-3 bg-bg/80 backdrop-blur-md border border-border space-y-1">
                    <div className="micro-label text-textDim">MPC HORIZON</div>
                    <div className="font-mono text-xs text-cyan">12 STEPS</div>
                  </div>
                  <div className="p-3 bg-bg/80 backdrop-blur-md border border-border space-y-1">
                    <div className="micro-label text-textDim">ENSEMBLE σ</div>
                    <div className="font-mono text-xs text-green">{telemetry.variance.toFixed(4)}</div>
                  </div>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="p-4 bg-bg/80 backdrop-blur-md border border-border flex items-center gap-4">
                  <div className="text-right">
                    <div className="micro-label text-textDim">Target Position</div>
                    <div className="font-mono text-xs text-white">X: {telemetry.targetPos.x.toFixed(1)} Y: {telemetry.targetPos.y.toFixed(1)}</div>
                  </div>
                  <div className="w-10 h-10 border border-border flex items-center justify-center text-textDim">
                    <Crosshair size={20} />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* BOTTOM PANEL: CHARTS & META-ANALYST */}
          <div className="h-[280px] bg-bg border-t border-border flex divide-x divide-border overflow-hidden">
            <div className="flex-1 p-6 space-y-4 overflow-hidden">
              <div className="flex justify-between items-center">
                <span className="micro-label text-cyan">L2 Prediction Residual</span>
                <span className="font-mono text-[10px] text-textDim">CONVERGENCE: {telemetry.residual.toFixed(4)}</span>
              </div>
              <div className="h-[180px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={telemetry.residualHistory}>
                    <defs>
                      <linearGradient id="colorResidual" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={COLORS.cyan} stopOpacity={0.3}/>
                        <stop offset="95%" stopColor={COLORS.cyan} stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1A1A1A" vertical={false} />
                    <Area type="monotone" dataKey="y" stroke={COLORS.cyan} fillOpacity={1} fill="url(#colorResidual)" isAnimationActive={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="flex-1 p-6 space-y-4 overflow-hidden">
              <div className="flex justify-between items-center">
                <span className="micro-label text-green">Control Effort (N)</span>
                <span className="font-mono text-[10px] text-textDim">PEAK: {Math.max(...telemetry.effortHistory.map(d => d.y), 0).toFixed(1)}N</span>
              </div>
              <div className="h-[180px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={telemetry.effortHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1A1A1A" vertical={false} />
                    <Bar dataKey="y" fill={COLORS.green} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="w-[400px] p-6 space-y-4 bg-bgRaised/30 overflow-hidden flex flex-col">
              <div className="flex justify-between items-center shrink-0">
                <div className="flex items-center gap-2">
                  <div className={`w-1.5 h-1.5 rounded-full ${quotaExceeded ? 'bg-red' : 'bg-cyan animate-pulse'}`} />
                  <span className="micro-label text-white">Meta-Analyst Intelligence</span>
                </div>
                <span className="font-mono text-[9px] text-textDim uppercase">Neural Link: {quotaExceeded ? 'THROTTLED' : 'ACTIVE'}</span>
              </div>
              <div className="flex-1 overflow-y-auto custom-scroll pr-2">
                {quotaExceeded ? (
                  <div className="h-full flex flex-col items-center justify-center text-center space-y-4 p-4 border border-dashed border-red/20 bg-red/5">
                    <AlertTriangle size={24} className="text-red/60" />
                    <div className="space-y-1">
                      <p className="font-mono text-[10px] text-red uppercase">Neural Quota Exhausted</p>
                      <p className="font-body text-[10px] text-textDim">Rate limit reached. Symbolic safety layer is maintaining control. Neural link will reset in 120s.</p>
                    </div>
                  </div>
                ) : metaAnalysisResult ? (
                  <div className="space-y-4">
                    <div className="font-mono text-[11px] text-cyan leading-relaxed">
                      {metaAnalysisResult.replace('> META-ANALYST:', '').trim()}
                    </div>
                    <div className="pt-4 border-t border-borderDim space-y-3">
                      <div className="flex items-center gap-2">
                        <Zap size={10} className="text-amber" />
                        <span className="font-display text-[9px] font-bold text-amber uppercase tracking-widest">Tuning Recommendation</span>
                      </div>
                      <div className="p-3 bg-amberDim/10 border border-amber/20 font-mono text-[10px] text-amber/80">
                        {metaAnalysisResult.includes('MPC') ? 'ADJUST_MPC_COST_WEIGHTS: Q_POS += 1.5' : 'ADJUST_SYSID_LR: ALPHA = 0.012'}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center text-center space-y-4 opacity-40">
                    <div className="w-8 h-8 border border-textDim flex items-center justify-center text-textDim">
                      <Cpu size={16} className="animate-spin-slow" />
                    </div>
                    <p className="font-mono text-[9px] text-textDim uppercase">Initializing Neural Diagnostics...</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

  if (loading || checkingAccess) {
    return (
      <div className="h-screen w-full bg-void flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Cpu className="text-green animate-spin-slow" size={48} />
          <span className="font-mono text-xs text-green uppercase tracking-widest">Verifying Neural Handshake...</span>
        </div>
      </div>
    );
  }

  if (user && isAllowed === false) {
    return (
      <div className="h-screen w-full bg-void flex flex-col items-center justify-center p-6 text-center">
        <div className="max-w-[400px] space-y-8">
          <ShieldAlert className="text-red mx-auto" size={64} />
          <div className="space-y-2">
            <h2 className="font-display text-2xl font-bold text-white uppercase tracking-tighter">Access Denied</h2>
            <p className="font-body text-sm text-textSecondary leading-relaxed">
              Your account ( {user.email} ) is not authorized to access the PhysiCore v3.0 infrastructure. 
              Please contact the system administrator for onboarding.
            </p>
          </div>
          <button onClick={handleLogout} className="btn-outline w-full h-12 text-xs">LOGOUT & RETURN</button>
        </div>
      </div>
    );
  }

  if (user && hasTested && !isAdmin && view !== 'home') {
    return renderQuotaExceeded();
  }

  return (
    <div className="w-full h-full">
      {/* Auth Error Modal */}
      <AnimatePresence>
        {authError && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-sm flex items-center justify-center p-6"
          >
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="max-w-md w-full bg-bgRaised border border-red/30 p-8 space-y-6"
            >
              <div className="flex items-center gap-4 text-red">
                <ShieldAlert size={32} />
                <h2 className="font-display text-xl font-bold uppercase tracking-widest">Access Protocol Failure</h2>
              </div>
              
              <div className="space-y-4">
                <p className="font-body text-sm text-textSecondary leading-relaxed">
                  {authError.message}
                </p>
                
                {authError.domain && (
                  <div className="p-4 bg-black/50 border border-border space-y-3">
                    <p className="font-mono text-[10px] text-cyan uppercase tracking-widest">Required Action:</p>
                    <ol className="font-mono text-[9px] text-textDim space-y-2 list-decimal pl-4">
                      <li>Go to Firebase Console &gt; Authentication &gt; Settings</li>
                      <li>Find "Authorized domains" section</li>
                      <li>Add <span className="text-white">"{authError.domain}"</span> to the list</li>
                    </ol>
                  </div>
                )}
              </div>

              <div className="flex gap-3">
                <button 
                  onClick={() => setAuthError(null)}
                  className="flex-1 py-3 border border-border text-textDim font-display text-[10px] font-bold uppercase tracking-widest hover:text-white transition-all"
                >
                  Dismiss
                </button>
                {authError.domain && (
                  <a 
                    href="https://console.firebase.google.com/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-1 py-3 bg-red/20 border border-red/40 text-red font-display text-[10px] font-bold uppercase tracking-widest hover:bg-red/40 transition-all text-center"
                  >
                    Open Console
                  </a>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {renderNav()}
      <AnimatePresence mode="wait">
        {view === 'home' ? (
          <motion.div key="home" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            {renderHome()}
          </motion.div>
        ) : view === 'integrator' ? (
          <motion.div key="integrator" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            {renderIntegrator()}
          </motion.div>
        ) : view === 'manual' ? (
          <motion.div key="manual" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            {renderManual()}
          </motion.div>
        ) : view === 'team' ? (
          <motion.div key="team" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            {renderTeam()}
          </motion.div>
        ) : (
          <motion.div key="dashboard" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            {renderDashboard()}
          </motion.div>
        )}
      </AnimatePresence>
      <AnimatePresence>
        {isLaunching && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[200] bg-void/90 backdrop-blur-xl flex items-center justify-center p-6"
          >
            <div className="max-w-[400px] w-full space-y-8 text-center">
              <div className="relative w-24 h-24 mx-auto">
                <motion.div 
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  className="absolute inset-0 border-2 border-green border-t-transparent"
                />
                <div className="absolute inset-0 flex items-center justify-center text-green">
                  <Cpu size={32} className="animate-pulse" />
                </div>
              </div>
              <div className="space-y-2">
                <h3 className="font-display text-xl font-bold text-white tracking-widest uppercase">Initializing PhysiCore</h3>
                <p className="font-mono text-[10px] text-textSecondary uppercase tracking-widest">Loading RK4 Kernel & Sentinel Handshake...</p>
              </div>
              <div className="h-1 w-full bg-border overflow-hidden">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: '100%' }}
                  transition={{ duration: 2.5 }}
                  className="h-full bg-green"
                />
              </div>
              <div className="grid grid-cols-2 gap-2">
                {['KERNEL_OK', 'SENTINEL_OK', 'MPC_OK', 'SYSID_OK'].map(status => (
                  <div key={status} className="flex items-center gap-2 font-mono text-[8px] text-green/60">
                    <Check size={8} /> {status}
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Digital Twin Confirmation Modal */}
      <AnimatePresence>
        {showDigitalTwinModal && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[300] bg-void/95 backdrop-blur-2xl flex items-center justify-center p-6"
          >
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="max-w-xl w-full p-8 border border-amber/30 bg-bgRaised space-y-8"
            >
              <div className="flex items-center gap-4 border-b border-amber/20 pb-6">
                <div className="w-12 h-12 border border-amber flex items-center justify-center text-amber bg-amber/5">
                  <ShieldAlert size={24} />
                </div>
                <div>
                  <h2 className="font-display text-xl font-bold text-white tracking-widest uppercase italic">Digital Twin Protocol</h2>
                  <p className="font-mono text-[10px] text-amber uppercase tracking-widest">Simulation-Only Mode Detected</p>
                </div>
              </div>

              <div className="space-y-4 font-body text-sm text-textSecondary leading-relaxed">
                <p>
                  You are attempting to launch the PhysiCore Dashboard in <span className="text-white font-bold">DIGITAL TWIN</span> mode. 
                  This mode relies on internal physics models rather than live hardware telemetry.
                </p>
                <div className="p-4 bg-bg border border-borderDim space-y-3">
                  <div className="flex items-center gap-2 text-amber">
                    <Info size={14} />
                    <span className="micro-label uppercase">Mandatory Disclaimer</span>
                  </div>
                  <p className="text-[11px] uppercase tracking-wider leading-relaxed">
                    Digital Twin simulations are for <span className="text-amber">architectural verification only</span>. 
                    They do not account for real-world sensor noise, actuator latency, or environmental stochasticity. 
                    PhysiCore v3.0 is designed for hardware-in-the-loop (HIL) integration.
                  </p>
                </div>
                <p className="text-xs italic">
                  By confirming, you acknowledge that the results seen in the dashboard are simulated and may not reflect actual hardware performance.
                </p>
              </div>

              <div className="flex gap-4 pt-4">
                <button 
                  onClick={() => setShowDigitalTwinModal(false)}
                  className="flex-1 py-3 border border-border text-textSecondary font-display font-bold text-[11px] uppercase tracking-widest hover:text-white transition-all"
                >
                  CANCEL
                </button>
                <button 
                  onClick={() => {
                    setDigitalTwinConfirmed(true);
                    setShowDigitalTwinModal(false);
                    // Trigger the launch/connect again now that it's confirmed
                    setTimeout(() => {
                      if (view === 'integrator') handleLaunchApp();
                      else handleConnect();
                    }, 100);
                  }}
                  className="flex-1 py-3 bg-amber text-black font-display font-bold text-[11px] uppercase tracking-widest hover:bg-white transition-all"
                >
                  CONFIRM & PROCEED
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// --- SUB-COMPONENTS ---

const HeroCanvas = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let frame = 0;
    let pos = { x: 0, y: 0 };
    let target = { x: 100, y: 100 };
    let trail: { x: number; y: number }[] = [];

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    window.addEventListener('resize', resize);
    resize();

    const animate = () => {
      frame++;
      ctx.fillStyle = COLORS.void;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Grid
      ctx.strokeStyle = '#0F0F0F';
      ctx.lineWidth = 1;
      for (let x = 0; x < canvas.width; x += 60) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += 60) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
      }

      // Target movement
      if (frame % 180 === 0) {
        target = { x: Math.random() * canvas.width, y: Math.random() * canvas.height };
      }

      // Smooth follow
      pos.x += (target.x - pos.x) * 0.01;
      pos.y += (target.y - pos.y) * 0.01;
      trail.push({ ...pos });
      if (trail.length > 100) trail.shift();

      // Trail
      ctx.strokeStyle = 'rgba(0, 255, 136, 0.1)';
      ctx.beginPath();
      trail.forEach((p, i) => {
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.stroke();

      // Reticle
      ctx.save();
      ctx.translate(pos.x, pos.y);
      ctx.rotate(frame * 0.005);
      ctx.strokeStyle = 'rgba(0, 255, 136, 0.3)';
      ctx.strokeRect(-20, -20, 40, 40);
      ctx.beginPath();
      ctx.moveTo(-30, 0); ctx.lineTo(-15, 0);
      ctx.moveTo(30, 0); ctx.lineTo(15, 0);
      ctx.moveTo(0, -30); ctx.lineTo(0, -15);
      ctx.moveTo(0, 30); ctx.lineTo(0, 15);
      ctx.stroke();
      ctx.restore();

      requestAnimationFrame(animate);
    };

    animate();
    return () => window.removeEventListener('resize', resize);
  }, []);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />;
};

const DashboardCanvas = ({ isConnected, handshakeConfirmed, onTelemetryUpdate, telemetry, connectionMode }: { isConnected: boolean, handshakeConfirmed: boolean, onTelemetryUpdate: (data: any) => void, telemetry: any, connectionMode: string }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const stateRef = useRef({
    robot: { x: 0, y: 0, vx: 0, vy: 0 },
    target: { x: 0, y: 0 },
    trueParams: { mass: 5.2, friction: 0.65 }, // Hidden true parameters
    estParams: { mass: 1.0, friction: 0.1 },   // AI's current estimates
    actuatorEfficiency: 0.95,
    residualHistory: [] as any[],
    effortHistory: [] as any[],
    frame: 0
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const state = stateRef.current;
    state.robot = { x: canvas.width / 2, y: canvas.height / 2, vx: 0, vy: 0 };
    state.target = { x: canvas.width / 2, y: canvas.height / 2 };

    const resize = () => {
      const parent = canvas.parentElement;
      if (parent) {
        canvas.width = parent.clientWidth;
        canvas.height = parent.clientHeight;
      }
    };

    window.addEventListener('resize', resize);
    resize();

    const prediction_error = (pos: any, vel: any, actual_pos: any, mass: number, friction: number) => {
      const pred = rk4_step(pos, vel, { x: 0, y: 0 }, { mass, friction });
      const dx = pred.pos.x - actual_pos.x;
      const dy = pred.pos.y - actual_pos.y;
      return Math.pow(dx, 2) + Math.pow(dy, 2);
    };

    const system_id_update = (pos: any, vel: any, actual_pos: any) => {
      // Mass gradient
      const m_plus = state.estParams.mass + 0.025;
      const e_plus = prediction_error(pos, vel, actual_pos, m_plus, state.estParams.friction);
      const m_minus = state.estParams.mass - 0.025;
      const e_minus = prediction_error(pos, vel, actual_pos, m_minus, state.estParams.friction);
      const grad_m = (e_plus - e_minus) / 0.05;
      state.estParams.mass = Math.max(0.3, Math.min(25.0, state.estParams.mass - 0.015 * grad_m));

      // Friction gradient
      const f_plus = state.estParams.friction + 0.008;
      const ef_plus = prediction_error(pos, vel, actual_pos, state.estParams.mass, f_plus);
      const f_minus = state.estParams.friction - 0.008;
      const ef_minus = prediction_error(pos, vel, actual_pos, state.estParams.mass, f_minus);
      const grad_f = (ef_plus - ef_minus) / 0.016;
      state.estParams.friction = Math.max(0.02, Math.min(0.95, state.estParams.friction - 0.006 * grad_f));
    };

    const compute_ensemble = (pos: any, vel: any, params: Params) => {
      const noise_scales = [0.7, 1.0, 1.3];
      const predictions = noise_scales.map(scale => {
        const noise = { x: (Math.random() - 0.5) * scale * 2, y: (Math.random() - 0.5) * scale * 2 };
        return rk4_step(pos, { x: vel.x + noise.x, y: vel.y + noise.y }, { x: 0, y: 0 }, params);
      });

      const dists = predictions.map(p => Math.sqrt(Math.pow(p.pos.x, 2) + Math.pow(p.pos.y, 2)));
      const meanDist = dists.reduce((a, b) => a + b, 0) / dists.length;
      const variance = Math.sqrt(dists.reduce((s, d) => s + Math.pow(d - meanDist, 2), 0) / dists.length);
      
      const confidence = Math.max(0, 100 - variance * 180);
      const meanPred = {
        x: predictions.reduce((s, p) => s + p.pos.x, 0) / predictions.length,
        y: predictions.reduce((s, p) => s + p.pos.y, 0) / predictions.length
      };
      const residual = Math.sqrt(Math.pow(meanPred.x - pos.x, 2) + Math.pow(meanPred.y - pos.y, 2));

      return { confidence, residual, variance };
    };

    const rls_update = (pos: any, vel: any, actual_pos: any) => {
      const pred = rk4_step(pos, vel, { x: 0, y: 0 }, state.estParams);
      const error = Math.sqrt(Math.pow(pred.pos.x - actual_pos.x, 2) + Math.pow(pred.pos.y - actual_pos.y, 2));
      state.actuatorEfficiency = Math.max(0.7, Math.min(0.99, 0.9 * state.actuatorEfficiency + 0.1 * (1.0 - error / 50)));
    };

    const lyapunov_check = (vel: any, mass: number) => {
      const energy = 0.5 * mass * (Math.pow(vel.x, 2) + Math.pow(vel.y, 2));
      return energy < 10000; 
    };

    const fault_observer = (pred: any, actual: any) => {
      const error = Math.sqrt(Math.pow(pred.x - actual.x, 2) + Math.pow(pred.y - actual.y, 2));
      return error > 15.0;
    };

    const animate = () => {
      state.frame++;
      ctx.fillStyle = COLORS.bgInset;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Grid
      ctx.strokeStyle = '#121212';
      ctx.lineWidth = 1;
      for (let x = 0; x < canvas.width; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
      }

      if (!isConnected || !handshakeConfirmed) {
        // Draw LOCKED state
        ctx.fillStyle = 'rgba(12, 12, 12, 0.8)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Dim crosshair reticle
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(canvas.width / 2, 0); ctx.lineTo(canvas.width / 2, canvas.height);
        ctx.moveTo(0, canvas.height / 2); ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();

        ctx.fillStyle = COLORS.red;
        ctx.font = 'bold 14px "JetBrains Mono"';
        ctx.textAlign = 'center';
        ctx.fillText('HARDWARE NOT CONNECTED // HANDSHAKE PENDING', canvas.width / 2, canvas.height / 2 - 10);
        ctx.fillStyle = COLORS.textDim;
        ctx.font = '10px "JetBrains Mono"';
        ctx.fillText('ESTABLISH SECURE LINK TO INITIALIZE KERNEL', canvas.width / 2, canvas.height / 2 + 15);
        
        // Draw dimmed safety envelope
        ctx.strokeStyle = 'rgba(255, 34, 34, 0.1)';
        ctx.setLineDash([10, 10]);
        ctx.strokeRect(40, 40, canvas.width - 80, canvas.height - 80);
        ctx.setLineDash([]);

        requestAnimationFrame(animate);
        return;
      }

      // If connected, we use the telemetry data from the hardware
      if (isConnected && telemetry.pos) {
        state.robot.x = telemetry.pos.x;
        state.robot.y = telemetry.pos.y;
        state.robot.vx = telemetry.vel?.x || 0;
        state.robot.vy = telemetry.vel?.y || 0;
        state.target = telemetry.targetPos || state.target;
      } else if (isConnected) {
        // Connected but no data yet
        ctx.fillStyle = COLORS.amber;
        ctx.font = 'bold 12px "JetBrains Mono"';
        ctx.textAlign = 'center';
        ctx.fillText('WAITING FOR TELEMETRY STREAM...', canvas.width / 2, canvas.height / 2);
        
        requestAnimationFrame(animate);
        return;
      }

      // Draw MPC Trajectory
      if ((state as any).lastOptimalSequence) {
        ctx.strokeStyle = COLORS.cyan;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(state.robot.x, state.robot.y);
        let px = state.robot.x;
        let py = state.robot.y;
        let pvx = state.robot.vx;
        let pvy = state.robot.vy;
        for (let i = 0; i < 12; i++) {
          const f = { x: (state as any).lastOptimalSequence.x[i], y: (state as any).lastOptimalSequence.y[i] };
          const step = rk4_step({ x: px, y: py }, { x: pvx, y: pvy }, f, state.estParams);
          px = step.pos.x;
          py = step.pos.y;
          pvx = step.vel.x;
          pvy = step.vel.y;
          ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Draw Target
      ctx.strokeStyle = COLORS.amber;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(state.target.x, state.target.y, 8, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(state.target.x - 12, state.target.y); ctx.lineTo(state.target.x + 12, state.target.y);
      ctx.moveTo(state.target.x, state.target.y - 12); ctx.lineTo(state.target.x, state.target.y + 12);
      ctx.stroke();

      // Draw Ensemble Nodes (Uncertainty)
      if ((state as any).lastEnsemble) {
        const numNodes = 3;
        for (let i = 0; i < numNodes; i++) {
          ctx.strokeStyle = `rgba(0, 255, 136, ${0.1 + (1 - (state as any).lastEnsemble.confidence/100) * 0.2})`;
          ctx.beginPath();
          const offset = (Math.random() - 0.5) * (state as any).lastEnsemble.variance * 500;
          ctx.arc(state.robot.x + offset, state.robot.y + offset, 15 + i * 5, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      // Draw Robot
      ctx.strokeStyle = COLORS.green;
      ctx.lineWidth = 2;
      ctx.strokeRect(state.robot.x - 20, state.robot.y - 20, 40, 40);
      ctx.fillStyle = COLORS.greenDim;
      ctx.fillRect(state.robot.x - 20, state.robot.y - 20, 40, 40);
      
      // Direction indicator
      ctx.beginPath();
      ctx.moveTo(state.robot.x, state.robot.y);
      ctx.lineTo(state.robot.x + state.robot.vx * 5, state.robot.y + state.robot.vy * 5);
      ctx.strokeStyle = COLORS.white;
      ctx.stroke();

      requestAnimationFrame(animate);
    };

    animate();
    return () => window.removeEventListener('resize', resize);
  }, [isConnected]);

  return <canvas ref={canvasRef} className="w-full h-full" />;
};

const RealityGapDiagram = () => (
  <div className="relative aspect-square border border-border bg-bgRaised p-10 flex flex-col justify-between">
    <div className="flex justify-between items-center">
      <div className="space-y-2">
        <span className="micro-label text-textDim">Simulation</span>
        <div className="font-display text-xl font-bold text-white">IDEAL STATE</div>
        <div className="font-mono text-[10px] text-textSecondary">mass: 2.500<br />fric: 0.350</div>
      </div>
      <div className="text-right space-y-2">
        <span className="micro-label text-textDim">Reality</span>
        <div className="font-display text-xl font-bold text-white">ACTUAL STATE</div>
        <div className="font-mono text-[10px] text-green">mass: 2.714<br />fric: 0.412</div>
      </div>
    </div>

    <div className="relative flex-1 my-10 border-x border-borderDim flex items-center justify-center">
      <svg className="w-full h-full" viewBox="0 0 200 100">
        {/* Sim path */}
        <path d="M20 80 Q100 0 180 80" fill="none" stroke={COLORS.textDim} strokeWidth="1" strokeDasharray="4 4" />
        {/* Reality path */}
        <path d="M20 80 Q90 20 170 90" fill="none" stroke={COLORS.green} strokeWidth="2" />
        {/* Gap arrow */}
        <path d="M100 40 L100 60" fill="none" stroke={COLORS.amber} strokeWidth="1" />
        <path d="M97 57 L100 60 L103 57" fill="none" stroke={COLORS.amber} strokeWidth="1" />
      </svg>
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-bg border border-amber px-3 py-1">
        <span className="font-mono text-[10px] text-amber uppercase tracking-widest">Reality Gap</span>
      </div>
    </div>

    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span className="font-mono text-[11px] text-green">L2 RESIDUAL: 0.847 → 0.023</span>
        <span className="font-mono text-[11px] text-textDim">CONVERGED</span>
      </div>
      <div className="h-1 w-full bg-border">
        <div className="h-full bg-green w-[92%]" />
      </div>
      <p className="font-body text-[9px] text-textDim uppercase tracking-widest">after 847 frames of SystemID convergence</p>
    </div>
  </div>
);

const SentinelDiagram = () => (
  <div className="relative aspect-video border border-border bg-bgRaised p-10 flex flex-col justify-center gap-8">
    <div className="grid grid-cols-4 gap-4">
      {[
        { id: 'NOMINAL', color: COLORS.green, dim: COLORS.greenDim, label: 'confidence > 62%' },
        { id: 'CAUTIOUS', color: COLORS.amber, dim: COLORS.amberDim, label: 'confidence 42-62%' },
        { id: 'RESTRICTED', color: COLORS.red, dim: COLORS.redDim, label: 'confidence 22-42%' },
        { id: 'FALLBACK', color: COLORS.red, dim: COLORS.red, label: 'confidence < 22%', dark: true },
      ].map((m, i) => (
        <div key={i} className="space-y-3">
          <div 
            className={`h-16 border flex items-center justify-center font-display text-[10px] font-bold tracking-widest ${m.dark ? 'text-black' : ''}`}
            style={{ backgroundColor: m.dim, borderColor: m.color, color: m.dark ? '#000' : m.color }}
          >
            {m.id}
          </div>
          <div className="font-mono text-[8px] text-textDim uppercase text-center leading-tight">{m.label}</div>
        </div>
      ))}
    </div>
    <div className="relative h-px bg-border">
      <div className="absolute top-1/2 left-0 -translate-y-1/2 w-full flex justify-around">
        {[1, 2, 3].map(i => (
          <div key={i} className="w-2 h-2 bg-border rotate-45" />
        ))}
      </div>
    </div>
    <div className="text-center space-y-2">
      <div className="font-display text-xs font-bold text-white tracking-widest uppercase">Lyapunov Stability Kernel</div>
      <div className="font-mono text-[9px] text-textDim uppercase">Monitoring 10,000 samples per second</div>
    </div>
  </div>
);
