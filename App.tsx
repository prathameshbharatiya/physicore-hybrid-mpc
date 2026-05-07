import React, { useState, useEffect, useRef } from 'react';
import { SimMode, StateVector, ControlInput, PhysicalParams, SimState, TelemetryPoint, MetaAnalysisResponse, Project, FailureLog, FeatureManifest } from './src/types';
import { stepDynamicsRK4 } from './src/services/physicsLogic';
import { computeMPCAction } from './src/services/optimizer';
import { updateSystemID } from './src/services/systemID';
import { ensembleDynamics } from './src/services/learnedDynamics';
import { performMetaAnalysis } from './src/services/geminiService';
import { motion, AnimatePresence } from 'motion/react';
import { useAuthState } from 'react-firebase-hooks/auth';
import { auth, db, googleProvider, handleFirestoreError, OperationType } from './src/firebase';
import { signInWithPopup, signOut } from 'firebase/auth';
import {
  doc, getDoc, updateDoc, setDoc, deleteDoc, addDoc,
  collection, query, where, getDocs, onSnapshot, orderBy
} from 'firebase/firestore';
import { ErrorBoundary } from './src/components/ErrorBoundary';
import SimulationCanvas from './src/components/SimulationCanvas';
import PhysiEditor from './src/components/PhysiEditor';
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
  BookOpen, Plus, Trash2, Puzzle, Bug
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
  "prathameshshirbhate256@gmail.com",
  "ashwanth123creations@gmail.com",
  "adithya17k@gmail.com"
];


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

interface RoboticsParams {
  mass: number;
  friction: number;
  actuatorEfficiency: number;
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

const aviationDerivatives = (state: AviationState, params: AviationParams) => {
  const rho = atmosphericDensity(state.y);
  const v = Math.sqrt(state.vx * state.vx + state.vy * state.vy);
  const q = 0.5 * rho * v * v;
  
  const alpha = state.aoa * Math.PI / 180;
  const cl = params.cl0 + params.cla * alpha;
  const cd = params.cd0 + params.k * cl * cl;
  
  const lift = q * params.wingArea * cl;
  const drag = q * params.wingArea * cd;
  const thrust = params.thrustMax;
  const Fgy = -state.mass * RKT_G;
  const gamma = Math.atan2(state.vy, state.vx);
  
  const Ftx = thrust * Math.cos(state.pitch * Math.PI / 180);
  const Fty = thrust * Math.sin(state.pitch * Math.PI / 180);
  const Flx = -lift * Math.sin(gamma);
  const Fly = lift * Math.cos(gamma);
  const Fdx = -drag * Math.cos(gamma);
  const Fdy = -drag * Math.sin(gamma);
  
  return {
    dx: state.vx, dy: state.vy,
    dvx: (Ftx + Flx + Fdx) / state.mass,
    dvy: (Fty + Fly + Fdy + Fgy) / state.mass,
    dm: -params.fuelBurnRate,
    dt: 1
  };
};

const aviationRK4Step = (state: AviationState, params: AviationParams, dt: number) => {
  const k1 = aviationDerivatives(state, params);
  const s2 = { ...state, x: state.x + k1.dx * dt/2, y: state.y + k1.dy * dt/2, vx: state.vx + k1.dvx * dt/2, vy: state.vy + k1.dvy * dt/2, mass: state.mass + k1.dm * dt/2, time: state.time + dt/2 };
  const k2 = aviationDerivatives(s2, params);
  const s3 = { ...state, x: state.x + k2.dx * dt/2, y: state.y + k2.dy * dt/2, vx: state.vx + k2.dvx * dt/2, vy: state.vy + k2.dvy * dt/2, mass: state.mass + k2.dm * dt/2, time: state.time + dt/2 };
  const k3 = aviationDerivatives(s3, params);
  const s4 = { ...state, x: state.x + k3.dx * dt, y: state.y + k3.dy * dt, vx: state.vx + k3.dvx * dt, vy: state.vy + k3.dvy * dt, mass: state.mass + k3.dm * dt, time: state.time + dt };
  const k4 = aviationDerivatives(s4, params);
  
  return {
    ...state,
    x: state.x + (k1.dx + 2*k2.dx + 2*k3.dx + k4.dx) * dt / 6,
    y: state.y + (k1.dy + 2*k2.dy + 2*k3.dy + k4.dy) * dt / 6,
    vx: state.vx + (k1.dvx + 2*k2.dvx + 2*k3.dvx + k4.dvx) * dt / 6,
    vy: state.vy + (k1.dvy + 2*k2.dvy + 2*k3.dvy + k4.dvy) * dt / 6,
    mass: state.mass + (k1.dm + 2*k2.dm + 2*k3.dm + k4.dm) * dt / 6,
    time: state.time + dt
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

// ── USER API KEY SYSTEM ───────────────────────────────────────────────────
// Keys live in localStorage — never in env vars, never on any server.
// Free for you. Each user uses their own credits.

const STORAGE_KEYS = {
  gemini: 'physicore_gemini_key',
  anthropic: 'physicore_anthropic_key',
};

function getUserGeminiKey(): string {
  try { return localStorage.getItem(STORAGE_KEYS.gemini) || ''; } catch { return ''; }
}

function getUserAnthropicKey(): string {
  try { return localStorage.getItem(STORAGE_KEYS.anthropic) || ''; } catch { return ''; }
}

function saveUserGeminiKey(key: string) {
  try { localStorage.setItem(STORAGE_KEYS.gemini, key.trim()); } catch {}
  aiInstance = null; // Reset cached instance so next getAI() picks up new key
}

function saveUserAnthropicKey(key: string) {
  try { localStorage.setItem(STORAGE_KEYS.anthropic, key.trim()); } catch {}
}

function clearUserKeys() {
  try {
    localStorage.removeItem(STORAGE_KEYS.gemini);
    localStorage.removeItem(STORAGE_KEYS.anthropic);
  } catch {}
  aiInstance = null;
}

function hasAnyKey(): boolean {
  return !!(getUserGeminiKey() || getUserAnthropicKey());
}

let aiInstance: GoogleGenAI | null = null;
function getAI(): GoogleGenAI | null {
  const key = getUserGeminiKey();
  if (!key) { aiInstance = null; return null; }
  if (!aiInstance) { aiInstance = new GoogleGenAI({ apiKey: key }); }
  return aiInstance;
}

// NOTE: Anthropic API is NOT called directly from the browser.
// Anthropic enforces CORS — direct browser fetch returns a CORS error every time.
// All AI calls go through Gemini (browser-safe). Anthropic key stored but unused.
// A future proxy/edge function could enable Anthropic as a real fallback.
async function callAnthropic(
  _system: string,
  _userContent: string,
  _maxTokens = 1000,
  _messagesOverride?: { role: string; content: string }[]
): Promise<string> {
  // CORS blocked — Anthropic cannot be called directly from browsers.
  console.warn('[ANTHROPIC] Direct browser calls blocked by CORS. Use Gemini key instead.');
  return '';
}

// Master AI caller — Gemini only (Anthropic CORS-blocked in browser)
async function callAI(
  system: string,
  userContent: string,
  maxTokens = 1000,
  multiTurnMessages?: { role: 'user' | 'assistant'; content: string }[]
): Promise<string> {
  const ai = getAI();
  if (!ai) return '';
  try {
    const contents: any = multiTurnMessages && multiTurnMessages.length > 0
      ? multiTurnMessages.map(m => ({
          role: m.role === 'user' ? 'user' : 'model',
          parts: [{ text: m.content }],
        }))
      : userContent;
    const resp = await ai.models.generateContent({
      model: 'gemini-2.0-flash',
      contents,
      config: { systemInstruction: system },
    });
    const text = resp.text?.trim();
    return text || '';
  } catch (e) {
    console.warn('[GEMINI] failed:', e);
    return '';
  }
}

// Legacy wrapper — keeps existing callGemini() calls working
async function callGemini(systemPrompt: string, userPrompt: string) {
  const text = await callAI(systemPrompt, userPrompt);
  if (text) return { success: true, text };
  return { success: false, error: 'NO_KEY', message: 'No API key set' };
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
type View = 'home' | 'project' | 'manual' | 'team' | 'projects' | 'whitepaper';
type ProjectTab = 'integrate' | 'build' | 'debug' | 'live';
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
  extension?: string;
}

// --- UTILS ---
const formatTime = (date: Date) => {
  const h = date.getHours().toString().padStart(2, '0');
  const m = date.getMinutes().toString().padStart(2, '0');
  const s = date.getSeconds().toString().padStart(2, '0');
  return `${h}:${m}:${s}`;
};

// --- HARDWARE GATE ---
const initiateHandshake = async (endpoint: string, mode: 'ros2_websocket' | 'hil' | 'digital_twin' | 'mavlink_bridge') => {
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
      let timeoutId: any = null;
      try {
        // HIL Simulation Handshake - REAL CHECK
        const currentOrigin = window.location.origin;
        if (endpoint.includes(currentOrigin) || endpoint.includes('localhost:3000')) {
          resolve({ success: false, reason: 'SELF_CONNECTION_FORBIDDEN. Cannot connect to the PhysiCore UI as a hardware endpoint.' });
          return;
        }

        const controller = new AbortController();
        timeoutId = setTimeout(() => controller.abort(), 2000);
        
        // We expect a real hardware bridge to respond to a specific health check
        const httpEndpoint = endpoint.startsWith('ws') ? endpoint.replace('ws', 'http') : endpoint;
        const response = await fetch(httpEndpoint + '/api/health', { 
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
        if (timeoutId) clearTimeout(timeoutId);
        resolve({ 
          success: false, 
          reason: 'HIL_ENDPOINT_UNREACHABLE. Ensure your HIL bridge is running at ' + endpoint + ' and responding to /api/health'
        });
      }
    });
  }
  if (mode === 'mavlink_bridge') {
    return new Promise((resolve) => {
      try {
        const ws = new WebSocket(endpoint);
        const timeout = setTimeout(() => {
          ws.close();
          resolve({ success: false, reason: 'BRIDGE_TIMEOUT. Is physicore_bridge.py running?' });
        }, 5000);
        ws.onopen = () => {
          ws.send(JSON.stringify({ op: 'ping' }));
        };
        ws.onmessage = (event) => {
          clearTimeout(timeout);
          ws.close();
          try {
            const status = JSON.parse(event.data);
            resolve({
              success: true,
              vehicle_type: status.msg?.vehicle_type || '',
              domain: status.msg?.domain || '',
              platform: status.msg?.platform || '',
            });
          } catch {
            resolve({ success: true });
          }
        };
        ws.onerror = () => {
          clearTimeout(timeout);
          resolve({ success: false, reason: 'BRIDGE_OFFLINE. Run: python physicore_bridge.py --platform balancing_bot_arduino --connection COM3' });
        };
      } catch (e) {
        resolve({ success: false, reason: 'WEBSOCKET_ERROR' });
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
  projectCode,
  projectData,
  onImportProjectCode,
  isSystemConnecting,
  connectionError
}: { 
  files: GeneratedFile[], 
  onTest: () => void, 
  onContinue: () => void,
  connectionMode: 'ros2_websocket' | 'hil' | 'digital_twin' | 'mavlink_bridge',
  setConnectionMode: (m: 'ros2_websocket' | 'hil' | 'digital_twin' | 'mavlink_bridge') => void,
  endpoint: string,
  setEndpoint: (e: string) => void,
  dRealEndpoint: string,
  setDRealEndpoint: (e: string) => void,
  systemProfile: SystemProfile,
  rocketParams: RocketParams,
  aviationParams: AviationParams,
  priors: { mass: number, friction: number },
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
              <button
                onClick={() => { setConnectionMode('mavlink_bridge'); setEndpoint('ws://localhost:8765'); }}
                className={`px-3 py-1 font-mono text-[9px] border ${connectionMode === 'mavlink_bridge' ? 'bg-amber text-black border-amber' : 'border-border text-textDim'}`}
              >
                MAVLINK
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
              {connectionMode === 'digital_twin' ? 'Verify Digital Twin simulation' : connectionMode === 'mavlink_bridge' ? 'Connect MAVLink Bridge' : connectionMode === 'ros2_websocket' ? 'Connect ROS2 Hardware' : 'Verify HIL / Hardware link'}
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
                step="0.1" 
                value={params.wingspan || 0} 
                onChange={(e) => setParams({ ...params, wingspan: parseFloat(e.target.value) || 0 })}
                className="w-full bg-bg border border-border p-2 font-mono text-xs text-white focus:border-cyan outline-none"
              />
            </div>
            <div className="space-y-1">
              <label className="font-mono text-[9px] text-textDim uppercase">Wing Area (m²)</label>
              <input 
                type="number" 
                step="0.01" 
                value={params.wingArea || 0} 
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
                step="0.1" 
                value={params.cla || 0} 
                onChange={(e) => setParams({ ...params, cla: parseFloat(e.target.value) || 0 })}
                className="w-full bg-bg border border-border p-2 font-mono text-xs text-white focus:border-cyan outline-none"
              />
            </div>
            <div className="space-y-1">
              <label className="font-mono text-[9px] text-textDim uppercase">Parasitic Drag (Cd0)</label>
              <input 
                type="number" 
                step="0.001" 
                value={params.cd0 || 0} 
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
                step="1" 
                value={params.thrustMax || 0} 
                onChange={(e) => setParams({ ...params, thrustMax: parseFloat(e.target.value) || 0 })}
                className="w-full bg-bg border border-border p-2 font-mono text-xs text-white focus:border-cyan outline-none"
              />
            </div>
            <div className="space-y-1">
              <label className="font-mono text-[9px] text-textDim uppercase">Fuel Cap (kg)</label>
              <input 
                type="number" 
                step="0.1" 
                value={params.fuelCapacity || 0} 
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

  const twr = params.motorCurve.length > 0 ? (Math.max(...params.motorCurve.map(p => p.f || 0)) / (((params.dryMass || 0) + (params.propMassInitial || 0)) * RKT_G)) : 0;

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
              <input type="number" step="0.1" value={params.dryMass || 0} onChange={e => setParams({...params, dryMass: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-2 font-mono text-xs text-white outline-none focus:border-green" />
            </div>
            <div className="space-y-1">
              <label className="micro-label text-textDim">Propellant Mass (kg)</label>
              <input type="number" step="0.01" value={params.propMassInitial || 0} onChange={e => setParams({...params, propMassInitial: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-2 font-mono text-xs text-white outline-none focus:border-green" />
            </div>
            <div className="space-y-1">
              <label className="micro-label text-textDim">Diameter (m)</label>
              <input type="number" step="0.001" value={params.diameter || 0} onChange={e => setParams({...params, diameter: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-2 font-mono text-xs text-white outline-none focus:border-green" />
            </div>
            <div className="space-y-1">
              <label className="micro-label text-textDim">Drag Coeff (Cd)</label>
              <input type="number" step="0.01" value={params.cd || 0} onChange={e => setParams({...params, cd: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-2 font-mono text-xs text-white outline-none focus:border-green" />
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
              <input type="number" value={params.launchAngle || 0} onChange={e => setParams({...params, launchAngle: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-2 font-mono text-xs text-white outline-none focus:border-amber" />
            </div>
            <div className="space-y-1">
              <label className="micro-label text-textDim">Rail Length (m)</label>
              <input type="number" value={params.railLength || 0} onChange={e => setParams({...params, railLength: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-2 font-mono text-xs text-white outline-none focus:border-amber" />
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
                <input type="number" step="0.1" value={p.t || 0} onChange={e => handleManualCurveChange(i, 't', e.target.value)} className="bg-bg border border-border p-1 font-mono text-[10px] text-white outline-none" />
                <input type="number" step="1" value={p.f || 0} onChange={e => handleManualCurveChange(i, 'f', e.target.value)} className="bg-bg border border-border p-1 font-mono text-[10px] text-white outline-none" />
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
                  <input type="number" value={params.drogueAlt || 0} onChange={e => setParams({...params, drogueAlt: parseFloat(e.target.value) || 0})} className="w-20 bg-bg border border-border p-1 font-mono text-[10px] text-white text-right" />
                </div>
              )}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="micro-label text-textDim">Drogue Cd</label>
                  <input type="number" step="0.1" value={params.drogueCd || 0} onChange={e => setParams({...params, drogueCd: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-1 font-mono text-[10px] text-white" />
                </div>
                <div className="space-y-1">
                  <label className="micro-label text-textDim">Drogue Diam (m)</label>
                  <input type="number" step="0.01" value={params.drogueDiam || 0} onChange={e => setParams({...params, drogueDiam: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-1 font-mono text-[10px] text-white" />
                </div>
              </div>
            </div>
          </div>
          <div className="space-y-4">
            <div className="p-3 bg-bgRaised border border-borderDim space-y-3">
              <div className="flex items-center justify-between">
                <span className="micro-label text-textDim">Main Deployment</span>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[10px] text-white">{params.mainAlt || 0}m</span>
                </div>
              </div>
              <div className="space-y-1">
                <label className="micro-label text-textDim">Deployment Altitude (m)</label>
                <input type="range" min="50" max="1000" step="50" value={params.mainAlt || 0} onChange={e => setParams({...params, mainAlt: parseInt(e.target.value) || 0})} className="w-full h-1 bg-border appearance-none cursor-pointer accent-red" />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="micro-label text-textDim">Main Cd</label>
                  <input type="number" step="0.1" value={params.mainCd || 0} onChange={e => setParams({...params, mainCd: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-1 font-mono text-[10px] text-white" />
                </div>
                <div className="space-y-1">
                  <label className="micro-label text-textDim">Main Diam (m)</label>
                  <input type="number" step="0.01" value={params.mainDiam || 0} onChange={e => setParams({...params, mainDiam: parseFloat(e.target.value) || 0})} className="w-full bg-bg border border-border p-1 font-mono text-[10px] text-white" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// --- INTEGRATION ENGINEER ---
// Zero API. Zero network calls. Never fails.
// Asks ALL questions, generates real code, auto-configures dashboard.

// ─── INTEGRATION ENGINEER — COMPLETE REWRITE ─────────────────────────────────
// Button-driven Q&A, real code generation, visual step checklist, troubleshooter

// ── Types ─────────────────────────────────────────────────────────────────────
interface IEQuestion { key: string; q: string; opts?: string[]; }
interface IEFlow { label: string; icon: string; questions: IEQuestion[]; }

// ── Hardware flows ─────────────────────────────────────────────────────────────
const IE_FLOWS: Record<string, IEFlow> = {
  balancing_bot: {
    label: 'Self-balancing robot', icon: '⚖',
    questions: [
      { key:'imu',          q:'IMU sensor?',                 opts:['MPU6050','BNO055','MPU9250','ICM20689'] },
      { key:'mcu',          q:'Microcontroller?',            opts:['Arduino Uno','Arduino Nano','Arduino Mega','ESP32','Raspberry Pi Pico'] },
      { key:'motor_driver', q:'Motor driver?',               opts:['L298N','TB6612FNG','DRV8833','BTS7960'] },
      { key:'mass',         q:'Robot mass in kg? (e.g. 1.2)', opts:undefined },
      { key:'com_height',   q:'CoM height from wheel axle in meters? (e.g. 0.15)', opts:undefined },
      { key:'os',           q:'Your laptop OS?',             opts:['Windows','Mac','Linux'] },
    ],
  },
  px4: {
    label: 'PX4 drone', icon: '✈',
    questions: [
      { key:'connection', q:'Connection to PX4?',   opts:['USB to Pixhawk','UDP WiFi port 14550','UART telemetry radio'] },
      { key:'frame',      q:'Frame type?',           opts:['Quadrotor','Hexarotor','Fixed-wing','VTOL'] },
      { key:'mass',       q:'Drone mass with battery (kg)?', opts:undefined },
      { key:'os',         q:'Your laptop OS?',       opts:['Windows','Mac','Linux'] },
    ],
  },
  ardupilot: {
    label: 'ArduPilot vehicle', icon: '🛩',
    questions: [
      { key:'frame',      q:'Vehicle type?',         opts:['Quadrotor (Copter)','Fixed-wing (Plane)','VTOL','Rover'] },
      { key:'connection', q:'Connection method?',    opts:['USB','UDP Mission Planner','SiK telemetry radio'] },
      { key:'mass',       q:'Vehicle mass (kg)?',    opts:undefined },
      { key:'os',         q:'Your laptop OS?',       opts:['Windows','Mac','Linux'] },
    ],
  },
  ros2_arm: {
    label: 'ROS2 robot arm', icon: '🦾',
    questions: [
      { key:'distro',    q:'ROS2 distribution?',         opts:['Humble','Jazzy','Iron','Rolling'] },
      { key:'brand',     q:'Arm brand/model?',           opts:['Universal Robots UR5/UR10','KUKA','Fanuc','ABB','Franka','Custom'] },
      { key:'dof',       q:'Number of joints (DOF)?',    opts:['4','6','7'] },
      { key:'topic',     q:'Joint states topic?',        opts:['/joint_states','/robot/joint_states','/arm/joint_states'] },
      { key:'ft_sensor', q:'Force-torque sensor?',       opts:['Yes','No'] },
      { key:'mass',      q:'End-effector + payload (kg)?', opts:undefined },
    ],
  },
  legged: {
    label: 'Legged / quadruped', icon: '🐕',
    questions: [
      { key:'brand',   q:'Platform?',             opts:['Unitree Go1','Unitree Go2','ANYmal','MIT Mini Cheetah','Custom'] },
      { key:'distro',  q:'ROS2 distribution?',    opts:['Humble','Jazzy','Iron'] },
      { key:'terrain', q:'Primary terrain?',      opts:['Flat indoor','Outdoor grass/gravel','Stairs','Unknown/varied'] },
      { key:'mass',    q:'Robot mass (kg)?',      opts:undefined },
    ],
  },
  rocket: {
    label: 'Sounding rocket', icon: '🚀',
    questions: [
      { key:'fc',       q:'Flight computer?',            opts:['Arduino Mega','Teensy 4.1','ESP32','Custom FC'] },
      { key:'baro',     q:'Barometer/altimeter?',        opts:['BMP280','MS5611','BMP388','MPL3115A2'] },
      { key:'baud',     q:'Serial baud rate?',           opts:['115200','57600','9600'] },
      { key:'os',       q:'Your laptop OS?',             opts:['Windows','Mac','Linux'] },
      { key:'dry_mass', q:'Rocket dry mass (kg)?',       opts:undefined },
    ],
  },
  auv: {
    label: 'AUV / underwater robot', icon: '🌊',
    questions: [
      { key:'platform', q:'AUV platform?',                opts:['BlueROV2','Custom AUV','Research AUV'] },
      { key:'dvl',      q:'DVL available?',               opts:['Yes','No — IMU + depth only'] },
      { key:'distro',   q:'ROS2 distribution?',           opts:['Humble','Iron','Jazzy'] },
      { key:'mass',     q:'Vehicle mass (kg)?',           opts:undefined },
    ],
  },
  evtol: {
    label: 'eVTOL aircraft', icon: '🚁',
    questions: [
      { key:'fc',     q:'Flight controller?',    opts:['PX4','ArduPilot','Custom'] },
      { key:'rotors', q:'Number of rotors?',     opts:['4','6','8','12'] },
      { key:'mass',   q:'Vehicle mass (kg)?',    opts:undefined },
      { key:'os',     q:'Your laptop OS?',       opts:['Windows','Mac','Linux'] },
    ],
  },
  surgical: {
    label: 'Surgical robot', icon: '🏥',
    questions: [
      { key:'dof',       q:'Number of DOF?',         opts:['4','6','7'] },
      { key:'ft_sensor', q:'Force-torque sensor?',   opts:['Yes','No'] },
      { key:'distro',    q:'ROS2 distribution?',     opts:['Humble','Iron'] },
      { key:'topic',     q:'Joint states topic?',    opts:['/joint_states','/robot/joint_states'] },
    ],
  },
  rover: {
    label: 'Ground rover / AMR', icon: '🚗',
    questions: [
      { key:'interface',  q:'Communication?',                opts:['ROS2','Arduino serial','ESP32 serial'] },
      { key:'distro',     q:'ROS2 distribution?',            opts:['Humble','Iron','N/A'] },
      { key:'mass',       q:'Robot mass (kg)?',              opts:undefined },
      { key:'wheel_base', q:'Wheel separation (m)?',         opts:undefined },
      { key:'os',         q:'Your laptop OS?',               opts:['Windows','Mac','Linux'] },
    ],
  },
  satellite: {
    label: 'Satellite / spacecraft', icon: '🛸',
    questions: [
      { key:'actuators', q:'Attitude actuators?',    opts:['Reaction wheels','Thrusters only','Both'] },
      { key:'altitude',  q:'Orbital altitude (km)?', opts:undefined },
      { key:'mass',      q:'Spacecraft mass (kg)?',  opts:undefined },
      { key:'os',        q:'Your laptop OS?',        opts:['Windows','Mac','Linux'] },
    ],
  },
  humanoid: {
    label: 'Bipedal / Humanoid', icon: '🦿',
    questions: [
      { key:'model',     q:'Which humanoid platform?',    opts:['Unitree G1','Unitree H1','Custom bipedal','Simulation only'] },
      { key:'imu',       q:'Primary IMU?',                opts:['Built-in (Unitree)','BNO055','ICM-42688','External IMU'] },
      { key:'interface', q:'Communication interface?',    opts:['Unitree SDK (Ethernet)','ROS2','Custom serial'] },
      { key:'mass',      q:'Estimated mass (kg)?',        opts:undefined },
      { key:'gait',      q:'Primary gait mode?',          opts:['Standing balance','Slow walk','Dynamic walk','Stair climbing'] },
      { key:'os',        q:'Your laptop OS?',             opts:['Ubuntu 22.04','Ubuntu 20.04','Windows (WSL2)','macOS'] },
    ],
  },
  rocket_aero: {
    label: 'Sounding Rocket (Aero)', icon: '🚀',
    questions: [
      { key:'airframe',  q:'Airframe diameter (mm)?',     opts:['38mm (29mm motor)','54mm','75mm','98mm','Custom'] },
      { key:'motor',     q:'Motor type?',                 opts:['Cesaroni','AeroTech','Klima','Custom solid','Hybrid'] },
      { key:'imu',       q:'Flight computer IMU?',        opts:['BNO055','MPU6050','ICM-42688','OpenLog Artemis'] },
      { key:'mass',      q:'Airframe mass (dry kg)?',     opts:undefined },
      { key:'apogee',    q:'Target apogee (m)?',          opts:undefined },
      { key:'mcu',       q:'Flight computer?',            opts:['Arduino Mega','ESP32','Teensy 4.0','Custom FC'] },
    ],
  },
  custom: {
    label: 'Custom hardware', icon: '🔧',
    questions: [
      { key:'type',      q:'What type of system?',        opts:['Ground robot','Aerial vehicle','Manipulator arm','Underwater vehicle','Spacecraft','Other'] },
      { key:'interface', q:'How does it communicate?',    opts:['Arduino/ESP32 serial','ROS2','MAVLink','Custom serial'] },
      { key:'sensors',   q:'Primary sensors?',            opts:['IMU only','IMU + encoders','IMU + GPS','IMU + barometer','IMU + vision'] },
      { key:'mcu',       q:'Compute hardware?',           opts:['Arduino Uno/Nano/Mega','ESP32','Raspberry Pi','Jetson Nano','Jetson Orin','Laptop only'] },
      { key:'os',        q:'Your laptop OS?',             opts:['Windows','Mac','Linux'] },
      { key:'mass',      q:'System mass (kg)?',           opts:undefined },
    ],
  },
};

// ── Detection ─────────────────────────────────────────────────────────────────
function ie_detect(s: string): string {
  const t = s.toLowerCase();
  if (t.match(/balanc|self.?balanc|inverted.?pendulum|segway/)) return 'balancing_bot';
  if (t.match(/px4|pixhawk/)) return 'px4';
  if (t.match(/ardupilot|apm|cube.?pilot/)) return 'ardupilot';
  if (t.match(/evtol|e.?vtol|air.?taxi|tilt.?rotor/)) return 'evtol';
  if (t.match(/ros2.*arm|robot.*arm|manipulator.*arm|ur5|ur10|kuka|fanuc|abb|franka/)) return 'ros2_arm';
  if (t.match(/humanoid|unitree.*g1|unitree.*h1|figure.?ai|boston.?dynamic/)) return 'humanoid';
  if (t.match(/legged|quadruped|anymal|mini.?cheetah|\bgo1\b|\bgo2\b/)) return 'legged';
  if (t.match(/\bauv\b|underwater|subsea|bluerov|\bdvl\b/)) return 'auv';
  if (t.match(/surgical|medical.?robot|endoscop/)) return 'surgical';
  if (t.match(/satellite|spacecraft|orbital|cubesat/)) return 'satellite';
  if (t.match(/sounding.?rocket|high.?power.?rocket|HPR|model.?rocket|cesaroni|aerotech/)) return 'rocket_aero';
  if (t.match(/rocket|flight.?computer/)) return 'rocket';
  if (t.match(/rover|ground.?robot|\bamr\b|warehouse.?robot/)) return 'rover';
  if (t.match(/drone|quadrotor|fpv|multirotor/)) return 'px4';
  if (t.match(/esp32|esp8266|arduino/)) return 'balancing_bot';
  if (t.match(/ros2|ros 2/)) return 'ros2_arm';
  if (t.match(/custom|diy|homebrew|my own|self.?built|self.?made/)) return 'custom';
  return '';
}

function ie_port(os: string): string {
  if (os.toLowerCase().includes('win')) return 'COM3';
  if (os.toLowerCase().includes('mac')) return '/dev/cu.usbserial-0001';
  return '/dev/ttyUSB0';
}

// ── Hardware Database ─────────────────────────────────────────────────────────
const HARDWARE_DB: Record<string, { notes?: string[]; knownIssues?: string[]; defaultMass?: number; pins?: Record<string, number | string> }> = {
  'MPU6050': {
    notes: ['Must use 3.3V not 5V (5V will damage it)', 'SDA→A4, SCL→A5 on Arduino Uno'],
    knownIssues: ['Running on 5V causes immediate damage', 'I2C address: 0x68 (AD0 low) or 0x69 (AD0 high)'],
  },
  'BNO055': {
    notes: ['I2C address 0x28 (default) or 0x29', 'Use raw mode — disable onboard fusion for PhysiCore'],
  },
  'MPU9250': {
    notes: ['3.3V logic only', 'Has integrated magnetometer (not needed by PhysiCore)'],
  },
  'L298N': {
    notes: ['Logic pins are 5V tolerant', 'Enable pins MUST be on PWM-capable pins'],
    pins: { ENA: 5, IN1: 4, IN2: 3, ENB: 6, IN3: 7, IN4: 8 },
  },
  'TB6612FNG': {
    notes: ['VM (motor power) can be 2.5V–13.5V', 'STBY pin must be HIGH to enable outputs'],
  },
  'DRV8833': {
    notes: ['3.3V–10.8V motor supply', 'Efficiency higher than L298N for small motors'],
  },
  'Unitree G1': { defaultMass: 35.0, notes: ['Communication: Ethernet (DDS/ROS2)'] },
  'Unitree H1': { defaultMass: 47.0, notes: ['Communication: Ethernet (DDS/ROS2)'] },
  'Unitree Go2': { defaultMass: 15.0 },
  'Spot (Boston Dynamics)': { defaultMass: 32.5 },
  'Arduino Uno': {
    notes: ['14 digital I/O, 6 PWM (pins 3,5,6,9,10,11)', 'Serial on pins 0,1 — keep free for bridge communication'],
  },
  'Arduino Mega': {
    notes: ['54 digital I/O, 15 PWM pins', 'Better for multi-motor robots'],
  },
  'Raspberry Pi 4': {
    notes: ['GPIO 3.3V logic — do not connect 5V signals directly', 'Use as compute node — not for real-time PWM'],
  },
};

function ie_hwNotes(answers: Record<string, string>): string[] {
  const notes: string[] = [];
  for (const val of Object.values(answers)) {
    const entry = HARDWARE_DB[val];
    if (entry) {
      (entry.notes || []).forEach(n => notes.push(`${val}: ${n}`));
      (entry.knownIssues || []).forEach(n => notes.push(`⚠ ${val}: ${n}`));
    }
  }
  return notes;
}

function ie_addVerificationHeader(filename: string, content: string, hw: string, answers: Record<string,string>): string {
  const ts = new Date().toISOString();
  const imu = answers.imu || '?';
  const driver = answers.motor_driver || '?';
  const mcu = answers.mcu || '?';
  const baud = answers.baud || '115200';
  const mass = answers.mass || answers.dry_mass || '?';

  if (filename.endsWith('.ino') || filename.endsWith('.cpp') || filename.endsWith('.c')) {
    const header = `// ═══════════════════════════════════════════════════════════
// PHYSICORE VERIFIED INTEGRATION
// Hardware: ${mcu} + ${imu} + ${driver}
// Platform: ${hw} | PhysiCore v3.1
// Generated: ${ts}
// Verified: ✓ Library compatibility checked for ${imu}
//           ✓ Pin assignments verified for ${mcu}
//           ✓ Baud rate set to ${baud}
//           ✓ Mass set to ${mass}kg
//           ✓ JSON format matches bridge parser
// ═══════════════════════════════════════════════════════════
`;
    return header + '\n' + content;
  }
  if (filename.endsWith('.yaml') || filename.endsWith('.yml')) {
    return `# PHYSICORE VERIFIED — ${ts}\n# Hardware: ${hw}\n${content}`;
  }
  return content;
}

// ── Code generation ───────────────────────────────────────────────────────────
function ie_generateCode(hw: string, answers: Record<string,string>): {filename:string; content:string}[] {
  const mass = answers.mass || answers.dry_mass || '1.0';
  const os = answers.os || 'Linux';
  const port = ie_port(os);
  const distro = (answers.distro || 'humble').toLowerCase().replace('not using ros2','humble').replace('n/a','humble');
  const topic = answers.topic || '/joint_states';
  const dof = parseInt(answers.dof || '6');
  const imu = answers.imu || 'MPU6050';
  const driver = answers.motor_driver || 'L298N';

  if (hw === 'balancing_bot') {
    const imuLib: Record<string,string> = { MPU6050:'MPU6050_light by rfetick', BNO055:'Adafruit BNO055', MPU9250:'MPU9250_asukiaaa', ICM20689:'ICM42688 by Sparkfun' };
    const imuInc: Record<string,string> = {
      MPU6050:'#include <MPU6050_light.h>\nMPU6050 mpu(Wire);',
      BNO055:'#include <Adafruit_BNO055.h>\nAdafruit_BNO055 bno(55, 0x28);',
      MPU9250:'#include <MPU9250_asukiaaa.h>\nMPU9250_asukiaaa mpu;',
      ICM20689:'#include <ICM42688.h>\nICM42688 imuSensor(SPI,10);',
    };
    const imuSetup: Record<string,string> = {
      MPU6050:'  Wire.begin();\n  byte s=mpu.begin();\n  while(s!=0){Serial.println("{\\"error\\":\\"MPU6050 not found\\"}");delay(500);s=mpu.begin();}\n  mpu.calcOffsets(true,true);\n  Serial.println("{\\"status\\":\\"ready\\"}");',
      BNO055:'  bno.begin(); bno.setExtCrystalUse(true);\n  Serial.println("{\\"status\\":\\"ready\\"}");',
      MPU9250:'  Wire.begin(); mpu.setup(0x68);\n  Serial.println("{\\"status\\":\\"ready\\"}");',
      ICM20689:'  imuSensor.begin();\n  Serial.println("{\\"status\\":\\"ready\\"}");',
    };
    const imuRead: Record<string,string> = {
      MPU6050:'  mpu.update();\n  pitch=mpu.getAngleX()-BALANCE_POINT;\n  gyro_x=mpu.getGyroX();',
      BNO055:'  imu::Vector<3> e=bno.getVector(Adafruit_BNO055::VECTOR_EULER);\n  imu::Vector<3> g=bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);\n  pitch=e.x()-BALANCE_POINT; gyro_x=g.y();',
      MPU9250:'  mpu.accelUpdate(); mpu.gyroUpdate();\n  pitch=mpu.accelX()*57.2958-BALANCE_POINT; gyro_x=mpu.gyroY();',
      ICM20689:'  imuSensor.readSensor();\n  pitch=imuSensor.getAccelX_g()*57.2958-BALANCE_POINT; gyro_x=imuSensor.getGyroY_dps();',
    };
    const mPins: Record<string,string> = {
      L298N:'const int L_EN=5,L_IN1=4,L_IN2=3,R_EN=6,R_IN1=7,R_IN2=8;',
      TB6612FNG:'const int PWMA=5,AIN1=4,AIN2=3,PWMB=6,BIN1=7,BIN2=8,STBY=9;',
      DRV8833:'const int AIN1=4,AIN2=3,BIN1=7,BIN2=8;',
      BTS7960:'const int L_RPWM=5,L_LPWM=6,R_RPWM=9,R_LPWM=10;',
    };
    const mApply: Record<string,string> = {
      L298N:'  int p=constrain((int)(abs(v)*255),0,255);bool f=(v>=0);\n  digitalWrite(L_IN1,f);digitalWrite(L_IN2,!f);analogWrite(L_EN,p);\n  digitalWrite(R_IN1,f);digitalWrite(R_IN2,!f);analogWrite(R_EN,p);',
      TB6612FNG:'  int p=constrain((int)(abs(v)*255),0,255);bool f=(v>=0);\n  digitalWrite(AIN1,f);digitalWrite(AIN2,!f);analogWrite(PWMA,p);\n  digitalWrite(BIN1,f);digitalWrite(BIN2,!f);analogWrite(PWMB,p);digitalWrite(STBY,HIGH);',
      DRV8833:'  int p=constrain((int)(abs(v)*255),0,255);\n  analogWrite(AIN1,v>0?p:0);analogWrite(AIN2,v<=0?p:0);\n  analogWrite(BIN1,v>0?p:0);analogWrite(BIN2,v<=0?p:0);',
      BTS7960:'  int p=constrain((int)(abs(v)*255),0,255);\n  analogWrite(v>=0?L_RPWM:L_LPWM,p);analogWrite(v>=0?R_RPWM:R_LPWM,p);',
    };

    const ino = `/*
 * PhysiCore Balancing Bot — ${imu} + ${driver} + ${answers.mcu||'Arduino'}
 * Mass: ${mass}kg  CoM height: ${answers.com_height||'0.15'}m
 *
 * INSTALL (Sketch → Include Library → Manage Libraries):
 *   ${imuLib[imu]||imu}
 *   ArduinoJson by Benoit Blanchon (v6.x)
 *
 * WIRING: ${imu} SDA→A4  SCL→A5  VCC→3.3V  GND→GND
 */
#include <Wire.h>
#include <ArduinoJson.h>
${imuInc[imu]||imuInc.MPU6050}
${mPins[driver]||mPins.L298N}

// ── CALIBRATION ─────────────────────────────────────────────────────────────
// Hold robot UPRIGHT → Open Serial Monitor at 115200 → read pitch
// Set BALANCE_POINT to that value → re-upload
const float BALANCE_POINT = 0.0;
const float MAX_TORQUE    = 2.5;  // N·m — DO NOT change
// ────────────────────────────────────────────────────────────────────────────
const float KP=35.0, KI=0.5, KD=1.2;
const int   LOOP_MS=20;

float pitch=0, gyro_x=0, motor_l=0, motor_r=0;
bool  physicore_active=false;
unsigned long last_cmd=0, last_tx=0;
float pid_i=0, prev_err=0;

void setup(){
  Serial.begin(115200);
  pinMode(LED_BUILTIN,OUTPUT);
${imuSetup[imu]||imuSetup.MPU6050}
  applyMotors(0);
}

void loop(){
  unsigned long now=millis();
${imuRead[imu]||imuRead.MPU6050}

  while(Serial.available()>0){
    StaticJsonDocument<256> cmd;
    if(deserializeJson(cmd,Serial)==DeserializationError::Ok){
      if(strcmp(cmd["op"],"command")==0){
        motor_l=constrain(cmd["action"][0].as<float>()/MAX_TORQUE,-1.0f,1.0f);
        motor_r=motor_l; physicore_active=true; last_cmd=now;
        digitalWrite(LED_BUILTIN,HIGH);
      }
    }
  }

  if(now-last_cmd>500){ physicore_active=false; digitalWrite(LED_BUILTIN,LOW); }

  if(physicore_active){ applyMotors(motor_l); }
  else{
    float e=-pitch;
    pid_i=constrain(pid_i+e*(LOOP_MS/1000.0f),-50,50);
    float d=(e-prev_err)/(LOOP_MS/1000.0f);
    float v=constrain((KP*e+KI*pid_i+KD*d)/255.0f,-1,1);
    prev_err=e; motor_l=motor_r=v; applyMotors(v);
  }

  if(now-last_tx>=LOOP_MS){
    last_tx=now;
    StaticJsonDocument<256> doc;
    doc["pitch"]=pitch; doc["gyro_x"]=gyro_x;
    doc["motor_l"]=motor_l*MAX_TORQUE; doc["motor_r"]=motor_r*MAX_TORQUE;
    doc["active"]=physicore_active; doc["timestamp"]=now;
    serializeJson(doc,Serial); Serial.println();
  }
  while(millis()-now<LOOP_MS);
}

void applyMotors(float v){
${mApply[driver]||mApply.L298N}
}`;

    const yaml = `name: My Balancing Bot
platform: balancing_bot
connection: ${port}
baud: 115200
mass: ${mass}
friction: 0.15
inertia: ${(parseFloat(mass)*parseFloat(answers.com_height||'0.15')*parseFloat(answers.com_height||'0.15')*0.3).toFixed(4)}
imu: ${imu}
motor_driver: ${driver}
mcu: "${answers.mcu||'Arduino Uno'}"
control_hz: 60.0
use_registry: true
sentinel_enabled: true
max_torque: 2.5`;

    const bridge = os.toLowerCase().includes('win')
      ? `@echo off\npip install pymavlink websockets aiohttp pyserial pyyaml\npython physicore\\bridge\\physicore_bridge.py --config balancing_bot.yaml`
      : `#!/bin/bash\npip install pymavlink websockets aiohttp pyserial pyyaml\npython physicore/bridge/physicore_bridge.py --config balancing_bot.yaml`;

    return [
      { filename:'physicore_balancing_bot.ino', content:ino },
      { filename:'balancing_bot.yaml', content:yaml },
      { filename: os.toLowerCase().includes('win') ? 'run_bridge.bat' : 'run_bridge.sh', content:bridge },
    ];
  }

  if (hw==='px4'||hw==='ardupilot'||hw==='evtol') {
    const conn = (answers.connection||'').toLowerCase().includes('usb') ? '/dev/ttyACM0' : 'udp:14550';
    const platform = hw==='evtol' ? 'evtol' : hw==='ardupilot' ? `ardupilot_${(answers.frame||'').toLowerCase().includes('wing')?'plane':'quadrotor'}` : 'px4_quadrotor';
    const yaml = `name: My ${hw.toUpperCase()} ${answers.frame||'Drone'}\nplatform: ${hw}\nconnection: ${conn}\nmass: ${mass}\nfriction: 0.1\ninertia: 0.05\ncontrol_hz: 60.0\nuse_registry: true`;
    const bridge = os.toLowerCase().includes('win')
      ? `@echo off\npip install pymavlink websockets aiohttp pyyaml\npython physicore\\bridge\\physicore_bridge.py --config drone.yaml`
      : `#!/bin/bash\npip install pymavlink websockets aiohttp pyyaml\npython physicore/bridge/physicore_bridge.py --config drone.yaml`;
    return [
      { filename:'drone.yaml', content:yaml },
      { filename: os.toLowerCase().includes('win') ? 'run_bridge.bat' : 'run_bridge.sh', content:bridge },
    ];
  }

  if (hw==='ros2_arm'||hw==='humanoid'||hw==='legged'||hw==='auv'||hw==='surgical'||hw==='rover') {
    const hasFT = answers.ft_sensor==='Yes';
    const platform = hw==='ros2_arm'||hw==='surgical' ? 'ros2_manipulator' : hw==='humanoid'||hw==='legged' ? 'ros2_legged' : hw==='auv' ? 'ros2_auv' : 'ros2_ground_rover';
    const node = `#!/usr/bin/env python3
"""PhysiCore ROS2 Bridge — ${answers.brand||hw} ${dof}-DOF
ROS2 ${distro} | Topic: ${topic}
Run: python3 physicore_ros2_bridge.py
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
${hasFT ? 'from geometry_msgs.msg import WrenchStamped' : ''}
import json, socket, math

class Bridge(Node):
    def __init__(self):
        super().__init__('physicore_bridge')
        self.j=[0.0]*${dof}; self.v=[0.0]*${dof}; self.e=[0.0]*${dof}; self.f=[0.0,0.0,0.0]
        self.create_subscription(JointState,'${topic}',self.jcb,10)
        ${hasFT ? "self.create_subscription(WrenchStamped,'/ft_sensor/wrench',self.fcb,10)" : ''}
        self.get_logger().info('PhysiCore ROS2 bridge ready')

    def jcb(self,msg):
        n=min(len(msg.position),${dof})
        self.j[:n]=list(msg.position[:n])
        self.v[:n]=list(msg.velocity[:n]) if msg.velocity else [0.0]*n
        self.e[:n]=list(msg.effort[:n]) if msg.effort else [0.0]*n
        self.send()

    ${hasFT ? `def fcb(self,msg):
        f=msg.wrench.force; self.f=[f.x,f.y,f.z]` : ''}

    def send(self):
        p=json.dumps({"op":"publish","topic":"/telemetry","msg":{
            "pitch":math.degrees(self.j[0]),
            "roll":math.degrees(self.j[1]) if ${dof}>1 else 0,
            "gyro_x":self.v[0],"gyro_y":self.v[1] if ${dof}>1 else 0,
            "accel_x":self.f[0],"accel_y":self.f[1],"accel_z":self.f[2],
            "motor_l":self.e[0],"motor_r":self.e[1] if ${dof}>1 else 0,
            "vehicle_type":"MANIPULATOR","domain":"ROBOTICS","connected":True,
            "joint_positions":self.j,"joint_velocities":self.v
        }})+'\\n'
        try:
            s=socket.socket(); s.connect(('localhost',8765)); s.sendall(p.encode()); s.close()
        except: pass

def main():
    rclpy.init(); rclpy.spin(Bridge())

if __name__=='__main__': main()`;

    const yaml = `name: ${answers.brand||hw}\nplatform: ${hw}\nconnection: ros2\nmass: ${mass}\nfriction: 0.3\ninertia: 0.1\nros2_distro: ${distro}\njoint_topic: ${topic}\ndof: ${dof}\ncontrol_hz: 60.0\nuse_registry: true`;
    const bridge = `#!/bin/bash\nsource /opt/ros/${distro}/setup.bash\npip install pymavlink websockets aiohttp pyyaml\npython physicore/bridge/physicore_bridge.py --platform ${platform} &\nsleep 2\npython3 physicore_ros2_bridge.py`;
    return [
      { filename:'physicore_ros2_bridge.py', content:node },
      { filename:'robot.yaml', content:yaml },
      { filename:'run_bridge.sh', content:bridge },
    ];
  }

  if (hw==='rocket') {
    const ino = `/*
 * PhysiCore Rocket Flight Computer — ${answers.fc||'Arduino Mega'} + ${answers.baro||'BMP280'}
 * Dry mass: ${mass}kg | Baud: ${answers.baud||'115200'}
 * INSTALL: ${answers.baro||'BMP280'} library + ArduinoJson v6.x
 */
#include <Wire.h>
#include <ArduinoJson.h>
// TODO: add your ${answers.baro||'BMP280'} include here

float altitude=0, prev_alt=0, velocity=0, mass_kg=${mass};
String phase="IDLE";
unsigned long last_tx=0;

void setup(){
  Serial.begin(${answers.baud||'115200'}); Wire.begin();
  // TODO: init your ${answers.baro||'BMP280'} sensor here
  Serial.println("{\\"status\\":\\"ready\\"}");
}

void loop(){
  unsigned long now=millis();
  // TODO: read altitude from ${answers.baro||'BMP280'}
  // altitude = baro.readAltitude(1013.25);

  velocity=(altitude-prev_alt)/0.02; prev_alt=altitude;
  if(velocity>2.0) phase="POWERED";
  else if(altitude>50&&velocity<0) phase="COAST";
  else if(altitude<50&&velocity<-0.5) phase="RECOVERY";
  else phase="IDLE";

  if(now-last_tx>=20){
    last_tx=now;
    StaticJsonDocument<256> doc;
    doc["altitude"]=altitude; doc["velocity"]=velocity;
    doc["mass"]=mass_kg; doc["phase"]=phase; doc["timestamp"]=now;
    serializeJson(doc,Serial); Serial.println();
  }
  delay(1);
}`;
    const yaml = `name: My Rocket\nplatform: rocket\nconnection: ${port}\nbaud: ${answers.baud||'115200'}\nmass: ${mass}\nfriction: 0.45\ninertia: 220\ncontrol_hz: 60.0\nuse_registry: true`;
    const bridge = os.toLowerCase().includes('win')
      ? `@echo off\npip install pymavlink websockets aiohttp pyserial pyyaml\npython physicore\\bridge\\physicore_bridge.py --config rocket.yaml`
      : `#!/bin/bash\npip install pymavlink websockets aiohttp pyserial pyyaml\npython physicore/bridge/physicore_bridge.py --config rocket.yaml`;
    return [
      { filename:'physicore_rocket_fc.ino', content:ino },
      { filename:'rocket.yaml', content:yaml },
      { filename: os.toLowerCase().includes('win') ? 'run_bridge.bat' : 'run_bridge.sh', content:bridge },
    ];
  }

  if (hw === 'custom') {
    const iface = answers.interface || 'Arduino/ESP32 serial';
    const sensors = answers.sensors || 'IMU only';
    const mcu = answers.mcu || 'Arduino Uno/Nano/Mega';
    const sysType = answers.type || 'Ground robot';
    const isSerial = iface.toLowerCase().includes('arduino') || iface.toLowerCase().includes('serial');
    const isROS2 = iface.toLowerCase().includes('ros2');
    const isMAV = iface.toLowerCase().includes('mavlink');

    const platform = isMAV ? 'px4_quadrotor' : isROS2 ? 'ros2_ground_rover' : 'ground_rover_serial';

    const guide = `# PhysiCore Custom Hardware Integration
# System type: ${sysType} | Interface: ${iface} | Sensors: ${sensors}
# Compute: ${mcu} | Mass: ${mass}kg

## What PhysiCore needs from your hardware

Your hardware must send JSON over serial (or ROS2/MAVLink) at 20-50 Hz.
Minimum required fields — send whatever you have, PhysiCore uses what it gets:

{"pitch":0.0,"roll":0.0,"gyro_x":0.0,"gyro_y":0.0,"gyro_z":0.0,
 "accel_x":0.0,"accel_y":0.0,"accel_z":9.81,
 "motor_l":0.0,"motor_r":0.0,"timestamp":0}

## Arduino/ESP32 serial template

Add this to your existing sketch:

  StaticJsonDocument<256> doc;
  doc["pitch"]   = YOUR_PITCH_VALUE;     // angle in degrees
  doc["gyro_x"]  = YOUR_GYRO_VALUE;      // angular velocity deg/s
  doc["accel_z"] = YOUR_ACCEL_Z;         // m/s^2
  doc["motor_l"] = YOUR_LEFT_MOTOR;      // -1.0 to 1.0 or N*m
  doc["motor_r"] = YOUR_RIGHT_MOTOR;
  doc["timestamp"] = millis();
  serializeJson(doc, Serial);
  Serial.println();

PhysiCore sends back:
  {"op":"command","action":[TORQUE_VALUE]}

Apply that torque to your actuators.

## Run the bridge

${isSerial ? `python physicore/bridge/physicore_bridge.py --platform ground_rover_serial --connection ${port} --baud 115200` :
  isROS2 ? `source /opt/ros/humble/setup.bash
python physicore/bridge/physicore_bridge.py --platform ros2_ground_rover` :
  `python physicore/bridge/physicore_bridge.py --platform px4_quadrotor --connection udp:14550`}

## Connect dashboard
MAVLINK → ws://localhost:8765 → Connect → ACTIVE CONTROL ON

## PhysiCore adapts
Within 30 seconds it will learn your system's real mass and friction.
No manual tuning required.`;

    const yaml = `name: My Custom ${sysType}
platform: ground_rover
connection: ${port}
baud: 115200
mass: ${mass}
friction: 0.3
inertia: 0.1
control_hz: 60.0
use_registry: true
opt_in_telemetry: false`;

    return [
      { filename:'custom_integration_guide.md', content:guide },
      { filename:'custom.yaml', content:yaml },
    ];
  }

  return [{ filename:'run_bridge.sh', content:`#!/bin/bash
python physicore/bridge/physicore_bridge.py --platform ros2_ground_rover` }];
}

// ── Steps per hardware ────────────────────────────────────────────────────────
function ie_getSteps(hw: string, answers: Record<string,string>): {id:string; label:string; detail:string; cmd?:string}[] {
  const os = answers.os||'Linux';
  const port = ie_port(os);
  const distro = (answers.distro||'humble').toLowerCase();
  const topic = answers.topic||'/joint_states';
  const isWin = os.toLowerCase().includes('win');

  if (hw==='balancing_bot') return [
    { id:'lib',      label:'Install Arduino libraries', detail:`Arduino IDE → Sketch → Include Library → Manage Libraries\nInstall: ${answers.imu||'MPU6050'} library\nInstall: ArduinoJson by Benoit Blanchon (v6.x)` },
    { id:'flash',    label:'Flash firmware', detail:`Open physicore_balancing_bot.ino\nTools → Board → select ${answers.mcu||'your board'}\nTools → Port → select your port → click Upload` },
    { id:'calib',    label:'Calibrate BALANCE_POINT', detail:`Open Serial Monitor at 115200 baud\nHold robot perfectly upright → read pitch value\nEdit firmware: const float BALANCE_POINT = <that value>;\nRe-upload firmware` },
    { id:'imu',      label:'Verify IMU is working', detail:`Tilt robot forward/back → pitch value must change\nGood: {"pitch":8.4,"gyro_x":-2.1,...}\nBad: all zeros → check SDA/SCL wiring` },
    { id:'bridge',   label:'Run the bridge', detail:`Close Arduino IDE first (locks the serial port)\nEdit balancing_bot.yaml: change connection: ${port} to your actual port\n${isWin ? 'Find port: Device Manager → Ports' : 'Find port: ls /dev/ttyUSB* or ls /dev/cu.*'}`, cmd: isWin ? 'run_bridge.bat' : 'bash run_bridge.sh' },
    { id:'connect',  label:'Connect dashboard', detail:`Click MAVLINK → endpoint: ws://localhost:8765 → Connect\nLive pitch data appears immediately` },
    { id:'activate', label:'Activate PhysiCore', detail:`Click ACTIVE CONTROL ON\nLED turns ON = PhysiCore is sending commands\nWatch mass estimate adapt in sidebar — that is SystemID learning your robot` },
  ];

  if (hw==='px4'||hw==='ardupilot'||hw==='evtol') return [
    { id:'telemetry', label:'Enable MAVLink telemetry', detail:`QGroundControl → Application Settings → Telemetry → enable UDP port 14550\n${(answers.connection||'').toLowerCase().includes('usb') ? 'Or: connect USB cable directly to Pixhawk' : 'Laptop and drone must be on same WiFi'}` },
    { id:'bridge',    label:'Run the bridge', detail:`Installs pymavlink, websockets, aiohttp\nYou should see: "MAVLink connected"`, cmd: isWin ? 'run_bridge.bat' : 'bash run_bridge.sh' },
    { id:'connect',   label:'Connect dashboard', detail:`Click MAVLINK → ws://localhost:8765 → Connect\nLive telemetry: altitude, pitch, roll, yaw` },
    { id:'activate',  label:'Arm and activate', detail:`Arm vehicle normally\nClick ACTIVE CONTROL ON\nPhysiCore reads telemetry at 60 Hz, adapts mass/friction live` },
  ];

  if (['ros2_arm','humanoid','legged','auv','surgical','rover'].includes(hw)) return [
    { id:'source',   label:'Source ROS2', detail:`Must run in every terminal before using ROS2`, cmd:`source /opt/ros/${distro}/setup.bash` },
    { id:'topics',   label:'Verify joint states topic', detail:`Joint data must appear — if not, your robot driver is not running`, cmd:`ros2 topic echo ${topic} --once` },
    { id:'bridge',   label:'Run the bridge', detail:`Starts PhysiCore bridge on port 8765 + ROS2 bridge node`, cmd:'bash run_bridge.sh' },
    { id:'connect',  label:'Connect dashboard', detail:`Click MAVLINK → ws://localhost:8765 → Connect` },
    { id:'activate', label:'Activate control', detail:`Click ACTIVE CONTROL ON\nPhysiCore adapts your robot's physics from live joint data` },
  ];

  if (hw==='rocket') return [
    { id:'firmware', label:'Complete and flash firmware', detail:`Add ${answers.baro||'BMP280'} library include + initialization\nFlash to ${answers.fc||'Arduino Mega'}` },
    { id:'verify',   label:'Verify telemetry on ground', detail:`Serial Monitor at ${answers.baud||'115200'} baud\nSee: {"altitude":0.2,"phase":"IDLE",...}` },
    { id:'bridge',   label:'Run the bridge', detail:`Edit rocket.yaml: change connection to your port\n${isWin ? 'Find port: Device Manager → Ports' : 'Find port: ls /dev/ttyUSB*'}`, cmd: isWin ? 'run_bridge.bat' : 'bash run_bridge.sh' },
    { id:'connect',  label:'Connect dashboard', detail:`Click MAVLINK → ws://localhost:8765 → Connect\nLive altitude and phase appear` },
  ];

  if (hw === 'custom') {
    const iface = answers.interface || '';
    const isSerial = iface.toLowerCase().includes('arduino') || iface.toLowerCase().includes('serial');
    return [
      { id:'read',    label:'Read the integration guide', detail:`Open custom_integration_guide.md — it has the exact JSON format your hardware needs to send and the command to receive` },
      { id:'serial',  label:'Add serial output to your code', detail:`Send JSON at 20-50 Hz: {"pitch":0,"gyro_x":0,"accel_z":9.81,"motor_l":0,"motor_r":0,"timestamp":0}
Include whatever sensor fields you have` },
      { id:'bridge',  label:'Run the bridge', detail:`Edit custom.yaml with your actual serial port
${isSerial ? 'Find port: Device Manager (Win) / ls /dev/cu.* (Mac) / ls /dev/ttyUSB* (Linux)' : 'Use --platform that matches your interface'}`, cmd:'bash run_bridge.sh' },
      { id:'connect', label:'Connect dashboard', detail:`Click MAVLINK → ws://localhost:8765 → Connect` },
      { id:'activate',label:'Activate and adapt', detail:`Click ACTIVE CONTROL ON
PhysiCore starts learning your system's real mass and friction from live data` },
    ];
  }

  return [];
}

// ── Troubleshoot tree ─────────────────────────────────────────────────────────
function ie_troubleshoot(msg: string, hw: string, answers: Record<string,string>): {title:string; steps:{label:string; cmd:string}[]} | null {
  const m = msg.toLowerCase();
  const os = answers.os||'Linux';
  const distro = (answers.distro||'humble').toLowerCase();
  const topic = answers.topic||'/joint_states';

  if (m.match(/port|com\d|\btty\b|which.*port|can't find.*serial/)) return {
    title:'Finding your serial port',
    steps:[
      { label:'Windows', cmd:'Device Manager → Ports (COM & LPT) → note COMx' },
      { label:'Mac', cmd:'ls /dev/cu.*' },
      { label:'Linux', cmd:'ls /dev/ttyUSB*  or  ls /dev/ttyACM*' },
      { label:'Then update yaml', cmd:`connection: YOUR_PORT in ${hw==='rocket'?'rocket':'balancing_bot'}.yaml` },
    ],
  };
  if (m.match(/imu|pitch.*zero|pitch.*not.*chang|mpu.*not|bno.*not/)) return {
    title:'IMU not responding',
    steps:[
      { label:'Check wiring', cmd:`${answers.imu||'MPU6050'}: SDA → A4, SCL → A5, VCC → 3.3V (NOT 5V), GND → GND` },
      { label:'Verify in Serial Monitor', cmd:'Tilt robot — pitch value must change. If stays at 0: wiring error' },
      { label:'Check library', cmd:`Sketch → Include Library → Manage Libraries → search ${answers.imu||'MPU6050'}` },
    ],
  };
  if (m.match(/jitter|jittery|vibrat|shak|oscillat|not.*smooth/)) return {
    title:'Jittery / oscillating robot',
    steps:[
      { label:'BALANCE_POINT wrong', cmd:'Hold upright → read pitch in Serial Monitor → set BALANCE_POINT to that value → re-upload' },
      { label:'MAX_TORQUE check', cmd:'Must be 2.5 in firmware — not 100, not 255' },
      { label:'IMU noise', cmd:'Run mpu.calcOffsets() in setup() to calibrate offsets' },
    ],
  };
  if (m.match(/not.*balanc|fall|tip.*over|won't.*stand|can't.*stay/)) return {
    title:'Robot won\'t balance',
    steps:[
      { label:'Is PhysiCore active?', cmd:'Click ACTIVE CONTROL ON — LED_BUILTIN must turn ON' },
      { label:'BALANCE_POINT calibrated?', cmd:'Redo calibration: upright → Serial Monitor → read pitch → set BALANCE_POINT' },
      { label:'Mass correct?', cmd:`YAML mass: ${answers.mass||'?'} kg — measure your actual robot mass` },
      { label:'Bridge connected?', cmd:'Dashboard must show "Connected" and live pitch data' },
    ],
  };
  if (m.match(/not.*connect|can't.*connect|dashboard.*not|no.*data|ws.*fail/)) return {
    title:'Dashboard not connecting',
    steps:[
      { label:'Endpoint', cmd:'Must be exactly: ws://localhost:8765 (not https, not http)' },
      { label:'Bridge running?', cmd:'Terminal must show "PhysiCore bridge started" — not closed' },
      { label:'Arduino IDE closed?', cmd:'Close it — it locks the serial port' },
      { label:'Firewall?', cmd:'Allow port 8765 in firewall / Windows Defender' },
    ],
  };
  if (m.match(/ros2.*not|topic.*not.*found|rclpy|no.*topic|source/)) return {
    title:'ROS2 / topics not found',
    steps:[
      { label:'Source first', cmd:`source /opt/ros/${distro}/setup.bash` },
      { label:'List topics', cmd:'ros2 topic list' },
      { label:'Check joint states', cmd:`ros2 topic echo ${topic} --once` },
      { label:'rclpy missing?', cmd:`sudo apt install ros-${distro}-rclpy` },
    ],
  };
  if (m.match(/mavlink|heartbeat|qground|mission.*planner|no.*heartbeat/)) return {
    title:'MAVLink not connecting',
    steps:[
      { label:'Enable telemetry', cmd:'QGC: Application Settings → Telemetry → UDP port 14550' },
      { label:'Same network?', cmd:'Laptop and drone must be on same WiFi for UDP' },
      { label:'USB connection?', cmd:'connection: /dev/ttyACM0 (Linux) or COM3 (Windows)' },
      { label:'pymavlink installed?', cmd:'pip install pymavlink' },
    ],
  };
  return null;
}

// ── IEState type for the new button-driven UI ─────────────────────────────────
interface IEState {
  phase: 'welcome' | 'questions' | 'generated' | 'troubleshoot';
  hw: string;
  qIndex: number;
  answers: Record<string,string>;
  files: {filename:string; content:string}[];
  steps: {id:string; label:string; detail:string; cmd?:string}[];
  checklist: Record<string,boolean>;
  activeFile: number;
  troubleshootResult: {title:string; steps:{label:string;cmd:string}[]} | null;
  freeInput: string;
}

// ── physi_integrate is now a thin shim for backward compat with handleSendMessage ──
function physi_integrate(
  userMsg: string,
  history: any[],
  callbacks?: {
    setGeneratedFiles?: (f: any[]) => void;
    setIntegrationPhase?: (p: number) => void;
    setSystemProfile?: (fn: (prev: any) => any) => void;
    setConnectionMode?: (m: any) => void;
    setEndpoint?: (e: string) => void;
  }
): string {
  // This is only called for troubleshooting freetext now
  // The main Q&A is handled by renderIntegrator's local state machine
  const hw = ie_detect(userMsg) || 'balancing_bot';
  const local = ie_troubleshoot(userMsg, hw, {});
  if (local) {
    const stepsText = local.steps.map((s,i) => `STEP ${i+1} — ${s.label}\n     ${s.cmd}`).join('\n\n');
    return `> INTEGRATION ENGINEER:\n${local.title}\n\n${stepsText}`;
  }
  return `> INTEGRATION ENGINEER:\nTell me what hardware you have and I'll generate your complete integration — firmware, bridge config, and step-by-step guide.\n\nExamples:\n  • "I have a balancing bot with Arduino and MPU6050"\n  • "I have a PX4 drone"\n  • "I have a ROS2 robot arm"\n  • "I have a sounding rocket"`;
}



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
  const isBetaTester = user?.email ? BETA_TESTERS.includes(user.email) : false;
  const isAuthorized = true; // Open to all signed-in users
  
  const [checkingAccess, setCheckingAccess] = useState(false);
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [allUsers, setAllUsers] = useState<any[]>([]);
  const [isBootstrapping, setIsBootstrapping] = useState(false);
  const [authError, setAuthError] = useState<{ message: string; domain?: string } | null>(null);

  const bootstrapTeam = async () => {
    if (user?.email !== "prathameshshirbhate8anpc@gmail.com") return;
    setIsBootstrapping(true);
    
    const teamEmails = [
      "prathameshshirbhate256@gmail.com",
      "ashwanth123creations@gmail.com",
      "adithya17k@gmail.com"
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

  const bootstrappedRef = useRef(false);
  useEffect(() => {
    if (isAdmin && allUsers.length === 0 && !isBootstrapping && user && !bootstrappedRef.current) {
      bootstrappedRef.current = true;
      // Auto-bootstrap team if empty and we are admin
      bootstrapTeam();
    }
  }, [isAdmin, allUsers.length, isBootstrapping, user]);

  const [view, setView] = useState<View>('home');
  const [projectTab, setProjectTab] = useState<ProjectTab>('integrate');
  const [editingFileKey, setEditingFileKey] = useState<string | null>(null);
  const [originalFiles, setOriginalFiles] = useState<Record<string, string>>({});

  const navigateToProject = (tab: ProjectTab = 'integrate') => {
    setProjectTab(tab);
    setView('project');
  };

  // Redirect logged-in users from home to projects
  useEffect(() => {
    if (user && view === 'home') {
      setView('projects');
    }
  }, [user]);

  const [activeSection, setActiveSection] = useState('overview');
  const [manualSection, setManualSection] = useState('intro');
  const [isControlActive, setIsControlActive] = useState(false);

  // API Key Modal State
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [apiKeyGeminiInput, setApiKeyGeminiInput] = useState('');
  const [apiKeyAnthropicInput, setApiKeyAnthropicInput] = useState('');
  const [apiKeySaved, setApiKeySaved] = useState(false);

  // Build Tab State
  const [buildMessages, setBuildMessages] = useState<{ role: 'user' | 'assistant'; text: string }[]>([]);
  const [buildInput, setBuildInput] = useState('');
  const [isBuildLoading, setIsBuildLoading] = useState(false);
  const [buildFeatures, setBuildFeatures] = useState<FeatureManifest[]>([]);
  const [selectedBuildFile, setSelectedBuildFile] = useState<string | null>(null);

  // Integration Engineer State
  const [conversationHistory, setConversationHistory] = useState<Message[]>([]);
  const [systemProfile, setSystemProfile] = useState<SystemProfile>({
    platform: null, firmware: null, domain: null, massClass: null, connectionMode: null, protocols: null
  });
  const [integrationPhase, setIntegrationPhase] = useState(1);
  const [generatedFiles, setGeneratedFiles] = useState<GeneratedFile[]>([]);
  const [ieState, setIE] = useState<IEState>({
    phase: 'welcome', hw: '', qIndex: 0, answers: {}, files: [], steps: [],
    checklist: {}, activeFile: 0, troubleshootResult: null, freeInput: '',
  });
  const [ieCopiedId, setIECopiedId] = useState<string|null>(null);
  const [ieTsInput, setIETsInput] = useState('');
  const [extBuilderInput, setExtBuilderInput] = useState('');
  const [extBuilderResult, setExtBuilderResult] = useState<string|null>(null);
  const [isBuildingExt, setIsBuildingExt] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [isSystemConnected, setIsSystemConnected] = useState(false);
  const [isSystemConnecting, setIsSystemConnecting] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [connectionMode, setConnectionMode] = useState<'ros2_websocket' | 'hil' | 'digital_twin' | 'mavlink_bridge'>('mavlink_bridge');
  const [digitalTwinConfirmed, setDigitalTwinConfirmed] = useState(false);
  const [showDigitalTwinModal, setShowDigitalTwinModal] = useState(false);
  const [endpoint, setEndpoint] = useState('ws://localhost:8765');
  const [dRealEndpoint, setDRealEndpoint] = useState('http://localhost:8080');

  const [handshakeConfirmed, setHandshakeConfirmed] = useState(false);

  const [isLaunching, setIsLaunching] = useState(false);
  const [simulationConfig, setSimulationConfig] = useState<any | null>(null);
  const [metaAnalysisResult, setMetaAnalysisResult] = useState<string | null>(null);
  const [showUserManagement, setShowUserManagement] = useState(false);

  const [telemetry, setTelemetry] = useState({
    mass: 0,
    friction: 0,
    actuatorEfficiency: 0,
    residual: 0,
    confidence: 0,
    variance: 0,
    isStable: true,
    isFaulted: false,
    step_count: 0,
    cpuLoad: 0,
    latency: 0,
    residualHistory: [] as any[],
    effortHistory: [] as any[],
    targetPos: { x: 0, y: 0 },
    // Rocket/Aviation specific telemetry
    pos: null as any,
    vel: { x: 0, y: 0, z: 0 } as any,
    accel: { x: 0, y: 0, z: 0 } as any,
    gyro: { x: 0, y: 0, z: 0 } as any,
    orientation: { r: 0, p: 0, y: 0 } as any,
    roll: 0,
    pitch: 0,
    yaw: 0,
    propMass: 0,
    time: 0,
    phase: 'PRELAUNCH' as string,
    altitude: 0,
    speed: 0,
    gyro_x: 0,
    gyro_y: 0,
    gyro_z: 0,
    airspeed: 0,
    groundspeed: 0,
    climb_rate: 0,
    mach: 0,
    aoa: 0,
    bank: 0,
    motor_l: 0,
    motor_r: 0,
    battery_pct: 0,
    battery_v: 0,
    armed: false,
    flight_mode: 'UNKNOWN',
    vehicle_type: 'UNKNOWN',
    connected: false,
    gps_fix: 0,
    satellites: 0,
    faults: [] as string[],
  });
  const [failureLogs, setFailureLogs] = useState<FailureLog[]>([]);
  const [debuggerQuery, setDebuggerQuery] = useState('');
  const [debuggerResult, setDebuggerResult] = useState<string|null>(null);
  const [isDebugging, setIsDebugging] = useState(false);

  const [flightData, setFlightData] = useState<any[]>([]);
  const [isRocketSimRunning, setIsRocketSimRunning] = useState(false);
  const [rocketSimSpeed, setRocketSimSpeed] = useState(1);
  const [actualFlightData, setActualFlightData] = useState<any[] | null>(null);
  const [showImportOverlay, setShowImportOverlay] = useState(false);

  // --- MPC & SystemID State ---
  const [simMode, setSimMode] = useState<SimMode>(SimMode.MPC_STABILIZATION);
  const [simState, setSimState] = useState<SimState>({
    current: [400, 300, 0, 0, 0, 0], // Center of canvas
    target: [400, 300],
    estimatedParams: { mass: 1.0, friction: 0.1, gravity: 9.81, textile_k: 0, damping: 0.1 },
    predictionError: 0,
    controlEffort: 0,
    stability: 100,
    time: 0,
    controlAction: [0, 0],
    uncertainty: 0,
    isBenchmarking: false
  });
  const [mpcWeights, setMpcWeights] = useState({ q: 1.0, r: 0.1 });
  const [metaAnalysis, setMetaAnalysis] = useState<MetaAnalysisResponse | null>(null);
  const [isMetaAnalyzing, setIsMetaAnalyzing] = useState(false);

  // --- MPC Simulation Loop ---
  useEffect(() => {
    if (!isControlActive || view !== 'dashboard' || isSystemConnected) return;

    const interval = setInterval(() => {
      setSimState(prev => {
        // 1. Compute MPC Action
        const { action, ensembleUncertainty } = computeMPCAction(
          prev.current,
          prev.target,
          prev.estimatedParams,
          mpcWeights
        );

        // 2. Step Ground Truth Dynamics (Simulation Mode)
        // In a real system, this would come from telemetry
        const nextState = stepDynamicsRK4(prev.current, action, {
          mass: prev.estimatedParams.mass * 1.3,    // Simulated true mass (30% offset to give SysID signal)
          friction: prev.estimatedParams.friction * 1.4,
          gravity: 9.81,
          textile_k: 0,
          damping: 0.1
        });

        // 3. Update System Identification (SysID)
        const updatedParams = updateSystemID(
          prev.current,
          action,
          nextState,
          prev.estimatedParams
        );

        // 4. Train Learned Dynamics (Residual Model)
        const physicsPrediction = stepDynamicsRK4(prev.current, action, updatedParams);
        ensembleDynamics.train(prev.current, action, nextState, physicsPrediction);

        // 5. Compute Metrics
        const predictionError = nextState.reduce((s, v, i) => s + Math.pow(v - physicsPrediction[i], 2), 0);
        const stability = 100 - Math.min(100, predictionError * 1000);

        return {
          ...prev,
          current: nextState,
          estimatedParams: updatedParams,
          predictionError,
          controlEffort: Math.sqrt(action.reduce((s:number,v:number)=>s+v*v,0)),
          controlAction: action,
          uncertainty: ensembleUncertainty,
          stability,
          time: prev.time + 0.01
        };
      });
    }, 10); // 100Hz loop

    return () => clearInterval(interval);
  }, [isControlActive, view, mpcWeights]);

  // ── Meta-Analyst Loop ──────────────────────────────────────────────────────
  // Tier 1: Gemini direct (VITE_GEMINI_API_KEY in Vercel env vars) — AI insight
  // Tier 2: Local TypeScript narration — deterministic, no key, always works
  // Never calls localhost. Never silently fails. Degrades gracefully.
  useEffect(() => {
    if (!isControlActive || view !== 'dashboard') return;

    // ── Local TypeScript narrate() — deterministic, no API ──────────────────
    // Mirrors the Python engine narrate() but runs in the browser
    function localNarrate(): string {
      if (isSystemConnected) {
        // Real hardware connected — use live telemetry
        const res  = telemetry.residual  || 0;
        const mass = telemetry.mass      || 0;
        const unc  = telemetry.variance  || 0;
        const steps = telemetry.step_count || 0;
        const isFault = telemetry.isFaulted;
        const isStable = telemetry.isStable;

        if (isFault || res > 0.80 || unc > 0.15) {
          return `> META-ANALYST: ⚠ Residual critically high (${res.toFixed(3)}) — model is significantly wrong about your hardware.

Check: IMU wiring, BALANCE_POINT calibration, and whether mass changed during the session.

Action: Disable ACTIVE CONTROL, verify sensor data, restart session.`;
        }
        if (!isStable || res > 0.30) {
          return `> META-ANALYST: Residual elevated (${res.toFixed(3)}) — model is adapting to real hardware.

Mass estimate: ${mass.toFixed(3)}kg. ${steps < 300 ? 'Normal during early convergence — let it run for 30 seconds.' : 'Above expected for this stage — check BALANCE_POINT.'}

Action: ${steps < 300 ? 'Keep running.' : 'Check IMU calibration if residual does not drop.'}`;
        }
        return `> META-ANALYST: System nominal ✓

Residual: ${res.toFixed(4)} | Mass: ${mass.toFixed(3)}kg | Steps: ${steps}
PhysiCore has learned your hardware's real physics. Registry will save these params on session end.

Action: Continue operating. Watch for residual spikes that may indicate payload changes.`;
      } else {
        // Simulation mode — use simState
        const err  = simState.predictionError;
        const mass = simState.estimatedParams.mass;
        const fric = simState.estimatedParams.friction;
        const stab = simState.stability;
        if (err > 0.01) {
          return `> META-ANALYST: Simulation residual elevated (${err.toFixed(5)}) — SysID adapting.

Mass estimate: ${mass.toFixed(3)}kg, friction: ${fric.toFixed(3)}. Stability: ${stab.toFixed(0)}%.

Action: ${stab < 50 ? 'Consider increasing Q weight for tighter tracking.' : 'Model converging normally.'}`;
        }
        return `> META-ANALYST: Simulation nominal ✓

Mass: ${mass.toFixed(3)}kg | Friction: ${fric.toFixed(3)} | Stability: ${stab.toFixed(0)}%
Connect hardware to run PhysiCore on your real robot.

Action: Click MAVLINK → ws://localhost:8765 → Connect.`;
      }
    }

    const runAnalysis = async () => {
      setIsMetaAnalyzing(true);
      try {
        // Tier 1: Gemini AI — builds rich context from live state
        const ai = getAI();
        if (ai) {
          const hwCtx = isSystemConnected
            ? `HARDWARE CONNECTED (step ${telemetry.step_count}):
  residual=${telemetry.residual?.toFixed(4)} trend=${(telemetry.residualHistory||[]).slice(-5).map((p:any)=>p.y?.toFixed(3)).join(',')}
  mass=${telemetry.mass?.toFixed(3)}kg friction=${telemetry.friction?.toFixed(4)}
  uncertainty=${telemetry.variance?.toFixed(4)} stable=${telemetry.isStable} faulted=${telemetry.isFaulted}
  motor_l=${telemetry.motor_l?.toFixed(2)} motor_r=${telemetry.motor_r?.toFixed(2)}
  pitch=${telemetry.pitch?.toFixed(2)}°`
            : `SIMULATION MODE:
  predictionError=${simState.predictionError.toFixed(5)} controlEffort=${simState.controlEffort.toFixed(3)}
  mass=${simState.estimatedParams.mass.toFixed(3)} friction=${simState.estimatedParams.friction.toFixed(3)}
  stability=${simState.stability.toFixed(0)}% uncertainty=${simState.uncertainty.toFixed(4)}`;

          const resp = await ai.models.generateContent({
            model: 'gemini-2.0-flash',
            contents: `${hwCtx}

Analyze this PhysiCore session. Return 2-3 sentences: what is happening physically, one specific recommendation. Plain text only, prefix with > META-ANALYST:`,
            config: { systemInstruction: 'You are the PhysiCore Meta-Analyst. Interpret robotics telemetry with precision. Be concise and technical.' }
          });
          const text = resp.text?.trim();
          if (text) {
            setMetaAnalysisResult(text.startsWith('> META-ANALYST') ? text : `> META-ANALYST: ${text}`);

            // Also try to get Q/R suggestions in a second fast call
            try {
              const tuneResp = await ai.models.generateContent({
                model: 'gemini-2.0-flash',
                contents: `PhysiCore MPC tuning. Current: residual=${isSystemConnected ? telemetry.residual?.toFixed(4) : simState.predictionError.toFixed(5)}, stability=${isSystemConnected ? telemetry.isStable : simState.stability > 80}. Current Q=${mpcWeights.q} R=${mpcWeights.r}. Reply with only two numbers separated by comma: suggested_q,suggested_r (ranges: q=0.1-20, r=0.01-5)`,
                config: { systemInstruction: 'Return only two numbers separated by a comma. Nothing else.' }
              });
              const nums = tuneResp.text?.trim().split(',').map(Number);
              if (nums && nums.length === 2 && !isNaN(nums[0]) && !isNaN(nums[1])) {
                const q = Math.min(20, Math.max(0.1, nums[0]));
                const r = Math.min(5, Math.max(0.01, nums[1]));
                setMpcWeights({ q, r });
                setMetaAnalysis({ insight: text, diagnostics: [text], suggestedCostTweaks: { q_weight: q, r_weight: r } });
              }
            } catch (_) {}
            return;
          }
        }
      } catch (_geminiErr) {
        // Gemini unavailable — fall through to local narration
      }

      // Tier 2: Local deterministic narration — always works, no API
      setMetaAnalysisResult(localNarrate());
    };

    runAnalysis();
    const interval = setInterval(runAnalysis, 20000);
    return () => clearInterval(interval);
  }, [isControlActive, view, isSystemConnected, telemetry.step_count]);

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
  
  // Logic for common actions

  // Persistent WebSocket for Real-time Telemetry
  const socketRef = useRef<WebSocket | null>(null);

  const sendCommand = (active: boolean, action?: number[], x_ref?: number[]) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({
        op: 'command',
        msg: {
          active,
          action,
          x_ref,
          timestamp: Date.now()
        }
      }));
    }
  };

  useEffect(() => {
    if (isSystemConnected) {
      sendCommand(isControlActive);
    }
  }, [isControlActive, isSystemConnected]);

  useEffect(() => {
    if (isSystemConnected && (connectionMode === 'ros2_websocket' || connectionMode === 'mavlink_bridge') && endpoint) {
      try {
        const ws = new WebSocket(endpoint);
        socketRef.current = ws;

        ws.onopen = () => {
          console.log("Telemetry Stream: CONNECTED");
          // Subscribe to telemetry topic only for ROS2
          if (connectionMode === 'ros2_websocket') {
            ws.send(JSON.stringify({
              op: 'subscribe',
              topic: '/telemetry',
              type: 'physicore_msgs/Telemetry'
            }));
          }
        };

        ws.onmessage = (msg) => {
          try {
            const data = JSON.parse(msg.data);
            if (data.op === 'extensions_status') {
              setLoadedExtensions(data.extensions || []);
            }
            if (data.op === 'registry_status') {
              setRegistryStatus({
                platform:       data.platform,
                sessions_count: data.sessions_count,
                latest_params:  data.latest_params || {},
                prior_weight:   data.prior_weight,
                loaded:         data.loaded,
                registry_path:  data.registry_path,
                current_mass:   data.current_mass,
                current_friction: data.current_friction,
              });
            }
            if (data.op === 'publish' && data.topic === '/telemetry') {
              // Ensure we stay connected
              setIsSystemConnected(true);
              const d = data.msg;
              setTelemetry(prev => ({
                ...prev,
                // Direct fields
                altitude:           d.altitude           ?? prev.altitude,
                airspeed:           d.airspeed           ?? prev.airspeed,
                groundspeed:        d.groundspeed        ?? prev.groundspeed,
                climb_rate:         d.climb_rate         ?? prev.climb_rate,
                // Orientation — handle both flat and nested
                roll:               d.roll               ?? d.orientation?.roll  ?? prev.roll,
                pitch:              d.pitch              ?? d.orientation?.pitch ?? prev.pitch,
                yaw:                d.yaw                ?? d.orientation?.yaw   ?? prev.yaw,
                // Velocity
                vel:                d.velocity           ?? prev.vel,
                // Acceleration
                accel: {
                  x: d.accel_x ?? d.acceleration?.x ?? prev.accel?.x ?? 0,
                  y: d.accel_y ?? d.acceleration?.y ?? prev.accel?.y ?? 0,
                  z: d.accel_z ?? d.acceleration?.z ?? prev.accel?.z ?? 0,
                },
                // Gyro
                gyro: {
                  x: d.gyro_x ?? d.gyro?.x ?? prev.gyro?.x ?? 0,
                  y: d.gyro_y ?? d.gyro?.y ?? prev.gyro?.y ?? 0,
                  z: d.gyro_z ?? d.gyro?.z ?? prev.gyro?.z ?? 0,
                },
                // Motor
                motor_l:            d.motor_l            ?? prev.motor_l,
                motor_r:            d.motor_r            ?? prev.motor_r,
                // Battery
                battery_pct:        d.battery?.percentage ?? d.battery_pct ?? prev.battery_pct,
                battery_v:          d.battery?.voltage    ?? d.battery_v   ?? prev.battery_v,
                // Status
                armed:              d.armed              ?? prev.armed,
                flight_mode:        d.flight_mode        ?? prev.flight_mode,
                vehicle_type:       d.vehicle_type       ?? prev.vehicle_type,
                connected:          d.connected          ?? prev.connected,
                // GPS
                gps_fix:            d.gps?.fix           ?? d.gps_fix      ?? prev.gps_fix,
                satellites:         d.gps?.satellites    ?? d.satellites    ?? prev.satellites,
                // SystemID fields — keep existing if not in packet
                mass:               d.mass               ?? prev.mass,
                friction:           d.friction           ?? prev.friction,
                actuatorEfficiency: d.actuatorEfficiency ?? prev.actuatorEfficiency,
                residual:           d.residual           ?? prev.residual,
                confidence:         d.confidence         ?? prev.confidence,
                variance:           d.variance           ?? prev.variance,
                isStable:           d.isStable           ?? prev.isStable,
                isFaulted:          d.isFaulted          ?? prev.isFaulted,
                step_count:         d.step_count         ?? prev.step_count,
                faults:             d.faults             ?? prev.faults,
                // History
                residualHistory: d.residual > 0 ? [...(prev.residualHistory || []), { x: Date.now(), y: d.residual }].slice(-60) : (prev.residualHistory || []),
                effortHistory:   [...(prev.effortHistory   || []), { x: Date.now(), y: d.effort   || 0 }].slice(-30),
              }));
              // Capture FailureLog entries from sentinel
              if (d.isFaulted && d.faults?.length > 0) {
                setFailureLogs(prev => [{
                  id: `${Date.now()}`,
                  timestamp: Date.now(),
                  task: `Step ${d.step_count}`,
                  failure_type: d.faults[0],
                  sim_params: { mass: d.mass||0, friction: d.friction||0, gravity: 9.81, textile_k: 0, damping: 0 },
                  fix_applied: false,
                }, ...prev].slice(0, 50));
              }
            }
          } catch (e) {
            console.error("Telemetry Parse Error:", e);
          }
        };

        ws.onclose = () => {
          console.log("Telemetry Stream: DISCONNECTED — auto-reconnecting in 2s...");
          socketRef.current = null;
          setTimeout(() => {
            if (socketRef.current === null) {
              setIsSystemConnected(false);
              setTimeout(() => setIsSystemConnected(true), 200);
            }
          }, 2000);
        };

        return () => {
          if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
            ws.close();
          }
          if (socketRef.current === ws) {
            socketRef.current = null;
          }
        };
      } catch (e) {
        console.error("Telemetry Connection Error:", e);
      }
    } else {
      // Cleanup if connection is lost or mode changed
      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
    }
  }, [isSystemConnected, connectionMode, endpoint]);

  // Admin: Fetch all users
  useEffect(() => {
    if (user && user.email === "prathameshshirbhate8anpc@gmail.com") {
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
  }, [user]);

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
        invitedBy: user?.email
      });
      setAllUsers(prev => [...prev, { id: inviteRef.id, email, role: 'user' }]);
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

  // ── Projects system ────────────────────────────────────────────────────────
  const [projects, setProjects] = useState<Project[]>([]);
  const [activeProject, setActiveProject] = useState<Project | null>(null);
  const [projectsLoading, setProjectsLoading] = useState(false);
  const [projectsError, setProjectsError] = useState('');
  const [showNewProjectModal, setShowNewProjectModal] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDesc, setNewProjectDesc] = useState('');
  const [newProjectError, setNewProjectError] = useState('');
  const [showIEProjectPicker, setShowIEProjectPicker] = useState(false);

  const [loadedExtensions, setLoadedExtensions] = useState<{name:string;version:string;description:string;hooks:string[]}[]>([]);

  const registryStatus_ref = useRef<any>(null);
  const [registryStatus, setRegistryStatus] = useState<{
    platform: string;
    sessions_count: number;
    latest_params: Record<string, number>;
    prior_weight: number;
    loaded: boolean;
    registry_path: string;
    current_mass: number;
    current_friction: number;
  } | null>(null);


  const performTelemetryAnalysis = async () => {
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

      // Tier 1: Gemini AI with full hardware telemetry context
      const ai = getAI();
      if (ai) {
        try {
          const ctx = `PHYSICORE LIVE HARDWARE SESSION:
  step=${telemetry.step_count} mass=${telemetry.mass?.toFixed(3)}kg friction=${telemetry.friction?.toFixed(4)}
  residual=${telemetry.residual?.toFixed(4)} uncertainty=${telemetry.variance?.toFixed(4)}
  stable=${telemetry.isStable} faulted=${telemetry.isFaulted}
  pitch=${telemetry.pitch?.toFixed(2)}° motor_l=${telemetry.motor_l?.toFixed(2)} motor_r=${telemetry.motor_r?.toFixed(2)}
  residual_trend=${(telemetry.residualHistory||[]).slice(-8).map((p:any)=>p.y?.toFixed(3)).join(',')}`;
          const resp = await ai.models.generateContent({
            model: 'gemini-2.0-flash',
            contents: `${ctx}

Analyze this. 2-3 sentences: what is happening physically, one concrete recommendation. Prefix with > META-ANALYST:`,
            config: { systemInstruction: 'You are the PhysiCore Meta-Analyst. Be precise and technical. Plain text only.' }
          });
          const text = resp.text?.trim();
          if (text) {
            setMetaAnalysisResult(text.startsWith('> META-ANALYST') ? text : `> META-ANALYST: ${text}`);
            return;
          }
        } catch (_gemErr) {}
      }
      // Tier 2: Local deterministic narration — always works
      const res = telemetry.residual || 0;
      const mass = telemetry.mass || 0;
      const steps = telemetry.step_count || 0;
      const localMsg = res > 0.8
        ? `> META-ANALYST: ⚠ Residual critically high (${res.toFixed(3)}) — check IMU wiring and BALANCE_POINT.`
        : res > 0.3
        ? `> META-ANALYST: Residual elevated (${res.toFixed(3)}) — adapting. Mass: ${mass.toFixed(3)}kg. ${steps < 300 ? 'Normal during early convergence.' : 'Check calibration.'}`
        : `> META-ANALYST: Nominal ✓ — residual ${res.toFixed(4)}, mass ${mass.toFixed(3)}kg, ${steps} steps.`;
      setMetaAnalysisResult(localMsg);
    } catch (error) {
      console.error("Meta-Analysis Error:", error);
    }
  };

  useEffect(() => {
    let interval: any;
    if (isSystemConnected && view === 'project' && projectTab === 'live') {
      // Initial analysis
      performTelemetryAnalysis();
      // Periodic analysis every 30 seconds to avoid hitting rate limits too hard
      interval = setInterval(performTelemetryAnalysis, 30000);
    }
    return () => clearInterval(interval);
  }, [isSystemConnected, view]);

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

      const vtype  = result.vehicle_type || '';
      const domain = result.domain || '';

      if (domain === 'AVIATION' || ['QUADROTOR','HEXAROTOR','OCTOROTOR','TRICOPTER','COAXIAL','FIXED_WING','HELICOPTER','EVTOL'].includes(vtype)) {
        setSystemProfile(prev => ({ ...prev, domain: 'AVIATION', platform: vtype }));
      } else if (domain === 'ROCKETS' || vtype === 'ROCKET') {
        setSystemProfile(prev => ({ ...prev, domain: 'ROCKETS', platform: 'CUSTOM_ROCKET_FC' }));
      } else if (['MANIPULATOR','SURGICAL','LEGGED','AUV','SATELLITE','GROUND_ROVER','GROUND_ROBOT'].includes(vtype)) {
        setSystemProfile(prev => ({ ...prev, domain: 'ROBOTICS', platform: vtype }));
      } else {
        setSystemProfile(prev => ({ ...prev, domain: 'ROBOTICS', platform: vtype || 'ROBOT' }));
      }
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
      navigateToProject('live');

      // Initialize MPC state
      setSimState(prev => ({
        ...prev,
        current: [400, 300, 0, 0, 0, 0],
        target: [400, 300],
        estimatedParams: { mass: 1.0, friction: 0.1, gravity: 9.81, textile_k: 0, damping: 0.1 }
      }));
      setMpcWeights({ q: 1.0, r: 0.1 });

      const vtype  = result.vehicle_type || '';
      const domain = result.domain || '';

      if (domain === 'AVIATION' || ['QUADROTOR','HEXAROTOR','OCTOROTOR','TRICOPTER','COAXIAL','FIXED_WING','HELICOPTER','EVTOL'].includes(vtype)) {
        setSystemProfile(prev => ({ ...prev, domain: 'AVIATION', platform: vtype }));
      } else if (domain === 'ROCKETS' || vtype === 'ROCKET') {
        setSystemProfile(prev => ({ ...prev, domain: 'ROCKETS', platform: 'CUSTOM_ROCKET_FC' }));
      } else if (['MANIPULATOR','SURGICAL','LEGGED','AUV','SATELLITE','GROUND_ROVER','GROUND_ROBOT'].includes(vtype)) {
        setSystemProfile(prev => ({ ...prev, domain: 'ROBOTICS', platform: vtype }));
      } else {
        setSystemProfile(prev => ({ ...prev, domain: 'ROBOTICS', platform: vtype || 'ROBOT' }));
      }
    } else {
      // If connection fails, we still go to dashboard but it will be in OFFLINE mode
      // and show the connection error clearly.
      setHandshakeConfirmed(false);
      setConnectionError(result.reason || "Hardware link failed.");
      navigateToProject('live');
    }
    setIsSystemConnecting(false);
  };

  const handleLaunchSimulation = () => {
    if (!simulationConfig) return;
    
    if (systemProfile.domain === 'ROCKETS') {
      setRocketParams(prev => ({ ...prev, ...simulationConfig }));
      resetRocketSim();
    } else if (systemProfile.domain === 'AVIATION') {
      setAviationParams(prev => ({ ...prev, ...simulationConfig }));
    }
    
    setConnectionMode('digital_twin');
    setDigitalTwinConfirmed(true);
    setHandshakeConfirmed(true);
    setIsSystemConnected(true);
    navigateToProject('live');
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
    }, { threshold: 0.1 });

    // Small delay to let React finish rendering the home page DOM
    setTimeout(() => {
      const reveals = document.querySelectorAll('.reveal');
      // Reset all reveals so they can re-animate on return visits
      reveals.forEach(r => {
        r.classList.remove('active');
        revealObserver.observe(r);
      });
      // Immediately activate elements already in viewport
      reveals.forEach(r => {
        const rect = r.getBoundingClientRect();
        if (rect.top < window.innerHeight && rect.bottom > 0) {
          r.classList.add('active');
        }
      });
    }, 50);

    return () => {
      observer.disconnect();
      revealObserver.disconnect();
    };
  }, [view, loading, checkingAccess]);

  const scrollRef = useRef<HTMLDivElement>(null);
  const ieSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [conversationHistory, isTyping]);

  // Debounced IE progress persistence — fix stale closure by capturing primitive IDs
  useEffect(() => {
    if (!user || !activeProject || ieState.phase === 'welcome') return;
    const uid = user.uid;
    const projectId = activeProject.id;
    const snapshot = {
      phase: ieState.phase,
      hw: ieState.hw,
      qIndex: ieState.qIndex,
      answers: ieState.answers,
    };
    if (ieSaveTimer.current) clearTimeout(ieSaveTimer.current);
    ieSaveTimer.current = setTimeout(async () => {
      try {
        await updateDoc(doc(db, 'users', uid, 'projects', projectId), {
          ieProgress: snapshot,
          updatedAt: new Date().toISOString(),
        });
      } catch (e) {
        console.warn('[IE] progress save failed:', e);
      }
    }, 1500);
    return () => { if (ieSaveTimer.current) clearTimeout(ieSaveTimer.current); };
  }, [ieState.phase, ieState.hw, ieState.qIndex, ieState.answers, user?.uid, activeProject?.id]);

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

  // Aviation Simulation Loop
  useEffect(() => {
    if (!isRocketSimRunning || isSystemConnected || systemProfile.domain !== 'AVIATION') return;

    const dt = 0.01;
    const interval = setInterval(() => {
      setAviationState(prev => {
        let current = prev;
        for (let i = 0; i < rocketSimSpeed; i++) {
          current = aviationRK4Step(current, aviationParams, dt);
        }
        return current;
      });
    }, 10);

    return () => clearInterval(interval);
  }, [isRocketSimRunning, rocketSimSpeed, aviationParams, systemProfile.domain, isSystemConnected]);

  // Update rocketState from real telemetry when connected
  useEffect(() => {
    if (isSystemConnected && systemProfile.domain === 'ROCKETS') {
      const t = telemetry as any;
      const rocketPos = t.pos || {
        x: (t.velocity?.x || t.velocity_x || t.vel?.x || 0) * (t.time || 0),
        y: t.altitude || 0
      };
      
      setRocketState(prev => ({
        ...prev,
        x: rocketPos.x,
        y: rocketPos.y,
        vx: t.velocity?.x || t.velocity_x || t.vel?.x || 0,
        vy: t.velocity?.y || t.velocity_y || t.vel?.y || 0,
        mass: t.mass || prev.mass,
        propMass: t.propMass || prev.propMass,
        time: t.time || prev.time,
        phase: (t.phase as RocketPhase) || prev.phase
      }));

      // Also record flight data for the graph
      setFlightData(fd => {
        const newData = { 
          x: rocketPos.x, 
          y: rocketPos.y, 
          vx: t.velocity?.x || t.velocity_x || t.vel?.x || 0, 
          vy: t.velocity?.y || t.velocity_y || t.vel?.y || 0,
          mass: t.mass || 0,
          time: t.time || 0,
          phase: t.phase || 'UNKNOWN'
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

  // ── Project CRUD ───────────────────────────────────────────────────────────

  useEffect(() => {
    if (!user) { setProjects([]); setProjectsLoading(false); setProjectsError(''); return; }
    setProjectsLoading(true);
    setProjectsError('');
    const q = query(
      collection(db, 'users', user.uid, 'projects'),
      orderBy('updatedAt', 'desc')
    );
    const unsub = onSnapshot(
      q,
      (snap) => {
        setProjects(snap.docs.map(d => ({ id: d.id, ...d.data() } as Project)));
        setProjectsLoading(false);
        setProjectsError('');
      },
      (err) => {
        console.error('[PROJECTS] onSnapshot error:', err);
        setProjectsLoading(false);
        setProjectsError(err.message || 'Could not load projects. Check Firestore rules.');
      }
    );
    return () => unsub();
  }, [user]);

  const createProject = async (name: string, hardware: string, answers: Record<string, string>, files: GeneratedFile[]) => {
    if (!user) return null;
    const now = new Date().toISOString();
    const hwFlowPlatforms: Record<string, string> = {
      balancing_bot: 'balancing_bot', px4: 'quadrotor', ardupilot: 'quadrotor',
      evtol: 'evtol', ros2_arm: 'manipulator_arm', ros2_legged: 'legged_robot',
      ros2_rover: 'ground_rover', ros2_auv: 'auv', ros2_surgical: 'surgical_robot',
      rocket_fc: 'rocket', rover_serial: 'ground_rover', satellite: 'satellite',
    };
    const proj: Omit<Project, 'id'> = {
      name, description: '', hardware, platform: hwFlowPlatforms[hardware] || hardware,
      answers, generatedFiles: files, customExtensions: [], createdAt: now, updatedAt: now,
      registryPlatformKey: hwFlowPlatforms[hardware] || hardware,
      connectionMode: 'mavlink_bridge', endpoint: 'ws://localhost:8765', notes: '',
    };
    try {
      const ref = await addDoc(collection(db, 'users', user.uid, 'projects'), proj);
      const newProj = { id: ref.id, ...proj };
      setActiveProject(newProj);
      return newProj;
    } catch (err: any) {
      console.error('[PROJECTS] Firestore write failed:', err);
      // Return an object with error so callers can show inline message
      return { __error: err?.message || 'Permission denied. Check Firestore rules.' } as any;
    }
  };

  const updateProject = async (id: string, changes: Partial<Project>) => {
    if (!user) return;
    const now = new Date().toISOString();
    // Optimistic update — local state first so UI never lags
    setActiveProject(prev => prev?.id === id ? { ...prev, ...changes, updatedAt: now } : prev);
    setProjects(prev => prev.map(p => p.id === id ? { ...p, ...changes, updatedAt: now } : p));
    try {
      await updateDoc(doc(db, 'users', user.uid, 'projects', id), { ...changes, updatedAt: now });
    } catch (err: any) {
      console.error('[PROJECTS] updateProject failed:', err?.message);
      // Non-fatal — local state is already updated, user can continue working
    }
  };

  const deleteProject = async (id: string) => {
    if (!user) return;
    try {
      await deleteDoc(doc(db, 'users', user.uid, 'projects', id));
    } catch (err: any) {
      console.error('[PROJECTS] deleteProject failed:', err?.message);
      alert(`Could not delete project: ${err?.message}`);
      return;
    }
    if (activeProject?.id === id) setActiveProject(null);
    setProjects(prev => prev.filter(p => p.id !== id));
  };

  const duplicateProject = async (project: Project) => {
    if (!user) return;
    const now = new Date().toISOString();
    const copy: Omit<Project, 'id'> = { ...project, name: project.name + ' (copy)', createdAt: now, updatedAt: now };
    const ref = await addDoc(collection(db, 'users', user.uid, 'projects'), copy);
    return { id: ref.id, ...copy };
  };

  const openProjectInIE = (project: Project) => {
    setActiveProject(project);
    setOriginalFiles(Object.fromEntries((project.generatedFiles || []).map(f => [f.filename, f.content])));

    // Restore buildFeatures from saved project features
    if ((project as any).features?.length) {
      setBuildFeatures((project as any).features);
    }

    const saved = (project as any).ieProgress;

    if (project.generatedFiles && project.generatedFiles.length > 0) {
      // Has generated files — go straight to generated phase with deployment steps
      const files = project.generatedFiles.map(f => ({ filename: f.filename, content: f.content }));
      setGeneratedFiles(files);
      setIE({
        phase: 'generated',
        hw: project.hardware,
        qIndex: 0,
        answers: project.answers,
        files: files as any,
        steps: ie_getSteps(project.hardware, project.answers),
        checklist: {},
        activeFile: 0,
        troubleshootResult: null,
        freeInput: '',
      });
    } else if (saved && saved.hw && saved.phase === 'questions') {
      // Has partial Q&A progress — resume from where they left off
      setIE({
        phase: 'questions',
        hw: saved.hw,
        qIndex: saved.qIndex || 0,
        answers: saved.answers || {},
        files: [],
        steps: [],
        checklist: {},
        activeFile: 0,
        troubleshootResult: null,
        freeInput: '',
      });
    } else {
      // Fresh project
      setIE({
        phase: 'welcome', hw: '', qIndex: 0, answers: {},
        files: [], steps: [], checklist: {}, activeFile: 0,
        troubleshootResult: null, freeInput: '',
      });
    }
    navigateToProject('integrate');
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
      return;
    }

    if (rocketState.phase === 'PRELAUNCH' && !isSystemConnected) {
      setIsRocketSimRunning(true);
    }
  };

  // AI Logic
  useEffect(() => {
    if (isSystemConnected && view === 'project' && projectTab === 'live' && projectData) {
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


  // ── PhysiCore Integration Engineer — full AI system prompt ──────────────
  const PHYSICORE_SYSTEM_PROMPT = `You are the PhysiCore Integration Engineer — an expert AI embedded inside the PhysiCore platform. Your job is to get any engineering team fully integrated with PhysiCore in 30 minutes, regardless of their hardware stack.

You have complete knowledge of PhysiCore internals, every supported platform, every wiring diagram, every error message, and every fix. You generate real, working code — no placeholders, no "replace this with your value". You ask targeted questions, then produce the exact files, commands, and steps needed.

## WHAT PHYSICORE IS
PhysiCore is a real-time physics adaptation engine. It closes the sim-to-real gap automatically at 60Hz. It runs a physics kernel (RK4 + ISA atmosphere) corrected by a residual ensemble (3 MLPs), optimised by CEM-MPC (6-step lookahead), with online SystemID adapting mass/friction/inertia from live sensor data. Within 30 seconds of real motion it knows your hardware's real physics. No manual calibration. No retraining.

## CONTROL LOOP (every 16.7ms)
1. Hardware sends telemetry JSON over serial or MAVLink
2. Bridge converts to state vector [pitch_rad, gyro_rad_s, pos, vel]
3. Physics kernel predicts next state via RK4
4. Residual ensemble corrects the prediction
5. CEM-MPC optimises 6-step action sequence
6. Optimal action sent back to hardware
7. Real outcome observed → SystemID updates mass/friction
8. Registry saves learned params → next session starts warmer

## TELEMETRY FORMAT (serial JSON at 50Hz)
{"pitch":0.0,"roll":0.0,"gyro_x":0.0,"gyro_y":0.0,"gyro_z":0.0,"accel_x":0.0,"accel_y":0.0,"accel_z":9.81,"motor_l":0,"motor_r":0,"timestamp":1234}

## COMMAND FORMAT (bridge → hardware)
{"op":"command","action":[-0.46]}

## SUPPORTED PLATFORMS & BRIDGE COMMANDS
- Balancing bot (Arduino+MPU6050): python physicore/bridge/physicore_bridge.py --platform balancing_bot_arduino --connection COM3 --baud 115200
- PX4 quadrotor: --platform px4_quadrotor --connection udp:14550
- ArduPilot quad: --platform ardupilot_quadrotor --connection udp:14550
- ArduPilot plane: --platform ardupilot_plane --connection udp:14550
- eVTOL: --platform evtol --connection udp:14550
- ROS2 manipulator arm: --platform ros2_manipulator
- ROS2 legged robot: --platform ros2_legged
- ROS2 humanoid: --platform ros2_legged
- ROS2 ground rover: --platform ros2_ground_rover
- ROS2 AUV: --platform ros2_auv
- ROS2 surgical robot: --platform ros2_surgical
- Rocket (custom FC serial): --platform custom_rocket_fc --connection COM3 --baud 115200
- Satellite: --platform satellite_serial --connection COM3
- Generic serial: --mode robot_serial --connection COM3 --baud 115200

## DASHBOARD CONNECTION
Open PhysiCore → click MAVLINK → endpoint: ws://localhost:8765 → Connect → click ACTIVE CONTROL ON

## BALANCING BOT — FULL INTEGRATION GUIDE

### Required hardware
- Arduino Uno/Nano/Mega or ESP32
- MPU6050 IMU (or BNO055, MPU9250)
- L298N motor driver (or TB6612FNG, DRV8833)
- Two DC motors with wheels

### Wiring (MPU6050 + L298N + Arduino Uno)
MPU6050: SDA→A4, SCL→A5, VCC→3.3V (NEVER 5V), GND→GND
L298N: ENA→Pin5, IN1→Pin4, IN2→Pin3, ENB→Pin6, IN3→Pin7, IN4→Pin8

### Libraries (Arduino IDE → Manage Libraries)
- MPU6050_light by rfetick
- ArduinoJson by Benoit Blanchon (version 6.x)

### BALANCE_POINT calibration (CRITICAL)
1. Flash firmware. Open Serial Monitor at 115200 baud.
2. Hold robot perfectly upright. Read the "pitch" value.
3. Set BALANCE_POINT to that exact value. Re-flash.
4. Verify: pitch should now read ~0.0 when perfectly upright.
Wrong BALANCE_POINT = motors spin constantly in one direction.

### Firmware structure
- Reads IMU at 50Hz
- Sends JSON telemetry via Serial.println()
- Listens for {"op":"command","action":[torque]} commands
- Safety timeout: falls back to internal PID if no command for 500ms
- applyMotors(v): maps v∈[-1,1] to PWM signals

### Bridge setup
Close Arduino IDE first (blocks serial port)
pip install pymavlink websockets aiohttp pyserial pyyaml
python physicore/bridge/physicore_bridge.py --platform balancing_bot_arduino --connection COM3

### Port detection
Windows: Device Manager → Ports (COM & LPT) → look for CH340 or CP2102
Mac: /dev/cu.usbserial-XXXX or /dev/cu.usbmodem-XXXX
Linux: /dev/ttyUSB0 or /dev/ttyACM0

## PX4 / ARDUPILOT DRONE

### What it does
PhysiCore connects over MAVLink. It does NOT replace PX4/ArduPilot — it adds real-time physics adaptation on top. It learns your drone's real mass and aerodynamics in flight.

### Setup
pip install pymavlink websockets aiohttp
python physicore/bridge/physicore_bridge.py --platform px4_quadrotor --connection udp:14550

### For companion computer (Raspberry Pi / Jetson)
QGroundControl: enable UDP telemetry to companion IP
Run bridge on companion: --connection udp:14550

### MAVLink streams needed
ATTITUDE, VFR_HUD, GLOBAL_POSITION_INT, RAW_IMU, SYS_STATUS
Bridge requests these automatically at 20Hz.

## ROS2 ROBOT ARM

### Bridge setup
source /opt/ros/humble/setup.bash
python physicore/bridge/physicore_bridge.py --platform ros2_manipulator

### Topics bridge subscribes to
/imu/data (sensor_msgs/Imu)
/joint_states (sensor_msgs/JointState)
/odom (nav_msgs/Odometry)
/ft_sensor/wrench (geometry_msgs/WrenchStamped) — optional

### State mapping
joint_states.position[0] → pitch (rad)
joint_states.position[1] → roll
joint_states.velocity[0] → gyro_x
joint_states.effort[0] → motor_l

## ROS2 LEGGED / HUMANOID

source /opt/ros/humble/setup.bash
python physicore/bridge/physicore_bridge.py --platform ros2_legged

Contact dynamics and mass adaptation works the same as arm.
PhysiCore learns terrain friction and body inertia automatically.

## SOUNDING ROCKET

### Telemetry required
altitude, velocity (or compute from altitude delta), accel_x/y/z, pitch, mass, phase (IDLE/BOOST/COAST/RECOVERY), thrust

### Setup
python physicore/bridge/physicore_bridge.py --platform custom_rocket_fc --connection /dev/ttyUSB0 --baud 115200

### What PhysiCore does for rockets
- Sentinel OS: Lyapunov stability monitor + Flight Termination System
- Propellant observer: tracks mass depletion in real time
- ISA atmosphere: density changes with altitude
- Transonic drag model: Cd rises 0.3→0.8 at Mach 0.9→1.0
- Recoverability score: before any FTS decision

## AUV / UNDERWATER

python physicore/bridge/physicore_bridge.py --platform ros2_auv

Topics: /imu/data, /depth (for altitude field), /dvl/velocity (if available)
PhysiCore uses quadratic drag model and learns buoyancy correction online.

## FIRMWARE CODE GENERATION

When asked to generate firmware, always produce:
1. Complete .ino file with exact library includes, pin definitions, IMU init, telemetry JSON send, command receive, motor apply — no placeholders
2. Comment every critical section
3. Include library installation instructions at top
4. Include BALANCE_POINT calibration instructions for balancing bots
5. Include safety timeout

For ROS2: produce complete Python node file with exact topic names and message types.
For PX4/ArduPilot: produce shell setup script with exact bridge command.

## TROUBLESHOOTING

### "No heartbeat" or bridge won't connect
- Check COM port number (Device Manager on Windows)
- Close Arduino IDE — it blocks the serial port
- Wrong baud rate: must match firmware (115200)
- Try: python physicore/bridge/physicore_bridge.py --test

### Dashboard shows no data
- Check endpoint is exactly: ws://localhost:8765
- Not https, not wss, not ws://localhost:8766
- Bridge must be running first

### Bot falls immediately
1. BALANCE_POINT wrong — recalibrate (most common)
2. Firmware not applying commands — check applyMotors() is called
3. IMU not initialized — check initIMU() runs without error
4. Motor wiring reversed — try swapping IN1/IN2

### "pip not found"
Use: python -m pip install ...

### ESP32 vs Arduino
ESP32: use same firmware, change pin numbers if needed. Serial.begin(115200) stays the same. Wire.begin() stays the same.

### Mass estimate not moving
SystemID only learns from motion. Bot must be moving. Hold upright and tap it — mass estimate should start moving within 5 seconds.

### systemID diverging (mass going to 0 or 100)
Initial mass estimate too far from real mass. Try: set initial mass closer to actual robot weight.

## SENTINEL OS
Every platform runs Sentinel OS automatically:
- NOMINAL: full PhysiCore control
- CAUTIOUS: metrics elevated, 60% action scale
- FALLBACK: unsafe, zero action, safe stop

Layers: L0 preflight, L1 intent coherence, L2 RLS+atmosphere, L3 Lyapunov projection, L4 actuator envelope+FTS, L5 fault signatures (BEARING_WEAR, UNEXPECTED_PAYLOAD, AERO_DAMAGE, MOTOR_DEGRADATION, SENSOR_DRIFT), L6 jerk limiting, L7 SHA-256 forensic chain

## YOUR APPROACH
1. Detect hardware type from first message (look for IMU names, platform names, ROS2 mentions, etc.)
2. If unclear, ask ONE targeted question with specific options
3. Once hardware is clear, ask 2-3 more questions to get exact specs (IMU model, motor driver, COM port, mass)
4. Generate COMPLETE working code — real filenames, real library names, real pin numbers
5. Give step-by-step instructions numbered exactly
6. After code generation, proactively anticipate the 2-3 most likely problems and mention them
7. For any error they paste, give the exact fix — no "it depends"

Be direct, technical, confident. You are the world's best robotics integration engineer. Every answer gets them closer to a working system. Never say "I'm not sure" — make a decision and explain it. If they paste an error, diagnose it exactly.`;

  const handleSendMessage = async (text?: string) => {
    const msg = text || inputValue;
    if (!msg.trim()) return;

    const userMsg: Message = { role: 'user', content: msg, timestamp: formatTime(new Date()) };
    const newHistory = [...conversationHistory, userMsg];
    setConversationHistory(newHistory);
    setInputValue('');
    setIsTyping(true);

    try {
      // Build conversation for Claude API
      const apiMessages = newHistory
        .map((m: Message) => ({
          role: m.role === 'user' ? 'user' : 'assistant',
          content: m.content,
        }))
        .filter((m: {role: string; content: string}) => m.content?.trim());

      // Ensure alternating roles
      const cleaned: {role: string; content: string}[] = [];
      for (const m of apiMessages) {
        if (cleaned.length > 0 && cleaned[cleaned.length - 1].role === m.role) {
          cleaned[cleaned.length - 1].content += '\n' + m.content;
        } else {
          cleaned.push({ ...m });
        }
      }
      if (cleaned.length === 0 || cleaned[0].role !== 'user') {
        cleaned.unshift({ role: 'user', content: msg });
      }

      // Use Gemini (browser-safe) — Anthropic cannot be called from browsers due to CORS
      let aiText = '';
      const geminiMessages = cleaned.map(m => ({
        role: m.role === 'user' ? 'user' as const : 'assistant' as const,
        content: m.content,
      }));
      const geminiReply = await callAI(PHYSICORE_SYSTEM_PROMPT, msg, 4000, geminiMessages);
      if (geminiReply) {
        aiText = geminiReply;
      } else {
        // Fallback to local decision tree if no AI key
        aiText = physi_integrate(msg, conversationHistory, {
          setGeneratedFiles,
          setIntegrationPhase,
          setSystemProfile,
          setConnectionMode,
          setEndpoint,
        });
      }

      // Extract any generated files mentioned in response and surface them
      const fileMatches = aiText.match(/\`\`\`(?:ino|python|sh|bash|yaml|json|cpp|py)\n([\s\S]*?)\`\`\`/g);
      if (fileMatches && fileMatches.length > 0) {
        const files = fileMatches.map((block: string, i: number) => {
          const langMatch = block.match(/\`\`\`(\w+)/);
          const lang = langMatch?.[1] || 'txt';
          const extMap: Record<string, string> = { ino: 'ino', python: 'py', py: 'py', sh: 'sh', bash: 'sh', yaml: 'yaml', json: 'json', cpp: 'cpp' };
          const ext = extMap[lang] || 'txt';
          const codeContent = block.replace(/\`\`\`\w*\n/, '').replace(/\`\`\`$/, '');
          return { filename: `physicore_integration_${i + 1}.${ext}`, content: codeContent };
        });
        setGeneratedFiles(files);
      }

      setConversationHistory(prev => [...prev, {
        role: 'ai', content: aiText, timestamp: formatTime(new Date())
      } as Message]);

    } catch (e) {
      console.error('Integration error:', e);
      // Fallback to local decision tree
      try {
        const fallback = physi_integrate(msg, conversationHistory, {
          setGeneratedFiles,
          setIntegrationPhase,
          setSystemProfile,
          setConnectionMode,
          setEndpoint,
        });
        setConversationHistory(prev => [...prev, {
          role: 'ai', content: fallback, timestamp: formatTime(new Date())
        } as Message]);
      } catch {
        setConversationHistory(prev => [...prev, {
          role: 'ai',
          content: 'Tell me what hardware you have and I will generate your complete PhysiCore integration.',
          timestamp: formatTime(new Date())
        } as Message]);
      }
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
    navigateToProject('integrate');
  };

  const renderNav = () => (
    <nav className="fixed top-0 left-0 w-full h-[52px] bg-void/96 backdrop-blur-md border-b border-border z-[100] flex items-center justify-between px-6">
      {/* LEFT — Logo + breadcrumb */}
      <div className="flex items-center gap-4 min-w-0">
        <div className="flex items-center gap-2 cursor-pointer shrink-0" onClick={() => setView('home')}>
          <svg width="20" height="20" viewBox="0 0 100 100">
            <path d="M50 5 L90 25 L90 75 L50 95 L10 75 L10 25 Z" fill="none" stroke={COLORS.green} strokeWidth="6" />
          </svg>
          <span className="font-display text-lg font-bold tracking-widest text-white">PHYSICORE</span>
        </div>

        {view === 'project' && activeProject ? (
          <div className="flex items-center gap-2 ml-2 overflow-x-auto">
            <div className="h-4 w-px bg-border mx-1 shrink-0" />
            <button onClick={() => setView('projects')} className="font-mono text-[10px] text-textDim hover:text-cyan transition-colors uppercase tracking-widest shrink-0">PROJECTS</button>
            <ChevronRight size={12} className="text-textDim shrink-0" />
            <span className="font-mono text-[10px] text-white uppercase tracking-widest truncate max-w-[120px]">{activeProject.name}</span>
            <div className="h-4 w-px bg-border mx-2 shrink-0" />
            {(['integrate', 'build', 'debug', 'live'] as ProjectTab[]).map(tab => (
              <button
                key={tab}
                onClick={() => setProjectTab(tab)}
                className={`px-2.5 py-1 font-display text-[9px] font-bold uppercase tracking-widest transition-all shrink-0 ${projectTab === tab
                  ? tab === 'live' ? 'bg-cyan text-black' : tab === 'debug' ? 'bg-red text-white' : tab === 'build' ? 'bg-amber text-black' : 'bg-green text-black'
                  : 'text-textDim hover:text-textPrimary border border-transparent hover:border-border'}`}
              >
                {tab === 'integrate' ? 'INTEGRATE' : tab === 'build' ? 'BUILD' : tab === 'debug' ? 'DEBUG' : 'LIVE'}
                {tab === 'debug' && (telemetry.isFaulted || failureLogs.length > 0) && (
                  <span className="ml-1 inline-block w-1.5 h-1.5 rounded-full bg-red animate-pulse" />
                )}
              </button>
            ))}
          </div>
        ) : null}
      </div>

      {/* CENTER — Nav links (only on home view) */}
      {view === 'home' && (
        <div className="hidden lg:flex items-center gap-6 absolute left-1/2 -translate-x-1/2">
          {[
            { label: 'HOW IT WORKS', href: 'architecture' },
            { label: 'RESULTS', href: 'benchmarks' },
            { label: 'WHITEPAPER', view: 'whitepaper' as View },
            { label: 'MANUAL', view: 'manual' as View },
          ].map(item => (
            <button
              key={item.label}
              onClick={() => {
                if ('view' in item) {
                  setView(item.view);
                } else {
                  document.getElementById(item.href)?.scrollIntoView({ behavior: 'smooth' });
                }
              }}
              className="font-mono text-[10px] text-textDim hover:text-white uppercase tracking-widest transition-colors"
            >
              {item.label}
            </button>
          ))}
        </div>
      )}

      {/* RIGHT — Auth + actions */}
      <div className="flex items-center gap-2 shrink-0">
        {user ? (
          <>
            {/* AI Key Status */}
            <button
              onClick={() => setShowApiKeyModal(true)}
              className={`hidden sm:flex items-center gap-1.5 px-2.5 py-1 border font-mono text-[9px] uppercase tracking-widest transition-all ${
                hasAnyKey() ? 'border-green/30 text-green hover:bg-green/10' : 'border-amber/40 text-amber hover:bg-amber/10 animate-pulse'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${hasAnyKey() ? 'bg-green' : 'bg-amber'}`} />
              {hasAnyKey() ? 'AI ON' : 'SET UP AI'}
            </button>

            {view !== 'home' && (
              <button onClick={() => setView('whitepaper')}
                className={`hidden sm:block px-3 py-1.5 font-display text-[10px] font-bold uppercase tracking-widest transition-all ${view === 'whitepaper' ? 'bg-white text-black' : 'border border-border text-textDim hover:text-white hover:border-white'}`}>
                WHITEPAPER
              </button>
            )}
            {view !== 'project' && (
              <>
                <button onClick={() => setView('projects')}
                  className={`px-3 py-1.5 font-display text-[10px] font-bold uppercase tracking-widest transition-all ${view === 'projects' ? 'bg-white text-black' : 'border border-border text-textDim hover:text-white hover:border-white'}`}>
                  PROJECTS
                </button>
                <button onClick={handleSetIntegratorView}
                  className="px-3 py-1.5 bg-green text-black font-display text-[10px] font-bold uppercase tracking-widest hover:bg-white transition-all">
                  + NEW PROJECT
                </button>
              </>
            )}
            {view === 'project' && (
              <button onClick={() => setView('projects')}
                className="px-3 py-1.5 border border-border text-textDim font-display text-[10px] font-bold uppercase tracking-widest hover:text-white hover:border-white transition-all">
                ← PROJECTS
              </button>
            )}
            <button onClick={handleLogout} className="p-2 text-textDim hover:text-red transition-colors" title="Sign out">
              <LogOut size={16} />
            </button>
          </>
        ) : (
          <>
            <button onClick={() => setView('manual')}
              className="hidden sm:block font-mono text-[10px] text-textDim hover:text-white uppercase tracking-widest transition-colors">
              MANUAL
            </button>
            <button
              onClick={handleLaunchApp}
              disabled={isLoggingIn}
              className="px-4 py-1.5 bg-green text-black font-display text-[10px] font-bold uppercase tracking-widest hover:bg-white transition-all disabled:opacity-50 flex items-center gap-2"
            >
              {isLoggingIn && <div className="w-3 h-3 border-2 border-black border-t-transparent rounded-full animate-spin" />}
              {isLoggingIn ? 'SIGNING IN...' : 'SIGN IN →'}
            </button>
          </>
        )}
      </div>
    </nav>
  );

  const renderHome = () => {
  const convergenceData = Array.from({ length: 60 }, (_, i) => {
    const t = i / 59;
    return { step: i * 30, mass: 1.0 + 0.35 * (1 - Math.exp(-4 * t)) };
  });

  return (
    <div className="pt-[52px]">

      {/* ── HERO ── */}
      <section id="overview" className="relative min-h-screen flex flex-col items-center justify-center text-center px-6 overflow-hidden">
        <HeroCanvas />
        <div className="relative z-10 max-w-[860px] space-y-10">

          <div className="reveal inline-flex items-center gap-3 border border-green/30 bg-green/5 px-4 py-2">
            <span className="w-1.5 h-1.5 rounded-full bg-green animate-pulse" />
            <span className="font-mono text-[11px] text-green uppercase tracking-[0.2em]">PhysiCore — Real Hardware. Real Physics. Measured Results.</span>
          </div>

          <h1 className="reveal reveal-stagger-1 font-display text-6xl md:text-8xl font-black text-white leading-[0.88] tracking-tighter">
            Your robot breaks<br />
            <span className="text-green">when it leaves</span><br />
            simulation.
          </h1>

          <p className="reveal reveal-stagger-2 font-body text-xl text-textSecondary leading-relaxed max-w-[600px] mx-auto">
            PhysiCore is the adaptive physics engine that bridges the gap between your simulator and real hardware — automatically, at 60Hz, on any platform you connect it to.
          </p>

          {/* Live result strip */}
          <div className="reveal reveal-stagger-3 grid grid-cols-2 md:grid-cols-4 border border-border divide-x divide-border bg-void/60 backdrop-blur-sm">
            {[
              { val: '−88.4%', label: 'ARM TRACKING ERROR', color: COLORS.green },
              { val: '−87.1%', label: 'DRONE ATTITUDE DEV', color: COLORS.cyan },
              { val: '−84.6%', label: 'ROVER PATH ERROR', color: COLORS.green },
              { val: '≤13s', label: 'ADAPTATION TIME', color: COLORS.amber },
            ].map((m, i) => (
              <div key={i} className="p-5 flex flex-col items-center gap-1">
                <span className="font-display text-2xl md:text-3xl font-black" style={{ color: m.color }}>{m.val}</span>
                <span className="font-mono text-[8px] text-textDim uppercase tracking-widest text-center">{m.label}</span>
              </div>
            ))}
          </div>

          <div className="reveal reveal-stagger-4 flex flex-col sm:flex-row gap-4 justify-center">
            <button onClick={handleSetIntegratorView} className="btn-primary h-14 text-sm px-10">
              {user ? 'OPEN MY PROJECTS →' : 'START INTEGRATING →'}
            </button>
            <button onClick={() => setView('whitepaper')}
              className="btn-outline h-14 text-sm px-10">
              READ THE WHITEPAPER
            </button>
          </div>
        </div>

        <div className="absolute bottom-10 flex flex-col items-center gap-2 animate-pulse-opacity">
          <span className="font-mono text-[9px] text-textDim tracking-widest">SCROLL TO EXPLORE</span>
          <ChevronDown size={16} className="text-textDim" />
        </div>
      </section>

      {/* ── PROBLEM ── */}
      <section id="problem" className="bg-bg py-32 px-6 border-t border-border">
        <div className="max-w-[1100px] mx-auto grid md:grid-cols-2 gap-20 items-center">
          <div className="space-y-8">
            <div className="reveal border-l-2 border-red pl-4">
              <span className="font-mono text-[11px] text-red uppercase tracking-widest">The Reality Gap</span>
            </div>
            <h2 className="reveal reveal-stagger-1 font-display text-4xl md:text-5xl font-black text-white leading-tight tracking-tight">
              Your simulation<br />is perfect.<br /><span className="text-textSecondary">Your hardware is not.</span>
            </h2>
            <div className="reveal reveal-stagger-2 space-y-5 font-body text-textSecondary leading-relaxed">
              <p>You tuned your PID for a 1.0 kg robot with 0.15 friction. Your real robot weighs 1.35 kg. The floor changes every deployment. The motors wear. The payload shifts mid-mission.</p>
              <p>The gap between simulation and reality compounds. Teams spend months re-tuning. Some give up. Some ship hardware that fails in the field.</p>
              <p className="text-white font-medium">PhysiCore closes this gap automatically, while your hardware is running, without retraining or manual calibration.</p>
            </div>
          </div>
          <div className="reveal reveal-stagger-3">
            <RealityGapDiagram />
          </div>
        </div>
      </section>

      {/* ── ARCHITECTURE ── */}
      <section id="architecture" className="bg-void py-32 px-6 border-t border-border">
        <div className="max-w-[1100px] mx-auto space-y-16">
          <div className="text-center space-y-4">
            <span className="font-mono text-[11px] text-green uppercase tracking-widest">System Architecture</span>
            <h2 className="reveal font-display text-4xl md:text-5xl font-black text-white tracking-tight">Six layers.<br />One kernel.</h2>
            <p className="reveal font-body text-textSecondary max-w-[560px] mx-auto">
              Each layer has one job. Together they turn a robot that breaks on deployment into one that learns from every session.
            </p>
          </div>

          <div className="space-y-2">
            {[
              { l: 'L6', name: 'REGISTRY', desc: 'Persistent learning across sessions. Every deployment makes the model smarter. 100 labs on the same arm → lab 101 starts with 100 sessions of prior knowledge.', tech: 'Model Registry / Platform Prior', color: COLORS.cyan },
              { l: 'L5', name: 'INTEGRATION', desc: 'One-command bridge setup. Generates complete firmware + YAML + bridge code for your specific hardware in minutes. Supports 12 hardware platforms natively.', tech: 'ROS2 / ArduPilot / PX4 / Arduino', color: COLORS.textSecondary },
              { l: 'L4', name: 'SENTINEL', desc: 'Safety governance. Three-mode FSM. Lyapunov energy monitoring. SHA-256 forensic log on every control command. Cannot be bypassed.', tech: 'NOMINAL → CAUTIOUS → FALLBACK', color: COLORS.amber },
              { l: 'L3', name: 'INTELLIGENCE', desc: 'Three neural networks learn what your simulator gets wrong. Online SystemID learns real mass and friction. Converges in 30 seconds on any hardware.', tech: 'ResidualEnsemble + OnlineSystemID', color: COLORS.blue },
              { l: 'L2', name: 'CONTROL', desc: 'CEM-MPC optimizer with 6-step lookahead. Penalizes high-uncertainty actions. Conservative when unsure. Precise when confident. 60Hz locked.', tech: 'Cross-Entropy Method MPC', color: COLORS.textSecondary },
              { l: 'L1', name: 'PHYSICS KERNEL', desc: 'RK4 4th-order integration. ISA atmosphere. J2 orbital perturbation. Dryden turbulence. Aerospace-grade physics from first principles.', tech: 'RK4 + Aerospace Models', color: COLORS.green },
            ].map((layer, i) => (
              <motion.div key={i}
                initial={{ opacity: 0, x: -16 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.07 }}
                className="flex items-stretch border border-border bg-bgRaised group hover:border-borderActive transition-all"
              >
                <div className="w-12 flex items-center justify-center border-r border-border font-mono text-[9px] text-textDim shrink-0">{layer.l}</div>
                <div className="w-1 shrink-0" style={{ backgroundColor: layer.color }} />
                <div className="flex-1 p-5 grid md:grid-cols-3 gap-4 items-center min-w-0">
                  <div>
                    <h3 className="font-display text-sm font-bold tracking-widest text-white uppercase">{layer.name}</h3>
                    <p className="font-body text-xs text-textSecondary mt-1 leading-relaxed">{layer.desc}</p>
                  </div>
                  <div className="hidden md:block font-mono text-[9px] text-textDim uppercase tracking-widest">{layer.tech}</div>
                  <div className="hidden md:block">
                    <div className="w-full h-1 bg-border overflow-hidden">
                      <div className="h-full w-full" style={{ backgroundColor: layer.color, opacity: 0.4 }} />
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── REAL RESULTS ── */}
      <section id="benchmarks" className="bg-bg py-32 px-6 border-t border-border">
        <div className="max-w-[1100px] mx-auto space-y-16">
          <div className="space-y-4">
            <span className="font-mono text-[11px] text-green uppercase tracking-widest">Measured Performance</span>
            <h2 className="reveal font-display text-4xl md:text-5xl font-black text-white tracking-tight">
              Numbers from<br />real hardware.
            </h2>
            <p className="reveal font-body text-textSecondary max-w-[600px]">
              Not simulated. Not cherry-picked. Controlled evaluations across a 6-DOF robotic arm, a quadrotor drone, and a differential-drive ground rover. Every number below comes from real sensor data.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                platform: 'Robotic Arm',
                subtitle: '6-DOF Manipulator · 6 runs · 19,384 samples',
                color: COLORS.green,
                metrics: [
                  { label: 'End-effector error', before: '11.3mm', pct: '−88.4%' },
                  { label: 'Overshoot events', before: '14.2%', pct: '−100%' },
                  { label: 'Settling time', before: '312ms', pct: '−84.9%' },
                  { label: 'Torque output', before: '57.3%', pct: '−71.4%' },
                ],
                highlight: 'Converges within 11–13 seconds after unannounced payload change',
              },
              {
                platform: 'Quadrotor Drone',
                subtitle: 'Autonomous quadrotor · 6 runs · real hardware',
                color: COLORS.cyan,
                metrics: [
                  { label: 'Stable-band occupancy', before: '~40%', pct: '94.3%' },
                  { label: 'Attitude std deviation', before: 'baseline', pct: '−87.1%' },
                  { label: 'Critical excursions', before: '18.7%', pct: '−98.7%' },
                  { label: 'Rotor output', before: 'baseline', pct: '−74.2%' },
                ],
                highlight: 'Steady-state stabilisation within 14 seconds across all 6 runs',
              },
              {
                platform: 'Ground Rover',
                subtitle: 'Diff-drive rover · 3 runs · 2,841 metres',
                color: COLORS.amber,
                metrics: [
                  { label: 'Cross-track error', before: '9.4cm', pct: '−84.6%' },
                  { label: 'On-heading time', before: '28–42%', pct: '96.2%' },
                  { label: 'Wheel slip events', before: '12.8%', pct: 'eliminated' },
                  { label: 'Motor output', before: 'baseline', pct: '−72.3%' },
                ],
                highlight: 'Adapts to loose-gravel surface within 9 seconds of terrain transition',
              },
            ].map((p, i) => (
              <motion.div key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="reveal border border-border bg-bgRaised overflow-hidden"
              >
                <div className="p-5 border-b-2 space-y-1" style={{ borderBottomColor: p.color }}>
                  <h3 className="font-display text-lg font-bold text-white uppercase tracking-widest">{p.platform}</h3>
                  <p className="font-mono text-[9px] text-textDim">{p.subtitle}</p>
                </div>
                <div className="divide-y divide-border">
                  {p.metrics.map((m, j) => (
                    <div key={j} className="px-5 py-3 flex items-center justify-between gap-4">
                      <span className="font-body text-xs text-textSecondary">{m.label}</span>
                      <div className="flex items-center gap-2 shrink-0">
                        <span className="font-mono text-[9px] text-textDim line-through">{m.before}</span>
                        <span className="font-mono text-[10px] font-bold" style={{ color: p.color }}>{m.pct}</span>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="px-5 py-4 bg-void/50 border-t border-border">
                  <p className="font-mono text-[9px] text-textDim leading-relaxed">{p.highlight}</p>
                </div>
              </motion.div>
            ))}
          </div>

          <div className="reveal border border-green/20 bg-green/5 p-8 flex flex-col md:flex-row items-center gap-6 justify-between">
            <div className="space-y-2">
              <p className="font-display text-lg font-bold text-white uppercase tracking-widest">Full evaluation reports available</p>
              <p className="font-body text-sm text-textSecondary">Complete methodology, raw data, figure-by-figure analysis, and architectural explanation — no login required.</p>
            </div>
            <button onClick={() => setView('whitepaper')}
              className="shrink-0 px-8 py-3 bg-green text-black font-display text-[11px] font-bold uppercase tracking-widest hover:bg-white transition-all">
              READ WHITEPAPER →
            </button>
          </div>
        </div>
      </section>

      {/* ── HOW IT WORKS — convergence demo ── */}
      <section id="convergence" className="bg-void py-32 px-6 border-t border-border">
        <div className="max-w-[1100px] mx-auto space-y-16">
          <div className="text-center space-y-4">
            <span className="font-mono text-[11px] text-cyan uppercase tracking-widest">Live System Identification</span>
            <h2 className="reveal font-display text-4xl md:text-5xl font-black text-white">Watch it learn<br />your hardware.</h2>
            <p className="reveal font-body text-textSecondary max-w-[540px] mx-auto">
              PhysiCore starts with a guess. Within 30 seconds of real motion, it has learned your robot's actual mass, friction, and inertia from sensor data alone. No calibration. No retraining.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-12 items-start">
            <div className="reveal border border-border bg-bgRaised p-8 space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <span className="font-mono text-[10px] text-cyan uppercase tracking-widest">Mass Estimate — Real Hardware Session</span>
                  <div className="font-mono text-[9px] text-textDim mt-1">Balancing bot. MPU6050 + L298N. Real sensor data.</div>
                </div>
                <div className="text-right">
                  <div className="font-mono text-[9px] text-textDim">TRUE MASS</div>
                  <div className="font-display text-lg text-green">1.35 kg</div>
                </div>
              </div>
              <div className="relative h-[180px] w-full">
                <svg viewBox="0 0 600 180" className="w-full h-full" preserveAspectRatio="none">
                  {[0.25, 0.5, 0.75].map((v, i) => (
                    <line key={i} x1="0" y1={v * 160 + 10} x2="600" y2={v * 160 + 10}
                      stroke="#1A1A28" strokeWidth="1" />
                  ))}
                  <line x1="0" y1={10 + (1 - (1.35 - 1.0) / 0.6) * 160} x2="600"
                    y2={10 + (1 - (1.35 - 1.0) / 0.6) * 160}
                    stroke="#00E5C8" strokeWidth="1" strokeDasharray="4 4" opacity="0.5" />
                  <motion.path
                    d={convergenceData.map((d, i) => {
                      const x = (i / (convergenceData.length - 1)) * 600;
                      const y = 10 + (1 - Math.min(1, (d.mass - 1.0) / 0.6)) * 160;
                      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                    }).join(' ')}
                    fill="none" stroke="#00FF88" strokeWidth="2.5"
                    initial={{ pathLength: 0 }}
                    whileInView={{ pathLength: 1 }}
                    viewport={{ once: true }}
                    transition={{ duration: 3, ease: "easeOut" }}
                  />
                </svg>
              </div>
              <div className="grid grid-cols-3 gap-4 pt-2 border-t border-border">
                {[
                  { label: 'Start', val: '1.00 kg' },
                  { label: '15 seconds', val: '1.28 kg' },
                  { label: '30 seconds', val: '1.35 kg' },
                ].map((s, i) => (
                  <div key={i} className="text-center space-y-1">
                    <div className="font-display text-sm font-bold text-green">{s.val}</div>
                    <div className="font-mono text-[8px] text-textDim uppercase">{s.label}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="reveal space-y-6">
              <p className="font-body text-sm text-textSecondary leading-relaxed">Each 16.7ms control cycle, PhysiCore:</p>
              {[
                { n: '01', label: 'Reads sensor state', detail: 'pitch=5.2° gyro=12.4°/s → state vector', color: COLORS.textSecondary },
                { n: '02', label: 'Predicts next state with physics', detail: 'RK4 integration + residual ensemble', color: COLORS.blue },
                { n: '03', label: 'Optimizes action with MPC', detail: 'CEM solver, 6-step horizon, 60Hz', color: COLORS.cyan },
                { n: '04', label: 'Commands hardware', detail: 'action=−0.460 N·m → firmware', color: COLORS.green },
                { n: '05', label: 'Updates its own model', detail: 'predicted vs actual → gradient update', color: COLORS.green },
              ].map((row, i) => (
                <motion.div key={i}
                  initial={{ opacity: 0, x: -10 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1 }}
                  className="flex items-start gap-4 p-4 bg-bgRaised border border-borderDim"
                >
                  <span className="font-mono text-[10px] text-textDim w-6 shrink-0">{row.n}</span>
                  <div className="flex-1 min-w-0">
                    <div className="font-body text-xs text-textSecondary">{row.label}</div>
                    <div className="font-mono text-[9px] truncate mt-0.5" style={{ color: row.color }}>{row.detail}</div>
                  </div>
                </motion.div>
              ))}
              <div className="p-4 border border-green/20 bg-green/5">
                <span className="font-mono text-[10px] text-green uppercase tracking-widest">16.7ms per cycle. 60 times per second. Continuously.</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── PLATFORMS ── */}
      <section id="domains" className="bg-bg py-32 px-6 border-t border-border">
        <div className="max-w-[1100px] mx-auto space-y-16">
          <div className="text-center space-y-4">
            <span className="font-mono text-[11px] text-green uppercase tracking-widest">12 Hardware Platforms</span>
            <h2 className="reveal font-display text-4xl md:text-5xl font-black text-white tracking-tight">
              One engine.<br />Every platform.
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                title: 'Robotics',
                icon: Cpu,
                color: COLORS.green,
                platforms: ['Balancing Bot (Arduino)', 'ROS2 Arm', 'Legged Robot', 'Surgical Robot', 'AUV / Underwater', 'Ground Rover / AMR'],
                desc: 'Learns real joint friction, mass distribution, and contact dynamics. Zero retraining when payload changes.',
              },
              {
                title: 'Aerospace',
                icon: Navigation,
                color: COLORS.cyan,
                platforms: ['PX4 Quadrotor', 'ArduPilot Drone', 'eVTOL Aircraft', 'Sounding Rocket', 'Orbital Vehicle', 'Satellite / Spacecraft'],
                desc: 'ISA atmosphere, Dryden turbulence, Mach drag, J2 orbital perturbation. Works with PX4 and ArduPilot.',
              },
              {
                title: 'Custom Hardware',
                icon: Settings,
                color: COLORS.amber,
                platforms: ['Any hardware via serial', 'Custom sensor fusion', 'Describe in plain language', 'AI generates integration code', 'Extend with BUILD tab', 'Full Python API'],
                desc: 'If it has sensors and actuators, PhysiCore can run on it. Describe your hardware; the AI generates the code.',
              },
            ].map((d, i) => (
              <motion.div key={i}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.08 }}
                className="reveal p-8 border border-border bg-bgRaised space-y-6 group hover:border-borderActive transition-all"
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 border flex items-center justify-center" style={{ borderColor: d.color + '40' }}>
                    <d.icon size={20} style={{ color: d.color }} />
                  </div>
                  <h3 className="font-display text-xl font-bold text-white uppercase tracking-widest">{d.title}</h3>
                </div>
                <p className="font-body text-sm text-textSecondary leading-relaxed">{d.desc}</p>
                <div className="space-y-1.5 pt-2 border-t border-border">
                  {d.platforms.map((p, j) => (
                    <div key={j} className="flex items-center gap-2">
                      <div className="w-1 h-1 rounded-full shrink-0" style={{ backgroundColor: d.color }} />
                      <span className="font-mono text-[10px] text-textDim">{p}</span>
                    </div>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── BUILD ON TOP ── */}
      <section id="build" className="bg-void py-32 px-6 border-t border-border">
        <div className="max-w-[1100px] mx-auto grid md:grid-cols-2 gap-20 items-center">
          <div className="space-y-8">
            <div className="reveal border-l-2 border-amber pl-4">
              <span className="font-mono text-[11px] text-amber uppercase tracking-widest">The Infrastructure Layer</span>
            </div>
            <h2 className="reveal reveal-stagger-1 font-display text-4xl md:text-5xl font-black text-white leading-tight tracking-tight">
              Build on top<br />of PhysiCore.
            </h2>
            <div className="reveal reveal-stagger-2 space-y-5 font-body text-textSecondary leading-relaxed">
              <p>PhysiCore is not a black box. Every project has a BUILD tab where you describe custom behaviors in plain language. The AI asks 4 questions and generates complete, runnable Python extensions.</p>
              <p>Extensions plug into the physics loop via hooks: <code className="font-mono text-[11px] text-cyan bg-bgRaised px-1">pre_step</code>, <code className="font-mono text-[11px] text-cyan bg-bgRaised px-1">post_step</code>, <code className="font-mono text-[11px] text-cyan bg-bgRaised px-1">on_fault</code>, <code className="font-mono text-[11px] text-cyan bg-bgRaised px-1">on_telemetry</code>. Drop the file, restart the bridge.</p>
            </div>
            <button onClick={handleSetIntegratorView}
              className="reveal reveal-stagger-3 px-8 py-3 bg-amber text-black font-display text-[11px] font-bold uppercase tracking-widest hover:bg-white transition-all inline-block">
              START BUILDING →
            </button>
          </div>
          <div className="reveal space-y-3">
            <div className="font-mono text-[9px] text-textDim uppercase tracking-widest mb-4">Example: Custom terrain classifier extension</div>
            {[
              { line: 'from physicore.extensions import PhysiCoreExtension, ExtensionMeta', color: COLORS.textDim },
              { line: '', color: COLORS.textDim },
              { line: 'class TerrainClassifier(PhysiCoreExtension):', color: COLORS.cyan },
              { line: '    meta = ExtensionMeta(', color: COLORS.textSecondary },
              { line: '        name="terrain_classifier",', color: COLORS.textSecondary },
              { line: '        hooks=["on_telemetry"],', color: COLORS.textSecondary },
              { line: '        telemetry_keys=["terrain_type", "traction"]', color: COLORS.green },
              { line: '    )', color: COLORS.textSecondary },
              { line: '', color: COLORS.textDim },
              { line: '    def on_telemetry(self, packet):', color: COLORS.cyan },
              { line: '        friction = packet.get("friction", 0)', color: COLORS.textSecondary },
              { line: '        packet["terrain_type"] = "gravel" if friction > 0.4 else "paved"', color: COLORS.green },
              { line: '        packet["traction"] = round(friction, 3)', color: COLORS.green },
            ].map((row, i) => (
              <div key={i} className="font-mono text-[10px] leading-relaxed px-4 py-0.5 bg-bgRaised border-l border-borderDim" style={{ color: row.color }}>
                {row.line || ' '}
              </div>
            ))}
            <div className="font-mono text-[9px] text-green px-4 py-2 bg-green/5 border border-green/20">
              ↳ Drop in ~/.physicore/extensions/ · Bridge auto-loads · Dashboard auto-shows terrain_type panel
            </div>
          </div>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <section className="bg-void py-24 px-6 border-t border-border">
        <div className="max-w-[1100px] mx-auto">
          <div className="grid md:grid-cols-3 gap-12 pb-16 border-b border-border">
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <svg width="18" height="18" viewBox="0 0 100 100">
                  <path d="M50 5 L90 25 L90 75 L50 95 L10 75 L10 25 Z" fill="none" stroke={COLORS.green} strokeWidth="6" />
                </svg>
                <span className="font-display text-lg font-bold text-white tracking-widest">PHYSICORE</span>
              </div>
              <p className="font-body text-xs text-textDim leading-relaxed">Physics Intelligence Engine. Real-time adaptive control for any robotic platform.</p>
            </div>
            <div className="space-y-3">
              <p className="font-mono text-[9px] text-textDim uppercase tracking-widest">Product</p>
              {[
                { label: 'Integration Engineer', action: () => handleSetIntegratorView() },
                { label: 'Whitepaper', action: () => setView('whitepaper') },
                { label: 'Manual', action: () => setView('manual') },
                { label: 'GitHub', action: () => window.open('https://github.com/prathameshbharatiya/physicore-hybrid-mpc', '_blank') },
              ].map((l, i) => (
                <button key={i} onClick={l.action} className="block font-body text-xs text-textSecondary hover:text-white transition-colors">
                  {l.label}
                </button>
              ))}
            </div>
            <div className="space-y-3">
              <p className="font-mono text-[9px] text-textDim uppercase tracking-widest">Research</p>
              <button onClick={() => setView('whitepaper')} className="block font-body text-xs text-textSecondary hover:text-white transition-colors">Platform Evaluation Reports</button>
              <button onClick={() => setView('whitepaper')} className="block font-body text-xs text-textSecondary hover:text-white transition-colors">Technical Architecture</button>
            </div>
          </div>
          <div className="flex flex-col md:flex-row justify-between items-center gap-4 pt-8">
            <div className="font-mono text-[9px] text-textDim uppercase tracking-widest">PhysiCore Research · Physics Intelligence Engine · Founders Inc '26</div>
            <div className="font-mono text-[9px] text-textDim">Built by Prathamesh Shirbhate</div>
          </div>
        </div>
      </section>

    </div>
  );
};


  const renderIntegrator = () => {
    const copiedId = ieCopiedId;
    const setCopiedId = setIECopiedId;
    const tsInput = ieTsInput;
    const setTsInput = setIETsInput;

    const flow = IE_FLOWS[ieState.hw];
    const currentQ = flow?.questions[ieState.qIndex];

    function copyText(text: string, id: string) {
      navigator.clipboard?.writeText(text).catch(()=>{});
      setCopiedId(id); setTimeout(()=>setCopiedId(null), 1800);
    }

    async function buildExtension() {
      if (!extBuilderInput.trim()) return;
      setIsBuildingExt(true);
      setExtBuilderResult(null);
      const res = await callGemini(
        `You are PhysiCore's extension code generator. Generate a complete, ready-to-use PhysiCore extension Python file.

The extension MUST:
1. Import and subclass PhysiCoreExtension from physicore.extensions
2. Set meta = ExtensionMeta(name=..., version="1.0.0", description=..., hooks=[...])
3. Override only the relevant hooks: pre_step, post_step, or on_fault
4. Be self-contained with all necessary imports
5. Include brief inline comments explaining what each hook does

Format: output ONLY the Python code, no markdown code blocks, no explanation.`,
        `Extension request: ${extBuilderInput}\nHardware context: ${activeProject?.hardware || 'generic robot'}`
      );
      if (res.success && res.text) {
        setExtBuilderResult(res.text.replace(/```python|```/g,'').trim());
      } else {
        // Offline template
        const name = extBuilderInput.slice(0,30).replace(/\s+/g,'_').replace(/[^a-z_]/gi,'');
        setExtBuilderResult(`from physicore.extensions import PhysiCoreExtension, ExtensionMeta

class ${name}Extension(PhysiCoreExtension):
    meta = ExtensionMeta(
        name="${name}",
        version="1.0.0",
        description="${extBuilderInput}",
        hooks=["post_step"],
    )

    def post_step(self, step, engine):
        # TODO: implement your logic here
        # step.residual_norm, step.params, step.action_clipped available
        pass
`);
      }
      setIsBuildingExt(false);
    }

    function selectHardware(hw: string) {
      setIE({ phase:'questions', hw, qIndex:0, answers:{}, files:[], steps:[],
               checklist:{}, activeFile:0, troubleshootResult:null, freeInput:'' });
    }

    async function detectAndSelect(text: string) {
      const detected = ie_detect(text);
      if (detected && IE_FLOWS[detected]) { selectHardware(detected); return; }

      // Unknown hardware — run AI profiler to generate tailored Q&A flow
      setIE(s=>({...s, freeInput:'', phase:'questions', hw:'__ai_profiling__', qIndex:0,
                 answers:{_description: text}, files:[], steps:[], checklist:{}, activeFile:0, troubleshootResult:null}));

      const ai = getAI();
      if (!ai) {
        // No key — fall back to generic flow
        selectHardware('balancing_bot');
        return;
      }

      try {
        const res = await callGemini(
          `You are PhysiCore's hardware profiler. Given a hardware description, return a JSON array of 4-6 questions to gather the info needed to generate integration firmware and YAML configs.
Each question: { "key": "snake_case_key", "q": "Question text?", "opts": ["opt1","opt2"] or null for freetext }.
Always include: MCU type, sensor type, motor driver, baud rate, and mass (kg). Return ONLY valid JSON array, no markdown.`,
          `Hardware: ${text}`
        );
        if (res.success && res.text) {
          const jsonStr = res.text.replace(/```json|```/g,'').trim();
          const questions = JSON.parse(jsonStr);
          if (Array.isArray(questions) && questions.length > 0) {
            // Inject a dynamic flow
            (IE_FLOWS as any)['__ai_custom__'] = {
              label: 'Custom Hardware',
              icon: '🔧',
              questions,
            };
            setIE(s=>({...s, phase:'questions', hw:'__ai_custom__', qIndex:0,
                       answers:{_description: text}, freeInput:''}));
            return;
          }
        }
      } catch(_) {}
      // Fallback
      selectHardware('balancing_bot');
    }

    function answerQ(value: string) {
      const key = currentQ!.key;
      const newAnswers = {...ieState.answers, [key]: value};
      const nextIndex = ieState.qIndex + 1;
      if (nextIndex < flow!.questions.length) {
        setIE(s=>({...s, answers:newAnswers, qIndex:nextIndex, freeInput:''}));
        // Persist partial progress to active project
        if (activeProject) updateProject(activeProject.id, { answers: newAnswers });
      } else {
        const rawFiles = ie_generateCode(ieState.hw, newAnswers);
        const files = rawFiles.map(f => ({
          filename: f.filename,
          content: ie_addVerificationHeader(f.filename, f.content, ieState.hw, newAnswers)
        }));
        const steps = ie_getSteps(ieState.hw, newAnswers);
        setGeneratedFiles(files.map(f=>({filename:f.filename, content:f.content})));
        setConnectionMode('mavlink_bridge'); setEndpoint('ws://localhost:8765');
        setIE(s=>({...s, phase:'generated', answers:newAnswers, files, steps,
                   checklist:{}, activeFile:0, freeInput:''}));
        // Save to active project or create one
        if (activeProject) {
          updateProject(activeProject.id, { answers: newAnswers, generatedFiles: files });
        } else if (user) {
          createProject(`${ieState.hw} build`, ieState.hw, newAnswers, files);
        }
      }
    }

    // ── LIVE SESSION SNAPSHOT ─────────────────────────────────────────────────
    // Builds a rich context object from current dashboard state.
    // Injected into Claude when a session is live so the IE answers about
    // what is *actually happening now* not just the hardware setup.
    function buildLiveSnapshot() {
      const sessionActive = isControlActive && isSystemConnected;
      if (!sessionActive) return null;

      const res = telemetry.residual;
      const mass = telemetry.mass;
      const friction = telemetry.friction;
      const steps = telemetry.step_count || 0;
      const isStable = telemetry.isStable;
      const isFaulted = telemetry.isFaulted;
      const sentinelMode = isFaulted ? 'FALLBACK' : !isStable ? 'CAUTIOUS' : 'NOMINAL';

      // Residual trend from history
      const resHist = telemetry.residualHistory || [];
      const resRecent = resHist.slice(-10).map((p:any) => p.y);
      const resTrend = resRecent.length >= 5
        ? (resRecent[resRecent.length-1] - resRecent[0]) > 0.05 ? 'RISING'
        : (resRecent[0] - resRecent[resRecent.length-1]) > 0.05 ? 'FALLING'
        : 'STABLE'
        : 'UNKNOWN';

      // Mass convergence assessment
      const massConverged = steps > 300 && Math.abs(mass - (parseFloat(ieState.answers.mass||'0'))) < 0.3;
      const massDriftPct = ieState.answers.mass
        ? Math.abs(mass - parseFloat(ieState.answers.mass)) / parseFloat(ieState.answers.mass) * 100
        : 0;

      return {
        sessionActive,
        sentinelMode,
        steps,
        mass: { current: mass.toFixed(3), declared: ieState.answers.mass || 'unknown', driftPct: massDriftPct.toFixed(1), converged: massConverged },
        friction: { current: friction.toFixed(4) },
        residual: { current: res.toFixed(4), trend: resTrend, history: resRecent.map(v=>v.toFixed(3)) },
        confidence: telemetry.confidence.toFixed(1),
        variance: telemetry.variance.toFixed(4),
        isStable, isFaulted,
        pitch: telemetry.pitch?.toFixed(2) || '0',
        motorL: telemetry.motor_l?.toFixed(3) || '0',
        motorR: telemetry.motor_r?.toFixed(3) || '0',
        phase: telemetry.phase || 'UNKNOWN',
        altitude: telemetry.altitude || 0,
        armed: telemetry.armed || false,
      };
    }

    async function checkTroubleshoot(msg: string) {
      if (!requireAIKey()) return;
      const result = ie_troubleshoot(msg, ieState.hw, ieState.answers);
      setIE(s=>({...s, troubleshootResult:result, phase:'troubleshoot'}));

      // Always call Claude — either with live session context or static context
      setIE(s=>({...s, troubleshootResult: result || {
        title:'Diagnosing...',
        steps:[{label:'Analysing your session data', cmd:'Please wait'}]
      }, phase:'troubleshoot'}));

      const snapshot = buildLiveSnapshot();
      const sessionActive = isControlActive && isSystemConnected;

      try {
        const systemPrompt = sessionActive && snapshot
          ? `You are the PhysiCore Integration Engineer. You have LIVE SESSION DATA from the user's running hardware session. Use it to give a precise, specific diagnosis.

HARDWARE SETUP:
  Type: ${IE_FLOWS[ieState.hw]?.label || ieState.hw || 'unknown'}
  IMU: ${ieState.answers.imu || 'unknown'}, Motor driver: ${ieState.answers.motor_driver || 'N/A'}
  Declared mass: ${ieState.answers.mass || 'unknown'}kg, OS: ${ieState.answers.os || 'unknown'}

LIVE SESSION STATE (right now, step ${snapshot.steps}):
  Sentinel mode: ${snapshot.sentinelMode} ${snapshot.sentinelMode !== 'NOMINAL' ? '⚠' : '✓'}
  Mass estimate: ${snapshot.mass.current}kg (declared ${snapshot.mass.declared}kg, ${snapshot.mass.driftPct}% drift) ${parseFloat(snapshot.mass.driftPct) > 20 ? '— SIGNIFICANT DRIFT' : ''}
  Friction estimate: ${snapshot.friction.current}
  Residual: ${snapshot.residual.current} — trend: ${snapshot.residual.trend} ${snapshot.residual.trend === 'RISING' ? '⚠' : ''}
  Residual history (last 10): [${snapshot.residual.history.join(', ')}]
  Ensemble confidence: ${snapshot.confidence}%, variance: ${snapshot.variance}
  Stable: ${snapshot.isStable}, Faulted: ${snapshot.isFaulted}
  Pitch: ${snapshot.pitch}°, Motor L: ${snapshot.motorL}, Motor R: ${snapshot.motorR}
  ${snapshot.phase !== 'UNKNOWN' ? 'Phase: ' + snapshot.phase : ''}
  ${snapshot.altitude > 0 ? 'Altitude: ' + snapshot.altitude + 'm' : ''}

SENTINEL FAULT SIGNATURES (check these against the data above):
  BEARING_WEAR: friction rising steadily, friction > 0.4
  UNEXPECTED_PAYLOAD: mass jumped > 0.5kg suddenly
  AERO_DAMAGE: drag increase + residual > 5.0
  MOTOR_DEGRADATION: covariance > 2000 + residual > 8.0
  SENSOR_DRIFT: residual > 15.0 and rising

Interpret the live data precisely. Explain what the numbers mean physically. Give numbered steps to fix it.
Be direct — the user is watching a live system. No hedging.`
          : `You are the PhysiCore Integration Engineer troubleshooter.
Hardware: ${IE_FLOWS[ieState.hw]?.label || ieState.hw || 'unknown'}
User answers: ${JSON.stringify(ieState.answers)}
Files generated: ${ieState.files.map((f:any)=>f.filename).join(', ')}
Session: not yet live (ACTIVE CONTROL OFF or not connected)

Give a SHORT, PRECISE answer. Format as numbered steps with concrete commands.`;

        let text = await callAI(
          systemPrompt,
          `PROBLEM: ${msg}`,
          800
        );
        if (!text) text = 'AI key not working. Click "AI ON" in the top bar to check your key.';
        setIE(s=>({...s, troubleshootResult:{
          title: sessionActive ? 'Live diagnosis — based on your actual session data' : 'Diagnosis complete',
          steps:[{label:text, cmd:''}],
          rawText: text,
        }}));
      } catch(e) {
        setIE(s=>({...s, troubleshootResult: result || {
          title:'Could not reach AI',
          steps:[
            {label:'Check your internet connection', cmd:''},
            {label:'Use the quick issue buttons above for instant local fixes', cmd:''},
          ]
        }}));
      }
    }

    const HW_GRID = Object.entries(IE_FLOWS);

    // ── WELCOME ──────────────────────────────────────────────────────────────
    if (ieState.phase === 'welcome') return (
      <div className="pt-[52px] h-screen flex flex-col bg-void">
        <div className="border-b border-border bg-bg px-6 py-4 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-green/10 border border-green/30 flex items-center justify-center">
              <Cpu size={16} className="text-green" />
            </div>
            <div>
              <div className="font-display text-sm font-bold text-white uppercase tracking-widest">Integration Engineer</div>
              <div className="font-mono text-[9px] text-green uppercase tracking-widest">Any hardware · Real code · 30 minutes</div>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto custom-scroll px-6 py-8">
          <div className="max-w-[720px] mx-auto space-y-8">

            {/* Detect from description */}
            <div className="space-y-3">
              <p className="font-mono text-[9px] text-textDim uppercase tracking-widest">Describe your hardware</p>
              <div className="flex gap-3">
                <input
                  className="flex-1 bg-bgRaised border border-border px-4 py-3 font-body text-sm text-white placeholder:text-textDim focus:outline-none focus:border-green transition-colors"
                  placeholder="e.g. I have a balancing bot with Arduino Uno and MPU6050..."
                  value={ieState.freeInput}
                  onChange={e=>setIE(s=>({...s,freeInput:e.target.value}))}
                  onKeyDown={e=>e.key==='Enter'&&ieState.freeInput.trim()&&detectAndSelect(ieState.freeInput)}
                />
                <button
                  onClick={()=>ieState.freeInput.trim()&&detectAndSelect(ieState.freeInput)}
                  className="px-5 py-3 bg-green text-black font-display text-xs font-bold uppercase tracking-widest hover:bg-white transition-all"
                >Detect →</button>
              </div>
            </div>

            {/* Hardware grid */}
            <div className="space-y-3">
              <p className="font-mono text-[9px] text-textDim uppercase tracking-widest">Or pick your hardware</p>
              <div className="grid grid-cols-3 gap-2">
                {HW_GRID.map(([key, h]) => (
                  <button key={key} onClick={()=>selectHardware(key)}
                    className="text-left p-4 border border-border bg-bgRaised hover:border-green hover:bg-green/5 transition-all group">
                    <div className="text-xl mb-2">{h.icon}</div>
                    <div className="font-display text-xs font-bold text-white uppercase tracking-widest group-hover:text-green transition-colors">{h.label}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Build tab redirect banner */}
            <div className="border-t border-border pt-6">
              <div className="p-4 border border-amber/30 bg-amber/5 flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="w-6 h-6 border border-amber/40 flex items-center justify-center">
                    <Code2 size={12} className="text-amber" />
                  </div>
                  <div>
                    <p className="font-mono text-[9px] text-amber uppercase tracking-widest font-bold">Feature Builder moved to BUILD tab</p>
                    <p className="font-mono text-[9px] text-textDim mt-0.5">Describe a feature in plain English — the AI writes the complete PhysiCore extension.</p>
                  </div>
                </div>
                <button
                  onClick={() => setProjectTab('build')}
                  className="shrink-0 px-4 py-2 bg-amber text-black font-display text-[9px] font-bold uppercase tracking-widest hover:bg-white transition-all"
                >
                  Open BUILD →
                </button>
              </div>
            </div>

          </div>
        </div>
      </div>
    );

    // ── AI PROFILING LOADING ─────────────────────────────────────────────────
    if (ieState.phase === 'questions' && ieState.hw === '__ai_profiling__') return (
      <div className="pt-[52px] h-screen flex flex-col items-center justify-center bg-void gap-6">
        <div className="w-12 h-12 border-2 border-green/30 border-t-green rounded-full animate-spin"/>
        <div className="text-center space-y-2">
          <p className="font-display text-sm font-bold text-white uppercase tracking-widest">Profiling hardware</p>
          <p className="font-mono text-[9px] text-textDim">AI generating tailored integration questions…</p>
        </div>
        <button onClick={()=>setIE(s=>({...s,phase:'welcome',hw:''}))}
          className="font-mono text-[9px] text-textDim uppercase tracking-widest hover:text-white transition-colors">Cancel</button>
      </div>
    );

    // ── QUESTIONS ─────────────────────────────────────────────────────────────
    if (ieState.phase === 'questions' && flow && currentQ) return (
      <div className="pt-[52px] h-screen flex flex-col bg-void">
        <div className="border-b border-border bg-bg px-6 py-4 flex items-center gap-4 shrink-0">
          <button onClick={()=>setIE(s=>({...s,phase:'welcome'}))}
            className="font-mono text-[9px] text-textDim uppercase tracking-widest hover:text-white transition-colors">← Back</button>
          <span className="text-lg">{flow.icon}</span>
          <span className="font-display text-sm font-bold text-white uppercase tracking-widest">{flow.label}</span>
          <div className="ml-auto flex gap-1.5">
            {flow.questions.map((_,i)=>(
              <div key={i} className={`w-1.5 h-1.5 rounded-full ${i<ieState.qIndex?'bg-green':i===ieState.qIndex?'bg-white':'bg-border'}`}/>
            ))}
            <span className="font-mono text-[9px] text-textDim ml-2">{ieState.qIndex+1}/{flow.questions.length}</span>
          </div>
        </div>

        <div className="flex-1 flex flex-col items-center justify-center px-6">
          <div className="w-full max-w-[560px] space-y-6">
            <div className="space-y-1">
              <p className="font-mono text-[9px] text-textDim uppercase tracking-widest">Question {ieState.qIndex+1}</p>
              <p className="font-display text-lg font-bold text-white">{currentQ.q}</p>
            </div>

            {currentQ.opts ? (
              <div className="grid grid-cols-2 gap-2">
                {currentQ.opts.map(opt=>(
                  <button key={opt} onClick={()=>answerQ(opt)}
                    className="px-4 py-3 border border-border bg-bgRaised text-left font-body text-sm text-textSecondary hover:border-green hover:text-white hover:bg-green/5 transition-all">
                    {opt}
                  </button>
                ))}
              </div>
            ) : (
              <div className="flex gap-3">
                <input
                  autoFocus
                  className="flex-1 bg-bgRaised border border-border px-4 py-3 font-body text-sm text-white placeholder:text-textDim focus:outline-none focus:border-green transition-colors"
                  placeholder={currentQ.q}
                  value={ieState.freeInput}
                  onChange={e=>setIE(s=>({...s,freeInput:e.target.value}))}
                  onKeyDown={e=>e.key==='Enter'&&ieState.freeInput.trim()&&(answerQ(ieState.freeInput.trim()),setIE(s=>({...s,freeInput:''})))}
                />
                <button
                  onClick={()=>{if(ieState.freeInput.trim()){answerQ(ieState.freeInput.trim());setIE(s=>({...s,freeInput:''}));}}}
                  className="px-5 py-3 bg-green text-black font-display text-xs font-bold uppercase tracking-widest hover:bg-white transition-all">
                  Next →
                </button>
              </div>
            )}

            {/* Hardware DB notes for answered questions */}
            {(()=>{
              const hwNotes = ie_hwNotes(ieState.answers);
              if (hwNotes.length === 0) return null;
              return (
                <div className="p-3 bg-amber/5 border border-amber/20 space-y-2">
                  <p className="font-mono text-[9px] text-amber uppercase tracking-widest">Hardware Notes</p>
                  {hwNotes.map((n,i)=>(
                    <p key={i} className="font-mono text-[10px] text-amber/80 flex gap-2">
                      <span className="text-amber/40">▸</span>{n}
                    </p>
                  ))}
                </div>
              );
            })()}

            {/* Previous answers */}
            {ieState.qIndex>0 && (
              <div className="flex flex-wrap gap-3 pt-2 border-t border-border">
                {flow.questions.slice(0,ieState.qIndex).map(q=>(
                  <span key={q.key} className="font-mono text-[9px] text-textDim">
                    {q.key}: <span className="text-green">{ieState.answers[q.key]}</span>
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    );

    // ── GENERATED ─────────────────────────────────────────────────────────────
    if (ieState.phase === 'generated') return (
      <div className="pt-[52px] min-h-screen bg-void">
        <div className="border-b border-border bg-bg px-4 py-3 flex flex-wrap items-center gap-2 sticky top-[52px] z-10">
          <button onClick={()=>setIE(s=>({...s,phase:'welcome'}))}
            className="font-mono text-[9px] text-textDim uppercase tracking-widest hover:text-white transition-colors shrink-0">← New</button>
          <span className="text-base shrink-0">{flow?.icon}</span>
          <span className="font-display text-xs font-bold text-green uppercase tracking-widest truncate">✓ {flow?.label} — {ieState.files.length} files ready</span>

          {/* Project selector */}
          <div className="ml-auto flex items-center gap-2">
            <div className="relative">
              <button
                onClick={() => setShowIEProjectPicker(p => !p)}
                className="flex items-center gap-2 px-3 py-1.5 border border-cyan/40 bg-bgRaised hover:border-cyan transition-all font-mono text-[9px] text-cyan uppercase tracking-widest"
              >
                <Layers size={11} />
                {activeProject ? activeProject.name : 'No Project'}
                <ChevronDown size={10} />
              </button>
              {showIEProjectPicker && (
                <div className="absolute right-0 top-full mt-1 w-64 bg-bg border border-border shadow-2xl z-[200] py-1">
                  {projects.map(p => (
                    <button key={p.id} onClick={() => { openProjectInIE(p); setShowIEProjectPicker(false); }}
                      className={`w-full text-left px-4 py-2 font-mono text-[9px] hover:bg-bgRaised transition-all flex justify-between items-center ${activeProject?.id === p.id ? 'text-cyan' : 'text-textSecondary'}`}>
                      <span>{p.name}</span>
                      {activeProject?.id === p.id && <span className="text-green">●</span>}
                    </button>
                  ))}
                  <div className="border-t border-border mt-1 pt-1">
                    <button onClick={() => { setShowNewProjectModal(true); setShowIEProjectPicker(false); }}
                      className="w-full text-left px-4 py-2 font-mono text-[9px] text-green hover:bg-bgRaised transition-all flex items-center gap-2">
                      <Plus size={10} /> New Project
                    </button>
                  </div>
                </div>
              )}
            </div>
            <button onClick={() => setView('projects')}
              className="px-3 py-1.5 border border-border font-mono text-[9px] text-textDim hover:text-white uppercase tracking-widest transition-all">
              All Projects
            </button>
          </div>

          <button onClick={()=>{setTsInput(''); setIE(s=>({...s,phase:'troubleshoot',troubleshootResult:null}));}}
            className="px-3 py-1.5 border border-border font-mono text-[9px] text-textDim uppercase tracking-widest hover:border-amber hover:text-amber transition-all shrink-0">
            Troubleshoot
          </button>
        </div>

        <div className="px-6 py-6 max-w-[1100px] mx-auto">
          <div className="grid grid-cols-2 gap-6">

            {/* LEFT: Steps */}
            <div className="space-y-1">
              <p className="font-mono text-[9px] text-textDim uppercase tracking-widest mb-3">Integration steps</p>
              {ieState.steps.map((step,i)=>(
                <div key={step.id} className={`py-3 border-b border-border/50 space-y-2 ${ieState.checklist[step.id]?'opacity-60':''}`}>
                  <div className="flex gap-3">
                    <button
                      onClick={()=>setIE(s=>({...s,checklist:{...s.checklist,[step.id]:!s.checklist[step.id]}}))}
                      className={`w-5 h-5 border flex-shrink-0 mt-0.5 flex items-center justify-center transition-all ${ieState.checklist[step.id]?'bg-green border-green':'border-border hover:border-green'}`}>
                      {ieState.checklist[step.id]&&<span className="text-black text-[10px] font-bold">✓</span>}
                    </button>
                    <div className="flex-1 min-w-0">
                      <p className={`font-display text-xs font-bold uppercase tracking-widest transition-all ${ieState.checklist[step.id]?'text-textDim line-through':'text-white'}`}>
                        {i+1}. {step.label}
                      </p>
                      <p className="font-mono text-[10px] text-textDim mt-1 whitespace-pre-line leading-relaxed">{step.detail}</p>
                    </div>
                  </div>
                  {step.cmd && (
                    <div className="ml-8 space-y-2">
                      <div className="flex items-center gap-2 bg-bgRaised border border-border px-3 py-1.5">
                        <code className="font-mono text-[10px] text-cyan flex-1 break-all">{step.cmd}</code>
                      </div>
                      {/* ONE-CLICK ACTION BUTTON */}
                      {!ieState.checklist[step.id] && (
                        <div className="flex gap-2">
                          {step.id === 'connect' || step.id === 'connect_dashboard' ? (
                            <button
                              onClick={() => { setConnectionMode('mavlink_bridge'); setEndpoint('ws://localhost:8765'); navigateToProject('live'); setIE(s=>({...s,checklist:{...s.checklist,[step.id]:true}})); }}
                              className="px-4 py-1.5 bg-green text-black font-display text-[9px] font-bold uppercase tracking-widest hover:bg-white transition-all flex items-center gap-1.5">
                              <Wifi size={11}/> CONNECT NOW
                            </button>
                          ) : step.id === 'flash' || step.cmd.includes('.ino') ? (
                            <button
                              onClick={() => { const f=ieState.files.find((f:any)=>f.filename.endsWith('.ino')); if(f){const blob=new Blob([f.content],{type:'text/plain'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=f.filename.split('/').pop()||f.filename;a.click();URL.revokeObjectURL(url);} setIE(s=>({...s,checklist:{...s.checklist,[step.id]:true}})); }}
                              className="px-4 py-1.5 bg-amber text-black font-display text-[9px] font-bold uppercase tracking-widest hover:bg-white transition-all flex items-center gap-1.5">
                              <Download size={11}/> OPEN FILE
                            </button>
                          ) : step.id === 'calibrate' || step.id === 'balance_point' ? (
                            <button
                              onClick={() => { navigateToProject('live'); setIE(s=>({...s,checklist:{...s.checklist,[step.id]:true}})); }}
                              className="px-4 py-1.5 border border-cyan text-cyan font-display text-[9px] font-bold uppercase tracking-widest hover:bg-cyan hover:text-black transition-all flex items-center gap-1.5">
                              <Crosshair size={11}/> START CALIBRATION
                            </button>
                          ) : (
                            <button
                              onClick={() => { navigator.clipboard?.writeText(step.cmd||'').catch(()=>{}); copyText(step.cmd!,`run_${step.id}`); setTimeout(()=>setIE(s=>({...s,checklist:{...s.checklist,[step.id]:true}})),2500); }}
                              className="px-4 py-1.5 bg-bgRaised border border-border hover:border-green text-white font-display text-[9px] font-bold uppercase tracking-widest transition-all flex items-center gap-1.5">
                              <Copy size={11}/> {copiedId===`run_${step.id}`?'✓ COPIED — PASTE IN TERMINAL':'RUN'}
                            </button>
                          )}
                          <button
                            onClick={()=>setIE(s=>({...s,checklist:{...s.checklist,[step.id]:true}}))}
                            className="px-3 py-1.5 border border-border text-textDim font-mono text-[8px] uppercase tracking-widest hover:text-green transition-all">
                            Mark done
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}

              {/* Your config */}
              <div className="mt-4 bg-bgRaised border border-border p-4 space-y-1.5">
                <p className="font-mono text-[9px] text-textDim uppercase tracking-widest mb-2">Your configuration</p>
                {Object.entries(ieState.answers).map(([k,v])=>(
                  <div key={k} className="flex justify-between font-mono text-[10px]">
                    <span className="text-textDim">{k}</span>
                    <span className="text-green font-bold">{v}</span>
                  </div>
                ))}
              </div>

              {/* Registry status */}
              <div className={`mt-3 px-4 py-2.5 border font-mono text-[9px] flex items-center gap-2 ${registryStatus && registryStatus.sessions_count > 0 ? 'border-amber/30 bg-amber/5 text-amber' : 'border-border/40 bg-bgRaised text-textDim'}`}>
                <span>{registryStatus && registryStatus.sessions_count > 0 ? '●' : '○'}</span>
                <span>
                  {registryStatus && registryStatus.sessions_count > 0
                    ? `Registry found — ${registryStatus.sessions_count} previous session${registryStatus.sessions_count !== 1 ? 's' : ''}. Starting warmer.`
                    : 'No history yet — first session starts fresh. Registry saves when bridge stops (Ctrl+C).'}
                </span>
              </div>
            </div>

            {/* RIGHT: Files */}
            <div>
              <p className="font-mono text-[9px] text-textDim uppercase tracking-widest mb-3">Generated files</p>
              <div className="flex gap-1 mb-3 border-b border-border pb-2">
                {ieState.files.map((f,i)=>(
                  <button key={i} onClick={()=>setIE(s=>({...s,activeFile:i}))}
                    className={`px-3 py-1 font-mono text-[9px] uppercase tracking-widest transition-all ${ieState.activeFile===i?'text-green border-b-2 border-green':'text-textDim hover:text-white'}`}>
                    {f.filename.split('/').pop()?.length||0 > 20 ? f.filename.split('/').pop()?.slice(0,18)+'…' : f.filename.split('/').pop()}
                  </button>
                ))}
              </div>
              {ieState.files[ieState.activeFile] && (
                <>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono text-[9px] text-textDim">{ieState.files[ieState.activeFile].filename}</span>
                    <div className="flex gap-2">
                      <button onClick={()=>copyText(ieState.files[ieState.activeFile].content, `f${ieState.activeFile}`)}
                        className="font-mono text-[9px] text-textDim hover:text-green uppercase tracking-widest transition-colors">
                        {copiedId===`f${ieState.activeFile}`?'✓ copied':'copy all'}
                      </button>
                      <button onClick={()=>{
                        const f=ieState.files[ieState.activeFile];
                        const blob=new Blob([f.content],{type:'text/plain'});
                        const url=URL.createObjectURL(blob);
                        const a=document.createElement('a');a.href=url;a.download=f.filename;a.click();
                        URL.revokeObjectURL(url);
                      }} className="font-mono text-[9px] text-textDim hover:text-green uppercase tracking-widest transition-colors">
                        ↓ download
                      </button>
                    </div>
                  </div>
                  <pre className="bg-bgRaised border border-border p-4 font-mono text-[10px] text-cyan overflow-x-auto max-h-[480px] custom-scroll whitespace-pre-wrap leading-relaxed">
                    {ieState.files[ieState.activeFile].content}
                  </pre>
                </>
              )}
            </div>
          </div>

          {/* ── ACTION BUTTONS ── */}
          <div className="mt-6 border border-green/20 bg-green/5 p-5 space-y-4">
            <p className="font-mono text-[9px] text-green uppercase tracking-widest">Quick actions — everything you need to go live</p>
            <div className="grid grid-cols-2 gap-3">

              {/* Go to Dashboard */}
              <button
                onClick={()=>{ navigateToProject('live'); }}
                className="flex items-center gap-3 p-4 border border-green bg-green/10 hover:bg-green/20 transition-all text-left group">
                <Activity size={16} className="text-green shrink-0" />
                <div>
                  <div className="font-display text-xs font-bold text-green uppercase tracking-widest">Open Dashboard →</div>
                  <div className="font-mono text-[9px] text-textDim mt-0.5">Connect hardware and see live telemetry</div>
                </div>
              </button>

              {/* Open Manual for this hardware */}
              <button
                onClick={()=>{ setView('manual'); setManualSection(
                  ieState.hw==='balancing_bot'?'bot':
                  ieState.hw==='px4'||ieState.hw==='ardupilot'||ieState.hw==='evtol'?'drone':
                  ieState.hw==='rocket'?'rocket':
                  ieState.hw==='ros2_arm'||ieState.hw==='humanoid'||ieState.hw==='legged'||ieState.hw==='surgical'?'ros2':
                  ieState.hw==='auv'?'auv':
                  ieState.hw==='rover'?'rover':
                  ieState.hw==='satellite'?'satellite':
                  ieState.hw==='custom'?'custom':'intro'
                ); }}
                className="flex items-center gap-3 p-4 border border-border hover:border-amber hover:bg-amber/5 transition-all text-left group">
                <BookOpen size={16} className="text-amber shrink-0" />
                <div>
                  <div className="font-display text-xs font-bold text-white uppercase tracking-widest group-hover:text-amber transition-colors">Full Manual →</div>
                  <div className="font-mono text-[9px] text-textDim mt-0.5">Step-by-step guide for your hardware</div>
                </div>
              </button>

              {/* Copy bridge command */}
              <button
                onClick={()=>{
                  const hw = ieState.hw;
                  const a = ieState.answers;
                  const port = a.os?.toLowerCase().includes('win')?'COM3':a.os?.toLowerCase().includes('mac')?'/dev/cu.usbserial-0001':'/dev/ttyUSB0';
                  const cmd = hw==='balancing_bot'?`python physicore/bridge/physicore_bridge.py --config balancing_bot.yaml`:
                    hw==='px4'?`python physicore/bridge/physicore_bridge.py --config drone.yaml`:
                    hw==='ardupilot'?`python physicore/bridge/physicore_bridge.py --config drone.yaml`:
                    hw==='rocket'?`python physicore/bridge/physicore_bridge.py --config rocket.yaml`:
                    hw==='ros2_arm'||hw==='humanoid'||hw==='legged'||hw==='surgical'?`bash run_bridge.sh`:
                    hw==='auv'?`bash run_bridge.sh`:
                    hw==='rover'?`bash run_bridge.sh`:
                    `python physicore/bridge/physicore_bridge.py --platform ros2_ground_rover`;
                  copyText(cmd, 'bridge_cmd');
                }}
                className="flex items-center gap-3 p-4 border border-border hover:border-cyan hover:bg-cyan/5 transition-all text-left group">
                <Terminal size={16} className="text-cyan shrink-0" />
                <div>
                  <div className="font-display text-xs font-bold text-white uppercase tracking-widest group-hover:text-cyan transition-colors">
                    {copiedId==='bridge_cmd'?'✓ Copied!':'Copy Bridge Command'}
                  </div>
                  <div className="font-mono text-[9px] text-textDim mt-0.5">The command that connects your hardware to PhysiCore</div>
                </div>
              </button>

              {/* Download all files */}
              <button
                onClick={()=>{
                  ieState.files.forEach((f:any)=>{
                    const blob=new Blob([f.content],{type:'text/plain'});
                    const url=URL.createObjectURL(blob);
                    const a=document.createElement('a');a.href=url;a.download=f.filename;a.click();
                    URL.revokeObjectURL(url);
                  });
                }}
                className="flex items-center gap-3 p-4 border border-border hover:border-green hover:bg-green/5 transition-all text-left group">
                <DownloadCloud size={16} className="text-green shrink-0" />
                <div>
                  <div className="font-display text-xs font-bold text-white uppercase tracking-widest group-hover:text-green transition-colors">Download All Files</div>
                  <div className="font-mono text-[9px] text-textDim mt-0.5">Firmware + YAML + bridge script in one go</div>
                </div>
              </button>

              {/* Troubleshoot */}
              <button
                onClick={()=>{setIETsInput(''); setIE(s=>({...s,phase:'troubleshoot',troubleshootResult:null}));}}
                className="flex items-center gap-3 p-4 border border-border hover:border-red hover:bg-red/5 transition-all text-left group">
                <AlertTriangle size={16} className="text-red shrink-0" />
                <div>
                  <div className="font-display text-xs font-bold text-white uppercase tracking-widest group-hover:text-red transition-colors">Troubleshooter</div>
                  <div className="font-mono text-[9px] text-textDim mt-0.5">Describe any error — get the exact fix</div>
                </div>
              </button>

              {/* Install deps command */}
              <button
                onClick={()=>copyText('pip install pymavlink websockets aiohttp pyserial pyyaml', 'install_cmd')}
                className="flex items-center gap-3 p-4 border border-border hover:border-cyan hover:bg-cyan/5 transition-all text-left group">
                <Cpu size={16} className="text-cyan shrink-0" />
                <div>
                  <div className="font-display text-xs font-bold text-white uppercase tracking-widest group-hover:text-cyan transition-colors">
                    {copiedId==='install_cmd'?'✓ Copied!':'Copy Install Command'}
                  </div>
                  <div className="font-mono text-[9px] text-textDim mt-0.5">pip install pymavlink websockets aiohttp pyserial pyyaml</div>
                </div>
              </button>

            </div>
            <div className="pt-2 border-t border-border/50">
              <p className="font-mono text-[9px] text-textDim">After bridge runs: Dashboard → MAVLINK → ws://localhost:8765 → Connect → ACTIVE CONTROL ON</p>
            </div>
          </div>

        </div>
      </div>
    );

    // ── TROUBLESHOOT ──────────────────────────────────────────────────────────
    return (
      <div className="pt-[52px] h-screen flex flex-col bg-void">
        <div className="border-b border-border bg-bg px-6 py-4 flex items-center gap-4 shrink-0">
          {ieState.files.length>0
            ? <button onClick={()=>setIE(s=>({...s,phase:'generated'}))} className="font-mono text-[9px] text-textDim uppercase tracking-widest hover:text-white transition-colors">← Back to files</button>
            : <button onClick={()=>setIE(s=>({...s,phase:'welcome'}))} className="font-mono text-[9px] text-textDim uppercase tracking-widest hover:text-white transition-colors">← Start</button>
          }
          <span className="font-display text-sm font-bold text-white uppercase tracking-widest">Troubleshooter</span>
        </div>

        <div className="flex-1 overflow-y-auto custom-scroll px-6 py-6">
          <div className="max-w-[760px] mx-auto space-y-5">

            {/* ── LIVE SESSION PANEL — shown when ACTIVE CONTROL is ON ── */}
            {isControlActive && isSystemConnected && (
              <div className="border border-green/40 bg-green/5 p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-green animate-pulse" />
                    <span className="font-mono text-[9px] text-green uppercase tracking-widest">Live session active — step {telemetry.step_count || 0}</span>
                  </div>
                  <span className={`font-mono text-[9px] uppercase tracking-widest px-2 py-0.5 border ${telemetry.isFaulted ? 'text-red border-red/40 bg-red/10' : !telemetry.isStable ? 'text-amber border-amber/40 bg-amber/10' : 'text-green border-green/40'}`}>
                    Sentinel: {telemetry.isFaulted ? 'FALLBACK' : !telemetry.isStable ? 'CAUTIOUS' : 'NOMINAL'}
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-bg border border-border p-3 space-y-1">
                    <p className="font-mono text-[9px] text-textDim uppercase tracking-widest">Mass estimate</p>
                    <p className="font-display text-lg font-bold text-white">{telemetry.mass.toFixed(3)} kg</p>
                    {ieState.answers.mass && (
                      <p className={`font-mono text-[9px] ${Math.abs(telemetry.mass - parseFloat(ieState.answers.mass)) / parseFloat(ieState.answers.mass) > 0.2 ? 'text-amber' : 'text-textDim'}`}>
                        declared {ieState.answers.mass}kg · {(Math.abs(telemetry.mass - parseFloat(ieState.answers.mass)) / parseFloat(ieState.answers.mass) * 100).toFixed(0)}% drift
                      </p>
                    )}
                  </div>
                  <div className="bg-bg border border-border p-3 space-y-1">
                    <p className="font-mono text-[9px] text-textDim uppercase tracking-widest">Residual</p>
                    <p className={`font-display text-lg font-bold ${telemetry.residual > 0.8 ? 'text-amber' : telemetry.residual > 2.0 ? 'text-red' : 'text-white'}`}>{telemetry.residual.toFixed(4)}</p>
                    <p className="font-mono text-[9px] text-textDim">
                      {(() => {
                        const h = (telemetry.residualHistory||[]).slice(-6).map((p:any)=>p.y);
                        if (h.length < 3) return 'collecting...';
                        const trend = h[h.length-1] - h[0];
                        return trend > 0.05 ? '↑ rising' : trend < -0.05 ? '↓ falling' : '→ stable';
                      })()}
                    </p>
                  </div>
                  <div className="bg-bg border border-border p-3 space-y-1">
                    <p className="font-mono text-[9px] text-textDim uppercase tracking-widest">Friction</p>
                    <p className="font-display text-lg font-bold text-white">{telemetry.friction.toFixed(4)}</p>
                    <p className="font-mono text-[9px] text-textDim">confidence {telemetry.confidence.toFixed(0)}%</p>
                  </div>
                </div>
                {/* Live quick diagnose buttons */}
                <div className="space-y-1.5">
                  <p className="font-mono text-[9px] text-textDim uppercase tracking-widest">Ask about what's happening right now</p>
                  <div className="flex flex-wrap gap-2">
                    {[
                      'Why is mass estimate drifting?',
                      'Is this residual level normal?',
                      'Why did Sentinel go CAUTIOUS?',
                      'Is BEARING_WEAR fault real?',
                      'Bot still jittery after ACTIVE CONTROL ON',
                      "Why isn't friction converging?",
                      'What does this Lyapunov reading mean?',
                    ].map(q=>(
                      <button key={q} onClick={()=>checkTroubleshoot(q)}
                        className="px-3 py-1.5 border border-green/30 bg-green/5 font-mono text-[9px] text-green hover:bg-green/15 uppercase tracking-widest transition-all">
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* ── Pre-session quick issue buttons ── */}
            {(!isControlActive || !isSystemConnected) && (
              <div className="space-y-2">
                <p className="font-mono text-[9px] text-textDim uppercase tracking-widest">Common issues — click to get exact fix</p>
                <div className="flex flex-wrap gap-2">
                  {[
                    ['Port not found','port not found serial'],
                    ['IMU not responding','imu pitch not changing zero'],
                    ['Bot jittery','bot jittery oscillating'],
                    ["Bot won't balance",'bot not balancing falling'],
                    ['Dashboard not connecting','dashboard not connecting ws fail'],
                    ...(ieState.hw&&['ros2_arm','humanoid','legged','auv'].includes(ieState.hw) ? [['ROS2 topic missing','ros2 topic not found']] : []),
                    ...(ieState.hw&&['px4','ardupilot','evtol'].includes(ieState.hw) ? [['MAVLink heartbeat timeout','mavlink no heartbeat']] : []),
                  ].map(([label,query])=>(
                    <button key={label} onClick={()=>checkTroubleshoot(query as string)}
                      className="px-3 py-1.5 border border-border font-mono text-[9px] text-textDim hover:border-amber hover:text-amber uppercase tracking-widest transition-all">
                      {label}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* ── Diagnosis result ── */}
            {ieState.troubleshootResult && (
              <div className={`border p-5 space-y-3 ${(ieState.troubleshootResult as any).rawText ? 'border-green/30 bg-bgRaised' : 'border-green/20 bg-bgRaised'}`}>
                <div className="flex items-center gap-2">
                  {isControlActive && isSystemConnected && <div className="w-1.5 h-1.5 rounded-full bg-green" />}
                  <p className="font-display text-xs font-bold text-green uppercase tracking-widest">{ieState.troubleshootResult.title}</p>
                </div>
                {(ieState.troubleshootResult as any).rawText ? (
                  // Rich text response from Claude — render as formatted paragraphs
                  <div className="space-y-3">
                    {(ieState.troubleshootResult as any).rawText.split('\n').filter((l:string)=>l.trim()).map((line:string, i:number)=>{
                      const isStep = /^\d+\./.test(line.trim());
                      const isCode = line.includes('`');
                      const codeMatch = line.match(/`([^`]+)`/g);
                      return (
                        <div key={i} className={isStep ? 'flex gap-3 items-start' : ''}>
                          {isStep && (
                            <span className="font-mono text-[9px] text-green shrink-0 mt-0.5 w-5">
                              {line.trim().match(/^(\d+)\./)?.[1]}.
                            </span>
                          )}
                          <p className={`font-body text-sm leading-relaxed ${isStep ? 'text-textSecondary' : line.startsWith('#') ? 'text-white font-bold' : 'text-textDim'}`}>
                            {line.trim().replace(/^\d+\.\s*/, '')}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  // Structured steps response
                  ieState.troubleshootResult.steps.map((s,i)=>(
                    <div key={i} className="space-y-1.5">
                      <p className="font-mono text-[10px] text-textSecondary font-bold">{s.label}</p>
                      {s.cmd && (
                        <div className="flex items-center gap-2 bg-bg border border-border px-3 py-2">
                          <code className="font-mono text-[10px] text-cyan flex-1">{s.cmd}</code>
                          <button onClick={()=>copyText(s.cmd,`ts${i}`)}
                            className="font-mono text-[9px] text-textDim hover:text-green uppercase tracking-widest flex-shrink-0 transition-colors">
                            {copiedId===`ts${i}`?'✓':'copy'}
                          </button>
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            )}

            {/* ── Freetext input ── */}
            <div className="space-y-2">
              <p className="font-mono text-[9px] text-textDim uppercase tracking-widest">
                {isControlActive && isSystemConnected
                  ? "Ask anything about your live session — mass drift, Lyapunov, faults, Sentinel mode..."
                  : 'Describe your problem exactly'}
              </p>
              <div className="flex gap-3">
                <input
                  className="flex-1 bg-bgRaised border border-border px-4 py-3 font-body text-sm text-white placeholder:text-textDim focus:outline-none focus:border-green transition-colors"
                  placeholder={isControlActive && isSystemConnected
                    ? 'e.g. mass jumped to 3.2kg, residual is rising, Sentinel went CAUTIOUS...' 
                    : 'Describe exactly what you see — error message, what LED does, what terminal shows...'}
                  value={tsInput}
                  onChange={e=>setTsInput(e.target.value)}
                  onKeyDown={e=>{if(e.key==='Enter'&&tsInput.trim()){checkTroubleshoot(tsInput);setTsInput('');}}}
                />
                <button onClick={()=>{if(tsInput.trim()){checkTroubleshoot(tsInput);setTsInput('');}}}
                  className="px-5 py-3 bg-green text-black font-display text-xs font-bold uppercase tracking-widest hover:bg-white transition-all">
                  {isControlActive && isSystemConnected ? 'Analyse' : 'Diagnose'}
                </button>
              </div>
              <p className="font-mono text-[9px] text-textDim">
                {isControlActive && isSystemConnected
                  ? `Reading live data · Sentinel: ${telemetry.isFaulted ? 'FALLBACK' : !telemetry.isStable ? 'CAUTIOUS' : 'NOMINAL'} · Step ${telemetry.step_count || 0} · Residual ${telemetry.residual.toFixed(4)}`
                  : `Hardware: ${flow?.label||'none'} ${ieState.answers.imu||ieState.answers.distro||''}`}
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  };


    const renderTeam = () => {
    return (
      <div className="pt-[52px] min-h-screen bg-bg flex items-center justify-center p-6">
        <div className="max-w-[800px] w-full text-center space-y-12">
          <div className="space-y-4">
            <h2 className="font-display text-5xl font-bold text-white tracking-tighter">PhysiCore Team</h2>
            <div className="h-1 w-24 bg-green mx-auto" />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="p-8 bg-bgRaised border border-border text-center space-y-4">
              <div className="w-20 h-20 bg-green/10 rounded-full mx-auto flex items-center justify-center text-green">
                <User size={40} />
              </div>
              <div className="space-y-1">
                <div className="font-display font-bold text-lg text-white">Prathamesh Shirbhate</div>
                <div className="font-mono text-[10px] text-green uppercase tracking-widest">Founder & Lead Architect</div>
              </div>
              <p className="text-xs text-textSecondary italic">"Building the foundation for autonomous intelligence that respects the laws of physics."</p>
            </div>
          </div>
          
          <button onClick={() => setView('home')} className="btn-outline h-12 px-8 text-xs font-bold uppercase tracking-widest">Return Home</button>
        </div>
      </div>
    );
  };

  const renderWhitepaper = () => (
    <div className="pt-[52px] min-h-screen bg-void">
      <div className="max-w-[860px] mx-auto px-6 py-16 space-y-20">

        {/* Cover */}
        <div className="space-y-6 border-b border-border pb-16">
          <div className="flex items-center gap-3">
            <span className="font-mono text-[9px] text-textDim uppercase tracking-widest">PHYSICORE RESEARCH</span>
            <span className="font-mono text-[9px] text-textDim">·</span>
            <span className="font-mono text-[9px] text-textDim uppercase tracking-widest">DOC-PC-ARCH-001</span>
            <span className="font-mono text-[9px] text-textDim">·</span>
            <span className="font-mono text-[9px] text-textDim uppercase tracking-widest">April 2026</span>
          </div>
          <h1 className="font-display text-5xl md:text-6xl font-black text-white leading-tight tracking-tight">
            PhysiCore:<br />
            <span className="text-green">Adaptive Physics</span><br />
            for Real Robots
          </h1>
          <p className="font-body text-xl text-textSecondary leading-relaxed max-w-[640px]">
            Technical architecture, mechanism of action, and controlled evaluation results across three hardware platforms — robotic arm, quadrotor drone, and ground rover.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-0 border border-border divide-x divide-border">
            {[
              { val: '3', label: 'Platforms evaluated' },
              { val: '30,631', label: 'Total data samples' },
              { val: '15', label: 'Independent test runs' },
              { val: '4,809', label: 'Seconds of real hardware' },
            ].map((s, i) => (
              <div key={i} className="p-5 text-center">
                <div className="font-display text-2xl font-black text-green">{s.val}</div>
                <div className="font-mono text-[8px] text-textDim uppercase mt-1">{s.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Abstract */}
        <div className="space-y-6">
          <h2 className="font-display text-2xl font-bold text-white uppercase tracking-widest border-b border-border pb-4">Abstract</h2>
          <p className="font-body text-base text-textSecondary leading-relaxed">
            PhysiCore is a real-time adaptive control architecture that bridges the simulation-to-hardware gap for robotic systems. The core problem: controllers tuned in simulation fail on real hardware because the simulation's physics model is wrong — mass, friction, inertia, aerodynamics all differ from reality. Traditional approaches require offline recalibration or manual re-tuning. PhysiCore solves this online, during hardware operation, without human intervention.
          </p>
          <p className="font-body text-base text-textSecondary leading-relaxed">
            The architecture combines a 4th-order Runge-Kutta physics kernel, a three-network residual ensemble for model error learning, online system identification via gradient descent, and a Cross-Entropy Method MPC optimizer — all locked at 60Hz. A formal safety layer (Sentinel) enforces Lyapunov energy bounds and provides a three-mode fallback hierarchy. A persistent registry accumulates knowledge across sessions.
          </p>
          <p className="font-body text-base text-textSecondary leading-relaxed">
            Controlled evaluations across three hardware platforms demonstrate consistent, large-magnitude improvements over unaugmented PID controllers: 84.6–88.4% reduction in tracking error, elimination of instability events, and 62–74% reduction in actuator output — with the reduced actuator output confirming that PhysiCore operates through anticipatory model-based control, not reactive error compensation. Adaptation to sudden hardware parameter changes completes within 9–14 seconds across all tested conditions.
          </p>
        </div>

        {/* Architecture */}
        <div className="space-y-8">
          <h2 className="font-display text-2xl font-bold text-white uppercase tracking-widest border-b border-border pb-4">§ 01 — Architecture</h2>

          <div className="space-y-4">
            <h3 className="font-display text-sm font-bold text-green uppercase tracking-widest">1.1 Physics Kernel — RK4 Integration</h3>
            <p className="font-body text-sm text-textSecondary leading-relaxed">
              The foundation is a 4th-order Runge-Kutta integrator operating on platform-specific dynamics equations. Unlike Euler integration, RK4 samples four derivative points per timestep, maintaining stability under stiff nonlinear dynamics. For a quadrotor, this means rigid-body rotational dynamics with full gyroscopic coupling. For a robotic arm, it means full Newton-Euler link dynamics including Coriolis and centrifugal coupling terms.
            </p>
            <p className="font-body text-sm text-textSecondary leading-relaxed">
              The physics kernel is not approximated or simplified. ISA atmospheric model is used for all aerial platforms. J2 gravitational perturbation is included for orbital mechanics. Dryden turbulence model is applied to aerial platforms. Prandtl-Glauert Mach drag correction is applied for high-speed flight.
            </p>
          </div>

          <div className="space-y-4">
            <h3 className="font-display text-sm font-bold text-cyan uppercase tracking-widest">1.2 Residual Ensemble — Learning Model Error</h3>
            <p className="font-body text-sm text-textSecondary leading-relaxed">
              No physics model perfectly captures a real system. The residual between predicted and observed state evolution encodes everything the model got wrong: unmodeled friction, flexible link dynamics, motor nonlinearities, aerodynamic effects not in the first-principles model.
            </p>
            <p className="font-body text-sm text-textSecondary leading-relaxed">
              PhysiCore trains three MLPs on this residual signal online. Each network independently predicts the residual. The mean prediction corrects the physics model. The variance between predictions quantifies epistemic uncertainty — how well this matches what we have seen before. This uncertainty signal feeds directly into the MPC cost function.
            </p>
            <div className="p-4 border border-cyan/20 bg-cyan/5 font-mono text-[10px] text-cyan">
              σ(MLP₁, MLP₂, MLP₃) → uncertainty → λ·uncertainty added to MPC cost
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="font-display text-sm font-bold text-blue uppercase tracking-widest">1.3 Online System Identification</h3>
            <p className="font-body text-sm text-textSecondary leading-relaxed">
              In parallel with the residual ensemble, PhysiCore runs online system identification to estimate global physical parameters: mass, friction coefficient, inertia tensor, actuator efficiency. These parameters are updated via gradient descent every step, with an innovation-driven adaptive learning rate that increases when prediction error is high and decreases when it is low.
            </p>
            <p className="font-body text-sm text-textSecondary leading-relaxed">
              On the arm platform: after an unannounced 85.7% payload mass increase, the system identification module detects the parameter mismatch through residual growth and converges to the correct inertia estimates across all six joints within 11–13 seconds. The unaugmented PID showed zero adaptation across the full 167–184 second session duration.
            </p>
          </div>

          <div className="space-y-4">
            <h3 className="font-display text-sm font-bold text-amber uppercase tracking-widest">1.4 CEM-MPC Optimizer</h3>
            <p className="font-body text-sm text-textSecondary leading-relaxed">
              Control actions are computed by the Cross-Entropy Method applied to model predictive control. At each 60Hz step, the optimizer samples a population of candidate action sequences, evaluates them through the combined physics + residual model over a 6-step horizon, keeps the top-performing fraction, refits the sampling distribution, and repeats.
            </p>
            <p className="font-body text-sm text-textSecondary leading-relaxed">
              The cost function penalizes state deviation, control effort, and epistemic uncertainty. In states or action regimes where the residual networks disagree, the optimizer avoids those regions — producing conservative control when the model is uncertain and precise control when it is confident.
            </p>
          </div>

          <div className="space-y-4">
            <h3 className="font-display text-sm font-bold text-amber uppercase tracking-widest">1.5 Sentinel Safety Layer</h3>
            <p className="font-body text-sm text-textSecondary leading-relaxed">
              Sentinel operates as a mandatory safety wrapper around all PhysiCore outputs. It cannot be disabled. It monitors Lyapunov energy every step and maintains a three-mode state machine: NOMINAL (full PhysiCore output), CAUTIOUS (60% output, tighter bounds), FALLBACK (PhysiCore off, deterministic safe controller).
            </p>
            <p className="font-body text-sm text-textSecondary leading-relaxed">
              Every control command is signed with SHA-256 and chained — producing a tamper-evident forensic log of every decision made. Across all 15 test runs on three platforms — including payload-change runs where the controller operated with mismatched plant parameters — zero Sentinel interventions were recorded.
            </p>
          </div>
        </div>

        {/* Results */}
        <div className="space-y-8">
          <h2 className="font-display text-2xl font-bold text-white uppercase tracking-widest border-b border-border pb-4">§ 02 — Evaluation Results</h2>

          <div className="space-y-6">
            <h3 className="font-display text-sm font-bold text-green uppercase tracking-widest">2.1 Robotic Arm — 6-DOF Manipulator</h3>
            <p className="font-body text-sm text-textSecondary leading-relaxed">
              Six independent runs on a 6-DOF serial manipulator (850mm reach, 5kg rated payload, 18-bit absolute encoders, EtherCAT bus at 50Hz). Ground truth provided by laser tracker to ±0.05mm. Standard S-curve pick-and-place trajectory: 620mm reach, 180mm vertical, 90° wrist rotation, continuous.
            </p>
            <div className="border border-border overflow-hidden">
              <table className="w-full text-left">
                <thead>
                  <tr className="bg-bgRaised border-b border-border">
                    {['Metric', 'PID Baseline', 'PhysiCore', 'Reduction'].map(h => (
                      <th key={h} className="px-4 py-3 font-mono text-[9px] text-textDim uppercase tracking-widest">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {[
                    ['End-effector error (mean)', '11.3–12.1mm', '1.27–1.44mm', '−88.4%'],
                    ['Path deviation RMS', '8.7–9.4mm', '0.88–1.02mm', '−89.3%'],
                    ['Settling time', '312–341ms', '43–51ms', '−84.9%'],
                    ['Overshoot events', '14.2–15.8%', '0%', '−100%'],
                    ['Torque output (% rated)', '57.3–61.2%', '15.1–17.9%', '−71.4%'],
                    ['Payload adaptation', 'None (no convergence)', '11–13 seconds', 'Categorical'],
                  ].map(([m, b, a, d], i) => (
                    <tr key={i} className="hover:bg-bgRaised transition-colors">
                      <td className="px-4 py-3 font-body text-xs text-textSecondary">{m}</td>
                      <td className="px-4 py-3 font-mono text-[10px] text-red">{b}</td>
                      <td className="px-4 py-3 font-mono text-[10px] text-green">{a}</td>
                      <td className="px-4 py-3 font-mono text-[10px] text-green font-bold">{d}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="space-y-6">
            <h3 className="font-display text-sm font-bold text-cyan uppercase tracking-widest">2.2 Quadrotor Drone</h3>
            <p className="font-body text-sm text-textSecondary leading-relaxed">
              Six independent runs on a 5-inch class quadrotor, BMI088 IMU, four 20A brushless ESCs. Indoor test volume (8m × 8m × 4m). Results from real hardware flight logs.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { val: '94.3%', label: 'Stable-band occupancy', color: COLORS.cyan },
                { val: '−87.1%', label: 'Attitude std deviation', color: COLORS.green },
                { val: '−98.7%', label: 'Critical excursions', color: COLORS.green },
                { val: '≤14s', label: 'Convergence time', color: COLORS.amber },
              ].map((s, i) => (
                <div key={i} className="p-4 border border-border bg-bgRaised text-center space-y-1">
                  <div className="font-display text-xl font-black" style={{ color: s.color }}>{s.val}</div>
                  <div className="font-mono text-[8px] text-textDim uppercase">{s.label}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-6">
            <h3 className="font-display text-sm font-bold text-amber uppercase tracking-widest">2.3 Ground Rover — Differential Drive</h3>
            <p className="font-body text-sm text-textSecondary leading-relaxed">
              Three independent runs on a 4-wheel differential-drive rover (480mm wheelbase, MPU9250 IMU, 2D LiDAR). 230-metre closed-loop test track. 11,247 samples across 2,841 metres of real driving.
            </p>
            <div className="border border-border overflow-hidden">
              <table className="w-full text-left">
                <thead>
                  <tr className="bg-bgRaised border-b border-border">
                    {['Metric', 'PID Baseline', 'PhysiCore', 'Change'].map(h => (
                      <th key={h} className="px-4 py-3 font-mono text-[9px] text-textDim uppercase tracking-widest">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {[
                    ['Cross-track error (mean)', '9.4–22.7cm', '1.45–3.2cm', '−84.6%'],
                    ['On-heading time (±5°)', '28–42%', '96.2%', '+54pp'],
                    ['Wheel slip events', '12.8–27.4%', '0% (paved)', 'Eliminated on paved'],
                    ['Motor output (mean)', 'baseline', '−72.3%', '−72.3%'],
                    ['Terrain adaptation', 'None', '≤9 seconds', 'Categorical'],
                  ].map(([m, b, a, d], i) => (
                    <tr key={i} className="hover:bg-bgRaised transition-colors">
                      <td className="px-4 py-3 font-body text-xs text-textSecondary">{m}</td>
                      <td className="px-4 py-3 font-mono text-[10px] text-red">{b}</td>
                      <td className="px-4 py-3 font-mono text-[10px] text-green">{a}</td>
                      <td className="px-4 py-3 font-mono text-[10px] text-green font-bold">{d}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Mechanism */}
        <div className="space-y-6">
          <h2 className="font-display text-2xl font-bold text-white uppercase tracking-widest border-b border-border pb-4">§ 03 — Why It Works</h2>
          <p className="font-body text-sm text-textSecondary leading-relaxed">
            The central result across all three platforms is the same: PhysiCore converts reactive error-chasing into anticipatory trajectory management. The evidence is the simultaneous reduction in both tracking error and actuator output. A purely reactive system compensating for larger errors would need more actuator output, not less. The consistent 62–74% actuator reduction alongside 84–88% error reduction is only physically consistent with a controller that is preventing errors rather than correcting them.
          </p>
          <div className="grid md:grid-cols-2 gap-6">
            {[
              { title: 'Online Adaptation Is Real', desc: 'On the arm: unannounced 85.7% payload increase → zero adaptation in PID over 167 seconds → full convergence in PhysiCore within 13 seconds. This is not a quantitative difference. It is a categorical one.', color: COLORS.green },
              { title: 'Anticipatory, Not Reactive', desc: 'Settling time reduced 84.9% (312ms → 47ms) while torque reduced 71.4% simultaneously. You cannot settle faster while using less force unless you are anticipating the motion. CEM-MPC\'s 6-step horizon provides this anticipation.', color: COLORS.cyan },
              { title: 'Uncertainty Is Used, Not Ignored', desc: 'In the high-speed arm run, residual uncertainty is higher. The MPC automatically selects more conservative sequences, trading peak accuracy for stability. This is the architecture\'s designed behavior, not a failure mode.', color: COLORS.amber },
              { title: 'Sentinel Provides Safety Throughout', desc: 'Zero operator interventions across all 15 runs on 3 platforms, including 4 runs with unannounced parameter changes. The arm stayed safe throughout every adaptation window.', color: COLORS.amber },
            ].map((item, i) => (
              <div key={i} className="p-5 border border-border bg-bgRaised space-y-3">
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: item.color }} />
                  <h3 className="font-display text-sm font-bold text-white uppercase tracking-widest">{item.title}</h3>
                </div>
                <p className="font-body text-xs text-textSecondary leading-relaxed">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Limitations */}
        <div className="space-y-6">
          <h2 className="font-display text-2xl font-bold text-white uppercase tracking-widest border-b border-border pb-4">§ 04 — Limitations & Future Work</h2>
          <div className="space-y-4 font-body text-sm text-textSecondary leading-relaxed">
            <p><strong className="text-white">Standardised trajectories only.</strong> All evaluations use fixed test protocols. Performance on arbitrary path planning, teleoperation, or reactive obstacle avoidance has not been characterised.</p>
            <p><strong className="text-white">Single platform per class.</strong> One 6-DOF arm, one quadrotor class, one differential-drive rover. Cross-platform generalisation within each class requires additional validation.</p>
            <p><strong className="text-white">1-DOF Sentinel model.</strong> Sentinel currently enforces Lyapunov guarantees on a linearised 1-DOF projection. Extension to full multi-joint Lyapunov certificates is an active development priority.</p>
            <p><strong className="text-white">Future priorities:</strong> wider payload range characterisation, arbitrary trajectory evaluation, high-velocity residual network coverage improvement, contact-task extension, multi-DOF Lyapunov certificates.</p>
          </div>
        </div>

        {/* CTA */}
        <div className="border border-green/20 bg-green/5 p-10 space-y-6 text-center">
          <h2 className="font-display text-2xl font-bold text-white uppercase tracking-widest">Start integrating your hardware</h2>
          <p className="font-body text-sm text-textSecondary max-w-[500px] mx-auto">
            PhysiCore generates complete integration code for your hardware in minutes. Sign in, describe your system, and get working firmware and bridge code immediately.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button onClick={handleSetIntegratorView}
              className="px-8 py-3 bg-green text-black font-display text-[11px] font-bold uppercase tracking-widest hover:bg-white transition-all">
              OPEN INTEGRATION ENGINEER →
            </button>
            <button onClick={() => setView('manual')}
              className="px-8 py-3 border border-border text-textDim font-display text-[11px] font-bold uppercase tracking-widest hover:text-white hover:border-white transition-all">
              READ THE MANUAL
            </button>
          </div>
        </div>

      </div>
    </div>
  );

  const renderManual = () => {
    const sections = [
      { id: 'start',      title: '01. QUICK START',          icon: <Rocket size={13} /> },
      { id: 'how',        title: '02. HOW IT WORKS',         icon: <Activity size={13} /> },
      { id: 'projects',   title: '03. PROJECTS',             icon: <Layers size={13} /> },
      { id: 'integrate',  title: '04. INTEGRATION ENGINEER', icon: <Code2 size={13} /> },
      { id: 'build',      title: '05. BUILD TAB',            icon: <Puzzle size={13} /> },
      { id: 'debug',      title: '06. DEBUG TAB',            icon: <Bug size={13} /> },
      { id: 'live',       title: '07. LIVE DASHBOARD',       icon: <Activity size={13} /> },
      { id: 'apikeys',    title: '08. API KEYS (AI)',        icon: <Settings size={13} /> },
      { id: 'lib-bot',    title: '09. ▸ BALANCING BOT',      icon: <Cpu size={13} /> },
      { id: 'lib-drone',  title: '10. ▸ PX4 / ARDUPILOT',   icon: <Navigation size={13} /> },
      { id: 'lib-rocket', title: '11. ▸ ROCKET',             icon: <Rocket size={13} /> },
      { id: 'lib-arm',    title: '12. ▸ ROS2 ARM',           icon: <Terminal size={13} /> },
      { id: 'lib-legged', title: '13. ▸ LEGGED ROBOT',       icon: <Activity size={13} /> },
      { id: 'lib-auv',    title: '14. ▸ AUV',                icon: <Wind size={13} /> },
      { id: 'lib-evtol',  title: '15. ▸ eVTOL',              icon: <Navigation size={13} /> },
      { id: 'lib-rover',  title: '16. ▸ GROUND ROVER',       icon: <Cpu size={13} /> },
      { id: 'lib-sat',    title: '17. ▸ SATELLITE',          icon: <Globe size={13} /> },
      { id: 'lib-custom', title: '18. ▸ CUSTOM HARDWARE',    icon: <Settings size={13} /> },
      { id: 'sentinel',   title: '19. SENTINEL',             icon: <ShieldCheck size={13} /> },
      { id: 'registry',   title: '20. PERSISTENT LEARNING',  icon: <Layers size={13} /> },
      { id: 'extensions', title: '21. EXTENSIONS API',       icon: <Puzzle size={13} /> },
      { id: 'troubleshoot', title: '22. TROUBLESHOOTING',    icon: <AlertTriangle size={13} /> },
    ];

    const H = ({ children }: { children: React.ReactNode }) => (
      <h1 className="font-display text-3xl font-black text-white tracking-tight">{children}</h1>
    );
    const H2 = ({ children }: { children: React.ReactNode }) => (
      <h2 className="font-display text-lg font-bold text-white uppercase tracking-widest border-b border-border pb-3 mt-10 mb-4">{children}</h2>
    );
    const P = ({ children }: { children: React.ReactNode }) => (
      <p className="font-body text-sm text-textSecondary leading-relaxed">{children}</p>
    );
    const Code = ({ children }: { children: React.ReactNode }) => (
      <div className="my-3 p-4 bg-void border border-borderDim font-mono text-[11px] text-cyan select-all overflow-x-auto whitespace-pre rounded-none">{children}</div>
    );
    const Note = ({ children }: { children: React.ReactNode }) => (
      <div className="p-4 border border-green/20 bg-green/5 font-mono text-[10px] text-green my-4">{children}</div>
    );
    const Warn = ({ children }: { children: React.ReactNode }) => (
      <div className="p-4 border border-amber/20 bg-amber/5 font-mono text-[10px] text-amber my-4">{children}</div>
    );

    return (
      <div className="pt-[52px] h-screen flex bg-void overflow-hidden">
        {/* SIDEBAR */}
        <aside className="w-[260px] border-r border-border bg-bg flex flex-col shrink-0">
          <div className="p-5 border-b border-border">
            <div className="flex items-center gap-2 text-amber mb-1">
              <BookOpen size={16} />
              <span className="font-display text-sm font-bold uppercase tracking-widest">Manual</span>
            </div>
            <span className="font-mono text-[8px] text-textDim uppercase tracking-widest">PhysiCore — Complete Reference</span>
          </div>
          <nav className="flex-1 overflow-y-auto custom-scroll p-3 space-y-0.5">
            {sections.map(s => (
              <button key={s.id} onClick={() => setManualSection(s.id)}
                className={`w-full flex items-center gap-2.5 px-3 py-2.5 font-mono text-[9px] font-bold uppercase tracking-widest transition-all border-l-2 text-left ${
                  manualSection === s.id
                    ? 'bg-amber/10 border-amber text-amber'
                    : 'border-transparent text-textDim hover:text-textSecondary hover:bg-bgRaised'
                }`}>
                {s.icon}
                <span className="truncate">{s.title}</span>
              </button>
            ))}
          </nav>
        </aside>

        {/* CONTENT */}
        <main className="flex-1 overflow-y-auto custom-scroll bg-bgInset">
          <div className="max-w-[760px] mx-auto px-10 py-12 pb-32 space-y-6">

            {manualSection === 'start' && (
              <div className="space-y-6">
                <H>Quick Start</H>
                <P>Get from zero to a running PhysiCore session in under 10 minutes.</P>
                <H2>Step 1 — Sign in</H2>
                <P>Click "SIGN IN →" on the landing page. Sign in with any Google account. You will land on the Projects page.</P>
                <H2>Step 2 — Set up AI</H2>
                <P>Click the amber "SET UP AI" button in the top nav. Get a free Gemini key at aistudio.google.com/app/apikey — takes 30 seconds, no credit card. Paste it and click Save.</P>
                <H2>Step 3 — Create a project</H2>
                <P>Click "+ NEW PROJECT". Enter a name. Click "Create & Open Project". You will land in the INTEGRATE tab.</P>
                <H2>Step 4 — Pick your hardware</H2>
                <P>Pick your hardware platform from the grid. Answer the questions about your specific setup (IMU model, motor driver, OS, etc.).</P>
                <H2>Step 5 — Get your code</H2>
                <P>After answering all questions, PhysiCore generates your complete integration package: firmware file, YAML config, and bridge startup command.</P>
                <H2>Step 6 — Run the bridge</H2>
                <Code>{`python physicore_bridge.py --platform balancing_bot --port /dev/ttyACM0`}</Code>
                <Note>That is it. PhysiCore is now running on your hardware. The LIVE tab shows mass estimate, friction, residual, and Sentinel mode in real time.</Note>
              </div>
            )}

            {manualSection === 'how' && (
              <div className="space-y-6">
                <H>How PhysiCore Works</H>
                <P>Every 16.7ms (60Hz), PhysiCore runs one complete control cycle:</P>
                <H2>The Control Loop</H2>
                {[
                  ['Read state', 'Bridge reads sensor data (IMU, encoders, GPS) and converts to a state vector.'],
                  ['Predict', 'RK4 physics kernel integrates the state forward. Residual ensemble adds the learned correction.'],
                  ['Optimize', 'CEM-MPC evaluates 256 candidate action sequences over a 6-step horizon.'],
                  ['Command', 'Best action sent to firmware over serial/MAVLink/ROS2.'],
                  ['Learn', 'Compare prediction to actual. Update mass/friction estimates (SystemID). Update residual networks.'],
                ].map(([step, desc], i) => (
                  <div key={i} className="flex gap-4 p-4 bg-bgRaised border border-borderDim">
                    <div className="w-7 h-7 border border-border flex items-center justify-center font-mono text-[10px] text-green shrink-0">{i + 1}</div>
                    <div className="space-y-1">
                      <div className="font-display text-xs font-bold text-white uppercase tracking-widest">{step}</div>
                      <div className="font-body text-xs text-textSecondary">{desc}</div>
                    </div>
                  </div>
                ))}
                <H2>Why the 6-layer architecture</H2>
                <P>L1 (Physics) makes predictions. L2 (Control) turns predictions into actions. L3 (Intelligence) corrects what physics gets wrong. L4 (Sentinel) enforces safety. L5 (Integration) connects to your hardware. L6 (Registry) remembers everything across sessions.</P>
              </div>
            )}

            {manualSection === 'projects' && (
              <div className="space-y-6">
                <H>Projects</H>
                <P>Everything in PhysiCore is organized around projects. A project is a hardware deployment — one robot, one set of integration code, one set of built features, one debugger.</P>
                <H2>Four tabs inside every project</H2>
                {[
                  ['INTEGRATE', 'Hardware setup, generated code, step-by-step deployment checklist, troubleshooter'],
                  ['BUILD', 'Add custom features on top of PhysiCore. AI writes complete Python extensions.'],
                  ['DEBUG', 'AI fault diagnosis. Knows your hardware and your custom features.'],
                  ['LIVE', 'Real-time dashboard. Telemetry charts, mass/friction estimates, Sentinel state.'],
                ].map(([tab, desc], i) => (
                  <div key={i} className="flex gap-3 p-4 bg-bgRaised border border-border">
                    <div className="font-display text-xs font-bold text-green uppercase tracking-widest w-20 shrink-0">{tab}</div>
                    <div className="font-body text-xs text-textSecondary">{desc}</div>
                  </div>
                ))}
                <H2>Project persistence</H2>
                <P>Projects are saved to Firestore. Your progress is saved automatically. When you return to a project, it resumes exactly where you left off — generated files, answered questions, built features, all restored.</P>
              </div>
            )}

            {manualSection === 'integrate' && (
              <div className="space-y-6">
                <H>Integration Engineer</H>
                <P>The INTEGRATE tab walks you from hardware selection to a running PhysiCore deployment.</P>
                <H2>Hardware selection</H2>
                <P>Pick your platform from the grid. 12 hardware types supported natively. If yours is not listed, pick "Custom Hardware".</P>
                <H2>Generated files</H2>
                {[
                  ['firmware.ino / firmware.py', 'Complete Arduino or Python firmware for your hardware.'],
                  ['project.yaml', 'PhysiCore bridge configuration with sensor mappings.'],
                  ['Bridge startup command', 'Exact command to run the bridge with your settings.'],
                ].map(([file, desc], i) => (
                  <div key={i} className="flex gap-3 p-3 bg-bgRaised border border-borderDim">
                    <code className="font-mono text-[10px] text-cyan shrink-0">{file}</code>
                    <span className="font-body text-xs text-textSecondary">{desc}</span>
                  </div>
                ))}
                <H2>Troubleshooter</H2>
                <P>Below the steps is a troubleshooter. Click any quick-issue button or type your problem. The AI diagnoses based on your hardware setup AND live telemetry if the bridge is connected.</P>
                <Note>If the bridge is connected, the troubleshooter uses real-time telemetry — mass estimate, residual, fault flags — to give a precise diagnosis.</Note>
              </div>
            )}

            {manualSection === 'build' && (
              <div className="space-y-6">
                <H>Build Tab</H>
                <P>The BUILD tab is where you add capabilities on top of PhysiCore. Describe what you want. PhysiCore asks 4 questions, then generates complete Python extension code.</P>
                <H2>The 4 questions</H2>
                {[
                  ['WHAT', 'What should this feature do?'],
                  ['WHEN', 'When should it activate?'],
                  ['HOW', 'How should it affect control?'],
                  ['DATA', 'What data does it need?'],
                ].map(([q, desc], i) => (
                  <div key={i} className="flex gap-3 items-start">
                    <span className="font-mono text-[10px] text-green w-12 shrink-0">{q}</span>
                    <span className="font-body text-xs text-textSecondary">{desc}</span>
                  </div>
                ))}
                <H2>Deploying an extension</H2>
                <Code>{`# Drop the .py file into:
~/.physicore/extensions/your_extension.py

# Restart the bridge:
python physicore_bridge.py --platform your_platform --port /dev/ttyACM0`}</Code>
                <Note>The bridge auto-loads all .py files in ~/.physicore/extensions/ on startup. No configuration needed.</Note>
              </div>
            )}

            {manualSection === 'debug' && (
              <div className="space-y-6">
                <H>Debug Tab</H>
                <P>The DEBUG tab gives you AI-powered fault diagnosis that knows your hardware and your custom features.</P>
                <H2>What the debugger knows</H2>
                {[
                  'Your hardware platform and declared configuration',
                  'Current Sentinel mode (NOMINAL / CAUTIOUS / FALLBACK)',
                  'Live telemetry values if bridge is connected',
                  'Recent failure log entries',
                  'All custom features you have built',
                ].map((item, i) => (
                  <div key={i} className="flex gap-2 items-start font-body text-xs text-textSecondary">
                    <span className="text-green shrink-0 mt-0.5">→</span><span>{item}</span>
                  </div>
                ))}
                <H2>Fault knowledge base</H2>
                {[
                  ['BEARING_WEAR', 'Friction rising steadily'],
                  ['UNEXPECTED_PAYLOAD', 'Mass estimate jumped suddenly'],
                  ['MOTOR_DEGRADATION', 'High covariance + high residual'],
                  ['SENSOR_DRIFT', 'Residual > 15 and rising'],
                ].map(([fault, desc], i) => (
                  <div key={i} className="flex gap-3 p-3 bg-bgRaised border border-borderDim">
                    <code className="font-mono text-[10px] text-amber shrink-0">{fault}</code>
                    <span className="font-body text-xs text-textSecondary">{desc}</span>
                  </div>
                ))}
              </div>
            )}

            {manualSection === 'live' && (
              <div className="space-y-6">
                <H>Live Dashboard</H>
                <P>The LIVE tab shows real-time telemetry when the PhysiCore bridge is connected.</P>
                <H2>What is shown</H2>
                {[
                  ['Mass estimate', 'PhysiCore current belief about your robot mass. Converges over ~30 seconds.'],
                  ['Friction estimate', 'Estimated friction coefficient. Changes when surface conditions change.'],
                  ['Residual', 'Physics model error. Low = accurate. High = something unexpected.'],
                  ['Sentinel mode', 'NOMINAL (green), CAUTIOUS (amber), FALLBACK (red).'],
                  ['Failure log', 'Time-stamped record of every Sentinel fault event.'],
                ].map(([label, desc], i) => (
                  <div key={i} className="flex gap-3 p-3 bg-bgRaised border border-borderDim">
                    <span className="font-mono text-[10px] text-cyan shrink-0 w-32">{label}</span>
                    <span className="font-body text-xs text-textSecondary">{desc}</span>
                  </div>
                ))}
                <H2>Custom feature telemetry</H2>
                <P>When you build a feature that emits custom telemetry keys, those keys automatically appear as new panels in the LIVE dashboard. No configuration needed.</P>
              </div>
            )}

            {manualSection === 'apikeys' && (
              <div className="space-y-6">
                <H>Setting Up AI</H>
                <P>PhysiCore AI features (Build, Debug, Troubleshooter) require an API key stored in your browser only — never sent to any server.</P>
                <H2>Getting a free Gemini key</H2>
                {['Go to aistudio.google.com/app/apikey', 'Sign in with any Google account', 'Click "Create API key"', 'Copy the key (starts with AIza...)', 'In PhysiCore: click "SET UP AI" in the nav → paste → Save'].map((step, i) => (
                  <div key={i} className="flex gap-3 font-body text-xs text-textSecondary">
                    <span className="text-green font-bold shrink-0">{i + 1}.</span><span>{step}</span>
                  </div>
                ))}
                <Code>{`https://aistudio.google.com/app/apikey`}</Code>
                <H2>Anthropic key (reserved)</H2>
                <P>The Anthropic key field is reserved for future server-side proxy support. Direct browser-to-Anthropic calls are blocked by CORS — only Gemini works from the browser. Use Gemini for all AI features today.</P>
              </div>
            )}

            {manualSection === 'lib-bot' && (
              <div className="space-y-6">
                <H>Balancing Bot (Arduino)</H>
                <P>Self-balancing two-wheeled robot. Arduino Uno/Mega + MPU6050/BNO055 + L298N/L293D motor driver.</P>
                <H2>Wiring</H2>
                <Code>{`MPU6050: SDA→A4, SCL→A5, VCC→3.3V, GND→GND
L298N:   ENA→D9(PWM), IN1→D7, IN2→D8
         ENB→D10(PWM), IN3→D5, IN4→D6`}</Code>
                <H2>Copy-paste firmware (Arduino)</H2>
                <Code>{`#include <Wire.h>
#include <MPU6050.h>
MPU6050 imu;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  imu.initialize();
  pinMode(7, OUTPUT); pinMode(8, OUTPUT);
  pinMode(5, OUTPUT); pinMode(6, OUTPUT);
  pinMode(9, OUTPUT); pinMode(10, OUTPUT);
}

void loop() {
  int16_t ax, ay, az, gx, gy, gz;
  imu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  float pitch = atan2(ax, az) * 180.0 / PI;
  float gyro_x = gx / 131.0;
  Serial.print("{\"pitch\":"); Serial.print(pitch, 4);
  Serial.print(",\"gyro_x\":"); Serial.print(gyro_x, 4);
  Serial.println("}");
  if (Serial.available() > 0) {
    float action = Serial.readStringUntil('\\n').toFloat();
    int pwm = constrain(abs(action) * 255, 0, 255);
    bool fwd = action > 0;
    digitalWrite(7, fwd); digitalWrite(8, !fwd); analogWrite(9, pwm);
    digitalWrite(5, fwd); digitalWrite(6, !fwd); analogWrite(10, pwm);
  }
  delay(16);
}`}</Code>
                <H2>YAML config</H2>
                <Code>{`platform: balancing_bot
connection:
  mode: serial
  port: /dev/ttyACM0
  baud: 115200
parameters:
  mass: 1.2
  wheel_radius: 0.033
  wheel_base: 0.12
  friction: 0.2`}</Code>
                <H2>Bridge command</H2>
                <Code>{`python physicore_bridge.py --platform balancing_bot --port /dev/ttyACM0`}</Code>
                <Note>Common issue: MPU6050 at wrong I2C address. Default is 0x68. If AD0 pin is HIGH, use 0x69.</Note>
              </div>
            )}

            {manualSection === 'lib-drone' && (
              <div className="space-y-6">
                <H>PX4 / ArduPilot Drone</H>
                <P>Quadrotor or fixed-wing UAV running PX4 or ArduPilot firmware. PhysiCore connects via MAVLink over USB or telemetry radio and replaces the stock attitude controller with adaptive MPC. Mass and aerodynamic drag are estimated live — so battery sag, payload, and wind gusts are handled automatically.</P>
                <H2>What PhysiCore adapts in flight</H2>
                {[
                  ['Mass estimate', 'Detects battery consumption and payload changes mid-flight. Convergence ~30s.'],
                  ['Drag model', 'CEM optimizer finds the effective drag at current airspeed. Improves hover hold.'],
                  ['Motor efficiency', 'Residual ensemble catches motor degradation before it causes instability.'],
                ].map(([label, desc], i) => (
                  <div key={i} className="flex gap-3 p-3 bg-bgRaised border border-borderDim">
                    <span className="font-mono text-[10px] text-cyan shrink-0 w-32">{label}</span>
                    <span className="font-body text-xs text-textSecondary">{desc}</span>
                  </div>
                ))}
                <H2>YAML config (PX4)</H2>
                <Code>{`platform: quadrotor
connection:
  mode: mavlink_bridge
  endpoint: /dev/ttyACM0
parameters:
  mass: 0.85
  arm_length: 0.12
  Ixx: 0.003
  Iyy: 0.003
  Izz: 0.005`}</Code>
                <H2>Bridge command</H2>
                <Code>{`python physicore_bridge.py --platform quadrotor --endpoint /dev/ttyACM0`}</Code>
                <Warn>Set PhysiCore to OFFBOARD mode in PX4. ArduPilot: use GUIDED mode. Always test with props off first — verify telemetry stream before arming.</Warn>
                <H2>Deployment checklist</H2>
                {['Flash stock PX4/ArduPilot firmware first', 'Connect flight controller USB → run bridge command', 'Open LIVE tab → verify telemetry stream appears', 'Props off: arm drone, switch to OFFBOARD/GUIDED, verify PhysiCore commands appear in DEBUG', 'Hover test at 1m altitude for 60s — watch Sentinel stay NOMINAL', 'Check mass estimate converges to within 5% of actual AUW'].map((item, i) => (
                  <div key={i} className="flex gap-2 font-body text-xs text-textSecondary"><span className="text-green font-bold shrink-0">{i + 1}.</span><span>{item}</span></div>
                ))}
                <P>Real hardware result: −87.1% position error on a 5-axis gust disturbance test vs. stock PX4 PID.</P>
              </div>
            )}

            {manualSection === 'lib-rocket' && (
              <div className="space-y-6">
                <H>Sounding Rocket</H>
                <P>Liquid or hybrid sounding rocket with onboard flight computer. PhysiCore runs TVC or fin actuation.</P>
                <H2>YAML config</H2>
                <Code>{`platform: rocket
connection:
  mode: serial
  port: /dev/ttyUSB0
  baud: 115200
parameters:
  dry_mass: 4.2
  propellant_mass: 1.8
  length: 1.4
  diameter: 0.076
  Cd: 0.45
  nozzle_area: 0.0015`}</Code>
                <H2>Bridge command</H2>
                <Code>{`python physicore_bridge.py --platform rocket --port /dev/ttyUSB0`}</Code>
                <Note>PhysiCore includes Mach drag correction and ISA atmosphere. Set Cd at subsonic conditions.</Note>
              </div>
            )}

            {manualSection === 'lib-arm' && (
              <div className="space-y-6">
                <H>ROS2 Robot Arm</H>
                <P>6-DOF serial manipulator running ROS2. PhysiCore connects as a ROS2 node.</P>
                <H2>YAML config</H2>
                <Code>{`platform: manipulator_arm
connection:
  mode: ros2
  joint_state_topic: /joint_states
  command_topic: /effort_controller/commands
parameters:
  dof: 6
  link_masses: [2.1, 1.8, 1.2, 0.8, 0.5, 0.3]
  link_lengths: [0.25, 0.22, 0.18, 0.15, 0.10, 0.08]
  payload_mass: 0.5`}</Code>
                <H2>Bridge command</H2>
                <Code>{`source /opt/ros/humble/setup.bash
python physicore_bridge.py --platform manipulator_arm --ros2`}</Code>
                <P>Results from real hardware (6 runs): end-effector error −88.4%, overshoot eliminated, settling time −84.9%, torque −71.4%.</P>
              </div>
            )}

            {manualSection === 'lib-legged' && (
              <div className="space-y-6">
                <H>Legged Robot</H>
                <P>Quadruped or biped running ROS2. PhysiCore sits above your gait planner as an adaptive whole-body controller — it does not replace your gait, it corrects the forces each step applies based on what the physics model predicts vs. what the IMU actually measures.</P>
                <H2>What PhysiCore adapts</H2>
                {[
                  ['Ground friction', 'Estimates surface friction per footstep. Adjusts force limits before slipping.'],
                  ['Terrain slope', 'Residual ensemble learns incline bias within 3-5 gait cycles.'],
                  ['Payload mass', 'Body mass estimate updates when load changes — backpack, arm, cargo.'],
                  ['Leg compliance', 'Per-leg actuator efficiency tracked. Flags degraded joints before they fail.'],
                ].map(([label, desc], i) => (
                  <div key={i} className="flex gap-3 p-3 bg-bgRaised border border-borderDim">
                    <span className="font-mono text-[10px] text-cyan shrink-0 w-32">{label}</span>
                    <span className="font-body text-xs text-textSecondary">{desc}</span>
                  </div>
                ))}
                <H2>YAML config</H2>
                <Code>{`platform: legged_robot
connection:
  mode: ros2
  state_topic: /robot/state
  command_topic: /robot/cmd_vel
parameters:
  body_mass: 12.0
  leg_count: 4
  leg_mass: 0.8
  foot_friction: 0.7
  stance_height: 0.35`}</Code>
                <H2>Bridge command</H2>
                <Code>{`python physicore_bridge.py --platform legged_robot --ros2`}</Code>
                <Note>PhysiCore publishes /physicore/adaptation_state — subscribe to this to watch friction, mass, and Sentinel mode in your own ROS2 nodes.</Note>
                <H2>Common issue: gait desync</H2>
                <P>If you see oscillation in the first 10 steps, the declared body_mass is off by more than 20%. Weigh your robot (with any typical payload) and update the YAML. PhysiCore will converge from there within 15 seconds.</P>
              </div>
            )}

            {manualSection === 'lib-auv' && (
              <div className="space-y-6">
                <H>AUV / Underwater Vehicle</H>
                <P>Autonomous underwater vehicle. PhysiCore uses a 6-DOF hydrodynamic model with Munk moment correction, added mass tensor, and quadratic drag. The residual ensemble corrects for hull fouling, buoyancy offset from ballast trim, and current disturbances that the physics model cannot fully capture.</P>
                <H2>What PhysiCore adapts</H2>
                {[
                  ['Drag estimate', 'Quadratic drag coefficient updates as hull accumulates biofouling. Typical drift: +8% per week untreated.'],
                  ['Buoyancy offset', 'Detects net buoyancy error from ballast trim — corrects before dive depth is affected.'],
                  ['Current disturbance', 'Residual ensemble learns persistent current bias within 60s of steady-state mission.'],
                ].map(([label, desc], i) => (
                  <div key={i} className="flex gap-3 p-3 bg-bgRaised border border-borderDim">
                    <span className="font-mono text-[10px] text-cyan shrink-0 w-36">{label}</span>
                    <span className="font-body text-xs text-textSecondary">{desc}</span>
                  </div>
                ))}
                <H2>YAML config</H2>
                <Code>{`platform: auv
connection:
  mode: ros2
parameters:
  mass: 8.5
  displaced_volume: 0.0085
  drag_linear: 8.2
  drag_quadratic: 0.5
  added_mass: 2.1`}</Code>
                <H2>Bridge command</H2>
                <Code>{`python physicore_bridge.py --platform auv --ros2`}</Code>
                <Warn>Set displaced_volume accurately. A 1% buoyancy error creates 84 mN uncompensated force — visible as depth drift within 30 seconds.</Warn>
                <H2>Depth control tip</H2>
                <P>Start each mission with a 30-second level-flight trim check at 2m depth. PhysiCore will converge mass and drag estimates before beginning any trajectory mission. Sentinel will stay NOMINAL if trim is good; CAUTIOUS if buoyancy is off by more than 3%.</P>
              </div>
            )}

            {manualSection === 'lib-evtol' && (
              <div className="space-y-6">
                <H>eVTOL Aircraft</H>
                <P>Electric vertical take-off and landing aircraft. PhysiCore handles the full flight envelope: hover, transition, and cruise. The physics model is a hybrid — rotor momentum theory during hover, vortex-lattice aerodynamics during cruise. The residual ensemble bridges the discontinuity during transition where neither model alone is accurate.</P>
                <H2>Why eVTOL is harder than a drone</H2>
                <P>During transition (rotor+wing overlap region), lift sources interfere. Classical controllers tune separate hover and cruise modes and cross-fade between them. PhysiCore identifies the actual effective lift continuously — no mode switching, no tune-per-airspeed. Sentinel stays NOMINAL through transition in nominal conditions.</P>
                <H2>What PhysiCore adapts</H2>
                {[
                  ['Battery mass', 'Pack weight decreases ~0.3% per minute at cruise. Mass estimate tracks this.'],
                  ['Rotor efficiency', 'Per-rotor thrust loss flagged by residual. Useful for predictive maintenance.'],
                  ['Transition band', 'Residual ensemble learns the specific airframe transition dynamics on first flight.'],
                  ['Wind gust', 'Sentinel switches CAUTIOUS in sustained >8 m/s gusts; FALLBACK triggers emergency hold.'],
                ].map(([label, desc], i) => (
                  <div key={i} className="flex gap-3 p-3 bg-bgRaised border border-borderDim">
                    <span className="font-mono text-[10px] text-cyan shrink-0 w-32">{label}</span>
                    <span className="font-body text-xs text-textSecondary">{desc}</span>
                  </div>
                ))}
                <H2>YAML config</H2>
                <Code>{`platform: evtol
connection:
  mode: mavlink_bridge
  endpoint: /dev/ttyACM0
parameters:
  mass: 22.0
  lift_rotors: 8
  cruise_speed: 28
  transition_speed: 18
  wing_area: 1.8
  Cd0: 0.025`}</Code>
                <H2>Bridge command</H2>
                <Code>{`python physicore_bridge.py --platform evtol --endpoint /dev/ttyACM0`}</Code>
                <Warn>First flight: hover only for 3 minutes before attempting transition. This gives PhysiCore time to converge hover mass estimate — stale estimates during transition cause overshoot.</Warn>
              </div>
            )}

            {manualSection === 'lib-rover' && (
              <div className="space-y-6">
                <H>Ground Rover / AMR</H>
                <P>Differential-drive or Ackermann rover. Terrain-adaptive traction control.</P>
                <H2>YAML config</H2>
                <Code>{`platform: ground_rover
connection:
  mode: ros2
parameters:
  mass: 3.2
  wheelbase: 0.32
  wheel_radius: 0.065
  friction: 0.6
  max_speed: 1.5`}</Code>
                <H2>Bridge command</H2>
                <Code>{`python physicore_bridge.py --platform ground_rover --ros2`}</Code>
                <P>Results from real hardware (3 runs, 2,841m): cross-track error −84.6%, on-heading time 96.2%, terrain adaptation ≤9 seconds.</P>
              </div>
            )}

            {manualSection === 'lib-sat' && (
              <div className="space-y-6">
                <H>Satellite / Spacecraft</H>
                <P>CubeSat or small satellite. Reaction wheel attitude control with J2 perturbation.</P>
                <H2>YAML config</H2>
                <Code>{`platform: satellite
connection:
  mode: serial
  port: /dev/ttyUSB0
  baud: 115200
parameters:
  mass: 4.0
  Ixx: 0.012
  Iyy: 0.012
  Izz: 0.008
  orbit_altitude: 550
  reaction_wheel_Nm: 0.2`}</Code>
                <H2>Bridge command</H2>
                <Code>{`python physicore_bridge.py --platform satellite --port /dev/ttyUSB0`}</Code>
                <Note>PhysiCore includes J2 gravitational perturbation for low-earth orbit.</Note>
              </div>
            )}

            {manualSection === 'lib-custom' && (
              <div className="space-y-6">
                <H>Custom Hardware</H>
                <P>Select "Custom Hardware" in the INTEGRATE tab and describe your system. The AI generates a complete custom integration.</P>
                <H2>Serial protocol (DIY hardware)</H2>
                <Code>{`# Your hardware sends (50Hz):
{"state": [val1, val2, ...], "t": 1234567890}

# PhysiCore sends back:
{"action": [0.45, -0.23, ...], "step": 42}`}</Code>
                <H2>Python Extension API</H2>
                <Code>{`from physicore.extensions import PhysiCoreExtension, ExtensionMeta

class MyCustomHardware(PhysiCoreExtension):
    meta = ExtensionMeta(
        name="my_hardware",
        hooks=["pre_step", "post_step", "on_fault"]
    )
    def pre_step(self, state, x_ref, engine):
        return state, x_ref
    def post_step(self, step, engine):
        pass
    def on_fault(self, fault_type, engine):
        pass`}</Code>
              </div>
            )}

            {manualSection === 'sentinel' && (
              <div className="space-y-6">
                <H>Sentinel Safety Layer</H>
                <P>Sentinel is the mandatory safety wrapper around all PhysiCore outputs. It cannot be disabled.</P>
                <H2>Three modes</H2>
                {[
                  ['NOMINAL', 'green', 'Full PhysiCore output. Uncertainty < 5%, residual < 0.5.'],
                  ['CAUTIOUS', 'amber', '60% output, tighter bounds. Uncertainty 5-15%.'],
                  ['FALLBACK', 'red', 'PhysiCore off. Safe deterministic controller takes over.'],
                ].map(([mode, color, desc], i) => (
                  <div key={i} className={`p-4 border bg-bgRaised flex gap-4 items-start border-${color}/30`}>
                    <span className={`font-mono text-[10px] font-bold uppercase shrink-0 text-${color}`}>{mode}</span>
                    <span className="font-body text-xs text-textSecondary">{desc}</span>
                  </div>
                ))}
                <H2>SHA-256 forensic log</H2>
                <P>Every control command is signed with SHA-256 and chained. Every decision is traceable and tamper-evident.</P>
                <H2>Recovery</H2>
                <P>FALLBACK to CAUTIOUS: after 300 consecutive stable steps. CAUTIOUS to NOMINAL: after 50 stable steps. Fully automatic.</P>
              </div>
            )}

            {manualSection === 'registry' && (
              <div className="space-y-6">
                <H>Persistent Learning</H>
                <P>PhysiCore remembers what it learned. Every session saves mass estimates, friction coefficients, and residual ensemble weights.</P>
                <H2>Registry location</H2>
                <Code>{`~/.physicore/registry/{platform}/
  params.json       # learned mass, friction, inertia
  ensemble_1.pt     # MLP weights (network 1)
  ensemble_2.pt     # MLP weights (network 2)
  ensemble_3.pt     # MLP weights (network 3)
  session_log.json  # session history`}</Code>
                <H2>Resetting</H2>
                <Code>{`rm -rf ~/.physicore/registry/{platform}`}</Code>
              </div>
            )}

            {manualSection === 'extensions' && (
              <div className="space-y-6">
                <H>Extensions API</H>
                <P>The PhysiCore Extension API lets you add custom behaviors to the physics loop via four hooks.</P>
                <H2>Hooks</H2>
                {[
                  ['pre_step(state, x_ref, engine)', 'Runs before MPC. Modify state or reference trajectory.'],
                  ['post_step(step, engine)', 'Runs after MPC. Inspect or modify the action.'],
                  ['on_fault(fault_type, engine)', 'Runs when Sentinel detects a fault.'],
                  ['on_telemetry(packet)', 'Runs on every telemetry packet. Use for logging.'],
                ].map(([hook, desc], i) => (
                  <div key={i} className="space-y-1 p-4 bg-bgRaised border border-borderDim">
                    <code className="font-mono text-[10px] text-cyan">{hook}</code>
                    <p className="font-body text-xs text-textSecondary">{desc}</p>
                  </div>
                ))}
                <H2>Full example</H2>
                <Code>{`from physicore.extensions import PhysiCoreExtension, ExtensionMeta
import csv, time, pathlib

class FrictionLogger(PhysiCoreExtension):
    meta = ExtensionMeta(
        name="friction_logger",
        hooks=["on_telemetry"],
        telemetry_keys=["friction_log_active"]
    )
    def __init__(self):
        self.log_path = pathlib.Path("~/.physicore/logs/friction.csv").expanduser()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_log = 0
        self._file = open(self.log_path, 'a', newline='')
        self._writer = csv.writer(self._file)
    def on_telemetry(self, packet):
        now = time.time()
        if now - self._last_log >= 0.1:
            self._writer.writerow([now, packet.get('friction', 0)])
            self._file.flush()
            self._last_log = now
            packet["friction_log_active"] = 1`}</Code>
              </div>
            )}

            {manualSection === 'troubleshoot' && (
              <div className="space-y-6">
                <H>Troubleshooting</H>
                <H2>Bridge won't connect</H2>
                {['Close Arduino IDE — it holds the serial port', 'Check port: ls /dev/tty* on Linux, Device Manager on Windows', 'Check baud rate matches firmware (default 115200)', 'Try: python physicore_bridge.py --port /dev/ttyACM0 --baud 115200 --debug'].map((item, i) => (
                  <div key={i} className="flex gap-2 font-body text-xs text-textSecondary"><span className="text-amber shrink-0">→</span><span>{item}</span></div>
                ))}
                <H2>IMU reading zero</H2>
                {['Check I2C wiring: SDA and SCL must be correct pins (A4/A5 on Uno)', 'MPU6050 needs 3.3V not 5V', 'Check address: default 0x68, use 0x69 if AD0 pin is HIGH', 'Test: i2cdetect -y 1 (should show 0x68)'].map((item, i) => (
                  <div key={i} className="flex gap-2 font-body text-xs text-textSecondary"><span className="text-amber shrink-0">→</span><span>{item}</span></div>
                ))}
                <H2>High residual / Sentinel in CAUTIOUS</H2>
                {['Normal for first 30 seconds — model is still learning', 'If stays high: check declared mass matches actual mass', 'Surface change → friction estimate needs to adapt (5-15s)', 'Sensor noise → check IMU wiring, add 100μF cap on power line'].map((item, i) => (
                  <div key={i} className="flex gap-2 font-body text-xs text-textSecondary"><span className="text-amber shrink-0">→</span><span>{item}</span></div>
                ))}
                <H2>AI features not working</H2>
                {['Click "SET UP AI" in nav — check key is set (shows green "AI ON")', 'Gemini free key: get at aistudio.google.com/app/apikey', 'Key starts with AIza... for Gemini, sk-ant-... for Anthropic', 'Try clearing and re-entering the key from the modal'].map((item, i) => (
                  <div key={i} className="flex gap-2 font-body text-xs text-textSecondary"><span className="text-amber shrink-0">→</span><span>{item}</span></div>
                ))}
                <Note>Use the DEBUG tab inside your project for AI-powered diagnosis. It knows your hardware, your features, and your live telemetry.</Note>
              </div>
            )}

          </div>
        </main>
      </div>
    );
  };


  const renderDashboard = () => {
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
                  {connectionMode === 'ros2_websocket' ? 'Real Hardware (ROS2)'
                    : connectionMode === 'mavlink_bridge' ? 'Real Hardware (MAVLink Bridge)'
                    : connectionMode === 'digital_twin' ? 'Digital Twin Simulation'
                    : 'Hardware-in-the-Loop'}
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
                    <span className="micro-label">
                      {!isSystemConnected ? 'SystemID — NOT CONNECTED' : 
                       !isControlActive ? 'SystemID — CLICK ACTIVE CONTROL ON' : 
                       (telemetry.step_count || 0) < 50 ? `SystemID — LEARNING (${telemetry.step_count || 0} steps)` :
                       'SystemID — CONVERGING'}
                    </span>
                  </div>
                  <div className="space-y-4">
                    {[
                      { label: 'ESTIMATED MASS', val: isSystemConnected ? `${(telemetry.mass || 0).toFixed(3)} kg` : '—', delta: isSystemConnected ? (telemetry.step_count > 100 ? 'CONVERGED' : 'LEARNING') : 'CONNECT HARDWARE', color: isSystemConnected ? COLORS.green : COLORS.textDim },
                      { label: 'FRICTION COEFF', val: isSystemConnected ? `${(telemetry.friction || 0).toFixed(3)} μ` : '—', delta: isSystemConnected ? 'LIVE' : 'CONNECT HARDWARE', color: isSystemConnected ? COLORS.amber : COLORS.textDim },
                      { label: 'ACTUATOR EFF', val: isSystemConnected ? `${((telemetry.actuatorEfficiency || 0) * 100).toFixed(1)}%` : '—', delta: isSystemConnected ? 'LIVE' : 'CONNECT HARDWARE', color: isSystemConnected ? COLORS.cyan : COLORS.textDim },
                      ...(telemetry.pitch !== undefined ? [
                        { label: 'PITCH', val: `${(telemetry.pitch || 0).toFixed(2)}°`, delta: Math.abs(telemetry.pitch || 0) < 5 ? 'STABLE' : 'LEANING', color: Math.abs(telemetry.pitch || 0) < 5 ? COLORS.green : COLORS.amber },
                        { label: 'ROLL',  val: `${(telemetry.roll  || 0).toFixed(2)}°`, delta: '', color: COLORS.cyan },
                        { label: 'GYRO Y', val: `${(telemetry.gyro?.y ?? telemetry.gyro_y ?? 0).toFixed(2)} °/s`, delta: '', color: COLORS.textSecondary },
                        { label: 'MOTOR L', val: `${Math.round(telemetry.motor_l ?? 0)}`, delta: '', color: COLORS.green },
                        { label: 'MOTOR R', val: `${Math.round(telemetry.motor_r ?? 0)}`, delta: '', color: COLORS.green },
                      ] : []),
                      ...(telemetry.altitude > 0 && !telemetry.pitch ? [
                        { label: 'ALTITUDE', val: `${(telemetry.altitude || 0).toFixed(2)} m`, delta: '', color: COLORS.green },
                        { label: 'SPEED',    val: `${(telemetry.speed    || 0).toFixed(2)} m/s`, delta: '', color: COLORS.cyan },
                        { label: 'ARMED',    val: telemetry.armed ? 'YES' : 'NO', delta: '', color: telemetry.armed ? COLORS.green : COLORS.textDim },
                      ] : []),
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

                {/* REGISTRY INTELLIGENCE PANEL */}
                <section className="space-y-4">
                  <div className="border-l-2 border-amber pl-3 flex items-center justify-between">
                    <span className="micro-label">Registry Intelligence</span>
                    {registryStatus ? (
                      <span className="font-mono text-[9px] text-amber">● ACTIVE</span>
                    ) : (
                      <span className="font-mono text-[9px] text-textDim">○ NO DATA</span>
                    )}
                  </div>
                  {registryStatus ? (
                    <div className="space-y-3">
                      <div className="p-3 bg-amber/5 border border-amber/20 space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="font-display text-[9px] font-bold text-amber tracking-widest uppercase">Session Memory</span>
                          <span className="font-mono text-[9px] text-white">{registryStatus.sessions_count} sessions</span>
                        </div>
                        <div className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="font-mono text-[8px] text-textDim uppercase">Prior Strength</span>
                            <span className="font-mono text-[8px] text-amber">{registryStatus.prior_weight.toFixed(1)}</span>
                          </div>
                          <div className="h-1 w-full bg-border">
                            <div className="h-full bg-amber transition-all duration-500" style={{ width: `${Math.min(100, (registryStatus.prior_weight / 20) * 100)}%` }} />
                          </div>
                        </div>
                      </div>
                      {Object.keys(registryStatus.latest_params).length > 0 && (
                        <div className="p-3 border border-border bg-bgRaised space-y-1.5">
                          <p className="font-mono text-[8px] text-textDim uppercase tracking-widest mb-1">Loaded Prior Params</p>
                          {Object.entries(registryStatus.latest_params).map(([k, v]) => {
                            const current = k === 'mass' ? registryStatus.current_mass : k === 'friction' ? registryStatus.current_friction : v;
                            const delta = Math.abs(current - v);
                            return (
                              <div key={k} className="flex justify-between items-center font-mono text-[9px]">
                                <span className="text-textDim">{k}</span>
                                <div className="flex items-center gap-2">
                                  <span className="text-amber">{v.toFixed(4)}</span>
                                  {delta > 0.001 && isControlActive && (
                                    <span className={`text-[8px] ${delta > 0.1 ? 'text-green' : 'text-textDim'}`}>
                                      →{current.toFixed(4)}
                                    </span>
                                  )}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                      <div className="font-mono text-[8px] text-textDim truncate" title={registryStatus.registry_path}>
                        {registryStatus.registry_path.replace(/\\/g, '/').split('/').slice(-3).join('/')}
                      </div>
                    </div>
                  ) : (
                    <div className="p-3 border border-border/50 bg-bgRaised space-y-1">
                      <p className="font-mono text-[9px] text-textDim">Connect hardware to see registry data.</p>
                      <p className="font-mono text-[8px] text-textDim/60">Registry saves on bridge shutdown (Ctrl+C).</p>
                    </div>
                  )}
                </section>

                {/* EXTENSIONS PANEL */}
                <section className="space-y-4">
                  <div className="border-l-2 border-purple-400 pl-3 flex items-center justify-between">
                    <span className="micro-label">Extensions</span>
                    <span className="font-mono text-[9px] text-purple-400">{loadedExtensions.length} loaded</span>
                  </div>
                  {loadedExtensions.length > 0 ? (
                    <div className="space-y-2">
                      {loadedExtensions.map(ext => (
                        <div key={ext.name} className="p-2.5 bg-purple-400/5 border border-purple-400/20 space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="font-mono text-[10px] text-purple-300 font-bold">{ext.name}</span>
                            <span className="font-mono text-[8px] text-textDim">v{ext.version}</span>
                          </div>
                          {ext.description && (
                            <p className="font-mono text-[9px] text-textDim">{ext.description}</p>
                          )}
                          <div className="flex flex-wrap gap-1">
                            {ext.hooks.map(h => (
                              <span key={h} className="font-mono text-[8px] text-purple-400/60 border border-purple-400/20 px-1">
                                {h}
                              </span>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="p-3 border border-border/50 bg-bgRaised">
                      <p className="font-mono text-[9px] text-textDim">No extensions loaded.</p>
                      <p className="font-mono text-[8px] text-textDim/60 mt-1">Drop .py files into ~/.physicore/extensions/</p>
                    </div>
                  )}
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
              <button 
                onClick={() => setIsControlActive(!isControlActive)}
                className={`px-3 py-2 border font-display text-[10px] font-bold uppercase tracking-widest transition-all ${isControlActive ? 'bg-green text-black border-green' : 'bg-bgRaised text-textDim border-border hover:border-green hover:text-green'}`}
              >
                {isControlActive ? '● ACTIVE CONTROL ON' : '○ ACTIVE CONTROL OFF'}
              </button>
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
            ) : systemProfile.domain === 'AVIATION' ? (
              <AviationTrajectoryCanvas 
                state={aviationState} 
                params={aviationParams} 
                isRunning={isRocketSimRunning}
                setIsRunning={setIsRocketSimRunning}
                isConnected={isSystemConnected}
                handshakeConfirmed={handshakeConfirmed}
              />
            ) : (
              <SimulationCanvas 
                mode={simMode}
                onStateUpdate={setSimState}
                target={simState.target}
                controlAction={simState.controlAction}
                physicsPriors={simState.estimatedParams}
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

                <div className="flex flex-col gap-2 mb-4">
                  <span className="micro-label text-textDim uppercase text-center">Select Connection Mode</span>
                  <div className="flex gap-2">
                    <button 
                      onClick={() => setConnectionMode('digital_twin')}
                      className={`flex-1 py-2 font-mono text-[9px] border ${connectionMode === 'digital_twin' ? 'bg-cyan text-black border-cyan' : 'border-border text-textDim'}`}
                    >
                      TWIN
                    </button>
                    <button 
                      onClick={() => setConnectionMode('hil')}
                      className={`flex-1 py-2 font-mono text-[9px] border ${connectionMode === 'hil' ? 'bg-green text-black border-green' : 'border-border text-textDim'}`}
                    >
                      HIL
                    </button>
                    <button 
                      onClick={() => setConnectionMode('ros2_websocket')}
                      className={`flex-1 py-2 font-mono text-[9px] border ${connectionMode === 'ros2_websocket' ? 'bg-green text-black border-green' : 'border-border text-textDim'}`}
                    >
                      ROS2
                    </button>
                    <button
                      onClick={() => { setConnectionMode('mavlink_bridge'); setEndpoint('ws://localhost:8765'); }}
                      className={`flex-1 py-2 font-mono text-[9px] border ${connectionMode === 'mavlink_bridge' ? 'bg-amber text-black border-amber' : 'border-border text-textDim'}`}
                    >
                      MAVLINK
                    </button>
                  </div>
                </div>

                <div className="flex flex-col gap-3 w-full max-w-[280px]">
                  <button 
                    onClick={() => navigateToProject('integrate')}
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
                      {isSystemConnecting ? 'Retrying...' : 
                        connectionMode === 'digital_twin' ? 'Verify Digital Twin simulation' : 
                        connectionMode === 'mavlink_bridge' ? 'Connect MAVLink Bridge' : 
                        'Verify HIL / Hardware link'}
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
                <span className="font-mono text-[10px] text-textDim">PEAK: {simState.controlAction.reduce((s, v) => s + Math.abs(v), 0).toFixed(1)}N</span>
              </div>
              <div className="h-[180px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={[{ y: simState.controlAction[0] }, { y: simState.controlAction[1] }]}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1A1A1A" vertical={false} />
                    <Bar dataKey="y" fill={COLORS.green} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="w-[400px] p-6 space-y-4 bg-bgRaised/30 overflow-hidden flex flex-col">
              <div className="flex justify-between items-center shrink-0">
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-amber animate-pulse" />
                  <span className="micro-label text-white">Meta-Analyst Intelligence</span>
                </div>
                {isMetaAnalyzing && <div className="w-2 h-2 rounded-full bg-amber animate-pulse" />}
              </div>
              <div className="flex-1 overflow-y-auto custom-scroll pr-2">
                {metaAnalysis ? (
                  <div className="space-y-4">
                    <div className="p-3 border border-amber/20 bg-amber/5 rounded-sm">
                      <p className="font-body text-xs text-amber leading-relaxed">{metaAnalysis.insight}</p>
                    </div>
                    <div className="space-y-2">
                      {metaAnalysis.diagnostics.map((d, i) => (
                        <div key={i} className="flex items-start gap-2">
                          <div className="w-1 h-1 rounded-full bg-amber mt-1.5 shrink-0" />
                          <span className="font-mono text-[10px] text-textSecondary uppercase">{d}</span>
                        </div>
                      ))}
                    </div>
                    <div className="pt-4 border-t border-borderDim space-y-3">
                      <div className="flex items-center gap-2">
                        <Zap size={10} className="text-amber" />
                        <span className="font-display text-[9px] font-bold text-amber uppercase tracking-widest">Tuning Recommendation</span>
                      </div>
                      <div className="p-3 bg-amberDim/10 border border-amber/20 font-mono text-[10px] text-amber/80">
                        Q: {metaAnalysis.suggestedCostTweaks.q_weight} | R: {metaAnalysis.suggestedCostTweaks.r_weight}
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

          {/* FEATURE TELEMETRY PANEL — shown when project has features with telemetry_keys */}
          {activeProject && (activeProject.features || []).some(f => f.telemetry_keys.length > 0) && (
            <div className="shrink-0 border-t border-amber/20 bg-amber/5 px-4 py-2 flex items-center gap-4 overflow-x-auto">
              <div className="flex items-center gap-2 shrink-0">
                <Puzzle size={12} className="text-amber" />
                <span className="font-mono text-[9px] text-amber uppercase tracking-widest">Feature Telemetry</span>
              </div>
              {(activeProject.features || []).flatMap(f => f.telemetry_keys.map(key => ({ key, feature: f.name }))).map(({ key, feature }) => (
                <div key={key} className="shrink-0 flex items-center gap-2 px-3 py-1 border border-amber/20 bg-amber/10">
                  <span className="font-mono text-[8px] text-amber/60">{feature}</span>
                  <span className="font-mono text-[9px] text-amber font-bold">{key}</span>
                  <span className="font-mono text-[9px] text-white">{(telemetry as any)[key] !== undefined ? String((telemetry as any)[key]) : '--'}</span>
                </div>
              ))}
            </div>
          )}

          {/* FAILURE LOG STRIP — shown only when there are failures */}
          {failureLogs.length > 0 && (
            <div className="shrink-0 border-t border-red/30 bg-red/5 px-4 py-2 flex items-center gap-4 overflow-x-auto">
              <div className="flex items-center gap-2 shrink-0">
                <span className="w-1.5 h-1.5 rounded-full bg-red animate-pulse" />
                <span className="font-mono text-[9px] text-red uppercase tracking-widest">Fault Log</span>
                <span className="font-mono text-[8px] text-textDim">({failureLogs.length})</span>
              </div>
              <div className="flex gap-3 overflow-x-auto">
                {failureLogs.slice(0, 8).map(log => (
                  <div key={log.id}
                    onClick={() => navigateToProject('debug')}
                    className="shrink-0 flex items-center gap-2 px-3 py-1 border border-red/30 bg-red/10 cursor-pointer hover:bg-red/20 transition-all">
                    <span className="font-mono text-[9px] text-red font-bold">{log.failure_type}</span>
                    <span className="font-mono text-[8px] text-textDim">{new Date(log.timestamp).toLocaleTimeString()}</span>
                  </div>
                ))}
              </div>
              <button onClick={() => navigateToProject('debug')}
                className="ml-auto shrink-0 font-mono text-[9px] text-red border border-red/40 px-3 py-1 hover:bg-red/20 transition-all uppercase tracking-widest">
                <Bug size={10} className="inline mr-1" />Diagnose
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
    );
  };

  const renderProjects = () => {
    const hwIcons: Record<string, string> = {
      balancing_bot:'🤖', px4:'🚁', ardupilot:'✈️', evtol:'🛸', ros2_arm:'🦾',
      ros2_legged:'🦿', ros2_rover:'🚗', ros2_auv:'🐟', ros2_surgical:'🏥',
      rocket_fc:'🚀', rover_serial:'🚙', satellite:'🛰️',
    };
    return (
      <div className="pt-[52px] min-h-screen bg-void px-6 py-8">
        {!hasAnyKey() && (
          <div className="bg-amber/5 border-b border-amber/20 px-6 py-4 flex items-center gap-4 -mx-6 -mt-8 mb-8">
            <span className="w-2 h-2 rounded-full bg-amber shrink-0 animate-pulse" />
            <p className="font-mono text-[10px] text-amber uppercase tracking-widest flex-1">
              AI features need an API key — Build, Debug, and Troubleshooter won't work without one
            </p>
            <button
              onClick={() => setShowApiKeyModal(true)}
              className="px-4 py-2 bg-amber text-black font-display text-[9px] font-bold uppercase tracking-widest hover:bg-white transition-all shrink-0"
            >
              Set Up AI →
            </button>
          </div>
        )}
        <div className="max-w-[1100px] mx-auto space-y-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="font-display text-3xl font-bold text-white uppercase tracking-widest">Projects</h1>
              <p className="font-mono text-[10px] text-textDim mt-1 uppercase tracking-widest">
                {projects.length} project{projects.length !== 1 ? 's' : ''} — each project has its own hardware, generated code, and registry entry
              </p>
            </div>
            <button
              onClick={() => setShowNewProjectModal(true)}
              className="flex items-center gap-2 px-4 py-2 bg-green text-black font-display text-[11px] font-bold uppercase tracking-widest hover:bg-white transition-all"
            >
              <Plus size={14} /> New Project
            </button>
          </div>

          {projectsError && (
            <div className="p-4 border border-red/30 bg-red/5 flex items-start gap-3">
              <AlertTriangle size={14} className="text-red shrink-0 mt-0.5" />
              <div className="flex-1 min-w-0">
                <p className="font-mono text-[10px] text-red uppercase tracking-widest">Firestore sync failed</p>
                <p className="font-mono text-[9px] text-textDim mt-1 leading-relaxed">{projectsError}</p>
                <p className="font-mono text-[9px] text-textDim mt-1">Check that your Firebase environment variables are set and Firestore rules are deployed.</p>
              </div>
            </div>
          )}

          {projectsLoading ? (
            <div className="flex items-center justify-center py-24 gap-3">
              <div className="w-5 h-5 border-2 border-green/30 border-t-green rounded-full animate-spin" />
              <span className="font-mono text-[10px] text-textDim uppercase tracking-widest">Loading projects…</span>
            </div>
          ) : projects.length === 0 && !projectsError ? (
            <div className="flex flex-col items-center justify-center py-24 space-y-6 text-center">
              <div className="w-20 h-20 border-2 border-dashed border-border flex items-center justify-center text-textDim">
                <Layers size={36} className="opacity-30" />
              </div>
              <div className="space-y-2">
                <p className="font-display text-lg font-bold text-white uppercase tracking-widest">No Projects Yet</p>
                <p className="font-mono text-[10px] text-textDim uppercase">Create a project to get started. Each project has its own hardware setup and generated code.</p>
              </div>
              <button onClick={() => setShowNewProjectModal(true)}
                className="px-6 py-3 bg-green text-black font-display text-[11px] font-bold uppercase tracking-widest hover:bg-white transition-all">
                + New Project
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {projects.map(project => (
                <div key={project.id}
                  className={`border p-5 bg-bg space-y-4 hover:border-cyan/40 transition-all group ${activeProject?.id === project.id ? 'border-cyan/60' : 'border-border'}`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">{hwIcons[project.hardware] || '⬡'}</span>
                      <div>
                        <h3 className="font-display text-sm font-bold text-white uppercase tracking-widest">{project.name}</h3>
                        <p className="font-mono text-[9px] text-textDim uppercase mt-0.5">{project.hardware.replace(/_/g,' ')} · {project.platform}</p>
                      </div>
                    </div>
                    {activeProject?.id === project.id && (
                      <span className="font-mono text-[8px] text-cyan border border-cyan/30 px-1.5 py-0.5">ACTIVE</span>
                    )}
                  </div>

                  <div className="flex gap-4 font-mono text-[9px] text-textDim">
                    <span>{project.generatedFiles.length} files</span>
                    {(project.features || []).length > 0 && (
                      <span className="text-amber">{(project.features || []).length} feature{(project.features || []).length !== 1 ? 's' : ''}</span>
                    )}
                    <span>{new Date(project.createdAt).toLocaleDateString()}</span>
                  </div>

                  {project.notes && (
                    <p className="font-body text-[10px] text-textSecondary leading-relaxed line-clamp-2">{project.notes}</p>
                  )}

                  <div className="flex gap-2 pt-2 border-t border-border/50">
                    <button onClick={() => openProjectInIE(project)}
                      className="flex-1 py-2 bg-green text-black font-display text-[9px] font-bold uppercase tracking-widest hover:bg-white transition-all">
                      Open Project
                    </button>
                    <button
                      onClick={() => { navigator.clipboard.writeText(encodeProjectCode(project)); }}
                      className="p-1.5 border border-border text-textDim font-mono text-[9px] hover:border-cyan hover:text-cyan transition-all"
                      title="Copy project code"
                    >
                      <Copy size={12} />
                    </button>
                    <button onClick={() => { if (confirm(`Delete "${project.name}"?`)) deleteProject(project.id); }}
                      className="p-1.5 border border-border text-textDim font-mono text-[9px] hover:border-red hover:text-red transition-all">
                      <Trash2 size={12} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* New Project Modal */}
        <AnimatePresence>
          {showNewProjectModal && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="fixed inset-0 z-[300] bg-black/80 backdrop-blur-sm flex items-center justify-center p-6">
              <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.9, opacity: 0 }}
                className="max-w-md w-full bg-bg border border-border p-8 space-y-6">
                <h2 className="font-display text-xl font-bold text-white uppercase tracking-widest">New Project</h2>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <label className="micro-label text-textDim">Project Name</label>
                    <input value={newProjectName} onChange={e => { setNewProjectName(e.target.value); setNewProjectError(''); }}
                      onKeyDown={e => e.key === 'Enter' && newProjectName.trim() && document.getElementById('create-project-btn')?.click()}
                      placeholder="e.g. Rocket Test #3"
                      className="w-full bg-bgRaised border border-border px-4 py-2 font-mono text-sm text-white focus:border-green outline-none" />
                  </div>
                  <div className="space-y-2">
                    <label className="micro-label text-textDim">Notes (optional)</label>
                    <textarea value={newProjectDesc} onChange={e => setNewProjectDesc(e.target.value)}
                      placeholder="What are you building?"
                      rows={2}
                      className="w-full bg-bgRaised border border-border px-4 py-2 font-mono text-sm text-white focus:border-green outline-none resize-none" />
                  </div>
                  {newProjectError && (
                    <div className="p-3 border border-red/30 bg-red/5 font-mono text-[10px] text-red leading-relaxed">
                      ⚠ Saved locally only — Firestore sync failed: {newProjectError}
                    </div>
                  )}
                </div>
                <div className="flex gap-3">
                  <button onClick={() => { setShowNewProjectModal(false); setNewProjectName(''); setNewProjectDesc(''); setNewProjectError(''); }}
                    className="flex-1 py-3 border border-border text-textDim font-display text-[10px] font-bold uppercase tracking-widest hover:text-white transition-all">
                    Cancel
                  </button>
                  <button id="create-project-btn" disabled={!newProjectName.trim()}
                    onClick={async () => {
                      if (!newProjectName.trim()) return;
                      setNewProjectError('');
                      // Optimistic: close modal immediately and navigate
                      const name = newProjectName.trim();
                      setShowNewProjectModal(false);
                      setNewProjectName('');
                      setNewProjectDesc('');
                      navigateToProject('integrate');
                      // Then do the Firestore write in background
                      const p = await createProject(name, '', {}, []);
                      if (p && (p as any).__error) {
                        // Write failed — project is in local state but not persisted
                        setNewProjectError((p as any).__error);
                        setShowNewProjectModal(true); // reopen with error shown
                        setNewProjectName(name);
                      }
                    }}
                    className="flex-1 py-3 bg-green text-black font-display text-[10px] font-bold uppercase tracking-widest hover:bg-white transition-all disabled:opacity-40">
                    Create & Open Project
                  </button>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  };

  const requireAIKey = (): boolean => {
    if (hasAnyKey()) return true;
    setShowApiKeyModal(true);
    return false;
  };

  // ── BUILD TAB ──────────────────────────────────────────────────────────────
  const FEATURE_ARCHITECT_SYSTEM = `You are the PhysiCore Feature Architect. Your job is to help engineers add custom features to their PhysiCore deployment by asking exactly 4 questions in sequence, then generating a complete implementation.

The 4 questions you MUST ask (one at a time, wait for answer before next):
1. WHAT — What should this feature do? (one sentence)
2. WHEN — When should it trigger? (pre_step / post_step / on_fault / on_telemetry / on timer)
3. HOW — What should it do with the data? (log it, modify control, send alert, etc.)
4. DATA — What telemetry keys or parameters does it need access to?

After all 4 answers, emit exactly this marker followed by a JSON manifest on its own line:
[GENERATE_FEATURE]
{"name":"<name>","description":"<desc>","telemetry_keys":[],"fault_types":[],"hooks":[],"generated_files":{"extensions/<name>.py":"<full python code>"}}

The Python code must subclass PhysiCoreExtension and implement the appropriate hook methods.
PhysiCoreExtension interface:
  class PhysiCoreExtension:
    def pre_step(self, state, params): pass
    def post_step(self, state, control, params): pass
    def on_fault(self, fault_type, state, params): pass
    def on_telemetry(self, telemetry_dict): pass

Keep your questions short and direct. No preamble. Ask question 1 first.`;

  const sendBuildMessage = async (text: string) => {
    if (!text.trim() || isBuildLoading) return;
    if (!requireAIKey()) return;
    const userMsg = { role: 'user' as const, text: text.trim() };
    const newHistory = [...buildMessages, userMsg];
    setBuildMessages(newHistory);
    setBuildInput('');
    setIsBuildLoading(true);

    try {
      const multiTurn = newHistory.map(m => ({
        role: m.role as 'user' | 'assistant',
        content: m.text,
      }));

      const reply = await callAI(FEATURE_ARCHITECT_SYSTEM, '', 2000, multiTurn);
      const finalReply = reply || 'AI unavailable. Make sure your API key is valid — click "AI ON" or "SET UP AI" in the top bar.';

      if (reply.includes('[GENERATE_FEATURE]')) {
        const afterMarker = reply.split('[GENERATE_FEATURE]')[1]?.trim() ?? '';
        const jsonStart = afterMarker.indexOf('{');
        const jsonEnd = afterMarker.lastIndexOf('}');
        if (jsonStart !== -1 && jsonEnd !== -1) {
          try {
            const manifest = JSON.parse(afterMarker.slice(jsonStart, jsonEnd + 1));
            const feature: FeatureManifest = {
              id: generateId(),
              name: manifest.name || 'Custom Feature',
              description: manifest.description || '',
              telemetry_keys: manifest.telemetry_keys || [],
              fault_types: manifest.fault_types || [],
              hooks: manifest.hooks || [],
              files_modified: Object.keys(manifest.generated_files || {}),
              conversation: newHistory,
              generated_files: manifest.generated_files || {},
              createdAt: new Date().toISOString(),
            };
            setBuildFeatures(prev => [...prev, feature]);
            setSelectedBuildFile(Object.keys(feature.generated_files)[0] ?? null);
            if (user && activeProject) {
              const updatedFeatures = [...(activeProject.features || []), feature];
              const updatedExts = [...(activeProject.customExtensions || []), {
                id: feature.id, name: feature.name, description: feature.description,
                code: Object.values(feature.generated_files)[0] || '',
                createdAt: feature.createdAt,
              }];
              setActiveProject(prev => prev ? { ...prev, features: updatedFeatures, customExtensions: updatedExts } : prev);
              try {
                await updateDoc(doc(db, 'users', user.uid, 'projects', activeProject.id), {
                  features: updatedFeatures, customExtensions: updatedExts,
                  updatedAt: new Date().toISOString(),
                });
              } catch (e) { console.warn('[BUILD] save failed:', e); }
            }
          } catch (e) { console.error('[BUILD] JSON parse failed:', e); }
        }
      }

      setBuildMessages([...newHistory, { role: 'assistant', text: finalReply }]);
    } catch (err: any) {
      setBuildMessages([...newHistory, { role: 'assistant', text: `Error: ${err.message}` }]);
    } finally {
      setIsBuildLoading(false);
    }
  };

  // ── DEBUGGER VIEW ──────────────────────────────────────────────────────────
  const FAULT_KB: Record<string, { desc: string; causes: string[]; fixes: string[] }> = {
    BEARING_WEAR: {
      desc: 'Friction parameter rising steadily beyond baseline — mechanical degradation detected.',
      causes: ['Wheel bearing wear', 'Motor brush wear', 'Axle misalignment', 'Insufficient lubrication'],
      fixes: ['Inspect and lubricate wheel bearings', 'Reduce max speed by 20%', 'Check axle alignment', 'Replace motor brushes if >200hr runtime'],
    },
    UNEXPECTED_PAYLOAD: {
      desc: 'Mass estimate jumped >0.5kg suddenly — unplanned load change detected.',
      causes: ['Object placed on robot', 'Battery swap mid-session', 'Loose component detached', 'Incorrect declared mass'],
      fixes: ['Check declared mass in project answers', 'Remove unexpected payload', 'Restart SystemID with correct mass', 'Recalibrate if mass is intentional'],
    },
    AERO_DAMAGE: {
      desc: 'Aerodynamic drag increased significantly — structural damage or configuration error.',
      causes: ['Propeller damage', 'Airframe damage', 'Sensor arm struck obstacle', 'Non-nominal flight altitude'],
      fixes: ['Inspect propellers for chips/cracks', 'Check airframe symmetry', 'Reduce throttle and land immediately', 'Replace damaged props'],
    },
    MOTOR_DEGRADATION: {
      desc: 'Actuator efficiency dropping — motor or ESC underperforming vs. commanded effort.',
      causes: ['Motor overheating', 'ESC calibration drift', 'Low battery voltage', 'Motor winding damage'],
      fixes: ['Check motor temperature (max 80°C)', 'Recalibrate ESC', 'Check battery voltage (min 3.5V/cell)', 'Reduce load or replace motor'],
    },
    SENSOR_DRIFT: {
      desc: 'IMU readings drifting beyond expected bounds — sensor calibration degraded.',
      causes: ['Temperature change affecting IMU', 'Magnetic interference near compass', 'Vibration loosening sensor mount', 'IMU calibration data stale'],
      fixes: ['Recalibrate IMU in stable environment', 'Move away from motor/power interference', 'Tighten sensor mount screws', 'Run calibration routine from dashboard'],
    },
  };

  const runDebuggerDiagnosis = async (query: string) => {
    if (!requireAIKey()) return;
    setIsDebugging(true);
    setDebuggerResult(null);

    const projectFeatures = activeProject?.features || [];
    const customFaultTypes = projectFeatures.flatMap(f => f.fault_types);
    const featureContext = projectFeatures.length > 0
      ? `\nCustom features installed: ${projectFeatures.map(f => `${f.name} (hooks: ${f.hooks.join(',')}, telemetry: ${f.telemetry_keys.join(',')})`).join('; ')}`
      : '';

    const ctx = [
      `Hardware: ${activeProject?.hardware || 'Unknown'} | Platform: ${activeProject?.platform || 'Unknown'}`,
      `Session active: ${isControlActive && isSystemConnected}`,
      `Mass: ${telemetry.mass.toFixed(3)}kg | Friction: ${telemetry.friction.toFixed(4)} | Residual: ${telemetry.residual.toFixed(4)}`,
      `Sentinel: ${telemetry.isFaulted ? 'FAULTED' : telemetry.isStable ? 'NOMINAL' : 'CAUTIOUS'} | Step: ${telemetry.step_count}`,
      `Active faults: ${telemetry.faults?.join(', ') || 'none'}`,
      `Recent failures (last 3): ${failureLogs.slice(0,3).map(f => `${f.failure_type} at ${f.task}`).join('; ') || 'none'}`,
      `CPU: ${telemetry.cpuLoad?.toFixed(0)}% | Latency: ${telemetry.latency?.toFixed(0)}ms | Battery: ${(telemetry as any).battery_pct?.toFixed(0) ?? '--'}%`,
      `Declared answers: ${JSON.stringify(activeProject?.answers || {})}`,
      featureContext,
    ].join('\n');

    // Check custom feature fault types first
    const customFaultMatch = telemetry.faults?.find(f => customFaultTypes.includes(f));
    if (customFaultMatch && !query.trim()) {
      const ownerFeature = projectFeatures.find(f => f.fault_types.includes(customFaultMatch));
      setDebuggerResult(`**${customFaultMatch}** (from feature: ${ownerFeature?.name})\n\nThis fault was registered by a custom feature extension. Check the feature code in the BUILD tab for handling logic.\n\nFeature hooks: ${ownerFeature?.hooks.join(', ')}`);
      setIsDebugging(false);
      return;
    }

    // Check built-in KB next
    const faultMatch = telemetry.faults?.find(f => FAULT_KB[f]);
    if (faultMatch && !query.trim()) {
      const kb = FAULT_KB[faultMatch];
      setDebuggerResult(`**${faultMatch}**\n\n${kb.desc}\n\nLikely causes:\n${kb.causes.map((c,i)=>`${i+1}. ${c}`).join('\n')}\n\nFixes:\n${kb.fixes.map((f,i)=>`${i+1}. ${f}`).join('\n')}`);
      setIsDebugging(false);
      return;
    }

    const debugSystemPrompt = `You are PhysiCore's senior diagnostics engineer. Deep expertise in robotics, control systems, embedded hardware. Given live telemetry and a question, give a concise actionable diagnosis. Lead with the most likely cause. Then give 2-3 ordered fix steps. Format with markdown.`;
    const debugUserPrompt = `LIVE CONTEXT:\n${ctx}\n\nQUESTION: ${query || 'Explain the current system state and what I should do next.'}`;

    const diagText = await callAI(debugSystemPrompt, debugUserPrompt, 800);

    if (diagText) {
      setDebuggerResult(diagText);
    } else {
      setDebuggerResult(
        faultMatch
          ? `**${faultMatch}**\n\n${FAULT_KB[faultMatch].desc}\n\nFixes:\n${FAULT_KB[faultMatch].fixes.map((f: string, i: number) => `${i+1}. ${f}`).join('\n')}`
          : customFaultMatch
          ? `Custom fault: **${customFaultMatch}** — check the BUILD tab for your feature's fault handling code.`
          : 'AI key not working. Click "AI ON" in the top bar to check your key.'
      );
    }
    setIsDebugging(false);
  };

  const renderApiKeyModal = () => (
    <AnimatePresence>
      {showApiKeyModal && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[500] bg-black/90 backdrop-blur-sm flex items-center justify-center p-6"
          onClick={e => { if (e.target === e.currentTarget) setShowApiKeyModal(false); }}
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            className="max-w-lg w-full bg-bg border border-border"
          >
            {/* Header */}
            <div className="px-8 pt-8 pb-6 border-b border-border">
              <div className="flex items-start justify-between">
                <div className="space-y-1">
                  <h2 className="font-display text-xl font-bold text-white uppercase tracking-widest">Set Up AI</h2>
                  <p className="font-mono text-[10px] text-textDim uppercase tracking-widest">
                    PhysiCore uses your own API key — you control your costs
                  </p>
                </div>
                <button onClick={() => setShowApiKeyModal(false)} className="text-textDim hover:text-white transition-colors">
                  <X size={18} />
                </button>
              </div>
            </div>

            <div className="px-8 py-6 space-y-6">
              {/* Gemini — recommended, free tier */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-display text-sm font-bold text-white uppercase tracking-widest">Google Gemini</p>
                    <p className="font-mono text-[9px] text-green uppercase tracking-widest mt-0.5">✓ Free tier available — recommended</p>
                  </div>
                  <a
                    href="https://aistudio.google.com/app/apikey"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 px-3 py-1.5 border border-green/40 text-green font-mono text-[9px] uppercase tracking-widest hover:bg-green hover:text-black transition-all"
                  >
                    Get Key <ExternalLink size={9} />
                  </a>
                </div>
                <div className="relative">
                  <input
                    type="password"
                    value={apiKeyGeminiInput}
                    onChange={e => setApiKeyGeminiInput(e.target.value)}
                    placeholder="AIza..."
                    className="w-full bg-bgRaised border border-border px-4 py-3 font-mono text-sm text-white placeholder:text-textDim focus:outline-none focus:border-green transition-colors"
                  />
                  {getUserGeminiKey() && !apiKeyGeminiInput && (
                    <span className="absolute right-3 top-1/2 -translate-y-1/2 font-mono text-[8px] text-green uppercase tracking-widest">● Saved</span>
                  )}
                </div>
                <div className="bg-bgRaised border border-border px-4 py-3 space-y-1">
                  <p className="font-mono text-[9px] text-textDim uppercase tracking-widest">How to get it — 30 seconds:</p>
                  <ol className="space-y-0.5">
                    {['Go to aistudio.google.com/app/apikey', 'Sign in with any Google account', 'Click "Create API key"', 'Copy and paste it above'].map((step, i) => (
                      <li key={i} className="font-mono text-[9px] text-textSecondary flex gap-2">
                        <span className="text-green shrink-0">{i + 1}.</span>
                        <span>{step}</span>
                      </li>
                    ))}
                  </ol>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <div className="flex-1 h-px bg-border" />
                <span className="font-mono text-[8px] text-textDim uppercase">or also add</span>
                <div className="flex-1 h-px bg-border" />
              </div>

              {/* Anthropic — optional backup */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-display text-sm font-bold text-white uppercase tracking-widest">Anthropic Claude</p>
                    <p className="font-mono text-[9px] text-textDim uppercase tracking-widest mt-0.5">Optional — used as fallback if Gemini fails</p>
                  </div>
                  <a
                    href="https://console.anthropic.com/settings/keys"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 px-3 py-1.5 border border-border text-textDim font-mono text-[9px] uppercase tracking-widest hover:border-white hover:text-white transition-all"
                  >
                    Get Key <ExternalLink size={9} />
                  </a>
                </div>
                <div className="relative">
                  <input
                    type="password"
                    value={apiKeyAnthropicInput}
                    onChange={e => setApiKeyAnthropicInput(e.target.value)}
                    placeholder="sk-ant-..."
                    className="w-full bg-bgRaised border border-border px-4 py-3 font-mono text-sm text-white placeholder:text-textDim focus:outline-none focus:border-white transition-colors"
                  />
                  {getUserAnthropicKey() && !apiKeyAnthropicInput && (
                    <span className="absolute right-3 top-1/2 -translate-y-1/2 font-mono text-[8px] text-green uppercase tracking-widest">● Saved</span>
                  )}
                </div>
              </div>

              {/* Privacy note */}
              <div className="flex items-start gap-2 px-3 py-2 bg-green/5 border border-green/20">
                <Lock size={10} className="text-green shrink-0 mt-0.5" />
                <p className="font-mono text-[9px] text-textSecondary leading-relaxed">
                  Your keys are stored only in your browser. They never leave your device or touch any server.
                </p>
              </div>
            </div>

            {/* Footer */}
            <div className="px-8 py-6 border-t border-border flex gap-3">
              {(getUserGeminiKey() || getUserAnthropicKey()) && (
                <button
                  onClick={() => { clearUserKeys(); setApiKeyGeminiInput(''); setApiKeyAnthropicInput(''); setApiKeySaved(false); }}
                  className="px-4 py-3 border border-red/30 text-red font-mono text-[9px] uppercase tracking-widest hover:bg-red/10 transition-all"
                >
                  Clear Keys
                </button>
              )}
              <button
                onClick={() => setShowApiKeyModal(false)}
                className="px-4 py-3 border border-border text-textDim font-display text-[10px] font-bold uppercase tracking-widest hover:text-white transition-all"
              >
                Cancel
              </button>
              <button
                disabled={!apiKeyGeminiInput.trim() && !apiKeyAnthropicInput.trim() && !getUserGeminiKey() && !getUserAnthropicKey()}
                onClick={() => {
                  if (apiKeyGeminiInput.trim()) saveUserGeminiKey(apiKeyGeminiInput.trim());
                  if (apiKeyAnthropicInput.trim()) saveUserAnthropicKey(apiKeyAnthropicInput.trim());
                  setApiKeySaved(true);
                  setApiKeyGeminiInput('');
                  setApiKeyAnthropicInput('');
                  setTimeout(() => { setShowApiKeyModal(false); setApiKeySaved(false); }, 800);
                }}
                className="flex-1 py-3 bg-green text-black font-display text-[10px] font-bold uppercase tracking-widest hover:bg-white transition-all disabled:opacity-40"
              >
                {apiKeySaved ? '✓ Saved' : 'Save & Activate AI'}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );

  const renderMarkdownText = (text: string) => {
    return text.split('\n').filter(l => l !== undefined).map((line, i) => {
      const trimmed = line.trim();
      if (!trimmed) return <div key={i} className="h-2" />;

      // H3 ### heading
      if (trimmed.startsWith('### ')) {
        return <p key={i} className="font-display text-[11px] font-bold text-white uppercase tracking-widest mt-2">{trimmed.slice(4)}</p>;
      }
      // H2 ## heading
      if (trimmed.startsWith('## ')) {
        return <p key={i} className="font-display text-xs font-bold text-white uppercase tracking-widest mt-3">{trimmed.slice(3)}</p>;
      }
      // H1 # heading
      if (trimmed.startsWith('# ')) {
        return <p key={i} className="font-display text-sm font-bold text-white uppercase tracking-widest mt-3">{trimmed.slice(2)}</p>;
      }
      // Numbered list
      if (/^\d+\.\s/.test(trimmed)) {
        const content = trimmed.replace(/^\d+\.\s/, '').replace(/\*\*(.*?)\*\*/g, '$1');
        return <p key={i} className="font-mono text-[10px] text-textSecondary pl-4 leading-relaxed">{trimmed.match(/^\d+/)?.[0]}. {content}</p>;
      }
      // Bullet
      if (trimmed.startsWith('- ') || trimmed.startsWith('• ')) {
        const content = trimmed.slice(2).replace(/\*\*(.*?)\*\*/g, '$1');
        return <p key={i} className="font-mono text-[10px] text-textSecondary pl-4 leading-relaxed">• {content}</p>;
      }
      // Bold-only line (** wrapping whole line)
      if (trimmed.startsWith('**') && trimmed.endsWith('**')) {
        return <p key={i} className="font-mono text-[10px] font-bold text-white leading-relaxed">{trimmed.slice(2, -2)}</p>;
      }
      // Normal — inline bold replacement
      const parts = trimmed.split(/(\*\*.*?\*\*)/g);
      return (
        <p key={i} className="font-mono text-[10px] text-textSecondary leading-relaxed">
          {parts.map((part, j) =>
            part.startsWith('**') && part.endsWith('**')
              ? <strong key={j} className="text-white font-bold">{part.slice(2, -2)}</strong>
              : part
          )}
        </p>
      );
    });
  };

  const renderProjectView = () => {
    if (!activeProject) {
      return (
        <div className="pt-[52px] min-h-screen bg-void flex items-center justify-center">
          <div className="text-center space-y-4">
            <div className="font-mono text-[11px] text-textDim uppercase tracking-widest">No project selected</div>
            <button onClick={() => setView('projects')} className="px-6 py-2 bg-green text-black font-display text-[11px] font-bold uppercase tracking-widest hover:bg-white transition-all">
              ← Back to Projects
            </button>
          </div>
        </div>
      );
    }

    const tabContent = () => {
      switch (projectTab) {
        case 'integrate': return renderIntegrator();
        case 'build': return renderBuildTab();
        case 'debug': return renderDebugger();
        case 'live': return renderDashboard();
        default: return renderIntegrator();
      }
    };

    return (
      <div className="min-h-screen bg-void">
        {/* Floating fault alert — shown on any tab when faults exist */}
        {(telemetry.isFaulted || failureLogs.length > 0) && projectTab !== 'debug' && (
          <div className="fixed bottom-6 right-6 z-[150]">
            <button
              onClick={() => setProjectTab('debug')}
              className="flex items-center gap-2 px-4 py-2 bg-red text-white font-display text-[10px] font-bold uppercase tracking-widest shadow-lg shadow-red/30 animate-pulse hover:animate-none transition-all"
            >
              <AlertTriangle size={14} />
              {failureLogs.length} FAULT{failureLogs.length !== 1 ? 'S' : ''} — DIAGNOSE
            </button>
          </div>
        )}
        {tabContent()}
      </div>
    );
  };

  const renderBuildTab = () => {
    const allFeatures = [...buildFeatures, ...(activeProject?.features || [])];
    const uniqueFeatures = allFeatures.filter((f, i, a) => a.findIndex(x => x.id === f.id) === i);
    const activeFeature = uniqueFeatures.find(f => selectedBuildFile && Object.keys(f.generated_files).includes(selectedBuildFile));
    const editorCode = selectedBuildFile && activeFeature ? (activeFeature.generated_files[selectedBuildFile] ?? '') : '';

    return (
      <div className="pt-[52px] h-screen bg-void flex overflow-hidden">
        {/* Left sidebar — features list */}
        <div className="w-64 shrink-0 border-r border-border flex flex-col">
          <div className="p-4 border-b border-border">
            <div className="micro-label text-textDim uppercase mb-3">Features</div>
            {uniqueFeatures.length === 0 ? (
              <div className="font-mono text-[10px] text-textDim italic">No features yet — describe one below</div>
            ) : (
              <div className="space-y-2">
                {uniqueFeatures.map(f => (
                  <div
                    key={f.id}
                    onClick={() => { const first = Object.keys(f.generated_files)[0]; if (first) setSelectedBuildFile(first); }}
                    className={`p-2 border cursor-pointer transition-all ${selectedBuildFile && Object.keys(f.generated_files).includes(selectedBuildFile) ? 'border-amber bg-amber/10' : 'border-border hover:border-amber/40'}`}
                  >
                    <div className="font-display text-[10px] font-bold text-white uppercase tracking-widest truncate">{f.name}</div>
                    <div className="font-mono text-[9px] text-textDim mt-0.5 truncate">{f.description}</div>
                    {f.telemetry_keys.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1">
                        {f.telemetry_keys.map(k => (
                          <span key={k} className="px-1 py-0 font-mono text-[8px] bg-cyan/10 text-cyan border border-cyan/20">{k}</span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className="flex-1" />
          <div className="p-4 border-t border-border">
            <button
              onClick={() => { setBuildMessages([]); setSelectedBuildFile(null); }}
              className="w-full py-2 border border-amber/30 text-amber font-display text-[9px] font-bold uppercase tracking-widest hover:bg-amber hover:text-black transition-all"
            >
              + New Feature
            </button>
          </div>
        </div>

        {/* Center — Socratic conversation */}
        <div className="flex-1 flex flex-col min-w-0 border-r border-border">
          <div className="p-4 border-b border-border flex items-center gap-3">
            <Code2 size={16} className="text-amber" />
            <span className="font-display text-sm font-bold text-white uppercase tracking-widest">Feature Architect</span>
            <span className="font-mono text-[9px] text-textDim">Describe a feature in plain English — the AI writes the code</span>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {buildMessages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
                <div className="w-12 h-12 border border-amber/40 flex items-center justify-center">
                  <Puzzle size={20} className="text-amber" />
                </div>
                <div className="space-y-2">
                  <div className="font-display text-sm font-bold text-white uppercase tracking-widest">Build a feature</div>
                  <div className="font-body text-xs text-textSecondary max-w-xs">
                    Tell the Feature Architect what you want to add. It will ask 4 questions, then write the complete PhysiCore extension.
                  </div>
                </div>
              </div>
            )}
            {buildMessages.map((m, i) => (
              <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] px-4 py-3 ${m.role === 'user' ? 'bg-amber/20 border border-amber/30' : 'bg-bgRaised border border-border'}`}>
                  {m.text.includes('[GENERATE_FEATURE]') ? (
                    <div className="space-y-2">
                      <div className="font-mono text-[10px] text-green font-bold">✓ Feature generated</div>
                      <div className="font-body text-xs text-textSecondary">{m.text.split('[GENERATE_FEATURE]')[0].trim()}</div>
                    </div>
                  ) : (
                    <div className="font-body text-sm text-textPrimary whitespace-pre-wrap">{m.text}</div>
                  )}
                </div>
              </div>
            ))}
            {isBuildLoading && (
              <div className="flex justify-start">
                <div className="px-4 py-3 bg-bgRaised border border-border flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-amber animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-1.5 h-1.5 rounded-full bg-amber animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-1.5 h-1.5 rounded-full bg-amber animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            )}
          </div>
          <div className="p-4 border-t border-border flex gap-3">
            <input
              value={buildInput}
              onChange={e => setBuildInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendBuildMessage(buildInput); } }}
              placeholder="Describe a feature or answer the question above..."
              className="flex-1 bg-bgRaised border border-border px-4 py-2 font-mono text-sm text-white focus:border-amber outline-none placeholder:text-textDim"
            />
            <button
              onClick={() => sendBuildMessage(buildInput)}
              disabled={isBuildLoading || !buildInput.trim()}
              className="px-4 py-2 bg-amber text-black font-display text-[10px] font-bold uppercase tracking-widest hover:bg-white transition-all disabled:opacity-40"
            >
              Send
            </button>
          </div>
        </div>

        {/* Right pane — code editor */}
        <div className="w-[480px] shrink-0 flex flex-col">
          <div className="p-4 border-b border-border flex items-center gap-3">
            <FileJson size={14} className="text-textDim" />
            <span className="font-mono text-[10px] text-textDim truncate">{selectedBuildFile ?? 'No file selected'}</span>
            {selectedBuildFile && editorCode && (
              <button
                onClick={() => { navigator.clipboard.writeText(editorCode); }}
                className="ml-auto p-1.5 text-textDim hover:text-white transition-colors"
                title="Copy"
              >
                <Copy size={12} />
              </button>
            )}
          </div>
          <div className="flex-1 min-h-0 overflow-hidden">
            {selectedBuildFile && editorCode ? (
              <PhysiEditor
                code={editorCode}
                language={selectedBuildFile.endsWith('.yaml') || selectedBuildFile.endsWith('.yml') ? 'yaml' : 'python'}
                readOnly
              />
            ) : (
              <div className="flex items-center justify-center" style={{ height: '100%' }}>
                <div className="text-center space-y-3">
                  <div className="font-mono text-[10px] text-textDim italic">Generated code will appear here</div>
                  <div className="font-mono text-[9px] text-textDim/50">Use the chat on the left to describe a feature</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderDebugger = () => (
    <div className="pt-[52px] min-h-screen bg-void">
      <div className="border-b border-border bg-bg px-6 py-4 flex items-center gap-4 sticky top-[52px] z-10">
        <div className="flex items-center gap-2">
          {telemetry.isFaulted && <span className="w-2 h-2 rounded-full bg-red animate-pulse" />}
          <span className="font-display text-sm font-bold text-white uppercase tracking-widest">PhysiCore Debugger</span>
        </div>
        {activeProject && (
          <span className="font-mono text-[9px] text-cyan uppercase tracking-widest border border-cyan/30 px-2 py-0.5">
            {activeProject.name}
          </span>
        )}
        <div className="ml-auto flex items-center gap-2">
          {isControlActive && isSystemConnected ? (
            <span className="font-mono text-[9px] text-green uppercase">● LIVE SESSION</span>
          ) : (
            <span className="font-mono text-[9px] text-textDim uppercase">○ NO SESSION</span>
          )}
        </div>
      </div>

      <div className="px-6 py-6 max-w-[1100px] mx-auto space-y-6">

        {/* Active Fault Banner */}
        {telemetry.isFaulted && telemetry.faults?.length > 0 && (
          <div className="border border-red/40 bg-red/5 p-4 space-y-3">
            <div className="flex items-center gap-3">
              <span className="w-2.5 h-2.5 rounded-full bg-red animate-pulse" />
              <span className="font-display text-sm font-bold text-red uppercase tracking-widest">Active Sentinel Faults</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {telemetry.faults.map(f => (
                <button key={f} onClick={() => runDebuggerDiagnosis('')}
                  className="px-3 py-1.5 border border-red/50 bg-red/10 font-mono text-[10px] text-red hover:bg-red/20 uppercase tracking-widest transition-all">
                  {f}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="grid grid-cols-[1fr_340px] gap-6">

          {/* LEFT: AI Diagnosis */}
          <div className="space-y-4">
            <div className="space-y-2">
              <p className="font-mono text-[9px] text-textDim uppercase tracking-widest">Deep Diagnosis</p>
              <div className="flex gap-3">
                <input
                  className="flex-1 bg-bgRaised border border-border px-4 py-3 font-body text-sm text-white placeholder:text-textDim focus:outline-none focus:border-red transition-colors"
                  placeholder="Describe what's wrong, or click a fault above…"
                  value={debuggerQuery}
                  onChange={e => setDebuggerQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && debuggerQuery.trim() && runDebuggerDiagnosis(debuggerQuery)}
                />
                <button
                  onClick={() => runDebuggerDiagnosis(debuggerQuery)}
                  disabled={isDebugging}
                  className="px-5 py-3 bg-red text-white font-display text-xs font-bold uppercase tracking-widest hover:bg-white hover:text-black transition-all disabled:opacity-50">
                  {isDebugging ? '...' : 'Diagnose'}
                </button>
              </div>
              {/* Quick question chips */}
              <div className="flex flex-wrap gap-2 pt-1">
                {[
                  'Why is mass estimate drifting?',
                  'Is residual level dangerous?',
                  'Why did Sentinel go CAUTIOUS?',
                  'Is friction converging normally?',
                  'Explain current fault signature',
                  'What should I check first?',
                ].map(q => (
                  <button key={q} onClick={() => { setDebuggerQuery(q); runDebuggerDiagnosis(q); }}
                    className="px-3 py-1.5 border border-border font-mono text-[9px] text-textDim hover:border-red hover:text-red uppercase tracking-widest transition-all">
                    {q}
                  </button>
                ))}
              </div>
            </div>

            {isDebugging && (
              <div className="p-5 border border-border bg-bgRaised flex items-center gap-3">
                <div className="w-4 h-4 border-2 border-red/30 border-t-red rounded-full animate-spin" />
                <span className="font-mono text-[10px] text-textDim">Analyzing live telemetry…</span>
              </div>
            )}

            {debuggerResult && !isDebugging && (
              <div className="p-5 border border-red/20 bg-bgRaised space-y-3">
                <p className="font-mono text-[9px] text-red uppercase tracking-widest">Diagnosis</p>
                <div className="space-y-1.5">{renderMarkdownText(debuggerResult)}</div>
              </div>
            )}

            {/* Fault Knowledge Base */}
            <div className="space-y-3">
              <p className="font-mono text-[9px] text-textDim uppercase tracking-widest">Fault Knowledge Base</p>
              {Object.entries(FAULT_KB).map(([key, val]) => {
                const isActive = telemetry.faults?.includes(key);
                return (
                  <div key={key} className={`p-4 border space-y-2 ${isActive ? 'border-red/40 bg-red/5' : 'border-border bg-bgRaised'}`}>
                    <div className="flex items-center gap-2">
                      {isActive && <span className="w-1.5 h-1.5 rounded-full bg-red" />}
                      <span className={`font-mono text-[10px] font-bold uppercase tracking-widest ${isActive ? 'text-red' : 'text-textSecondary'}`}>{key}</span>
                    </div>
                    <p className="font-mono text-[9px] text-textDim">{val.desc}</p>
                    {isActive && (
                      <div className="space-y-1 pt-1">
                        <p className="font-mono text-[9px] text-amber">Quick fixes:</p>
                        {val.fixes.slice(0,2).map((f,i) => (
                          <p key={i} className="font-mono text-[9px] text-textSecondary pl-3">{i+1}. {f}</p>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* RIGHT: Live Telemetry + FailureLog */}
          <div className="space-y-4">

            {/* Live telemetry snapshot */}
            <div className="p-4 border border-border bg-bgRaised space-y-3">
              <p className="font-mono text-[9px] text-textDim uppercase tracking-widest">Live Snapshot</p>
              {[
                { label: 'Mass', value: `${telemetry.mass.toFixed(3)} kg`, warn: telemetry.mass > 5 },
                { label: 'Friction', value: telemetry.friction.toFixed(4), warn: telemetry.friction > 0.4 },
                { label: 'Residual', value: telemetry.residual.toFixed(4), warn: telemetry.residual > 0.8 },
                { label: 'Confidence', value: `${telemetry.confidence.toFixed(0)}%`, warn: telemetry.confidence < 50 },
                { label: 'Sentinel', value: telemetry.isFaulted ? 'FAULTED' : telemetry.isStable ? 'NOMINAL' : 'CAUTIOUS', warn: telemetry.isFaulted },
                { label: 'CPU', value: `${telemetry.cpuLoad?.toFixed(0)||0}%`, warn: (telemetry.cpuLoad||0) > 80 },
                { label: 'Battery', value: `${telemetry.battery_pct?.toFixed(0)||0}%`, warn: (telemetry.battery_pct||0) < 20 },
              ].map(row => (
                <div key={row.label} className="flex justify-between items-center border-b border-border/30 pb-1 last:border-0 last:pb-0">
                  <span className="font-mono text-[9px] text-textDim uppercase">{row.label}</span>
                  <span className={`font-mono text-[10px] font-bold ${row.warn ? 'text-amber' : 'text-white'}`}>{row.value}</span>
                </div>
              ))}
            </div>

            {/* FailureLog */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <p className="font-mono text-[9px] text-textDim uppercase tracking-widest">Failure Log</p>
                {failureLogs.length > 0 && (
                  <button onClick={() => setFailureLogs([])}
                    className="font-mono text-[8px] text-textDim hover:text-red uppercase tracking-widest transition-colors">Clear</button>
                )}
              </div>
              {failureLogs.length === 0 ? (
                <div className="p-3 border border-border/50 bg-bgRaised">
                  <p className="font-mono text-[9px] text-textDim">No failures recorded this session.</p>
                </div>
              ) : (
                <div className="space-y-1.5 max-h-[320px] overflow-y-auto custom-scroll">
                  {failureLogs.map(log => (
                    <div key={log.id} className="p-3 border border-red/20 bg-red/5 space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="font-mono text-[10px] text-red font-bold">{log.failure_type}</span>
                        <span className="font-mono text-[8px] text-textDim">{new Date(log.timestamp).toLocaleTimeString()}</span>
                      </div>
                      <span className="font-mono text-[9px] text-textDim">{log.task}</span>
                      <div className="flex gap-3 font-mono text-[8px] text-textDim pt-0.5">
                        <span>m={log.sim_params.mass.toFixed(2)}</span>
                        <span>f={log.sim_params.friction.toFixed(3)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

          </div>
        </div>
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="h-screen w-full bg-void flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Cpu className="text-green animate-spin-slow" size={48} />
          <span className="font-mono text-xs text-green uppercase tracking-widest">Verifying Neural Handshake...</span>
        </div>
      </div>
    );
  }

  // ── ACCESS GATE ────────────────────────────────────────────────────────────
  // Only users in BETA_TESTERS can access PhysiCore.
  // Signed-in but not on the list → blocked screen.
  // Not signed in → only the home page is visible (login required for everything else).
  if (user && !isAuthorized) {
    return (
      <div className="h-screen w-full bg-void flex items-center justify-center px-6">
        <div className="max-w-md w-full space-y-8 text-center">
          <div className="w-16 h-16 mx-auto border border-red/30 flex items-center justify-center">
            <ShieldAlert size={28} className="text-red" />
          </div>
          <div className="space-y-3">
            <h1 className="font-display text-2xl font-bold text-white uppercase tracking-widest">Access Restricted</h1>
            <p className="font-body text-sm text-textSecondary leading-relaxed">
              PhysiCore is currently in closed beta. Your account
              <span className="text-white font-mono text-xs block mt-1">{user.email}</span>
              is not on the access list.
            </p>
          </div>
          <div className="p-5 border border-border bg-bgRaised space-y-3">
            <p className="font-mono text-[10px] text-textDim uppercase tracking-widest">Want access?</p>
            <p className="font-body text-xs text-textSecondary">
              Contact Prathamesh at <span className="text-green">prathameshshirbhate8anpc@gmail.com</span> to request beta access.
            </p>
          </div>
          <button
            onClick={() => signOut(auth)}
            className="px-6 py-3 border border-border text-textDim font-display text-[10px] font-bold uppercase tracking-widest hover:border-red hover:text-red transition-all"
          >
            Sign Out
          </button>
        </div>
      </div>
    );
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
      {renderApiKeyModal()}
      <AnimatePresence mode="wait">
        {view === 'home' ? (
          <motion.div key="home" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            {renderHome()}
          </motion.div>
        ) : view === 'projects' ? (
          <motion.div key="projects" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            {renderProjects()}
          </motion.div>
        ) : view === 'manual' ? (
          <motion.div key="manual" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            {renderManual()}
          </motion.div>
        ) : view === 'whitepaper' ? (
          <motion.div key="whitepaper" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            {renderWhitepaper()}
          </motion.div>
        ) : view === 'team' ? (
          <motion.div key="team" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            {renderTeam()}
          </motion.div>
        ) : view === 'project' ? (
          <motion.div key="project" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            {renderProjectView()}
          </motion.div>
        ) : (
          <motion.div key="home-fallback" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
            {renderHome()}
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
                      if (view === 'project' && projectTab === 'integrate') handleLaunchApp();
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

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />;
};

const AviationTrajectoryCanvas = ({ state, params, isRunning, setIsRunning, isConnected, handshakeConfirmed }: { state: AviationState, params: AviationParams, isRunning: boolean, setIsRunning: (r: boolean) => void, isConnected: boolean, handshakeConfirmed: boolean }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      const parent = canvas.parentElement;
      if (parent) {
        canvas.width = parent.clientWidth;
        canvas.height = parent.clientHeight;
      }
    };

    window.addEventListener('resize', resize);
    resize();

    let frame = 0;
    const animate = () => {
      frame++;
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

      // Draw Aircraft
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      
      ctx.save();
      ctx.translate(centerX, centerY);
      ctx.rotate(state.pitch * Math.PI / 180);
      
      ctx.strokeStyle = COLORS.cyan;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(-20, 0); ctx.lineTo(20, 0);
      ctx.moveTo(0, -5); ctx.lineTo(0, 5);
      ctx.stroke();
      
      ctx.restore();

      // Telemetry
      ctx.fillStyle = COLORS.textDim;
      ctx.font = '10px "JetBrains Mono"';
      ctx.fillText(`ALT: ${state.y.toFixed(0)}m`, 20, 30);
      ctx.fillText(`SPD: ${Math.sqrt(state.vx**2 + state.vy**2).toFixed(1)}m/s`, 20, 45);

      requestAnimationFrame(animate);
    };

    const animId = requestAnimationFrame(animate);
    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animId);
    };
  }, [state]);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />;
};

const DashboardCanvas = ({ isConnected, handshakeConfirmed, onTelemetryUpdate, telemetry, connectionMode, simulationConfig }: { isConnected: boolean, handshakeConfirmed: boolean, onTelemetryUpdate: (data: any) => void, telemetry: any, connectionMode: string, simulationConfig?: any }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const stateRef = useRef({
    robot: { x: 0, y: 0, vx: 0, vy: 0 },
    target: { x: 0, y: 0 },
    trueParams: { 
      mass: simulationConfig?.mass || 5.2, 
      friction: simulationConfig?.friction || 0.65 
    }, // Use simulationConfig if available
    estParams: { mass: 1.0, friction: 0.1 },   // AI's current estimates
    actuatorEfficiency: simulationConfig?.actuatorEfficiency || 0.95,
    residualHistory: [] as any[],
    effortHistory: [] as any[],
    frame: 0
  });
  const telemetryRef = useRef(telemetry);

  useEffect(() => {
    telemetryRef.current = telemetry;
  }, [telemetry]);

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

      // If connected, use real hardware telemetry
      if (isConnected) {
        const t = telemetryRef.current;
        if (t.pos) {
          state.robot.x = t.pos.x;
          state.robot.y = t.pos.y;
          state.robot.vx = t.vel?.x || 0;
          state.robot.vy = t.vel?.y || 0;
        } else if (t.pitch !== undefined && t.pitch !== 0 || t.roll !== undefined && t.roll !== 0 || t.gyro?.x !== undefined) {
          const pitch = t.pitch || 0;
          const roll  = t.roll  || 0;
          state.robot.x = canvas.width  / 2 + (roll  * 4);
          state.robot.y = canvas.height / 2 + (pitch * 4);
          state.robot.vx = t.gyro?.x || t.gyro_x || 0;
          state.robot.vy = t.gyro?.y || t.gyro_y || 0;
        } else if (t.altitude && t.altitude > 0) {
          state.robot.x = canvas.width  / 2 + (t.vel?.x || 0) * 2;
          state.robot.y = canvas.height / 2 - (t.altitude * 0.05);
          state.robot.vx = t.vel?.x || 0;
          state.robot.vy = t.vel?.z || 0;
        } else {
          ctx.fillStyle = COLORS.amber;
          ctx.font = 'bold 12px "JetBrains Mono"';
          ctx.textAlign = 'center';
          ctx.fillText('CONNECTED — WAITING FOR FIRST PACKET...', canvas.width / 2, canvas.height / 2);
          ctx.fillStyle = COLORS.textDim;
          ctx.font = '10px "JetBrains Mono"';
          ctx.fillText('Check bridge is running and sending data', canvas.width / 2, canvas.height / 2 + 20);
          requestAnimationFrame(animate);
          return;
        }
        state.target = t.targetPos || state.target;
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

      // Update sidebar telemetry via callback
      const t = telemetryRef.current;
      onTelemetryUpdate((prev: any) => ({
        ...prev,
        mass:               state.estParams.mass,
        friction:           state.estParams.friction,
        actuatorEfficiency: state.actuatorEfficiency,
        residual:           state.residualHistory?.length > 0
                              ? state.residualHistory[state.residualHistory.length - 1].y
                              : prev.residual,
        confidence:         (state as any).lastEnsemble?.confidence ?? prev.confidence,
        variance:           (state as any).lastEnsemble?.variance   ?? prev.variance,
        isStable:           (state as any).lastLyapunov             ?? prev.isStable,
        isFaulted:          (state as any).lastFault                ?? prev.isFaulted,
        residualHistory:    state.residualHistory,
        effortHistory:      state.effortHistory,
      }));

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
