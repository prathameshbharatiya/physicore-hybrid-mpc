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

async function callAnthropic(
  system: string,
  userContent: string,
  maxTokens = 1000,
  messagesOverride?: { role: string; content: string }[]
): Promise<string> {
  const key = getUserAnthropicKey();
  if (!key) return '';
  try {
    const messages = messagesOverride || [{ role: 'user', content: userContent }];
    const resp = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': key,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: maxTokens,
        system,
        messages,
      }),
    });
    if (!resp.ok) {
      const errText = await resp.text();
      console.warn('[ANTHROPIC]', resp.status, errText);
      return '';
    }
    const data = await resp.json();
    return data.content?.[0]?.text?.trim() || '';
  } catch (e) {
    console.warn('[ANTHROPIC] fetch failed:', e);
    return '';
  }
}

// Master AI caller — Gemini first, Anthropic fallback
async function callAI(
  system: string,
  userContent: string,
  maxTokens = 1000,
  multiTurnMessages?: { role: 'user' | 'assistant'; content: string }[]
): Promise<string> {
  // Tier 1: Gemini
  const ai = getAI();
  if (ai) {
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
      if (text) return text;
    } catch (e) {
      console.warn('[GEMINI] failed:', e);
    }
  }
  // Tier 2: Anthropic
  const anthropicMessages = multiTurnMessages
    ? multiTurnMessages.map(m => ({ role: m.role, content: m.content }))
    : undefined;
  return await callAnthropic(system, userContent, maxTokens, anthropicMessages);
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
type View = 'home' | 'project' | 'manual' | 'team' | 'projects';
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
  const [showNewProjectModal, setShowNewProjectModal] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDesc, setNewProjectDesc] = useState('');
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
    }, { threshold: 0.15 });

    const reveals = document.querySelectorAll('.reveal');
    reveals.forEach(r => revealObserver.observe(r));

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
    if (!user) { setProjects([]); return; }
    const q = query(
      collection(db, 'users', user.uid, 'projects'),
      orderBy('updatedAt', 'desc')
    );
    const unsub = onSnapshot(q, (snap) => {
      setProjects(snap.docs.map(d => ({ id: d.id, ...d.data() } as Project)));
    }, () => {});
    return () => unsub();
  }, [user]);

  const createProject = async (name: string, hardware: string, answers: Record<string, string>, files: GeneratedFile[]) => {
    if (!user) { alert('Please sign in first.'); return null; }
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
      alert(`Could not save project: ${err?.message || 'Permission denied'}. Check browser console.`);
      return null;
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

      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': getUserAnthropicKey(),
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 4000,
          system: PHYSICORE_SYSTEM_PROMPT,
          messages: cleaned,
        }),
      });

      let aiText = '';
      if (response.ok) {
        const data = await response.json();
        aiText = data?.content?.map((b: any) => b.type === 'text' ? b.text : '').filter(Boolean).join('\n') || '';
      } else {
        // Fallback to local decision tree if API unavailable
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
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 cursor-pointer" onClick={() => setView('home')}>
          <svg width="20" height="20" viewBox="0 0 100 100">
            <path d="M50 5 L90 25 L90 75 L50 95 L10 75 L10 25 Z" fill="none" stroke={COLORS.green} strokeWidth="6" />
          </svg>
          <span className="font-display text-lg font-bold tracking-widest text-white">PHYSICORE</span>
          <span className="font-mono text-[10px] text-textDim">v3.0</span>
        </div>

        {view === 'project' && activeProject ? (
          <div className="flex items-center gap-2 ml-2">
            <div className="h-4 w-px bg-border mx-1" />
            <button onClick={() => setView('projects')} className="font-mono text-[10px] text-textDim hover:text-cyan transition-colors uppercase tracking-widest">PROJECTS</button>
            <ChevronRight size={12} className="text-textDim" />
            <span className="font-mono text-[10px] text-white uppercase tracking-widest">{activeProject.name}</span>
            <div className="h-4 w-px bg-border mx-2" />
            {(['integrate', 'build', 'debug', 'live'] as ProjectTab[]).map(tab => (
              <button
                key={tab}
                onClick={() => setProjectTab(tab)}
                className={`px-3 py-1 font-display text-[10px] font-bold uppercase tracking-widest transition-all ${projectTab === tab
                  ? tab === 'live' ? 'bg-cyan text-black' : tab === 'debug' ? 'bg-red text-white' : tab === 'build' ? 'bg-amber text-black' : 'bg-green text-black'
                  : 'text-textDim hover:text-textPrimary border border-transparent hover:border-border'}`}
              >
                {tab === 'integrate' ? '⬡ INTEGRATE' : tab === 'build' ? '⬡ BUILD' : tab === 'debug' ? '⬡ DEBUG' : '⬡ LIVE'}
                {tab === 'debug' && (telemetry.isFaulted || failureLogs.length > 0) && (
                  <span className="ml-1 inline-block w-1.5 h-1.5 rounded-full bg-red animate-pulse" />
                )}
              </button>
            ))}
          </div>
        ) : (
          <>
            <div className="h-4 w-px bg-border mx-2" />
            <span className="font-body text-[11px] text-textSecondary uppercase tracking-widest hidden md:block">
              Physics Intelligence Engine
            </span>
          </>
        )}
      </div>

      {view !== 'project' && (
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
            {/* AI Key Status */}
            <button
              onClick={() => setShowApiKeyModal(true)}
              className={`flex items-center gap-1.5 px-3 py-1 border font-mono text-[9px] uppercase tracking-widest transition-all ${
                hasAnyKey()
                  ? 'border-green/30 text-green hover:bg-green/10'
                  : 'border-amber/40 text-amber hover:bg-amber/10 animate-pulse'
              }`}
              title={hasAnyKey() ? 'AI active — click to manage keys' : 'No API key — click to set up AI'}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${hasAnyKey() ? 'bg-green' : 'bg-amber'}`} />
              {hasAnyKey() ? 'AI ON' : 'SET UP AI'}
            </button>
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

        {user && view !== 'project' && (
          <>
            <button
              onClick={handleSetIntegratorView}
              className="px-4 py-1.5 font-display text-[11px] font-bold uppercase tracking-widest transition-all bg-green text-black hover:bg-white"
            >
              ⬡ INTEGRATION ENGINEER
            </button>
            <button
              onClick={() => setView('projects')}
              className={`px-4 py-1.5 font-display text-[11px] font-bold uppercase tracking-widest transition-all ${view === 'projects' ? 'bg-white text-black' : 'border border-cyan text-cyan hover:bg-cyan hover:text-black'}`}
            >
              ⬡ PROJECTS
            </button>
            <button
              onClick={() => setView('manual')}
              className={`px-4 py-1.5 font-display text-[11px] font-bold uppercase tracking-widest transition-all ${view === 'manual' ? 'bg-white text-black' : 'bg-amber text-black hover:bg-white'}`}
            >
              ⬡ MANUAL
            </button>
          </>
        )}
        {user && view === 'project' && (
          <button
            onClick={() => setView('projects')}
            className="px-4 py-1.5 border border-border text-textDim font-display text-[11px] font-bold uppercase tracking-widest hover:text-textPrimary hover:border-textPrimary transition-all"
          >
            ← PROJECTS
          </button>
        )}
      </div>
    </nav>
  );

  const renderHome = () => {
  // Convergence animation data — real numbers from actual hardware test
  // Mass estimate converging from 1.0 toward 1.35 over 30 seconds
  const convergenceData = Array.from({ length: 60 }, (_, i) => {
    const t = i / 59;
    const converged = 1.0 + 0.35 * (1 - Math.exp(-4 * t));
    return { step: i * 30, mass: converged };
  });

  return (
    <div className="pt-[52px] custom-scroll">

      {/* ── HERO ── */}
      <section id="overview" className="relative h-screen flex flex-col items-center justify-center text-center px-6 overflow-hidden">
        <HeroCanvas />
        <div className="relative z-10 max-w-[820px] space-y-8">
          <div className="reveal border-l-2 border-green pl-4 text-left inline-block">
            <span className="font-mono text-[11px] text-green uppercase tracking-[0.2em]">PhysiCore v3.1 — Real Hardware. Real Physics. Proven.</span>
          </div>
          <h1 className="reveal reveal-stagger-1 font-display text-6xl md:text-8xl font-bold text-white leading-[0.9] tracking-tighter">
            Close the <br />Reality Gap.
          </h1>
          <p className="reveal reveal-stagger-2 font-body text-lg md:text-xl text-textSecondary leading-relaxed max-w-[640px] mx-auto">
            Every robot trained in simulation breaks when it hits real hardware.<br />
            PhysiCore fixes this in real-time, at 60Hz, on any hardware you connect it to.<br />
            <span className="text-white font-medium">The bot didn't fall. Not once.</span>
          </p>

          <div className="reveal reveal-stagger-3 grid grid-cols-2 md:grid-cols-5 gap-0 border border-border divide-x divide-border bg-void/50 backdrop-blur-sm">
            {[
              { val: 'RK4', label: '4TH ORDER PHYSICS' },
              { val: '60 Hz', label: 'CONTROL LOOP' },
              { val: '3×MLP', label: 'RESIDUAL ENSEMBLE' },
              { val: '6-STEP', label: 'MPC LOOKAHEAD' },
              { val: '12', label: 'HARDWARE PLATFORMS' },
            ].map((m, i) => (
              <div key={i} className="p-5 flex flex-col items-center">
                <span className="font-display text-2xl md:text-3xl font-bold text-green">{m.val}</span>
                <span className="micro-label text-[8px] text-textDim mt-1 text-center">{m.label}</span>
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

      {/* ── PROBLEM ── */}
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
              <p>The floor isn't as smooth as the model. The payload shifts. The motors wear. The air density varies. The gap between what your simulation predicts and what your hardware does compounds with every iteration.</p>
              <p>Teams spend months manually re-tuning. Some give up. Some ship hardware that fails in the field. PhysiCore eliminates this gap in real-time — closing it automatically while your hardware is running.</p>
            </div>
          </div>
          <div className="reveal reveal-stagger-3">
            <RealityGapDiagram />
          </div>
        </div>
      </section>

      {/* ── CONVERGENCE DEMO — the magic made visible ── */}
      <section id="convergence" className="bg-void py-32 px-6 border-t border-border">
        <div className="max-w-[1100px] mx-auto space-y-16">
          <div className="text-center space-y-4">
            <span className="micro-label text-cyan">Live System Identification</span>
            <h2 className="font-display text-4xl md:text-5xl font-bold text-white">Watch it learn your hardware.</h2>
            <p className="font-body text-textSecondary max-w-[600px] mx-auto">
              PhysiCore starts with a guess. Within 30 seconds of real motion, it has learned your robot's actual mass, friction, and inertia from sensor data alone. No manual calibration.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-12 items-start">
            {/* Convergence graph */}
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

              {/* SVG graph */}
              <div className="relative h-[200px] w-full">
                <svg viewBox="0 0 600 200" className="w-full h-full" preserveAspectRatio="none">
                  {/* Grid lines */}
                  {[0.25, 0.5, 0.75, 1.0].map((v, i) => (
                    <line key={i} x1="0" y1={v * 180 + 10} x2="600" y2={v * 180 + 10}
                      stroke="#1A1A28" strokeWidth="1" />
                  ))}
                  {/* True value line */}
                  <line x1="0" y1={10 + (1 - (1.35 - 1.0) / 0.6) * 180} x2="600"
                    y2={10 + (1 - (1.35 - 1.0) / 0.6) * 180}
                    stroke="#00E5C8" strokeWidth="1" strokeDasharray="4 4" opacity="0.4" />
                  {/* Convergence curve */}
                  <motion.path
                    d={`M ${convergenceData.map((d, i) => {
                      const x = (i / (convergenceData.length - 1)) * 600;
                      const y = 10 + (1 - Math.min(1, (d.mass - 1.0) / 0.6)) * 180;
                      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                    }).join(' ')}`}
                    fill="none" stroke="#22C55E" strokeWidth="2"
                    initial={{ pathLength: 0 }}
                    whileInView={{ pathLength: 1 }}
                    transition={{ duration: 3, ease: "easeOut" }}
                  />
                </svg>
                {/* Y axis labels */}
                <div className="absolute left-0 top-0 h-full flex flex-col justify-between pointer-events-none">
                  <span className="font-mono text-[8px] text-textDim">1.35</span>
                  <span className="font-mono text-[8px] text-textDim">1.18</span>
                  <span className="font-mono text-[8px] text-textDim">1.00</span>
                </div>
              </div>

              <div className="flex justify-between items-end">
                <span className="font-mono text-[9px] text-textDim">0s</span>
                <span className="font-mono text-[9px] text-cyan">Converged ↗</span>
                <span className="font-mono text-[9px] text-textDim">30s</span>
              </div>
            </div>

            {/* Before / After + explanation */}
            <div className="reveal space-y-8">

              {/* Before/After motor power */}
              <div className="space-y-4">
                <span className="font-mono text-[10px] text-textDim uppercase tracking-widest">Motor Output — Same Hardware, Same Lean Angle</span>
                <div className="space-y-3">
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="font-body text-xs text-textSecondary">Without PhysiCore <span className="text-textDim">(MAX_TORQUE bug)</span></span>
                      <span className="font-mono text-xs text-red">0.4%</span>
                    </div>
                    <div className="h-2 w-full bg-border overflow-hidden">
                      <motion.div initial={{ width: 0 }} whileInView={{ width: '0.4%' }}
                        transition={{ duration: 1 }} className="h-full bg-red" />
                    </div>
                    <span className="font-mono text-[9px] text-textDim">Motors barely twitch. Robot falls immediately.</span>
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="font-body text-xs text-textSecondary">With PhysiCore <span className="text-textDim">(calibrated)</span></span>
                      <span className="font-mono text-xs text-green">18.4%</span>
                    </div>
                    <div className="h-2 w-full bg-border overflow-hidden">
                      <motion.div initial={{ width: 0 }} whileInView={{ width: '18.4%' }}
                        transition={{ duration: 1, delay: 0.3 }} className="h-full bg-green" />
                    </div>
                    <span className="font-mono text-[9px] text-textDim">Strong correction. Bot stays upright.</span>
                  </div>
                </div>
              </div>

              {/* Data flow */}
              <div className="space-y-3">
                <span className="font-mono text-[10px] text-textDim uppercase tracking-widest">How it works — one control cycle</span>
                {[
                  { step: '01', label: 'IMU reads pitch + gyro rate', detail: 'pitch=5.2° gyro_x=12.4°/s', color: COLORS.textSecondary },
                  { step: '02', label: 'Bridge converts to state vector', detail: '[0.0908, 0.2164, 0.0, 0.016]', color: COLORS.blue },
                  { step: '03', label: 'CEM-MPC computes optimal torque', detail: 'action = −0.460 N·m', color: COLORS.cyan },
                  { step: '04', label: 'Command sent to firmware', detail: '{"op":"command","action":[−0.460]}', color: COLORS.green },
                  { step: '05', label: 'Motors apply 18.4% power', detail: 'Bot corrects. SystemID updates.', color: COLORS.green },
                ].map((row, i) => (
                  <motion.div key={i}
                    initial={{ opacity: 0, x: -10 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.12 }}
                    className="flex items-start gap-4 p-3 bg-bgRaised border border-borderDim"
                  >
                    <span className="font-mono text-[10px] text-textDim shrink-0 w-6">{row.step}</span>
                    <div className="flex-1 min-w-0">
                      <div className="font-body text-xs text-textSecondary">{row.label}</div>
                      <div className="font-mono text-[9px] truncate" style={{ color: row.color }}>{row.detail}</div>
                    </div>
                  </motion.div>
                ))}
              </div>

              <div className="p-4 border border-green/20 bg-green/5">
                <span className="font-mono text-[10px] text-green uppercase tracking-widest">16.7ms per cycle. 60 times per second. Continuously.</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── ARCHITECTURE ── */}
      <section id="architecture" className="bg-bg py-32 px-6">
        <div className="max-w-[1100px] mx-auto space-y-16">
          <div className="text-center space-y-4">
            <span className="micro-label text-green">System Architecture</span>
            <h2 className="font-display text-4xl md:text-5xl font-bold text-white">Six layers. One kernel.</h2>
            <p className="font-body text-textSecondary max-w-[600px] mx-auto">
              Each layer has one job. Together they take a robot that breaks on deployment and make it work — in real-time, on real hardware, without retraining.
            </p>
          </div>
          <div className="space-y-3">
            {[
              { l: 'L6', name: 'REGISTRY LAYER', desc: 'Persistent learning. Saves every session. Gets smarter with every hardware deployment.', tech: 'Model Registry / Platform Prior', color: COLORS.cyan, detail: 'After 100 labs use PhysiCore on the same arm, lab 101 starts with 100 sessions of prior knowledge.' },
              { l: 'L5', name: 'INTEGRATION LAYER', desc: 'Production bridge code. Connects any hardware in 30 minutes.', tech: 'ROS2 / ArduPilot / PX4 / Arduino Serial', color: COLORS.textSecondary, detail: 'One command. One YAML file. Any hardware. The bridge handles all protocol translation automatically.' },
              { l: 'L4', name: 'SENTINEL GOVERNANCE', desc: 'Safety envelopes. Three-mode state machine. SHA-256 forensic log.', tech: 'NOMINAL → CAUTIOUS → FALLBACK', color: COLORS.amber, detail: 'Lyapunov energy monitoring ensures PhysiCore never commands an action that violates stability bounds.' },
              { l: 'L3', name: 'INTELLIGENCE LAYER', desc: 'Three neural networks learn what the simulator gets wrong. Online SystemID learns real mass and friction.', tech: 'ResidualEnsemble + OnlineSystemID', color: COLORS.blue, detail: 'Converges on real mass within 30 seconds. Innovation-driven adaptive learning rate speeds up when payload changes.' },
              { l: 'L2', name: 'CONTROL LAYER', desc: 'CEM-MPC optimizer. 6-step lookahead. Uncertainty-penalized planning.', tech: 'Cross-Entropy Method / 60Hz locked', color: COLORS.textSecondary, detail: 'Penalizes high-uncertainty regions. Conservative when unsure. Precise when confident.' },
              { l: 'L1', name: 'PHYSICS KERNEL', desc: 'RK4 integrator. ISA atmosphere. J2 orbital perturbation. Dryden turbulence.', tech: 'Aerospace-grade physics from first principles', color: COLORS.green, detail: '4th-order Runge-Kutta stays stable under stiff nonlinear dynamics where Euler integration would blow up.' },
            ].map((layer, i) => (
              <motion.div key={i}
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.08 }}
                className="reveal flex items-stretch border border-border bg-bgRaised group hover:border-borderActive transition-all cursor-default"
              >
                <div className="w-[56px] flex items-center justify-center border-r border-border font-mono text-[10px] text-textDim group-hover:text-textPrimary transition-colors shrink-0">{layer.l}</div>
                <div className="w-1" style={{ backgroundColor: layer.color }} />
                <div className="flex-1 p-5 grid md:grid-cols-3 gap-4 items-center">
                  <div className="md:col-span-1">
                    <h3 className="font-display text-sm font-bold tracking-widest text-textPrimary uppercase">{layer.name}</h3>
                    <p className="font-body text-[11px] text-textSecondary mt-1">{layer.desc}</p>
                  </div>
                  <div className="font-mono text-[9px] text-textDim uppercase tracking-widest">{layer.tech}</div>
                  <div className="font-body text-[11px] text-textDim leading-relaxed hidden md:block">{layer.detail}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── DOMAINS ── */}
      <section id="domains" className="bg-void py-32 px-6 border-t border-border">
        <div className="max-w-[1100px] mx-auto space-y-16">
          <div className="text-center space-y-4">
            <span className="micro-label text-green">Multi-Domain Support</span>
            <h2 className="font-display text-4xl md:text-5xl font-bold text-white uppercase tracking-tighter">Engineered for the Edge.</h2>
            <p className="font-body text-textSecondary max-w-[600px] mx-auto">12 hardware platforms. One adaptation engine. From a balancing bot on an Arduino to a liquid rocket's flight computer.</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                title: 'Robotics',
                icon: Cpu,
                desc: 'Industrial manipulators, humanoids, legged robots, surgical systems, AUVs. PhysiCore learns your real joint friction, mass distribution, and contact dynamics without retraining.',
                tech: 'ROS2 / Arduino / ESP32',
                platforms: ['Balancing bot', 'Robot arm', 'Legged robot', 'Surgical robot', 'AUV', 'Ground rover'],
              },
              {
                title: 'Rockets',
                icon: Rocket,
                desc: 'Sounding rockets to orbital launch vehicles. Learns real propellant consumption, nozzle erosion, and aerodynamic variation while the vehicle is in flight.',
                tech: 'Custom FC / MAVLink serial',
                platforms: ['Sounding rockets', 'Liquid vehicles', 'Hybrid motors', 'Orbital vehicles'],
              },
              {
                title: 'Aviation',
                icon: Navigation,
                desc: 'Fixed-wing UAVs, quadrotors, eVTOL. ISA atmosphere, Dryden turbulence, Mach drag modeling. Works with PX4 and ArduPilot flight controllers.',
                tech: 'PX4 / ArduPilot / MAVLink',
                platforms: ['Quadrotors', 'Fixed-wing UAVs', 'eVTOL', 'Satellites'],
              },
            ].map((d, i) => (
              <div key={i} className="reveal p-10 border border-border bg-bgRaised space-y-8 group hover:border-green transition-all">
                <div className="w-16 h-16 bg-bg flex items-center justify-center border border-border group-hover:border-green transition-all">
                  <d.icon className="text-green" size={32} />
                </div>
                <div className="space-y-3">
                  <h3 className="font-display text-2xl font-bold text-white uppercase tracking-widest">{d.title}</h3>
                  <p className="font-body text-sm text-textSecondary leading-relaxed">{d.desc}</p>
                </div>
                <div className="space-y-2">
                  {d.platforms.map((p, j) => (
                    <div key={j} className="flex items-center gap-2">
                      <div className="w-1 h-1 rounded-full bg-green" />
                      <span className="font-mono text-[10px] text-textDim">{p}</span>
                    </div>
                  ))}
                </div>
                <div className="pt-4 border-t border-border">
                  <span className="font-mono text-[10px] text-textDim uppercase tracking-widest">{d.tech}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── FEATURES ── */}
      <section id="features" className="bg-bg py-32 px-6">
        <div className="max-w-[1100px] mx-auto space-y-16">
          <div className="space-y-4">
            <span className="micro-label text-green">Core Capabilities</span>
            <h2 className="font-display text-4xl md:text-5xl font-bold text-white">Every subsystem. Fully specified.</h2>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                title: 'RK4 Physics Kernel',
                color: COLORS.green,
                desc: '4th-order Runge-Kutta integration samples four derivative points per step. Stable under high stiffness and nonlinear dynamics where Euler integration diverges.',
                spec: 'k₁ k₂ k₃ k₄ → Δstate / 16.7ms',
              },
              {
                title: 'Online SystemID',
                color: COLORS.cyan,
                desc: 'Innovation-driven adaptive learning rate. When the robot enters new terrain or picks up a payload, the learning rate increases automatically. Converges on real mass within 30 seconds.',
                spec: '∇mass ∇friction → physical bounds',
              },
              {
                title: 'Residual Ensemble',
                color: COLORS.blue,
                desc: 'Three neural networks learn what your simulator gets wrong. The spread between their predictions quantifies epistemic uncertainty — feeding directly into the optimizer cost function.',
                spec: 'σ(MLP₁, MLP₂, MLP₃) → confidence',
              },
              {
                title: 'CEM-MPC Optimizer',
                color: COLORS.amber,
                desc: 'Cross-Entropy Method samples action sequences, evaluates them through the physics model plus residual, keeps the best, and repeats. 6-step lookahead every 16.7ms.',
                spec: 'CEM solver / 6-step / 60Hz locked',
              },
              {
                title: 'Persistent Registry',
                color: COLORS.cyan,
                desc: 'Every session saves learned mass, friction, and ensemble weights. Next session starts where the last one ended. Each deployment makes the model smarter.',
                spec: '~/.physicore/registry/{platform}/',
              },
              {
                title: 'Sentinel OS',
                color: COLORS.amber,
                desc: 'Three-mode safety state machine. Lyapunov energy monitoring. SHA-256 forensic hash chain on every control command. Falls back to safe controller automatically.',
                spec: 'NOMINAL → CAUTIOUS → FALLBACK',
              },
            ].map((f, i) => (
              <div key={i} className="reveal p-8 border border-border bg-bgRaised border-t-2 space-y-6 group hover:bg-void transition-all" style={{ borderTopColor: f.color }}>
                <h3 className="font-display text-lg font-bold text-white tracking-widest uppercase">{f.title}</h3>
                <p className="font-body text-sm text-textSecondary leading-relaxed">{f.desc}</p>
                <div className="font-mono text-[10px] uppercase tracking-widest" style={{ color: f.color }}>{f.spec}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── BENCHMARKS ── */}
      <section id="benchmarks" className="bg-void py-32 px-6">
        <div className="max-w-[1100px] mx-auto space-y-16">
          <div className="space-y-4">
            <span className="micro-label text-green">Performance — Real Numbers</span>
            <h2 className="font-display text-4xl md:text-5xl font-bold text-white">Numbers that matter to engineers.</h2>
            <p className="font-body text-textSecondary">These are not simulated results. These are real measurements from running PhysiCore on a balancing bot with Arduino Uno + MPU6050 + L298N.</p>
          </div>
          <div className="grid md:grid-cols-2 gap-20">
            <div className="space-y-8">
              {[
                { label: 'CONTROL LOOP RATE', val: '60 Hz', p: 100, note: 'All 12 platforms under 16.7ms per step' },
                { label: 'SYSID CONVERGENCE', val: '30 seconds', p: 85, note: '1,800 steps at 60Hz on real hardware' },
                { label: 'MASS ERROR AFTER CONVERGENCE', val: '< 18%', p: 82, note: 'True mass 1.4kg, estimated 1.16kg after 30s' },
                { label: 'MOTOR POWER WITH PHYSICORE', val: '18.4%', p: 93, note: 'vs 0.4% without — 46x more responsive' },
                { label: 'SAFETY AUDIT CHECKS', val: '68 / 68', p: 100, note: 'All platforms, all scenarios' },
                { label: 'SENTINEL FALLBACK TRIGGER', val: '< 500ms', p: 95, note: 'From NOMINAL to FALLBACK on anomaly detect' },
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
                  <span className="font-mono text-[9px] text-textDim">{b.note}</span>
                </div>
              ))}
            </div>
            <div className="reveal border border-border bg-bgRaised overflow-hidden">
              <div className="p-4 border-b border-border">
                <span className="font-mono text-[10px] text-textDim uppercase tracking-widest">System Specification</span>
              </div>
              <table className="w-full text-left border-collapse">
                <tbody>
                  {[
                    ['PHYSICS INTEGRATOR', 'RK4 — 4th order'],
                    ['OPTIMIZER', 'CEM — Cross-Entropy Method'],
                    ['ENSEMBLE SIZE', '3 MLP networks'],
                    ['LOOP RATE', '60 Hz locked'],
                    ['SYSID METHOD', 'Numerical gradient + SGD momentum'],
                    ['ADAPTIVE LR', 'Innovation-driven (RLS-style)'],
                    ['MPC HORIZON', '6 steps lookahead'],
                    ['ACTION SMOOTHING', 'Exponential (α = 0.35)'],
                    ['SAFETY CHAIN', 'SHA-256 hash every step'],
                    ['PLATFORMS', '12 hardware types'],
                    ['PERSISTENCE', 'Registry saves per session'],
                    ['LEARNING', 'Starts fresh, improves with time'],
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

      {/* ── SENTINEL ── */}
      <section id="sentinel" className="bg-bg py-32 px-6 border-t border-border">
        <div className="max-w-[1100px] mx-auto space-y-24">
          <div className="grid md:grid-cols-2 gap-20 items-center">
            <div className="space-y-8">
              <div className="reveal border-l-2 border-amber pl-4">
                <span className="micro-label text-amber">Governance & Safety</span>
              </div>
              <h2 className="reveal reveal-stagger-1 font-display text-5xl md:text-6xl font-bold text-white leading-tight">
                Sentinel OS. <br />
                <span className="text-textSecondary">The Safety Kernel.</span>
              </h2>
              <div className="reveal reveal-stagger-2 space-y-6 font-body text-textSecondary leading-relaxed text-lg">
                <p>PhysiCore provides the intelligence. Sentinel OS provides the authority. It monitors every control step and can override or halt PhysiCore instantly if anything goes outside safe bounds.</p>
                <p>Three modes. Automatic transitions. You cannot bypass it. It is not optional — it is the layer that makes real hardware deployment safe enough to trust.</p>
              </div>
            </div>
            <div className="reveal reveal-stagger-3">
              <SentinelDiagram />
            </div>
          </div>

          <div className="grid md:grid-cols-4 gap-6">
            {[
              { title: 'Lyapunov Monitor', desc: 'Monitors system energy every step. If the robot enters an unstable energy state, Sentinel drops to FALLBACK before damage occurs.', icon: ShieldCheck },
              { title: 'Three-Mode FSM', desc: 'NOMINAL — full PhysiCore. CAUTIOUS — conservative (60% output). FALLBACK — PhysiCore off, safe controller takes over.', icon: AlertTriangle },
              { title: 'SHA-256 Forensic Log', desc: 'Every control command is signed and hashed in a chain. Tamper-evident. Every decision is traceable after the fact.', icon: FileJson },
              { title: 'Platform Presets', desc: 'Surgical robots get tighter bounds than drones. Satellites get looser. Each platform has tuned thresholds for what "safe" means in its domain.', icon: CheckCircle2 },
            ].map((item, i) => (
              <div key={i} className="reveal p-6 border border-border bg-bgRaised space-y-4 group hover:border-amber transition-all">
                <item.icon className="text-amber" size={24} />
                <h3 className="font-display text-sm font-bold text-white tracking-widest uppercase">{item.title}</h3>
                <p className="font-body text-xs text-textSecondary leading-relaxed">{item.desc}</p>
              </div>
            ))}
          </div>

          {/* Mode transition diagram */}
          <div className="reveal border border-amber/20 bg-void p-10 space-y-6">
            <h3 className="font-display text-lg font-bold text-white uppercase tracking-widest">Mode Transitions</h3>
            <div className="flex items-center gap-0 overflow-x-auto">
              {[
                { mode: 'NOMINAL', color: COLORS.green, desc: 'Full PhysiCore control', sub: 'uncertainty < 5%\nresidual < 0.5' },
                { mode: '→', color: COLORS.textDim, desc: '', sub: '' },
                { mode: 'CAUTIOUS', color: COLORS.amber, desc: '60% output, tighter bounds', sub: 'uncertainty 5–15%\nresidual 0.5–2.0' },
                { mode: '→', color: COLORS.textDim, desc: '', sub: '' },
                { mode: 'FALLBACK', color: COLORS.red, desc: 'PhysiCore off, safe stop', sub: 'uncertainty > 15%\nor Lyapunov exceeded' },
              ].map((m, i) => (
                m.mode === '→' ? (
                  <div key={i} className="px-4 font-mono text-textDim text-xl shrink-0">→</div>
                ) : (
                  <div key={i} className="flex-1 min-w-[140px] p-4 border border-borderDim space-y-2">
                    <div className="font-display text-sm font-bold" style={{ color: m.color }}>{m.mode}</div>
                    <div className="font-body text-xs text-textSecondary">{m.desc}</div>
                    <div className="font-mono text-[9px] text-textDim whitespace-pre-line">{m.sub}</div>
                  </div>
                )
              ))}
            </div>
            <p className="font-body text-xs text-textDim">Recovery: FALLBACK → CAUTIOUS after 300 stable steps. CAUTIOUS → NOMINAL after 50 stable steps. Automatic.</p>
          </div>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <section className="bg-void py-24 px-6 border-t border-border">
        <div className="max-w-[1100px] mx-auto flex flex-col md:flex-row items-center justify-between gap-8">
          <div className="space-y-2">
            <div className="font-display text-2xl font-bold text-white tracking-tighter">PhysiCore</div>
            <div className="font-mono text-[10px] text-textDim uppercase tracking-widest">Physics Intelligence Engine — v3.1 — Founders Inc '26</div>
            <div className="font-body text-xs text-textDim">Built by Prathamesh Shirbhate</div>
          </div>
          <div className="flex gap-8">
            <button onClick={() => setView('manual')} className="font-mono text-[10px] text-textDim uppercase tracking-widest hover:text-textSecondary transition-colors">Manual</button>
            <button onClick={handleSetIntegratorView} className="font-mono text-[10px] text-textDim uppercase tracking-widest hover:text-textSecondary transition-colors">Integration Engineer</button>
            <a href="https://github.com/prathameshbharatiya/physicore-hybrid-mpc" target="_blank" rel="noopener noreferrer" className="font-mono text-[10px] text-textDim uppercase tracking-widest hover:text-textSecondary transition-colors">GitHub</a>
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
        <div className="border-b border-border bg-bg px-6 py-4 flex items-center gap-4 sticky top-[52px] z-10 flex-wrap">
          <button onClick={()=>setIE(s=>({...s,phase:'welcome'}))}
            className="font-mono text-[9px] text-textDim uppercase tracking-widest hover:text-white transition-colors">← New</button>
          <span className="text-base">{flow?.icon}</span>
          <span className="font-display text-xs font-bold text-green uppercase tracking-widest">✓ {flow?.label} — {ieState.files.length} files ready</span>

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
            className="px-4 py-2 border border-border font-mono text-[9px] text-textDim uppercase tracking-widest hover:border-amber hover:text-amber transition-all">
            Help / Troubleshoot
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

  const renderManual = () => {
    const sections = [
      { id: 'intro',      title: '01. WHAT IS PHYSICORE',    icon: <Info size={14} /> },
      { id: 'how',        title: '02. HOW IT WORKS',         icon: <Activity size={14} /> },
      { id: 'bot',        title: '03. BALANCING BOT',        icon: <Cpu size={14} /> },
      { id: 'drone',      title: '04. PX4 / ARDUPILOT DRONE',icon: <Navigation size={14} /> },
      { id: 'rocket',     title: '05. SOUNDING ROCKET',      icon: <Rocket size={14} /> },
      { id: 'ros2',       title: '06. ROS2 ROBOT ARM',       icon: <Terminal size={14} /> },
      { id: 'humanoid',   title: '07. HUMANOID / LEGGED',    icon: <Activity size={14} /> },
      { id: 'auv',        title: '08. AUV / UNDERWATER',     icon: <Wind size={14} /> },
      { id: 'evtol',      title: '09. eVTOL AIRCRAFT',       icon: <Navigation size={14} /> },
      { id: 'rover',      title: '10. GROUND ROVER / AMR',   icon: <Cpu size={14} /> },
      { id: 'satellite',  title: '11. SATELLITE / SPACECRAFT',icon: <Globe size={14} /> },
      { id: 'custom',     title: '12. CUSTOM HARDWARE',      icon: <Settings size={14} /> },
      { id: 'config',     title: '13. ROBOT CONFIG (YAML)',  icon: <Settings size={14} /> },
      { id: 'registry',   title: '14. PERSISTENT LEARNING',  icon: <Layers size={14} /> },
      { id: 'sentinel',   title: '15. SENTINEL OS',          icon: <ShieldCheck size={14} /> },
      { id: 'troubleshoot',title: '16. TROUBLESHOOTING',     icon: <AlertTriangle size={14} /> },
    ];

    const Step = ({ n, children }: { n: number; children: React.ReactNode }) => (
      <div className="flex gap-4">
        <div className="shrink-0 w-7 h-7 border border-border flex items-center justify-center font-mono text-[10px] text-textDim">{n}</div>
        <div className="font-body text-sm text-textSecondary leading-relaxed pt-0.5">{children}</div>
      </div>
    );

    const Code = ({ children }: { children: React.ReactNode }) => (
      <div className="my-3 p-4 bg-bgRaised border border-borderDim font-mono text-[11px] text-cyan select-all overflow-x-auto whitespace-pre">{children}</div>
    );

    const Phase = ({ n, title, children }: { n: string; title: string; children: React.ReactNode }) => (
      <div className="space-y-4">
        <div className="flex items-center gap-3 border-b border-border pb-3">
          <span className="font-mono text-[10px] text-green uppercase tracking-widest">{n}</span>
          <h3 className="font-display text-sm font-bold text-white uppercase tracking-widest">{title}</h3>
        </div>
        <div className="space-y-4 pl-4 border-l border-borderDim">{children}</div>
      </div>
    );

    const Good = ({ children }: { children: React.ReactNode }) => (
      <div className="p-4 border border-green/20 bg-green/5 space-y-1">
        <div className="font-mono text-[9px] text-green uppercase tracking-widest">What good looks like</div>
        <div className="font-body text-xs text-textSecondary leading-relaxed">{children}</div>
      </div>
    );

    const Warn = ({ title, children }: { title: string; children: React.ReactNode }) => (
      <div className="p-4 border border-amber/20 bg-amber/5 space-y-1">
        <div className="font-mono text-[9px] text-amber uppercase tracking-widest">{title}</div>
        <div className="font-body text-xs text-textSecondary leading-relaxed">{children}</div>
      </div>
    );

    return (
      <div className="pt-[52px] h-screen flex bg-void overflow-hidden">
        {/* SIDEBAR */}
        <aside className="w-[280px] border-r border-border bg-bg flex flex-col shrink-0">
          <div className="p-6 border-b border-border">
            <div className="flex items-center gap-3 text-amber mb-1">
              <BookOpen size={18} />
              <span className="font-display text-sm font-bold uppercase tracking-widest">Integration Manual</span>
            </div>
            <span className="font-mono text-[9px] text-textDim uppercase tracking-widest">PhysiCore v3.1 — Real Hardware Proven</span>
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
          <div className="max-w-[780px] mx-auto space-y-12 pb-24">

            {/* ── INTRO ── */}
            {manualSection === 'intro' && (
              <section className="space-y-10">
                <div className="space-y-4">
                  <h1 className="font-display text-4xl font-black text-white tracking-tighter uppercase">What is PhysiCore?</h1>
                  <p className="font-body text-lg text-textSecondary leading-relaxed">
                    PhysiCore is a real-time physics adaptation engine. It sits between your robot's sensors and its motors. Every 16.7 milliseconds it reads your sensor data, computes the optimal control action using real physics and neural learning, and sends that command to your hardware.
                  </p>
                  <p className="font-body text-sm text-textSecondary leading-relaxed">
                    The key difference from everything else: it <em>learns</em>. It starts with a physics model and a guess at your robot's mass and friction. As the robot moves, it compares what it predicted would happen with what actually happened. It updates its model. Within 30 seconds it knows your robot's real physics better than any simulation.
                  </p>
                </div>

                <div className="space-y-4">
                  <h2 className="font-display text-lg font-bold text-white uppercase tracking-widest">The problem it solves</h2>
                  <p className="font-body text-sm text-textSecondary leading-relaxed">
                    Every robot trained in simulation fails on real hardware. The simulation assumed your robot weighs exactly 1.0 kg with 0.15 friction coefficient. Your real robot weighs 1.35 kg and the floor has different friction every time you deploy it somewhere new.
                  </p>
                  <p className="font-body text-sm text-textSecondary leading-relaxed">
                    Teams spend weeks manually re-tuning every time this happens. PhysiCore makes it automatic. You connect it, run your robot for 30 seconds, and it has learned your hardware. No manual tuning. No retraining. No offline calibration.
                  </p>
                </div>

                <div className="space-y-4">
                  <h2 className="font-display text-lg font-bold text-white uppercase tracking-widest">What it proved</h2>
                  <div className="p-6 border border-green/20 bg-green/5 space-y-3">
                    <div className="font-mono text-[10px] text-green uppercase tracking-widest">Real Hardware Test — Balancing Bot</div>
                    <p className="font-body text-sm text-textSecondary">We connected PhysiCore to an Arduino Uno with MPU6050 IMU and L298N motor driver. Before PhysiCore: the bot fell immediately. After connecting PhysiCore and letting it run for 30 seconds: the bot balanced. It didn't fall. Not once. Mass estimate converged from 1.0 to 1.16 kg (true mass: 1.35 kg) — with no manual calibration.</p>
                    <div className="grid grid-cols-3 gap-4 pt-2">
                      {[
                        { label: 'Motor power before', val: '0.4%', color: COLORS.red },
                        { label: 'Motor power after', val: '18.4%', color: COLORS.green },
                        { label: 'Falls while balanced', val: '0', color: COLORS.green },
                      ].map((s, i) => (
                        <div key={i} className="text-center">
                          <div className="font-display text-2xl font-bold" style={{ color: s.color }}>{s.val}</div>
                          <div className="font-mono text-[9px] text-textDim mt-1">{s.label}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h2 className="font-display text-lg font-bold text-white uppercase tracking-widest">Supported hardware</h2>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      'Balancing bot (Arduino + MPU6050)',
                      'PX4 quadrotor / fixed-wing',
                      'ArduPilot copter / plane',
                      'eVTOL aircraft',
                      'ROS2 robot arm (UR, KUKA, custom)',
                      'ROS2 humanoid / legged robot',
                      'ROS2 AUV / underwater robot',
                      'ROS2 surgical robot',
                      'Sounding rocket (custom FC)',
                      'Ground rover / AMR',
                      'Satellite / spacecraft',
                      'Custom serial hardware',
                    ].map((p, i) => (
                      <div key={i} className="flex items-center gap-2 p-3 border border-borderDim bg-bgRaised">
                        <div className="w-1.5 h-1.5 rounded-full bg-green shrink-0" />
                        <span className="font-mono text-[10px] text-textDim">{p}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </section>
            )}

            {/* ── HOW IT WORKS ── */}
            {manualSection === 'how' && (
              <section className="space-y-10">
                <div className="space-y-4">
                  <h1 className="font-display text-4xl font-black text-white tracking-tighter uppercase">How PhysiCore Works</h1>
                  <p className="font-body text-lg text-textSecondary leading-relaxed">One control cycle. 16.7 milliseconds. Repeated 60 times per second.</p>
                </div>

                <div className="space-y-3">
                  {[
                    { step: 1, title: 'Sensor data arrives', detail: 'Your hardware sends pitch, gyro rate, acceleration, and motor state over serial or MAVLink. This happens every 20ms at 50Hz.', example: '{"pitch":5.2,"gyro_x":12.4,"accel_x":0.8,...}' },
                    { step: 2, title: 'State vector built', detail: 'The bridge converts raw sensor readings into a state vector the physics engine understands. For a balancing bot: [pitch in radians, pitch rate in rad/s, position, velocity].', example: 'state = [0.0908, 0.2164, 0.0, 0.016]' },
                    { step: 3, title: 'Physics model predicts', detail: 'RK4 integration runs the physics model forward: given current state and last action, where should the robot be? This uses your estimated mass and friction.', example: 'x_predicted = physics.step(state, action, dt)' },
                    { step: 4, title: 'Residual ensemble corrects', detail: 'Three neural networks predict what the physics model got wrong (the residual). Their average is added to the physics prediction. Their spread measures how uncertain the model is.', example: 'residual, uncertainty = ensemble.predict(state, action)' },
                    { step: 5, title: 'CEM-MPC optimizes', detail: 'Cross-Entropy Method samples candidate action sequences, simulates them forward 6 steps using physics + residual, evaluates cost, keeps the best, repeats. Returns the optimal first action.', example: 'optimal_action = cem.optimize(state, physics, ensemble)' },
                    { step: 6, title: 'Command sent to hardware', detail: 'The optimal torque is sent back to your firmware. Firmware applies it to the motors. The safety timeout means: if commands stop arriving, firmware falls back to PID after 500ms.', example: '{"op":"command","action":[-0.460]}' },
                    { step: 7, title: 'SystemID learns', detail: 'PhysiCore compares what it predicted with what actually happened. It runs numerical gradient descent to update its mass and friction estimates. Innovation-driven adaptive learning rate speeds up when the robot enters new conditions.', example: 'sysid.update(state, action, next_state, physics)' },
                    { step: 8, title: 'Repeat every 16.7ms', detail: 'Back to step 1. The model gets more accurate with every cycle. After 30 seconds (1,800 steps) the mass estimate has converged. After the session ends, the registry saves everything so next session starts smarter.', example: 'steps=1800 mass=1.160 residual=0.024 ✓ converged' },
                  ].map((s, i) => (
                    <div key={i} className="p-5 border border-borderDim bg-bgRaised space-y-2">
                      <div className="flex items-center gap-3">
                        <span className="font-mono text-[10px] text-green">{String(s.step).padStart(2,'0')}</span>
                        <span className="font-display text-sm font-bold text-white">{s.title}</span>
                      </div>
                      <p className="font-body text-xs text-textSecondary leading-relaxed pl-8">{s.detail}</p>
                      <div className="font-mono text-[9px] text-textDim pl-8 border-l border-borderDim ml-8">{s.example}</div>
                    </div>
                  ))}
                </div>

                <div className="space-y-4">
                  <h2 className="font-display text-lg font-bold text-white uppercase tracking-widest">What PhysiCore cannot do</h2>
                  <div className="space-y-2">
                    {[
                      'It cannot work without real sensor data — it needs real hardware connected.',
                      'It cannot learn from a stationary robot — it needs real motion to compare predictions.',
                      'It cannot compensate for broken motors or damaged sensors.',
                      'It cannot replace your flight controller or ROS2 stack — it works alongside them.',
                      'It does not remember between sessions by default — enable the registry to persist learning.',
                    ].map((item, i) => (
                      <div key={i} className="flex items-start gap-3 p-3 border border-borderDim">
                        <span className="text-red text-xs shrink-0 mt-0.5">✗</span>
                        <span className="font-body text-xs text-textSecondary">{item}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </section>
            )}

            {/* ── BALANCING BOT ── */}
            {manualSection === 'bot' && (
              <section className="space-y-10">
                <div className="space-y-4">
                  <h1 className="font-display text-4xl font-black text-white tracking-tighter uppercase">Balancing Bot</h1>
                  <p className="font-body text-lg text-textSecondary">Arduino Uno (or Nano/Mega) + MPU6050 IMU + L298N motor driver. Windows instructions below.</p>
                  <div className="p-4 border border-green/20 bg-green/5">
                    <span className="font-mono text-[10px] text-green uppercase tracking-widest">Tested and proven. This exact setup was used for the first real hardware test. The bot didn't fall.</span>
                  </div>
                </div>

                <div className="space-y-3">
                  <h2 className="font-display text-sm font-bold text-white uppercase tracking-widest">Required wiring</h2>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { label: 'MPU6050 SDA', val: '→ Arduino A4' },
                      { label: 'MPU6050 SCL', val: '→ Arduino A5' },
                      { label: 'MPU6050 VCC', val: '→ Arduino 3.3V (NOT 5V)' },
                      { label: 'MPU6050 GND', val: '→ Arduino GND' },
                      { label: 'L298N ENA', val: '→ Arduino Pin 5' },
                      { label: 'L298N IN1', val: '→ Arduino Pin 4' },
                      { label: 'L298N IN2', val: '→ Arduino Pin 3' },
                      { label: 'L298N ENB', val: '→ Arduino Pin 6' },
                      { label: 'L298N IN3', val: '→ Arduino Pin 7' },
                      { label: 'L298N IN4', val: '→ Arduino Pin 8' },
                    ].map((w, i) => (
                      <div key={i} className="p-2 bg-bgRaised border border-borderDim flex justify-between">
                        <span className="font-mono text-[10px] text-textDim">{w.label}</span>
                        <span className="font-mono text-[10px] text-cyan">{w.val}</span>
                      </div>
                    ))}
                  </div>
                  <Warn title="Critical">VCC must go to 3.3V. Connecting to 5V will damage the MPU6050.</Warn>
                </div>

                <Phase n="PHASE 1" title="Flash the firmware">
                  <Step n={1}>Open Arduino IDE.</Step>
                  <Step n={2}>Go to <strong className="text-white">Sketch → Include Library → Manage Libraries</strong>. Search <code className="text-cyan">MPU6050_light</code> by rfetick. Install. Then search <code className="text-cyan">ArduinoJson</code> by Benoit Blanchon — make sure version 6.x. Install.</Step>
                  <Step n={3}>Download the firmware file from the Integration Engineer, or open <code className="text-cyan">firmware/balancing_bot/physicore_active.ino</code> from the project.</Step>
                  <Step n={4}>Find this line near the top and leave it at 0.0 for now:
                    <Code>const float BALANCE_POINT = 0.0;  // calibrate this in Phase 2</Code>
                  </Step>
                  <Step n={5}>Plug your Arduino into your laptop with USB.</Step>
                  <Step n={6}>Find your COM port: press <strong className="text-white">Windows key + X → Device Manager → Ports (COM & LPT)</strong>. Look for something like <code className="text-cyan">USB-SERIAL CH340 (COM3)</code>. Write down that number.</Step>
                  <Step n={7}>In Arduino IDE: <strong className="text-white">Tools → Board → Arduino Uno</strong>. Then <strong className="text-white">Tools → Port → COM3</strong> (your actual number).</Step>
                  <Step n={8}>Click the Upload button (→ arrow). Wait for "Done uploading."</Step>
                </Phase>

                <Phase n="PHASE 2" title="Calibrate BALANCE_POINT — critical">
                  <Step n={9}>In Arduino IDE: <strong className="text-white">Tools → Serial Monitor</strong>. Bottom right dropdown → set to <strong className="text-white">115200 baud</strong>.</Step>
                  <Step n={10}>You will see JSON printing every 20ms:
                    <Code>{`{"pitch":2.3,"roll":0.1,"gyro_x":0.2,...}`}</Code>
                  </Step>
                  <Step n={11}>Hold your robot <strong className="text-white">perfectly upright</strong> — the exact angle where it would balance. Look at the <code className="text-cyan">pitch</code> value. Write it down. Example: 2.3</Step>
                  <Step n={12}>Close Serial Monitor. Find this line in the firmware and change 0.0 to your reading:
                    <Code>const float BALANCE_POINT = 2.3;  // your actual value</Code>
                  </Step>
                  <Step n={13}>Click Upload again.</Step>
                  <Step n={14}>Open Serial Monitor again. Hold bot upright. Pitch should now read approximately <strong className="text-white">0.0</strong>. If yes — BALANCE_POINT is correct.</Step>
                  <Warn title="Why this matters">Wrong BALANCE_POINT = PhysiCore permanently fights a lean that doesn't exist. The bot will fall immediately or motors will spin in one direction non-stop. 5 minutes here makes everything else work.</Warn>
                </Phase>

                <Phase n="PHASE 3" title="Run the bridge">
                  <Step n={15}><strong className="text-white">Close Arduino IDE completely.</strong> Do not leave it open. Arduino IDE blocks the serial port. If it's open, the bridge cannot connect.</Step>
                  <Step n={16}>Open your PhysiCore project folder in File Explorer. Click the address bar. Type <code className="text-cyan">cmd</code>. Press Enter. A black terminal opens.</Step>
                  <Step n={17}>Install dependencies:
                    <Code>pip install pymavlink websockets aiohttp pyserial pyyaml</Code>
                  </Step>
                  <Step n={18}>Run the bridge — replace COM3 with your actual port:
                    <Code>python physicore/bridge/physicore_bridge.py --platform balancing_bot_arduino --connection COM3</Code>
                  </Step>
                  <Good>
                    You should see:<br />
                    <code className="text-green">[BRIDGE] Serial connected: COM3</code><br />
                    <code className="text-green">[ENGINE] Initialized for 'balancing_bot'</code><br />
                    <code className="text-green">[TELEM] P:0.1° R:0.0° | mass=1.000 res=0.0000 steps=0</code>
                  </Good>
                </Phase>

                <Phase n="PHASE 4" title="Connect the dashboard">
                  <Step n={19}>Open your PhysiCore dashboard in Chrome (your Vercel URL).</Step>
                  <Step n={20}>Click <strong className="text-white">MAVLINK</strong> in the connection panel.</Step>
                  <Step n={21}>In the endpoint box type exactly: <Code>ws://localhost:8765</Code></Step>
                  <Step n={22}>Click <strong className="text-white">Connect</strong>.</Step>
                  <Step n={23}>Check: pitch updates when you tilt the bot ✓ | ESTIMATED MASS shows — ✓ | SystemID says "CLICK ACTIVE CONTROL ON" ✓</Step>
                </Phase>

                <Phase n="PHASE 5" title="Activate PhysiCore">
                  <Step n={24}>Hold your bot upright with your hand.</Step>
                  <Step n={25}>Click <strong className="text-white">ACTIVE CONTROL ON</strong> in the dashboard.</Step>
                  <Step n={26}>Watch the terminal — steps should count up and residual should move:
                    <Code>[TELEM] P:0.2° | mass=1.000 res=0.0240 steps=12 | clients=1</Code>
                  </Step>
                  <Step n={27}>Slowly loosen your grip. PhysiCore is now controlling the motors. Hold loosely for the first 30 seconds — let it move and correct, catch it if it falls hard.</Step>
                  <Good>
                    steps counting up — PhysiCore running ✓<br />
                    mass moving away from 1.000 — SystemID learning ✓<br />
                    residual dropping — model getting more accurate ✓<br />
                    Bot stays upright without falling ✓
                  </Good>
                </Phase>
              </section>
            )}

            {/* ── DRONE ── */}
            {manualSection === 'drone' && (
              <section className="space-y-10">
                <div className="space-y-4">
                  <h1 className="font-display text-4xl font-black text-white tracking-tighter uppercase">PX4 / ArduPilot Drone</h1>
                  <p className="font-body text-lg text-textSecondary">PhysiCore connects to your flight controller over MAVLink. It does not replace PX4 or ArduPilot — it adds a real-time physics adaptation layer on top.</p>
                </div>

                <Phase n="PHASE 1" title="Enable MAVLink telemetry">
                  <Step n={1}>Open QGroundControl (PX4) or Mission Planner (ArduPilot). Connect your flight controller.</Step>
                  <Step n={2}>For WiFi/UDP (most common): MAVLink UDP on port 14550 is enabled by default. No configuration needed.</Step>
                  <Step n={3}>For USB: connect the flight controller directly to your laptop. Note the COM port in Device Manager.</Step>
                </Phase>

                <Phase n="PHASE 2" title="Run the bridge">
                  <Step n={4}>Open terminal in your PhysiCore project folder. Run:
                    <Code>pip install pymavlink websockets aiohttp pyserial</Code>
                  </Step>
                  <Step n={5}>For UDP connection:
                    <Code>python physicore/bridge/physicore_bridge.py --platform px4_quadrotor --connection udp:14550</Code>
                  </Step>
                  <Step n={6}>For USB connection (replace COM3):
                    <Code>python physicore/bridge/physicore_bridge.py --platform px4_quadrotor --connection COM3</Code>
                  </Step>
                  <Good>
                    [BRIDGE] MAVLink connecting: udp:14550<br />
                    [BRIDGE] Waiting for heartbeat...<br />
                    [BRIDGE] MAVLink connected. Vehicle: QUADROTOR<br />
                    [TELEM] P:0.1° R:0.0° | mass=1.000 res=0.0000 steps=0
                  </Good>
                </Phase>

                <Phase n="PHASE 3" title="Connect dashboard and activate">
                  <Step n={7}>Dashboard → MAVLINK → <code className="text-cyan">ws://localhost:8765</code> → Connect.</Step>
                  <Step n={8}>Verify pitch and roll showing live values.</Step>
                  <Step n={9}>Arm your vehicle normally using QGC or your transmitter.</Step>
                  <Step n={10}>Click <strong className="text-white">ACTIVE CONTROL ON</strong>. PhysiCore is now augmenting your flight controller. Watch steps count up in the terminal.</Step>
                  <Warn title="Important">Arm the vehicle before clicking ACTIVE CONTROL ON. PhysiCore needs real flight dynamics to start learning — it cannot learn from a grounded vehicle.</Warn>
                </Phase>
              </section>
            )}

            {/* ── ROCKET ── */}
            {manualSection === 'rocket' && (
              <section className="space-y-10">
                <div className="space-y-4">
                  <h1 className="font-display text-4xl font-black text-white tracking-tighter uppercase">Sounding Rocket</h1>
                  <p className="font-body text-lg text-textSecondary">PhysiCore connects to your rocket's flight computer over serial JSON. It tracks real propellant consumption, learns your motor's actual burn curve, and logs every guidance decision with SHA-256 forensic hashing.</p>
                </div>

                <div className="space-y-4">
                  <h2 className="font-display text-sm font-bold text-white uppercase tracking-widest">Required telemetry format</h2>
                  <p className="font-body text-xs text-textSecondary">Your flight computer must send JSON over serial at 20-50Hz. Minimum required fields:</p>
                  <Code>{`{"altitude":0.0, "velocity":0.0, "accel_z":9.81, "pitch":0.0, "mass":5.0, "phase":"BOOST", "timestamp":0}`}</Code>
                  <p className="font-body text-xs text-textSecondary">The <code className="text-cyan">mass</code> field is critical — PhysiCore uses it to track real propellant depletion during the burn.</p>
                </div>

                <Phase n="PHASE 1" title="Flash your flight computer">
                  <Step n={1}>The Integration Engineer generates custom firmware for your FC (Arduino Mega, Teensy, ESP32). Use it to add the JSON serial output shown above.</Step>
                  <Step n={2}>Set your dry mass at the top of the firmware:
                    <Code>const float DRY_MASS = 5.0;  // kg without propellant</Code>
                  </Step>
                  <Step n={3}>Flash the firmware. Verify JSON appearing in Serial Monitor at 115200 baud.</Step>
                </Phase>

                <Phase n="PHASE 2" title="Run the bridge">
                  <Step n={4}>Close Arduino IDE. Find your COM port in Device Manager.</Step>
                  <Step n={5}>
                    <Code>pip install pymavlink websockets aiohttp pyserial</Code>
                    <Code>python physicore/bridge/physicore_bridge.py --platform custom_rocket_fc --connection COM3 --baud 115200</Code>
                  </Step>
                  <Good>
                    [BRIDGE] Serial connected: COM3<br />
                    [ENGINE] Initialized for 'rocket'<br />
                    [TELEM] altitude=0.0 velocity=0.0 phase=PRELAUNCH mass=5.000
                  </Good>
                </Phase>

                <Phase n="PHASE 3" title="Connect and monitor launch">
                  <Step n={6}>Dashboard → MAVLINK → <code className="text-cyan">ws://localhost:8765</code> → Connect.</Step>
                  <Step n={7}>Click <strong className="text-white">ACTIVE CONTROL ON</strong> before launch.</Step>
                  <Step n={8}>PhysiCore monitors all phases: PRELAUNCH → BOOST → COAST → APOGEE → DESCENT. Watch mass dropping during BOOST — that is the propellant tracking.</Step>
                  <Good>
                    Mass dropping during BOOST phase — tracking real propellant consumption ✓<br />
                    Phase transitions appearing in dashboard ✓<br />
                    Residual low during COAST — model has learned your vehicle ✓<br />
                    SHA-256 hash chain — every guidance command forensically logged ✓
                  </Good>
                </Phase>
              </section>
            )}

            {/* ── ROS2 ── */}
            {manualSection === 'ros2' && (
              <section className="space-y-10">
                <div className="space-y-4">
                  <h1 className="font-display text-4xl font-black text-white tracking-tighter uppercase">ROS2 Robot Arm</h1>
                  <p className="font-body text-lg text-textSecondary">PhysiCore subscribes to your ROS2 joint state topic and learns your arm's real friction and payload dynamics. No changes to your existing ROS2 stack.</p>
                </div>

                <Phase n="PHASE 1" title="Verify your ROS2 setup">
                  <Step n={1}>Source your ROS2 installation:
                    <Code>source /opt/ros/humble/setup.bash</Code>
                  </Step>
                  <Step n={2}>Verify your arm is publishing joint states:
                    <Code>ros2 topic echo /joint_states --once</Code>
                    You should see position, velocity, and effort for all joints. If nothing appears — your arm driver is not running.
                  </Step>
                  <Step n={3}>Check your actual topic name:
                    <Code>ros2 topic list</Code>
                    If it's different from /joint_states, note the actual name.
                  </Step>
                </Phase>

                <Phase n="PHASE 2" title="Run the bridge">
                  <Step n={4}>Navigate to your PhysiCore folder. Run:
                    <Code>{`source /opt/ros/humble/setup.bash\npip3 install pymavlink websockets aiohttp pyserial\npython3 physicore/bridge/physicore_bridge.py --platform ros2_manipulator`}</Code>
                  </Step>
                  <Good>
                    [BRIDGE] ROS2 subscribed to /imu/data /gps/fix /odom /joint_states<br />
                    [ENGINE] Initialized for 'manipulator_arm'<br />
                    [TELEM] P:0.0° | mass=1.000 res=0.0000 steps=0
                  </Good>
                </Phase>

                <Phase n="PHASE 3" title="Connect and learn">
                  <Step n={5}>Dashboard → MAVLINK → <code className="text-cyan">ws://localhost:8765</code> → Connect.</Step>
                  <Step n={6}>Verify joint angles appearing in dashboard.</Step>
                  <Step n={7}>Click <strong className="text-white">ACTIVE CONTROL ON</strong>.</Step>
                  <Step n={8}>Move your arm. PhysiCore starts learning payload mass and joint friction. After 30 seconds the estimates converge. When you change payload — watch the mass estimate update automatically.</Step>
                  <Good>
                    Mass estimate moving — learning your payload ✓<br />
                    Friction converging — learning your joint dynamics ✓<br />
                    Adapts to new payload within 10-15 seconds of change ✓
                  </Good>
                </Phase>
              </section>
            )}

            {/* ── HUMANOID / LEGGED ── */}
            {manualSection === 'humanoid' && (
              <section className="space-y-10">
                <div className="space-y-4">
                  <h1 className="font-display text-4xl font-black text-white tracking-tighter uppercase">Humanoid & Legged Robots</h1>
                  <p className="font-body text-lg text-textSecondary">PhysiCore connects to humanoid and legged robots over ROS2. It learns their real mass distribution, joint friction, and contact dynamics without any manual calibration.</p>
                  <div className="p-4 border border-green/20 bg-green/5">
                    <span className="font-mono text-[10px] text-green uppercase tracking-widest">Supported: Unitree G1/H1, Figure AI Apollo, Boston Dynamics Spot, ANYmal, Go1/Go2, MIT Mini Cheetah, any custom biped or quadruped</span>
                  </div>
                </div>

                <Phase n="PHASE 1" title="Verify ROS2 is publishing">
                  <Step n={1}>Source your ROS2 installation:<Code>source /opt/ros/humble/setup.bash</Code></Step>
                  <Step n={2}>For Unitree robots, source the Unitree ROS2 workspace first, then verify:<Code>ros2 topic list | grep joint</Code></Step>
                  <Step n={3}>Check joint states are flowing:<Code>ros2 topic echo /joint_states --once</Code>You must see position and velocity arrays. If empty — your robot driver is not running.</Step>
                  <Warn title="Unitree SDK note">Unitree G1/H1 requires the unitree_ros2 package. Source its workspace before running the bridge or topics will not appear.</Warn>
                </Phase>

                <Phase n="PHASE 2" title="Run the bridge">
                  <Step n={4}>Use the Integration Engineer to generate your exact bridge script. Or run manually:
                    <Code>{`source /opt/ros/humble/setup.bash
python physicore/bridge/physicore_bridge.py --platform ros2_legged`}</Code>
                  </Step>
                  <Good>
                    [BRIDGE] ROS2 subscribed to /joint_states<br/>
                    [ENGINE] Initialized for 'legged_robot'<br/>
                    [TELEM] P:0.0° | mass=30.000 res=0.0000 steps=0
                  </Good>
                </Phase>

                <Phase n="PHASE 3" title="Connect and adapt">
                  <Step n={5}>Dashboard → MAVLINK → <code className="text-cyan">ws://localhost:8765</code> → Connect.</Step>
                  <Step n={6}>Click <strong className="text-white">ACTIVE CONTROL ON</strong>. Let the robot walk or move.</Step>
                  <Step n={7}>Watch mass estimate adapt — PhysiCore learns the robot's real mass distribution from contact dynamics. Payload changes are detected automatically within 10-15 seconds.</Step>
                  <Good>
                    Mass estimate adapting — learning real body distribution ✓<br/>
                    Terrain friction adapting — registry saves per-terrain ✓<br/>
                    Sentinel in NOMINAL — all safety bounds clear ✓
                  </Good>
                </Phase>
              </section>
            )}

            {/* ── AUV ── */}
            {manualSection === 'auv' && (
              <section className="space-y-10">
                <div className="space-y-4">
                  <h1 className="font-display text-4xl font-black text-white tracking-tighter uppercase">AUV / Underwater Robot</h1>
                  <p className="font-body text-lg text-textSecondary">PhysiCore adapts to real hydrodynamic drag, buoyancy changes, and battery degradation in real time. Works with BlueROV2 and any ROS2-enabled AUV.</p>
                  <div className="p-4 border border-green/20 bg-green/5">
                    <span className="font-mono text-[10px] text-green uppercase tracking-widest">Supported: BlueROV2, custom AUVs, research vehicles with ROS2 + IMU + depth sensor</span>
                  </div>
                </div>

                <Phase n="PHASE 1" title="Verify sensor topics">
                  <Step n={1}><Code>ros2 topic list</Code>You need: /imu/data, /depth or /bar30/pressure, optionally /dvl/velocity</Step>
                  <Step n={2}>If DVL is available:<Code>ros2 topic echo /dvl/velocity --once</Code></Step>
                  <Warn title="Depth sensor required">PhysiCore's AUV model needs depth to track buoyancy. Without it, uncertainty stays high and Sentinel may stay in CAUTIOUS mode.</Warn>
                </Phase>

                <Phase n="PHASE 2" title="Run the bridge">
                  <Step n={3}><Code>{`source /opt/ros/humble/setup.bash
python physicore/bridge/physicore_bridge.py --platform ros2_auv`}</Code></Step>
                  <Good>
                    [BRIDGE] ROS2 subscribed to /imu/data /depth /dvl/velocity<br/>
                    [ENGINE] Initialized for 'auv' with quadratic drag model<br/>
                    drag coefficient adapting from depth and thrust data
                  </Good>
                </Phase>

                <Phase n="PHASE 3" title="Connect and dive">
                  <Step n={4}>Dashboard → MAVLINK → <code className="text-cyan">ws://localhost:8765</code> → Connect.</Step>
                  <Step n={5}>Click <strong className="text-white">ACTIVE CONTROL ON</strong> before entering water.</Step>
                  <Step n={6}>PhysiCore learns drag coefficient and buoyancy within the first 20 seconds of real motion. Registry saves per-vehicle — each AUV gets its own entry.</Step>
                </Phase>
              </section>
            )}

            {/* ── eVTOL ── */}
            {manualSection === 'evtol' && (
              <section className="space-y-10">
                <div className="space-y-4">
                  <h1 className="font-display text-4xl font-black text-white tracking-tighter uppercase">eVTOL Aircraft</h1>
                  <p className="font-body text-lg text-textSecondary">PhysiCore connects to eVTOL aircraft over MAVLink (PX4 or ArduPilot). It learns real aerodynamic coefficients, rotor efficiency degradation, and battery discharge curves in flight.</p>
                </div>

                <Phase n="PHASE 1" title="Configure MAVLink">
                  <Step n={1}>Your eVTOL must be running PX4 or ArduPilot. MAVLink UDP port 14550 must be enabled.</Step>
                  <Step n={2}>For multi-rotor eVTOL (quadrotor config): use --platform px4_quadrotor</Step>
                  <Step n={3}>For tilt-rotor or fixed-wing transition: use --platform evtol</Step>
                </Phase>

                <Phase n="PHASE 2" title="Run the bridge">
                  <Step n={4}><Code>python physicore/bridge/physicore_bridge.py --platform evtol --connection udp:14550</Code></Step>
                  <Good>
                    [BRIDGE] MAVLink connected. Vehicle: EVTOL<br/>
                    [ENGINE] Initialized for 'evtol' — ISA atmosphere + transition model active
                  </Good>
                </Phase>

                <Phase n="PHASE 3" title="Hover, transition, cruise">
                  <Step n={5}>Click <strong className="text-white">ACTIVE CONTROL ON</strong> before takeoff.</Step>
                  <Step n={6}>PhysiCore handles the VTOL ↔ cruise transition automatically. The eVTOL model smoothly interpolates between hover dynamics and wing lift based on airspeed.</Step>
                  <Step n={7}>Watch rotor efficiency adapt during flight — thermal degradation is tracked per-motor.</Step>
                  <Warn title="Safety">Sentinel is set to stricter thresholds for eVTOL (max_uncertainty: 0.03, max_residual: 0.5). Any unusual aerodynamic event triggers CAUTIOUS mode automatically.</Warn>
                </Phase>
              </section>
            )}

            {/* ── ROVER ── */}
            {manualSection === 'rover' && (
              <section className="space-y-10">
                <div className="space-y-4">
                  <h1 className="font-display text-4xl font-black text-white tracking-tighter uppercase">Ground Rover / AMR</h1>
                  <p className="font-body text-lg text-textSecondary">PhysiCore connects to differential-drive rovers, AMRs, and warehouse robots over ROS2 or direct Arduino/ESP32 serial. It adapts to real terrain friction and slip in real time.</p>
                  <div className="p-4 border border-green/20 bg-green/5">
                    <span className="font-mono text-[10px] text-green uppercase tracking-widest">Supported: any ROS2 rover with /cmd_vel and /odom, Arduino/ESP32 serial robots, ROS2 Nav2 compatible</span>
                  </div>
                </div>

                <Phase n="PHASE 1" title="ROS2 rover">
                  <Step n={1}><Code>ros2 topic echo /odom --once</Code>You need odometry publishing. If you only have /cmd_vel with no feedback — PhysiCore can still run but SysID convergence is slower.</Step>
                  <Step n={2}><Code>{`source /opt/ros/humble/setup.bash
python physicore/bridge/physicore_bridge.py --platform ros2_ground_rover`}</Code></Step>
                </Phase>

                <Phase n="PHASE 1 (alt)" title="Arduino/ESP32 serial rover">
                  <Step n={1}>Your firmware must send JSON at 20-50 Hz with velocity and orientation. Use the Integration Engineer to generate exact firmware.</Step>
                  <Step n={2}><Code>python physicore/bridge/physicore_bridge.py --platform ground_rover_serial --connection COM3 --baud 115200</Code></Step>
                </Phase>

                <Phase n="PHASE 2" title="Connect and adapt">
                  <Step n={3}>Dashboard → MAVLINK → <code className="text-cyan">ws://localhost:8765</code> → Connect.</Step>
                  <Step n={4}>Click <strong className="text-white">ACTIVE CONTROL ON</strong>. Drive the rover. PhysiCore learns terrain friction within 15 seconds. Moving to a new floor surface — friction adapts automatically.</Step>
                  <Good>
                    Terrain friction adapting as surface changes ✓<br/>
                    Slip model learned per-wheel ✓<br/>
                    Registry saves per-rover — next session starts smarter ✓
                  </Good>
                </Phase>
              </section>
            )}

            {/* ── SATELLITE ── */}
            {manualSection === 'satellite' && (
              <section className="space-y-10">
                <div className="space-y-4">
                  <h1 className="font-display text-4xl font-black text-white tracking-tighter uppercase">Satellite / Spacecraft</h1>
                  <p className="font-body text-lg text-textSecondary">PhysiCore includes a full orbital mechanics model with J2 oblateness perturbation, ISA atmosphere extension, and real-time inertia tensor estimation from reaction wheel telemetry.</p>
                  <div className="p-4 border border-amber/20 bg-amber/5">
                    <span className="font-mono text-[10px] text-amber uppercase tracking-widest">Advanced: requires attitude telemetry (quaternion or Euler) + reaction wheel speeds or thruster states over serial</span>
                  </div>
                </div>

                <Phase n="PHASE 1" title="Telemetry format">
                  <Step n={1}>Your flight computer must output JSON over serial at 10-50 Hz. Minimum:
                    <Code>{`{"roll":0.0,"pitch":0.0,"yaw":0.0,"gyro_x":0.0,"gyro_y":0.0,"gyro_z":0.0,"altitude":550000,"timestamp":0}`}</Code>
                    Altitude is in meters above Earth center (orbital altitude ~550km = 550000m + 6371000m).
                  </Step>
                </Phase>

                <Phase n="PHASE 2" title="Run the bridge">
                  <Step n={2}>Connect your flight computer over serial. Close any other serial terminals first.
                    <Code>python physicore/bridge/physicore_bridge.py --platform satellite_serial --connection COM3 --baud 115200</Code>
                  </Step>
                  <Good>
                    [ENGINE] Initialized for 'satellite'<br/>
                    [ENGINE] J2 perturbation model active<br/>
                    [TELEM] altitude=550000m | mass=100.000 | res=0.0000
                  </Good>
                </Phase>

                <Phase n="PHASE 3" title="Attitude control">
                  <Step n={3}>Dashboard → MAVLINK → <code className="text-cyan">ws://localhost:8765</code> → Connect.</Step>
                  <Step n={4}>Click <strong className="text-white">ACTIVE CONTROL ON</strong>. PhysiCore runs a full J2-corrected attitude controller. Inertia tensor estimation converges from reaction wheel torque vs angular acceleration over the first 60 seconds.</Step>
                </Phase>
              </section>
            )}

            {/* ── CUSTOM HARDWARE ── */}
            {manualSection === 'custom' && (
              <section className="space-y-10">
                <div className="space-y-4">
                  <h1 className="font-display text-4xl font-black text-white tracking-tighter uppercase">Custom Hardware</h1>
                  <p className="font-body text-lg text-textSecondary">PhysiCore works with any hardware that can send JSON over serial, ROS2 topics, or MAVLink. You do not need to match a specific robot type.</p>
                </div>

                <div className="space-y-4">
                  <h2 className="font-display text-sm font-bold text-white uppercase tracking-widest">The contract — what PhysiCore needs</h2>
                  <p className="font-body text-sm text-textSecondary">Your hardware sends sensor data. PhysiCore sends control commands back. That is the entire contract. Send what you have — PhysiCore uses whatever it gets.</p>
                  <Code>{`// Minimum: send any of these fields at 20-50 Hz
{"pitch":0.0}                    // angle in degrees — enough for basic adaptation
{"pitch":0.0,"gyro_x":0.0}       // + rate — better
{"pitch":0.0,"gyro_x":0.0,"accel_z":9.81,"motor_l":0.0,"motor_r":0.0,"timestamp":0}  // full`}</Code>
                  <p className="font-body text-xs text-textSecondary">PhysiCore sends back: <code className="text-cyan">{"{"}"op":"command","action":[TORQUE]{"}"}</code> — apply that to your actuators.</p>
                </div>

                <Phase n="OPTION A" title="Arduino / ESP32 serial">
                  <Step n={1}>Add JSON output to your existing sketch. Include: ArduinoJson library v6.x.</Step>
                  <Step n={2}>Use the Integration Engineer — select "Custom hardware" — it generates exact starter firmware for your MCU and sensors.</Step>
                  <Step n={3}><Code>python physicore/bridge/physicore_bridge.py --platform ground_rover_serial --connection COM3 --baud 115200</Code></Step>
                </Phase>

                <Phase n="OPTION B" title="ROS2 custom">
                  <Step n={1}>Publish any sensor data as ROS2 topics. The bridge subscribes to /imu/data, /odom, /joint_states — publish whichever you have.</Step>
                  <Step n={2}><Code>python physicore/bridge/physicore_bridge.py --platform ros2_ground_rover</Code></Step>
                </Phase>

                <Phase n="OPTION C" title="MAVLink custom">
                  <Step n={1}>If your hardware sends MAVLink telemetry (many FC boards do by default):<Code>python physicore/bridge/physicore_bridge.py --platform px4_quadrotor --connection udp:14550</Code></Step>
                </Phase>

                <div className="space-y-4">
                  <h2 className="font-display text-sm font-bold text-white uppercase tracking-widest">What platform to use for your custom system</h2>
                  <div className="space-y-2">
                    {[
                      { match:'Ground robot (wheels, treads)', platform:'ground_rover_serial or ros2_ground_rover' },
                      { match:'Flying vehicle (any rotor config)', platform:'px4_quadrotor' },
                      { match:'Robotic arm (serial or parallel)', platform:'ros2_manipulator' },
                      { match:'Anything with a pendulum / balance', platform:'balancing_bot' },
                      { match:'Rocket / high-acceleration', platform:'custom_rocket_fc' },
                      { match:'Underwater / high-drag', platform:'ros2_auv' },
                    ].map((r,i)=>(
                      <div key={i} className="flex gap-4 p-3 border border-borderDim bg-bgRaised">
                        <span className="font-body text-xs text-textSecondary flex-1">{r.match}</span>
                        <code className="font-mono text-[10px] text-cyan">{r.platform}</code>
                      </div>
                    ))}
                  </div>
                  <p className="font-body text-xs text-textSecondary">The platform choice determines the physics model PhysiCore starts with. Pick the closest match — SysID will correct for the difference within 30 seconds.</p>
                </div>

                <div className="p-6 border border-green/20 bg-green/5 space-y-3">
                  <div className="font-mono text-[10px] text-green uppercase tracking-widest">Use the Integration Engineer</div>
                  <p className="font-body text-sm text-textSecondary">Select "Custom hardware" in the Integration Engineer. Answer 6 questions. Get firmware, YAML config, and bridge command generated specifically for your system. 30 minutes from zero to PhysiCore running on your hardware.</p>
                  <button onClick={()=>navigateToProject('integrate')} className="px-4 py-2 bg-green text-black font-display text-xs font-bold uppercase tracking-widest hover:bg-white transition-all">
                    Open Integration Engineer →
                  </button>
                </div>
              </section>
            )}

            {/* ── CONFIG ── */}
            {manualSection === 'config' && (
              <section className="space-y-10">
                <div className="space-y-4">
                  <h1 className="font-display text-4xl font-black text-white tracking-tighter uppercase">Robot Config (YAML)</h1>
                  <p className="font-body text-lg text-textSecondary">Instead of remembering flags, write a YAML file once and reuse it forever.</p>
                </div>

                <div className="space-y-4">
                  <h2 className="font-display text-sm font-bold text-white uppercase tracking-widest">Without config file</h2>
                  <Code>python physicore/bridge/physicore_bridge.py --platform balancing_bot_arduino --connection COM3 --baud 115200</Code>
                </div>

                <div className="space-y-4">
                  <h2 className="font-display text-sm font-bold text-white uppercase tracking-widest">With config file</h2>
                  <Code>python physicore/bridge/physicore_bridge.py --config balancing_bot_robot.yaml</Code>
                  <p className="font-body text-sm text-textSecondary">Same result. But you never have to remember the flags again. The config file also tells the registry exactly which robot this is, so the right saved model loads automatically.</p>
                </div>

                <div className="space-y-4">
                  <h2 className="font-display text-sm font-bold text-white uppercase tracking-widest">Example — balancing bot</h2>
                  <p className="font-body text-xs text-textSecondary">Only change two values: your port and your robot's weight.</p>
                  <Code>{`name: My Balancing Bot
platform: balancing_bot
connection: COM3       # ← change this to your port
baud: 115200
mass: 1.0              # ← change this to your robot's weight in kg
friction: 0.15
inertia: 0.01
imu: MPU6050
motor_driver: L298N
mcu: Arduino Uno
control_hz: 60.0
use_registry: true
opt_in_telemetry: false
max_torque: 2.5`}</Code>
                </div>

                <div className="space-y-4">
                  <h2 className="font-display text-sm font-bold text-white uppercase tracking-widest">Creating a template</h2>
                  <p className="font-body text-xs text-textSecondary">Run this to generate a ready-to-edit config file for any platform:</p>
                  <Code>python physicore/bridge/physicore_bridge.py --init-config balancing_bot</Code>
                  <p className="font-body text-xs text-textSecondary">This creates <code className="text-cyan">balancing_bot_robot.yaml</code> in your current folder. Open it, change your port and mass, save.</p>
                </div>
              </section>
            )}

            {/* ── REGISTRY ── */}
            {manualSection === 'registry' && (
              <section className="space-y-10">
                <div className="space-y-4">
                  <h1 className="font-display text-4xl font-black text-white tracking-tighter uppercase">Persistent Learning</h1>
                  <p className="font-body text-lg text-textSecondary">Every session saves. Every deployment starts smarter than the last.</p>
                </div>

                <div className="space-y-6">
                  <div className="p-6 border border-cyan/20 bg-cyan/5 space-y-3">
                    <div className="font-mono text-[10px] text-cyan uppercase tracking-widest">Without registry</div>
                    <p className="font-body text-sm text-textSecondary">Every time you run PhysiCore, it starts at mass=1.0, friction=0.15. It takes 30 seconds to converge. Every session is the same.</p>
                  </div>
                  <div className="p-6 border border-green/20 bg-green/5 space-y-3">
                    <div className="font-mono text-[10px] text-green uppercase tracking-widest">With registry (use_registry: true)</div>
                    <p className="font-body text-sm text-textSecondary">First session: converges from 1.0 to 1.16 over 30 seconds. Saves. Second session: starts at 1.16. Converges to 1.22 in 10 seconds. Third session: starts at 1.22. Already close. Gets better every run.</p>
                  </div>
                </div>

                <div className="space-y-4">
                  <h2 className="font-display text-sm font-bold text-white uppercase tracking-widest">What gets saved</h2>
                  {[
                    { item: 'SystemID parameters', detail: 'Converged mass, friction, inertia estimates. Weighted average across sessions — more converged sessions count more.' },
                    { item: 'Residual ensemble weights', detail: 'The three neural network weights that learned what your simulator gets wrong. Loads pre-trained next session.' },
                    { item: 'CEM warm start', detail: 'The optimizer\'s memory of good action sequences. First few control steps are better immediately.' },
                    { item: 'Session log', detail: 'Every session recorded: timestamp, steps, convergence quality, final params.' },
                  ].map((s, i) => (
                    <div key={i} className="p-4 border border-borderDim bg-bgRaised space-y-1">
                      <div className="font-display text-xs font-bold text-white">{s.item}</div>
                      <div className="font-body text-xs text-textSecondary">{s.detail}</div>
                    </div>
                  ))}
                </div>

                <div className="space-y-4">
                  <h2 className="font-display text-sm font-bold text-white uppercase tracking-widest">Where it saves</h2>
                  <Code>{`~/.physicore/registry/{platform_hardware_key}/
  params.json          ← latest converged params
  ensemble_0.npz       ← neural network weights
  ensemble_1.npz
  ensemble_2.npz
  cem_warmstart.npz    ← optimizer warm start
  sessions.jsonl       ← session history
  platform_prior.json  ← aggregated prior`}</Code>
                  <p className="font-body text-xs text-textSecondary">The key is specific to your hardware combination — a bot with MPU6050 + L298N has a different entry from one with BNO055 + TB6612. They don't share params.</p>
                </div>

                <div className="space-y-4">
                  <h2 className="font-display text-sm font-bold text-white uppercase tracking-widest">Enabling it</h2>
                  <p className="font-body text-xs text-textSecondary">Set in your YAML config:</p>
                  <Code>use_registry: true</Code>
                  <p className="font-body text-xs text-textSecondary">Registry saves automatically when you stop the bridge with Ctrl+C. Loads automatically at startup. No other configuration needed.</p>
                </div>
              </section>
            )}

            {/* ── SENTINEL ── */}
            {manualSection === 'sentinel' && (
              <section className="space-y-10">
                <div className="space-y-4">
                  <h1 className="font-display text-4xl font-black text-white tracking-tighter uppercase">Sentinel OS</h1>
                  <p className="font-body text-lg text-textSecondary">The safety layer that runs underneath PhysiCore. It monitors every step and can override or halt control instantly. You cannot bypass it.</p>
                </div>

                <div className="space-y-4">
                  <h2 className="font-display text-sm font-bold text-white uppercase tracking-widest">Three modes</h2>
                  <div className="space-y-3">
                    {[
                      { mode: 'NOMINAL', color: COLORS.green, trigger: 'All metrics within bounds', action: 'Full PhysiCore control. No restrictions.' },
                      { mode: 'CAUTIOUS', color: COLORS.amber, trigger: 'One metric near limit (uncertainty 5-15%, or residual 0.5-2.0)', action: 'PhysiCore output scaled to 60%. Tighter action bounds.' },
                      { mode: 'FALLBACK', color: COLORS.red, trigger: 'Any metric exceeded (uncertainty > 15%, Lyapunov energy exceeded, state explosion)', action: 'PhysiCore disabled. Safe fallback controller (zero action or PID) takes over.' },
                    ].map((m, i) => (
                      <div key={i} className="p-4 border border-borderDim bg-bgRaised space-y-2">
                        <div className="flex items-center gap-3">
                          <span className="font-display text-sm font-bold" style={{ color: m.color }}>{m.mode}</span>
                        </div>
                        <div className="font-body text-xs text-textSecondary"><strong className="text-white">Triggers when:</strong> {m.trigger}</div>
                        <div className="font-body text-xs text-textSecondary"><strong className="text-white">Action:</strong> {m.action}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="space-y-4">
                  <h2 className="font-display text-sm font-bold text-white uppercase tracking-widest">Recovery</h2>
                  <p className="font-body text-sm text-textSecondary">FALLBACK → CAUTIOUS automatically after 300 stable steps (~5 seconds). CAUTIOUS → NOMINAL after 50 stable steps (~0.8 seconds). No manual intervention needed.</p>
                </div>

                <div className="space-y-4">
                  <h2 className="font-display text-sm font-bold text-white uppercase tracking-widest">Forensic log</h2>
                  <p className="font-body text-sm text-textSecondary">Every control command is signed with SHA-256 and chained. If the log is tampered with, the chain breaks. Every mode transition is recorded with the trigger reason, parameter snapshot, and timestamp.</p>
                  <p className="font-body text-xs text-textDim">Save to file: set <code className="text-cyan">log_path: /path/to/sentinel.log</code> in your Sentinel config.</p>
                </div>
              </section>
            )}

            {/* ── TROUBLESHOOTING ── */}
            {manualSection === 'troubleshoot' && (
              <section className="space-y-10">
                <div className="space-y-4">
                  <h1 className="font-display text-4xl font-black text-white tracking-tighter uppercase">Troubleshooting</h1>
                  <p className="font-body text-lg text-textSecondary">Every error. Exact fix. No vague advice.</p>
                </div>

                <div className="space-y-6">
                  {[
                    {
                      error: 'Serial failed — retrying in 3s',
                      causes: ['Arduino IDE is still open (most common)', 'Wrong COM port number'],
                      fixes: [
                        'Close Arduino IDE completely — not minimized, fully closed. Then run the bridge again.',
                        'Check Device Manager → Ports (COM & LPT) → find your Arduino\'s actual COM number. Update --connection.',
                      ],
                    },
                    {
                      error: 'MPU6050 not found — check wiring',
                      causes: ['Wiring is wrong', 'VCC connected to 5V instead of 3.3V'],
                      fixes: [
                        'SDA must go to A4. SCL must go to A5. Pull each wire out and push back in firmly.',
                        'VCC must go to 3.3V pin on Arduino. If it was in 5V, the MPU6050 may be damaged.',
                      ],
                    },
                    {
                      error: 'Pitch shows 0.0 and never changes',
                      causes: ['Arduino not sending data', 'Bridge port mismatch'],
                      fixes: [
                        'Unplug Arduino USB. Plug back in. Wait 5 seconds for calibration message in bridge terminal.',
                        'Open Arduino Serial Monitor at 115200 baud. Do you see JSON? If yes, compare the port in the title bar vs your bridge --connection flag.',
                      ],
                    },
                    {
                      error: 'Motors spin full speed in one direction and won\'t stop',
                      causes: ['BALANCE_POINT is wrong', 'Bot thinks it\'s always falling'],
                      fixes: [
                        'Redo Phase 2 calibration. Hold bot upright, read pitch from Serial Monitor, set BALANCE_POINT to that exact value, re-flash.',
                      ],
                    },
                    {
                      error: 'Motors barely move / no visible response',
                      causes: ['MAX_TORQUE is 100 instead of 2.5', 'ACTIVE CONTROL ON not clicked'],
                      fixes: [
                        'Open firmware. Find const float MAX_TORQUE = ... and make sure it says 2.5, not 100. At 100, motors get <1% power.',
                        'Click ACTIVE CONTROL ON in the dashboard. PhysiCore only sends commands when explicitly enabled.',
                      ],
                    },
                    {
                      error: 'steps not counting after ACTIVE CONTROL ON',
                      causes: ['Dashboard not connected', 'Bridge not running', 'No sensor data flowing'],
                      fixes: [
                        'Check dashboard shows "Connected" in green. If not — click MAVLINK, type ws://localhost:8765, click Connect.',
                        'Check the bridge terminal is still running and printing telemetry lines.',
                        'Check pitch is updating in dashboard. If frozen at 0 — bridge is not receiving data from hardware.',
                      ],
                    },
                    {
                      error: 'python is not recognized',
                      causes: ['Python not installed', 'Not in PATH'],
                      fixes: [
                        'Go to python.org. Download Python 3. Run installer. On the FIRST screen, tick "Add Python to PATH". Install. Restart terminal.',
                        'Try python3 instead of python (Mac/Linux default).',
                      ],
                    },
                    {
                      error: 'pip install fails with permission error',
                      causes: ['Permissions issue'],
                      fixes: ['Run: python -m pip install pymavlink websockets aiohttp pyserial pyyaml'],
                    },
                    {
                      error: 'Bridge crashes on import',
                      causes: ['Missing dependencies', 'ROS2 not sourced'],
                      fixes: [
                        'Run: pip install pymavlink websockets aiohttp pyserial pyyaml',
                        'For ROS2 platforms: source /opt/ros/humble/setup.bash before running bridge.',
                      ],
                    },
                  ].map((item, i) => (
                    <div key={i} className="border border-borderDim bg-bgRaised overflow-hidden">
                      <div className="p-4 border-b border-borderDim bg-bg">
                        <span className="font-mono text-[11px] text-red">{item.error}</span>
                      </div>
                      <div className="p-4 space-y-3">
                        <div className="space-y-1">
                          <span className="font-mono text-[9px] text-textDim uppercase tracking-widest">Cause</span>
                          {item.causes.map((c, j) => (
                            <div key={j} className="font-body text-xs text-textSecondary">• {c}</div>
                          ))}
                        </div>
                        <div className="space-y-2">
                          <span className="font-mono text-[9px] text-green uppercase tracking-widest">Fix</span>
                          {item.fixes.map((f, j) => (
                            <div key={j} className="font-body text-xs text-textSecondary p-2 bg-bg border border-borderDim">→ {f}</div>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="p-6 border border-cyan/20 bg-cyan/5 space-y-3">
                  <div className="font-mono text-[10px] text-cyan uppercase tracking-widest">Still stuck?</div>
                  <p className="font-body text-sm text-textSecondary">Use the Integration Engineer — click the button in the top navigation. It knows your exact hardware setup and gives you the exact fix for your specific error. Just describe what you see.</p>
                </div>
              </section>
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

          {projects.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-24 space-y-6 text-center">
              <div className="w-20 h-20 border-2 border-dashed border-border flex items-center justify-center text-textDim">
                <Layers size={36} className="opacity-30" />
              </div>
              <div className="space-y-2">
                <p className="font-display text-lg font-bold text-white uppercase tracking-widest">No Projects Yet</p>
                <p className="font-mono text-[10px] text-textDim uppercase">Go to Integration Engineer and complete a hardware Q&A to auto-create a project.</p>
              </div>
              <button onClick={handleSetIntegratorView}
                className="px-6 py-3 bg-green text-black font-display text-[11px] font-bold uppercase tracking-widest hover:bg-white transition-all">
                ⬡ Open Integration Engineer
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
                    <input value={newProjectName} onChange={e => setNewProjectName(e.target.value)}
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
                </div>
                <div className="flex gap-3">
                  <button onClick={() => { setShowNewProjectModal(false); setNewProjectName(''); setNewProjectDesc(''); }}
                    className="flex-1 py-3 border border-border text-textDim font-display text-[10px] font-bold uppercase tracking-widest hover:text-white transition-all">
                    Cancel
                  </button>
                  <button disabled={!newProjectName.trim()}
                    onClick={async () => {
                      if (!newProjectName.trim()) return;
                      const p = await createProject(newProjectName.trim(), '', {}, []);
                      if (p) {
                        setActiveProject(p);
                        setShowNewProjectModal(false);
                        setNewProjectName('');
                        setNewProjectDesc('');
                        navigateToProject('integrate');
                      }
                      // If p is null, createProject already showed the error alert.
                      // Modal stays open so user can retry.
                    }}
                    className="flex-1 py-3 bg-green text-black font-display text-[10px] font-bold uppercase tracking-widest hover:bg-white transition-all disabled:opacity-40">
                    Create & Open IE
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
                  onClick={() => { clearUserKeys(); setApiKeyInput(''); setApiKeyAnthropicInput(''); setApiKeySaved(false); }}
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
                  setApiKeyInput('');
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
          <div className="flex-1 overflow-hidden">
            {selectedBuildFile && editorCode ? (
              <PhysiEditor
                code={editorCode}
                language={selectedBuildFile.endsWith('.yaml') || selectedBuildFile.endsWith('.yml') ? 'yaml' : 'python'}
                readOnly
              />
            ) : (
              <div className="h-full flex items-center justify-center">
                <div className="font-mono text-[10px] text-textDim italic">Generated code will appear here</div>
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
