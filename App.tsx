import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Activity, Cpu, Shield, Zap, ChevronRight, ChevronLeft, 
  Play, Download, Terminal, AlertTriangle, CheckCircle2, 
  Settings, Crosshair, ArrowUpRight, ArrowDownRight, X,
  Maximize2, Activity as FrequencyIcon, RefreshCw, Globe,
  Link, Wifi, Radio, HardDrive, FileJson, Copy, Check,
  ArrowRight, MousePointer2, Layers, BarChart3, ShieldCheck,
  Code2, MessageSquare, DownloadCloud, ExternalLink, ChevronDown
} from 'lucide-react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, 
  ResponsiveContainer, BarChart, Bar, ReferenceLine
} from 'recharts';

import { GoogleGenAI } from "@google/genai";

// --- CONSTANTS ---
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "YOUR_API_KEY";
const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

const COLORS = {
  void: '#080808', bg: '#0C0C0C', bgRaised: '#111111', bgInset: '#0A0A0A',
  borderDim: '#1E1E1E', border: '#2A2A2A', borderActive: '#3D3D3D',
  textPrimary: '#EFEFEF', textSecondary: '#7A7A7A', textDim: '#444444',
  green: '#00FF88', greenDim: '#003320', amber: '#FFB800', amberDim: '#1A1000',
  red: '#FF2222', redDim: '#1A0000', blue: '#0099FF', blueDim: '#001020',
  white: '#FFFFFF', cyan: '#00DDCC',
};

// --- TYPES ---
type View = 'home' | 'integrator' | 'dashboard';
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

// --- COMPONENTS ---

const SyntaxHighlighter = ({ code }: { code: string }) => {
  const highlightCode = (text: string) => {
    // Escape HTML
    let escaped = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");

    // Keywords
    const keywords = /\b(def|import|from|class|if|else|return|async|await|try|except|with|as|for|in|while|pass|break|continue|yield|lambda|global|nonlocal|assert|del|is|not|and|or|True|False|None|public|private|protected|static|void|int|float|double|bool|string|char|using|namespace|std|vector|cout|endl|ros|rclcpp|node|publisher|subscriber|timer|callback|msg|srv|action|uint32_t|float32_t|double_t|bool_t|auto|const|constexpr|struct|enum|union|typedef|extern|inline|virtual|override|final|explicit|mutable|volatile|register|thread_local|alignas|alignof|sizeof|typeid|typename|template|concept|requires|decltype|noexcept|static_assert|static_cast|dynamic_cast|const_cast|reinterpret_cast|new|delete|this|throw|try|catch|operator|friend|export|module|import|co_await|co_yield|co_return)\b/g;
    escaped = escaped.replace(keywords, `<span style="color: ${COLORS.cyan}">$1</span>`);

    // Strings
    escaped = escaped.replace(/(&quot;.*?&quot;|&#039;.*?&#039;)/g, `<span style="color: #88DD88">$1</span>`);

    // Comments
    escaped = escaped.replace(/(\/\/.*|#.*|\/\*[\s\S]*?\*\/)/g, `<span style="color: ${COLORS.textDim}; font-style: italic">$1</span>`);

    // Numbers
    escaped = escaped.replace(/\b(\d+)\b/g, `<span style="color: ${COLORS.amber}">$1</span>`);

    // Types/Classes (approximate)
    escaped = escaped.replace(/\b([A-Z][a-zA-Z0-9_]*)\b/g, `<span style="color: ${COLORS.blue}">$1</span>`);

    return escaped;
  };

  return (
    <pre className="font-mono text-[11px] leading-relaxed whitespace-pre overflow-x-auto">
      <code dangerouslySetInnerHTML={{ __html: highlightCode(code) }} />
    </pre>
  );
};

const IntegrationActionPanel = ({ files, onTest, onContinue }: { files: GeneratedFile[], onTest: () => void, onContinue: () => void }) => {
  const [copied, setCopied] = useState(false);

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
          className="flex items-center justify-between p-4 border border-green/30 bg-bgRaised hover:bg-green hover:text-black transition-all group"
        >
          <div className="flex flex-col items-start">
            <span className="font-display text-[11px] font-bold tracking-widest uppercase">TEST CONNECTION</span>
            <span className="font-mono text-[9px] text-textDim group-hover:text-black/60 uppercase">Verify HIL / Digital Twin link</span>
          </div>
          <Wifi size={20} />
        </button>

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

// --- MAIN APP ---

export default function App() {
  const [view, setView] = useState<View>('home');
  const [activeSection, setActiveSection] = useState('overview');
  
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

  const [isLaunching, setIsLaunching] = useState(false);

  const handleLaunchApp = () => {
    setIsLaunching(true);
    setTimeout(() => {
      setIsLaunching(false);
      setView('dashboard');
    }, 3000);
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
  }, [view]);

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

  // AI Logic
  const handleSendMessage = async (text?: string) => {
    const msg = text || inputValue;
    if (!msg.trim()) return;

    const userMsg: Message = { role: 'user', content: msg, timestamp: formatTime(new Date()) };
    const newHistory = [...conversationHistory, userMsg];
    setConversationHistory(newHistory);
    setInputValue('');
    setIsTyping(true);

    try {
      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: newHistory.map(m => ({
          role: m.role === 'ai' ? 'model' : 'user',
          parts: [{ text: m.content }]
        })),
        config: {
          systemInstruction: `You are the PhysiCore Integration Engineer.
           You help robotics engineers integrate PhysiCore — a Physics Intelligence Engine — into their systems.

           PhysiCore capabilities you're integrating:
           — RK4 4th-order physics integrator at 60Hz
           — Online SystemID: learns mass and friction in real-time
           — 3-node ensemble: quantifies epistemic uncertainty
           — MPC lookahead: 12-step CEM trajectory planning
           — Hardware gate: LIVE / HIL / Digital Twin modes
           — Sentinel OS integration: safety governance layer
           — Export: JSON pack + ROS2 bridge code

           YOUR BEHAVIOR:
           — Follow the workflow phases strictly:
             Discovery (6 questions) → Confirm → Generate → Guide → Q&A
           — Ask ONE question at a time. Never multiple.
           — When generating code: inject real values, never placeholders.
           — Code must be complete and copy-paste ready.
           — Never use markdown fences (code blocks) in responses.
             Format code with inline syntax highlighting hints instead:
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
           CURRENT DATE: ${new Date().toISOString()}`,
          temperature: 0.2,
          maxOutputTokens: 2000,
        }
      });

      const aiText = response.text || "> INTEGRATION ENGINEER: NO RESPONSE RECEIVED.";
      const aiMsg: Message = { role: 'ai', content: aiText, timestamp: formatTime(new Date()) };
      setConversationHistory(prev => [...prev, aiMsg]);
      
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
        setIntegrationPhase(4); // Transition to action panel phase
      }

    } catch (error) {
      console.error("AI Error:", error);
      const errorMsg: Message = { role: 'ai', content: "> INTEGRATION ENGINEER: ERROR DETECTED. SYSTEM OFFLINE. PLEASE RETRY.", timestamp: formatTime(new Date()) };
      setConversationHistory(prev => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleTestConnection = () => {
    const testMsg: Message = { 
      role: 'ai', 
      content: `> INTEGRATION ENGINEER: INITIATING CONNECTION TEST...
      [SYSTEM] PINGING PHYSICORE KERNEL... SUCCESS (0.4ms)
      [SYSTEM] VERIFYING SENTINEL HANDSHAKE... SUCCESS
      [SYSTEM] CHECKING ROS2 BRIDGE... ACTIVE
      [SYSTEM] STREAMING TELEMETRY... 60Hz STABLE
      
      CONNECTION VERIFIED. SYSTEM IS LIVE.`, 
      timestamp: formatTime(new Date()) 
    };
    setConversationHistory(prev => [...prev, testMsg]);
    setIsSystemConnected(true);
  };

  // --- RENDERERS ---

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
      </div>

      {view !== 'dashboard' && (
        <div className="hidden lg:flex items-center gap-8">
          {['OVERVIEW', 'ARCHITECTURE', 'FEATURES', 'BENCHMARKS', 'SENTINEL'].map(item => (
            <a 
              key={item} 
              href={`#${item.toLowerCase()}`}
              className={`font-body text-[11px] uppercase tracking-widest transition-colors ${activeSection === item.toLowerCase() ? (item === 'SENTINEL' ? 'text-amber border-b border-amber' : 'text-green border-b border-green') : 'text-textSecondary hover:text-textPrimary'}`}
              onClick={(e) => {
                e.preventDefault();
                if (view !== 'home') {
                  setView('home');
                  setTimeout(() => {
                    document.getElementById(item.toLowerCase())?.scrollIntoView({ behavior: 'smooth' });
                  }, 100);
                } else {
                  document.getElementById(item.toLowerCase())?.scrollIntoView({ behavior: 'smooth' });
                }
              }}
            >
              {item}
            </a>
          ))}
        </div>
      )}

      <div className="flex items-center gap-4">
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
              onClick={() => setView('integrator')}
              className={`px-4 py-1.5 font-display text-[11px] font-bold uppercase tracking-widest transition-all ${view === 'integrator' ? 'bg-white text-black' : 'bg-green text-black hover:bg-white'}`}
            >
              ⬡ INTEGRATION ENGINEER
            </button>
            <button 
              onClick={handleLaunchApp}
              className="hidden sm:block px-4 py-1.5 border border-border font-display text-[11px] font-bold uppercase tracking-widest text-textSecondary hover:text-textPrimary transition-all"
            >
              ▣ LAUNCH APP
            </button>
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
            Physics Intelligence Engine for robotics deployment. <br />
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
            <button onClick={() => setView('integrator')} className="btn-primary h-14 text-sm">⬡ START INTEGRATION →</button>
            <button onClick={handleLaunchApp} className="btn-outline h-14 text-sm">▣ LAUNCH PHYSICORE APP</button>
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
              <p>Every robotics deployment hits the same wall. The physics that worked in simulation — the perfectly tuned mass, the ideal friction, the clean actuator response — fails the moment it meets real hardware.</p>
              <p>The floor isn't as smooth as the model. The payload shifts. The motors degrade. The gap between what your simulation predicts and what your hardware does compounds with every iteration.</p>
              <p>PhysiCore eliminates this gap in real-time. Not by making better simulations — by making the simulation learn the reality it's deployed into.</p>
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
              { title: 'Hardware Gate', color: COLORS.textSecondary, desc: 'Physics does not run until hardware is connected. Live ROS2, HIL simulation with processor-accurate latency profiles, or Digital Twin mode.', spec: 'LIVE → VERIFIED | HIL → HIL_VALIDATED' },
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
                    ['CERTIFICATION', 'VERIFIED / HIL / TWIN'],
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
                <div className="flex justify-between gap-10"><span>STATUS:</span> <span className="text-green">VERIFIED</span></div>
                <div className="flex justify-between gap-10"><span>GOVERNANCE:</span> <span>ACTIVE</span></div>
              </div>
              <button className="btn-outline border-amber text-amber w-full hover:bg-amber hover:text-black">LEARN ABOUT SENTINEL →</button>
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
              <button className="text-left hover:text-white transition-colors">▣ Launch App</button>
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

  const renderIntegrator = () => (
    <div className="pt-[52px] h-screen flex bg-void overflow-hidden">
      {/* LEFT PANEL */}
      <aside className="w-[300px] border-r border-border bg-bg flex flex-col overflow-hidden">
        <div className="h-12 border-b border-border flex items-center justify-between px-4 shrink-0">
          <div className="flex flex-col">
            <span className="font-display text-xs font-bold text-cyan tracking-widest uppercase">⬡ INTEGRATION ENGINEER</span>
            <span className="font-body text-[9px] text-textDim uppercase">PHYSICORE v3.0</span>
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
                {integrationPhase === 4 && (
                  <IntegrationActionPanel 
                    files={generatedFiles} 
                    onTest={handleTestConnection}
                    onContinue={() => setView('dashboard')}
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
            onKeyDown={e => e.key === 'Enter' && handleSendMessage()}
            className="flex-1 bg-transparent border-b border-border py-2 font-mono text-xs text-cyan outline-none focus:border-cyan transition-colors"
            placeholder="> DESCRIBE YOUR SYSTEM OR ASK ANYTHING..."
          />
          <button onClick={() => handleSendMessage()} className="font-display text-xs font-bold text-cyan uppercase tracking-widest border border-cyan px-6 py-2 hover:bg-cyan hover:text-black transition-all">⏎ SEND</button>
        </div>
      </div>
    </div>
  );

  const renderDashboard = () => (
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
            <section className="space-y-4">
              <div className="border-l-2 border-cyan pl-3">
                <span className="micro-label">SystemID Estimates</span>
              </div>
              <div className="space-y-4">
                {[
                  { label: 'ESTIMATED MASS', val: '2.714 kg', delta: '+8.5%', color: COLORS.green },
                  { label: 'FRICTION COEFF', val: '0.412 μ', delta: '+17.7%', color: COLORS.amber },
                  { label: 'ACTUATOR EFF', val: '94.2%', delta: '-1.2%', color: COLORS.cyan },
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
                  <span className="font-mono text-[9px] text-green">NOMINAL</span>
                </div>
                <div className="h-1 w-full bg-border">
                  <div className="h-full bg-amber w-[85%]" />
                </div>
                <div className="font-mono text-[9px] text-textSecondary uppercase">Energy: 12.4J / 15.0J</div>
              </div>
              <div className="space-y-2">
                {isSystemConnected ? [
                  { time: '23:34:01', msg: 'LYAPUNOV_BOUND_CHECK: OK' },
                  { time: '23:34:05', msg: 'MPC_TRAJECTORY_VALIDATED' },
                  { time: '23:34:12', msg: 'SYSID_CONVERGENCE_STABLE' },
                  { time: '23:34:18', msg: 'SENTINEL_HEARTBEAT_ACK' },
                ].map((log, i) => (
                  <div key={i} className="flex gap-3 font-mono text-[9px]">
                    <span className="text-textDim">{log.time}</span>
                    <span className="text-textSecondary">{log.msg}</span>
                  </div>
                )) : (
                  <div className="flex items-center justify-center h-20 border border-dashed border-borderDim">
                    <span className="font-mono text-[9px] text-textDim uppercase">Waiting for telemetry...</span>
                  </div>
                )}
              </div>
            </section>
          </div>
        </aside>

        {/* MAIN CONTENT: SIMULATION & CHARTS */}
        <main className="flex-1 flex flex-col bg-bgInset overflow-hidden">
          {/* SIMULATION VIEW */}
          <div className="flex-1 relative border-b border-border">
            <div className="absolute top-6 left-6 z-10 space-y-2">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 border border-green flex items-center justify-center text-green">
                  <Maximize2 size={20} />
                </div>
                <div>
                  <h2 className="font-display text-lg font-bold text-white tracking-widest uppercase">Live Simulation View</h2>
                  <p className="font-mono text-[10px] text-textDim uppercase">RK4 Integrator @ 60Hz | MPC Lookahead</p>
                </div>
              </div>
            </div>
            
            <div className="absolute top-6 right-6 z-10 flex gap-2">
              <button className="p-2 border border-border bg-bgRaised text-textSecondary hover:text-green transition-all">
                <RefreshCw size={16} />
              </button>
              <button className="p-2 border border-border bg-bgRaised text-textSecondary hover:text-green transition-all">
                <Settings size={16} />
              </button>
            </div>

            <DashboardCanvas isConnected={isSystemConnected} />

            {!isSystemConnected && (
              <div className="absolute inset-0 z-20 bg-void/40 backdrop-blur-[2px] flex items-center justify-center">
                <div className="p-8 border border-border bg-bg/90 max-w-[320px] text-center space-y-6">
                  <div className="w-12 h-12 border border-textDim flex items-center justify-center text-textDim mx-auto">
                    <Wifi size={24} className="animate-pulse" />
                  </div>
                  <div className="space-y-2">
                    <h3 className="font-display text-lg font-bold text-white tracking-widest uppercase">Hardware Offline</h3>
                    <p className="font-body text-xs text-textSecondary leading-relaxed">
                      Physics intelligence engine is idle. Connect your system via the Integration Engineer to stream live telemetry.
                    </p>
                  </div>
                  <button 
                    onClick={() => setView('integrator')}
                    className="w-full py-2 bg-white text-black font-display text-[11px] font-bold uppercase tracking-widest hover:bg-green transition-all"
                  >
                    ⬡ OPEN INTEGRATOR
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
                    <div className="font-mono text-xs text-green">0.014</div>
                  </div>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="p-4 bg-bg/80 backdrop-blur-md border border-border flex items-center gap-4">
                  <div className="text-right">
                    <div className="micro-label text-textDim">Target Position</div>
                    <div className="font-mono text-xs text-white">X: 412.4 Y: 188.2</div>
                  </div>
                  <div className="w-10 h-10 border border-border flex items-center justify-center text-textDim">
                    <Crosshair size={20} />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* BOTTOM PANEL: CHARTS */}
          <div className="h-[280px] bg-bg border-t border-border flex divide-x divide-border">
            <div className="flex-1 p-6 space-y-4">
              <div className="flex justify-between items-center">
                <span className="micro-label text-cyan">L2 Prediction Residual</span>
                <span className="font-mono text-[10px] text-textDim">CONVERGENCE: 0.023</span>
              </div>
              <div className="h-[180px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={Array.from({ length: 20 }, (_, i) => ({ x: i, y: Math.exp(-i / 5) * 0.8 + Math.random() * 0.05 }))}>
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
            <div className="flex-1 p-6 space-y-4">
              <div className="flex justify-between items-center">
                <span className="micro-label text-green">Control Effort (N)</span>
                <span className="font-mono text-[10px] text-textDim">PEAK: 14.2N</span>
              </div>
              <div className="h-[180px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={Array.from({ length: 20 }, (_, i) => ({ x: i, y: 5 + Math.random() * 10 }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1A1A1A" vertical={false} />
                    <Bar dataKey="y" fill={COLORS.green} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );

  return (
    <div className="w-full h-full">
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

const DashboardCanvas = ({ isConnected }: { isConnected: boolean }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let frame = 0;
    let target = { x: canvas.width / 2, y: canvas.height / 2 };
    let robot = { x: canvas.width / 2, y: canvas.height / 2 };

    const resize = () => {
      const parent = canvas.parentElement;
      if (parent) {
        canvas.width = parent.clientWidth;
        canvas.height = parent.clientHeight;
      }
    };

    window.addEventListener('resize', resize);
    resize();

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

      if (!isConnected) {
        // Static noise/searching effect when disconnected
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        for (let i = 0; i < 5; i++) {
          const x = Math.random() * canvas.width;
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
        }
        requestAnimationFrame(animate);
        return;
      }

      // Target
      if (frame % 120 === 0) {
        target = { x: Math.random() * canvas.width, y: Math.random() * canvas.height };
      }

      // Robot Physics (Simplified RK4-ish feel)
      const dx = target.x - robot.x;
      const dy = target.y - robot.y;
      robot.x += dx * 0.05;
      robot.y += dy * 0.05;

      // MPC Trajectory (Prediction)
      ctx.strokeStyle = COLORS.cyan;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(robot.x, robot.y);
      let px = robot.x;
      let py = robot.y;
      for (let i = 0; i < 12; i++) {
        px += (target.x - px) * 0.1;
        py += (target.y - py) * 0.1;
        ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Ensemble Nodes (Shadows)
      [0.8, 1.2].forEach(scale => {
        ctx.strokeStyle = 'rgba(0, 255, 136, 0.1)';
        ctx.beginPath();
        ctx.arc(robot.x + (Math.random() - 0.5) * 10, robot.y + (Math.random() - 0.5) * 10, 15, 0, Math.PI * 2);
        ctx.stroke();
      });

      // Robot
      ctx.strokeStyle = COLORS.green;
      ctx.lineWidth = 2;
      ctx.strokeRect(robot.x - 20, robot.y - 20, 40, 40);
      ctx.fillStyle = COLORS.greenDim;
      ctx.fillRect(robot.x - 20, robot.y - 20, 40, 40);
      
      // Target Reticle
      ctx.strokeStyle = COLORS.amber;
      ctx.beginPath();
      ctx.arc(target.x, target.y, 10, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(target.x - 15, target.y); ctx.lineTo(target.x + 15, target.y);
      ctx.moveTo(target.x, target.y - 15); ctx.lineTo(target.x, target.y + 15);
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
