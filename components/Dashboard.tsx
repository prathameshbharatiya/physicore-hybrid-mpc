
import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TelemetryPoint } from '../types';

interface DashboardProps {
  telemetry: TelemetryPoint[];
  avgVelocity: number;
  stability: number;
}

const Dashboard: React.FC<DashboardProps> = ({ telemetry, avgVelocity, stability }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 h-full">
      <div className="bg-slate-900/40 p-4 rounded-sm border border-white/5 flex flex-col">
        <h3 className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-4 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)] animate-pulse"></span>
          Dynamics Residual (L2)
        </h3>
        <div className="flex-1 min-h-[140px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={telemetry.slice(-40)}>
              <defs>
                <linearGradient id="colorErr" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="time" hide />
              <YAxis hide domain={[0, 'auto']} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#020617', border: '1px solid #1e293b', fontSize: '9px', fontFamily: 'monospace' }}
                itemStyle={{ color: '#6366f1' }}
              />
              <Area type="monotone" dataKey="value" stroke="#6366f1" strokeWidth={1} fillOpacity={1} fill="url(#colorErr)" animationDuration={100} isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 flex justify-between items-baseline">
          <span className="text-xl font-bold text-white tabular-nums">
            {telemetry.length > 0 ? telemetry[telemetry.length-1].value.toExponential(4) : '0.0000'}
          </span>
          <span className="text-[8px] text-indigo-400 font-mono">RESIDUAL_FLUX</span>
        </div>
      </div>

      <div className="bg-slate-900/40 p-4 rounded-sm border border-white/5 flex flex-col">
        <h3 className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-4 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]"></span>
          Control Stability
        </h3>
        <div className="flex-1 flex items-center justify-center">
           <div className="relative w-28 h-28">
             <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
               <circle cx="18" cy="18" r="16" fill="none" className="stroke-slate-800" strokeWidth="2" />
               <circle 
                 cx="18" cy="18" r="16" fill="none" 
                 className="stroke-emerald-500 transition-all duration-500" 
                 strokeWidth="2" 
                 strokeDasharray={`${stability}, 100`} 
                 strokeLinecap="round" 
               />
             </svg>
             <div className="absolute inset-0 flex items-center justify-center flex-col">
               <span className="text-xl font-bold text-white tabular-nums">{Math.round(stability)}%</span>
               <span className="text-[8px] text-slate-600 uppercase tracking-tighter">System Nominal</span>
             </div>
           </div>
        </div>
        <div className="mt-4 border-t border-white/5 pt-2 grid grid-cols-2 gap-2">
           <div className="text-[9px] text-slate-500 uppercase">Solver State: <span className="text-emerald-500 font-bold">RK4_STABLE</span></div>
           <div className="text-[9px] text-slate-500 uppercase">J_Cost: <span className="text-emerald-500 font-bold">{(Math.random() * 1.5).toFixed(3)}</span></div>
        </div>
      </div>

      <div className="bg-slate-900/40 p-4 rounded-sm border border-white/5 flex flex-col">
        <h3 className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2">Simulation Architecture</h3>
        <div className="space-y-2.5 mt-2 flex-1 overflow-y-auto custom-scroll">
          <div className="flex justify-between items-center text-[10px]">
            <span className="text-slate-500">Rigid Logic</span>
            <span className="text-indigo-400 font-bold">Velocity Verlet</span>
          </div>
          <div className="flex justify-between items-center text-[10px]">
            <span className="text-slate-500">Soft-Body</span>
            <span className="text-pink-400 font-bold">96 Node Mesh</span>
          </div>
          <div className="flex justify-between items-center text-[10px]">
            <span className="text-slate-500">Fluids</span>
            <span className="text-cyan-400 font-bold">Lagrangian SPH</span>
          </div>
          <div className="flex justify-between items-center text-[10px]">
            <span className="text-slate-500">Residuals</span>
            <span className="text-white font-mono">Ensemble MLP</span>
          </div>
          <div className="pt-2 mt-auto border-t border-white/5">
            <div className="flex justify-between text-[8px] mb-1 text-slate-400 uppercase font-bold tracking-widest">Training Buffer</div>
            <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
              <div className="bg-indigo-600 h-full w-[68%] shadow-[0_0_8px_#4f46e5]"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
