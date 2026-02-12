
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
          <span className="w-2 h-2 rounded-full bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)]"></span>
          Dynamics Residual
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
              <Area type="step" dataKey="value" stroke="#6366f1" strokeWidth={1} fillOpacity={1} fill="url(#colorErr)" animationDuration={300} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 text-xl font-bold text-white tabular-nums">
          {telemetry.length > 0 ? telemetry[telemetry.length-1].value.toExponential(4) : '0.0000'}
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
      </div>

      <div className="bg-slate-900/40 p-4 rounded-sm border border-white/5 flex flex-col">
        <h3 className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2">Architectural Priors</h3>
        <div className="space-y-3 mt-2 flex-1">
          <div className="flex justify-between items-center text-[10px]">
            <span className="text-slate-500">Integrator</span>
            <span className="text-indigo-400 font-bold">RK-4</span>
          </div>
          <div className="flex justify-between items-center text-[10px]">
            <span className="text-slate-500">Uncertainty</span>
            <span className="text-pink-400 font-bold uppercase">Epistemic</span>
          </div>
          <div className="flex justify-between items-center text-[10px]">
            <span className="text-slate-500">Loop Rate</span>
            <span className="text-emerald-400 font-bold">100Hz</span>
          </div>
          <div className="pt-4 mt-auto">
            <div className="flex justify-between text-[8px] mb-1"><span>BUFFER UTILIZATION</span><span>72%</span></div>
            <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
              <div className="bg-indigo-600 h-full w-[72%]"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
