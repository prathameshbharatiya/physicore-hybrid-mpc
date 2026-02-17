
import React from 'react';
import { PhysicalParams } from '../types';

interface ParameterEditorProps {
  params: PhysicalParams;
  onChange: (updates: Partial<PhysicalParams>) => void;
  onReset: () => void;
}

const ParameterEditor: React.FC<ParameterEditorProps> = ({ params, onChange, onReset }) => {
  return (
    <div className="bg-slate-900/40 p-5 rounded-lg border border-white/5 space-y-5 shadow-2xl">
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Hardware Priors</h3>
        <span className="text-[8px] px-1.5 py-0.5 bg-slate-800 text-slate-500 rounded font-mono">CALIBRATED</span>
      </div>
      
      <div className="space-y-5">
        <div className="group">
          <div className="flex justify-between text-[10px] mb-2 font-mono">
            <span className="text-slate-400 group-hover:text-indigo-400 transition-colors">Base Mass (kg)</span>
            <span className="text-white font-bold">{params.mass.toFixed(3)}</span>
          </div>
          <input 
            type="range" min="0.1" max="5.0" step="0.01" 
            value={params.mass} 
            onChange={(e) => onChange({ mass: parseFloat(e.target.value) })}
            className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
          />
        </div>

        <div className="group">
          <div className="flex justify-between text-[10px] mb-2 font-mono">
            <span className="text-slate-400 group-hover:text-indigo-400 transition-colors">Static Friction (μ)</span>
            <span className="text-white font-bold">{params.friction.toFixed(3)}</span>
          </div>
          <input 
            type="range" min="0" max="1.0" step="0.005" 
            value={params.friction} 
            onChange={(e) => onChange({ friction: parseFloat(e.target.value) })}
            className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
          />
        </div>

        <div className="group">
          <div className="flex justify-between text-[10px] mb-2 font-mono">
            <span className="text-slate-400 group-hover:text-pink-400 transition-colors">Tension Matrix (k)</span>
            <span className="text-white font-bold">{params.textile_k.toFixed(1)}</span>
          </div>
          <input 
            type="range" min="10" max="1000" step="1" 
            value={params.textile_k} 
            onChange={(e) => onChange({ textile_k: parseFloat(e.target.value) })}
            className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-pink-600"
          />
        </div>
      </div>

      <button 
        onClick={onReset}
        className="w-full py-2.5 bg-slate-800/50 border border-slate-700 text-slate-500 text-[9px] font-bold uppercase tracking-widest hover:border-slate-500 hover:text-white hover:bg-slate-800 transition-all rounded mt-2"
      >
        Reset to Defaults
      </button>
    </div>
  );
};

export default ParameterEditor;
