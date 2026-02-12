
import React, { useEffect, useRef } from 'react';
import { SimMode, SimState, StateVector, ControlInput } from '../types';
import { stepDynamicsRK4 } from '../services/physicsLogic';

declare const Matter: any;

interface SimulationCanvasProps {
  mode: SimMode;
  onStateUpdate: (state: SimState) => void;
  target: [number, number];
  controlAction: ControlInput;
  physicsPriors: { mass: number; friction: number; gravity: number };
}

const SimulationCanvas: React.FC<SimulationCanvasProps> = ({ mode, onStateUpdate, target, controlAction, physicsPriors }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<any>(null);
  const robotRef = useRef<any>(null);

  useEffect(() => {
    if (!containerRef.current || !canvasRef.current || typeof Matter === 'undefined') return;

    const engine = Matter.Engine.create();
    engineRef.current = engine;
    engine.world.gravity.y = 0; 

    // Use getBoundingClientRect for more accurate sizing, with fallbacks
    const rect = containerRef.current.getBoundingClientRect();
    const actualWidth = rect.width || window.innerWidth * 0.6 || 800;
    const actualHeight = rect.height || window.innerHeight * 0.6 || 600;

    const render = Matter.Render.create({
      element: containerRef.current,
      canvas: canvasRef.current,
      engine: engine,
      options: {
        width: actualWidth,
        height: actualHeight,
        wireframes: false,
        background: 'transparent',
      }
    });
    
    const robot = Matter.Bodies.rectangle(actualWidth / 2, actualHeight / 2, 60, 30, {
      friction: 0,
      airFriction: 0,
      restitution: 0.1,
      render: { fillStyle: '#4f46e5', strokeStyle: '#818cf8', lineWidth: 2 }
    });
    robotRef.current = robot;

    const ground = Matter.Bodies.rectangle(actualWidth / 2, actualHeight + 25, actualWidth, 50, { 
      isStatic: true, 
      render: { fillStyle: '#0f172a' } 
    });
    Matter.Composite.add(engine.world, [robot, ground]);

    const runner = Matter.Runner.create();
    Matter.Runner.run(runner, engine);
    Matter.Render.run(render);

    const controlLoop = setInterval(() => {
      const b = robotRef.current;
      if (!b) return;
      
      Matter.Body.applyForce(b, b.position, { 
        x: controlAction[0] * 0.001, 
        y: (controlAction[1] + physicsPriors.gravity * 10) * 0.001 
      });

      onStateUpdate({
        current: [b.position.x, b.position.y, b.velocity.x, b.velocity.y, b.angle, b.angularVelocity],
        target: target,
        estimatedParams: physicsPriors,
        predictionError: 0, 
        controlEffort: Math.abs(controlAction[0]) + Math.abs(controlAction[1]),
        stability: 100 - Math.min(100, (Math.abs(b.angularVelocity) * 100)),
        time: Date.now(),
        controlAction: controlAction,
        uncertainty: 0,
        isBenchmarking: false
      });
    }, 32);

    const drawOverlay = () => {
      const ctx = canvasRef.current?.getContext('2d');
      if (!ctx || !robotRef.current) return;

      let tempX: StateVector = [
        robotRef.current.position.x,
        robotRef.current.position.y,
        robotRef.current.velocity.x,
        robotRef.current.velocity.y,
        robotRef.current.angle,
        robotRef.current.angularVelocity
      ];
      
      ctx.beginPath();
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.4)';
      ctx.moveTo(tempX[0], tempX[1]);
      
      for (let i = 0; i < 15; i++) {
        tempX = stepDynamicsRK4(tempX, controlAction, physicsPriors);
        ctx.lineTo(tempX[0], tempX[1]);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.beginPath();
      ctx.arc(target[0], target[1], 8, 0, Math.PI * 2);
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 2;
      ctx.stroke();

      requestAnimationFrame(drawOverlay);
    };
    const overlayReq = requestAnimationFrame(drawOverlay);

    return () => {
      clearInterval(controlLoop);
      cancelAnimationFrame(overlayReq);
      Matter.Engine.clear(engine);
      Matter.Render.stop(render);
    };
  }, [mode]);

  return (
    <div ref={containerRef} className="w-full h-full relative overflow-hidden bg-slate-950/50 border border-slate-900 rounded-sm">
      <canvas ref={canvasRef} className="block w-full h-full" />
    </div>
  );
};

export default SimulationCanvas;
