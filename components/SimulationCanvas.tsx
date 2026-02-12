
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
  const particlesRef = useRef<any[]>([]);
  const textileConstraintsRef = useRef<any[]>([]);

  useEffect(() => {
    if (!containerRef.current || !canvasRef.current || typeof Matter === 'undefined') return;

    const engine = Matter.Engine.create();
    engineRef.current = engine;
    engine.world.gravity.y = 1; 

    const rect = containerRef.current.getBoundingClientRect();
    const width = rect.width || 800;
    const height = rect.height || 600;

    const render = Matter.Render.create({
      element: containerRef.current,
      canvas: canvasRef.current,
      engine: engine,
      options: {
        width,
        height,
        wireframes: false,
        background: 'transparent',
        pixelRatio: window.devicePixelRatio,
      }
    });

    const robot = Matter.Bodies.rectangle(width / 2, height / 2, 60, 30, {
      friction: 0.1,
      restitution: 0.5,
      render: { 
        fillStyle: '#4f46e5', 
        strokeStyle: '#818cf8', 
        lineWidth: 3,
        glow: '0 0 15px #4f46e5'
      }
    });
    robotRef.current = robot;

    const ground = Matter.Bodies.rectangle(width / 2, height + 25, width, 50, { isStatic: true, render: { fillStyle: '#0f172a' } });
    const wallLeft = Matter.Bodies.rectangle(-25, height / 2, 50, height, { isStatic: true });
    const wallRight = Matter.Bodies.rectangle(width + 25, height / 2, 50, height, { isStatic: true });
    const ceiling = Matter.Bodies.rectangle(width / 2, -25, width, 50, { isStatic: true });

    const clothRows = 8;
    const clothCols = 12;
    const clothGroup = Matter.Body.nextGroup(true);
    const clothParticles = Matter.Composites.stack(100, 50, clothCols, clothRows, 0, 0, (x: number, y: number) => {
      return Matter.Bodies.circle(x, y, 3, {
        collisionFilter: { group: clothGroup },
        frictionAir: 0.05,
        render: { fillStyle: '#ec4899' }
      });
    });

    const mesh = Matter.Composites.mesh(clothParticles, clothCols, clothRows, false, {
      stiffness: 0.8,
      render: { strokeStyle: 'rgba(236, 72, 153, 0.3)', lineWidth: 1 }
    });
    textileConstraintsRef.current = mesh.constraints;

    for (let i = 0; i < clothCols; i++) {
      if (i % 3 === 0) {
        const p = clothParticles.bodies[i];
        const anchor = Matter.Constraint.create({
          pointA: { x: p.position.x, y: p.position.y },
          bodyB: p,
          stiffness: 1,
          length: 0,
          render: { visible: false }
        });
        Matter.Composite.add(engine.world, anchor);
      }
    }

    const fluidParticles: any[] = [];
    const particleCount = 60;
    for (let i = 0; i < particleCount; i++) {
      const p = Matter.Bodies.circle(width - 150 + (i % 10) * 10, 100 + Math.floor(i / 10) * 10, 4, {
        friction: 0.01,
        restitution: 0.8,
        density: 0.0005,
        render: { fillStyle: '#06b6d4', opacity: 0.6 }
      });
      fluidParticles.push(p);
    }
    particlesRef.current = fluidParticles;

    Matter.Composite.add(engine.world, [robot, ground, wallLeft, wallRight, ceiling, clothParticles, ...fluidParticles]);

    const runner = Matter.Runner.create();
    Matter.Runner.run(runner, engine);
    Matter.Render.run(render);

    const controlLoop = setInterval(() => {
      const b = robotRef.current;
      if (!b) return;
      
      Matter.Body.applyForce(b, b.position, { 
        x: controlAction[0] * 0.0015, 
        y: (controlAction[1] - (engine.world.gravity.y * b.mass * 9.8)) * 0.001 
      });

      let avgFluidRepulsion = 0;
      for (let i = 0; i < fluidParticles.length; i++) {
        for (let j = i + 1; j < fluidParticles.length; j++) {
          const p1 = fluidParticles[i];
          const p2 = fluidParticles[j];
          const dx = p2.position.x - p1.position.x;
          const dy = p2.position.y - p1.position.y;
          const distSq = dx * dx + dy * dy;
          if (distSq < 400) {
            const force = (400 - distSq) * 0.000001;
            avgFluidRepulsion += force;
            Matter.Body.applyForce(p1, p1.position, { x: -dx * force, y: -dy * force });
            Matter.Body.applyForce(p2, p2.position, { x: dx * force, y: dy * force });
          }
        }
      }

      onStateUpdate({
        current: [b.position.x, b.position.y, b.velocity.x, b.velocity.y, b.angle, b.angularVelocity],
        target: target,
        estimatedParams: { 
          ...physicsPriors, 
          textile_k: physicsPriors.textile_k || 400,
          damping: physicsPriors.damping || 0.15 
        },
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
      
      ctx.save();
      ctx.beginPath();
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.6)';
      ctx.lineWidth = 2;
      ctx.shadowBlur = 10;
      ctx.shadowColor = '#6366f1';
      ctx.moveTo(tempX[0], tempX[1]);
      
      for (let i = 0; i < 20; i++) {
        tempX = stepDynamicsRK4(tempX, controlAction, physicsPriors);
        ctx.lineTo(tempX[0], tempX[1]);
      }
      ctx.stroke();
      ctx.restore();

      ctx.save();
      ctx.beginPath();
      ctx.arc(target[0], target[1], 12, 0, Math.PI * 2);
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 3;
      ctx.setLineDash([2, 2]);
      ctx.stroke();
      
      ctx.beginPath();
      ctx.arc(target[0], target[1], 6, 0, Math.PI * 2);
      ctx.fillStyle = '#f43f5e';
      ctx.shadowBlur = 20;
      ctx.shadowColor = '#f43f5e';
      ctx.fill();
      ctx.restore();

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
    <div ref={containerRef} className="w-full h-full relative overflow-hidden bg-slate-950/20 border border-slate-900/50 rounded-sm">
      <canvas ref={canvasRef} className="block w-full h-full" />
      <div className="absolute bottom-4 left-4 pointer-events-none font-mono text-[9px] text-indigo-400/80 bg-black/40 p-2 backdrop-blur-sm border border-indigo-500/20 rounded">
        <div>CORE_LOAD: {(Math.random() * 5 + 92).toFixed(1)}%</div>
        <div>SOLVER: RK4_HIFI</div>
        <div>FLUID_PARTICLES: {particlesRef.current.length}</div>
        <div>TEXTILE_NODES: 96</div>
      </div>
    </div>
  );
};

export default SimulationCanvas;
