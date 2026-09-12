"use client";

import {
  Activity,
  AlertOctagon,
  AlertTriangle,
  BarChart3,
  BatteryCharging,
  BrainCircuit,
  CheckCircle2,
  ClipboardList,
  Clock,
  Cpu,
  Gauge,
  Plane,
  Radio,
  Rotate3D,
  ShieldCheck,
  Thermometer,
  Volume2,
  VolumeX,
  Wrench,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import styles from "./page.module.css";

type ScenarioKey = "endurance" | "highAltitude" | "hotWeather" | "rapidThrottle" | "custom" | "recorded";

type Telemetry = {
  cycle: number;
  elapsedS: number; // mission clock (sim seconds since mission start)
  altitude: number;
  airspeed: number;
  throttle: number;
  rpm: number;
  cht: number;
  egt: number;
  oilTemp: number;
  oilPressure: number;
  fuelFlow: number;
  vibration: number;
  battery: number;
  alternator: number;
  injectionTiming: number;
  efficiency: number;
  rul: number;
  anomaly: number;
  health: number;
  phase: string;
  fault: string;
  faultComponent: string;
  shap: Array<{ feature: string; impact: number }>;
  powerKw: number;
  torqueNm: number;
  densityRatio: number;
  source: "physics-simulation";
  core: { thermal: number; electrical: number; combustion: number; lubrication: number; bsfc: number; status: string };
  ml: { available: boolean; anomaly: boolean; diagnosis: string; confidence: number; rul: number | null; modelType: string; detail: string | null };
};

type TwinPayload = {
  sequence: number;
  source: "physics-simulation";
  mission: { scenario: string; label: string; elapsed_s: number };
  telemetry: {
    altitude_m: number; airspeed_kt: number; throttle_pct: number; rpm: number; power_kw: number; torque_nm: number;
    fuel_flow_lph: number; egt_c: number; cht_c: number; oil_temp_c: number; oil_pressure_psi: number;
    vibration_g: number; battery_soc_pct: number; alternator_v: number; injection_timing_deg: number; density_ratio: number;
  };
  health: { overall_pct: number; anomaly_score: number; fault_mode: string; fault_component: string; rul_cycles: number; thermal_pct: number; electrical_pct: number; combustion_pct: number; lubrication_pct: number };
  xai: Array<{ feature: string; impact: number }>;
  engine_core: { status: string; warnings: string[]; thermal_load_pct: number; electrical_health_pct: number; combustion_quality_pct: number; lubrication_health_pct: number; bsfc_g_kwh: number };
  ml_assessment: { available: boolean; anomaly: boolean; diagnosis: string; confidence: number; rul_cycles: number | null; model_type: string; detail?: string };
};

const TWIN_API = process.env.NEXT_PUBLIC_TWIN_API_URL ?? "http://127.0.0.1:8000";
const TWIN_STREAM = `${TWIN_API.replace(/^http/, "ws")}/stream`;

const scenarios: Record<
  ScenarioKey,
  {
    label: string;
    environment: string;
    mission: string;
    altitudeBase: number;
    temp: number;
    throttleBase: number;
    fault: string;
    risk: string;
    apiScenario: string;
  }
> = {
  endurance: {
    label: "Endurance ISR",
    environment: "Sustained loiter profile",
    mission: "Long-duration steady ISR with slow degradation tracking",
    altitudeBase: 4200,
    temp: 14,
    throttleBase: 62,
    fault: "none",
    risk: "Low",
    apiScenario: "endurance",
  },
  highAltitude: {
    label: "High-Altitude ISR",
    environment: "Thin-air power margin",
    mission: "Density-ratio power loss, lean combustion and cooling margin",
    altitudeBase: 7200,
    temp: -18,
    throttleBase: 76,
    fault: "combustion_instability",
    risk: "Medium",
    apiScenario: "high_altitude",
  },
  hotWeather: {
    label: "Heat Stress",
    environment: "43 C thermal validation",
    mission: "Thermal load, cooling degradation and oil temperature rise",
    altitudeBase: 2100,
    temp: 43,
    throttleBase: 68,
    fault: "cooling_degradation",
    risk: "High",
    apiScenario: "hot_weather",
  },
  rapidThrottle: {
    label: "Throttle Transient",
    environment: "Injection and vibration response",
    mission: "Transient throttle, injection timing and vibration signatures",
    altitudeBase: 3600,
    temp: 28,
    throttleBase: 58,
    fault: "injector_abnormality",
    risk: "High",
    apiScenario: "rapid_throttle",
  },
  custom: {
    label: "Custom",
    environment: "User-defined envelope",
    mission: "Manual altitude / ambient / throttle — full authority",
    altitudeBase: 5000,
    temp: 15,
    throttleBase: 65,
    fault: "any",
    risk: "Variable",
    apiScenario: "custom",
  },
  recorded: {
    label: "Recorded Fault Flight",
    environment: "Real recorded mission rows",
    mission: "Dataset mission streamed live at 10 Hz — real fault growth, real ML diagnosis",
    altitudeBase: 5000,
    temp: 15,
    throttleBase: 65,
    fault: "recorded",
    risk: "High",
    apiScenario: "__replay__",
  },
};

const requirements = [
  "Digital Twin Core",
  "CAN/FADEC Ingestion",
  "Health Indices",
  "Fault Prediction",
  "RUL Estimation",
  "Scenario Simulation",
  "Mission Replay",
  "Edge AI Split",
  "Secure Telemetry",
];

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

// Mission clock T+MM:SS from sim seconds (real-time simulation readout).
function fmtClock(totalS: number) {
  const s = Math.max(0, Math.floor(totalS));
  const mm = String(Math.floor(s / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `T+${mm}:${ss}`;
}

function statusFor(health: number, anomaly: number, fault = "none") {
  if (fault !== "none") return "CAUTION";
  if (health < 55 || anomaly > 0.78) return "CRITICAL";
  if (health < 72 || anomaly > 0.52) return "CAUTION";
  return "NOMINAL";
}

// Recorded-fault -> subsystem, mirroring the backend fault_component map.
function faultComponentFor(fault: string) {
  if (fault === "cooling_degradation" || fault === "overheating") return "thermal";
  if (fault === "injector_abnormality") return "fuel";
  if (fault === "lubrication_issue") return "lubrication";
  if (fault === "sensor_drift") return "electrical";
  if (fault === "misfire" || fault === "combustion_instability") return "combustion";
  if (fault === "abnormal_vibration") return "propulsor";
  return "none";
}

// Dataset mission row -> live Telemetry shape (dataset throttle is a 0-1
// fraction; the live bus carries percent, hence x100).
function fromReplayRow(row: Record<string, number | string>, remaining: number): Telemetry {
  const num = (k: string) => Number(row[k] ?? 0);
  const fault = String(row["fault_type"] ?? "none");
  const rulRaw = Number(row["RUL"]);
  const health = num("health_index");
  return {
    cycle: num("cycle"),
    elapsedS: num("elapsed_time_sec"),
    altitude: num("altitude_m"),
    airspeed: num("airspeed_kt"),
    throttle: num("throttle_pct") * 100,
    rpm: num("rpm"),
    cht: num("cht"),
    egt: num("egt"),
    oilTemp: num("oil_temp"),
    oilPressure: num("oil_pressure"),
    fuelFlow: num("fuel_flow"),
    vibration: num("vibration"),
    battery: num("battery_soc"),
    alternator: num("alternator_voltage"),
    injectionTiming: num("injection_timing"),
    efficiency: clamp(38 - num("fuel_flow") * 0.42, 18, 37),
    rul: Number.isFinite(rulRaw) ? rulRaw : remaining,
    anomaly: num("degradation_severity"),
    health,
    phase: String(row["flight_phase"] ?? "cruise").toUpperCase(),
    fault,
    faultComponent: faultComponentFor(fault),
    shap: [],
    powerKw: num("power_kw"),
    torqueNm: num("torque_nm"),
    densityRatio: num("sigma"),
    source: "physics-simulation",
    core: {
      thermal: health,
      electrical: health,
      combustion: health,
      lubrication: health,
      bsfc: num("bsfc"),
      status: fault !== "none" ? (num("degradation_severity") > 0.6 ? "CRITICAL" : "CAUTION") : "NORMAL",
    },
    ml: { available: false, anomaly: false, diagnosis: "replay", confidence: 0, rul: null, modelType: "replay", detail: null },
  };
}

function fromTwinPayload(payload: TwinPayload): Telemetry {
  const data = payload.telemetry;
  const health = payload.health;
  return {
    cycle: payload.sequence,
    elapsedS: payload.mission.elapsed_s,
    altitude: data.altitude_m,
    airspeed: data.airspeed_kt,
    throttle: data.throttle_pct,
    rpm: data.rpm,
    cht: data.cht_c,
    egt: data.egt_c,
    oilTemp: data.oil_temp_c,
    oilPressure: data.oil_pressure_psi,
    fuelFlow: data.fuel_flow_lph,
    vibration: data.vibration_g,
    battery: data.battery_soc_pct,
    alternator: data.alternator_v,
    injectionTiming: data.injection_timing_deg,
    efficiency: clamp(38 - data.fuel_flow_lph * 0.42, 18, 37),
    rul: health.rul_cycles,
    anomaly: health.anomaly_score,
    health: health.overall_pct,
    phase: payload.mission.label.toUpperCase(),
    fault: health.fault_mode,
    faultComponent: health.fault_component,
    shap: [...payload.xai].sort((a, b) => b.impact - a.impact),
    powerKw: data.power_kw,
    torqueNm: data.torque_nm,
    densityRatio: data.density_ratio,
    source: payload.source,
    core: {
      thermal: payload.engine_core.thermal_load_pct,
      electrical: payload.engine_core.electrical_health_pct,
      combustion: payload.engine_core.combustion_quality_pct,
      lubrication: payload.engine_core.lubrication_health_pct,
      bsfc: payload.engine_core.bsfc_g_kwh,
      status: payload.engine_core.status,
    },
    ml: {
      available: payload.ml_assessment.available,
      anomaly: payload.ml_assessment.anomaly,
      diagnosis: payload.ml_assessment.diagnosis,
      confidence: payload.ml_assessment.confidence,
      rul: payload.ml_assessment.rul_cycles,
      modelType: payload.ml_assessment.model_type,
      detail: payload.ml_assessment.detail ?? null,
    },
  };
}

// Real recorded sounds (gcs-app/public/audio, via Wikimedia Commons):
//   engine-run.ogg    -- Ford TriMotor radial piston aero-engines running (CC BY-SA 3.0, EAA)
//   alert-caution.ogg -- diving alarm (public domain)
//   alert-critical.ogg-- WWII submarine dive klaxon (public domain)
// If a sample cannot be fetched/decoded (offline, no Ogg support), the
// manager falls back to the synthesized engine hum / beeps automatically.
const AUDIO_FILES = {
  engine: "/audio/engine-run.ogg",
  caution: "/audio/alert-caution.ogg",
  critical: "/audio/alert-critical.ogg",
};

class EngineAudioManager {
  ctx: AudioContext | null = null;
  masterGain: GainNode | null = null;
  faultGain: GainNode | null = null;
  // --- real-sample engine chain ---
  engineSrc: AudioBufferSourceNode | null = null;
  engineFilter: BiquadFilterNode | null = null;
  engineGain: GainNode | null = null;
  wobbleOsc: OscillatorNode | null = null;
  wobbleGain: GainNode | null = null;
  useSample = false;
  sampleCache: Record<string, AudioBuffer> = {};
  // --- synthesized fallback chain ---
  baseOsc: OscillatorNode | null = null;
  harmOsc: OscillatorNode | null = null;
  noiseOsc: OscillatorNode | null = null;
  playing = false;
  faultInterval: number | null = null;

  ensureCtx() {
    const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
    if (!this.ctx) this.ctx = new AudioContext();
    if (this.ctx.state === "suspended") void this.ctx.resume();
  }

  async loadSample(url: string): Promise<AudioBuffer> {
    if (this.sampleCache[url]) return this.sampleCache[url];
    if (!this.ctx) this.ensureCtx();
    const res = await fetch(url, { cache: "force-cache" });
    if (!res.ok) throw new Error(`audio HTTP ${res.status}`);
    const raw = await res.arrayBuffer();
    const buf = await this.ctx!.decodeAudioData(raw);
    this.sampleCache[url] = buf;
    return buf;
  }

  async start() {
    if (this.playing) return;
    this.ensureCtx();
    const ctx = this.ctx!;

    this.masterGain = ctx.createGain();
    this.masterGain.gain.value = 0;
    this.faultGain = ctx.createGain();
    this.faultGain.gain.value = 1;
    this.faultGain.connect(this.masterGain);
    this.masterGain.connect(ctx.destination);

    // Prefer the REAL recorded piston-engine loop, RPM-tracked via playbackRate.
    try {
      const buf = await this.loadSample(AUDIO_FILES.engine);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.loop = true;
      // Skip handling transients at the recording edges for a seamless loop.
      src.loopStart = buf.duration * 0.12;
      src.loopEnd = buf.duration * 0.92;
      this.engineFilter = ctx.createBiquadFilter();
      this.engineFilter.type = "lowpass";
      this.engineFilter.frequency.value = 900;
      this.engineGain = ctx.createGain();
      this.engineGain.gain.value = 0.85;
      // Fault roughness: LFO wobble on engine loudness (misfire shake you can hear).
      this.wobbleOsc = ctx.createOscillator();
      this.wobbleOsc.type = "sine";
      this.wobbleOsc.frequency.value = 9;
      this.wobbleGain = ctx.createGain();
      this.wobbleGain.gain.value = 0.02;
      this.wobbleOsc.connect(this.wobbleGain);
      this.wobbleGain.connect(this.engineGain.gain);
      src.connect(this.engineFilter);
      this.engineFilter.connect(this.engineGain);
      this.engineGain.connect(this.faultGain);
      src.start();
      this.wobbleOsc.start();
      this.engineSrc = src;
      this.useSample = true;
    } catch {
      // Offline / no Ogg decode: synthesized fallback (previous behaviour).
      this.useSample = false;
      this.baseOsc = ctx.createOscillator();
      this.baseOsc.type = "sawtooth";
      this.baseOsc.connect(this.faultGain);
      this.harmOsc = ctx.createOscillator();
      this.harmOsc.type = "square";
      this.harmOsc.connect(this.faultGain);
      this.noiseOsc = ctx.createOscillator();
      this.noiseOsc.type = "triangle";
      this.noiseOsc.connect(this.faultGain);
      this.baseOsc.start();
      this.harmOsc.start();
      this.noiseOsc.start();
    }

    // Warm the alert-sample cache in the background (real klaxon / alarm).
    void this.loadSample(AUDIO_FILES.caution).catch(() => undefined);
    void this.loadSample(AUDIO_FILES.critical).catch(() => undefined);

    this.masterGain.gain.setTargetAtTime(0.5, ctx.currentTime, 0.1);
    this.playing = true;
  }

  stop() {
    if (!this.playing || !this.ctx) return;
    if (this.masterGain) this.masterGain.gain.setTargetAtTime(0, this.ctx.currentTime, 0.1);
    const ctx = this.ctx;
    setTimeout(() => {
      try { this.engineSrc?.stop(); } catch { /* already stopped */ }
      try { this.wobbleOsc?.stop(); } catch { /* already stopped */ }
      this.baseOsc?.stop();
      this.harmOsc?.stop();
      this.noiseOsc?.stop();
      this.engineSrc = null;
      this.wobbleOsc = null;
      this.baseOsc = this.harmOsc = this.noiseOsc = null;
      this.playing = false;
      void ctx;
    }, 200);
    if (this.faultInterval) {
      clearInterval(this.faultInterval);
      this.faultInterval = null;
    }
  }

  updateParams(rpm: number, throttle: number, vibration: number, fault: string) {
    if (!this.playing || !this.ctx || !this.masterGain || !this.faultGain) return;

    if (this.useSample && this.engineSrc && this.engineFilter && this.wobbleOsc && this.wobbleGain) {
      // Real recording, RPM-tracked: the loop pitch follows the crankshaft.
      this.engineSrc.playbackRate.setTargetAtTime(0.5 + (rpm / 5500) * 1.0, this.ctx.currentTime, 0.15);
      const load = Math.min(1, Math.max(0, throttle / 100));
      this.engineFilter.frequency.setTargetAtTime(380 + load * 1700, this.ctx.currentTime, 0.15);
      this.masterGain.gain.setTargetAtTime(Math.max(0.08, load), this.ctx.currentTime, 0.15);
      const rough = fault !== "none";
      this.wobbleGain.gain.setTargetAtTime(rough ? 0.22 + vibration * 0.9 : 0.02, this.ctx.currentTime, 0.2);
      this.wobbleOsc.frequency.setTargetAtTime(rough ? 6 + vibration * 30 : 9, this.ctx.currentTime, 0.2);
      return;
    }

    if (!this.baseOsc || !this.harmOsc || !this.noiseOsc) return;
    const baseFreq = 40 + (rpm / 5500) * 120;
    this.baseOsc.frequency.setTargetAtTime(baseFreq, this.ctx.currentTime, 0.1);
    this.harmOsc.frequency.setTargetAtTime(baseFreq * 2, this.ctx.currentTime, 0.1);

    this.noiseOsc.frequency.setTargetAtTime(baseFreq * 0.5, this.ctx.currentTime, 0.1);
    this.noiseOsc.detune.setTargetAtTime(vibration * 1000, this.ctx.currentTime, 0.1);

    const targetVol = Math.max(0.1, throttle / 100);
    this.masterGain.gain.setTargetAtTime(targetVol, this.ctx.currentTime, 0.1);

    if (fault !== "none" && !this.faultInterval) {
      this.faultInterval = window.setInterval(() => {
        if (!this.ctx || !this.faultGain) return;
        this.faultGain.gain.setValueAtTime(0.2, this.ctx.currentTime);
        this.faultGain.gain.setTargetAtTime(1, this.ctx.currentTime + 0.05, 0.02);
      }, 150);
    } else if (fault === "none" && this.faultInterval) {
      clearInterval(this.faultInterval);
      this.faultInterval = null;
      this.faultGain.gain.setTargetAtTime(1, this.ctx.currentTime, 0.1);
    }
  }

  async playAlert(type: "caution" | "critical") {
    if (!this.ctx || !this.masterGain) return;
    const ctx = this.ctx;
    // Prefer the REAL recorded alarm; fall back to synth beeps offline.
    try {
      const buf = await this.loadSample(type === "caution" ? AUDIO_FILES.caution : AUDIO_FILES.critical);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      const g = ctx.createGain();
      g.gain.setValueAtTime(0.9, ctx.currentTime);
      g.gain.setTargetAtTime(0, ctx.currentTime + Math.min(1.6, buf.duration * 0.9), 0.1);
      src.connect(g);
      g.connect(this.masterGain);
      src.start();
      return;
    } catch {
      /* fall through to synth */
    }
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(this.masterGain);

    if (type === "caution") {
      osc.type = "sine";
      osc.frequency.value = 800;
      gain.gain.setValueAtTime(0, ctx.currentTime);
      gain.gain.setValueAtTime(0.5, ctx.currentTime + 0.01);
      gain.gain.setValueAtTime(0, ctx.currentTime + 0.1);
      gain.gain.setValueAtTime(0.5, ctx.currentTime + 0.2);
      gain.gain.setValueAtTime(0, ctx.currentTime + 0.3);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.35);
    } else {
      osc.type = "square";
      let t = ctx.currentTime;
      gain.gain.setValueAtTime(0.4, t);
      for (let i = 0; i < 5; i++) {
        osc.frequency.setValueAtTime(900, t);
        osc.frequency.setValueAtTime(700, t + 0.2);
        t += 0.4;
      }
      gain.gain.setTargetAtTime(0, t, 0.05);
      osc.start(ctx.currentTime);
      osc.stop(t + 0.1);
    }
  }
}

// Static placeholder for 3D when offline – not used for metrics, only to keep scene alive
const OFFLINE_PLACEHOLDER: Telemetry = {
  cycle: 0,
  elapsedS: 0,
  altitude: 0,
  airspeed: 0,
  throttle: 0,
  rpm: 0,
  cht: 0,
  egt: 0,
  oilTemp: 0,
  oilPressure: 0,
  fuelFlow: 0,
  vibration: 0,
  battery: 0,
  alternator: 0,
  injectionTiming: 0,
  efficiency: 0,
  rul: 0,
  anomaly: 0,
  health: 0,
  phase: "OFFLINE",
  fault: "none",
  faultComponent: "none",
  shap: [],
  powerKw: 0,
  torqueNm: 0,
  densityRatio: 0,
  source: "physics-simulation",
  core: { thermal: 0, electrical: 0, combustion: 0, lubrication: 0, bsfc: 0, status: "OFFLINE" },
  ml: { available: false, anomaly: false, diagnosis: "offline", confidence: 0, rul: null, modelType: "offline", detail: null },
};

function DroneScene({ telemetry, isLive, viewMode }: { telemetry: Telemetry; isLive: boolean; viewMode: "drone" | "engine" }) {
  const mountRef = useRef<HTMLDivElement>(null);
  const telemetryRef = useRef(telemetry);
  const liveRef = useRef(isLive);
  const viewRef = useRef(viewMode);

  useEffect(() => {
    telemetryRef.current = telemetry;
  }, [telemetry]);
  useEffect(() => {
    liveRef.current = isLive;
  }, [isLive]);
  useEffect(() => {
    viewRef.current = viewMode;
  }, [viewMode]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0xbcd7e8, 34, 200);

    const camera = new THREE.PerspectiveCamera(42, mount.clientWidth / mount.clientHeight, 0.1, 600);
    camera.position.set(4.8, 3.2, 7.2);
    camera.lookAt(0, 0.15, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.localClippingEnabled = true;
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(-0.35, 0.12, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    // Free 360° inspection: full vertical orbit, pan and zoom all enabled.
    controls.enablePan = true;
    controls.screenSpacePanning = true;
    controls.minDistance = 0.8;
    controls.maxDistance = 60;
    controls.minPolarAngle = 0;
    controls.maxPolarAngle = Math.PI;
    controls.update();
    // Camera auto-framing runs only briefly after a view switch and yields
    // instantly to user input, so it never fights manual orbiting.
    let camTransition = 0;
    let prevVm: "drone" | "engine" = viewRef.current;
    controls.addEventListener("start", () => { camTransition = 0; });

    const keyLight = new THREE.DirectionalLight(0xfff2df, 2.3);
    keyLight.position.set(10, 14, 6);
    keyLight.castShadow = true;
    scene.add(keyLight);
    scene.add(new THREE.HemisphereLight(0xbfe3f5, 0x3d5a3a, 0.85));
    scene.add(new THREE.AmbientLight(0x6f8ea8, 0.5));

    // (Hologram grid removed per design — open sky + farmland below.)

    // --- Sky dome (gradient shader, always behind everything) ---
    const skyMat = new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      fog: false,
      uniforms: {
        top: { value: new THREE.Color(0x2f7fc4) },
        mid: { value: new THREE.Color(0x9fd3ea) },
        bot: { value: new THREE.Color(0xe6e0cf) },
      },
      vertexShader: `varying vec3 vP; void main() { vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `varying vec3 vP; uniform vec3 top; uniform vec3 mid; uniform vec3 bot;
        void main() {
          float h = normalize(vP).y;
          vec3 c = h > 0.0 ? mix(mid, top, pow(h, 0.6)) : mix(mid, bot, pow(-h, 0.6));
          gl_FragColor = vec4(c, 1.0);
        }`,
    });
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(240, 24, 16), skyMat));

    // --- Ground far below (patchwork fields, scrolls past in flight) ---
    const fieldCanvas = document.createElement("canvas");
    fieldCanvas.width = 256;
    fieldCanvas.height = 256;
    const fctx = fieldCanvas.getContext("2d")!;
    fctx.fillStyle = "#4a6b41";
    fctx.fillRect(0, 0, 256, 256);
    const fieldCols = ["#425f3a", "#55794a", "#6b7a44", "#3d5a44", "#7a6f4a", "#4f7038"];
    for (let i = 0; i < 46; i++) {
      fctx.fillStyle = fieldCols[i % fieldCols.length];
      fctx.globalAlpha = 0.55 + Math.random() * 0.45;
      fctx.fillRect(Math.random() * 256, Math.random() * 256, 20 + Math.random() * 70, 14 + Math.random() * 50);
    }
    fctx.globalAlpha = 1;
    fctx.strokeStyle = "#8a8f96";
    fctx.lineWidth = 3;
    fctx.beginPath();
    fctx.moveTo(0, 128);
    fctx.lineTo(256, 128);
    fctx.moveTo(128, 0);
    fctx.lineTo(128, 256);
    fctx.stroke();
    const groundTex = new THREE.CanvasTexture(fieldCanvas);
    groundTex.wrapS = THREE.RepeatWrapping;
    groundTex.wrapT = THREE.RepeatWrapping;
    groundTex.repeat.set(10, 10);
    const ground = new THREE.Mesh(
      new THREE.CircleGeometry(220, 48),
      new THREE.MeshStandardMaterial({ map: groundTex, roughness: 1, metalness: 0 })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -26;
    scene.add(ground);

    // --- Clouds (drifting puffs the UAV flies past) ---
    const clouds: THREE.Group[] = [];
    const cloudMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, transparent: true, opacity: 0.92 });
    for (let c = 0; c < 14; c++) {
      const puff = new THREE.Group();
      const n = 3 + Math.floor(Math.random() * 3);
      for (let k = 0; k < n; k++) {
        const r = 1.2 + Math.random() * 1.8;
        const m = new THREE.Mesh(new THREE.SphereGeometry(r, 12, 10), cloudMat);
        m.position.set((Math.random() - 0.5) * 5, (Math.random() - 0.5) * 1.2, (Math.random() - 0.5) * 2.5);
        m.scale.y = 0.55;
        puff.add(m);
      }
      puff.position.set((Math.random() - 0.5) * 140, 8 + Math.random() * 15, (Math.random() - 0.5) * 120);
      scene.add(puff);
      clouds.push(puff);
    }

    // --- Wind streaks (airspeed-driven rushing air) ---
    const STREAKS = 130;
    const streakMesh = new THREE.InstancedMesh(
      new THREE.BoxGeometry(0.9, 0.012, 0.012),
      new THREE.MeshBasicMaterial({ color: 0xeaf7ff, transparent: true, opacity: 0.5 }),
      STREAKS
    );
    scene.add(streakMesh);
    const streakData: Array<{ x: number; y: number; z: number; s: number }> = [];
    for (let i = 0; i < STREAKS; i++) {
      streakData.push({ x: (Math.random() - 0.5) * 28, y: -2 + Math.random() * 7, z: (Math.random() - 0.5) * 18, s: 0.7 + Math.random() * 0.6 });
    }
    const streakDummy = new THREE.Object3D();

    // --- MQ-9 REAPER (user-supplied file, loaded whole and as-is) ---
    const drone = new THREE.Group();
    scene.add(drone);
    const mq9 = new THREE.Group();
    mq9.rotation.y = Math.PI; // file nose faces -X; scene forward is +X
    drone.add(mq9);

    // Flat factory-gray finish, no textures.
    const reaperMat = new THREE.MeshStandardMaterial({
      color: 0x8f9599,
      roughness: 0.6,
      metalness: 0.05,
      emissive: new THREE.Color(0x000000),
      emissiveIntensity: 0,
    });

    new OBJLoader().load(
      "/mq9-reaper.obj",
      (obj) => {
        // Auto-fit: wingspan onto ~7.4 scene units, fuselage onto the origin line
        const box = new THREE.Box3().setFromObject(obj);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const s = maxDim > 0 ? 7.4 / maxDim : 0.394;
        obj.scale.setScalar(s);
        obj.position.set(-center.x * s, -center.y * s + 0.1, -center.z * s);
        obj.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const mesh = child as THREE.Mesh;
            mesh.material = reaperMat;
            mesh.castShadow = true;
          }
        });
        mq9.add(obj);
      },
      undefined,
      (err) => console.warn("Reaper model failed to load:", err)
    );

    // --- ENGINE GROUP — real Rotax 912 shell only (public/engine.obj) ---
    const engine = new THREE.Group();
    const cadEngine = new THREE.Group();
    engine.add(cadEngine);
    const realEngineMaterials: THREE.MeshStandardMaterial[] = [];
    new OBJLoader().load(
      "/engine.obj",
      (loadedObj) => {
        const box = new THREE.Box3().setFromObject(loadedObj);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = maxDim > 0 ? 2.1 / maxDim : 0.005;
        loadedObj.scale.setScalar(scale);
        loadedObj.position.set(-center.x * scale, -center.y * scale + 0.05, -center.z * scale);
        loadedObj.rotation.y = -Math.PI / 2;
        loadedObj.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const mesh = child as THREE.Mesh;
            if (mesh.geometry) mesh.geometry.computeVertexNormals();
            const mat = new THREE.MeshStandardMaterial({
              color: 0x93a4b0,
              metalness: 0.68,
              roughness: 0.32,
              emissive: new THREE.Color(0x000000),
              emissiveIntensity: 0.16,
            });
            mesh.material = mat;
            mesh.castShadow = false; // 1M-face shell stays out of the shadow pass
            mesh.receiveShadow = false;
            realEngineMaterials.push(mat);
          }
        });
        cadEngine.add(loadedObj);
      },
      undefined,
      (err) => {
        console.warn("Real engine shell failed to load:", err);
      }
    );
    // Position engine group — center in scene when active
    engine.position.set(0, 0.15, 0);
    engine.scale.setScalar(1.35);
    engine.visible = false;
    scene.add(engine);

    // Position engine group — center in scene when active
    engine.position.set(0, 0.15, 0);
    engine.scale.setScalar(1.35);
    engine.visible = false;
    scene.add(engine);

    // --- Telemetry ring ---
    const telemetryArc = new THREE.Mesh(
      new THREE.TorusGeometry(2.75, 0.006, 8, 96, Math.PI * 1.55),
      new THREE.MeshBasicMaterial({ color: 0x42f5d7, transparent: true, opacity: 0.38 }),
    );
    telemetryArc.rotation.x = Math.PI / 2;
    telemetryArc.position.y = -0.1;
    scene.add(telemetryArc);

    let frame = 0;
    let raf = 0;
    const animate = () => {
      frame += 0.016;
      const live = telemetryRef.current;
      const isLiveNow = liveRef.current;
      const vm = viewRef.current;
      const load = live.throttle / 100;

      // Toggle visibility
      const showEngine = vm === "engine";
      drone.visible = !showEngine;
      engine.visible = showEngine;
      // Camera auto-frame: brief glide after a view switch, then full manual control
      if (vm !== prevVm) {
        prevVm = vm;
        camTransition = 1.4;
      }
      const targetDrone = new THREE.Vector3(-0.35, 0.12, 0);
      const targetEngine = new THREE.Vector3(0, 0.12, 0);
      const camDrone = new THREE.Vector3(4.8, 3.2, 7.2);
      const camEngine = new THREE.Vector3(1.85, 1.15, 1.85);
      if (camTransition > 0) {
        camTransition -= 0.016;
        controls.target.lerp(showEngine ? targetEngine : targetDrone, 0.08);
        camera.position.lerp(showEngine ? camEngine : camDrone, 0.06);
      }
      camera.lookAt(controls.target);
      // Sky life runs in every view: clouds drift past with airspeed
      const ias = isLiveNow ? Math.max(live.airspeed, 0) : 18;
      clouds.forEach((puff, i) => {
        puff.position.x -= (1.2 + ias * 0.1) * 0.016 * (1 + (i % 3) * 0.3);
        if (puff.position.x < -75) {
          puff.position.x = 75;
          puff.position.z = (Math.random() - 0.5) * 120;
        }
      });
      groundTex.offset.x += 0.002 + ias * 0.00012; // world flows -X past the +X-facing nose

      if (showEngine) {
        // Real Rotax shell: RPM tremor + vibration shake, slow inspection
        // turntable, and fault-driven heat glow across the casing.
        const t = frame * (isLiveNow ? 1 : 0.25);
        const tremor = isLiveNow ? 0.004 + (live.rpm / 5500) * 0.012 : 0.002;
        engine.position.x = Math.sin(t * 18) * live.vibration * 0.12 + Math.sin(t * 61) * tremor;
        engine.position.y = 0.15 + Math.sin(t * 22) * live.vibration * 0.06 + Math.sin(t * 53) * tremor;
        cadEngine.rotation.y = Math.sin(frame * 0.35) * 0.3; // inspection turntable
        // Real shell fault glow
        if (realEngineMaterials.length > 0) {
          const faultActive = live.fault !== "none";
          const pulse = Math.abs(Math.sin(frame * 6));
          let shellHex = 0x000000;
          let shellInt = 0.14;
          if (faultActive) {
            if (live.faultComponent === "thermal" || live.fault === "overheating" || live.fault === "cooling_degradation") {
              shellHex = 0xff3b1f;
              shellInt = 0.45 + pulse * 0.55;
            } else if (live.faultComponent === "lubrication" || live.fault === "lubrication_issue") {
              shellHex = 0xd69e2e;
              shellInt = 0.4 + pulse * 0.5;
            } else if (live.faultComponent === "electrical" || live.fault === "sensor_drift") {
              shellHex = 0x805ad5;
              shellInt = 0.4 + pulse * 0.5;
            } else {
              shellHex = 0xe53e3e;
              shellInt = 0.35 + pulse * 0.5;
            }
          }
          realEngineMaterials.forEach((mat) => {
            mat.emissive.setHex(shellHex);
            mat.emissiveIntensity = shellInt;
          });
        }
        streakMesh.visible = false; // test-stand view: no rushing air
      } else {
        // Drone animation
        if (isLiveNow) {
          // Parallax motion
          let phasePitch = 0;
          if (live.phase === "CLIMB") phasePitch = 0.1;
          else if (live.phase === "DESCENT") phasePitch = -0.1;

          drone.rotation.z = Math.sin(frame * 1.4) * 0.08 + (load - 0.62) * 0.22 + phasePitch;
          drone.rotation.x = Math.sin(frame * 0.9) * 0.035;
          drone.rotation.y = Math.sin(frame * 0.55) * 0.06;
          drone.position.y = Math.sin(frame * 1.1) * 0.08;
          // Whole-model fault tint on the Reaper shell
          if (live.fault !== "none") {
            reaperMat.emissive.setHex(0xff1208);
            reaperMat.emissiveIntensity = 0.45 + Math.abs(Math.sin(frame * 7.2)) * 1.0;
          } else {
            reaperMat.emissive.setHex(0x000000);
            reaperMat.emissiveIntensity = 0;
          }
          streakMesh.visible = true;
          for (let i = 0; i < STREAKS; i++) {
            const s = streakData[i];
            s.x -= (6 + ias * 0.55) * s.s * 0.016;
            if (s.x < -15) {
              s.x = 15;
              s.y = -2 + Math.random() * 7;
              s.z = (Math.random() - 0.5) * 18;
            }
            streakDummy.position.set(s.x, s.y, s.z);
            streakDummy.updateMatrix();
            streakMesh.setMatrixAt(i, streakDummy.matrix);
          }
          streakMesh.instanceMatrix.needsUpdate = true;
          telemetryArc.rotation.z += 0.004 + live.anomaly * 0.012;
        } else {
          drone.rotation.y = Math.sin(frame * 0.2) * 0.08;
          drone.position.y = Math.sin(frame * 0.6) * 0.04;
          reaperMat.emissive.setHex(0x000000);
          reaperMat.emissiveIntensity = 0;
          streakMesh.visible = false;
          telemetryArc.rotation.z += 0.001;
        }
      }
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };
    animate();

    const resize = () => {
      if (!mount) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener("resize", resize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      controls.dispose();
      mount.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  return <div ref={mountRef} className={styles.sceneCanvas} aria-label="3D MALE UAV digital twin simulation" />;
}

function MetricBar({ label, value, unit, max, tone = "normal" }: { label: string; value: number; unit: string; max: number; tone?: "normal" | "warn" | "danger" }) {
  const pct = clamp((value / max) * 100, 0, 100);
  return (
    <div className={styles.metricBar}>
      <div className={styles.metricHead}>
        <span>{label}</span>
        <strong>
          {value.toFixed(value < 10 ? 1 : 0)} {unit}
        </strong>
      </div>
      <div className={styles.track}>
        <span className={`${styles.fill} ${styles[tone]}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function SignalTrace({ history }: { history: Telemetry[] }) {
  const values = history.map((sample) => sample.egt);
  const points = values.length > 1
    ? values.map((value, index) => `${(index / (values.length - 1)) * 100},${100 - clamp((value - 620) / 240 * 100, 0, 100)}`).join(" ")
    : "0,70 100,70";
  const latest = values.at(-1) ?? 0;
  const hasData = history.length > 0;
  return (
    <div className={styles.signalTrace} aria-label="Live exhaust-gas-temperature trace">
      <div><span>EGT trend</span><strong>{hasData ? `${latest.toFixed(0)} C` : "--"}</strong></div>
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Rolling live EGT trace">
        <path d="M0 25H100M0 50H100M0 75H100" />
        {hasData ? <polyline points={points} /> : null}
      </svg>
      {!hasData ? <small className={styles.trendNote}>Awaiting backend stream…</small> : null}
    </div>
  );
}

export default function Home() {
  const [scenario, setScenario] = useState<ScenarioKey>("hotWeather");
  const [liveTelemetry, setLiveTelemetry] = useState<Telemetry | null>(null);
  const [history, setHistory] = useState<Telemetry[]>([]);
  const [linkState, setLinkState] = useState<"connecting" | "live" | "offline">("connecting");
  const [retryKey, setRetryKey] = useState(0);
  const [healthInfo, setHealthInfo] = useState<Record<string, unknown> | null>(null);
  const [missions, setMissions] = useState<Array<{ engine_id: string; mission_id: number; scenario: string; fault_type: string; cycles: number }>>([]);
  const [showCustom, setShowCustom] = useState(false);
  const [customAlt, setCustomAlt] = useState(5000);
  const [customAmb, setCustomAmb] = useState(15);
  const [customThr, setCustomThr] = useState(65);
  const [customLabel, setCustomLabel] = useState("Custom Mission");
  const [viewMode, setViewMode] = useState<"drone" | "engine">("drone");
  const [currentTime, setCurrentTime] = useState<Date | null>(null);
  const [soundEnabled, setSoundEnabled] = useState(false);
  const audioRef = useRef<EngineAudioManager | null>(null);
  const prevStatusRef = useRef("NOMINAL");

  useEffect(() => {
    setCurrentTime(new Date());
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!audioRef.current) audioRef.current = new EngineAudioManager();
    if (soundEnabled) {
      audioRef.current.start();
    } else {
      audioRef.current.stop();
    }
  }, [soundEnabled]);

  useEffect(() => {
    if (soundEnabled && audioRef.current && liveTelemetry) {
      audioRef.current.updateParams(liveTelemetry.rpm, liveTelemetry.throttle, liveTelemetry.vibration, liveTelemetry.fault);
    }
  }, [liveTelemetry, soundEnabled]);

  useEffect(() => {
    let active = true;
    let socket: WebSocket | undefined;
    let reconnectTimer: number | undefined;
    let attempt = 0;

    const connect = async (fresh: boolean) => {
      if (!active) return;
      setLinkState("connecting");
      try {
        const healthRes = await fetch(`${TWIN_API}/health`, { cache: "no-store" });
        if (!healthRes.ok) throw new Error(`health ${healthRes.status}`);
        const h = await healthRes.json();
        if (active) setHealthInfo(h);

        // A (re)start is only issued for explicit user action (scenario change
        // or manual retry). Silent auto-reconnects reattach to the ONGOING
        // mission so the real-time clock is never reset underneath the user.
        if (fresh) {
          if (scenario === "recorded") {
            const start = await fetch(`${TWIN_API}/mission/replay`, { method: "POST" });
            if (!start.ok) throw new Error(`mission/replay ${start.status}: ${await start.text()}`);
          } else if (scenario === "custom") {
            const qs = new URLSearchParams({ altitude: String(customAlt), ambient: String(customAmb), throttle: String(customThr), label: customLabel });
            const start = await fetch(`${TWIN_API}/mission/custom?${qs.toString()}`, { method: "POST" });
            if (!start.ok) throw new Error(`mission/custom ${start.status}: ${await start.text()}`);
          } else {
            const start = await fetch(`${TWIN_API}/mission/start?scenario=${scenarios[scenario].apiScenario}`, { method: "POST" });
            if (!start.ok) throw new Error(`mission/start ${start.status}: ${await start.text()}`);
          }
        }

        socket = new WebSocket(TWIN_STREAM);

        socket.onopen = () => {
          attempt = 0;
          if (active) setLinkState("live");
        };

        socket.onmessage = (event) => {
          if (!active) return;
          if (replayRef.current) return; // replay owns the telemetry state while active
          try {
            const next = fromTwinPayload(JSON.parse(event.data) as TwinPayload);
            setLiveTelemetry(next);
            setHistory((current) => [...current.slice(-59), next]);
            if (linkState !== "live") setLinkState("live");
          } catch {
            // ignore malformed frame
          }
        };

        socket.onerror = () => {
          if (active) setLinkState("offline");
        };

        socket.onclose = () => {
          if (!active) return;
          setLinkState("offline");
          attempt += 1;
          const delay = Math.min(800 * Math.pow(1.7, attempt), 7000);
          reconnectTimer = window.setTimeout(() => {
            if (active) void connect(false); // reattach only — never restart the mission
          }, delay) as unknown as number;
        };
      } catch {
        if (!active) return;
        setLinkState("offline");
        attempt += 1;
        const delay = Math.min(800 * Math.pow(1.7, attempt), 7000);
        reconnectTimer = window.setTimeout(() => {
          if (active) void connect(false);
        }, delay) as unknown as number;
      }
    };

    void connect(true); // explicit mount/scenario change: (re)start the mission

    return () => {
      active = false;
      try {
        socket?.close();
      } catch {}
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
    };
  }, [scenario, retryKey]);

  // Mission-wise health reports — real backend history, never mock
  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const res = await fetch(`${TWIN_API}/missions?limit=8`, { cache: "no-store" });
        if (!res.ok) return;
        const j = (await res.json()) as { missions: Array<{ engine_id: string; mission_id: number; scenario: string; fault_type: string; cycles: number }> };
        if (alive) setMissions(j.missions ?? []);
      } catch {}
    };
    void load();
    const id = window.setInterval(() => void load(), 12000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  // Predictive maintenance advisories — real backend /advisory, polled live
  const [advisories, setAdvisories] = useState<Array<{ level: string; action: string; due_cycles: string }>>([]);
  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        if (replayRef.current) return; // replay shows recorded faults; live advisory stays paused
        const res = await fetch(`${TWIN_API}/advisory`, { cache: "no-store" });
        if (!res.ok) return;
        const j = (await res.json()) as { advisory: Array<{ level: string; action: string; due_cycles: string }> };
        if (alive) setAdvisories(j.advisory ?? []);
      } catch {}
    };
    void load();
    const id = window.setInterval(() => void load(), 2000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  // Alert toasts — popup on every transition INTO caution/critical (and on
  // recovery), driven by the same live status as the banner. Works muted.
  type Toast = { id: number; level: "CAUTION" | "CRITICAL" | "NOMINAL"; title: string; msg: string };
  const [toasts, setToasts] = useState<Toast[]>([]);
  const toastId = useRef(0);
  const toastPrev = useRef("NOMINAL");
  const dismissToast = (id: number) => {
    setToasts((cur) => cur.filter((t) => t.id !== id));
  };
  const pushToast = (level: Toast["level"], title: string, msg: string, ttlMs: number) => {
    toastId.current += 1;
    const id = toastId.current;
    setToasts((cur) => [...cur.slice(-3), { id, level, title, msg }]);
    window.setTimeout(() => dismissToast(id), ttlMs);
  };

  // Mission replay — real recorded rows from GET /missions/{id}, played at 10 Hz
  // through the SAME liveTelemetry state, so 3D, gauges, trends all follow.
  type ReplayState = { missionId: number; engineId: string; fault: string; rows: Telemetry[]; idx: number; playing: boolean };
  const [replay, setReplay] = useState<ReplayState | null>(null);
  const replayRef = useRef<ReplayState | null>(null);
  useEffect(() => {
    replayRef.current = replay;
  }, [replay]);
  const replaying = replay !== null;

  const startReplay = async (missionId: number) => {
    try {
      const res = await fetch(`${TWIN_API}/missions/${missionId}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`missions/${missionId} ${res.status}`);
      const j = (await res.json()) as { engine_id: string; fault_type: string; data: Array<Record<string, number | string>> };
      const rows = j.data.map((r, i) => fromReplayRow(r, j.data.length - 1 - i));
      if (rows.length === 0) return;
      setAdvisories([]);
      setHistory([]);
      setLiveTelemetry(rows[0]);
      setReplay({ missionId, engineId: j.engine_id, fault: j.fault_type, rows, idx: 0, playing: true });
    } catch {
      // keep live view untouched when replay fetch fails
    }
  };
  const stopReplay = () => {
    setReplay(null);
    setHistory([]);
  };
  useEffect(() => {
    if (!replay || !replay.playing) return;
    // Side effects stay OUT of the state updater (StrictMode double-invokes
    // updaters in dev — index math would still be pure-safe via the ref).
    const id = window.setInterval(() => {
      const cur = replayRef.current;
      if (!cur || !cur.playing) return;
      const next = Math.min(cur.idx + 1, cur.rows.length - 1);
      const frame = cur.rows[next];
      setLiveTelemetry(frame);
      setHistory((h) => [...h.slice(-59), frame]);
      setReplay({ ...cur, idx: next, playing: next < cur.rows.length - 1 });
    }, 100);
    return () => window.clearInterval(id);
  }, [replay?.missionId, replay?.playing]);

  const telemetry = liveTelemetry;
  const isLive = linkState === "live" && telemetry !== null;
  const display: Telemetry | null = telemetry;
  const status = display ? statusFor(display.health, display.anomaly, display.fault) : "OFFLINE";

  useEffect(() => {
    if (soundEnabled && audioRef.current && status !== prevStatusRef.current) {
      if (status === "CAUTION") audioRef.current.playAlert("caution");
      if (status === "CRITICAL") audioRef.current.playAlert("critical");
      prevStatusRef.current = status;
    }
  }, [status, soundEnabled]);

  useEffect(() => {
    if (!display || status === toastPrev.current) return;
    const prev = toastPrev.current;
    toastPrev.current = status;
    if (status === "CAUTION" || status === "CRITICAL") {
      const vitals = `CHT ${display.cht.toFixed(0)}C · EGT ${display.egt.toFixed(0)}C · RUL ${display.rul} cycles`;
      pushToast(
        status,
        `${status} — ${display.fault === "none" ? "abnormal readings" : display.fault.replaceAll("_", " ")}`,
        status === "CRITICAL" ? `${vitals}. Recovery recommended before next sortie.` : vitals,
        status === "CRITICAL" ? 12000 : 8000
      );
    } else if (status === "NOMINAL" && (prev === "CAUTION" || prev === "CRITICAL")) {
      pushToast("NOMINAL", "Recovered to NOMINAL", `Fault cleared at ${fmtClock(display.elapsedS)}.`, 6000);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  const selectedScenario = scenarios[scenario];

  const handleRetry = () => {
    setLiveTelemetry(null);
    setHistory([]);
    setLinkState("connecting");
    setRetryKey((k) => k + 1);
  };

  return (
    <main className={styles.shell}>
      <section className={styles.topbar}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ background: 'var(--cyan)', color: '#000', padding: '4px 8px', borderRadius: '4px', fontWeight: 'bold', fontSize: '12px' }}>
            DRDO
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <h1>MALE UAV Mission Control</h1>
              <span style={{ color: 'var(--cyan)', fontSize: '12px', padding: '2px 8px', border: '1px solid var(--cyan)', borderRadius: '12px' }}>Powered by Digital Twin + AI/ML</span>
            </div>
          </div>
        </div>
        <div className={styles.statusCluster}>
          {currentTime && (
            <span style={{ fontFamily: 'var(--font-geist-mono)', fontSize: '13px', color: 'var(--muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Clock size={14} />
              {currentTime.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
            </span>
          )}
          <span className={`${styles.statusPill} ${styles[status.toLowerCase()]}`}>{status}</span>
          <span className={styles.livePill}>
            <Radio size={15} />
            {linkState === "live" ? "10 Hz stream" : linkState === "connecting" ? "Connecting to twin…" : "Backend offline"}
          </span>
          <button 
            type="button" 
            onClick={() => setSoundEnabled(!soundEnabled)}
            title="Real recorded piston aero-engine + klaxon/alarm audio (Wikimedia Commons: EAA Ford TriMotor CC BY-SA 3.0; alarms public domain)"
            style={{ display: "flex", alignItems: "center", gap: 6, padding: "4px 8px", border: "1px solid var(--line)", borderRadius: 6, background: soundEnabled ? "var(--cyan)" : "rgba(5,16,21,0.78)", color: soundEnabled ? "#071518" : "var(--cyan)", fontFamily: "var(--font-geist-mono)", fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", cursor: "pointer", marginLeft: '8px' }}
          >
            {soundEnabled ? <Volume2 size={14} /> : <VolumeX size={14} />} SOUND
          </button>
        </div>
      </section>

      {!isLive ? (
        <section className={styles.offlineBanner}>
          <span className={styles.offlineBannerTitle}>
            {linkState === "connecting" ? "Connecting to backend…" : "Backend offline — no mock data shown"}
          </span>
          <span className={styles.offlineBannerText}>
            This station only displays the backend physics-simulation stream (a DRDO-permitted simulated dataset,
            tagged <code>source: physics-simulation</code>).{" "}
            {linkState === "offline"
              ? "Start the FastAPI twin at http://127.0.0.1:8000, then retry."
              : "Waiting for 10 Hz telemetry…"}
            {healthInfo ? ` Health: ${String((healthInfo as Record<string, unknown>)["ml_detail"] ?? "")}` : null}
          </span>
          <button className={styles.offlineBannerButton} onClick={handleRetry} type="button">
            Retry connection
          </button>
          <span className={styles.offlineBannerMeta}>
            API {TWIN_API} · WS {TWIN_STREAM}
          </span>
        </section>
      ) : null}

      <section className={styles.controlStrip}>
        {Object.entries(scenarios).map(([key, item]) => {
          const riskColor = item.risk === 'Low' ? 'var(--green)' : item.risk === 'Medium' ? 'var(--amber)' : item.risk === 'High' ? 'var(--red)' : 'var(--cyan)';
          return (
          <button
            className={`${styles.scenarioButton} ${scenario === key ? styles.selected : ""}`}
            key={key}
            onClick={() => {
              if (key === "custom") setShowCustom(true);
              setScenario(key as ScenarioKey);
              setLiveTelemetry(null);
              setHistory([]);
            }}
            type="button"
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Plane size={14} /> {item.label}</span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '10px', color: riskColor }}>
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: riskColor, display: 'inline-block' }}></span>
                {item.risk}
              </span>
            </div>
            <small>{item.environment}</small>
          </button>
        )})}
      </section>

      {showCustom ? (
        <section className={styles.customPanel}>
          <div className={styles.customHeader}>
            <h3><Wrench size={14} /> Custom Mission — Adjust Anything (Live Physics)</h3>
            <button type="button" className={styles.btnGhost} onClick={() => setShowCustom(false)}>Close</button>
          </div>
          <div className={styles.customGrid}>
            <div className={styles.sliderRow}>
              <label><span>Altitude</span><span>{customAlt} m</span></label>
              <input type="range" min={0} max={11000} step={100} value={customAlt} onChange={(e) => setCustomAlt(Number(e.target.value))} />
              <input type="text" value={String(customAlt)} onChange={(e) => setCustomAlt(Number(e.target.value) || 0)} placeholder="0 - 11000" />
            </div>
            <div className={styles.sliderRow}>
              <label><span>Ambient Temp</span><span>{customAmb} °C</span></label>
              <input type="range" min={-50} max={55} step={1} value={customAmb} onChange={(e) => setCustomAmb(Number(e.target.value))} />
              <input type="text" value={String(customAmb)} onChange={(e) => setCustomAmb(Number(e.target.value) || 0)} placeholder="-50 - 55" />
            </div>
            <div className={styles.sliderRow}>
              <label><span>Throttle</span><span>{customThr} %</span></label>
              <input type="range" min={30} max={100} step={1} value={customThr} onChange={(e) => setCustomThr(Number(e.target.value))} />
              <input type="text" value={String(customThr)} onChange={(e) => setCustomThr(Number(e.target.value) || 0)} placeholder="30 - 100" />
            </div>
            <div className={styles.sliderRow}>
              <label><span>Mission Label</span><span>for GCS phase</span></label>
              <input type="text" value={customLabel} onChange={(e) => setCustomLabel(e.target.value)} placeholder="Custom Mission" />
              <div style={{ fontSize: 10, color: "var(--muted)", lineHeight: 1.4 }}>Custom drives Willans+ISA → power, CHT τ24s, vibration 1×/2× — same physics as DRDO 4.</div>
            </div>
          </div>
          <div className={styles.customActions}>
            <button
              type="button"
              className={styles.btnPrimary}
              onClick={async () => {
                try {
                  const qs = new URLSearchParams({ altitude: String(customAlt), ambient: String(customAmb), throttle: String(customThr), label: customLabel });
                  const res = await fetch(`${TWIN_API}/mission/custom?${qs.toString()}`, { method: "POST" });
                  if (!res.ok) throw new Error(await res.text());
                  setScenario("custom");
                  setLiveTelemetry(null);
                  setHistory([]);
                  setRetryKey((k) => k + 1);
                } catch (e) {
                  alert(`Custom start failed: ${e}`);
                }
              }}
            >
              Apply Custom &amp; Start 10 Hz
            </button>
            <button type="button" className={styles.btnGhost} onClick={() => setShowCustom(false)}>Hide Panel (still has Custom button)</button>
            <span style={{ fontFamily: "var(--font-geist-mono)", fontSize: 10, color: "var(--muted)" }}>Tip: drag sliders or type values — Apply re-seeds TwinState + DigitalTwinCore + ML pipeline</span>
          </div>
        </section>
      ) : (
        <div style={{ maxWidth: 1560, margin: "0 auto 10px", padding: "0 clamp(12px,1.6vw,16px)", display: "flex", justifyContent: "flex-end" }}>
          <button type="button" className={styles.btnGhost} onClick={() => setShowCustom(true)} style={{ padding: "6px 12px", fontSize: 11 }}>Open Custom Builder — Adjust Anything</button>
        </div>
      )}

      <section className={styles.dashboardGrid}>
        <div className={styles.scenePanel}>
          <DroneScene telemetry={display ?? OFFLINE_PLACEHOLDER} isLive={!!isLive} viewMode={viewMode} />
            <div style={{ position: "absolute", top: 14, left: 14, display: "flex", gap: 6, zIndex: 2 }}>
              <button type="button" onClick={() => setViewMode("drone")} style={{ display: "flex", alignItems: "center", gap: 6, padding: "6px 10px", border: "1px solid var(--line)", borderRadius: 6, background: viewMode === "drone" ? "var(--cyan)" : "rgba(5,16,21,0.78)", color: viewMode === "drone" ? "#071518" : "var(--cyan)", fontFamily: "var(--font-geist-mono)", fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", cursor: "pointer" }}><Plane size={12} /> MALE UAV</button>
              <button type="button" onClick={() => setViewMode("engine")} style={{ display: "flex", alignItems: "center", gap: 6, padding: "6px 10px", border: "1px solid var(--line)", borderRadius: 6, background: viewMode === "engine" ? "var(--cyan)" : "rgba(5,16,21,0.78)", color: viewMode === "engine" ? "#071518" : "var(--cyan)", fontFamily: "var(--font-geist-mono)", fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", cursor: "pointer" }}><Wrench size={12} /> AERO ENGINE — Rotax 912</button>
              {replay ? (
                <span style={{ display: "flex", alignItems: "center", padding: "6px 10px", border: "1px solid var(--amber)", borderRadius: 6, background: "rgba(232,172,61,0.12)", color: "var(--amber)", fontFamily: "var(--font-geist-mono)", fontSize: 10, fontWeight: 700, letterSpacing: "0.06em" }}>
                  REPLAY M{replay.missionId} · {fmtClock(replay.rows[replay.idx]?.elapsedS ?? 0)}
                </span>
              ) : null}
            </div>
          <span className={styles.inspectBadge} title="Drag to orbit, scroll to zoom"><Rotate3D size={16} /></span>
          <div className={styles.sceneOverlay}>
            <div>
              <span>ALT</span>
              <strong>{isLive && display ? `${display.altitude.toLocaleString()} m` : "--"}</strong>
            </div>
            <div>
              <span>RPM</span>
              <strong>{isLive && display ? display.rpm.toLocaleString() : "--"}</strong>
            </div>
            <div>
              <span>PHASE</span>
              <strong>{isLive && display ? display.phase : "OFFLINE"}</strong>
            </div>
            <div>
              <span>FAULT PART</span>
              <strong>{isLive && display ? (display.fault === "none" ? "NONE" : display.faultComponent.replaceAll("_", " ").toUpperCase()) : "--"}</strong>
            </div>
            <div>
              <span>SOURCE</span>
              <strong>{isLive && display ? "SIM" : "NO DATA"}</strong>
            </div>
          </div>
        </div>

        <aside className={styles.commandPanel}>
          <div className={styles.panelHeader}>
            <Plane size={18} />
            <div>
              <h2>{selectedScenario.label}</h2>
              <p>{selectedScenario.mission}</p>
            </div>
          </div>
          <div className={styles.healthDial}>
            <span style={{ "--health": `${(display?.health ?? 0) * 3.6}deg` } as CSSProperties} />
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2px' }}>
              <strong style={{ color: status === 'CRITICAL' ? 'var(--red)' : status === 'CAUTION' ? 'var(--amber)' : 'var(--green)' }}>
                {isLive && display ? `${display.health.toFixed(0)}%` : "--"}
              </strong>
              <small>Engine health</small>
              {isLive && display && (
                <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.1em', color: status === 'CRITICAL' ? 'var(--red)' : status === 'CAUTION' ? 'var(--amber)' : 'var(--green)' }}>
                  {status === 'NOMINAL' ? 'HEALTHY' : status}
                </span>
              )}
            </div>
          </div>
          <div className={styles.statGrid}>
            <div>
              <small>RUL</small>
              <strong style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                {isLive && display && <span style={{ width: 6, height: 6, borderRadius: '50%', background: display.rul > 100 ? 'var(--green)' : display.rul > 50 ? 'var(--amber)' : 'var(--red)' }} />}
                {isLive && display ? `${display.rul} cycles` : "--"}
              </strong>
            </div>
            <div>
              <small>Anomaly</small>
              <strong style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                {isLive && display && <span style={{ width: 6, height: 6, borderRadius: '50%', background: display.anomaly < 0.5 ? 'var(--green)' : display.anomaly < 0.8 ? 'var(--amber)' : 'var(--red)' }} />}
                {isLive && display ? `${(display.anomaly * 100).toFixed(0)}%` : "--"}
              </strong>
            </div>
            <div>
              <small>Brake power</small>
              <strong style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                {isLive && display && <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--green)' }} />}
                {isLive && display ? `${display.powerKw.toFixed(1)} kW` : "--"}
              </strong>
            </div>
            <div>
              <small>Air density</small>
              <strong style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                {isLive && display && <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--cyan)' }} />}
                {isLive && display ? `${display.densityRatio.toFixed(2)} sigma` : "--"}
              </strong>
            </div>
          </div>
          <SignalTrace history={isLive && display ? history : []} />
          {isLive && display ? (
            <div className={styles.metaLine}>
              {fmtClock(display.elapsedS)} · frame {display.cycle} @10Hz · {display.source} · {display.core.status}
            </div>
          ) : null}
        </aside>

        <section className={styles.telemetryPanel}>
          <div className={styles.panelTitle}>
            <Gauge size={18} />
            <h2>Live engine telemetry</h2>
          </div>
          {!isLive || !display ? (
            <div className={styles.emptyState}>
              Awaiting live backend stream. No mock data is rendered in this view.
              <br />
              Start backend: <code>python3 -m uvicorn api:app --host 127.0.0.1 --port 8000</code>
            </div>
          ) : (
            <>
              {(() => {
                const warns = [
                  display.rpm > 3700,
                  display.cht > 120 && display.cht <= 135,
                  display.egt > 760 && display.egt <= 800,
                  display.oilPressure < 36 && display.oilPressure >= 28,
                  display.oilTemp > 115 && display.oilTemp <= 130,
                  display.vibration > 0.3 && display.vibration <= 0.5,
                  display.battery < 50 && display.battery >= 30
                ].filter(Boolean).length;
                const dangers = [
                  display.cht > 135,
                  display.egt > 800,
                  display.oilPressure < 28,
                  display.oilTemp > 130,
                  display.vibration > 0.5,
                  display.battery < 30,
                  display.alternator < 13
                ].filter(Boolean).length;
                const nominals = 10 - warns - dangers;
                return (
                  <div style={{ fontSize: '11px', color: 'var(--muted)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Activity size={12} />
                     <span>{nominals}/8 sensors nominal · <span style={{ color: warns > 0 ? 'var(--amber)' : 'inherit' }}>{warns} caution</span> · <span style={{ color: dangers > 0 ? 'var(--red)' : 'inherit' }}>{dangers} danger</span></span>
                  </div>
                );
              })()}
              <MetricBar label="RPM" value={display.rpm} unit="RPM" max={3880} tone={display.rpm > 3700 ? "warn" : "normal"} />
              <MetricBar label="Cylinder Head Temp (CHT)" value={display.cht} unit="°C" max={150} tone={display.cht > 135 ? "danger" : display.cht > 120 ? "warn" : "normal"} />
              <MetricBar label="Exhaust Gas Temp (EGT)" value={display.egt} unit="°C" max={850} tone={display.egt > 800 ? "danger" : display.egt > 760 ? "warn" : "normal"} />
              <MetricBar label="Oil Pressure" value={display.oilPressure} unit="PSI" max={75} tone={display.oilPressure < 28 ? "danger" : display.oilPressure < 36 ? "warn" : "normal"} />
              <MetricBar label="Oil Temperature" value={display.oilTemp} unit="°C" max={150} tone={display.oilTemp > 130 ? "danger" : display.oilTemp > 115 ? "warn" : "normal"} />
              <MetricBar label="Vibration RMS" value={display.vibration} unit="g" max={0.6} tone={display.vibration > 0.5 ? "danger" : display.vibration > 0.3 ? "warn" : "normal"} />
              <MetricBar label="Fuel Flow" value={display.fuelFlow} unit="L/h" max={32} />
              <MetricBar label="Battery SoC" value={display.battery} unit="%" max={100} tone={display.battery < 30 ? "danger" : display.battery < 50 ? "warn" : "normal"} />
              <MetricBar label="Alternator" value={display.alternator} unit="V" max={15} tone={display.alternator < 13 ? "danger" : "normal"} />
              <MetricBar label="Injection Timing" value={display.injectionTiming} unit="°BTDC" max={30} />
            </>
          )}
        </section>

        <section className={styles.alertPanel}>
          <div className={styles.panelTitle}>
            <BrainCircuit size={18} />
            <h2>Fault prediction & XAI</h2>
          </div>
          {!isLive || !display ? (
            <div className={styles.emptyState}>Awaiting backend ML assessment. No mock diagnosis shown.</div>
          ) : (
            <>
              <div className={styles.faultBox} style={{ border: `1px solid ${status === 'CRITICAL' ? 'var(--red)' : status === 'CAUTION' ? 'var(--amber)' : 'var(--green)'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <small>{display.ml.available ? `${display.ml.modelType} model verdict · ${Math.round(display.ml.confidence * 100)}% confidence${display.ml.confidence < 0.5 ? " (low — monitoring)" : ""}` : (display.ml.modelType === "replay" ? "Replay mode — recorded fault label, no live inference" : "Model runtime initializing")}</small>
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: status === 'CRITICAL' ? 'var(--red)' : status === 'CAUTION' ? 'var(--amber)' : 'var(--green)', animation: status !== 'NOMINAL' ? 'pulse 1.5s infinite' : 'none', flexShrink: 0 }} />
                </div>
                <strong style={{ color: status === 'CRITICAL' ? 'var(--red)' : status === 'CAUTION' ? 'var(--amber)' : 'inherit' }}>
                  {display.fault === "none" ? "No active fault" : display.fault.replaceAll("_", " ")}
                </strong>
                <p>
                  {display.ml.available ? `ML diagnosis: ${display.ml.diagnosis.replaceAll("_", " ")}. Physical twin remains the controlling safety source.` : (display.ml.detail ?? "Physical twin remains active while the model runtime is restored.")}
                </p>
                <p className={styles.metaLine}>
                  Backend source: {display.source} · Core: {display.core.status} · Warnings: {display.core.status === "UNKNOWN" ? "none" : (display.fault !== "none" ? display.fault : "none")}
                </p>
              </div>
              <div className={styles.shapList}>
                {display.shap.map((item) => (
                  <div className={styles.shapRow} key={item.feature}>
                    <span>{item.feature}</span>
                    <div>
                      <i style={{ width: `${clamp(item.impact * 100, 4, 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </section>

        <section className={styles.subsystemPanel}>
          <div className={styles.panelTitle}>
            <Cpu size={18} />
            <h2>Subsystem health</h2>
          </div>
          {!isLive || !display ? (
            <div className={styles.emptyState}>Awaiting backend health indices…</div>
          ) : (
            <div className={styles.subsystemGrid}>
              {(() => {
                const thermal = 100 - display.core.thermal;
                const electrical = display.core.electrical;
                const combustion = display.core.combustion;
                const lubrication = display.core.lubrication;
                const getColor = (val: number) => val > 80 ? 'var(--green)' : val >= 50 ? 'var(--amber)' : 'var(--red)';
                return (
                  <>
                    <div style={{ borderTop: `3px solid ${getColor(thermal)}` }}>
                      <Thermometer size={18} />
                      <span>Thermal</span>
                      <strong style={{ color: getColor(thermal) }}>{thermal.toFixed(0)}%</strong>
                    </div>
                    <div style={{ borderTop: `3px solid ${getColor(electrical)}` }}>
                      <BatteryCharging size={18} />
                      <span>Electrical</span>
                      <strong style={{ color: getColor(electrical) }}>{electrical.toFixed(0)}%</strong>
                    </div>
                    <div style={{ borderTop: `3px solid ${getColor(combustion)}` }}>
                      <Activity size={18} />
                      <span>Combustion</span>
                      <strong style={{ color: getColor(combustion) }}>{combustion.toFixed(0)}%</strong>
                    </div>
                    <div style={{ borderTop: `3px solid ${getColor(lubrication)}` }}>
                      <Wrench size={18} />
                      <span>Lubrication</span>
                      <strong style={{ color: getColor(lubrication) }}>{lubrication.toFixed(0)}%</strong>
                    </div>
                  </>
                );
              })()}
            </div>
          )}
          {isLive && display ? (
            <div className={styles.metaLine}>
              BSFC {display.core.bsfc ? `${display.core.bsfc} g/kWh` : "--"} · Status {display.core.status} · Density {display.densityRatio.toFixed(3)}
            </div>
          ) : null}
        </section>

        <section className={styles.trendPanel}>
          <div className={styles.panelTitle}>
            <BarChart3 size={18} />
            <h2>Efficiency & health trends</h2>
          </div>
          {!isLive || history.length < 2 ? (
            <div className={styles.emptyState}>Awaiting live trend data — trends are derived only from the backend 10 Hz stream, never synthesized.</div>
          ) : (
            <div>
              {(() => {
                const eff = history.map((h) => h.efficiency);
                const hl = history.map((h) => h.health);
                const rul = history.map((h) => h.rul);
                const mkPoints = (vals: number[], min: number, max: number) =>
                  vals.map((v, i) => `${(i / Math.max(1, vals.length - 1)) * 100},${100 - clamp((v - min) / (max - min) * 100, 0, 100)}`).join(" ");
                return (
                  <>
                    <div className={styles.miniTrend}>
                      <div className={styles.miniTrendHeader}><span>Engine efficiency</span><strong>{eff.at(-1)?.toFixed(1)} %</strong></div>
                      <svg viewBox="0 0 100 36" preserveAspectRatio="none">
                        <polyline fill="none" stroke="var(--cyan)" strokeWidth={1.9} points={mkPoints(eff, 18, 38)} />
                      </svg>
                    </div>
                    <div className={styles.miniTrend}>
                      <div className={styles.miniTrendHeader}><span>Health</span><strong>{hl.at(-1)?.toFixed(0)} %</strong></div>
                      <svg viewBox="0 0 100 36" preserveAspectRatio="none">
                        <polyline fill="none" stroke="var(--green)" strokeWidth={1.9} points={mkPoints(hl, 30, 100)} />
                      </svg>
                    </div>
                    <div className={styles.miniTrend}>
                      <div className={styles.miniTrendHeader}><span>RUL</span><strong>{rul.at(-1)} cycles</strong></div>
                      <svg viewBox="0 0 100 36" preserveAspectRatio="none">
                        <polyline fill="none" stroke="var(--amber)" strokeWidth={1.9} points={mkPoints(rul, 0, 230)} />
                      </svg>
                    </div>
                    <div className={styles.trendNote}>Rolling 60 samples @10 Hz — physics-simulation stream + DigitalTwinCore health</div>
                  </>
                );
              })()}
            </div>
          )}
            <div className={styles.missionSection}>
              <div className={styles.missionHeading}>
                <ClipboardList size={14} />
                <span>Mission-wise health reports</span>
              </div>
              {replay ? (
                <div style={{ border: "1px solid var(--cyan)", borderRadius: 8, padding: "8px 10px", marginBottom: 8, background: "rgba(0,255,255,0.04)" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: 12 }}>
                    <strong style={{ color: "var(--cyan)" }}>REPLAY M{replay.missionId} {replay.engineId} · {replay.fault.replaceAll("_", " ")}</strong>
                    <span style={{ opacity: 0.8 }}>{replay.idx + 1} / {replay.rows.length}</span>
                  </div>
                  <div style={{ height: 4, borderRadius: 2, background: "rgba(255,255,255,0.1)", margin: "6px 0", overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${((replay.idx + 1) / replay.rows.length) * 100}%`, background: "var(--cyan)" }} />
                  </div>
                  <div style={{ display: "flex", gap: 6 }}>
                    <button type="button" className={styles.btnGhost} style={{ padding: "4px 10px", fontSize: 11 }}
                      onClick={() => setReplay((cur) => (cur ? { ...cur, playing: !cur.playing } : cur))}>
                      {replay.playing ? "Pause" : (replay.idx >= replay.rows.length - 1 ? "Replay again" : "Play")}
                    </button>
                    <button type="button" className={styles.btnGhost} style={{ padding: "4px 10px", fontSize: 11 }} onClick={stopReplay}>
                      Back to live
                    </button>
                  </div>
                </div>
              ) : null}
              {missions.length === 0 ? (
                <div className={styles.emptyState}>No mission history yet — run <code>python3 generate_engine_data.py</code> to seed <code>engine_data.csv</code>.</div>
              ) : (
                <div className={styles.missionList}>
                  {missions.slice(0, 6).map((m) => (
                    <button
                      type="button"
                      key={`${m.mission_id}-${m.engine_id}`}
                      title={`Replay mission ${m.mission_id} at 10 Hz`}
                      onClick={() => void startReplay(m.mission_id)}
                      className={`${styles.missionRow} ${m.fault_type === "none" ? styles.healthy : styles.faulted}`}
                      style={{ width: "100%", textAlign: "left", cursor: "pointer", font: "inherit", color: "inherit" }}
                    >
                      <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: m.fault_type === 'none' ? 'var(--green)' : 'var(--amber)' }} />
                        M{m.mission_id} {m.engine_id}
                      </span>
                      <span>{m.scenario}</span>
                      <span className={`${styles.missionFault} ${m.fault_type === "none" ? styles.healthy : styles.faulted}`}>
                        {m.fault_type.replaceAll("_", " ")}
                      </span>
                      <span>{m.cycles}cyc</span>
                    </button>
                  ))}
                </div>
              )}
              <div className={styles.trendNote}>Source: GET /missions from engine_data.csv · click a row for full 10 Hz replay</div>
          </div>
        </section>

        <section className={styles.advisoryPanel}>
          <div className={styles.panelTitle}>
            <ShieldCheck size={18} />
            <h2>Maintenance advisory</h2>
          </div>
          {!isLive || !display ? (
            <p>Awaiting live health status. Advisory is generated only from backend physics-simulation + ML, never from mock data.</p>
          ) : (
            <>
              <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start', padding: '12px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', borderLeft: `4px solid ${status === 'CRITICAL' ? 'var(--red)' : status === 'CAUTION' ? 'var(--amber)' : 'var(--green)'}` }}>
                <div style={{ marginTop: '2px', color: status === 'CRITICAL' ? 'var(--red)' : status === 'CAUTION' ? 'var(--amber)' : 'var(--green)' }}>
                  {status === 'CRITICAL' ? <AlertOctagon size={20} /> : status === 'CAUTION' ? <AlertTriangle size={20} /> : <CheckCircle2 size={20} />}
                </div>
                <p style={{ margin: 0, fontSize: '14px', lineHeight: '1.5', fontWeight: status !== 'NOMINAL' ? 500 : 400 }}>
                  {status === "CRITICAL"
                    ? "Recommend recovery window and inspection of the red highlighted propulsion subsystem before the next sortie."
                    : status === "CAUTION"
                      ? `Plan maintenance after mission. Inspect the highlighted ${display.faultComponent} subsystem and trend its related signals at full telemetry rate (10 Hz). Core warnings: ${display.core.status}.`
                      : "Continue mission. Store this run as healthy baseline data for model drift monitoring and future fleet learning."}
                </p>
              </div>
              {replay ? (
                <p style={{ opacity: 0.7, fontSize: '12px' }}>Replay mode — advisory follows the recorded fault above; live backend advisory is paused.</p>
              ) : advisories.length > 0 ? (
                <ul style={{ listStyle: 'none', margin: '10px 0 0 0', padding: 0, display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {advisories.map((a, i) => {
                    const lvl = a.level.toUpperCase();
                    const color = lvl === "CRITICAL" ? 'var(--red)' : (lvl === "URGENT" || lvl === "WARNING") ? 'var(--amber)' : 'var(--green)';
                    return (
                      <li key={i} style={{ borderLeft: `3px solid ${color}`, padding: '6px 10px', background: 'rgba(255,255,255,0.02)', borderRadius: '0 6px 6px 0', fontSize: '13px', lineHeight: '1.45' }}>
                        <strong style={{ color }}>[{lvl}]</strong>{' '}{a.action}
                        {a.due_cycles === "immediate" ? <span style={{ opacity: 0.75 }}> — immediate action</span>
                          : (a.due_cycles && a.due_cycles !== "N/A" ? <span style={{ opacity: 0.75 }}> — due in {a.due_cycles} cycles</span> : null)}
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <p style={{ opacity: 0.6, fontSize: '12px' }}>Contacting maintenance advisory service…</p>
              )}
            </>
          )}
          <div className={styles.canFrame}>
            <code>ID 0x101</code>
            <code>DLC 4</code>
            <code>{isLive && display ? `RPM ${display.rpm}` : "RPM --"}</code>
            <code>{isLive && display ? `CHT ${display.cht.toFixed(0)}` : "CHT --"}</code>
            <code>{isLive && display ? `PWR ${display.powerKw.toFixed(1)} kW` : "PWR --"}</code>
            <code>{isLive && display ? `ML ${display.ml.available ? display.ml.diagnosis.toUpperCase() : "LOADING"}` : "ML --"}</code>
            <code>{isLive && display ? display.source.toUpperCase() : "NO DATA"}</code>
          </div>
        </section>
      </section>

      <section className={styles.coverage}>
        {requirements.map((item) => (
          <span key={item}>
            <ShieldCheck size={14} />
            {item}
          </span>
        ))}
      </section>
      {/* Alert toasts — bottom-right stack, newest on top */}
      <div aria-live="assertive" style={{ position: "fixed", right: 16, bottom: 16, zIndex: 60, display: "flex", flexDirection: "column-reverse", gap: 8, maxWidth: 360 }}>
        {toasts.map((t) => {
          const color = t.level === "CRITICAL" ? "var(--red)" : t.level === "CAUTION" ? "var(--amber)" : "var(--green)";
          const Icon = t.level === "CRITICAL" ? AlertOctagon : t.level === "CAUTION" ? AlertTriangle : CheckCircle2;
          return (
            <div key={t.id} role="alert"
              style={{ display: "flex", gap: 10, alignItems: "flex-start", background: "rgba(8,18,24,0.95)", border: "1px solid var(--line)", borderLeft: `4px solid ${color}`, borderRadius: 8, padding: "10px 12px", boxShadow: "0 8px 28px rgba(0,0,0,0.5)", cursor: "pointer" }}
              onClick={() => dismissToast(t.id)} title="Click to dismiss">
              <span style={{ color, marginTop: 1, flexShrink: 0 }}><Icon size={18} /></span>
              <span style={{ fontSize: 13, lineHeight: 1.45 }}>
                <strong style={{ color }}>{t.title}</strong>
                <br />
                <span style={{ opacity: 0.85 }}>{t.msg}</span>
              </span>
            </div>
          );
        })}
      </div>
    </main>
  );
}
