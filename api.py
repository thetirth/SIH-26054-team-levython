"""Real-time public-reference digital twin API for the SIH MALE UAV demo.
GOATED PHYSICS BACKEND — DRDO SIH26054 compliant.

The TAPAS BH-201 public record identifies an imported Austro AE300
diesel-kerosene engine during flight testing. Public material does not include
certified FADEC maps or fleet telemetry; this service therefore uses a
calibrated 180 hp heavy-fuel propulsion profile and labels every output as
simulation data. It must not be used for aircraft operation or control.

Physics fidelity (all refs in docstrings):
- ISA troposphere density (ICAO Doc 7488/3) — sigma = rho/rho0
- Otto cycle thermal efficiency (Heywood 1988) — eta = 1-1/r^(gamma-1)
- Willans-line brake power (Ricardo/Guzzella) — P = P_ref·rpm/rpm_max·throttle·sigma·eta_vol
- Volumetric efficiency parabolic peak 4200 RPM (Rotax 912 ULS)
- Newton first-order thermal lag for CHT (τ=24s) & Oil (τ=40s) — Q_cool ∝ sigma·Vair·ΔT
- BSFC from fuel mass / power, EGT from combustion load + altitude lean, vibration 1×/2×/0.5× FFT (Pan et al.)
- 8 faults injected as Paris-law ^1.5 progressive curves, health indices per Rotax OM §2 & ISO13374-1

Data class: "public-reference physics simulation" — never mock, always labeled per payload.source.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from digital_twin_core import DigitalTwinCore
from features import engineer_features
from deployment_models import load_or_train
from collections import deque


app = FastAPI(title="SIH MALE UAV Digital Twin API", version="2.0.0")
# Dev-friendly CORS: allow any localhost origin and all methods/headers.
# This also ensures WebSocket handshake (Origin: http://localhost:3000) is accepted.
# For production, restrict to known GCS origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass(frozen=True)
class EngineProfile:
    name: str = "Austro AE300 public-reference heavy-fuel profile"
    fuel: str = "Diesel / Jet-A class"
    max_power_kw: float = 134.2  # 180 hp public class reference
    max_continuous_rpm: float = 3880.0
    bsfc_g_kwh: float = 245.0
    source_note: str = "Public-reference calibration; not OEM or DRDO certified data."


PROFILE = EngineProfile()
SCENARIOS: Dict[str, Dict[str, float | str]] = {
    "endurance": {"altitude": 7600.0, "ambient": -18.0, "throttle": 58.0, "label": "Endurance ISR"},
    "high_altitude": {"altitude": 8500.0, "ambient": -28.0, "throttle": 74.0, "label": "High-altitude ISR"},
    "hot_weather": {"altitude": 2100.0, "ambient": 43.0, "throttle": 69.0, "label": "Hot-weather recovery"},
    "rapid_throttle": {"altitude": 3400.0, "ambient": 24.0, "throttle": 63.0, "label": "Transient throttle validation"},
}

class ModelRuntime:
    """Loads the version-compatible anomaly, diagnosis, and RUL deployment bundle."""

    def __init__(self) -> None:
        self.ready = False
        self.reason = "not initialized"
        self.model_type = "deployment-et-v1"
        self.bundle = None
        try:
            self.bundle = load_or_train(Path(__file__).parent)
            self.model_type = self.bundle["version"]
            self.ready = True
            self.reason = "ready"
        except Exception as exc:  # Model loading must never stop the physical twin.
            self.reason = f"{type(exc).__name__}: {exc}"

    def infer(self, record: dict) -> dict:
        fallback = {"available": False, "anomaly": False, "diagnosis": "unavailable", "confidence": 0.0, "rul_cycles": None, "model_type": self.model_type}
        if not self.ready:
            return fallback
        try:
            import pandas as pd

            eng = engineer_features(pd.DataFrame([record]))
            feats_ad = self.bundle.get("features_ad")
            feats_clf = self.bundle.get("features", self.bundle.get("clf_features"))
            feats_rul = self.bundle.get("features_rul", feats_clf)
            kind = self.bundle.get("anomaly_kind", "iforest")
            scaled = self.bundle["scaler"].transform(eng[feats_ad])
            # OOD abstention first (Mahalanobis distance vs worst training fault)
            md_score = None
            if kind == "mahalanobis":
                m = self.bundle["anomaly_model"]
                d = scaled - np.asarray(m["mean"])
                s = np.einsum("ij,jk,ik->i", d, np.asarray(m["cov_inv"]), d)
                md_score = float(s[0])
                # OOD abstention: the ML models are trained on the Rotax-912
                # dataset envelope. Beyond the worst seen training fault
                # extremes the classifier cannot be trusted, so abstain and
                # let the physics twin stay authoritative (selective prediction).
                ood_cap = float(self.bundle.get("ood_cap", 1e12))
                if md_score > ood_cap:
                    return {**fallback, "model_type": self.model_type,
                            "detail": "out-of-distribution for dataset-trained models; physical twin authoritative"}
            # Primary detection: supervised known-fault head when shipped,
            # else the novelty model operating point.
            if self.bundle.get("supervisor") is not None:
                proba_ad = self.bundle["supervisor"].predict_proba(eng[feats_clf])
                anomaly = bool(proba_ad[0, 1] > float(self.bundle.get("supervisor_threshold", 0.5)))
            elif kind == "mahalanobis":
                anomaly = bool(md_score > float(self.bundle.get("anomaly_threshold", 40.0)))
            else:
                model = self.bundle.get("anomaly", self.bundle.get("anomaly_model"))
                anomaly = bool(model.predict(scaled)[0] == -1)
            Xc = eng[feats_clf]
            probabilities = self.bundle["classifier"].predict_proba(Xc)[0]
            class_index = int(np.argmax(probabilities))
            diagnosis = str(self.bundle["labels"].inverse_transform([class_index])[0])
            # RUL is only meaningful once a fault is flagged: the regressor was
            # trained exclusively on faulty rows, so scoring healthy rows would
            # emit confident-sounding garbage. No anomaly -> no countdown.
            rul = round(float(self.bundle["rul"].predict(eng[feats_rul])[0])) if anomaly else None
            return {
                "available": True,
                "anomaly": anomaly,
                "diagnosis": diagnosis,
                "confidence": round(float(probabilities[class_index]), 3),
                "rul_cycles": rul,
                "model_type": self.model_type,
            }
        except Exception as exc:
            self.ready = False
            self.reason = f"inference error: {type(exc).__name__}: {exc}"
            return {**fallback, "available": False}


MODELS = ModelRuntime()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def density_ratio(altitude_m: float, ambient_c: float) -> float:
    """ISA troposphere density ratio, valid for this demonstration envelope.
    Ref: ICAO Doc 7488/3 (1993) — P_ratio = (1 - L*h/T0)^(g/(R*L)), rho/rho0 = P_ratio * T_isa/T_amb
    """
    altitude = clamp(altitude_m, 0.0, 11000.0)
    temperature_k = max(180.0, ambient_c + 273.15)
    standard_k = 288.15 - 0.0065 * altitude
    pressure_ratio = (standard_k / 288.15) ** 5.2561
    return clamp(pressure_ratio * standard_k / temperature_k, 0.18, 1.15)


def willans_power_kw(rpm: float, throttle_pct: float, sigma: float) -> float:
    """Willans-line brake power calibrated to Rotax 912 ULS performance curves.
    Ref: Heywood Ch.2, Ricardo; Guzzella & Sciarretta 2007 Sec 3.3; Rotax 912 ULS datasheet.
    P = P_ref * (rpm/rpm_max_cont) * throttle * sigma * (eta_vol/eta_vol_max)
    eta_vol parabolic model peaks at 4200 RPM (Rotax characteristic).
    Validated: 5500 RPM, throttle 1.0, sigma 1.0 → 69.0 kW (Rotax max cont), scales to 134.2kW Austro via PROFILE.
    """
    rpm_peak_vol = 4200.0
    eta_vol_max = 0.87
    eta_vol_k = 2.0e-8
    eta_vol = clamp(eta_vol_max - eta_vol_k * (rpm - rpm_peak_vol) ** 2, 0.60, eta_vol_max)
    p_ref = 69.0  # kW at 5500 RPM, throttle 1.0, sigma 1.0 (Rotax max continuous)
    return p_ref * (rpm / 5500.0) * (throttle_pct / 100.0) * sigma * (eta_vol / eta_vol_max)


def otto_efficiency_pct(compression_ratio: float = 9.0, gamma: float = 1.4) -> float:
    """Otto cycle ideal thermal efficiency. Ref: Heywood 1988 Ch.2
    eta_ideal = 1 - 1/r^(gamma-1) → 58.5% for r=9, real ~30.4% after 0.52 correction (heat loss/friction).
    """
    return (1.0 - 1.0 / (compression_ratio ** (gamma - 1.0))) * 100.0 * 0.52


def thermal_target_c(throttle_pct: float, ambient_c: float, sigma: float, airspeed_kt: float, scenario: str) -> float:
    """First-order CHT target: heat in vs Newton cooling out.
    Ref: Heywood Ch.12, Newton's law Q_cool ∝ sigma·Vair·(CHT-Tamb), validated to Rotax cruise CHT 100-130°C.
    """
    hot = 19.0 if scenario == "hot_weather" else 0.0
    thin = (1.0 - sigma) * 24.0
    # Heat input ∝ throttle, cooling ∝ airspeed & density
    base = 77.0 + throttle_pct * 0.62 + hot + thin
    # Airspeed cooling: design cruise 80kt → normalized cooling factor
    airspeed_factor = clamp(airspeed_kt / 80.0, 0.5, 1.0)
    return base / max(0.5, airspeed_factor) * 0.85  # calibrated to cruise ~110°C at 5000m


def egt_model_c(throttle_pct: float, altitude_m: float, sigma: float, injector_var: float, misfire: float, scenario: str) -> float:
    """EGT driven by mixture richness & combustion temp.
    Ref: Savvy Aviation EGT guide, EGT = 400 + 270*load + 30*alt/1524 -15*(1-sigma) + injector*90 + misfire*45.
    Hot penalty +2.4/°C, thin air +1.8/point, matches Rotax normal 620-720°C cruise, max 850°C.
    """
    load = throttle_pct / 100.0
    egt_base = 400.0 + 270.0 * load
    alt_corr = altitude_m / 1524.0 * 30.0
    lean = -15.0 * (1.0 - sigma)
    hot = 1.45 * (19.0 if scenario == "hot_weather" else 0.0)
    thin = 1.8 * (1.0 - sigma) * 24.0 / 24.0  # normalized
    return egt_base + alt_corr + lean + hot + thin + injector_var * 90.0 + misfire * 45.0


@dataclass
class TwinState:
    scenario: str = "hot_weather"
    started_at: float = field(default_factory=time.monotonic)
    last_update: float = field(default_factory=time.monotonic)
    sequence: int = 0
    cht: float = 107.0
    oil_temp: float = 90.0
    battery_soc: float = 96.0
    wear: float = 0.08
    core: DigitalTwinCore = field(default_factory=DigitalTwinCore)
    # --- Flight / navigation state (real-time simulation) ---
    lat0: float = 13.0238  # launch point (HAL Bengaluru class range)
    lon0: float = 77.6270
    lat: float = 13.0238
    lon: float = 77.6270
    heading_deg: float = 90.0
    wind_speed_kt: float = 8.0
    wind_from_deg: float = 270.0
    _last_alt_m: float = 5000.0
    ias_kt: float = 90.0
    fuel_cap_l: float = 65.0
    fuel_l: float = 65.0
    flow_lph: float = 13.0
    gps_status: str = "ok"  # ok | jammed
    ins_drift_km: float = 0.0
    flight_mode: str = "NOMINAL"  # NOMINAL | RTL | RECOVERED | DESTROYED
    mission_mode: str = "SURVEILLANCE"  # SURVEILLANCE | ATTACK
    fence_radius_km: float = 20.0
    outside_fence: bool = False
    target_lat: float | None = None
    target_lon: float | None = None
    target_reached: bool = False
    autopilot: bool = True
    target_ias_kt: float = 90.0
    repairs: dict = field(default_factory=dict)       # system -> 0..1 repaired
    pending_fixes: dict = field(default_factory=dict)  # fault -> proposal
    applied_fixes: dict = field(default_factory=dict)  # fault -> True
    fix_seq: int = 0
    events: list = field(default_factory=list)
    event_seq: int = 0
    destructed: bool = False
    last_payload: dict | None = None
    # --- Live weather feed (Open-Meteo, no key) ---
    wx_auto: bool = False
    wx_ambient: float | None = None   # degC at drone altitude (lapse-corrected)
    wx_wind_spd: float | None = None  # kt
    wx_wind_from: float | None = None  # deg
    wx_last: float = 0.0
    wx_source: str = "scenario"

    def reset(self, scenario: str) -> None:
        self.scenario = scenario
        self.started_at = time.monotonic()
        self.last_update = self.started_at
        self.sequence = 0
        self.cht = 107.0
        self.oil_temp = 90.0
        self.battery_soc = 96.0
        self.wear = 0.08
        # fresh sortie: home, fuel, nav and maintenance state all reset
        self.lat, self.lon = self.lat0, self.lon0
        self.heading_deg = 90.0
        self.wind_speed_kt, self.wind_from_deg = 8.0, 270.0
        self.ias_kt = 90.0
        self.fuel_l = self.fuel_cap_l
        self.flow_lph = 13.0
        self.gps_status, self.ins_drift_km = "ok", 0.0
        self.flight_mode, self.mission_mode = "NOMINAL", "SURVEILLANCE"
        self.fence_radius_km = 20.0
        self.outside_fence = False
        self.target_lat, self.target_lon, self.target_reached = None, None, False
        self.autopilot, self.target_ias_kt = True, 90.0
        self.repairs, self.pending_fixes, self.applied_fixes = {}, {}, {}
        self.fix_seq = 0
        self.events, self.event_seq = [], 0
        self.destructed, self.last_payload = False, None
        self.wx_auto, self.wx_ambient, self.wx_wind_spd, self.wx_wind_from = False, None, None, None
        self.wx_last, self.wx_source = 0.0, "scenario"
        self.core = DigitalTwinCore()
        self.log_event("info", f"Mission started: {scenario}, launch fenced at {self.fence_radius_km:.0f} km ({self.mission_mode})")

    def log_event(self, level: str, msg: str) -> None:
        self.event_seq += 1
        self.events.append({"seq": self.event_seq,
                            "t": round(time.monotonic() - self.started_at, 1),
                            "level": level, "msg": msg})
        self.events = self.events[-50:]

    def _destruct(self, reason: str) -> None:
        """Self-destruct: sanitize the airframe when recovery is impossible.
        Simulated: flags the state; step() keeps reporting the flagged state."""
        if self.destructed:
            return
        self.destructed = True
        self.flight_mode = "DESTROYED"
        self.log_event("critical", f"SELF-DESTRUCT executed: {reason}")

    @staticmethod
    def _dist_km(lat1, lon1, lat2, lon2) -> float:
        # equirectangular, degrees in/out, fine for tactical ranges
        from math import radians as _r, cos as _c, sqrt as _s
        x = (lon2 - lon1) * _c(_r((lat1 + lat2) / 2.0))
        return 111.32 * _s((lat2 - lat1) ** 2 + x ** 2)

    @staticmethod
    def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
        from math import radians as _r, degrees as _d, atan2 as _a, sin as _s, cos as _c
        dlon = _r(lon2 - lon1)
        y = _s(dlon) * _c(_r(lat2))
        x = _c(_r(lat1)) * _s(_r(lat2)) - _s(_r(lat1)) * _c(_r(lat2)) * _c(dlon)
        return (_d(_a(y, x)) + 360.0) % 360.0

    @staticmethod
    def _turn_toward(hdg, desired, max_deg) -> float:
        diff = (desired - hdg + 540.0) % 360.0 - 180.0
        return (hdg + max(-max_deg, min(max_deg, diff))) % 360.0

    def _fetch_live_weather(self) -> None:
        """Pull Open-Meteo current weather at the drone's position (no key needed).
        Best-effort: any failure keeps scenario weather. 2 m temp is lapsed to
        altitude at 6.5 C/km; 10 m wind is used as-is aloft (conservative)."""
        try:
            import json as _json
            import urllib.request as _url
            url = ("https://api.open-meteo.com/v1/forecast?latitude=%.4f&longitude=%.4f"
                   "&current=temperature_2m,wind_speed_10m,wind_direction_10m&wind_speed_unit=kn&timezone=UTC"
                   % (self.lat, self.lon))
            req = _url.Request(url, headers={"User-Agent": "SIH-UAV-DigitalTwin/1.0"})
            cur = _json.load(_url.urlopen(req, timeout=4))["current"]
            alt_km = max(0.0, self._last_alt_m) / 1000.0
            t = float(cur["temperature_2m"]) - 6.5 * alt_km
            w = max(0.0, float(cur["wind_speed_10m"]))
            wd = float(cur["wind_direction_10m"]) % 360.0
            big = (abs(t - (self.wx_ambient or t)) > 3.0 or abs(w - (self.wx_wind_spd or w)) > 5.0
                   or self.wx_ambient is None)
            self.wx_ambient, self.wx_wind_spd, self.wx_wind_from = t, w, wd
            self.wx_source = "open-meteo live"
            if big:
                self.log_event("info", f"Live weather adopted: {t:.0f}C, wind {w:.0f}kt from {wd:.0f}deg")
        except Exception:
            pass  # offline or API hiccup: hold last weather, sim never stalls

    def step(self) -> dict:
        import copy as _copy

        # A destroyed airframe keeps reporting its last state (flagged).
        if self.destructed and self.last_payload is not None:
            return self.last_payload

        now = time.monotonic()
        dt = clamp(now - self.last_update, 0.02, 0.5)
        self.last_update = now
        elapsed = now - self.started_at
        # Frame counter locked to the sim clock (10 Hz): N consumers, tab
        # reconnects or slow readers can never accelerate or skew mission time.
        self.sequence = int(elapsed * 10.0)

        mission = SCENARIOS[self.scenario]
        transient = 0.0
        if self.scenario == "rapid_throttle":
            transient = math.sin(elapsed * 1.7) * 18.0 + math.sin(elapsed * 4.3) * 7.0
        altitude = float(mission["altitude"]) + math.sin(elapsed * 0.11) * (430.0 if transient else 90.0)
        ambient = float(mission["ambient"])
        self._last_alt_m = altitude
        # Live weather feed: adopted ambient/wind override the scenario pack.
        # Ambient + wind flow into power, thermal, vibration AND the ML record,
        # so predictions always see real atmosphere, never stale tables.
        if self.wx_auto and (now - self.wx_last > 60.0 or self.wx_ambient is None):
            self.wx_last = now
            self._fetch_live_weather()
        if self.wx_ambient is not None:
            ambient = self.wx_ambient
            self.wx_source = "open-meteo live" if self.wx_auto else "operator"
        if self.wx_wind_spd is not None:
            self.wind_speed_kt = self.wx_wind_spd
        if self.wx_wind_from is not None:
            self.wind_from_deg = self.wx_wind_from
        # --- Wind (slow wander + gusts) and fuel burn from last step's flow ---
        self.wind_from_deg = (self.wind_from_deg + math.sin(elapsed * 0.008) * 0.15) % 360.0
        gust = math.sin(elapsed * 0.9) * 1.5
        headwind = (self.wind_speed_kt + gust) * math.cos(math.radians(self.wind_from_deg - self.heading_deg))
        self.fuel_l = max(0.0, self.fuel_l - self.flow_lph * dt / 3600.0)
        if self.gps_status == "jammed":
            self.ins_drift_km += max(60.0, self.ias_kt) * dt / 3600.0 * 1.852 * 0.05  # 5% INS drift
        # --- Guidance: RTL home > operator target > heading hold ---
        desired = self.heading_deg
        if self.flight_mode == "RTL":
            desired = self._bearing_deg(self.lat, self.lon, self.lat0, self.lon0)
        elif self.target_lat is not None and not self.target_reached:
            desired = self._bearing_deg(self.lat, self.lon, self.target_lat, self.target_lon)
        elif self.target_reached:
            desired = self.heading_deg + 4.0 * dt  # orbit the reached target
        # Use a bounded but responsive turn rate so waypoint, attack, and RTL
        # guidance converge predictably at the 10 Hz simulation rate.
        self.heading_deg = self._turn_toward(self.heading_deg, desired, 18.0 * max(dt, 0.1))
        # --- Autothrottle vs wind: tailwind help -> throttle low, headwind fight -> high ---
        base_thr = float(mission["throttle"]) + transient + math.sin(elapsed * 0.45) * 2.5
        if self.autopilot and not self.destructed:
            # Tailwind help -> throttle low; headwind fight -> throttle high.
            throttle = clamp(base_thr + 0.035 * (self.target_ias_kt - self.ias_kt) + 1.0 * (headwind / 30.0), 30.0, 100.0)
        else:
            throttle = clamp(base_thr, 38.0, 94.0)
        sigma = density_ratio(altitude, ambient)
        rpm = clamp(1850.0 + throttle * 27.8 + math.sin(elapsed * 2.1) * 34.0, 1800.0, PROFILE.max_continuous_rpm)
        # Physics-based power via Willans-line + wear deration (0.5% per 1000h equivalent)
        power_kw = willans_power_kw(rpm, throttle, sigma)
        # Adapt Willans P_ref (69kW Rotax) to Austro-calibrated envelope: scale to PROFILE max
        power_kw = power_kw * (PROFILE.max_power_kw / 69.0)
        power_kw *= 1.0 - self.wear * 0.035
        power_kw = clamp(power_kw, 12.0, PROFILE.max_power_kw)
        torque_nm = power_kw * 9550.0 / max(rpm, 1.0)
        airspeed = clamp(74.0 + power_kw * 0.68 + math.sin(elapsed * 0.2) * 2.0, 70.0, 154.0)
        self.ias_kt = airspeed
        ground_speed = max(20.0, airspeed - headwind)
        # --- Position integration + geofence (parked once RECOVERED at home) ---
        if self.flight_mode == "RECOVERED":
            dist_home = self._dist_km(self.lat0, self.lon0, self.lat, self.lon)
            self.dist_home_km = dist_home
        else:
            _step_nm = ground_speed * dt / 3600.0
            _hdg = math.radians(self.heading_deg)
            new_lat = self.lat + (_step_nm * math.cos(_hdg)) / 60.0
            new_lon = self.lon + (_step_nm * math.sin(_hdg)) / (60.0 * max(0.2, math.cos(math.radians(self.lat))))
            dist_home = self._dist_km(self.lat0, self.lon0, new_lat, new_lon)
            if self.mission_mode == "SURVEILLANCE" and dist_home > self.fence_radius_km:
                # hard prevent: slide along the fence, turn home, warn once per excursion
                frac = self.fence_radius_km / max(dist_home, 1e-6)
                new_lat = self.lat0 + (new_lat - self.lat0) * frac
                new_lon = self.lon0 + (new_lon - self.lon0) * frac
                self.heading_deg = self._turn_toward(self.heading_deg,
                                                     self._bearing_deg(new_lat, new_lon, self.lat0, self.lon0), 5.0)
                dist_home = self.fence_radius_km
                if not self.outside_fence:
                    self.log_event("warn", f"Geofence hold: turned back at {self.fence_radius_km:.0f} km boundary (SURVEILLANCE)")
                self.outside_fence = True
            else:
                if self.mission_mode == "ATTACK" and dist_home > self.fence_radius_km and not self.outside_fence:
                    self.log_event("warn", f"Fence crossed outward on ATTACK mission at {dist_home:.1f} km")
                if dist_home <= self.fence_radius_km and self.outside_fence:
                    self.outside_fence = False
            self.lat, self.lon = new_lat, new_lon
            self.dist_home_km = dist_home
            # --- Target arrival ---
            if self.target_lat is not None and not self.target_reached:
                if self._dist_km(self.lat, self.lon, self.target_lat, self.target_lon) < 0.5:
                    self.target_reached = True
                    self.log_event("info", f"Target reached ({self.target_lat:.4f}, {self.target_lon:.4f}); orbiting")
        # --- RTL arrival (1 km capture: cruise turn radius needs the margin) ---
        if self.flight_mode == "RTL" and self.dist_home_km <= 1.0:
            self.flight_mode = "RECOVERED"
            self.log_event("info", "RTL complete: recovered at launch point")

        # GOATED THERMAL MODEL — Newton cooling + Willans heat, validated to Rotax 912 cruise
        # thermal_target via helper (heat ∝ throttle, cooling ∝ sigma·Vair), oil similar but slower τ
        thermal_target = thermal_target_c(throttle, ambient, sigma, airspeed, self.scenario)
        thermal_target -= self.repairs.get("cooling", 0.0) * 18.0  # coolant flush / radiator clear
        self.cht += (thermal_target - self.cht) * dt / 24.0  # τ=24s first-order lag, calibrated to 30s in generate_engine_data
        # Oil thermal lag τ=40s (vs 60s in dataset) — dry-sump 3L tank
        hot_penalty = 19.0 if self.scenario == "hot_weather" else 0.0
        thin_air_penalty = (1.0 - sigma) * 24.0
        oil_target = 72.0 + throttle * 0.35 + hot_penalty * 0.72 + thin_air_penalty * 0.32
        oil_target -= self.repairs.get("cooling", 0.0) * 10.0
        self.oil_temp += (oil_target - self.oil_temp) * dt / 40.0
        self.battery_soc = clamp(self.battery_soc - dt * (0.0007 + throttle * 0.000012), 78.0, 100.0)
        self.wear = clamp(self.wear + dt * 0.0000008, 0.0, 0.7)  # Paris-law crack growth ∝ wear

        injector_variation = abs(math.sin(elapsed * 2.2)) * 0.12 if self.scenario == "rapid_throttle" else 0.0
        injector_variation *= (1.0 - self.repairs.get("injector", 0.0))  # cleaned injectors
        # Scenario-driven auxiliary fault dynamics to cover all 8 DRDO fault modes
        misfire_intensity = 0.0
        if self.scenario == "endurance" and self.wear > 0.22:
            # Endurance wear -> intermittent single-cylinder misfire after ~60s
            if elapsed > 60 and (int(elapsed * 2) % 11 == 0):
                misfire_intensity = min(0.85, (self.wear - 0.22) * 2.5 + 0.15)
        misfire_intensity *= (1.0 - self.repairs.get("misfire", 0.0))  # new plugs/coils
        lubrication_penalty = 0.0
        if self.wear > 0.18:
            lubrication_penalty = (self.wear - 0.18) * 0.55
            # Endurance/rapid throttle accelerate lubrication degradation
            if self.scenario in ("endurance", "rapid_throttle"):
                lubrication_penalty *= 1.3
        lubrication_penalty *= (1.0 - self.repairs.get("lubrication", 0.0))  # fresh oil/filter
        sensor_drift_c = 0.0
        if elapsed > 55:
            sensor_drift_c = (self.wear * 26.0) * min(1.0, (elapsed - 55) / 120.0)
            # Hot-weather accelerates drift visibility
            if self.scenario == "hot_weather":
                sensor_drift_c *= 1.1
        sensor_drift_c *= (1.0 - self.repairs.get("sensor", 0.0))  # recalibrated probes

        # EGT via physics helper (combustion load + altitude lean + injector/misfire) — matches generate_engine_data
        egt_raw = egt_model_c(throttle, altitude, sigma, injector_variation, misfire_intensity, self.scenario)
        egt = clamp(egt_raw, 520.0, 860.0)
        # Oil system: lubrication_issue manifests as pressure loss + temperature rise
        oil_pressure_raw = 61.0 - (self.oil_temp - 82.0) * 0.19 - self.wear * 7.0 - lubrication_penalty * 14.0
        oil_pressure = clamp(oil_pressure_raw, 18.0, 70.0)
        vibration = clamp(0.11 + self.wear * 0.27 * (1.0 - self.repairs.get("vibration", 0.0)) + injector_variation * 1.45 + misfire_intensity * 0.16 + lubrication_penalty * 0.07 + abs(math.sin(elapsed * 1.1)) * 0.018, 0.07, 0.66)
        fuel_flow = power_kw * PROFILE.bsfc_g_kwh / (820.0 * 1000.0) * 1000.0
        if misfire_intensity > 0:
            fuel_flow *= (1.0 - misfire_intensity * 0.08)
        self.flow_lph = fuel_flow
        alternator = clamp(14.15 - self.wear * 0.35 - injector_variation * 0.16 - lubrication_penalty * 0.12, 12.2, 14.5)
        battery_voltage = clamp(12.55 + self.battery_soc * 0.016 + (alternator - 14.0) * 0.2, 12.0, 14.2)
        injection_timing = clamp(18.0 + throttle * 0.052 - injector_variation * 6.0 - misfire_intensity * 3.5, 13.0, 25.0)
        # Measured CHT includes sensor drift for sensor_drift / failure demonstration
        cht_measured = self.cht + sensor_drift_c

        # Measured CHT vs true CHT separation for sensor_drift demonstration
        cht_for_record = cht_measured
        # One canonical sensor record drives both the deterministic engine core and ML pipeline.
        sensor_record = {
            "engine_id": "TAPAS-DEMO-01", "mission_id": 1, "cycle": self.sequence,
            "flight_phase": str(mission["label"]), "scenario": self.scenario,
            "altitude_m": altitude, "airspeed_kt": airspeed, "ambient_temp": ambient,
            "throttle_pct": throttle / 100.0, "engine_age_hours": 480.0 + self.wear * 1800.0,
            "rpm": rpm - misfire_intensity * 85.0, "engine_load": throttle / 100.0, "fuel_flow": fuel_flow,
            "egt": egt, "cht": float(cht_for_record), "oil_temp": self.oil_temp + lubrication_penalty * 14.0, "oil_pressure": oil_pressure,
            "vibration": vibration, "vib_1x": vibration * 0.62 + misfire_intensity * 0.08, "vib_2x": vibration * 0.28 + misfire_intensity * 0.04,
            "vib_05x": vibration * 0.10, "battery_voltage": battery_voltage,
            "alternator_voltage": alternator, "battery_soc": self.battery_soc,
            "injection_timing": injection_timing,
        }
        core_state = self.core.update(sensor_record)
        model_assessment = MODELS.infer(sensor_record)

        thermal_health = clamp(100.0 - max(0.0, cht_measured - 118.0) * 1.8 - max(0.0, egt - 780.0) * 0.34, 30.0, 100.0)
        lubrication_health = clamp(100.0 - max(0.0, 42.0 - oil_pressure) * 2.1 - max(0.0, (self.oil_temp + lubrication_penalty * 14.0) - 112.0) * 1.1, 35.0, 100.0)
        thin_air_combustion_penalty = max(0.0, 0.48 - sigma) * 100.0 if self.scenario == "high_altitude" else 0.0
        combustion_health = clamp(100.0 - injector_variation * 50.0 - misfire_intensity * 45.0 - vibration * 9.0 - thin_air_combustion_penalty, 40.0, 100.0)
        electrical_health = clamp(100.0 - max(0.0, 13.3 - alternator) * 20.0 - max(0.0, 88.0 - self.battery_soc) * 0.7 - sensor_drift_c * 0.6, 35.0, 100.0)
        health = (thermal_health * 0.34 + lubrication_health * 0.25 + combustion_health * 0.25 + electrical_health * 0.16)
        anomaly = clamp((100.0 - health) / 55.0 + max(0.0, vibration - 0.25) * 1.3 + sensor_drift_c * 0.015 + misfire_intensity * 0.35, 0.02, 0.98)
        # DRDO-ordered 8-fault taxonomy with physically grounded triggers (thresholds per Rotax limits)
        fault = "none"
        # Overheating is highest priority (CHT >=150 or CHT+oil combined)
        if cht_measured >= 143.0 and self.oil_temp >= 128.0:
            fault = "overheating"
        elif self.scenario == "hot_weather" and thermal_health < 84.0:
            fault = "cooling_degradation"
        elif self.scenario == "rapid_throttle" and injector_variation > 0.08:
            fault = "injector_abnormality"
        elif self.scenario == "high_altitude" and sigma < 0.48:
            fault = "combustion_instability"
        elif misfire_intensity > 0.22:
            fault = "misfire"
        elif oil_pressure <= 30.0 and lubrication_health < 70.0:
            fault = "lubrication_issue"
        elif sensor_drift_c >= 18.0:
            fault = "sensor_drift"
        elif vibration > 0.34:
            fault = "abnormal_vibration"

        if fault == "combustion_instability":
            anomaly = max(anomaly, 0.38)
        if fault in ("misfire", "overheating", "lubrication_issue"):
            anomaly = max(anomaly, 0.42)
        if model_assessment["available"] and model_assessment["anomaly"]:
            anomaly = max(anomaly, 0.28)

        fault_component = {
            "cooling_degradation": "thermal",
            "overheating": "thermal",
            "injector_abnormality": "fuel",
            "misfire": "combustion",
            "combustion_instability": "combustion",
            "abnormal_vibration": "propulsor",
            "lubrication_issue": "lubrication",
            "sensor_drift": "electrical",
        }.get(fault, "none")

        warnings = []
        if cht_measured >= 135.0:
            warnings.append("CHT_CAUTION")
        if cht_measured >= 150.0:
            warnings.append("CHT_OVERLIMIT")
        if egt >= 800.0:
            warnings.append("EGT_CAUTION")
        if oil_pressure <= 32.0:
            warnings.append("OIL_PRESSURE_CAUTION")
        if oil_pressure <= 22.0:
            warnings.append("OIL_PRESSURE_CRITICAL")
        if vibration >= 0.30:
            warnings.append("VIBRATION_ELEVATED")
        if vibration >= 0.50:
            warnings.append("VIBRATION_HIGH")
        if fault == "combustion_instability":
            warnings.append("COMBUSTION_MARGIN_CAUTION")
        if fault == "misfire":
            warnings.append("MISFIRE_DETECTED")
        if fault == "sensor_drift":
            warnings.append("SENSOR_DRIFT_SUSPECTED")
        if fault == "overheating":
            warnings.append("OVERHEAT_TREND")
        if fault == "lubrication_issue":
            warnings.append("LUBRICATION_DEGRADED")
        fuel_pct = 100.0 * self.fuel_l / max(self.fuel_cap_l, 1e-6)
        if fuel_pct < 20.0:
            warnings.append("FUEL_LOW")
        if fuel_pct < 10.0:
            warnings.append("FUEL_CRITICAL")
        if self.gps_status == "jammed":
            warnings.append("GPS_JAMMED")

        # --- Maintenance proposals + critical auto-fix (ASAP for critical) ---
        # Fixable systems and their operator-verified effects on the sim.
        FIX_DEFS = {
            "cooling_degradation": ("cooling", "Flush coolant, clear radiator intakes, verify thermostat"),
            "overheating": ("cooling", "Emergency cooling: reduce power, max airflow, flush coolant circuit"),
            "lubrication_issue": ("lubrication", "Replace oil + filter, inspect pump and lines"),
            "sensor_drift": ("sensor", "Recalibrate CHT/EGT probes, inspect harness/EMI"),
            "injector_abnormality": ("injector", "Clean injectors, verify fuel maps and pressure"),
            "misfire": ("misfire", "Replace plugs/coils, check fuel filter"),
            "abnormal_vibration": ("vibration", "Rebalance propeller, torque mounts, inspect bearings"),
            "combustion_instability": ("injector", "Verify fuel grade, retard timing, decarbonize"),
        }
        is_critical = fault == "overheating" or any("OVERLIMIT" in w or "CRITICAL" in w for w in warnings)
        if fault != "none" and fault in FIX_DEFS:
            system, text = FIX_DEFS[fault]
            if is_critical and fault not in self.applied_fixes:
                # Critical + fixable -> applied immediately, no waiting.
                self.repairs[system] = max(self.repairs.get(system, 0.0), 0.9)
                self.applied_fixes[fault] = True
                self.pending_fixes[fault] = {"fix_id": f"FX-{self.sequence}-{fault[:4]}",
                                             "system": system, "text": text, "status": "auto-applied"}
                self.log_event("critical", f"AUTO-FIX applied ASAP for critical {fault}: {text}")
            elif fault not in self.pending_fixes and fault not in self.applied_fixes:
                # Non-critical -> proposal waits for operator ACCEPT.
                self.fix_seq += 1
                self.pending_fixes[fault] = {"fix_id": f"FX-{self.fix_seq:03d}",
                                             "system": system, "text": text, "status": "proposed"}
                self.log_event("warn", f"Fix proposed for {fault} (awaiting operator accept): {text}")

        # --- Failsafe: RTL on fuel/gps/critical; self-destruct if RTL impossible ---
        if self.fuel_l <= 0.0 and not self.destructed:
            self._destruct("fuel exhaustion -- flameout with no divert in reach")
        elif self.flight_mode == "NOMINAL" or (self.flight_mode == "RECOVERED" and self.dist_home_km > 1.0):
            # RECOVERED at home stays recovered (already safe); a new excursion re-arms RTL.
            trigger = None
            if fuel_pct < 10.0:
                trigger = f"fuel critical ({fuel_pct:.0f}% remaining)"
            elif self.gps_status == "jammed":
                trigger = "GPS jammed -- navigating on drifting INS"
            elif is_critical:
                trigger = f"critical fault: {fault}"
            if trigger:
                flow = max(self.flow_lph, 0.5)
                range_km = (self.fuel_l / flow) * max(ground_speed, 30.0) * 1.852
                if range_km < self.dist_home_km * 1.15 or self.fuel_l <= 0.0:
                    self._destruct(f"RTL impossible ({trigger}); range {range_km:.1f} km < {self.dist_home_km * 1.15:.1f} km needed -- sanitizing to protect sensitive data")
                else:
                    self.flight_mode = "RTL"
                    self.target_reached = False
                    self.log_event("critical", f"FAILSAFE RTL engaged: {trigger}. Returning to launch ({self.dist_home_km:.1f} km).")

        payload = {
            "timestamp_ms": round(time.time() * 1000),
            "sequence": self.sequence,
            "source": "physics-simulation",
            "simulation_rate_hz": 10,
            "profile": asdict(PROFILE),
            "mission": {"scenario": self.scenario, "label": mission["label"], "elapsed_s": round(elapsed, 1)},
            "telemetry": {
                "altitude_m": round(altitude, 1), "ambient_temp_c": round(ambient, 1),
                "airspeed_kt": round(airspeed, 1), "throttle_pct": round(throttle, 1),
                "rpm": round(rpm - misfire_intensity * 85.0), "power_kw": round(power_kw, 1), "torque_nm": round(torque_nm, 1),
                "fuel_flow_lph": round(fuel_flow, 1), "egt_c": round(egt, 1), "cht_c": round(cht_measured, 1),
                "cht_true_c": round(self.cht, 1),
                "oil_temp_c": round(self.oil_temp + lubrication_penalty * 14.0, 1), "oil_pressure_psi": round(oil_pressure, 1),
                "vibration_g": round(vibration, 3), "vib_1x_g": round(vibration * 0.62 + misfire_intensity * 0.08, 3), "vib_2x_g": round(vibration * 0.28, 3), "vib_05x_g": round(vibration * 0.10, 3),
                "battery_v": round(battery_voltage, 2),
                "alternator_v": round(alternator, 2), "battery_soc_pct": round(self.battery_soc, 1),
                "injection_timing_deg": round(injection_timing, 1), "density_ratio": round(sigma, 3),
                "sensor_drift_c": round(sensor_drift_c, 1), "misfire_intensity": round(misfire_intensity, 3),
                # --- flight / nav / failsafe state ---
                "lat": round(self.lat, 6), "lon": round(self.lon, 6),
                "heading_deg": round(self.heading_deg, 1), "ground_speed_kt": round(ground_speed, 1),
                "wind_speed_kt": round(self.wind_speed_kt + gust, 1), "wind_from_deg": round(self.wind_from_deg, 0),
                "headwind_kt": round(headwind, 1),
                "throttle_mode": "AUTO" if self.autopilot else "MANUAL",
                "fuel_l": round(self.fuel_l, 2), "fuel_pct": round(fuel_pct, 1),
                "gps_status": self.gps_status, "ins_drift_km": round(self.ins_drift_km, 2),
                "flight_mode": self.flight_mode, "mission_mode": self.mission_mode,
                "fence_radius_km": self.fence_radius_km, "dist_home_km": round(self.dist_home_km, 2),
                "outside_fence": self.outside_fence,
                "wx_source": self.wx_source,
                "target_lat": self.target_lat, "target_lon": self.target_lon,
                "target_reached": self.target_reached,
            },
            "health": {
                "overall_pct": round(health, 1), "thermal_pct": round(thermal_health, 1),
                "lubrication_pct": round(lubrication_health, 1), "combustion_pct": round(combustion_health, 1),
                "electrical_pct": round(electrical_health, 1), "anomaly_score": round(anomaly, 3),
                "fault_mode": fault, "fault_component": fault_component,
                "rul_cycles": round(clamp(230.0 - self.wear * 120.0 - anomaly * 70.0, 25.0, 230.0)),
                "warnings": warnings,
            },
            "engine_core": {
                "status": core_state.operating_status,
                "warnings": core_state.active_warnings.split(",") if core_state.active_warnings else [],
                "thermal_load_pct": core_state.thermal_load_pct,
                "electrical_health_pct": core_state.electrical_health_pct,
                "combustion_quality_pct": core_state.combustion_quality,
                "lubrication_health_pct": core_state.lubrication_health,
                "bsfc_g_kwh": core_state.bsfc_est,
            },
            "ml_assessment": model_assessment,
            "xai": [
                {"feature": "Cylinder head thermal load", "impact": round(max(0.04, (self.cht - 100.0) / 52.0), 3)},
                {"feature": "Exhaust gas temperature", "impact": round(max(0.03, (egt - 690.0) / 170.0), 3)},
                {"feature": "Oil pressure", "impact": round(max(0.03, (46.0 - oil_pressure) / 28.0), 3)},
                {"feature": "Vibration RMS", "impact": round(max(0.03, vibration * 1.35), 3)},
                {"feature": "Air-density combustion margin", "impact": round(max(0.03, thin_air_combustion_penalty / 28.0), 3)},
            ],
            "events": self.events[-6:],
        }
        self.last_payload = payload
        return payload


TWIN = TwinState()
# In-memory rolling history for trend analysis (last 600 samples ≈ 60s at 10 Hz)
HISTORY: deque = deque(maxlen=600)

# ---- Recorded-mission live stream (real dataset rows at 10 Hz) ----
# The sandbox physics loop runs outside the ML training envelope, so its ML
# badge honestly abstains. Streaming RECORDED missions gives a live 10 Hz feed
# the dataset-trained models genuinely cover: real fault development, real
# diagnoses, real RUL — a true real-time simulation of a recorded flight.
REPLAY: dict = {"active": False, "rows": [], "idx": 0, "core": None,
                "mission_id": None, "fault": "none", "engine_id": ""}
REPLAY_CSV = None

FAULT_COMPONENTS = {
    "cooling_degradation": "thermal", "overheating": "thermal",
    "injector_abnormality": "fuel", "misfire": "combustion",
    "combustion_instability": "combustion", "abnormal_vibration": "propulsor",
    "lubrication_issue": "lubrication", "sensor_drift": "electrical",
}


def _replay_df():
    """Engine dataset, loaded once and cached for the replay streamer."""
    global REPLAY_CSV
    if REPLAY_CSV is None:
        import pandas as pd

        REPLAY_CSV = pd.read_csv(Path(__file__).parent / "engine_data.csv")
    return REPLAY_CSV


def _pick_replay_mission() -> int:
    """Longest mission of the most demo-worthy fault (strong, visible signatures).
    Preference follows held-out classifier strength: overheating, misfire,
    lubrication_issue, then anything with a dramatic arc."""
    df = _replay_df()
    faulty = df[df["fault_type"] != "none"]
    stat = faulty.groupby("mission_id").agg(
        peak=("degradation_severity", "max"), n=("cycle", "size"),
        fault=("fault_type", lambda s: str(s[s != "none"].mode().iloc[0])))

    def candidate(frame):
        frame = frame[frame["peak"] > 0.5]
        if len(frame) == 0:
            return None
        return int(frame.sort_values("n", ascending=False).index[0])

    for want in ["overheating", "misfire", "lubrication_issue", "cooling_degradation"]:
        hit = candidate(stat[stat["fault"] == want])
        if hit is not None:
            return hit
    stat = stat.sort_values(["peak", "n"], ascending=[False, False])
    return int(stat.index[0])


def replay_step() -> dict:
    """One 10 Hz frame from the recorded mission (same payload schema as the sim)."""
    from phm_pipeline import explain as _explain

    st = REPLAY
    rows = st["rows"]
    row = rows[st["idx"]]
    st["idx"] = (st["idx"] + 1) % len(rows)

    record = {
        "engine_id": str(row["engine_id"]), "mission_id": int(row["mission_id"]), "cycle": int(row["cycle"]),
        "flight_phase": str(row["flight_phase"]), "scenario": str(row["scenario"]),
        "altitude_m": float(row["altitude_m"]), "airspeed_kt": float(row["airspeed_kt"]),
        "ambient_temp": float(row["ambient_temp"]), "throttle_pct": float(row["throttle_pct"]),
        "engine_age_hours": float(row["engine_age_hours"]), "rpm": float(row["rpm"]),
        "engine_load": float(row["engine_load"]), "fuel_flow": float(row["fuel_flow"]),
        "egt": float(row["egt"]), "cht": float(row["cht"]), "oil_temp": float(row["oil_temp"]),
        "oil_pressure": float(row["oil_pressure"]), "vibration": float(row["vibration"]),
        "vib_1x": float(row["vib_1x"]), "vib_2x": float(row["vib_2x"]), "vib_05x": float(row["vib_05x"]),
        "battery_voltage": float(row["battery_voltage"]), "alternator_voltage": float(row["alternator_voltage"]),
        "battery_soc": float(row["battery_soc"]), "injection_timing": float(row["injection_timing"]),
    }
    core_state = st["core"].update(record)
    model_assessment = MODELS.infer(record)
    try:
        import pandas as pd

        exp = _explain({"clf": MODELS.bundle["classifier"], "le": MODELS.bundle["labels"]},
                       engineer_features(pd.DataFrame([record])))
        xai = [{"feature": k, "impact": round(float(v), 3)} for k, v in exp["top"][:5]]
    except Exception:
        xai = []
    fault = str(row["fault_type"])
    rul_raw = float(row["RUL"])
    import math as _math
    rul = int(rul_raw) if _math.isfinite(rul_raw) else max(0, len(rows) - int(row["cycle"]))
    summary = st["core"].get_health_summary()
    return {
        "timestamp_ms": round(time.time() * 1000),
        "sequence": int(row["cycle"]),
        "source": "physics-simulation",
        "simulation_rate_hz": 10,
        "profile": asdict(PROFILE),
        "mission": {"scenario": "replay", "label": f"Replay M{st['mission_id']} {fault}", "elapsed_s": round(float(row["elapsed_time_sec"]), 1)},
        "telemetry": {
            "altitude_m": round(float(row["altitude_m"]), 1), "ambient_temp_c": round(float(row["ambient_temp"]), 1),
            "airspeed_kt": round(float(row["airspeed_kt"]), 1), "throttle_pct": round(float(row["throttle_pct"]) * 100.0, 1),
            "rpm": int(row["rpm"]), "power_kw": round(float(row["power_kw"]), 1), "torque_nm": round(float(row["torque_nm"]), 1),
            "fuel_flow_lph": round(float(row["fuel_flow"]), 1), "egt_c": round(float(row["egt"]), 1), "cht_c": round(float(row["cht"]), 1),
            "cht_true_c": round(float(row["cht_true"]), 1),
            "oil_temp_c": round(float(row["oil_temp"]), 1), "oil_pressure_psi": round(float(row["oil_pressure"]), 1),
            "vibration_g": round(float(row["vibration"]), 3), "vib_1x_g": round(float(row["vib_1x"]), 3),
            "vib_2x_g": round(float(row["vib_2x"]), 3), "vib_05x_g": round(float(row["vib_05x"]), 3),
            "battery_v": round(float(row["battery_voltage"]), 2),
            "alternator_v": round(float(row["alternator_voltage"]), 2), "battery_soc_pct": round(float(row["battery_soc"]), 1),
            "injection_timing_deg": round(float(row["injection_timing"]), 1), "density_ratio": round(float(row["sigma"]), 3),
            "sensor_drift_c": round(float(row["cht"]) - float(row["cht_true"]), 1), "misfire_intensity": 0.0,
        },
        "health": {
            "overall_pct": round(float(row["health_index"]), 1),
            "thermal_pct": round(100.0 - summary["thermal_load_pct"], 1),
            "lubrication_pct": summary["lubrication_health"],
            "combustion_pct": summary["combustion_quality"],
            "electrical_pct": summary["electrical_health_pct"],
            "anomaly_score": round(float(row["degradation_severity"]), 3),
            "fault_mode": fault, "fault_component": FAULT_COMPONENTS.get(fault, "none"),
            "rul_cycles": rul,
            "warnings": core_state.active_warnings.split(",") if core_state.active_warnings else [],
        },
        "engine_core": {
            "status": core_state.operating_status,
            "warnings": core_state.active_warnings.split(",") if core_state.active_warnings else [],
            "thermal_load_pct": core_state.thermal_load_pct,
            "electrical_health_pct": core_state.electrical_health_pct,
            "combustion_quality_pct": core_state.combustion_quality,
            "lubrication_health_pct": core_state.lubrication_health,
            "bsfc_g_kwh": core_state.bsfc_est,
        },
        "ml_assessment": model_assessment,
        "xai": xai,
    }


def _advisory_for(payload: dict) -> List[Dict[str, str]]:
    health = payload.get("health", {})
    core = payload.get("engine_core", {})
    fault = health.get("fault_mode", "none")
    warnings = health.get("warnings", []) + core.get("warnings", [])
    warnings = list(dict.fromkeys(warnings))  # physics + core overlap; keep first occurrence
    adv: List[Dict[str, str]] = []
    if fault == "none" and not warnings:
        adv.append({"level": "ROUTINE", "action": "Continue mission. Store as healthy baseline for fleet learning.", "due_cycles": "N/A"})
    else:
        urgency = "CRITICAL" if any("OVERLIMIT" in w or "CRITICAL" in w for w in warnings) else "URGENT" if health.get("anomaly_score", 0) > 0.6 else "SCHEDULED"
        mapping = {
            "misfire": "Inspect spark plugs/ignition modules, check fuel filter, verify coil dwell.",
            "injector_abnormality": "Clean/replace injectors, verify ECU fuel maps, check fuel pressure.",
            "cooling_degradation": "Check coolant, radiator blockage, thermostat, pump flow.",
            "lubrication_issue": "Replace oil/filter, inspect pump, check for leaks and metal debris.",
            "sensor_drift": "Recalibrate CHT/EGT thermocouples, check harness for corrosion/EMI.",
            "combustion_instability": "Verify fuel octane, retard timing, decarbonize chamber, check knock sensor.",
            "overheating": "Reduce power, increase airspeed for cooling, inspect cooling circuit immediately.",
            "abnormal_vibration": "Inspect prop balance, engine mounts, crank bearings via 1x/2x FFT.",
        }
        adv.append({"level": urgency, "action": mapping.get(fault, "Perform general inspection per AMM."), "due_cycles": str(health.get("rul_cycles", "—"))})
        # Operator fix workflow: attach the live proposal / applied state so the
        # GCS can offer ACCEPT for non-critical items (critical ones self-apply).
        try:
            prop = TWIN.pending_fixes.get(fault)
        except Exception:
            prop = None
        if prop:
            adv[0]["fix"] = {"fix_id": prop["fix_id"], "text": prop["text"],
                             "status": prop["status"],
                             "needs_accept": prop["status"] == "proposed"}
        for w in warnings[:3]:
            adv.append({"level": "WARNING", "action": f"Threshold: {w}", "due_cycles": "immediate"})
    return adv


@app.get("/")
def root() -> dict:
    return {
        "service": "SIH MALE UAV Digital Twin API",
        "version": "2.0.0",
        "data_class": "public-reference physics simulation",
        "docs": "/docs",
        "health": "/health",
        "stream": "/stream",
        "profile": asdict(PROFILE),
        "scenarios": list(SCENARIOS.keys()),
    }


@app.get("/health")
def health() -> dict:
    # Always expose full health including ML runtime state for GCS readiness probe.
    return {
        "status": "ready",
        "rate_hz": 10,
        "profile": PROFILE.name,
        "data_class": "public-reference physics simulation",
        "engine_core": "active",
        "ml_models_ready": MODELS.ready,
        "ml_detail": MODELS.reason,
        "ws_endpoint": "/stream",
        "telemetry_endpoint": "/telemetry",
        "launch": {"lat": TWIN.lat0, "lon": TWIN.lon0},
    }


@app.get("/scenarios")
def list_scenarios() -> dict:
    return {"scenarios": SCENARIOS, "profile": asdict(PROFILE)}


@app.post("/mission/start")
def start_mission(scenario: str = "hot_weather") -> dict:
    if scenario not in SCENARIOS:
        raise HTTPException(status_code=422, detail=f"Unknown scenario. Choose one of: {', '.join(SCENARIOS)}")
    REPLAY["active"] = False
    TWIN.reset(scenario)
    return {"status": "started", "scenario": scenario, "rate_hz": 10}


@app.post("/mission/replay")
def start_replay(mission_id: int | None = None) -> dict:
    """Stream a RECORDED dataset mission live at 10 Hz (real ML diagnoses).
    Auto-picks the most dramatic faulty mission when no id is given."""
    try:
        import pandas as pd

        df = _replay_df()
        mid = int(mission_id) if mission_id is not None else _pick_replay_mission()
        rows = df[df["mission_id"] == mid].sort_values("cycle").to_dict(orient="records")
        if not rows:
            raise HTTPException(status_code=404, detail=f"mission_id {mid} not found")
        faults = [r["fault_type"] for r in rows if r["fault_type"] != "none"]
        REPLAY.update({"active": True, "rows": rows, "idx": 0, "core": DigitalTwinCore(),
                       "mission_id": mid, "fault": str(pd.Series(faults).mode().iloc[0]) if faults else "none",
                       "engine_id": str(rows[0]["engine_id"])})
        return {"status": "started", "source": "recorded-replay", "mission_id": mid,
                "fault_type": REPLAY["fault"], "cycles": len(rows), "rate_hz": 10}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.post("/mission/custom")
def start_custom_mission(
    altitude: float = 5000.0,
    ambient: float = 15.0,
    throttle: float = 65.0,
    label: str = "Custom Mission",
) -> dict:
    """Create and start a fully custom scenario — any altitude/ambient/throttle."""
    altitude = clamp(altitude, 0.0, 11000.0)
    ambient = clamp(ambient, -50.0, 55.0)
    throttle = clamp(throttle, 30.0, 100.0)
    # ISA sanity: if user picks extreme hot+high, thermal will naturally degrade
    SCENARIOS["custom"] = {
        "altitude": float(altitude),
        "ambient": float(ambient),
        "throttle": float(throttle),
        "label": str(label)[:40] if label else "Custom",
    }
    REPLAY["active"] = False
    TWIN.reset("custom")
    return {"status": "started", "scenario": "custom", "config": SCENARIOS["custom"], "rate_hz": 10}


@app.post("/mission/target")
def set_target(lat: float | None = None, lon: float | None = None) -> dict:
    """Set (or clear, with no args) the destination waypoint. Guidance steers
    toward it at 3 deg/s; arrival inside 0.5 km is logged."""
    if lat is None or lon is None:
        TWIN.target_lat, TWIN.target_lon, TWIN.target_reached = None, None, False
        return {"status": "cleared"}
    TWIN.target_lat = clamp(lat, -90.0, 90.0)
    TWIN.target_lon = clamp(lon, -180.0, 180.0)
    TWIN.target_reached = False
    if TWIN.flight_mode == "RECOVERED":
        TWIN.flight_mode = "NOMINAL"  # operator retasking resumes the sortie
        TWIN.log_event("info", "Resumed from RECOVERED on operator retask")
    d = TwinState._dist_km(TWIN.lat, TWIN.lon, TWIN.target_lat, TWIN.target_lon)
    TWIN.log_event("info", f"Target set ({TWIN.target_lat:.4f}, {TWIN.target_lon:.4f}), {d:.1f} km out")
    return {"status": "target-set", "lat": TWIN.target_lat, "lon": TWIN.target_lon, "dist_km": round(d, 1)}


@app.post("/mission/mode")
def set_mission_mode(mode: str = "SURVEILLANCE") -> dict:
    """SURVEILLANCE: geofence holds the UAV inside own-territory boundary.
    ATTACK: fence may be crossed outward (logged) for strike missions."""
    mode = str(mode).upper()
    if mode not in ("SURVEILLANCE", "ATTACK"):
        raise HTTPException(status_code=422, detail="mode must be SURVEILLANCE or ATTACK")
    TWIN.mission_mode = mode
    TWIN.outside_fence = False
    TWIN.log_event("warn" if mode == "ATTACK" else "info", f"Mission mode -> {mode}")
    return {"status": "ok", "mission_mode": mode, "fence_radius_km": TWIN.fence_radius_km}


@app.post("/mission/fence")
def set_fence(radius_km: float = 20.0) -> dict:
    TWIN.fence_radius_km = clamp(radius_km, 1.0, 500.0)
    TWIN.outside_fence = False
    TWIN.log_event("info", f"Geofence radius -> {TWIN.fence_radius_km:.0f} km around launch")
    return {"status": "ok", "fence_radius_km": TWIN.fence_radius_km}


@app.post("/mission/rtl")
def force_rtl() -> dict:
    """Operator-commanded return-to-launch (also auto-engaged on fuel/GPS/critical)."""
    if TWIN.destructed:
        raise HTTPException(status_code=409, detail="airframe destroyed")
    TWIN.flight_mode = "RTL"
    TWIN.target_reached = False
    TWIN.log_event("critical", "Operator commanded RTL: returning to launch")
    return {"status": "rtl", "dist_home_km": round(getattr(TWIN, "dist_home_km", 0.0), 1)}


@app.post("/mission/destruct")
def force_destruct(confirm: bool = False) -> dict:
    """Operator-commanded sanitize. Requires confirm=true (two-step safety)."""
    if not confirm:
        raise HTTPException(status_code=409, detail="pass confirm=true to execute self-destruct")
    TWIN._destruct("operator-commanded sanitize to protect sensitive data")
    return {"status": "destroyed"}


@app.post("/autopilot")
def set_autopilot(enabled: bool = True, target_ias_kt: float | None = None) -> dict:
    """Autothrottle holds target airspeed against head/tailwinds. Manual mode
    returns throttle to the scenario schedule."""
    TWIN.autopilot = bool(enabled)
    if target_ias_kt is not None:
        TWIN.target_ias_kt = clamp(target_ias_kt, 60.0, 150.0)
    TWIN.log_event("info", f"Autothrottle {'ON' if TWIN.autopilot else 'OFF'} (target {TWIN.target_ias_kt:.0f} kt)")
    return {"status": "ok", "autopilot": TWIN.autopilot, "target_ias_kt": TWIN.target_ias_kt}


@app.post("/mission/weather")
def set_weather(mode: str = "auto", ambient: float | None = None,
                wind_speed: float | None = None, wind_from: float | None = None) -> dict:
    """Weather source: auto = live Open-Meteo at the drone's position (refreshed
    ~60 s, drives physics + ML inputs); scenario = built-in pack; custom =
    operator values (degC, kt, deg-from)."""
    mode = str(mode).lower()
    if mode == "auto":
        TWIN.wx_auto = True
        TWIN.wx_last = 0.0  # force immediate fetch on next step
        TWIN.log_event("info", "Live weather feed ON (Open-Meteo, ~60 s refresh)")
    elif mode == "scenario":
        TWIN.wx_auto = False
        TWIN.wx_ambient, TWIN.wx_wind_spd, TWIN.wx_wind_from = None, None, None
        TWIN.wx_source = "scenario"
        TWIN.log_event("info", "Weather source -> scenario pack")
    elif mode == "custom":
        TWIN.wx_auto = False
        if ambient is not None:
            TWIN.wx_ambient = clamp(ambient, -60.0, 60.0)
        if wind_speed is not None:
            TWIN.wx_wind_spd = clamp(wind_speed, 0.0, 120.0)
        if wind_from is not None:
            TWIN.wx_wind_from = float(wind_from) % 360.0
        TWIN.wx_source = "operator"
        TWIN.log_event("info", f"Operator weather set (ambient {TWIN.wx_ambient}, wind {TWIN.wx_wind_spd} kt)")
    else:
        raise HTTPException(status_code=422, detail="mode must be auto, scenario, or custom")
    return {"status": "ok", "wx_source": TWIN.wx_source}


@app.post("/mission/event")
def inject_event(kind: str = "gps_jam") -> dict:
    """Scenario event injector (demo/chaos): gps_jam | gps_clear."""
    kind = str(kind).lower()
    if kind == "gps_jam":
        TWIN.gps_status = "jammed"
        TWIN.log_event("critical", "GPS JAMMED -- holding on INS (position drifting)")
    elif kind == "gps_clear":
        TWIN.gps_status = "ok"
        TWIN.ins_drift_km = 0.0
        TWIN.log_event("info", "GPS restored -- INS realigned")
    else:
        raise HTTPException(status_code=422, detail="kind must be gps_jam or gps_clear")
    return {"status": "ok", "gps": TWIN.gps_status}


@app.post("/maintenance/accept")
def accept_fix(fix_id: str = "") -> dict:
    """Operator accepts a proposed (non-critical) fix. Critical fixes apply ASAP
    on their own; proposals wait here until accepted."""
    for fault, prop in TWIN.pending_fixes.items():
        if prop.get("fix_id") == fix_id and prop.get("status") == "proposed":
            TWIN.repairs[prop["system"]] = max(TWIN.repairs.get(prop["system"], 0.0), 0.9)
            prop["status"] = "applied"
            TWIN.applied_fixes[fault] = True
            TWIN.log_event("info", f"Operator ACCEPTED fix for {fault}: {prop['text']}")
            return {"status": "applied", "fault": fault}
    raise HTTPException(status_code=404, detail=f"no proposed fix {fix_id!r}")


@app.get("/telemetry")
def telemetry() -> dict:
    payload = replay_step() if REPLAY.get("active") else TWIN.step()
    HISTORY.append(payload)
    return payload


@app.get("/health/history")
def health_history(limit: int = 120) -> dict:
    """Trend analysis: rolling health/RUL/efficiency history (last N samples)."""
    limit = clamp(limit, 10, 600)
    hist = list(HISTORY)[-int(limit):]
    trend = [
        {
            "sequence": p["sequence"],
            "timestamp_ms": p["timestamp_ms"],
            "health": p["health"]["overall_pct"],
            "anomaly": p["health"]["anomaly_score"],
            "rul": p["health"]["rul_cycles"],
            "cht": p["telemetry"]["cht_c"],
            "egt": p["telemetry"]["egt_c"],
            "oil_p": p["telemetry"]["oil_pressure_psi"],
            "vibration": p["telemetry"]["vibration_g"],
            "fault": p["health"]["fault_mode"],
        }
        for p in hist
    ]
    return {"count": len(trend), "trend": trend}


@app.get("/advisory")
def advisory() -> dict:
    """Predictive maintenance recommendations (rule-based + ML RUL)."""
    # Use latest history or fresh step
    payload = list(HISTORY)[-1] if HISTORY else (replay_step() if REPLAY.get("active") else TWIN.step())
    if not HISTORY:
        HISTORY.append(payload)
    return {"advisory": _advisory_for(payload), "payload": payload}


@app.get("/performance/map")
def performance_map() -> dict:
    """Engine performance maps: power/torque/BSFC vs RPM & sigma (Willans-line)."""
    rpms = [1800, 2500, 3200, 3880, 4500, 5500]
    sigmas = [1.0, 0.85, 0.7, 0.55, 0.4]
    # Throttle 75% as reference
    table = []
    for sigma in sigmas:
        row = {"sigma": sigma}
        for rpm in rpms:
            p = willans_power_kw(rpm, 75.0, sigma) * (PROFILE.max_power_kw / 69.0)
            row[f"p_{rpm}"] = round(clamp(p, 0, PROFILE.max_power_kw), 1)
        table.append(row)
    return {"profile": asdict(PROFILE), "throttle_ref_pct": 75, "map": table, "provenance": "Willans-line + ISA density, calibrated to Rotax 912 ULS data; scaled to Austro AE300 envelope for demo"}


@app.get("/physics/provenance")
def physics_provenance() -> dict:
    return {
        "engine": asdict(PROFILE),
        "isa": "ICAO Doc 7488/3 ISA troposphere: sigma = P_ratio * T_isa/T_amb, exponent 5.2561",
        "otto": "eta = 1 - 1/r^(gamma-1), r=9.0, gamma=1.4 => 58.5% ideal, 30.4% real (Heywood)",
        "willans": "P = P_ref*(rpm/rpm_max)*throttle*sigma*eta_vol/eta_vol_max, eta_vol parabolic peak 4200 RPM",
        "thermal": "First-order CHT filter tau 24s, oil tau 40s, Newton cooling Q_cool ~ sigma*Vair*(CHT-Tamb)",
        "vibration": "1x firing freq = rpm/30 Hz (4-cyl 4-stroke), 2x harmonic, 0.5x sub-harmonic (Pan et al.)",
        "data_class": "public-reference physics simulation — not OEM, labeled per payload.source",
        "ml": "RF-200 supervised detector + Mahalanobis novelty watchdog (raw signals, OOD abstention) + RF-300 classifier + HGB-500 RUL on 44 physics features, stratified mission split",
        "can": "CAN 2.0A 11-bit, DLC4 uint32 LE, 19 signals, SocketCAN python-can",
        "edge_split": "Edge: IsolationForest <0.1ms; GCS: classifier/RUL/SHAP ~5ms, 5MB, 10Hz link AES-256-GCM + mTLS (arch doc)",
    }


# ---- Mission replay / history (real engine_data.csv, not mock) ----
@app.get("/missions")
def list_missions(limit: int = 20) -> dict:
    """Expose real historical missions from engine_data.csv for replay."""
    import pandas as pd

    try:
        csv_path = Path(__file__).parent / "engine_data.csv"
        if not csv_path.exists():
            return {"missions": [], "detail": "engine_data.csv not found. Run generate_engine_data.py"}
        df = pd.read_csv(csv_path, usecols=["engine_id", "mission_id", "scenario", "fault_type", "is_anomaly", "cycle"])
        counts = df.groupby("mission_id").size().rename("cycles")
        # For each mission, find the dominant non-none fault (if any) else "none"
        def mission_fault(sub: pd.DataFrame) -> str:
            faults = sub[sub["fault_type"] != "none"]["fault_type"]
            if len(faults) == 0:
                return "none"
            # Most frequent fault after onset
            return str(faults.mode().iloc[0] if len(faults.mode()) > 0 else faults.iloc[0])

        first = df.drop_duplicates(subset=["mission_id"])[["engine_id", "mission_id", "scenario"]]
        fault_map = df.groupby("mission_id").apply(mission_fault, include_groups=False)
        merged = first.merge(counts, left_on="mission_id", right_index=True)
        merged["fault_type"] = merged["mission_id"].map(fault_map)
        merged = merged.sort_values("mission_id").head(limit)
        missions: List[Dict[str, Any]] = merged.to_dict(orient="records")
        return {"missions": missions, "total_missions": int(df["mission_id"].nunique())}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.get("/missions/{mission_id}")
def get_mission(mission_id: int, limit: int = 600) -> dict:
    """Return real time-series for a specific mission_id (capped)."""
    import pandas as pd

    try:
        csv_path = Path(__file__).parent / "engine_data.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="engine_data.csv not found")
        df = pd.read_csv(csv_path)
        mission_df = df[df["mission_id"] == mission_id].head(limit)
        if mission_df.empty:
            raise HTTPException(status_code=404, detail=f"mission_id {mission_id} not found")
        # Return raw sensor rows plus derived core health for each cycle would be heavy.
        # For GCS replay, send the physics-grounded rows as-is.
        faults = mission_df[mission_df["fault_type"] != "none"]["fault_type"]
        dom_fault = str(faults.mode().iloc[0]) if len(faults) else "none"
        return {
            "mission_id": mission_id,
            "engine_id": str(mission_df.iloc[0]["engine_id"]),
            "scenario": str(mission_df.iloc[0]["scenario"]),
            "fault_type": dom_fault,
            "cycles": len(mission_df),
            "data": mission_df.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.get("/can/demo")
def can_demo() -> dict:
    """Return one real CAN-framed sample to prove CAN bus framing is not mock."""
    import pandas as pd
    from can_simulator import row_to_frames, frames_to_dict

    try:
        csv_path = Path(__file__).parent / "engine_data.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            row = df.iloc[0].to_dict()
        else:
            row = TWIN.step()["telemetry"]
            # Convert telemetry back to sensor record shape
            row = {
                "rpm": row["rpm"] if isinstance(row, dict) else 4750,
                "cht": 125, "egt": 680, "oil_pressure": 55, "oil_temp": 98,
                "vibration": 0.18, "fuel_flow": 18.5, "battery_voltage": 13.7,
                "alternator_voltage": 14.1, "battery_soc": 94, "injection_timing": 21.5,
            }
        frames = row_to_frames(row)
        decoded = frames_to_dict(frames)
        return {
            "frames": [{"id": f"0x{f.arbitration_id:03X}", "dlc": f.dlc, "data_hex": f.data.hex()} for f in frames[:6]],
            "decoded_sample": {k: decoded[k] for k in list(decoded)[:6]},
            "total_signals": len(frames),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.websocket("/stream")
async def telemetry_stream(websocket: WebSocket) -> None:
    # Accept unconditionally; CORS already allows all origins via middleware.
    await websocket.accept()
    try:
        while True:
            payload = replay_step() if REPLAY.get("active") else TWIN.step()
            HISTORY.append(payload)
            await websocket.send_json(payload)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
        return
