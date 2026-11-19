from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import plc as P
from .estimator import estimate_seam_offset
from .scene import WeldSim, random_coupon

CYCLE = 0.01


@dataclass
class EpisodeResult:
    rms_mm: float
    max_mm: float
    final_state: int
    x_final: float
    max_y_cmd_dev_mm: float


def run_episode(
    seed: int,
    plc=None,
    use_ml: bool = True,
    ml_rate_hz: float = 30.0,
    ml_latency_s: float = 0.05,
    stall_after_s: float | None = None,
    outlier_at_s: float | None = None,
    max_time_s: float = 60.0,
    trace: Callable[[dict], None] | None = None,
) -> EpisodeResult:
    plc = plc or P.MockPLC()
    sim = WeldSim(random_coupon(seed))
    rng = np.random.default_rng(seed + 1000)
    plc.write_by_name("MAIN.bUseMl", use_ml)
    plc.write_by_name("MAIN.bStart", True)

    t, next_ml, pending, hb, errs, dev = 0.0, 0.0, [], 0, [], 0.0
    while t < max_time_s:
        x, y = sim.torch_xy()
        plc.write_by_name("MAIN.fTorchX_Act", x)
        plc.write_by_name("MAIN.fTorchY_Act", y)

        if t >= next_ml:
            next_ml = t + 1.0 / ml_rate_hz
            yl, z = sim.laser_profile(rng)
            pending.append((t + ml_latency_s, y, yl, z))
        while pending and pending[0][0] <= t:
            _, y_at, yl, z = pending.pop(0)
            if stall_after_s is not None and t >= stall_after_s:
                continue
            off = estimate_seam_offset(yl, z)
            if off is None:
                continue
            est = y_at + off
            if outlier_at_s is not None and t >= outlier_at_s:
                est += 0.05
            hb += 1
            plc.write_by_name("MAIN.fMlSeamY", est)
            plc.write_by_name("MAIN.nMlHeartbeat", hb)

        plc.tick(CYCLE)
        state = plc.read_by_name("MAIN.eState")
        x_cmd = plc.read_by_name("MAIN.fTorchX_Cmd")
        y_cmd = plc.read_by_name("MAIN.fTorchY_Cmd")
        dev = max(dev, abs(y_cmd - plc.cfg.y_nom))
        sim.step(x_cmd, y_cmd)
        if trace is not None:
            xa, ya = sim.torch_xy()
            trace({
                "time_s": round(t, 4),
                "plc_state": ("Idle", "Seam find", "Weld", "Fault", "Done")[state],
                "torch_x_m": float(xa), "torch_y_m": float(ya),
                "seam_offset_mm": float((ya - sim.coupon.seam_y(xa)) * 1e3),
                "ml_valid": bool(plc.read_by_name("MAIN.bMlValid")),
            })
        if state == P.WELD:
            xa, ya = sim.torch_xy()
            errs.append(ya - sim.coupon.seam_y(xa))
        if state in (P.DONE, P.FAULT):
            break
        t += CYCLE

    e = np.array(errs) if errs else np.array([np.nan])
    return EpisodeResult(
        rms_mm=float(np.sqrt(np.mean(e**2)) * 1e3),
        max_mm=float(np.max(np.abs(e)) * 1e3),
        final_state=int(plc.read_by_name("MAIN.eState")),
        x_final=float(plc.read_by_name("MAIN.fTorchX_Cmd")),
        max_y_cmd_dev_mm=dev * 1e3,
    )
