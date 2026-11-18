import math
from dataclasses import dataclass

IDLE, SEAM_FIND, WELD, FAULT, DONE = 0, 1, 2, 3, 4


@dataclass
class PlcConfig:
    watchdog_s: float = 0.2
    fault_s: float = 1.0
    max_offset: float = 0.008
    slew: float = 0.02
    travel_speed: float = 0.01
    stale_speed_scale: float = 0.5
    x_end: float = 0.38
    y_nom: float = 0.0


class MockPLC:
    """In-process stand-in for a TwinCAT PLC; same read/write surface as pyads.Connection."""

    def __init__(self, cfg: PlcConfig | None = None):
        self.cfg = cfg or PlcConfig()
        self.t = 0.0
        self._last_hb = 0
        self._last_hb_t = 0.0
        self._have_ml = False
        self.sym = {
            "MAIN.eState": IDLE,
            "MAIN.bStart": False,
            "MAIN.bUseMl": True,
            "MAIN.bMlValid": False,
            "MAIN.nMlHeartbeat": 0,
            "MAIN.fMlSeamY": 0.0,
            "MAIN.fTorchX_Act": 0.0,
            "MAIN.fTorchY_Act": 0.0,
            "MAIN.fTorchX_Cmd": 0.0,
            "MAIN.fTorchY_Cmd": self.cfg.y_nom,
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def open(self):
        pass

    def close(self):
        pass

    def read_by_name(self, name, plc_datatype=None):
        return self.sym[name]

    def write_by_name(self, name, value, plc_datatype=None):
        if name not in self.sym:
            raise KeyError(name)
        self.sym[name] = value

    def tick(self, dt):
        s, c = self.sym, self.cfg
        self.t += dt
        hb = s["MAIN.nMlHeartbeat"]
        y_est = s["MAIN.fMlSeamY"]
        if hb != self._last_hb and math.isfinite(y_est):
            self._last_hb, self._last_hb_t, self._have_ml = hb, self.t, True
        fresh = self._have_ml and (self.t - self._last_hb_t) <= c.watchdog_s
        s["MAIN.bMlValid"] = fresh
        state = s["MAIN.eState"]

        if state == IDLE and s["MAIN.bStart"]:
            s["MAIN.bStart"] = False
            self._have_ml, self._last_hb_t = False, self.t
            state = SEAM_FIND if s["MAIN.bUseMl"] else WELD

        elif state == SEAM_FIND:
            if fresh:
                state = WELD
            elif self.t - self._last_hb_t > c.fault_s:
                state = FAULT

        elif state == WELD:
            speed = c.travel_speed
            if s["MAIN.bUseMl"]:
                if fresh:
                    target = min(max(y_est, c.y_nom - c.max_offset), c.y_nom + c.max_offset)
                else:
                    target = s["MAIN.fTorchY_Cmd"]
                    speed *= c.stale_speed_scale
                    if self.t - self._last_hb_t > c.fault_s:
                        state = FAULT
            else:
                target = c.y_nom
            step = min(max(target - s["MAIN.fTorchY_Cmd"], -c.slew * dt), c.slew * dt)
            s["MAIN.fTorchY_Cmd"] += step
            if state == WELD:
                s["MAIN.fTorchX_Cmd"] += speed * dt
                if s["MAIN.fTorchX_Cmd"] >= c.x_end:
                    state = DONE

        s["MAIN.eState"] = state


class AdsPLC:
    """Thin pyads wrapper for a real TwinCAT runtime; tick is a no-op (PLC runs in real time)."""

    def __init__(self, ams_net_id: str, port: int = 851, ip: str | None = None):
        import pyads

        self._c = pyads.Connection(ams_net_id, port, ip)

    def __enter__(self):
        self._c.open()
        return self

    def __exit__(self, *exc):
        self._c.close()
        return False

    def read_by_name(self, name, plc_datatype=None):
        return self._c.read_by_name(name, plc_datatype)

    def write_by_name(self, name, value, plc_datatype=None):
        return self._c.write_by_name(name, value, plc_datatype)

    def tick(self, dt):
        pass
