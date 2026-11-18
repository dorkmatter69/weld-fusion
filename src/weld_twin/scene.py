from dataclasses import dataclass

import mujoco
import numpy as np

PLATE_T = 0.006
PLATE_HALF_W = 0.05
SEG = 0.01
LASER_STANDOFF = 0.05
Y_LOCAL = np.linspace(-0.02, 0.02, 81)


@dataclass(frozen=True)
class Coupon:
    length: float
    gap: float
    hilo: float
    seam_offset: float
    amps: tuple
    periods: tuple
    phases: tuple

    def seam_y(self, x):
        y = self.seam_offset
        for a, p, ph in zip(self.amps, self.periods, self.phases):
            y = y + a * np.sin(2 * np.pi * x / p + ph)
        return y


def random_coupon(seed: int, length: float = 0.4) -> Coupon:
    r = np.random.default_rng(seed)
    return Coupon(
        length=length,
        gap=r.uniform(1.5e-3, 3.5e-3),
        hilo=r.uniform(0.0, 1.0e-3),
        seam_offset=r.uniform(-3e-3, 3e-3),
        amps=(r.uniform(1e-3, 4e-3), r.uniform(0.0, 1.5e-3)),
        periods=(r.uniform(0.15, 0.4), r.uniform(0.05, 0.12)),
        phases=(r.uniform(0, 2 * np.pi), r.uniform(0, 2 * np.pi)),
    )


def _plate(name, xc, y0, y1, top):
    hy = (y1 - y0) / 2
    return (
        f'<geom name="{name}" type="box" contype="0" conaffinity="0" '
        f'pos="{xc:.6f} {(y0 + y1) / 2:.6f} {top - PLATE_T / 2:.6f}" '
        f'size="{SEG / 2:.6f} {hy:.6f} {PLATE_T / 2:.6f}"/>'
    )


def build_xml(c: Coupon) -> str:
    plates = []
    for i in range(int(round(c.length / SEG))):
        xc = (i + 0.5) * SEG
        ys = float(c.seam_y(xc))
        plates.append(_plate(f"L{i}", xc, -PLATE_HALF_W, ys - c.gap / 2, 0.0))
        plates.append(_plate(f"R{i}", xc, ys + c.gap / 2, PLATE_HALF_W, c.hilo))
    return f"""
<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <geom name="backing" type="box" contype="0" conaffinity="0" pos="0.2 0 -0.02" size="0.5 0.2 0.005"/>
    {''.join(plates)}
    <body name="torch" pos="0 0 0.015">
      <joint name="tx" type="slide" axis="1 0 0" range="-0.05 0.6"/>
      <joint name="ty" type="slide" axis="0 1 0" range="-0.05 0.05"/>
      <joint name="tz" type="slide" axis="0 0 1" range="-0.02 0.05"/>
      <inertial pos="0 0 0" mass="0.5" diaginertia="1e-4 1e-4 1e-4"/>
      <geom name="tip" type="sphere" size="0.003" contype="0" conaffinity="0" group="1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="ax" joint="tx" kp="400" kv="30"/>
    <position name="ay" joint="ty" kp="400" kv="30"/>
    <position name="az" joint="tz" kp="400" kv="30"/>
  </actuator>
</mujoco>
"""


class WeldSim:
    SUBSTEPS = 5

    def __init__(self, coupon: Coupon):
        self.coupon = coupon
        self.m = mujoco.MjModel.from_xml_string(build_xml(coupon))
        self.d = mujoco.MjData(self.m)
        self.dt = self.m.opt.timestep * self.SUBSTEPS
        self._group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self._geomid = np.zeros(1, dtype=np.int32)
        self.d.ctrl[2] = 0.015
        mujoco.mj_forward(self.m, self.d)

    def step(self, x_cmd: float, y_cmd: float):
        self.d.ctrl[0], self.d.ctrl[1] = x_cmd, y_cmd
        for _ in range(self.SUBSTEPS):
            mujoco.mj_step(self.m, self.d)

    def torch_xy(self):
        return float(self.d.qpos[0]), float(self.d.qpos[1])

    def laser_profile(self, rng, noise=5e-5, dropout=0.02):
        x, y = self.torch_xy()
        z = np.full(Y_LOCAL.shape, np.nan)
        vec = np.array([0.0, 0.0, -1.0])
        for k, yl in enumerate(Y_LOCAL):
            pnt = np.array([x, y + yl, LASER_STANDOFF])
            dist = mujoco.mj_ray(self.m, self.d, pnt, vec, self._group, 1, -1, self._geomid)
            if dist >= 0:
                z[k] = LASER_STANDOFF - dist
        z = z + rng.normal(0.0, noise, z.shape)
        z[rng.random(z.shape) < dropout] = np.nan
        return Y_LOCAL.copy(), z
