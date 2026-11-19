from weld_twin import plc as P
from weld_twin.episode import run_episode

NAMES = {P.IDLE: "IDLE", P.SEAM_FIND: "SEAM_FIND", P.WELD: "WELD", P.FAULT: "FAULT", P.DONE: "DONE"}

print(f"{'seed':>4} {'mode':<12} {'rms mm':>7} {'max mm':>7} {'end state':<9}")
for seed in range(5):
    for label, kw in [("open-loop", dict(use_ml=False)), ("closed-loop", dict())]:
        r = run_episode(seed, P.MockPLC(), **kw)
        print(f"{seed:>4} {label:<12} {r.rms_mm:>7.2f} {r.max_mm:>7.2f} {NAMES[r.final_state]:<9}")
