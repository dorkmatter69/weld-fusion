export const STATES = Object.freeze({ IDLE: 0, SEAM_FIND: 1, WELD: 2, FAULT: 3, DONE: 4 });
export const STATE_NAMES = Object.freeze(["Idle", "Seam find", "Weld", "Fault", "Done"]);

export function runVirtualEpisode(fault = "none") {
  if (!["none", "gas_loss", "arc_dropout", "stale_vision", "seam_drift"].includes(fault)) {
    throw new Error("Unknown fault");
  }
  const episode = {
    schema_version: "1.0.0",
    episode_id: `sim-${fault}-${Date.now()}`,
    source: { kind: "simulation", generator: "WeldOps virtual cell 0.1" },
    process: { weld_type: "flat_butt_coupon", position: "simulation_only" },
    recipe: { current_a: 165, voltage_v: 21, gas_bar: 4.5 },
    timeline: { time_basis: "simulation_elapsed_s", clock_alignment_verified: true },
    streams: {}, telemetry: [], events: [], reviews: [],
  };
  let state = STATES.IDLE;
  let faultSince = null;
  for (let step = 0; step <= 400; step++) {
    const t = Number((step * 0.1).toFixed(1));
    if (state === STATES.IDLE && t >= 0.1) state = STATES.SEAM_FIND;
    if (state === STATES.SEAM_FIND && t >= 1) state = STATES.WELD;
    const injected = fault !== "none" && t >= 17;
    const gas = injected && fault === "gas_loss" ? 0.3 : 4.5;
    const visionAge = injected && fault === "stale_vision" ? t - 17 : 0.03;
    const seamOffset = injected && fault === "seam_drift" ? 12 : 0.8 * Math.sin(t * 0.7);
    const current = state === STATES.WELD && !(injected && (fault === "arc_dropout" || fault === "gas_loss")) ? 165 + 4 * Math.sin(t * 2) : 0;
    const voltage = current > 0 ? 21 + 0.6 * Math.cos(t * 1.3) : 0;
    const unsafe = state === STATES.WELD && (
      gas < 1 || visionAge > 1 || Math.abs(seamOffset) > 8 ||
      (injected && fault === "arc_dropout")
    );
    faultSince = unsafe ? (faultSince ?? t) : null;
    if (faultSince !== null && t - faultSince >= 0.6) {
      state = STATES.FAULT;
      episode.events.push({ time_s: t, type: "simulated_fault", cause: fault });
    }
    if (state === STATES.WELD && t >= 38) state = STATES.DONE;
    const row = {
      time_s: t, plc_state: STATE_NAMES[state], current_a: state === STATES.WELD ? current : 0,
      voltage_v: state === STATES.WELD ? voltage : 0, gas_bar: gas,
      seam_offset_mm: seamOffset, vision_age_s: visionAge,
    };
    episode.telemetry.push(row);
    if (state === STATES.FAULT || state === STATES.DONE) break;
  }
  episode.reviews = reviewEpisode(episode);
  return episode;
}

export function reviewEpisode(episode) {
  const rows = episode.telemetry;
  const first = (predicate) => rows.find(predicate);
  const hasProcess = rows.some(r => Number.isFinite(r.gas_bar) && Number.isFinite(r.current_a));
  const hasSeam = rows.some(r => Number.isFinite(r.seam_offset_mm));
  const process = hasProcess && first(r => r.plc_state === "Weld" && (r.gas_bar < 1 || r.current_a < 10));
  const seam = hasSeam && first(r => r.plc_state === "Weld" && (Math.abs(r.seam_offset_mm) > 8 || r.vision_age_s > 1));
  const fault = first(r => r.plc_state === "Fault");
  const reviewers = [
    { agent: "Process review", verdict: !hasProcess ? "unavailable" : process ? "flag" : "pass", evidence_time_s: process?.time_s ?? null,
      finding: !hasProcess ? "Current and gas not recorded" : process ? "Gas or arc signal outside simulated limits" : "No process threshold crossed" },
    { agent: "Seam review", verdict: !hasSeam ? "unavailable" : seam ? "flag" : "pass", evidence_time_s: seam?.time_s ?? null,
      finding: !hasSeam ? "Seam offset not recorded" : seam ? "Seam offset or vision age outside simulated limits" : "No seam offset threshold crossed" },
    { agent: "State review", verdict: fault ? "flag" : "pass", evidence_time_s: fault?.time_s ?? null,
      finding: fault ? "Virtual PLC entered Fault and disabled arc" : "No PLC fault recorded" },
  ];
  const status = reviewers.some(r => r.verdict === "flag") ? "review" :
    reviewers.some(r => r.verdict === "unavailable") ? "incomplete" : "pass";
  return [...reviewers, {
    agent: "Adjudication", verdict: status,
    evidence_time_s: null, finding: "Deterministic rules; human review required for real weld quality",
  }];
}
