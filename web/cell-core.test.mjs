import test from "node:test";
import assert from "node:assert/strict";
import { runVirtualEpisode, reviewEpisode } from "./cell-core.mjs";

test("normal episode reaches Done with independent passing reviews", () => {
  const episode = runVirtualEpisode();
  assert.equal(episode.telemetry.at(-1).plc_state, "Done");
  assert.equal(episode.reviews.at(-1).verdict, "pass");
  assert.equal(episode.source.kind, "simulation");
});

for (const fault of ["gas_loss", "arc_dropout", "stale_vision", "seam_drift"]) {
  test(`${fault} faults and never leaves arc powered`, () => {
    const episode = runVirtualEpisode(fault);
    assert.equal(episode.telemetry.at(-1).plc_state, "Fault");
    assert.equal(episode.telemetry.at(-1).current_a, 0);
    assert.equal(episode.reviews.at(-1).verdict, "review");
    assert.equal(episode.events[0].cause, fault);
  });
}

test("MuJoCo trace never passes unmeasured current and gas", () => {
  const reviews = reviewEpisode({telemetry: [
    {time_s: 0, plc_state: "Weld", seam_offset_mm: 0.2},
    {time_s: 1, plc_state: "Done", seam_offset_mm: 0.1},
  ]});
  assert.equal(reviews[0].verdict, "unavailable");
  assert.equal(reviews.at(-1).verdict, "incomplete");
});
