// Automated simulation scenarios — each is a short sequence of keyframe
// conditions that gets linearly interpolated into a full timeline. Playing
// a scenario steps through the timeline, calling /api/risk-map at each
// point and charting how the risk-class distribution evolves.
//
// keyframes.water_level_m is calibrated against NARAYANI/DEVGHAT only: dry
// baseline near 0mm/2m, and Devghat's danger level (9.0m) as the ceiling for
// the more extreme scenarios. SimulationPanel derives Rapti/Rajaiya's value
// from this by scaling it to the same proportion of Rapti's own (much
// lower, ~3.7m) danger level, rather than mirroring the identical number --
// see utils/risk.js's scaleWaterLevelToRapti. Don't read this file's
// water_level_m as "what Rapti will show," it's the Narayani reference
// value both rivers are derived from.
//
// Only rainfall_24h_mm and water_level_m are varied: the model (app/ml/
// features.py, updated 2026-08) only takes rainfall_24h_mm as a rainfall
// input now — the 1h/3h/6h/12h windows this used to also ramp were dropped
// from training and are ignored by the backend, so animating them here
// would suggest they affect the outcome when they don't.

const RAINFALL_KEYS = ["rainfall_24h_mm"];

export const SIMULATION_SCENARIOS = [
  {
    id: "dry-to-extreme",
    label: "Dry → Extreme Ramp",
    description:
      "Steady linear ramp from a dry baseline to an extreme cyclonic event. Good for finding roughly where each risk class kicks in.",
    steps: 12,
    keyframes: [
      { rainfall_24h_mm: 0, water_level_m: 2 },
      { rainfall_24h_mm: 140, water_level_m: 9.6 },
    ],
  },
  {
    id: "monsoon-storm",
    label: "Monsoon Storm Event",
    description:
      "Typical monsoon pattern: rainfall builds, peaks, then tapers — while the river keeps rising after the rain eases (routing lag).",
    steps: 12,
    keyframes: [
      { rainfall_24h_mm: 18, water_level_m: 4.0 },
      { rainfall_24h_mm: 90, water_level_m: 6.4 },
      { rainfall_24h_mm: 95, water_level_m: 8.3 },
    ],
  },
  {
    id: "flash-flood",
    label: "Flash Flood Spike",
    description:
      "A short, intense burst of rain with a fast-rising river — stress-tests how quickly the model flags Danger.",
    steps: 8,
    keyframes: [
      { rainfall_24h_mm: 20, water_level_m: 4.5 },
      { rainfall_24h_mm: 75, water_level_m: 9.2 },
    ],
  },
  {
    id: "recede",
    label: "Recession (Danger → Dry)",
    description: "Reverse case: starts at a Danger-level scenario and recedes back to dry, to check the map clears sensibly.",
    steps: 10,
    keyframes: [
      { rainfall_24h_mm: 145, water_level_m: 9.4 },
      { rainfall_24h_mm: 20, water_level_m: 3.0 },
    ],
  },
];

const lerp = (a, b, t) => a + (b - a) * t;

/** Build `steps + 1` condition points by piecewise-linear interpolation across `keyframes`. */
export function buildTimeline(scenario) {
  const { keyframes, steps } = scenario;
  const segments = keyframes.length - 1;
  const points = [];

  for (let i = 0; i <= steps; i += 1) {
    const t = i / steps; // 0..1 along the whole path
    const segFloat = t * segments;
    const segIndex = Math.min(segments - 1, Math.floor(segFloat));
    const segT = segFloat - segIndex;
    const from = keyframes[segIndex];
    const to = keyframes[segIndex + 1];

    const point = { water_level_m: Math.round(lerp(from.water_level_m, to.water_level_m, segT) * 100) / 100 };
    RAINFALL_KEYS.forEach((key) => {
      point[key] = Math.round(lerp(from[key], to[key], segT) * 10) / 10;
    });
    points.push(point);
  }

  return points;
}
