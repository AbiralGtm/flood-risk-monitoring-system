import { useEffect, useMemo, useRef, useState } from "react";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import IconButton from "@mui/material/IconButton";
import Slider from "@mui/material/Slider";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import ToggleButton from "@mui/material/ToggleButton";
import Chip from "@mui/material/Chip";
import Alert from "@mui/material/Alert";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import SkipNextIcon from "@mui/icons-material/SkipNext";
import SkipPreviousIcon from "@mui/icons-material/SkipPrevious";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend as RLegend } from "recharts";
import { SIMULATION_SCENARIOS, buildTimeline } from "../utils/simulation";
import { RISK_COLORS, WATER_LEVEL_SLIDER, scaleWaterLevelToRapti } from "../utils/risk";

const SPEED_OPTIONS = [0.5, 1, 2, 4];
const BASE_INTERVAL_MS = 2200;

function SimulationPanel({ baseConditions, onApplyStep, loading }) {
  const [scenarioId, setScenarioId] = useState(SIMULATION_SCENARIOS[0].id);
  const [stepIndex, setStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");

  // Backend river-station-resolution fix: each grid scores against its own
  // nearest river, so a single scripted water-level number can't represent
  // "Narayani rising, Rapti steady" -- the scenario this whole feature line
  // exists to test. The scripted timeline (below) still drives BOTH rivers
  // by default (a scenario like "flash flood" is a general storm severity
  // script, not tied to one specific river) -- but either river can be
  // manually pulled off the script at any time via its own slider, and that
  // override then sticks across steps/playback until the scenario is reset
  // or changed (handled by the effect below). null = "follow the script".
  const [riverOverrides, setRiverOverrides] = useState({ rv01: null, rv02: null });

  const scenario = useMemo(() => SIMULATION_SCENARIOS.find((s) => s.id === scenarioId), [scenarioId]);
  const timeline = useMemo(() => buildTimeline(scenario), [scenario]);
  const currentPoint = timeline[stepIndex];

  // The script is calibrated against Narayani/Devghat -- Rapti/Rajaiya's
  // scripted value is scaled to the same proportion of ITS OWN danger level
  // (much lower, ~3.7m vs ~9.0m), not mirrored to the identical number, or
  // both rivers would misleadingly "rise at the same level" regardless of
  // their very different real scales. See scaleWaterLevelToRapti.
  const rv01Value = riverOverrides.rv01 ?? currentPoint.water_level_m;
  const rv02Value = riverOverrides.rv02 ?? scaleWaterLevelToRapti(currentPoint.water_level_m, baseConditions);

  // Read inside runStep via ref, not as a state dependency of that effect --
  // putting riverOverrides directly in the effect's deps would restart the
  // playback timer (and re-score the current step) every time a slider is
  // dragged mid-playback, purely from a stale-closure workaround. The ref
  // always has the latest value without that side effect.
  const riverOverridesRef = useRef(riverOverrides);
  riverOverridesRef.current = riverOverrides;

  // Reset playback state whenever the scenario changes.
  useEffect(() => {
    setStepIndex(0);
    setHistory([]);
    setIsPlaying(false);
    setError("");
    setRiverOverrides({ rv01: null, rv02: null });
  }, [scenarioId]);

  useEffect(() => {
    if (!isPlaying) {
      return undefined;
    }
    let cancelled = false;

    const runStep = async () => {
      const point = timeline[stepIndex];
      // Each river independently follows the script unless the user has
      // pulled it off with its own slider (riverOverrides), in which case
      // that manual value is used instead and does NOT get overwritten by
      // the next scripted step -- see riverOverrides state comment above.
      // The script itself is calibrated against Narayani -- Rapti's
      // scripted value is ratio-scaled to its own (much lower) danger
      // level, not mirrored identically, same as rv02Value above.
      // water_level_m (used by predict/compare if a grid is clicked while
      // this map is showing) mirrors rv01/Narayani, same "primary station"
      // convention as elsewhere.
      const overrides = riverOverridesRef.current;
      const rv01 = overrides.rv01 ?? point.water_level_m;
      const rv02 = overrides.rv02 ?? scaleWaterLevelToRapti(point.water_level_m, baseConditions);
      const conditions = {
        ...baseConditions,
        rainfall_24h_mm: point.rainfall_24h_mm,
        water_level_m: rv01,
        water_level_rv01_m: rv01,
        water_level_rv02_m: rv02,
      };
      try {
        const result = await onApplyStep(conditions);
        if (cancelled) return;
        setHistory((previous) => [
          ...previous.filter((entry) => entry.step !== stepIndex),
          { step: stepIndex, ...result.counts, skipped: result.skipped },
        ]);
      } catch (stepError) {
        if (!cancelled) setError(stepError.message || "Simulation step failed.");
        setIsPlaying(false);
        return;
      }

      if (cancelled) return;

      if (stepIndex >= timeline.length - 1) {
        setIsPlaying(false);
        return;
      }

      await new Promise((resolve) => setTimeout(resolve, BASE_INTERVAL_MS / speed));
      if (!cancelled) {
        setStepIndex((previous) => previous + 1);
      }
    };

    runStep();
    // eslint-disable-next-line react-hooks/exhaustive-deps
    return () => {
      cancelled = true;
    };
  }, [isPlaying, stepIndex, timeline, speed]);

  const handlePlayPause = () => {
    if (isPlaying) {
      setIsPlaying(false);
      return;
    }
    setError("");
    if (stepIndex >= timeline.length - 1) {
      setStepIndex(0);
      setHistory([]);
    }
    setIsPlaying(true);
  };

  const handleReset = () => {
    setIsPlaying(false);
    setStepIndex(0);
    setHistory([]);
    setError("");
    setRiverOverrides({ rv01: null, rv02: null });
  };

  const applyManualStep = async (nextIndex) => {
    const point = timeline[nextIndex];
    // See riverOverrides / scaleWaterLevelToRapti comments in runStep above.
    const rv01 = riverOverrides.rv01 ?? point.water_level_m;
    const rv02 = riverOverrides.rv02 ?? scaleWaterLevelToRapti(point.water_level_m, baseConditions);
    const conditions = {
      ...baseConditions,
      rainfall_24h_mm: point.rainfall_24h_mm,
      water_level_m: rv01,
      water_level_rv01_m: rv01,
      water_level_rv02_m: rv02,
    };
    setStepIndex(nextIndex);
    try {
      const result = await onApplyStep(conditions);
      setHistory((previous) => [
        ...previous.filter((entry) => entry.step !== nextIndex),
        { step: nextIndex, ...result.counts, skipped: result.skipped },
      ]);
    } catch (stepError) {
      setError(stepError.message || "Simulation step failed.");
    }
  };

  const handleScrub = (_event, value) => {
    if (isPlaying) return;
    applyManualStep(value);
  };

  const handleStep = (direction) => {
    if (isPlaying || loading) return;
    const next = Math.min(timeline.length - 1, Math.max(0, stepIndex + direction));
    applyManualStep(next);
  };

  return (
    <Stack spacing={2}>
      <Typography variant="body2" color="text.secondary">
        Plays a scripted rainfall/river sequence through the live model, one step at a time, so you can watch
        risk classes shift in real time on the map instead of testing one static value.
      </Typography>

      <FormControl size="small" fullWidth>
        <InputLabel id="scenario-label">Scenario</InputLabel>
        <Select
          labelId="scenario-label"
          label="Scenario"
          value={scenarioId}
          onChange={(event) => setScenarioId(event.target.value)}
          disabled={isPlaying}
        >
          {SIMULATION_SCENARIOS.map((option) => (
            <MenuItem key={option.id} value={option.id}>
              {option.label}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      <Typography variant="caption" color="text.secondary">
        {scenario.description}
      </Typography>

      {error ? <Alert severity="error">{error}</Alert> : null}

      <Stack direction="row" spacing={1} alignItems="center" justifyContent="center">
        <IconButton onClick={() => handleStep(-1)} disabled={isPlaying || stepIndex === 0 || loading}>
          <SkipPreviousIcon />
        </IconButton>
        <IconButton color="primary" onClick={handlePlayPause} disabled={loading && !isPlaying}>
          {isPlaying ? <PauseIcon fontSize="large" /> : <PlayArrowIcon fontSize="large" />}
        </IconButton>
        <IconButton onClick={() => handleStep(1)} disabled={isPlaying || stepIndex >= timeline.length - 1 || loading}>
          <SkipNextIcon />
        </IconButton>
        <IconButton onClick={handleReset} disabled={isPlaying}>
          <RestartAltIcon />
        </IconButton>
      </Stack>

      <Box>
        <Slider
          size="small"
          min={0}
          max={timeline.length - 1}
          value={stepIndex}
          onChange={handleScrub}
          disabled={isPlaying}
          valueLabelDisplay="auto"
          valueLabelFormat={(value) => `Step ${value + 1}/${timeline.length}`}
        />
      </Box>

      <Stack spacing={1.5}>
        <Typography variant="caption" color="text.secondary">
          Rivers follow the scripted scenario by default — Rapti scaled to the same relative severity as
          Narayani (its danger level is much lower), not the identical number. Drag either slider to hold
          that river independently instead (e.g. Narayani rising per script while Rapti stays flat). Reset
          clears any override.
        </Typography>
        <Box>
          <Stack direction="row" justifyContent="space-between">
            <Typography variant="body2">
              Narayani (Devghat){riverOverrides.rv01 !== null ? " — manual" : ""}
            </Typography>
            <Typography variant="body2" fontWeight={700}>
              {rv01Value.toFixed(2)} m
            </Typography>
          </Stack>
          <Slider
            size="small"
            min={WATER_LEVEL_SLIDER.min}
            max={WATER_LEVEL_SLIDER.max}
            step={WATER_LEVEL_SLIDER.step}
            value={rv01Value}
            onChange={(_event, value) => setRiverOverrides((previous) => ({ ...previous, rv01: value }))}
            color="secondary"
          />
        </Box>
        <Box>
          <Stack direction="row" justifyContent="space-between">
            <Typography variant="body2">
              Rapti (Rajaiya){riverOverrides.rv02 !== null ? " — manual" : ""}
            </Typography>
            <Typography variant="body2" fontWeight={700}>
              {rv02Value.toFixed(2)} m
            </Typography>
          </Stack>
          <Slider
            size="small"
            min={WATER_LEVEL_SLIDER.min}
            max={WATER_LEVEL_SLIDER.max}
            step={WATER_LEVEL_SLIDER.step}
            value={rv02Value}
            onChange={(_event, value) => setRiverOverrides((previous) => ({ ...previous, rv02: value }))}
            color="secondary"
          />
        </Box>
      </Stack>

      <Stack direction="row" spacing={1} alignItems="center" justifyContent="center">
        <Typography variant="caption" color="text.secondary">
          Speed
        </Typography>
        <ToggleButtonGroup
          size="small"
          exclusive
          value={speed}
          onChange={(_event, value) => value && setSpeed(value)}
        >
          {SPEED_OPTIONS.map((option) => (
            <ToggleButton key={option} value={option}>
              {option}x
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
      </Stack>

      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap justifyContent="center">
        <Chip size="small" label={`24h rainfall: ${currentPoint.rainfall_24h_mm}mm`} />
        <Chip size="small" label={`Scripted level (Narayani ref): ${currentPoint.water_level_m}m`} color="primary" variant="outlined" />
      </Stack>

      <Box sx={{ width: "100%", height: 200 }}>
        {history.length ? (
          <ResponsiveContainer>
            <LineChart data={history} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
              <Tooltip />
              <RLegend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="Low" stroke={RISK_COLORS.Low} dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="Medium" stroke={RISK_COLORS.Medium} dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="High" stroke={RISK_COLORS.High} dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="Danger" stroke={RISK_COLORS.Danger} dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <Typography variant="caption" color="text.secondary">
            Press play to start — the chart fills in as each step scores against the live model.
          </Typography>
        )}
      </Box>
    </Stack>
  );
}

export default SimulationPanel;
