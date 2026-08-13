import { useEffect, useState } from "react";
import Stack from "@mui/material/Stack";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Slider from "@mui/material/Slider";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import Alert from "@mui/material/Alert";
import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import AccordionDetails from "@mui/material/AccordionDetails";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import RefreshIcon from "@mui/icons-material/Refresh";
import {
  RIVER_STATIONS,
  RAINFALL_SLIDER_CONFIG,
  WATER_LEVEL_SLIDER,
  CONDITION_PRESETS,
  getActiveStationWaterLevel,
  getActiveStationWarningLevel,
  getActiveStationDangerLevel,
  withRatioScaledWaterLevel,
  getModelOptions,
} from "../utils/risk";

function ConditionsForm({ conditions, onSubmit, loading, liveStatus, onRefreshLive, refreshingLive, modelsData }) {
  const [form, setForm] = useState(conditions);
  // Prefers the live GET /api/models list so a model added/removed on the
  // backend (e.g. xgboost_tuned, added 2026-08) shows up here without a
  // frontend code change; falls back to a static list only until that
  // request resolves. See utils/risk.js.
  const modelOptions = getModelOptions(modelsData);

  useEffect(() => {
    setForm(conditions);
  }, [conditions]);

  const updateField = (key, value) => {
    setForm((previous) => ({ ...previous, [key]: value }));
  };

  // Narayani (RV01) / Rapti (RV02) each have their own slider now (backend
  // river-station-resolution fix, 2026-08) -- water_level_m is kept in sync
  // as a DERIVED value (whichever river the gauge-station accordion has
  // selected), since predict/compare and the accordion's own warning/danger
  // display still only take one number for one grid's one river.
  const updateRiverLevel = (rvKey, value) => {
    setForm((previous) => {
      const next = { ...previous, [rvKey]: value };
      next.water_level_m = getActiveStationWaterLevel(next);
      return next;
    });
  };

  // Switching the gauge station must re-derive ALL THREE of water_level_m/
  // warning_level_m/danger_level_m together -- a real bug this app shipped:
  // this used to only recompute water_level_m, so the threshold fields
  // stayed frozen on whichever station's data was live-fetched first
  // (always Devghat by default), showing the same 7.3/9.0 warning/danger
  // for Rajaiya too instead of its real 3.3/3.7.
  const updateStation = (riverStationId) => {
    setForm((previous) => {
      const next = { ...previous, river_station_id: riverStationId };
      next.water_level_m = getActiveStationWaterLevel(next);
      next.warning_level_m = getActiveStationWarningLevel(next);
      next.danger_level_m = getActiveStationDangerLevel(next);
      return next;
    });
  };

  // Quick scenarios were calibrated against Narayani/Devghat's danger level
  // (9.0m) -- e.g. "Extreme" = 9.6m, just past danger. Applying that same
  // absolute number to Rapti/Rajaiya (danger 3.7m) would put it at ~2.6x
  // ITS danger level, not "just past danger" -- withRatioScaledWaterLevel
  // scales Rapti's value to the same proportion of its own danger level
  // instead, so both rivers read the same relative severity rather than
  // rising in lockstep to an identical, scale-mismatched number. It also
  // explicitly overwrites water_level_rv01_m/rv02_m, not just water_level_m,
  // or a preset click after a live fetch would silently keep sending the
  // old real per-river numbers instead of the preset's value (toRiskMapParams
  // has no way to tell "intentionally set" apart from "stale leftover" once
  // the fields are present).
  const applyPreset = (preset) => {
    const next = withRatioScaledWaterLevel({ ...form, ...preset.values }, preset.values.water_level_m);
    setForm(next);
    onSubmit(next);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit(form);
  };

  const isLive = liveStatus?.mode === "live";
  const isManual = liveStatus?.mode === "manual";

  return (
    <Box component="form" onSubmit={handleSubmit}>
      <Stack spacing={2}>
        <Alert
          severity={isLive ? "success" : isManual ? "warning" : "info"}
          action={
            <Button
              size="small"
              startIcon={<RefreshIcon />}
              onClick={() => onRefreshLive?.()}
              disabled={refreshingLive}
            >
              {refreshingLive ? "…" : isManual ? "Retry" : "Refresh"}
            </Button>
          }
        >
          {isLive
            ? `Live from BIPAD${liveStatus.fetchedAt ? ` · updated ${liveStatus.fetchedAt.toLocaleTimeString()}` : ""}${
                liveStatus.stationsUsed?.length ? ` · gauges: ${liveStatus.stationsUsed.join(", ")}` : ""
              }`
            : isManual
              ? "Live BIPAD data unavailable — showing dry-day defaults, not a real reading."
              : "Loading live conditions…"}
        </Alert>

        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Quick scenarios
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.5 }}>
            Sets Narayani to the scenario's level and scales Rapti to the same relative severity (its own
            danger level is much lower, ~3.7m vs. ~9.0m) — not the identical number.
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            {CONDITION_PRESETS.map((preset) => (
              <Chip
                key={preset.id}
                label={preset.label}
                onClick={() => applyPreset(preset)}
                sx={{
                  borderColor: preset.color,
                  color: preset.color,
                  fontWeight: 700,
                }}
                variant="outlined"
                disabled={loading}
              />
            ))}
          </Stack>
        </Box>

        <FormControl size="small" fullWidth>
          <InputLabel id="model-select-label">Model</InputLabel>
          <Select
            labelId="model-select-label"
            label="Model"
            value={form.model}
            onChange={(event) => updateField("model", event.target.value)}
          >
            {modelOptions.map((option) => (
              <MenuItem key={option.id} value={option.id}>
                {option.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <Stack spacing={2.5}>
          {RAINFALL_SLIDER_CONFIG.map((field) => (
            <Box key={field.key}>
              <Stack direction="row" justifyContent="space-between">
                <Typography variant="body2">{field.label}</Typography>
                <Typography variant="body2" fontWeight={700}>
                  {Number(form[field.key]).toFixed(1)} mm
                </Typography>
              </Stack>
              <Slider
                size="small"
                min={0}
                max={field.max}
                step={0.5}
                value={Number(form[field.key]) || 0}
                onChange={(_event, value) => updateField(field.key, value)}
              />
            </Box>
          ))}

          <Box>
            <Stack direction="row" justifyContent="space-between">
              <Typography variant="body2">Narayani water level (Devghat)</Typography>
              <Typography variant="body2" fontWeight={700}>
                {Number(form.water_level_rv01_m).toFixed(2)} m
              </Typography>
            </Stack>
            <Slider
              size="small"
              min={WATER_LEVEL_SLIDER.min}
              max={WATER_LEVEL_SLIDER.max}
              step={WATER_LEVEL_SLIDER.step}
              value={Number(form.water_level_rv01_m) || 0}
              onChange={(_event, value) => updateRiverLevel("water_level_rv01_m", value)}
              color="secondary"
            />
          </Box>

          <Box>
            <Stack direction="row" justifyContent="space-between">
              <Typography variant="body2">Rapti water level (Rajaiya)</Typography>
              <Typography variant="body2" fontWeight={700}>
                {Number(form.water_level_rv02_m).toFixed(2)} m
              </Typography>
            </Stack>
            <Slider
              size="small"
              min={WATER_LEVEL_SLIDER.min}
              max={WATER_LEVEL_SLIDER.max}
              step={WATER_LEVEL_SLIDER.step}
              value={Number(form.water_level_rv02_m) || 0}
              onChange={(_event, value) => updateRiverLevel("water_level_rv02_m", value)}
              color="secondary"
            />
          </Box>
        </Stack>

        <Typography variant="caption" color="text.secondary">
          Each grid on the map scores against whichever river it's actually nearest to — a rising Narayani no
          longer bleeds into Rapti-side grids' scores, or vice versa.
        </Typography>

        <Accordion disableGutters elevation={0} sx={{ border: 1, borderColor: "divider", "&:before": { display: "none" } }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="body2">Grid-detail station &amp; thresholds</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={1.5}>
              <Typography variant="caption" color="text.secondary">
                Only affects the single-grid detail panel (clicking a grid) — the map itself always scores each
                grid against its own nearest river regardless of this selection.
              </Typography>
              <FormControl size="small" fullWidth>
                <InputLabel id="station-select-label">Gauge station</InputLabel>
                <Select
                  labelId="station-select-label"
                  label="Gauge station"
                  value={form.river_station_id || "devghat"}
                  onChange={(event) => updateStation(event.target.value)}
                >
                  {RIVER_STATIONS.map((station) => (
                    <MenuItem key={station.id} value={station.id}>
                      {station.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <Stack direction="row" spacing={1.5}>
                <TextField
                  size="small"
                  label="Warning level (m)"
                  type="number"
                  fullWidth
                  value={form.warning_level_m}
                  onChange={(event) => updateField("warning_level_m", event.target.value)}
                  placeholder="auto"
                />
                <TextField
                  size="small"
                  label="Danger level (m)"
                  type="number"
                  fullWidth
                  value={form.danger_level_m}
                  onChange={(event) => updateField("danger_level_m", event.target.value)}
                  placeholder="auto"
                />
              </Stack>
              <Typography variant="caption" color="text.secondary">
                Auto-filled from the selected gauge when live data loads. If blank and the gauge is
                unrecognized, a documented fallback score (0.3) is used for the river component instead.
              </Typography>
            </Stack>
          </AccordionDetails>
        </Accordion>

        <Button type="submit" variant="contained" size="large" disabled={loading} fullWidth>
          {loading ? "Scoring grids…" : "Update risk map"}
        </Button>
      </Stack>
    </Box>
  );
}

export default ConditionsForm;
