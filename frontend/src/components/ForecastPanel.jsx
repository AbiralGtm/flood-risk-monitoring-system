import { useState } from "react";
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
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import ToggleButton from "@mui/material/ToggleButton";
import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import AccordionDetails from "@mui/material/AccordionDetails";
import Table from "@mui/material/Table";
import TableHead from "@mui/material/TableHead";
import TableBody from "@mui/material/TableBody";
import TableRow from "@mui/material/TableRow";
import TableCell from "@mui/material/TableCell";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import UndoIcon from "@mui/icons-material/Undo";
import CloudDownloadIcon from "@mui/icons-material/CloudDownload";
import {
  RIVER_STATIONS,
  RAINFALL_SLIDER_CONFIG,
  WATER_LEVEL_SLIDER,
  RISK_CLASS_ORDER,
  RISK_COLORS,
  getActiveStationWaterLevel,
  getActiveStationWarningLevel,
  getActiveStationDangerLevel,
  getModelOptions,
  withRatioScaledWaterLevel,
} from "../utils/risk";
import { FORECAST_HORIZONS, FORECAST_PRESETS, buildDefaultForecastConditions, getHorizonLabel } from "../utils/forecast";

function ForecastPanel({
  baseConditions,
  horizonId,
  onHorizonChange,
  onSubmit,
  loading,
  forecastContext,
  forecastHistory,
  onExitForecast,
  onFetchForecast,
  modelsData,
}) {
  // See ConditionsForm's identical use of getModelOptions.
  const modelOptions = getModelOptions(modelsData);
  // Seeded once from whatever gauge station the live/manual conditions
  // currently use, via the useState initializer -- deliberately not kept in
  // sync on every baseConditions change afterwards, or editing the forecast
  // form while live conditions refresh in the background would wipe out
  // what the user just typed.
  const [form, setForm] = useState(() => buildDefaultForecastConditions(baseConditions));

  // "Fill from forecast" -- pulls real forward-looking rainfall (Open-Meteo,
  // via GET /api/conditions/forecast) for the selected horizon and drops it
  // into the rainfall slider above. Kept separate from `form` state: this is
  // just a one-shot autofill action, not a live-synced field, so the user
  // can still hand-edit the slider afterwards without it snapping back.
  const [forecastFetch, setForecastFetch] = useState({ status: "idle", error: "", data: null });

  const updateField = (key, value) => {
    setForm((previous) => ({ ...previous, [key]: value }));
  };

  // Two independent river sliders now (SIMULATION_FORECAST_RIVER_FRONTEND_
  // PROMPT.md) -- same pattern as ConditionsForm's updateRiverLevel: keep
  // the derived water_level_m (used by predict/compare if a grid is clicked
  // while a forecast is showing) in sync with whichever station the gauge-
  // station accordion below has selected.
  const updateRiverLevel = (rvKey, value) => {
    setForm((previous) => {
      const next = { ...previous, [rvKey]: value };
      next.water_level_m = getActiveStationWaterLevel(next);
      return next;
    });
  };

  // Same fix as ConditionsForm's updateStation -- switching the gauge
  // station here must re-derive water_level_m/warning_level_m/danger_level_m
  // from that station's own pair, or they stay frozen on whichever was
  // active when the form was seeded (see buildDefaultForecastConditions).
  const updateStation = (riverStationId) => {
    setForm((previous) => {
      const next = { ...previous, river_station_id: riverStationId };
      next.water_level_m = getActiveStationWaterLevel(next);
      next.warning_level_m = getActiveStationWarningLevel(next);
      next.danger_level_m = getActiveStationDangerLevel(next);
      return next;
    });
  };

  const handleFillFromForecast = async () => {
    if (!onFetchForecast) return;
    setForecastFetch({ status: "loading", error: "", data: null });
    try {
      const result = await onFetchForecast(horizonId);
      updateField("rainfall_24h_mm", result.rainfall_24h_mm);
      setForecastFetch({ status: "success", error: "", data: result });
    } catch (fetchError) {
      console.error("Forecast conditions fetch error:", fetchError);
      setForecastFetch({
        status: "error",
        error: fetchError.message || "Forecast unavailable — Open-Meteo unreachable.",
        data: null,
      });
    }
  };

  // Quick scenarios are calibrated against Narayani/Devghat's danger level
  // (9.0m) -- withRatioScaledWaterLevel scales Rapti's value to the same
  // proportion of ITS danger level (3.7m) instead of mirroring the
  // identical absolute number, same fix as ConditionsForm's applyPreset
  // (see utils/risk.js for the full reasoning). Also explicitly overwrites
  // water_level_rv01_m/rv02_m, or a preset click would leave whatever was
  // seeded (real live readings, per buildDefaultForecastConditions) sitting
  // there stale and still winning in toRiskMapParams.
  const applyPreset = (preset) => {
    setForm((previous) => withRatioScaledWaterLevel({ ...previous, ...preset.values }, preset.values.water_level_m));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit(horizonId, form);
  };

  const isViewingForecast = Boolean(forecastContext);
  const historyRows = FORECAST_HORIZONS.map((horizon) => ({
    horizon,
    entry: forecastHistory?.[horizon.id],
  })).filter((row) => row.entry);

  return (
    <Box component="form" onSubmit={handleSubmit}>
      <Stack spacing={2}>
        <Alert severity="info">
          Forecast mode scores the map for a future horizon. Use <strong>Fill from forecast</strong> below to
          pull a real rainfall forecast (Open-Meteo) for the selected window, or enter your own "what would
          happen if" scenario. Neither river has a forecast source — no public API forecasts river level, for
          Narayani or Rapti — so both start from your current/live reading as a starting point; edit either
          one manually if you expect it to change. The model has a single rainfall input, so the horizon
          label itself doesn't change the computation — only the numbers you submit do.
        </Alert>

        {isViewingForecast ? (
          <Alert
            severity="warning"
            action={
              <Button size="small" startIcon={<UndoIcon />} onClick={onExitForecast} disabled={loading}>
                Back to current
              </Button>
            }
          >
            Viewing a <strong>{getHorizonLabel(forecastContext.horizonId)}</strong> forecast, generated{" "}
            {forecastContext.generatedAt.toLocaleTimeString()} — not the current live/manual risk map.
          </Alert>
        ) : null}

        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Horizon
          </Typography>
          <ToggleButtonGroup
            fullWidth
            size="small"
            exclusive
            value={horizonId}
            onChange={(_event, value) => value && onHorizonChange(value)}
          >
            {FORECAST_HORIZONS.map((horizon) => (
              <ToggleButton key={horizon.id} value={horizon.id}>
                {horizon.label}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        </Box>

        <Box>
          <Button
            size="small"
            variant="outlined"
            startIcon={<CloudDownloadIcon />}
            onClick={handleFillFromForecast}
            disabled={forecastFetch.status === "loading" || !onFetchForecast}
            fullWidth
          >
            {forecastFetch.status === "loading"
              ? "Fetching forecast…"
              : `Fill rainfall from forecast (${getHorizonLabel(horizonId)})`}
          </Button>
          {forecastFetch.status === "success" ? (
            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
              {forecastFetch.data.rainfall_24h_mm} mm expected from Open-Meteo, {forecastFetch.data.window_start} to{" "}
              {forecastFetch.data.window_end}.
            </Typography>
          ) : null}
          {forecastFetch.status === "error" ? (
            <Alert severity="warning" sx={{ mt: 1 }}>
              {forecastFetch.error}
            </Alert>
          ) : null}
        </Box>

        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Quick scenarios
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.5 }}>
            Sets Narayani to the scenario's level and scales Rapti to the same relative severity (its own
            danger level is much lower, ~3.7m vs. ~9.0m) — not the identical number.
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            {FORECAST_PRESETS.map((preset) => (
              <Chip
                key={preset.id}
                label={preset.label}
                onClick={() => applyPreset(preset)}
                sx={{ borderColor: preset.color, color: preset.color, fontWeight: 700 }}
                variant="outlined"
                disabled={loading}
              />
            ))}
          </Stack>
        </Box>

        <FormControl size="small" fullWidth>
          <InputLabel id="forecast-model-select-label">Model</InputLabel>
          <Select
            labelId="forecast-model-select-label"
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
                <Typography variant="body2">Expected {field.label}</Typography>
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
              <Typography variant="body2">Expected Narayani water level (Devghat)</Typography>
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
              <Typography variant="body2">Expected Rapti water level (Rajaiya)</Typography>
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

        <Accordion disableGutters elevation={0} sx={{ border: 1, borderColor: "divider", "&:before": { display: "none" } }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="body2">Gauge station &amp; thresholds</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={1.5}>
              <FormControl size="small" fullWidth>
                <InputLabel id="forecast-station-select-label">Gauge station</InputLabel>
                <Select
                  labelId="forecast-station-select-label"
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
            </Stack>
          </AccordionDetails>
        </Accordion>

        <Button type="submit" variant="contained" size="large" disabled={loading} fullWidth>
          {loading ? "Scoring grids…" : `Generate ${getHorizonLabel(horizonId)} forecast risk map`}
        </Button>

        {historyRows.length ? (
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Forecast runs this session
            </Typography>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Horizon</TableCell>
                  {RISK_CLASS_ORDER.map((riskClass) => (
                    <TableCell key={riskClass} sx={{ color: RISK_COLORS[riskClass] }}>
                      {riskClass}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {historyRows.map(({ horizon, entry }) => (
                  <TableRow key={horizon.id} selected={forecastContext?.horizonId === horizon.id}>
                    <TableCell>{horizon.label}</TableCell>
                    {RISK_CLASS_ORDER.map((riskClass) => (
                      <TableCell key={riskClass}>{entry.counts?.[riskClass] ?? 0}</TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        ) : null}
      </Stack>
    </Box>
  );
}

export default ForecastPanel;
