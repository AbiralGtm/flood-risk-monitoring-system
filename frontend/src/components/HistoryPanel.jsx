import { useEffect, useState } from "react";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import IconButton from "@mui/material/IconButton";
import Typography from "@mui/material/Typography";
import Alert from "@mui/material/Alert";
import Table from "@mui/material/Table";
import TableHead from "@mui/material/TableHead";
import TableBody from "@mui/material/TableBody";
import TableRow from "@mui/material/TableRow";
import TableCell from "@mui/material/TableCell";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Stack from "@mui/material/Stack";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import CloseIcon from "@mui/icons-material/Close";
import RefreshIcon from "@mui/icons-material/Refresh";
import {
  getRainfallHistory,
  getRiverLevelHistory,
  getForecastHistory,
  getPredictionHistory,
  getRiskResultHistory,
} from "../services/api";
import { RISK_COLORS, formatPercent } from "../utils/risk";

// Every table here is a thin read-only view over the five history tables the
// backend has added since 2026-08 (see app/routes/history.py). Rows are
// written best-effort, server-side, as a side effect of calls the app
// already makes (GET /api/conditions/live, GET /api/conditions/forecast,
// POST /api/predict, GET /api/risk-map) -- there's no manual-entry path and
// no guarantee every call produced a row, so this is "recent activity," not
// a guaranteed-complete audit trail.

const SOURCES = {
  rainfall: {
    label: "Rainfall",
    fetcher: (limit) => getRainfallHistory(limit),
    loggedBy: "Logged by GET /api/conditions/live (BIPAD).",
    columns: ["Recorded at", "Source", "Rainfall (24h, mm)", "Stations used", "Logged at"],
    row: (r) => [r.recorded_at ?? "—", r.source ?? "—", r.rainfall_24h_mm, r.rain_stations_used ?? "—", r.created_at],
  },
  riverLevel: {
    label: "River level",
    fetcher: (limit) => getRiverLevelHistory(undefined, limit),
    loggedBy: "Logged by GET /api/conditions/live (BIPAD).",
    columns: ["Recorded at", "Station", "Water level (m)", "Warning (m)", "Danger (m)", "Logged at"],
    row: (r) => [r.recorded_at ?? "—", r.station_id ?? "—", r.water_level_m, r.warning_level_m ?? "—", r.danger_level_m ?? "—", r.created_at],
  },
  forecasts: {
    label: "Forecasts",
    fetcher: (limit) => getForecastHistory(undefined, limit),
    loggedBy: "Logged by GET /api/conditions/forecast (Open-Meteo).",
    columns: ["Horizon", "Rainfall (mm)", "Window start", "Window end", "Source", "Logged at"],
    row: (r) => [r.horizon ?? "—", r.rainfall_24h_mm, r.window_start ?? "—", r.window_end ?? "—", r.source ?? "—", r.created_at],
  },
  predictions: {
    label: "Predictions",
    fetcher: (limit) => getPredictionHistory(undefined, limit),
    loggedBy: "Logged by POST /api/predict — not predict/compare or risk-map runs (see note below).",
    columns: ["Grid", "Model", "Risk class", "Confidence", "Rainfall (mm)", "Water level (m)", "River fallback?", "Logged at"],
    row: (r) => [
      r.grid_id ?? "—",
      r.model_id ?? "—",
      r.predicted_risk_class,
      formatPercent(r.risk_probability),
      r.rainfall_24h_mm,
      r.water_level_m,
      r.river_level_score_used_fallback ? "Yes" : "No",
      r.created_at,
    ],
  },
  alerts: {
    label: "Alerts",
    fetcher: (limit) => getRiskResultHistory(undefined, undefined, limit),
    loggedBy: "Logged by GET /api/risk-map — only High/Danger grids, debounced to at most one write per ~60s per conditions (see note below). This is the project's persisted alert history.",
    columns: ["Grid", "Model", "Risk class", "Rainfall (mm)", "Water level (m)", "Run", "Logged at"],
    row: (r) => [r.grid_id ?? "—", r.model_id ?? "—", r.predicted_risk_class, r.rainfall_24h_mm, r.water_level_m, r.run_id ?? "—", r.created_at],
  },
};

const LIMIT_OPTIONS = [25, 50, 100, 200];

function formatCell(value, columnLabel) {
  if (value === null || value === undefined) return "—";
  if (columnLabel === "Logged at") {
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
  }
  if (columnLabel === "Risk class") {
    return value ?? "—";
  }
  if (columnLabel === "Run") {
    // run_id is a UUID grouping every row one risk-map call wrote -- the
    // first 8 chars are enough to visually group rows without wrapping the table.
    return typeof value === "string" ? value.slice(0, 8) : value;
  }
  return String(value);
}

function HistoryPanel({ onClose }) {
  const [activeSource, setActiveSource] = useState("rainfall");
  const [limit, setLimit] = useState(50);
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const source = SOURCES[activeSource];

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const result = await source.fetcher(limit);
      setRecords(result?.records ?? []);
    } catch (fetchError) {
      console.error("History API error:", fetchError);
      setError(fetchError.message || "History unavailable.");
      setRecords([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSource, limit]);

  return (
    <Dialog open onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        Database history
        <IconButton onClick={onClose} size="small">
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        <Alert severity="info" sx={{ mb: 2 }}>
          A live, read-only view of what this app has logged to its database (SQLite in dev / MySQL-ready).
          Rows are written automatically as a side effect of using the app — refreshing live conditions,
          fetching a forecast, or checking a grid — never from a dedicated "save" action. Logging is
          best-effort: a DB write failure never blocks the request it came from, so treat this as recent
          activity, not a guaranteed-complete record.
        </Alert>

        <Tabs value={activeSource} onChange={(_event, value) => setActiveSource(value)} sx={{ mb: 1.5 }}>
          {Object.entries(SOURCES).map(([key, config]) => (
            <Tab key={key} value={key} label={config.label} />
          ))}
        </Tabs>

        <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={2} sx={{ mb: 1.5 }}>
          <Typography variant="caption" color="text.secondary">
            {source.loggedBy}
          </Typography>
          <Stack direction="row" spacing={1.5} alignItems="center">
            <FormControl size="small" sx={{ minWidth: 100 }}>
              <InputLabel id="history-limit-label">Rows</InputLabel>
              <Select
                labelId="history-limit-label"
                label="Rows"
                value={limit}
                onChange={(event) => setLimit(event.target.value)}
              >
                {LIMIT_OPTIONS.map((option) => (
                  <MenuItem key={option} value={option}>
                    {option}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <Button size="small" startIcon={<RefreshIcon />} onClick={load} disabled={loading}>
              Refresh
            </Button>
          </Stack>
        </Stack>

        {activeSource === "predictions" ? (
          <Alert severity="warning" sx={{ mb: 1.5 }}>
            The grid detail drawer's model comparison uses /predict/compare, which isn't logged (to avoid
            counting the same click 3x). Opening a grid's detail panel also fires one single-model /predict
            call purely so it shows up here.
          </Alert>
        ) : null}

        {activeSource === "alerts" ? (
          <Alert severity="warning" sx={{ mb: 1.5 }}>
            Not every High/Danger grid that ever appeared on the map is here — writes are debounced
            (~60s per conditions) so dragging a slider doesn't spam the table. Treat this as "alert-worthy
            grids over time," not a complete log of every risk-map request. Rows sharing the same Run were
            written by the same map refresh.
          </Alert>
        ) : null}

        {error ? <Alert severity="error">{error}</Alert> : null}

        {loading ? (
          <Stack alignItems="center" sx={{ py: 4 }}>
            <CircularProgress size={28} />
          </Stack>
        ) : records.length === 0 && !error ? (
          <Typography variant="body2" color="text.secondary" sx={{ py: 3, textAlign: "center" }}>
            No {source.label.toLowerCase()} rows logged yet — use the app (refresh live conditions, fetch a
            forecast, or click a grid) to generate some.
          </Typography>
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                {source.columns.map((column) => (
                  <TableCell key={column}>{column}</TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {records.map((record) => (
                <TableRow key={record.id}>
                  {source.row(record).map((value, index) => {
                    const columnLabel = source.columns[index];
                    const isRiskCell = columnLabel === "Risk class";
                    return (
                      <TableCell key={columnLabel} sx={isRiskCell ? { color: RISK_COLORS[value], fontWeight: 700 } : undefined}>
                        {formatCell(value, columnLabel)}
                      </TableCell>
                    );
                  })}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </DialogContent>
    </Dialog>
  );
}

export default HistoryPanel;
