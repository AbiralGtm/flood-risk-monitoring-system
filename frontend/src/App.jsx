import { useCallback, useEffect, useMemo, useState } from "react";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import Chip from "@mui/material/Chip";
import Button from "@mui/material/Button";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Stack from "@mui/material/Stack";
import CircularProgress from "@mui/material/CircularProgress";
import Alert from "@mui/material/Alert";
import Tooltip from "@mui/material/Tooltip";
import MenuIcon from "@mui/icons-material/Menu";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import HistoryIcon from "@mui/icons-material/History";
import CircleIcon from "@mui/icons-material/Circle";

import FloodMap from "./components/FloodMap";
import Legend from "./components/Legend";
import ConditionsForm from "./components/ConditionsForm";
import SimulationPanel from "./components/SimulationPanel";
import ForecastPanel from "./components/ForecastPanel";
import LayersTab from "./components/LayersTab";
import WardReportPanel from "./components/WardReportPanel";
import SummaryTab from "./components/SummaryTab";
import ModelComparePanel from "./components/ModelComparePanel";
import AboutPanel from "./components/AboutPanel";
import HistoryPanel from "./components/HistoryPanel";
import {
  getRiskMap,
  getAffectedArea,
  getModels,
  getModelsEvaluationInfo,
  getGisOutputTestInfo,
  getRivers,
  getWards,
  getWardsRiskReport,
  getAllGrids,
  getStaticRisk,
  getLiveConditions,
  getForecastConditions,
  predictCompare,
  predictGrid,
  getPredictionHistory,
} from "./services/api";
import { DEFAULT_CONDITIONS, countByRiskClass, bucketSkippedGrids, toDynamicConditions, toRiskMapParams } from "./utils/risk";
import { DEFAULT_FORECAST_HORIZON, getHorizonLabel } from "./utils/forecast";

const DRAWER_WIDTH = 380;

// Backend river-station-resolution fix (2026-08): GET /api/conditions/live
// has always accepted river_station_id=devghat (RV01/Narayani) or
// =rajaiya (RV02/Rapti) individually -- the app just never called it twice
// before, since there was previously no reason to (one blended reading was
// applied to every grid regardless of which river it was actually near).
// This combines both calls into the single conditions object the rest of
// the app expects. rainfall_24h_mm is basin-wide (identical in both
// responses, not per-river) so either copy is fine -- devghat's is used.
// Only rainfall_24h_mm is carried in (not the sub-hourly windows -- see
// utils/risk.js toDynamicConditions, the model doesn't use them).
const combineLiveConditions = (devghatLive, rajaiyaLive, model, riverStationId = "devghat") => {
  const activeLive = riverStationId === "rajaiya" ? rajaiyaLive : devghatLive;
  return {
    model,
    rainfall_24h_mm: devghatLive.rainfall_24h_mm,
    water_level_rv01_m: devghatLive.water_level_m,
    water_level_rv02_m: rajaiyaLive.water_level_m,
    // Both stations' own threshold pair stored (not just the active one) --
    // ConditionsForm's gauge-station dropdown needs both to correctly
    // re-derive warning_level_m/danger_level_m when the user switches
    // stations. Storing only the "active at fetch time" pair was the bug:
    // switching stations had nothing to derive the other one's real
    // threshold from, so it stayed frozen on whichever was fetched first.
    warning_level_rv01_m: devghatLive.warning_level_m ?? "",
    danger_level_rv01_m: devghatLive.danger_level_m ?? "",
    warning_level_rv02_m: rajaiyaLive.warning_level_m ?? "",
    danger_level_rv02_m: rajaiyaLive.danger_level_m ?? "",
    water_level_m: activeLive.water_level_m,
    river_station_id: riverStationId,
    warning_level_m: activeLive.warning_level_m ?? "",
    danger_level_m: activeLive.danger_level_m ?? "",
  };
};

function App() {
  const [conditions, setConditions] = useState(DEFAULT_CONDITIONS);
  const [conditionsSubmittedAt, setConditionsSubmittedAt] = useState(null);
  const [riskMapData, setRiskMapData] = useState(null);
  const [skippedGrids, setSkippedGrids] = useState([]);
  const [affectedAreaData, setAffectedAreaData] = useState(null);
  const [modelsData, setModelsData] = useState(null);
  const [evaluationInfo, setEvaluationInfo] = useState(null);
  const [gisOutputTestInfo, setGisOutputTestInfo] = useState(null);
  const [riversData, setRiversData] = useState(null);
  const [wardsData, setWardsData] = useState(null);
  const [wardsRiskReport, setWardsRiskReport] = useState(null);
  const [allGridsData, setAllGridsData] = useState(null);
  // static_susceptibility_class is NOT present on /api/risk-map features —
  // verified live: it only exists on GET /api/risk/static. That endpoint
  // isn't conditions-dependent, so it's fetched once and used as its own
  // dataset for the "static" view mode. It reads from the same canonical
  // grid file as /api/risk-map (app/config.py:DATA_PATH), so its grid count
  // tracks whatever that file currently contains rather than a fixed number.
  const [staticRiskData, setStaticRiskData] = useState(null);

  const [viewMode, setViewMode] = useState("predicted");
  const [showAffectedArea, setShowAffectedArea] = useState(false);
  const [showRivers, setShowRivers] = useState(true);
  // Defaults off (unlike rivers) -- wards are an orientation/reference layer,
  // not something that changes how risk should be read, so it's opt-in
  // rather than adding default clutter on top of the risk grid.
  const [showWards, setShowWards] = useState(false);
  const [showHatch, setShowHatch] = useState(false);
  const [showAbout, setShowAbout] = useState(false);
  const [showHistory, setShowHistory] = useState(false);

  const [drawerOpen, setDrawerOpen] = useState(true);
  const [activeTab, setActiveTab] = useState("conditions");

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [apiConnected, setApiConnected] = useState(false);

  const [liveStatus, setLiveStatus] = useState({ mode: "loading", fetchedAt: null, stationsUsed: [], error: "" });
  const [refreshingLive, setRefreshingLive] = useState(false);
  // Set from the risk-map response's grids_using_river_fallback count (added
  // 2026-08 -- replaces an earlier top-level river_level_score_used_fallback
  // boolean of the same intent, which no longer exists on this response;
  // reading the old field now silently returns undefined). A nonzero count
  // means the backend couldn't resolve a river gauge/thresholds for at least
  // one grid in this request and used a flat 0.3 river_level_score for it —
  // the exact condition that caused the map/detail-panel disagreement bug
  // this app hit before. Surfaced as a banner so a regression is visible in
  // the UI, not just findable by diffing API responses.
  const [riverFallbackCount, setRiverFallbackCount] = useState(0);

  const [selectedGridId, setSelectedGridId] = useState(null);
  const [compareData, setCompareData] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState("");
  // Only POST /api/predict (single best-model call) returns early_warning/
  // early_warning_reasons -- /api/predict/compare doesn't (app/routes/
  // predict.py). Previously this predictGrid call was pure fire-and-forget,
  // used only to give the grid a history row; now its result is also kept
  // so ModelComparePanel can show the early-warning badge on the best
  // model's card. Independent of which model is "best" -- it's a property
  // of the grid's river/rainfall state, not any one model's output.
  const [earlyWarning, setEarlyWarning] = useState(null);
  // Past predictions logged for the currently-open grid (see runCompare) --
  // history.py only has rows for POST /api/predict, not /predict/compare
  // (the call the detail panel actually needs), so runCompare also fires a
  // fire-and-forget single-model predict purely to keep this populated.
  const [gridHistory, setGridHistory] = useState([]);

  const [flyTarget, setFlyTarget] = useState(null);

  // Forecast Risk Mode (proposal §3.2.2). `forecastContext` is set while the
  // map/detail-panel are showing a forecast run instead of current live/
  // manual conditions -- it carries the exact conditions object used for
  // that run (not just the horizon id) so grid clicks can build their
  // /api/predict/compare payload from the SAME conditions the map used,
  // rather than silently falling back to `conditions` (the live/manual
  // state) and reintroducing the map-vs-detail-panel disagreement bug this
  // app already had to fix once -- see toDynamicConditions/toRiskMapParams
  // in utils/risk.js and Task #40.
  const [forecastHorizonId, setForecastHorizonId] = useState(DEFAULT_FORECAST_HORIZON);
  const [forecastContext, setForecastContext] = useState(null);
  const [forecastHistory, setForecastHistory] = useState({});

  const fetchRiskMap = useCallback(async (nextConditions) => {
    setLoading(true);
    setError("");
    try {
      // Built through the same toRiskMapParams/toDynamicConditions helper
      // that runCompare uses for /api/predict/compare, so the map and the
      // detail panel can never silently disagree on which fields (e.g.
      // river_station_id) got sent for the same nominal conditions.
      const riskMapParams = toRiskMapParams(nextConditions);
      const [riskMap, affectedArea, wardReport] = await Promise.all([
        getRiskMap(riskMapParams),
        getAffectedArea(riskMapParams).catch((affectedAreaError) => {
          console.error("Affected-area API error:", affectedAreaError);
          return null;
        }),
        // Same condition-dependence as the map itself -- refreshed alongside
        // it rather than lazily on Wards-tab open, so the tab is never
        // showing a ward breakdown for stale conditions. Non-blocking: the
        // Wards tab just shows nothing if this fails, it never fails the map.
        getWardsRiskReport(riskMapParams).catch((wardReportError) => {
          console.error("Ward risk-report API error:", wardReportError);
          return null;
        }),
      ]);

      setRiskMapData(riskMap);
      setSkippedGrids(riskMap?.skipped_grids ?? []);
      setAffectedAreaData(affectedArea);
      setWardsRiskReport(wardReport);
      setApiConnected(true);
      setConditionsSubmittedAt(new Date());
      setRiverFallbackCount(riskMap?.grids_using_river_fallback ?? 0);

      return {
        counts: countByRiskClass(riskMap?.features ?? [], "predicted_risk_class"),
        skipped: (riskMap?.skipped_grids ?? []).length,
      };
    } catch (loadError) {
      console.error("Risk-map API error:", loadError);
      setError(loadError.message || "API unavailable.");
      setApiConnected(false);
      throw loadError;
    } finally {
      setLoading(false);
    }
  }, []);

  const bootstrapLiveConditions = useCallback(async () => {
    try {
      // Both stations fetched together -- each is server-cached ~a few
      // minutes, so this isn't wasteful. If EITHER fails, fall back to the
      // dry-day default for both rather than blending one real reading with
      // a guessed/zero one: a missing Rapti reading paired with a real
      // Narayani one would otherwise silently misrepresent Rapti as "calm"
      // instead of "unknown," the same false-calm failure mode
      // DEFAULT_CONDITIONS exists to avoid.
      const [devghatLive, rajaiyaLive] = await Promise.all([getLiveConditions("devghat"), getLiveConditions("rajaiya")]);
      const nextConditions = combineLiveConditions(devghatLive, rajaiyaLive, "xgboost_tuned", "devghat");
      setConditions(nextConditions);
      setLiveStatus({
        mode: "live",
        fetchedAt: devghatLive.fetched_at ? new Date(devghatLive.fetched_at) : new Date(),
        stationsUsed: devghatLive.rain_stations_used ?? [],
        error: "",
      });
      await fetchRiskMap(nextConditions);
    } catch (liveError) {
      console.error("Live conditions API error:", liveError);
      setLiveStatus({ mode: "manual", fetchedAt: null, stationsUsed: [], error: liveError.message || "unavailable" });
      setConditions(DEFAULT_CONDITIONS);
      await fetchRiskMap(DEFAULT_CONDITIONS).catch(() => {});
    }
  }, [fetchRiskMap]);

  useEffect(() => {
    bootstrapLiveConditions();
    getModels()
      .then(setModelsData)
      .catch((modelsError) => console.error("Models API error:", modelsError));
    // Metadata only (when/on-what-split the /api/models metrics were
    // computed) -- not required for the map or any prediction to work, so a
    // failure here just leaves the About panel's "last evaluated" caption
    // off rather than blocking anything.
    getModelsEvaluationInfo()
      .then(setEvaluationInfo)
      .catch((evalInfoError) => console.error("Models evaluation-info API error:", evalInfoError));
    // Same non-blocking treatment -- report/methodology material, not
    // something any live interaction depends on.
    getGisOutputTestInfo()
      .then(setGisOutputTestInfo)
      .catch((gisTestError) => console.error("GIS output test API error:", gisTestError));
    getRivers()
      .then(setRiversData)
      .catch((riversError) => console.error("Rivers layer API error:", riversError));
    // Fixed reference layer, same fetch-once-and-draw treatment as rivers.
    getWards()
      .then(setWardsData)
      .catch((wardsError) => console.error("Wards layer API error:", wardsError));
    getAllGrids()
      .then(setAllGridsData)
      .catch((allGridsError) => console.error("Grids API error:", allGridsError));
    getStaticRisk()
      .then(setStaticRiskData)
      .catch((staticRiskError) => console.error("Static risk API error:", staticRiskError));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleConditionsSubmit = (nextConditions) => {
    setConditions(nextConditions);
    setForecastContext(null);
    setSelectedGridId(null);
    setCompareData(null);
    setEarlyWarning(null);
    fetchRiskMap(nextConditions).catch(() => {});
  };

  // Always refreshes BOTH rivers (there's no such thing as "refresh just
  // Narayani" anymore -- the map needs both to score correctly). The
  // currently-selected gauge station (conditions.river_station_id) is kept
  // as whichever one stays "active" for the single-grid predict/compare
  // panel and the warning/danger display.
  const handleRefreshLive = async () => {
    setRefreshingLive(true);
    try {
      const [devghatLive, rajaiyaLive] = await Promise.all([getLiveConditions("devghat"), getLiveConditions("rajaiya")]);
      const nextConditions = combineLiveConditions(devghatLive, rajaiyaLive, conditions.model, conditions.river_station_id);
      setConditions(nextConditions);
      setForecastContext(null);
      setLiveStatus({
        mode: "live",
        fetchedAt: devghatLive.fetched_at ? new Date(devghatLive.fetched_at) : new Date(),
        stationsUsed: devghatLive.rain_stations_used ?? [],
        error: "",
      });
      await fetchRiskMap(nextConditions);
    } catch (liveError) {
      console.error("Live conditions refresh error:", liveError);
      setLiveStatus((previous) => ({ ...previous, mode: "manual", error: liveError.message || "unavailable" }));
    } finally {
      setRefreshingLive(false);
    }
  };

  // Used by SimulationPanel: applies one step's conditions to the live map
  // (same code path as a manual submit) and hands back the resulting risk
  // counts so the simulation's own timeline chart can plot them.
  const handleSimulationStep = async (nextConditions) => {
    setConditions(nextConditions);
    setForecastContext(null);
    return fetchRiskMap(nextConditions);
  };

  // Runs a forecast scenario through the same fetchRiskMap pipeline as
  // current conditions -- the model doesn't know or care whether a
  // rainfall/water-level pair represents "now" or "+48h," so there's no
  // separate endpoint needed for a v1. What matters is *labeling* it
  // clearly as a forecast (forecastContext) and keeping its exact
  // conditions object around so grid clicks stay consistent with what's on
  // the map (see activeConditions below).
  const handleForecastSubmit = async (horizonId, forecastConditions) => {
    setSelectedGridId(null);
    setCompareData(null);
    setEarlyWarning(null);
    try {
      const result = await fetchRiskMap(forecastConditions);
      const generatedAt = new Date();
      setForecastContext({ horizonId, generatedAt, conditions: forecastConditions });
      setForecastHistory((previous) => ({
        ...previous,
        [horizonId]: { ...result, generatedAt },
      }));
    } catch (forecastError) {
      // fetchRiskMap already recorded `error` for display; don't mark a
      // failed run as the active forecast context.
      console.error("Forecast risk-map error:", forecastError);
    }
  };

  const handleExitForecast = () => {
    setForecastContext(null);
    fetchRiskMap(conditions).catch(() => {});
  };

  // Whichever conditions are actually reflected on the map right now --
  // the forecast run's conditions while viewing a forecast, otherwise the
  // live/manual `conditions` state. Grid detail lookups (runCompare) MUST
  // read from this, not from `conditions` directly, or clicking a grid
  // while viewing a forecast would silently score the detail panel against
  // current conditions instead of the forecast -- the same class of bug
  // Task #40 fixed for the map/predict-compare split.
  const activeConditions = forecastContext?.conditions ?? conditions;

  const runCompare = async (gridId) => {
    setSelectedGridId(gridId);
    setCompareData(null);
    setCompareError("");
    setCompareLoading(true);
    setGridHistory([]);
    setEarlyWarning(null);

    const dynamicConditions = toDynamicConditions(activeConditions);

    try {
      const result = await predictCompare(gridId, dynamicConditions);
      setCompareData(result);

      // predictCompare (above) isn't logged server-side -- only POST
      // /api/predict is (see app/routes/predict.py: "/predict/compare does
      // not log, to avoid 3x-counting"). Fire a single-model predict with
      // the same conditions, primarily so this grid picks up a history row
      // for the "recently checked" list below -- but this is also currently
      // the ONLY endpoint that returns early_warning/early_warning_reasons
      // (app/early_warning.py), so its result is kept for that badge too.
      // Errors here must never affect the comparison already shown above,
      // so early warning just silently stays unset (no banner) on failure.
      predictGrid(gridId, dynamicConditions)
        .then((result) => {
          setEarlyWarning({
            flag: Boolean(result?.early_warning),
            reasons: result?.early_warning_reasons ?? [],
          });
        })
        .catch(() => {});
    } catch (compareErr) {
      console.error("Predict/compare API error:", compareErr);
      setCompareError(compareErr.message || "Comparison unavailable.");
    } finally {
      setCompareLoading(false);
    }

    getPredictionHistory(gridId, 10)
      .then((history) => setGridHistory(history?.records ?? []))
      .catch((historyError) => console.error("Grid history API error:", historyError));
  };

  // Pulled by ForecastPanel's "Fill from forecast" button. Kept as a thin
  // passthrough here (rather than having ForecastPanel import services/api
  // directly) to match every other component's pattern of not touching the
  // API layer itself -- see ConditionsForm/onRefreshLive for the same shape.
  const handleFetchForecast = (horizonId) => getForecastConditions(horizonId);

  const handleGridClick = async (properties) => {
    const gridId = properties?.grid_id;
    if (!gridId || properties?.__excluded) {
      return;
    }
    await runCompare(gridId);
  };

  const handleJumpToGrid = (gridId) => {
    setFlyTarget({ gridId, nonce: Date.now() });
  };

  const handleSearchSelect = (gridId) => {
    setFlyTarget({ gridId, nonce: Date.now() });
    runCompare(gridId);
  };

  const features = riskMapData?.features ?? [];

  const { slivers: sliverGrids, gaps: gapGrids } = useMemo(() => bucketSkippedGrids(skippedGrids), [skippedGrids]);

  // Merges each skipped grid's actual backend-reported `reason` string into
  // its geometry's properties (as __skipReason), so the map popup can show
  // the real cause instead of a hardcoded guess. Without this, the gap
  // popup used to always say "missing elevation/slope" regardless of what
  // actually failed -- misleading once other required fields (e.g. a newly
  // added model feature) started being the real cause for every grid.
  const makeExcludedCollection = (entries) => {
    if (!allGridsData?.features?.length || !entries.length) {
      return { type: "FeatureCollection", features: [] };
    }
    const reasonByGridId = new Map(entries.map((entry) => [entry.grid_id, entry.reason]));
    return {
      type: "FeatureCollection",
      features: allGridsData.features
        .filter((feature) => reasonByGridId.has(feature.properties?.grid_id))
        .map((feature) => ({
          ...feature,
          properties: {
            ...feature.properties,
            __skipReason: reasonByGridId.get(feature.properties?.grid_id),
          },
        })),
    };
  };

  const sliverGridsData = useMemo(() => makeExcludedCollection(sliverGrids), [allGridsData, sliverGrids]);
  const gapGridsData = useMemo(() => makeExcludedCollection(gapGrids), [allGridsData, gapGrids]);

  const stats = {
    totalGrids: features.length,
    skippedCount: skippedGrids.length,
    sliverCount: sliverGrids.length,
    gapCount: gapGrids.length,
    counts: countByRiskClass(features, "predicted_risk_class"),
    model: conditions.model,
  };

  // "static" mode shows the terrain-only baseline from a completely
  // separate endpoint (no live-model exclusions, since it doesn't run the
  // ML model at all); "predicted" mode shows the live-scored risk-map plus
  // its excluded sliver/gap layers, if any. Don't mix the two.
  const isStaticView = viewMode === "static";
  const activeMapData = isStaticView ? staticRiskData : riskMapData;
  const emptyCollection = { type: "FeatureCollection", features: [] };

  return (
    <Box sx={{ height: "100vh", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <AppBar position="static" elevation={1}>
        <Toolbar variant="dense" sx={{ flexWrap: "wrap", rowGap: 0.5, py: 0.5 }}>
          <IconButton edge="start" onClick={() => setDrawerOpen((previous) => !previous)} sx={{ mr: 1 }}>
            {drawerOpen ? <ChevronLeftIcon /> : <MenuIcon />}
          </IconButton>
          <Typography variant="h6" noWrap sx={{ mr: 2, fontSize: "1.1rem" }}>
            Bharatpur Flood Risk Monitor
          </Typography>
          <Chip
            size="small"
            icon={<CircleIcon sx={{ fontSize: "10px !important" }} color={apiConnected ? "success" : "error"} />}
            label={apiConnected ? "API connected" : "API disconnected"}
            variant="outlined"
          />
          {forecastContext ? (
            <Chip
              size="small"
              color="info"
              label={`Forecast ${getHorizonLabel(forecastContext.horizonId)} · ${forecastContext.generatedAt.toLocaleTimeString()}`}
              sx={{ ml: 1 }}
            />
          ) : (
            <Chip
              size="small"
              color={liveStatus.mode === "live" ? "success" : liveStatus.mode === "manual" ? "warning" : "default"}
              label={
                liveStatus.mode === "live"
                  ? `Live · ${conditionsSubmittedAt ? conditionsSubmittedAt.toLocaleTimeString() : ""}`
                  : liveStatus.mode === "manual"
                    ? "Manual / dry-day"
                    : "Loading…"
              }
              sx={{ ml: 1 }}
            />
          )}
          <Chip size="small" label="Prototype — not an official DHM warning" color="warning" variant="outlined" sx={{ ml: 1 }} />
          <Box sx={{ flexGrow: 1 }} />
          <Button startIcon={<HistoryIcon />} onClick={() => setShowHistory(true)}>
            History
          </Button>
          <Button startIcon={<InfoOutlinedIcon />} onClick={() => setShowAbout(true)}>
            About
          </Button>
        </Toolbar>
      </AppBar>

      <Box sx={{ flex: 1, display: "flex", minHeight: 0, position: "relative" }}>
        <Drawer
          variant="persistent"
          open={drawerOpen}
          sx={{
            width: drawerOpen ? DRAWER_WIDTH : 0,
            flexShrink: 0,
            "& .MuiDrawer-paper": { width: DRAWER_WIDTH, position: "relative", boxSizing: "border-box" },
          }}
        >
          <Tabs
            value={activeTab}
            onChange={(_event, value) => setActiveTab(value)}
            variant="scrollable"
            scrollButtons="auto"
            allowScrollButtonsMobile
            sx={{
              borderBottom: 1,
              borderColor: "divider",
              minHeight: 40,
              "& .MuiTab-root": {
                minHeight: 40,
                minWidth: "auto",
                px: 1.5,
                fontSize: "0.78rem",
                whiteSpace: "nowrap",
              },
            }}
          >
            <Tab value="conditions" label="Conditions" />
            <Tab value="forecast" label="Forecast" />
            <Tab value="simulation" label="Simulate" />
            <Tab value="layers" label="Layers" />
            <Tab value="wards" label="Wards" />
            <Tab value="summary" label="Summary" />
          </Tabs>

          <Box sx={{ p: 2, overflowY: "auto" }}>
            {error ? (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            ) : null}

            {riverFallbackCount > 0 ? (
              <Alert severity="warning" sx={{ mb: 2 }}>
                River gauge not resolved for {riverFallbackCount} grid{riverFallbackCount === 1 ? "" : "s"} — using a
                flat fallback score (0.3) for {riverFallbackCount === 1 ? "it" : "them"} instead of the actual water
                level.
              </Alert>
            ) : null}

            {activeTab === "conditions" ? (
              <ConditionsForm
                conditions={conditions}
                onSubmit={handleConditionsSubmit}
                loading={loading}
                liveStatus={liveStatus}
                onRefreshLive={handleRefreshLive}
                refreshingLive={refreshingLive}
                modelsData={modelsData}
              />
            ) : null}

            {activeTab === "forecast" ? (
              <ForecastPanel
                baseConditions={conditions}
                horizonId={forecastHorizonId}
                onHorizonChange={setForecastHorizonId}
                onSubmit={handleForecastSubmit}
                loading={loading}
                forecastContext={forecastContext}
                forecastHistory={forecastHistory}
                onExitForecast={handleExitForecast}
                onFetchForecast={handleFetchForecast}
                modelsData={modelsData}
              />
            ) : null}

            {activeTab === "simulation" ? (
              <SimulationPanel baseConditions={conditions} onApplyStep={handleSimulationStep} loading={loading} />
            ) : null}

            {activeTab === "layers" ? (
              <LayersTab
                viewMode={viewMode}
                setViewMode={setViewMode}
                showAffectedArea={showAffectedArea}
                setShowAffectedArea={setShowAffectedArea}
                showRivers={showRivers}
                setShowRivers={setShowRivers}
                showWards={showWards}
                setShowWards={setShowWards}
                showHatch={showHatch}
                setShowHatch={setShowHatch}
              />
            ) : null}

            {activeTab === "wards" ? <WardReportPanel wardsRiskReport={wardsRiskReport} loading={loading} /> : null}

            {activeTab === "summary" ? (
              <SummaryTab
                features={features}
                onJumpToGrid={handleJumpToGrid}
                stats={stats}
                modelsData={modelsData}
                activeModelId={conditions.model}
                submittedAt={conditionsSubmittedAt}
                liveStatus={liveStatus}
                forecastContext={forecastContext}
                onOpenAbout={() => setShowAbout(true)}
              />
            ) : null}
          </Box>
        </Drawer>

        <Box sx={{ flex: 1, position: "relative", minWidth: 0 }}>
          {loading && !riskMapData ? (
            <Stack alignItems="center" justifyContent="center" sx={{ height: "100%" }} spacing={1}>
              <CircularProgress />
              <Typography color="text.secondary">Scoring grids…</Typography>
            </Stack>
          ) : (
            <FloodMap
              geoJsonData={activeMapData}
              sliverGridsData={isStaticView ? emptyCollection : sliverGridsData}
              gapGridsData={isStaticView ? emptyCollection : gapGridsData}
              affectedAreaData={affectedAreaData}
              showAffectedArea={showAffectedArea && !isStaticView}
              riversData={riversData}
              showRivers={showRivers}
              wardsData={wardsData}
              showWards={showWards}
              showHatch={showHatch}
              viewMode={viewMode}
              dataVersion={conditionsSubmittedAt ? conditionsSubmittedAt.getTime() : 0}
              onGridClick={handleGridClick}
              flyTarget={flyTarget}
              onSearchSelect={handleSearchSelect}
              legend={
                <Legend
                  geoJsonData={activeMapData}
                  viewMode={viewMode}
                  sliverCount={isStaticView ? 0 : stats.sliverCount}
                  gapCount={isStaticView ? 0 : stats.gapCount}
                />
              }
            />
          )}

          {!drawerOpen ? (
            <Tooltip title="Open controls">
              <IconButton
                onClick={() => setDrawerOpen(true)}
                sx={{
                  position: "absolute",
                  top: 12,
                  left: 12,
                  zIndex: 1200,
                  bgcolor: "background.paper",
                  boxShadow: 3,
                  "&:hover": { bgcolor: "background.paper" },
                }}
              >
                <MenuIcon />
              </IconButton>
            </Tooltip>
          ) : null}
        </Box>
      </Box>

      <ModelComparePanel
        gridId={selectedGridId}
        compareData={compareData}
        loading={compareLoading}
        error={compareError}
        gridHistory={gridHistory}
        onClose={() => setSelectedGridId(null)}
        modelsData={modelsData}
        earlyWarning={earlyWarning}
      />

      {showAbout ? (
        <AboutPanel
          modelsData={modelsData}
          evaluationInfo={evaluationInfo}
          gisOutputTestInfo={gisOutputTestInfo}
          stats={stats}
          onClose={() => setShowAbout(false)}
        />
      ) : null}
      {showHistory ? <HistoryPanel onClose={() => setShowHistory(false)} /> : null}
    </Box>
  );
}

export default App;
