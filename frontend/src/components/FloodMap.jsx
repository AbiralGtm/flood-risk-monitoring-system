import { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import { GeoJSON, MapContainer, Popup, TileLayer, useMap } from "react-leaflet";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Fab from "@mui/material/Fab";
import Tooltip from "@mui/material/Tooltip";
import FullscreenIcon from "@mui/icons-material/Fullscreen";
import FullscreenExitIcon from "@mui/icons-material/FullscreenExit";
import { getRiskColor, MISSING_DATA_COLOR, RISK_CLASS_ORDER, RIVER_STATIONS } from "../utils/risk";

// Label lookup for the popup's "River gauge used" row -- properties.river_station_id
// on a risk-map feature is a raw id (devghat/rajaiya), RIVER_STATIONS has the
// human labels already.
const stationLabel = (stationId) => RIVER_STATIONS.find((station) => station.id === stationId)?.label ?? stationId;
import GridSearch from "./GridSearch";

const SLIVER_COLOR = "#a9bcd0";
const GAP_COLOR = MISSING_DATA_COLOR;

const RISK_KEY_BY_VIEW = {
  predicted: "predicted_risk_class",
  static: "static_susceptibility_class",
};

const safeNumber = (value) => {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue : null;
};

const formatValue = (value, digits = 2, suffix = "") => {
  const numericValue = safeNumber(value);
  if (numericValue === null) {
    return "N/A";
  }

  return `${numericValue.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  })}${suffix}`;
};

const formatPercent = (value) => {
  const numericValue = safeNumber(value);
  if (numericValue === null) {
    return "N/A";
  }
  return `${(numericValue * 100).toFixed(1)}%`;
};

// Only re-frames the map when `resetKey` changes (e.g. switching between
// predicted/static view, which can have different geometry), not on every
// geoJsonData update -- geoJsonData is a new object reference on every
// fetch (new conditions, a simulation step, etc.), and re-fitting bounds
// every time would yank the map's pan/zoom out from under the user
// mid-simulation.
function FitBounds({ geoJsonData, resetKey }) {
  const map = useMap();
  const lastFitKeyRef = useRef(null);

  useEffect(() => {
    if (!geoJsonData?.features?.length || lastFitKeyRef.current === resetKey) {
      return;
    }
    const layer = L.geoJSON(geoJsonData);
    const bounds = layer.getBounds();
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [24, 24] });
      lastFitKeyRef.current = resetKey;
    }
  }, [geoJsonData, resetKey, map]);

  return null;
}

function FlyToGrid({ flyTarget, findFeatureById, onFound }) {
  const map = useMap();

  useEffect(() => {
    if (!flyTarget?.gridId) {
      return;
    }
    const feature = findFeatureById(flyTarget.gridId);
    if (!feature) {
      return;
    }
    const lat = safeNumber(feature.properties?.centroid_lat);
    const lon = safeNumber(feature.properties?.centroid_lon);
    if (lat === null || lon === null) {
      return;
    }
    map.flyTo([lat, lon], Math.max(map.getZoom(), 15), { duration: 0.6 });
    onFound({ latlng: L.latLng(lat, lon), properties: feature.properties });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flyTarget, map]);

  return null;
}

// Hidden SVG defs for hatch-pattern fills — referenced by url(#hatch-<class>)
// from Leaflet feature styles regardless of which <svg> root they live in.
function HatchDefs() {
  const classes = [...RISK_CLASS_ORDER, "gap", "sliver"];
  const colorFor = { gap: GAP_COLOR, sliver: SLIVER_COLOR };
  return (
    <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden="true">
      <defs>
        {classes.map((riskClass) => {
          const color = colorFor[riskClass] ?? getRiskColor(riskClass);
          return (
            <pattern
              key={riskClass}
              id={`hatch-${riskClass}`}
              patternUnits="userSpaceOnUse"
              width="6"
              height="6"
              patternTransform="rotate(45)"
            >
              <rect width="6" height="6" fill={color} fillOpacity="0.35" />
              <line x1="0" y1="0" x2="0" y2="6" stroke={color} strokeWidth="3" />
            </pattern>
          );
        })}
      </defs>
    </svg>
  );
}

function FloodMap({
  geoJsonData,
  sliverGridsData,
  gapGridsData,
  affectedAreaData,
  showAffectedArea,
  riversData,
  showRivers,
  wardsData,
  showWards,
  showHatch,
  viewMode,
  dataVersion,
  legend,
  onGridClick,
  flyTarget,
  onSearchSelect,
}) {
  const [selectedGrid, setSelectedGrid] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const mapShellRef = useRef(null);
  const riskKey = RISK_KEY_BY_VIEW[viewMode];

  useEffect(() => {
    setSelectedGrid(null);
  }, [geoJsonData, viewMode]);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(document.fullscreenElement === mapShellRef.current);
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
    };
  }, []);

  const toggleFullscreen = async () => {
    if (!mapShellRef.current) {
      return;
    }

    if (document.fullscreenElement === mapShellRef.current) {
      await document.exitFullscreen();
      return;
    }

    await mapShellRef.current.requestFullscreen();
  };

  const styleFeature = (feature) => {
    const riskClass = feature?.properties?.[riskKey];
    const fillColor = riskClass
      ? showHatch
        ? `url(#hatch-${riskClass})`
        : getRiskColor(riskClass)
      : MISSING_DATA_COLOR;
    return {
      color: "#ffffff",
      weight: 0.75,
      fillColor,
      fillOpacity: 0.82,
    };
  };

  const gapStyle = () => ({
    color: "#8a93a1",
    weight: 0.75,
    fillColor: showHatch ? "url(#hatch-gap)" : GAP_COLOR,
    fillOpacity: 0.75,
  });

  const sliverStyle = () => ({
    color: "#8fa3b8",
    weight: 0.5,
    fillColor: showHatch ? "url(#hatch-sliver)" : SLIVER_COLOR,
    fillOpacity: 0.55,
  });

  const affectedAreaStyle = () => ({
    color: "#c0392b",
    weight: 2.5,
    fill: false,
    dashArray: "6 4",
  });

  // Rendered ON TOP of the (near-opaque) risk grid, as a bold outline with
  // only a faint fill — a translucent fill underneath a choropleth layer is
  // invisible, since the grid tiles completely cover the river corridor's
  // extent. Cartographically, rivers read better as an overlaid line here
  // anyway.
  const riverStyle = () => ({
    color: "#0b3d91",
    weight: 2.5,
    fillColor: "#2b6cb0",
    fillOpacity: 0.12,
    dashArray: "1 0",
  });

  // Same "bold outline over the near-opaque risk grid" treatment as rivers
  // (a translucent fill would be invisible underneath the choropleth), but
  // visually distinct from the river corridors -- neutral grey/dashed since
  // this is an administrative reference layer, not a hydrological one.
  const wardStyle = () => ({
    color: "#4a5568",
    weight: 1.5,
    fill: false,
    dashArray: "4 3",
  });

  // Ward number labels: a permanent Leaflet tooltip anchored at each
  // polygon's own bounds center (the backend doesn't provide a centroid) --
  // good enough for orientation ("which ward am I looking at") without
  // pulling in a geometry library just for this.
  const onEachWardFeature = (feature, layer) => {
    const wardNumber = feature?.properties?.ward_number;
    if (wardNumber == null) {
      return;
    }
    layer.bindTooltip(String(wardNumber), {
      permanent: true,
      direction: "center",
      className: "ward-label",
    });
  };

  const onEachFeature = (feature, layer) => {
    const properties = feature?.properties ?? {};
    layer.on("click", (event) => {
      const grid = { latlng: event.latlng, properties };
      setSelectedGrid(grid);
      onGridClick?.(properties);
    });
  };

  const onEachExcludedFeature = (excludedType) => (feature, layer) => {
    const properties = feature?.properties ?? {};
    layer.on("click", (event) => {
      setSelectedGrid({
        latlng: event.latlng,
        properties: { ...properties, __excluded: true, __excludedType: excludedType },
      });
    });
  };

  const popupProperties = selectedGrid?.properties ?? null;
  const isExcludedPopup = Boolean(popupProperties?.__excluded);
  const isSliverPopup = popupProperties?.__excludedType === "sliver";

  const hasAffectedGeometry = Boolean(affectedAreaData?.geometry);
  const hasRivers = Boolean(riversData?.features?.length);
  const hasWards = Boolean(wardsData?.features?.length);
  const hasSlivers = Boolean(sliverGridsData?.features?.length);
  const hasGaps = Boolean(gapGridsData?.features?.length);

  const allGridIds = useMemo(() => {
    const ids = (geoJsonData?.features ?? []).map((f) => f.properties?.grid_id);
    const sliverIds = (sliverGridsData?.features ?? []).map((f) => f.properties?.grid_id);
    const gapIds = (gapGridsData?.features ?? []).map((f) => f.properties?.grid_id);
    return [...new Set([...ids, ...sliverIds, ...gapIds])].filter(Boolean);
  }, [geoJsonData, sliverGridsData, gapGridsData]);

  const findFeatureById = (gridId) => {
    const inMain = (geoJsonData?.features ?? []).find((f) => f.properties?.grid_id === gridId);
    if (inMain) {
      return inMain;
    }
    const inSliver = (sliverGridsData?.features ?? []).find((f) => f.properties?.grid_id === gridId);
    if (inSliver) {
      return inSliver;
    }
    return (gapGridsData?.features ?? []).find((f) => f.properties?.grid_id === gridId);
  };

  return (
    <Box ref={mapShellRef} sx={{ position: "relative", width: "100%", height: "100%" }}>
      <HatchDefs />

      <Stack
        direction="row"
        spacing={1}
        alignItems="center"
        sx={{ position: "absolute", top: 12, right: 12, zIndex: 1200 }}
      >
        <GridSearch gridIds={allGridIds} onSelect={onSearchSelect} />
        <Tooltip title={isFullscreen ? "Exit full screen" : "Full screen"}>
          <Fab size="small" onClick={toggleFullscreen}>
            {isFullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
          </Fab>
        </Tooltip>
      </Stack>

      <MapContainer center={[27.67, 84.43]} zoom={11} scrollWheelZoom style={{ width: "100%", height: "100%" }}>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {hasSlivers ? (
          <GeoJSON
            key={`sliver-${showHatch}-${dataVersion}`}
            data={sliverGridsData}
            style={sliverStyle}
            onEachFeature={onEachExcludedFeature("sliver")}
          />
        ) : null}
        {hasGaps ? (
          <GeoJSON
            key={`gap-${showHatch}-${dataVersion}`}
            data={gapGridsData}
            style={gapStyle}
            onEachFeature={onEachExcludedFeature("gap")}
          />
        ) : null}
        <GeoJSON
          // react-leaflet's GeoJSON only reads `data` once and never
          // redraws on prop changes (documented behavior) -- remounting via
          // `key` on every new fetch is the only way to get fresh colors.
          key={`${viewMode}-${showHatch}-${dataVersion}`}
          data={geoJsonData}
          style={styleFeature}
          onEachFeature={onEachFeature}
        />
        {showRivers && hasRivers ? (
          <GeoJSON key="rivers" data={riversData} style={riverStyle} interactive={false} />
        ) : null}
        {showWards && hasWards ? (
          <GeoJSON key="wards" data={wardsData} style={wardStyle} interactive={false} onEachFeature={onEachWardFeature} />
        ) : null}
        {showAffectedArea && hasAffectedGeometry ? (
          <GeoJSON key={`affected-area-${dataVersion}`} data={affectedAreaData} style={affectedAreaStyle} />
        ) : null}
        {selectedGrid ? (
          <Popup position={selectedGrid.latlng} eventHandlers={{ remove: () => setSelectedGrid(null) }}>
            <div className="popup-content">
              <h3>{popupProperties?.grid_id ?? "Unknown Grid"}</h3>
              {isExcludedPopup ? (
                isSliverPopup ? (
                  <p>
                    <strong>Outside modeled area</strong> — this is a sliver polygon (area below the
                    0.005 km² threshold), never part of the model's trained population. This is by
                    design, not a data gap.
                  </p>
                ) : (
                  <p>
                    <strong>Excluded — missing data</strong>: this grid is within the modeled
                    population but is missing a required field in the source dataset, so no prediction
                    is available for it.
                    {popupProperties?.__skipReason ? (
                      <>
                        <br />
                        <em>Reason: {popupProperties.__skipReason}</em>
                      </>
                    ) : null}
                  </p>
                )
              ) : viewMode === "static" ? (
                <>
                  <p>
                    <strong>Static susceptibility:</strong>{" "}
                    {popupProperties?.static_susceptibility_class ?? "N/A"}
                  </p>
                  <p>
                    <strong>Susceptibility score:</strong>{" "}
                    {formatValue(popupProperties?.static_susceptibility_score, 3)}
                  </p>
                  <p style={{ fontStyle: "italic" }}>
                    Terrain-only baseline — switch to "Predicted risk" to see the live model output for
                    this grid.
                  </p>
                </>
              ) : (
                <>
                  <p>
                    <strong>Predicted risk:</strong>{" "}
                    {popupProperties?.predicted_risk_class ?? "N/A (missing terrain data)"}
                  </p>
                  {popupProperties?.predicted_risk_class ? (
                    <p>
                      <strong>Confidence:</strong> {formatPercent(popupProperties?.risk_probability)}
                    </p>
                  ) : null}
                  {popupProperties?.river_station_id ? (
                    <p>
                      <strong>River gauge used:</strong> {stationLabel(popupProperties.river_station_id)} ·{" "}
                      {formatValue(popupProperties?.water_level_m, 2, " m")}
                      {popupProperties?.river_level_score_used_fallback ? (
                        <em> (fallback score — gauge unresolved)</em>
                      ) : null}
                    </p>
                  ) : null}
                </>
              )}
              <p>
                <strong>Elevation (mean):</strong> {formatValue(popupProperties?.elevation_mean, 2, " m")}
              </p>
              <p>
                <strong>Slope (mean):</strong> {formatValue(popupProperties?.slope_mean)}
              </p>
              <p>
                <strong>Dist. to Narayani / Rapti:</strong>{" "}
                {formatValue(popupProperties?.distance_to_narayani_m, 0, " m")} /{" "}
                {formatValue(popupProperties?.distance_to_rapti_m, 0, " m")}
              </p>
              <p>
                <strong>Nearest major river:</strong> {popupProperties?.nearest_major_river ?? "N/A"}
              </p>
              <p>
                <strong>Dominant LULC:</strong> {popupProperties?.lulc_dominant ?? "N/A"}
              </p>
              <p>
                <strong>Centroid:</strong> {formatValue(popupProperties?.centroid_lat, 6)},{" "}
                {formatValue(popupProperties?.centroid_lon, 6)}
              </p>
            </div>
          </Popup>
        ) : null}
        <FitBounds geoJsonData={geoJsonData} resetKey={viewMode} />
        <FlyToGrid flyTarget={flyTarget} findFeatureById={findFeatureById} onFound={setSelectedGrid} />
      </MapContainer>

      <Box sx={{ position: "absolute", bottom: 12, left: 12, zIndex: 1200 }}>{legend}</Box>
    </Box>
  );
}

export default FloodMap;
