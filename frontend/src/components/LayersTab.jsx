import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import ToggleButton from "@mui/material/ToggleButton";
import FormControlLabel from "@mui/material/FormControlLabel";
import Switch from "@mui/material/Switch";
import Divider from "@mui/material/Divider";

function LayersTab({
  viewMode,
  setViewMode,
  showAffectedArea,
  setShowAffectedArea,
  showRivers,
  setShowRivers,
  showWards,
  setShowWards,
  showHatch,
  setShowHatch,
}) {
  return (
    <Stack spacing={2.5}>
      <Stack spacing={1}>
        <Typography variant="subtitle2">Risk view</Typography>
        <ToggleButtonGroup
          exclusive
          fullWidth
          value={viewMode}
          onChange={(_event, value) => value && setViewMode(value)}
          size="small"
        >
          <ToggleButton value="predicted">Predicted risk</ToggleButton>
          <ToggleButton value="static">Static susceptibility</ToggleButton>
        </ToggleButtonGroup>
        <Typography variant="caption" color="text.secondary">
          {viewMode === "predicted"
            ? "Live model output under the current conditions."
            : "Terrain-only baseline — doesn't change with conditions."}
        </Typography>
      </Stack>

      <Divider />

      <Stack spacing={0.5}>
        <Typography variant="subtitle2">Reference layers</Typography>
        <FormControlLabel
          control={<Switch checked={showRivers} onChange={(event) => setShowRivers(event.target.checked)} />}
          label="Narayani / Rapti river corridors"
        />
        <FormControlLabel
          control={<Switch checked={showAffectedArea} onChange={(event) => setShowAffectedArea(event.target.checked)} />}
          label="Affected-area outline (High + Danger)"
        />
        <FormControlLabel
          control={<Switch checked={showWards} onChange={(event) => setShowWards(event.target.checked)} />}
          label="Ward boundaries (29 wards)"
        />
      </Stack>

      <Divider />

      <Stack spacing={0.5}>
        <Typography variant="subtitle2">Accessibility</Typography>
        <FormControlLabel
          control={<Switch checked={showHatch} onChange={(event) => setShowHatch(event.target.checked)} />}
          label="Hatch-pattern fill (colorblind-friendly)"
        />
      </Stack>
    </Stack>
  );
}

export default LayersTab;
