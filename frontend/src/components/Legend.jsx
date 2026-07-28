import Paper from "@mui/material/Paper";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import { RISK_CLASS_ORDER, RISK_COLORS, countByRiskClass } from "../utils/risk";

const RISK_KEY_BY_VIEW = {
  predicted: "predicted_risk_class",
  static: "static_susceptibility_class",
};

const TITLE_BY_VIEW = {
  predicted: "Predicted flood risk",
  static: "Static susceptibility",
};

const SLIVER_COLOR = "#a9bcd0";
const GAP_COLOR = "#b8bfc8";

function Legend({ geoJsonData, viewMode, sliverCount, gapCount }) {
  const features = geoJsonData?.features ?? [];
  const riskKey = RISK_KEY_BY_VIEW[viewMode];
  const counts = countByRiskClass(features, riskKey);
  const total = features.length;

  return (
    <Paper elevation={4} sx={{ p: 1.5, width: 220 }}>
      <Typography variant="subtitle2">{TITLE_BY_VIEW[viewMode]}</Typography>
      <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
        {total.toLocaleString()} grid cells scored
      </Typography>
      <Stack spacing={0.5} sx={{ mt: 1 }}>
        {RISK_CLASS_ORDER.map((riskClass) => (
          <Stack key={riskClass} direction="row" alignItems="center" spacing={1}>
            <Box sx={{ width: 12, height: 12, borderRadius: 0.5, bgcolor: RISK_COLORS[riskClass], flexShrink: 0 }} />
            <Typography variant="caption" sx={{ flexGrow: 1 }}>
              {riskClass}
            </Typography>
            <Typography variant="caption" fontWeight={700}>
              {counts[riskClass].toLocaleString()}
            </Typography>
          </Stack>
        ))}
        {sliverCount ? (
          <Stack direction="row" alignItems="center" spacing={1}>
            <Box sx={{ width: 12, height: 12, borderRadius: 0.5, bgcolor: SLIVER_COLOR, flexShrink: 0 }} />
            <Typography variant="caption" sx={{ flexGrow: 1 }}>
              Outside modeled area
            </Typography>
            <Typography variant="caption" fontWeight={700}>
              {sliverCount.toLocaleString()}
            </Typography>
          </Stack>
        ) : null}
        {gapCount ? (
          <Stack direction="row" alignItems="center" spacing={1}>
            <Box sx={{ width: 12, height: 12, borderRadius: 0.5, bgcolor: GAP_COLOR, flexShrink: 0 }} />
            <Typography variant="caption" sx={{ flexGrow: 1 }}>
              Missing data
            </Typography>
            <Typography variant="caption" fontWeight={700}>
              {gapCount.toLocaleString()}
            </Typography>
          </Stack>
        ) : null}
      </Stack>
    </Paper>
  );
}

export default Legend;
