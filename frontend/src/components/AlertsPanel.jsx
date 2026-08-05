import Alert from "@mui/material/Alert";
import AlertTitle from "@mui/material/AlertTitle";
import Stack from "@mui/material/Stack";
import Chip from "@mui/material/Chip";
import Typography from "@mui/material/Typography";
import { RISK_COLORS, ALERT_CLASSES, getRiskRecommendation } from "../utils/risk";

function AlertsPanel({ features, onJumpToGrid }) {
  const alertFeatures = features.filter((feature) => ALERT_CLASSES.includes(feature?.properties?.predicted_risk_class));

  if (!alertFeatures.length) {
    return (
      <Alert severity="success">
        <AlertTitle>No high-risk areas currently</AlertTitle>
        All scored grids are Low or Medium under the submitted conditions.
      </Alert>
    );
  }

  const dangerCount = alertFeatures.filter((f) => f.properties.predicted_risk_class === "Danger").length;
  const highCount = alertFeatures.length - dangerCount;
  // Worst class present drives the recommendation shown -- Danger takes
  // priority over High if both are present on the map right now.
  const recommendation = getRiskRecommendation(dangerCount ? "Danger" : "High");

  return (
    <Alert severity="error">
      <AlertTitle>
        {dangerCount ? `${dangerCount} grid${dangerCount === 1 ? "" : "s"} at Danger` : null}
        {dangerCount && highCount ? " · " : null}
        {highCount ? `${highCount} grid${highCount === 1 ? "" : "s"} at High` : null}
      </AlertTitle>
      {recommendation ? (
        <Typography variant="body2" sx={{ mb: 0.75 }}>
          {recommendation}
        </Typography>
      ) : null}
      <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap sx={{ mt: 0.5 }}>
        {alertFeatures.slice(0, 20).map((feature) => {
          const riskClass = feature.properties.predicted_risk_class;
          return (
            <Chip
              key={feature.properties.grid_id}
              size="small"
              label={feature.properties.grid_id}
              onClick={() => onJumpToGrid?.(feature.properties.grid_id)}
              sx={{
                borderColor: RISK_COLORS[riskClass],
                color: RISK_COLORS[riskClass],
                fontWeight: 700,
              }}
              variant="outlined"
            />
          );
        })}
        {alertFeatures.length > 20 ? (
          <Typography variant="caption" color="text.secondary" sx={{ alignSelf: "center" }}>
            +{alertFeatures.length - 20} more
          </Typography>
        ) : null}
      </Stack>
    </Alert>
  );
}

export default AlertsPanel;
