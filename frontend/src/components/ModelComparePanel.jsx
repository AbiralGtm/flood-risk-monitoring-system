import { useEffect, useState } from "react";
import Drawer from "@mui/material/Drawer";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
import Collapse from "@mui/material/Collapse";
import LinearProgress from "@mui/material/LinearProgress";
import CircularProgress from "@mui/material/CircularProgress";
import Alert from "@mui/material/Alert";
import CloseIcon from "@mui/icons-material/Close";
import HistoryIcon from "@mui/icons-material/History";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import Divider from "@mui/material/Divider";
import { RISK_CLASS_ORDER, RISK_COLORS, ALERT_CLASSES, formatPercent, getRiskRecommendation, getModelLabel } from "../utils/risk";

// Proposal §3.7: "Rule-based explanation ... describe the major
// contributing factors behind the predicted risk." Backend-computed
// (app/ml/explain.py), deterministic thresholds -- describes the input
// feature values, not any one model's output, so it's the same regardless
// of which model card it's shown near (compareData.contributing_factors is
// top-level, not per-model).
const SEVERITY_COLOR = {
  high: RISK_COLORS.Danger,
  moderate: RISK_COLORS.High,
  low: "#64748b",
};

function ModelComparePanel({ gridId, compareData, loading, error, gridHistory, onClose, modelsData, earlyWarning }) {
  // Proposal §1.5: "alert information and basic recommendations based on
  // the predicted risk level of selected locations" -- this drawer IS the
  // location-based risk check, so the recommendation is tied to whichever
  // model is authoritative (compareData.best_model), not any of the
  // comparison-only models.
  const bestPrediction = compareData?.predictions?.[compareData?.best_model];
  const bestRiskClass = bestPrediction && !bestPrediction.error ? bestPrediction.predicted_risk_class : null;
  const isAlert = ALERT_CLASSES.includes(bestRiskClass);
  const recommendation = getRiskRecommendation(bestRiskClass);

  // Independent early-warning layer (app/early_warning.py) -- rate-of-rise
  // and catchment-rainfall rules that flag "worth a closer look" ahead of
  // the model's own risk_class crossing a threshold. Purely additive: never
  // changes bestRiskClass/RISK_COLORS above, only adds a badge alongside it.
  // Collapsed by default, reset whenever a different grid is opened so a
  // stale expanded reasons list from the previous grid doesn't linger.
  const [reasonsExpanded, setReasonsExpanded] = useState(false);
  useEffect(() => {
    setReasonsExpanded(false);
  }, [gridId]);

  return (
    <Drawer anchor="right" open={Boolean(gridId)} onClose={onClose}>
      <Box sx={{ width: 340, p: 2.5 }} role="presentation">
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2 }}>
          <Stack direction="row" alignItems="center" spacing={1}>
            <Typography variant="h6">Grid {gridId}</Typography>
            {isAlert ? <Chip size="small" color="error" label="ALERT" /> : null}
          </Stack>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Stack>

        {loading ? (
          <Stack alignItems="center" sx={{ py: 4 }}>
            <CircularProgress size={28} />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Comparing models…
            </Typography>
          </Stack>
        ) : null}

        {error ? <Alert severity="error">{error}</Alert> : null}

        {compareData?.diagnostics?.river_level_score_used_fallback ? (
          <Alert severity="warning" sx={{ mb: 1.5 }}>
            No river gauge resolved for this grid — river level used a flat fallback score (0.3) instead of the actual water level.
          </Alert>
        ) : null}

        {recommendation ? (
          <Alert severity={isAlert ? "error" : "info"} sx={{ mb: 1.5 }}>
            <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.25 }}>
              {bestRiskClass} — recommendation
            </Typography>
            {recommendation}
          </Alert>
        ) : null}

        {compareData?.contributing_factors?.length ? (
          <Box sx={{ mb: 1.5 }}>
            <Typography variant="subtitle2" sx={{ mb: 0.75 }}>
              Why this risk level
            </Typography>
            <Stack spacing={0.5}>
              {compareData.contributing_factors.map((factor) => (
                <Stack key={factor.factor} direction="row" alignItems="center" spacing={0.75}>
                  <Box
                    sx={{
                      width: 8,
                      height: 8,
                      borderRadius: "50%",
                      flexShrink: 0,
                      bgcolor: SEVERITY_COLOR[factor.severity] ?? SEVERITY_COLOR.low,
                    }}
                  />
                  <Typography variant="body2">{factor.label}</Typography>
                </Stack>
              ))}
            </Stack>
            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
              Rule-based, from the same conditions/terrain values used for this prediction — not a statistical
              breakdown of the model's internal reasoning.
            </Typography>
          </Box>
        ) : null}

        {compareData ? (
          <Stack spacing={1.5}>
            {Object.entries(compareData.predictions ?? {}).map(([modelId, prediction]) => {
              const isBest = modelId === compareData.best_model;
              const hasError = Boolean(prediction?.error);

              return (
                <Card
                  key={modelId}
                  variant="outlined"
                  sx={{
                    borderColor: isBest ? "primary.main" : "divider",
                    borderWidth: isBest ? 2 : 1,
                    opacity: hasError ? 0.6 : 1,
                  }}
                >
                  <CardContent>
                    <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
                      <Typography variant="subtitle2">{getModelLabel(modelId, modelsData)}</Typography>
                      {isBest ? <Chip size="small" color="primary" label="Authoritative" /> : null}
                    </Stack>

                    {isBest && earlyWarning?.flag ? (
                      <Box sx={{ mb: 1.25 }}>
                        <Stack direction="row" alignItems="center" spacing={0.5}>
                          <Chip
                            size="small"
                            icon={<WarningAmberIcon fontSize="small" />}
                            label="Early Warning"
                            sx={{
                              bgcolor: "#fff4e5",
                              color: "#9a5b00",
                              fontWeight: 700,
                              border: "1px solid #e67e22",
                              "& .MuiChip-icon": { color: "#e67e22" },
                            }}
                          />
                          {earlyWarning.reasons?.length ? (
                            <IconButton
                              size="small"
                              onClick={() => setReasonsExpanded((previous) => !previous)}
                              aria-label={reasonsExpanded ? "Hide early warning details" : "Show early warning details"}
                            >
                              <ExpandMoreIcon
                                fontSize="small"
                                sx={{ transition: "transform 0.15s", transform: reasonsExpanded ? "rotate(180deg)" : "none" }}
                              />
                            </IconButton>
                          ) : null}
                        </Stack>
                        {earlyWarning.reasons?.length ? (
                          <Collapse in={reasonsExpanded}>
                            <Stack spacing={0.25} sx={{ mt: 0.5, pl: 0.5 }}>
                              {earlyWarning.reasons.map((reason) => (
                                <Typography key={reason} variant="caption" color="text.secondary">
                                  • {reason}
                                </Typography>
                              ))}
                            </Stack>
                          </Collapse>
                        ) : null}
                      </Box>
                    ) : null}

                    {hasError ? (
                      <Alert severity="warning" sx={{ py: 0 }}>
                        Unavailable: {prediction.error}
                      </Alert>
                    ) : (
                      <>
                        <Typography variant="h5" sx={{ color: RISK_COLORS[prediction.predicted_risk_class] }}>
                          {prediction.predicted_risk_class}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Confidence: {formatPercent(prediction.risk_probability)}
                        </Typography>
                        <Stack spacing={0.5} sx={{ mt: 1.5 }}>
                          {RISK_CLASS_ORDER.map((riskClass) => (
                            <Box key={riskClass}>
                              <Stack direction="row" justifyContent="space-between">
                                <Typography variant="caption">{riskClass}</Typography>
                                <Typography variant="caption">
                                  {formatPercent(prediction.probabilities?.[riskClass])}
                                </Typography>
                              </Stack>
                              <LinearProgress
                                variant="determinate"
                                value={(prediction.probabilities?.[riskClass] ?? 0) * 100}
                                sx={{
                                  height: 5,
                                  borderRadius: 3,
                                  bgcolor: "action.hover",
                                  "& .MuiLinearProgress-bar": { bgcolor: RISK_COLORS[riskClass] },
                                }}
                              />
                            </Box>
                          ))}
                        </Stack>
                      </>
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </Stack>
        ) : null}

        {gridHistory?.length ? (
          <>
            <Divider sx={{ my: 2 }} />
            <Stack direction="row" alignItems="center" spacing={0.75} sx={{ mb: 1 }}>
              <HistoryIcon fontSize="small" color="action" />
              <Typography variant="subtitle2">Recently checked</Typography>
            </Stack>
            <Stack spacing={0.75}>
              {gridHistory.map((record) => (
                <Stack key={record.id} direction="row" justifyContent="space-between" alignItems="baseline">
                  <Typography variant="body2" sx={{ color: RISK_COLORS[record.predicted_risk_class], fontWeight: 600 }}>
                    {record.predicted_risk_class}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {record.created_at ? new Date(record.created_at).toLocaleString() : "—"}
                  </Typography>
                </Stack>
              ))}
            </Stack>
            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
              From this app's prediction history — recorded automatically, not a guaranteed-complete log.
            </Typography>
          </>
        ) : null}
      </Box>
    </Drawer>
  );
}

export default ModelComparePanel;
