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
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import Box from "@mui/material/Box";
import CloseIcon from "@mui/icons-material/Close";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend as RLegend, CartesianGrid } from "recharts";
import { RISK_CLASS_ORDER } from "../utils/risk";

const METRIC_KEYS = [
  { key: "test_accuracy", label: "Accuracy", isPercent: true },
  { key: "macro_f1", label: "Macro F1", isPercent: false },
  { key: "danger_recall", label: "Danger recall", isPercent: false },
  { key: "danger_precision", label: "Danger precision", isPercent: false },
  { key: "roc_auc", label: "ROC-AUC", isPercent: false },
];

function AboutPanel({ modelsData, evaluationInfo, gisOutputTestInfo, stats, onClose }) {
  const models = modelsData?.models ?? [];

  // GET /api/models/gis-output-test (added 2026-08, proposal §3.13.2):
  // pre-computed, not live. The response carries THREE figures -- always
  // lead with `historical_evidence_override_only` (the methodologically
  // clean one, ground truth from independent evidence rather than the
  // model's own training-label rule) rather than `combined_all_historical_rows`,
  // which looks much better (IoU 0.82 vs 0.29) but is ~96% circular. Showing
  // the favorable-looking number here would misrepresent the model's actual
  // validated performance against independent historical flood records.
  const gisTest = gisOutputTestInfo?.available ? gisOutputTestInfo.results?.historical_evidence_override_only : null;

  // GET /api/models/evaluation-info (added 2026-08): the metrics below come
  // from a reproducible offline eval script (app/ml/evaluate.py), not a live
  // computation -- this caption makes that provenance visible instead of
  // implying the numbers update themselves. available: false (eval never run
  // in this environment) or the fetch simply failing both just suppress the
  // caption rather than showing broken text.
  const evalInfoText =
    evaluationInfo && evaluationInfo.available
      ? `Metrics last computed ${new Date(evaluationInfo.computed_at).toLocaleString()} — trained/validated/tested on a ${evaluationInfo.train_row_count.toLocaleString()} / ${evaluationInfo.val_row_count.toLocaleString()} / ${evaluationInfo.test_row_count.toLocaleString()}-row time-based split of ${evaluationInfo.dataset_row_count.toLocaleString()} total rows (${evaluationInfo.dataset_path}). ${evaluationInfo.test_rows_with_nan_river_level_score.toLocaleString()} test rows had no river-level reading and were filled with the ${evaluationInfo.river_level_score_fill_value} production fallback rather than dropped.`
      : null;

  // Mirrors gridExclusionText below: computed from the live response instead
  // of a hardcoded claim, so this line disappears on its own the moment real
  // numbers are wired in (app/config.py MODEL_METRICS) instead of silently
  // going stale like the old hardcoded version of this bullet did.
  const modelsMissingFullMetrics = models.filter(
    (m) => m.metrics?.test_accuracy == null || m.metrics?.roc_auc == null || m.metrics?.confusion_matrix == null,
  );
  const metricsGapText =
    modelsMissingFullMetrics.length > 0
      ? `${modelsMissingFullMetrics.map((m) => m.display_name).join(" and ")} ${
          modelsMissingFullMetrics.length === 1 ? "has" : "have"
        } no validated accuracy/F1/ROC-AUC/confusion-matrix numbers yet.`
      : null;

  const totalScored = stats?.totalGrids ?? 0;
  const skippedCount = stats?.skippedCount ?? 0;
  const sliverCount = stats?.sliverCount ?? 0;
  const gapCount = stats?.gapCount ?? 0;
  const totalPopulation = totalScored + skippedCount;

  // Computed from the live /api/risk-map response (via App.jsx's `stats`)
  // rather than hardcoded -- the underlying static grid dataset has been
  // replaced at least once already (2,575 grids with 141 exclusions ->
  // a pre-filtered 2,434-grid canonical file), and a hardcoded count here
  // silently went stale the moment that happened. Recomputing it live means
  // this can't drift from reality again the next time the dataset changes.
  const gridExclusionText =
    totalPopulation > 0
      ? skippedCount > 0
        ? `${skippedCount} of ${totalPopulation.toLocaleString()} grids (${((skippedCount / totalPopulation) * 100).toFixed(1)}%) are excluded from every prediction: ${sliverCount} are sliver polygons below the 0.005 km² area threshold the model was trained on — by design, not a data problem — and ${gapCount} are genuine data gaps (missing a required field in the source dataset).`
        : `All ${totalPopulation.toLocaleString()} grids in the current dataset have complete data and are scored on every prediction — no sliver or data-gap exclusions right now.`
      : null;

  const chartData = METRIC_KEYS.map((metric) => {
    const row = { metric: metric.label };
    models.forEach((model) => {
      const value = model.metrics?.[metric.key];
      row[model.display_name] = value != null ? Number((value * 100).toFixed(1)) : null;
    });
    return row;
  });

  // Keyed by model_id (stable/versioned by the backend), not display_name
  // (a human-editable label) -- a display_name rename shouldn't silently
  // lose its assigned color. Falls back to gray (below) for any model_id
  // not in this list, so a future 5th model doesn't break, just looks
  // undifferentiated until a color is picked for it.
  const MODEL_COLORS_BY_ID = {
    xgboost_tuned: "#1d6fa5",
    xgboost: "#5b9bd5",
    random_forest: "#8fa3b8",
    logreg: "#c9a227",
  };

  return (
    <Dialog open onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        About this prototype
        <IconButton onClick={onClose} size="small">
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        <Alert severity="info" sx={{ mb: 2 }}>
          This is a decision-support prototype for Bharatpur Metropolitan City, Chitwan, Nepal — not an
          official Department of Hydrology and Meteorology (DHM) flood warning system. Predicted risk
          depends entirely on the rainfall/river conditions in effect (live from BIPAD when available, or
          manually entered when it isn't); treat results as indicative, not authoritative.
        </Alert>

        <Typography variant="subtitle1" gutterBottom>
          Model accuracy comparison
        </Typography>
        <Box sx={{ width: "100%", height: 220, mb: 1 }}>
          <ResponsiveContainer>
            <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} unit="%" />
              <Tooltip />
              <RLegend wrapperStyle={{ fontSize: 11 }} />
              {models.map((model) => (
                <Bar key={model.model_id} dataKey={model.display_name} fill={MODEL_COLORS_BY_ID[model.model_id] ?? "#999"} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </Box>
        <Typography variant="caption" color="text.secondary">
          {modelsMissingFullMetrics.length > 0
            ? `Bars are absent for ${modelsMissingFullMetrics
                .map((m) => m.display_name)
                .join(" and ")} where the backend reports \`null\` (not yet benchmarked).`
            : `All ${models.length} model${models.length === 1 ? " has" : "s have"} validated metrics.`}
        </Typography>

        <Typography variant="subtitle1" sx={{ mt: 3 }} gutterBottom>
          Model metrics
        </Typography>
        {evalInfoText ? (
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
            {evalInfoText}
          </Typography>
        ) : null}
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Model</TableCell>
              <TableCell>Accuracy</TableCell>
              <TableCell>Macro F1</TableCell>
              <TableCell>Danger recall</TableCell>
              <TableCell>Danger precision</TableCell>
              <TableCell>ROC-AUC</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {models.map((model) => (
              <TableRow key={model.model_id} selected={model.is_best}>
                <TableCell>
                  {model.display_name}
                  {model.is_best ? " (best)" : ""}
                </TableCell>
                <TableCell>
                  {model.metrics?.test_accuracy != null ? `${(model.metrics.test_accuracy * 100).toFixed(1)}%` : "Not yet benchmarked"}
                </TableCell>
                <TableCell>{model.metrics?.macro_f1 ?? "—"}</TableCell>
                <TableCell>{model.metrics?.danger_recall ?? "—"}</TableCell>
                <TableCell>{model.metrics?.danger_precision ?? "—"}</TableCell>
                <TableCell>{model.metrics?.roc_auc ?? "—"}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>

        <Typography variant="subtitle1" sx={{ mt: 3 }} gutterBottom>
          Confusion matrices
        </Typography>
        {models.some((m) => m.metrics?.confusion_matrix) ? (
          models
            .filter((m) => m.metrics?.confusion_matrix)
            .map((model) => (
              <Box key={model.model_id} sx={{ mb: 2 }}>
                <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>
                  {model.display_name}
                  {model.is_best ? " (best)" : ""}
                </Typography>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ color: "text.secondary" }}>Actual \ Predicted</TableCell>
                      {RISK_CLASS_ORDER.map((riskClass) => (
                        <TableCell key={riskClass}>{riskClass}</TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {model.metrics.confusion_matrix.map((row, rowIndex) => (
                      <TableRow key={RISK_CLASS_ORDER[rowIndex]}>
                        <TableCell sx={{ color: "text.secondary" }}>{RISK_CLASS_ORDER[rowIndex]}</TableCell>
                        {row.map((count, colIndex) => (
                          <TableCell
                            key={colIndex}
                            sx={rowIndex === colIndex ? { fontWeight: 700, bgcolor: "action.hover" } : undefined}
                          >
                            {count}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Box>
            ))
        ) : (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Not yet computed for any model — see Known gaps below.
          </Typography>
        )}

        {gisTest ? (
          <>
            <Typography variant="subtitle1" sx={{ mt: 3 }} gutterBottom>
              Historical flood-extent validation
            </Typography>
            <Alert severity="warning" sx={{ mb: 1 }}>
              Comparing predicted High/Danger zones against {gisTest.event_count} independently-verified
              historical flood events ({gisTest.row_count.toLocaleString()} grid×event records): Intersection
              over Union <strong>{gisTest.intersection_over_union.toFixed(2)}</strong>, precision{" "}
              <strong>{gisTest.precision.toFixed(2)}</strong>, recall <strong>{gisTest.recall.toFixed(2)}</strong>.
              The model misses roughly {Math.round((1 - gisTest.recall) * 100)}% of these independently-verified
              flood grids — the honest number to cite, not a higher-looking figure from the same test that
              mostly re-derives the model's own training-label rule (see project docs for the full methodology
              and all three reported figures).
            </Alert>
          </>
        ) : null}

        <Typography variant="subtitle1" gutterBottom>
          Known gaps
        </Typography>
        <List dense disablePadding>
          {[
            gridExclusionText,
            metricsGapText,
            "Live rainfall/river conditions come from BIPAD Portal's public API, which has no published documentation or stability guarantee — it was integrated by inspecting BIPAD's own dashboard network requests, so it could change without notice. If it's unreachable the dashboard falls back to manual entry rather than guessing.",
            "Forecast mode (24h/48h/72h) can pull a real rainfall forecast (Open-Meteo, via \"Fill from forecast\") for the selected window, or take manually entered rainfall/river values. There's no river-level forecast source, so water level always carries over the current/live reading unless edited by hand. The model also has a single rainfall input, so different horizons only produce different results if the rainfall number itself differs between them; the horizon label doesn't change the computation on its own.",
          ]
            .filter(Boolean)
            .map((text) => (
            <ListItem key={text} sx={{ display: "list-item", listStyleType: "disc", ml: 2, py: 0.4 }}>
              <ListItemText primaryTypographyProps={{ variant: "body2" }}>{text}</ListItemText>
            </ListItem>
          ))}
        </List>
      </DialogContent>
    </Dialog>
  );
}

export default AboutPanel;
