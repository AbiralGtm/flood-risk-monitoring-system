import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Alert from "@mui/material/Alert";
import AlertTitle from "@mui/material/AlertTitle";
import Chip from "@mui/material/Chip";
import Table from "@mui/material/Table";
import TableHead from "@mui/material/TableHead";
import TableBody from "@mui/material/TableBody";
import TableRow from "@mui/material/TableRow";
import TableCell from "@mui/material/TableCell";
import Divider from "@mui/material/Divider";
import CircularProgress from "@mui/material/CircularProgress";
import { RISK_COLORS, RISK_CLASS_ORDER } from "../utils/risk";

// Server-side ranking (app's GET /api/wards/risk-report) already sorts
// riskiest-first and already distinguishes "scored, came back low risk"
// from "not covered by the model's study-area population" -- this panel
// just renders that, no client-side aggregation/point-in-polygon anymore
// (an earlier version of this component did that client-side; replaced
// once the backend shipped a proper server-side version that also handles
// the no-coverage distinction, which the client-side version didn't).

// Same threshold concept as before (a ward counts as "high risk" once at
// least a quarter of its scored grids are High/Danger) -- now applied to
// the backend's own high_danger_pct instead of a client-recomputed share.
const HIGH_RISK_WARD_PCT_THRESHOLD = 25;

function WardReportPanel({ wardsRiskReport, loading }) {
  if (loading && !wardsRiskReport) {
    return (
      <Stack alignItems="center" sx={{ py: 4 }}>
        <CircularProgress size={24} />
      </Stack>
    );
  }

  if (!wardsRiskReport) {
    return (
      <Alert severity="info">
        Ward report not available yet — submit conditions on the Conditions tab, or the backend's
        /api/wards/risk-report call may have failed (see the browser console).
      </Alert>
    );
  }

  const { wards, wards_with_no_grid_coverage: noCoverageWards, total_grids_scored: totalScored, skipped_grid_count: skippedCount } =
    wardsRiskReport;

  const highRiskWards = (wards ?? []).filter((ward) => ward.high_danger_pct >= HIGH_RISK_WARD_PCT_THRESHOLD);

  return (
    <Stack spacing={2}>
      <Typography variant="body2" color="text.secondary">
        {totalScored?.toLocaleString()} grids scored{skippedCount ? ` (${skippedCount.toLocaleString()} excluded)` : ""},
        aggregated server-side into {wards?.length ?? 0} wards with coverage, ranked riskiest first.
      </Typography>

      {highRiskWards.length ? (
        <Alert severity="error">
          <AlertTitle>
            {highRiskWards.length} ward{highRiskWards.length === 1 ? "" : "s"} at high risk
          </AlertTitle>
          <Typography variant="body2" sx={{ mb: 0.75 }}>
            At least {HIGH_RISK_WARD_PCT_THRESHOLD}% of scored grids in these wards are currently High or Danger:
          </Typography>
          <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
            {highRiskWards.map((ward) => (
              <Chip
                key={ward.ward_number}
                size="small"
                label={`Ward ${ward.ward_number} — ${Math.round(ward.high_danger_pct)}%`}
                sx={{ borderColor: RISK_COLORS.Danger, color: RISK_COLORS.Danger, fontWeight: 700 }}
                variant="outlined"
              />
            ))}
          </Stack>
        </Alert>
      ) : (
        <Alert severity="success">
          <AlertTitle>No high-risk wards currently</AlertTitle>
          No ward has {HIGH_RISK_WARD_PCT_THRESHOLD}% or more of its scored grids at High/Danger under the
          submitted conditions.
        </Alert>
      )}

      <Divider />

      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Rank</TableCell>
            <TableCell>Ward</TableCell>
            <TableCell>Grids</TableCell>
            {RISK_CLASS_ORDER.map((riskClass) => (
              <TableCell key={riskClass}>{riskClass}</TableCell>
            ))}
            <TableCell>% High+Danger</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {(wards ?? []).map((ward) => {
            const isHighRisk = ward.high_danger_pct >= HIGH_RISK_WARD_PCT_THRESHOLD;
            return (
              <TableRow key={ward.ward_number} sx={isHighRisk ? { bgcolor: "error.50" } : undefined}>
                <TableCell>{ward.rank}</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>{ward.ward_number}</TableCell>
                <TableCell>{ward.grid_count}</TableCell>
                {RISK_CLASS_ORDER.map((riskClass) => (
                  <TableCell key={riskClass}>{ward.risk_class_counts?.[riskClass] ?? 0}</TableCell>
                ))}
                <TableCell sx={{ fontWeight: isHighRisk ? 700 : 400, color: isHighRisk ? RISK_COLORS.Danger : undefined }}>
                  {Math.round(ward.high_danger_pct)}%
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>

      {noCoverageWards?.length ? (
        <Alert severity="info" sx={{ mt: 1 }}>
          <AlertTitle>No data for {noCoverageWards.length} ward{noCoverageWards.length === 1 ? "" : "s"}</AlertTitle>
          Ward{noCoverageWards.length === 1 ? "" : "s"} {noCoverageWards.join(", ")} {noCoverageWards.length === 1 ? "has" : "have"}{" "}
          zero grids in the model's study-area population (the scored grids are concentrated near the rivers, not the
          full municipal footprint) — this means <strong>not scored</strong>, not "verified low risk." Treat these as
          unknown, not safe.
        </Alert>
      ) : null}
    </Stack>
  );
}

export default WardReportPanel;
