import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Link from "@mui/material/Link";

function TransparencyCard({ modelsData, activeModelId, submittedAt, liveStatus, forecastContext, onOpenAbout }) {
  // display_name is already on this object -- no separate hardcoded label
  // map needed (there used to be one here, independently of AboutPanel's
  // and StatCards'/ModelComparePanel's own copies, and all three had
  // drifted out of sync with the backend's actual model list at once).
  const activeModel = modelsData?.models?.find((model) => model.model_id === activeModelId);
  const accuracy = activeModel?.metrics?.test_accuracy;

  const timeLabel = submittedAt
    ? submittedAt.toLocaleString(undefined, {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        month: "short",
        day: "numeric",
      })
    : "Not yet loaded";

  // forecastContext takes priority over liveStatus: while viewing a
  // forecast run, `liveStatus` still describes the underlying live/manual
  // conditions state (untouched by forecast submissions), not what's
  // actually driving the map right now -- showing it here would be exactly
  // the kind of silently-wrong label this card exists to avoid.
  const sourceLabel = forecastContext
    ? `forecast scenario (${forecastContext.horizonId}) — manually entered, not live data`
    : liveStatus?.mode === "live"
      ? `live from BIPAD${liveStatus.stationsUsed?.length ? ` (${liveStatus.stationsUsed.join(", ")})` : ""}`
      : "manual entry — live data unavailable";

  return (
    <Stack spacing={0.75}>
      <Typography variant="caption" color="text.secondary">
        Active model
      </Typography>
      <Typography variant="body2" fontWeight={700}>
        {activeModel?.display_name ?? activeModelId}
        {accuracy != null ? ` (${(accuracy * 100).toFixed(1)}% test accuracy)` : ""}
      </Typography>
      <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
        Conditions as of
      </Typography>
      <Typography variant="body2">
        {timeLabel} — {sourceLabel}
      </Typography>
      <Link component="button" type="button" variant="body2" onClick={onOpenAbout} sx={{ textAlign: "left", mt: 1 }}>
        Full methodology &amp; known limitations →
      </Link>
    </Stack>
  );
}

export default TransparencyCard;
