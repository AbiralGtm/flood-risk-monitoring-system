import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Divider from "@mui/material/Divider";
import AlertsPanel from "./AlertsPanel";
import StatCards from "./StatCards";
import RiskBreakdownChart from "./RiskBreakdownChart";
import TransparencyCard from "./TransparencyCard";

function SummaryTab({ features, onJumpToGrid, stats, modelsData, activeModelId, submittedAt, liveStatus, forecastContext, onOpenAbout }) {
  return (
    <Stack spacing={2.5}>
      <AlertsPanel features={features} onJumpToGrid={onJumpToGrid} />

      <StatCards stats={stats} modelsData={modelsData} />

      <Divider />

      <Stack spacing={1}>
        <Typography variant="subtitle2">Risk breakdown</Typography>
        <RiskBreakdownChart counts={stats.counts} total={stats.totalGrids} />
      </Stack>

      <Divider />

      <TransparencyCard
        modelsData={modelsData}
        activeModelId={activeModelId}
        submittedAt={submittedAt}
        liveStatus={liveStatus}
        forecastContext={forecastContext}
        onOpenAbout={onOpenAbout}
      />
    </Stack>
  );
}

export default SummaryTab;
