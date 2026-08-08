import Grid from "@mui/material/Grid";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import { RISK_CLASS_ORDER, RISK_COLORS, getModelLabel } from "../utils/risk";

function StatCards({ stats, modelsData }) {
  const { totalGrids, skippedCount, sliverCount, gapCount, counts, model } = stats;

  const cards = [
    { label: "Grids Scored", value: (totalGrids ?? 0).toLocaleString() },
    ...RISK_CLASS_ORDER.map((riskClass) => ({
      label: riskClass,
      value: (counts?.[riskClass] ?? 0).toLocaleString(),
      color: RISK_COLORS[riskClass],
      emphasize: riskClass === "Danger",
    })),
    {
      label: "Excluded",
      value: (skippedCount ?? 0).toLocaleString(),
      sub: `${(sliverCount ?? 0).toLocaleString()} sliver · ${(gapCount ?? 0).toLocaleString()} gap`,
    },
    { label: "Active Model", value: model ? getModelLabel(model, modelsData) : "N/A" },
  ];

  return (
    <Grid container spacing={1}>
      {cards.map((card) => (
        <Grid key={card.label} size={6}>
          <Paper
            variant="outlined"
            sx={{
              p: 1.25,
              borderColor: card.emphasize ? RISK_COLORS.Danger : undefined,
              borderWidth: card.emphasize ? 2 : 1,
            }}
          >
            <Typography variant="caption" color={card.color ?? "text.secondary"} fontWeight={700}>
              {card.label}
            </Typography>
            <Typography variant="subtitle1" fontWeight={700} noWrap>
              {card.value}
            </Typography>
            {card.sub ? (
              <Typography variant="caption" color="text.secondary" display="block">
                {card.sub}
              </Typography>
            ) : null}
          </Paper>
        </Grid>
      ))}
    </Grid>
  );
}

export default StatCards;
