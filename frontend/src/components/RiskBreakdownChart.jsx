import Box from "@mui/material/Box";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell, LabelList } from "recharts";
import { RISK_CLASS_ORDER, RISK_COLORS } from "../utils/risk";

function RiskBreakdownChart({ counts, total }) {
  const data = RISK_CLASS_ORDER.map((riskClass) => ({
    name: riskClass,
    count: counts?.[riskClass] ?? 0,
    pct: total ? (((counts?.[riskClass] ?? 0) / total) * 100).toFixed(1) : "0.0",
  }));

  return (
    <Box sx={{ width: "100%", height: 180 }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 4 }}>
          <XAxis type="number" hide />
          <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={64} />
          <Tooltip formatter={(value, _name, item) => [`${value} grids (${item.payload.pct}%)`, item.payload.name]} />
          <Bar dataKey="count" radius={[0, 6, 6, 0]}>
            {data.map((entry) => (
              <Cell key={entry.name} fill={RISK_COLORS[entry.name]} />
            ))}
            <LabelList dataKey="count" position="right" style={{ fontSize: 12, fontWeight: 700 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Box>
  );
}

export default RiskBreakdownChart;
