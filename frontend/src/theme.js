import { createTheme } from "@mui/material/styles";

// Central MUI theme. Risk-class colors live in utils/risk.js (RISK_COLORS)
// and are used directly via `sx`/style props rather than theme palette keys,
// since they're semantic (Low/Medium/High/Danger) rather than UI-role colors.
const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#1d6fa5",
      dark: "#155276",
      light: "#4a90c2",
    },
    secondary: {
      main: "#2c3b41",
    },
    error: {
      main: "#e74c3c",
    },
    warning: {
      main: "#e67e22",
    },
    success: {
      main: "#2ecc71",
    },
    background: {
      default: "#eef1f5",
      paper: "#ffffff",
    },
  },
  shape: {
    borderRadius: 10,
  },
  typography: {
    fontFamily: '"Source Sans 3", "Roboto", "Segoe UI", sans-serif',
    h6: { fontWeight: 700 },
    subtitle2: { fontWeight: 700 },
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: "#ffffff",
          color: "#243447",
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: "none",
          fontWeight: 600,
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 600,
        },
      },
    },
  },
});

export default theme;
