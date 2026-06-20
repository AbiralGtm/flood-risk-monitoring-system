import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // No manualChunks: an earlier vendor-splitting config grouped mui/leaflet/
  // recharts into fixed chunks by package name, but those libs cross-import
  // each other internally, which produced real circular chunk dependencies
  // ("Circular chunk: charts -> mui -> charts", "mui -> leaflet -> mui" at
  // build time). Rollup still built successfully but the resulting chunks
  // could load in an order where one chunk references another before it's
  // initialized — invisible in `vite dev` (which doesn't chunk this way) but
  // a blank white page in the production build served from Docker/nginx.
  // Letting Rollup pick chunks automatically avoids the cycle.
  build: {},
});
