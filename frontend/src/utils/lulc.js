export const LULC_CLASS_CONFIG = [
  { code: 10, keys: ["lulc10", "lulc_10"], label: "Tree cover", color: "#2b7a46" },
  { code: 20, keys: ["lulc20", "lulc_20"], label: "Shrubland", color: "#7e8b3c" },
  { code: 30, keys: ["lulc30", "lulc_30"], label: "Grassland", color: "#89b96a" },
  { code: 40, keys: ["lulc40", "lulc_40"], label: "Cropland", color: "#c9a227" },
  { code: 50, keys: ["lulc50", "lulc_50"], label: "Built-up", color: "#d06224" },
  { code: 60, keys: ["lulc60", "lulc_60"], label: "Bare/sparse", color: "#b78d62" },
  { code: 80, keys: ["lulc80", "lulc_80"], label: "Water", color: "#2b6cb0" },
  { code: 90, keys: ["lulc90", "lulc_90"], label: "Wetland", color: "#2f8f8b" },
];

export const LULC_COLOR_BY_LABEL = Object.fromEntries(
  LULC_CLASS_CONFIG.map((item) => [item.label, item.color]),
);

const safeNumber = (value) => {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue : null;
};

export const getExplicitDominantLulc = (properties) => {
  const value = properties?.lulc_dominant;
  if (value === null || value === undefined || String(value).trim() === "") {
    return null;
  }

  return String(value).trim();
};

export const getLulcBreakdown = (properties) =>
  LULC_CLASS_CONFIG.map((item) => {
    const matchedKey = item.keys.find((key) => safeNumber(properties?.[key]) !== null);
    return {
      ...item,
      value: matchedKey ? safeNumber(properties?.[matchedKey]) ?? 0 : 0,
    };
  });

export const getDominantLulcLabel = (properties) => {
  const explicitLabel = getExplicitDominantLulc(properties);
  if (explicitLabel) {
    return explicitLabel;
  }

  const breakdown = getLulcBreakdown(properties);
  const dominant = breakdown
    .filter((item) => item.value > 0)
    .sort((left, right) => right.value - left.value)[0];

  return dominant?.label ?? null;
};

export const getLulcColor = (label) => {
  if (label === null || label === undefined || String(label).trim() === "") {
    return "#b8bfc8";
  }

  return LULC_COLOR_BY_LABEL[String(label).trim()] ?? "#b8bfc8";
};
