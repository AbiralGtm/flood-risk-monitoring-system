import { useState } from "react";
import Autocomplete from "@mui/material/Autocomplete";
import TextField from "@mui/material/TextField";
import Paper from "@mui/material/Paper";

function GridSearch({ gridIds, onSelect }) {
  const [inputValue, setInputValue] = useState("");

  return (
    <Autocomplete
      size="small"
      options={gridIds}
      value={null}
      inputValue={inputValue}
      onInputChange={(_event, value) => setInputValue(value)}
      onChange={(_event, value) => {
        if (value) {
          onSelect(value);
          setInputValue("");
        }
      }}
      PaperComponent={(props) => <Paper elevation={4} {...props} />}
      sx={{ width: 220, bgcolor: "background.paper", borderRadius: 1 }}
      renderInput={(params) => <TextField {...params} placeholder="Jump to grid…" />}
    />
  );
}

export default GridSearch;
