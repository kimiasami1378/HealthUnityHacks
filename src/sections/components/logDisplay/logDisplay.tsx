// src/sections/logDisplay.tsx

import React from "react";
import { FaTrash } from "react-icons/fa";
import { Textarea } from "@mantine/core";
import "./logDisplay.css";

type LogDisplayProps = {
  entry: string;
  time: string;
  onDelete: () => void;
};

const LogDisplay: React.FC<LogDisplayProps> = ({ entry, time, onDelete }) => {
  return (
    <div className="log-display">
      <button className="delete-button" onClick={onDelete}>
        <FaTrash />
      </button>
      <Textarea
        value={entry}
        readOnly // Makes the text box uneditable
        autosize
        minRows={1}
        maxRows={5}
        className="entry-textarea"
        variant="unstyled"
      />
      <div className="time">{time}</div>
    </div>
  );
};

export default LogDisplay;
