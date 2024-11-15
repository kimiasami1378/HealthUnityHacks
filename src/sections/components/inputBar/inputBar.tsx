// src/sections/inputBar.tsx

import React, { useState } from "react";
import { Textarea, Button } from "@mantine/core";
import { IoArrowForward } from "react-icons/io5";
import "./inputBar.css";

type InputBarProps = {
  onSend: (text: string) => void;
};

const InputBar: React.FC<InputBarProps> = ({ onSend }) => {
  const [inputValue, setInputValue] = useState("");

  const handleSend = () => {
    if (inputValue.trim() !== "") {
      onSend(inputValue.trim());
      setInputValue(""); // Clear the input after sending
    }
  };

  return (
    <div className="input-bar">
      <Textarea
        value={inputValue}
        onChange={(e) => setInputValue(e.currentTarget.value)}
        placeholder="Type your log..."
        autosize
        minRows={1}
        maxRows={5} // Adjust the maximum number of rows as needed
        className="input-box"
      />
      <Button
        onClick={handleSend}
        variant="light"
        className="send-button"
        rightSection={<IoArrowForward size={14} />} // Correct prop for the icon
      >
        Send
      </Button>
    </div>
  );
};

export default InputBar;
