// src/sections/loggingPage.tsx

import React from "react";
import { v4 as uuidv4 } from "uuid";
import "./loggingPage.css";

import { useAppContext } from "../AppContext";
import { useFetchData } from "../dataFetching";
import LogDisplay from "./components/logDisplay/logDisplay";
import InputBar from "./components/inputBar/inputBar";

const LoggingPage: React.FC = () => {
  const { state } = useAppContext();
  const { addJournalLog, deleteJournalLog } = useFetchData();

  const handleAddLog = (text: string) => {
    const newLog = {
      id: uuidv4(),
      entry: text,
      date: new Date().toISOString(),
    };
    addJournalLog(newLog);
  };

  const handleDeleteLog = (id: string) => {
    deleteJournalLog(id);
  };

  return (
    <div className="logging-page">
      <div className="log-area">
        {state.journalLogs.map((log) => (
          <LogDisplay
            key={log.id}
            entry={log.entry}
            time={new Date(log.date).toLocaleString()}
            onDelete={() => handleDeleteLog(log.id)}
          />
        ))}
      </div>
      <div className="inp-container">
        <InputBar onSend={handleAddLog} />
      </div>
    </div>
  );
};

export default LoggingPage;
