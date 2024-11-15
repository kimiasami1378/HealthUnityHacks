// src/dataFetching.ts

import { useAppContext } from "./AppContext";

export const useFetchData = () => {
  const { dispatch } = useAppContext();

  const fetchJournalLogs = async () => {
    try {
      const response = await fetch("/api/journal-logs");
      const data = await response.json();
      dispatch({ type: "SET_JOURNAL_LOGS", payload: data });
    } catch (error) {
      console.error("Error fetching journal logs:", error);
    }
  };

  const fetchConversationLogs = async () => {
    try {
      const response = await fetch("/api/conversation-logs");
      const data = await response.json();
      dispatch({ type: "SET_CONVERSATION_LOGS", payload: data });
    } catch (error) {
      console.error("Error fetching conversation logs:", error);
    }
  };

  const fetchInsights = async () => {
    try {
      const response = await fetch("/api/insights");
      const data = await response.json();
      dispatch({ type: "SET_INSIGHTS", payload: data });
    } catch (error) {
      console.error("Error fetching insights:", error);
    }
  };

  // Modify Data Functions
  const addJournalLog = async (newLog: any) => {
    dispatch({ type: "ADD_JOURNAL_LOG", payload: newLog });
  };

  const deleteJournalLog = async (id: string) => {
    dispatch({ type: "DELETE_JOURNAL_LOG", payload: id });
  };

  // Repeat add, update, and delete functions for conversation logs and insights similarly

  return {
    fetchJournalLogs,
    fetchConversationLogs,
    fetchInsights,
    addJournalLog,
    deleteJournalLog,
    // Include other add, update, delete functions here
  };
};
