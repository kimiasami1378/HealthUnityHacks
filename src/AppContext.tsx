// src/AppContext.tsx

import React, { createContext, useContext, useReducer } from "react";

// Define types for data
type JournalLog = { id: string; entry: string; date: string };
type ConversationLog = { id: string; messages: string[]; date: string };
type Insight = { id: string; content: string; date: string };

type AppState = {
  journalLogs: JournalLog[];
  conversationLogs: ConversationLog[];
  insights: Insight[];
};

type AppContextType = {
  state: AppState;
  dispatch: React.Dispatch<any>;
};

const AppContext = createContext<AppContextType | undefined>(undefined);

const initialState: AppState = {
  journalLogs: [],
  conversationLogs: [],
  insights: [],
};

// Reducer to manage state
function appReducer(state: AppState, action: any): AppState {
  switch (action.type) {
    case "SET_JOURNAL_LOGS":
      return { ...state, journalLogs: action.payload };
    case "SET_CONVERSATION_LOGS":
      return { ...state, conversationLogs: action.payload };
    case "SET_INSIGHTS":
      return { ...state, insights: action.payload };

    case "ADD_JOURNAL_LOG":
      return { ...state, journalLogs: [...state.journalLogs, action.payload] };
    case "DELETE_JOURNAL_LOG":
      return {
        ...state,
        journalLogs: state.journalLogs.filter(
          (log) => log.id !== action.payload
        ),
      };

    case "ADD_CONVERSATION_LOG":
      return {
        ...state,
        conversationLogs: [...state.conversationLogs, action.payload],
      };
    case "UPDATE_CONVERSATION_LOG":
      return {
        ...state,
        conversationLogs: state.conversationLogs.map((log) =>
          log.id === action.payload.id ? action.payload : log
        ),
      };
    case "DELETE_CONVERSATION_LOG":
      return {
        ...state,
        conversationLogs: state.conversationLogs.filter(
          (log) => log.id !== action.payload
        ),
      };

    case "ADD_INSIGHT":
      return { ...state, insights: [...state.insights, action.payload] };
    case "UPDATE_INSIGHT":
      return {
        ...state,
        insights: state.insights.map((insight) =>
          insight.id === action.payload.id ? action.payload : insight
        ),
      };
    case "DELETE_INSIGHT":
      return {
        ...state,
        insights: state.insights.filter(
          (insight) => insight.id !== action.payload
        ),
      };

    default:
      return state;
  }
}

// Provider component
export const AppProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
};

// Custom hook to use the context
export const useAppContext = () => {
  const context = useContext(AppContext);
  if (!context)
    throw new Error("useAppContext must be used within an AppProvider");
  return context;
};
