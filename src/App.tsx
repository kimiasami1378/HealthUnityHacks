import "./App.css";
import "./sections/theme.css";

import { AppProvider } from "./AppContext";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";

import MainLayout from "./sections/mainLayout";
import LoggingPage from "./sections/loggingPage";
import CalendarPage from "./sections/calendarPage";
import ChatbotPage from "./sections/chatbotPage";

function App() {
  return (
    <AppProvider>
      <Router>
        <Routes>
          <Route element={<MainLayout />}>
            <Route path="/" element={<LoggingPage />} />
            <Route path="/calendar" element={<CalendarPage />} />
            <Route path="/chatbot" element={<ChatbotPage />} />
          </Route>
        </Routes>
      </Router>
    </AppProvider>
  );
}

export default App;
