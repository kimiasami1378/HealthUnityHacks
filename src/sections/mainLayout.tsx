import React from "react";
import { Link, Outlet } from "react-router-dom";
import { FaHome, FaCalendarAlt, FaRobot } from "react-icons/fa";
import { IoLogoElectron } from "react-icons/io5";

import "./mainLayout.css";

const MainLayout: React.FC = () => {
  return (
    <div className="layout">
      <header>
        <IoLogoElectron className="logo-icon" />
        <h3 className="app-name">TrackSense</h3>
      </header>
      <main>
        <Outlet /> {/* This renders child routes */}
      </main>
      <footer>
        <ul>
          <li>
            <Link to="/">
              <FaHome />
            </Link>
          </li>
          <li>
            <Link to="/calendar">
              <FaCalendarAlt />
            </Link>
          </li>
          <li>
            <Link to="/chatbot">
              <FaRobot />
            </Link>
          </li>
        </ul>
      </footer>
    </div>
  );
};

export default MainLayout;
