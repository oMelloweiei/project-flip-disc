// component/NavBar.js

import { NavLink } from "react-router-dom";
import logo from "../assets/images/Logo_White.webp";
import "./navstyle.css";

const NavBar = () => {
  return (
    <nav className="flex justify-between align-center bg-black py-4 px-6">
       <NavLink to="/">
          <img className="object-cover w-30 h-12" src={logo} alt="Logo" />
          </NavLink>
      <ul className="flex space-x-6">
        {/* <li>
          <NavLink
            to="/"
            className={({ isActive }) =>
              `text-white text-lg font-semibold hover:text-gray-400 ${
                isActive ? "underline" : ""
              }`
            }
          >
            Home
          </NavLink>
        </li>
        <li>
          <NavLink
            to="/about"
            className={({ isActive }) =>
              `text-white text-lg font-semibold hover:text-gray-400 ${
                isActive ? "underline" : ""
              }`
            }
          >
            About
          </NavLink>
        </li>
        <li>
          <NavLink
            to="/contact"
            className={({ isActive }) =>
              `text-white text-lg font-semibold hover:text-gray-400 ${
                isActive ? "underline" : ""
              }`
            }
          >
            Contact
          </NavLink>
        </li> */}
        <li>
        <NavLink
            to="/test"
            className={({ isActive }) =>
              `text-white text-lg font-semibold hover:text-gray-400 ${
                isActive ? "underline" : ""
              }`
            }
          >
            Test
          </NavLink>
        </li>
      </ul>
    </nav>
  );
};

export default NavBar;
