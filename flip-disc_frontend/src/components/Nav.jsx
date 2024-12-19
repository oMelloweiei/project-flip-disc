import React, { useState } from "react";
import { Link } from "react-router-dom";
import logo from "../assets/images/Logo_White.webp";

function Nav() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <nav className="w-full bg-black px-4 sm:px-8 py-3 relative z-50">
      <div className="flex items-center justify-between">
        <div className="hover:text-gray-400 text-white text-xl font-sans">
          <Link to="/">
            <img className="object-cover w-30 h-12" src={logo} alt="Logo" />
          </Link>
        </div>

        <div className="md:hidden">
          <button className="text-white" onClick={toggleMenu}>
            <svg
              fill="none"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              viewBox="0 0 24 24"
              className="w-6 h-6"
            >
              <path d="M4 6h16M4 12H16M4 18h16"></path>
            </svg>
          </button>
        </div>

        <ul className="hidden md:flex space-x-8">
          {["Home", "About", "Contact", "Test"].map((item) => (
            <li key={item} className="relative group">
              <Link
                className="text-white text-lg font-serif relative z-10 hover:text-gray-400"
                to={item === "Home" ? "/" : `/${item.toLowerCase()}`} // Set '/' for Control
              >
                {item}
              </Link>
              <span className="absolute left-0 bottom-0 h-0.5 w-0 bg-gray-400 transition-all duration-300 group-hover:w-full"></span>
            </li>
          ))}
        </ul>
      </div>

      {isMenuOpen && (
        <div className="absolute right-4 mt-2 w-48 bg-white rounded-md shadow-lg overflow-hidden">
          <ul className="flex flex-col space-y-4 p-4">
            {["Control", "About", "Contact", "Test"].map((item) => (
              <li key={item}>
                <Link
                  className="block px-4 py-2 text-black text-lg font-serif hover:bg-gray-100 rounded-md"
                  to={item === "Control" ? "/" : `/${item.toLowerCase()}`}
                  onClick={() => setIsMenuOpen(false)}
                >
                  {item}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      )}
    </nav>
  );
}

export default Nav;
