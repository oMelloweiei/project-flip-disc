import React, { useState } from "react";
import cat from "../assets/images/nyancat.webp";
import excite from "../assets/images/excite.webp";
import dance from "../assets/images/dance.gif";
import cli from "../assets/images/gifs-on-cli.gif";
import mode3 from "../assets/images/mode-3.webp";
import wow from "../assets/images/Wow.webp";

function MainBox() {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [selectedMode, setSelectedMode] = useState("Mode 1");

  const modeImages = {
    "Mode 1": cat,
    "Mode 2": excite,
    "Mode 3": mode3,
    "Mode 4": dance,
    "Mode 5": cli,
    "Mode 6": wow,
  };

  const modeDescriptions = {
    "Mode 1":
      "Mode 1 is designed to provide optimal performance for task A. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
    "Mode 2":
      "Mode 2 is ideal for handling intensive processes with efficiency. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
    "Mode 3":
      "Mode 3 focuses on achieving balance between speed and quality. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
    "Mode 4":
      "Mode 4 introduces advanced features for power users. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
    "Mode 5":
      "Mode 5 is tailored for lightweight tasks and quick operations. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
    "Mode 6":
      "Mode 6 enhances security and reliability for sensitive operations. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
  };

  const toggleDropdown = () => {
    setIsDropdownOpen((prev) => !prev);
  };

  const handleModeSelect = (mode) => {
    setSelectedMode(mode);
    setIsDropdownOpen(false); // Close dropdown after selecting
  };

  return (
    <div className="relative flex-col w-11/12 max-w-screen-xl bg-white shadow-md rounded-md mx-auto my-10 mt-10 p-6 sm:p-8 min-h-[80vh] sm:min-h-[100vh] h-auto">
      <div className="flex justify-between items-center gap-4">
        <div className="flex items-end gap-4">
          <p className="text-lg sm:text-2xl font-medium">Preview Mode</p>
          <p id="modeName" className="text-base sm:text-xl text-gray-600">
            {selectedMode}
          </p>
        </div>

        {/* Dropdown button */}
        <div className="relative">
          <button
            type="button"
            className="inline-flex gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50"
            id="dropdownButton"
            aria-expanded={isDropdownOpen ? "true" : "false"}
            aria-haspopup="true"
            onClick={toggleDropdown}
          >
            Options
            <svg
              className="-mr-1 size-5 text-gray-400"
              viewBox="0 0 20 20"
              fill="currentColor"
              aria-hidden="true"
            >
              <path
                fillRule="evenodd"
                d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
                clipRule="evenodd"
              />
            </svg>
          </button>

          {/* Dropdown Box */}
          <div
            className={`absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg overflow-hidden transform transition-all duration-500 ease-in-out ${
              isDropdownOpen
                ? "opacity-100 scale-100"
                : "opacity-0 scale-95 pointer-events-none"
            }`}
          >
            <DropdownItem onModeSelect={handleModeSelect} />
          </div>
        </div>
      </div>

      {/* Dynamically change the image based on the selected mode */}
      <img
        className="object-cover w-full sm:h-96 sm:min-h-[65vh] rounded-md my-4"
        src={modeImages[selectedMode] || cat}
        alt={`Preview for ${selectedMode}`}
      />

      {/* Description Section */}
      <div className="mt-6">
        <p className="text-center text-3xl sm:text-5xl font-semibold mb-4">
          Description
        </p>
        <p className="text-center text-base sm:text-lg mx-6 sm:mx-12 my-4 text-gray-700 leading-relaxed sm:leading-loose">
          {modeDescriptions[selectedMode] ||
            "Select a mode to view its description. Lorem Ipsum is simply dummy text of the printing and typesetting industry."}
        </p>
      </div>
    </div>
  );
}

function DropdownItem({ onModeSelect }) {
  const modes = ["Mode 1", "Mode 2", "Mode 3", "Mode 4", "Mode 5", "Mode 6"];

  return (
    <ul className="flex flex-col space-y-4 p-4">
      {modes.map((mode) => (
        <li key={mode} className="group">
          <button
            onClick={() => onModeSelect(mode)}
            className="block px-4 py-2 text-black text-lg font-serif hover:bg-gray-100 rounded-md transition-all duration-300 text-left w-full"
          >
            {mode}
          </button>
        </li>
      ))}
    </ul>
  );
}

export default MainBox;
