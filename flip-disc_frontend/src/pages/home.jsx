import React from "react";
import MainBox from "../components/Mainbox";

function Home({ data, error }) {
  return (
    // The real one
    <div>
      <h1 className="text-3xl font-bold underline text-white">
        React to C++ Backend Connection
      </h1>
      {error && <p style={{ color: "red" }}>Error: {error}</p>}
      {data ? (
        <p style={{ color: "green" }}>Message from backend: {data.message}</p>
      ) : (
        <p>Loading...</p>
      )}
      <MainBox />
    </div>
  );
}

export default Home;
