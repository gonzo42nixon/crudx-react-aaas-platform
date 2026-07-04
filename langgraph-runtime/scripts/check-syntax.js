"use strict";

const { spawnSync } = require("node:child_process");

for (const file of ["src/server.js", "src/graph.js", "src/graph.mjs"]) {
  const result = spawnSync(process.execPath, ["--check", file], { stdio: "inherit" });
  if (result.status !== 0) process.exit(result.status || 1);
}
