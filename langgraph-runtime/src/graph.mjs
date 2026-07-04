import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createExternalLangGraph } = require("./server.js");

export const graph = createExternalLangGraph();
