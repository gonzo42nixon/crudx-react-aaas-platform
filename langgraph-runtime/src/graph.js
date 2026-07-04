"use strict";

const { createExternalLangGraph } = require("./server");

const graph = createExternalLangGraph();

module.exports = {
  graph
};
