"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { buildCamHelp, isHelpRequest } = require("../help-route");

test("recognizes only the reserved help command", () => {
  assert.equal(isHelpRequest(" ? "), true);
  assert.equal(isHelpRequest("show ?"), false);
  assert.equal(isHelpRequest(""), false);
});

test("builds deterministic Markdown help from a CAM", () => {
  const result = buildCamHelp({
    agent_id: "test-agent",
    meta: { name: "Test Agent", description: "Tests the help route." },
    example_prompts: ["?", "First question?", "Second question?", "First question?"],
    info_key: "CRUDX-5C583-8X4U2-3D323",
    help_command: {
      input: "?",
      help_url: "https://example.test/help"
    },
    okf: {}
  }, "CRUDX-5C583-4U458-54RC3");

  assert.equal(result.name, "Test Agent");
  assert.deepEqual(result.examples, ["First question?", "Second question?"]);
  assert.match(result.answer, /^# Test Agent/m);
  assert.match(result.answer, /1\. First question\?/);
  assert.match(result.answer, /\[Open Help\]\(https:\/\/example\.test\/help\)/);
  assert.doesNotMatch(result.answer, /^\d+\. \?$/m);
});

test("derives a Markdown help URL from info_key", () => {
  const result = buildCamHelp({
    agent_id: "test-agent",
    description: "Test purpose.",
    examples: ["Example"],
    info_key: "CRUDX-5C583-8X4U2-3D323",
    okf: {}
  }, "CRUDX-5C583-4U458-54RC3");

  assert.match(result.help_url, /key=CRUDX-5C583-8X4U2-3D323/);
  assert.match(result.help_url, /app=CRUDX-CORE-APP-MARKD/);
});
