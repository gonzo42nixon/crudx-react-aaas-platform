"use strict";

function isHelpRequest(input) {
  return String(input || "").trim() === "?";
}

function buildCamHelp(envelope, fallbackKey, fallbackAgentId) {
  const cam = envelope && envelope.data && typeof envelope.data === "object"
    ? envelope.data
    : envelope || {};
  const sections = [cam, cam.manifest, cam.okf].filter((value) => value && typeof value === "object");
  const meta = sections.map((section) => section.meta).find((value) => value && typeof value === "object") || {};
  const name = firstText(
    ...sections.map((section) => section.name),
    meta.name,
    ...sections.map((section) => section.label),
    ...sections.map((section) => section.agent_id),
    fallbackAgentId,
    fallbackKey,
    "CRUDX Agent"
  );
  const purpose = firstText(
    ...sections.map((section) => section.description),
    meta.description,
    ...sections.map((section) => section.purpose),
    "This agent provides help for its configured CRUDX role."
  );
  const examples = uniqueStrings(sections.flatMap((section) => [
    ...(Array.isArray(section.example_prompts) ? section.example_prompts : []),
    ...(Array.isArray(section.examples) ? section.examples : [])
  ])).filter((value) => value !== "?");
  const helpCommand = sections
    .map((section) => section.help_command)
    .find((value) => value && typeof value === "object") || {};
  const infoKey = firstText(
    ...sections.map((section) => section.info_key),
    ...sections.map((section) => section.agent_info_key),
    meta.info_key
  );
  const helpUrl = firstText(helpCommand.help_url, infoKey ? markdownHelpUrl(infoKey) : "");
  const lines = [`# ${name}`, "", purpose];

  if (examples.length) {
    lines.push("", "## Example questions", "");
    examples.forEach((example, index) => lines.push(`${index + 1}. ${example}`));
  } else {
    lines.push("", "No example questions are configured in this CAM yet.");
  }

  if (helpUrl) lines.push("", `[Open Help](${helpUrl})`);

  return {
    name,
    purpose,
    examples,
    help_url: helpUrl || null,
    answer: lines.join("\n")
  };
}

function markdownHelpUrl(infoKey) {
  const encoded = encodeURIComponent(infoKey);
  return `https://europe-west3-crudx-e0599.cloudfunctions.net/calendarApi/v1/launch?key=${encoded}&data=${encoded}&app=CRUDX-CORE-APP-MARKD`;
}

function firstText(...values) {
  for (const value of values) {
    const text = String(value || "").trim();
    if (text) return text;
  }
  return "";
}

function uniqueStrings(values) {
  const seen = new Set();
  const result = [];
  for (const value of values) {
    const text = String(value || "").trim();
    if (!text || seen.has(text)) continue;
    seen.add(text);
    result.push(text);
  }
  return result;
}

module.exports = { buildCamHelp, isHelpRequest };
