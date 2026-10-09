import { afterEach, beforeEach, describe, expect, it } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { formatModelLine, modelIdsChanged, modelsHeading, normalizeBaseUrl, readSettings, writeSettings } from "./index.ts";

describe("normalizeBaseUrl", () => {
  it("appends /v1 when missing", () => {
    expect(normalizeBaseUrl("http://localhost:8000")).toBe("http://localhost:8000/v1");
  });

  it("leaves an existing /v1 suffix alone", () => {
    expect(normalizeBaseUrl("http://localhost:8000/v1")).toBe("http://localhost:8000/v1");
  });

  it("strips trailing slashes before checking the suffix", () => {
    expect(normalizeBaseUrl("http://localhost:8000/v1/")).toBe("http://localhost:8000/v1");
    expect(normalizeBaseUrl("http://localhost:8000/")).toBe("http://localhost:8000/v1");
  });

  it("defaults to http:// for a bare host:port with no scheme", () => {
    expect(normalizeBaseUrl("localhost:11434")).toBe("http://localhost:11434/v1");
    expect(normalizeBaseUrl("192.168.1.50:8000")).toBe("http://192.168.1.50:8000/v1");
  });

  it("preserves an explicit https:// scheme", () => {
    expect(normalizeBaseUrl("https://my.server.com")).toBe("https://my.server.com/v1");
  });

  it("trims surrounding whitespace", () => {
    expect(normalizeBaseUrl("  localhost:11434  ")).toBe("http://localhost:11434/v1");
  });
});

describe("formatModelLine", () => {
  it("formats context window and max tokens in k, with no capability tags", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "some-model",
        contextWindow: 65536,
        maxTokens: 8192,
        reasoning: false,
        input: ["text"],
      }),
    ).toBe("  • some-model  (ctx 64k, max 8k)");
  });

  it("appends reasoning and vision tags when present", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "vision-model",
        contextWindow: 65536,
        maxTokens: 8192,
        reasoning: true,
        input: ["text", "image"],
      }),
    ).toBe("  • vision-model  (ctx 64k, max 8k, reasoning, vision)");
  });

  it("shows sub-1024 windows without a k suffix", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "tiny",
        contextWindow: 512,
        maxTokens: 256,
        reasoning: false,
        input: ["text"],
      }),
    ).toBe("  • tiny  (ctx 512, max 256)");
  });

  it("prefixes a checkmark when loaded is true", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "m",
        contextWindow: 4096,
        maxTokens: 2048,
        reasoning: false,
        input: ["text"],
        loaded: true,
      }),
    ).toBe("  • ✓ m  (ctx 4k, max 2k)");
  });

  it("prefixes a hollow circle when loaded is false", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "m",
        contextWindow: 4096,
        maxTokens: 2048,
        reasoning: false,
        input: ["text"],
        loaded: false,
      }),
    ).toBe("  • ○ m  (ctx 4k, max 2k)");
  });

  it("omits the loaded prefix entirely when loaded is unknown", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "m",
        contextWindow: 4096,
        maxTokens: 2048,
        reasoning: false,
        input: ["text"],
      }),
    ).toBe("  • m  (ctx 4k, max 2k)");
  });

  it("shows size and quantization when present, in order before capability tags", () => {
    expect(
      formatModelLine({
        id: "m1",
        name: "m",
        contextWindow: 4096,
        maxTokens: 2048,
        reasoning: true,
        input: ["text", "image"],
        sizeBytes: 4912898304,
        quantization: "Q4_K_M",
      }),
    ).toBe("  • m  (ctx 4k, max 2k, 4.6G, Q4_K_M, reasoning, vision)");
  });
});

describe("modelIdsChanged", () => {
  const baseModel = {
    id: "m1",
    name: "m1",
    contextWindow: 4096,
    maxTokens: 2048,
    reasoning: false,
    input: ["text"] as ("text" | "image")[],
  };

  it("is false when the same single model is refreshed with new metadata", () => {
    expect(modelIdsChanged([baseModel], [{ ...baseModel, contextWindow: 8192 }])).toBe(false);
  });

  it("is false when the same set of models comes back in a different order", () => {
    const a = { ...baseModel, id: "a" };
    const b = { ...baseModel, id: "b" };
    expect(modelIdsChanged([a, b], [b, a])).toBe(false);
  });

  it("is true when the model count changes", () => {
    const a = { ...baseModel, id: "a" };
    const b = { ...baseModel, id: "b" };
    expect(modelIdsChanged([a], [a, b])).toBe(true);
  });

  it("is true when a same-count refresh swaps in a different model id", () => {
    const a = { ...baseModel, id: "a" };
    const c = { ...baseModel, id: "c" };
    expect(modelIdsChanged([a], [c])).toBe(true);
  });

  it("is false for two empty lists", () => {
    expect(modelIdsChanged([], [])).toBe(false);
  });
});

describe("modelsHeading", () => {
  const baseModel = {
    id: "m1",
    name: "m1",
    contextWindow: 4096,
    maxTokens: 2048,
    reasoning: false,
    input: ["text"] as ("text" | "image")[],
  };

  it("adds the loaded-state legend when at least one model reports it", () => {
    expect(modelsHeading([{ ...baseModel, loaded: true }])).toBe(
      "Models:  (✓ = loaded in memory, ○ = will be loaded on first message)",
    );
    expect(modelsHeading([{ ...baseModel }, { ...baseModel, loaded: false }])).toBe(
      "Models:  (✓ = loaded in memory, ○ = will be loaded on first message)",
    );
  });

  it("omits the legend when no model reports loaded state", () => {
    expect(modelsHeading([baseModel])).toBe("Models:");
    expect(modelsHeading([])).toBe("Models:");
  });
});

describe("localllm.json persistence", () => {
  let dir: string;

  const server = {
    id: "abc123",
    name: "Test server",
    baseUrl: "http://localhost:8000/v1",
    apiKey: "",
    apiType: "omlx" as const,
    models: [
      {
        id: "m1",
        name: "m1",
        contextWindow: 4096,
        maxTokens: 2048,
        reasoning: false,
        input: ["text"] as ("text" | "image")[],
      },
    ],
  };

  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "localllm-test-"));
  });

  afterEach(() => {
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it("returns empty settings when nothing is configured", () => {
    expect(readSettings(dir)).toEqual({ servers: [] });
  });

  it("writes localllm.json only — settings.json is never created or touched", () => {
    writeSettings({ servers: [server] }, dir);
    expect(JSON.parse(fs.readFileSync(path.join(dir, "localllm.json"), "utf8"))).toEqual({
      servers: [server],
    });
    expect(fs.existsSync(path.join(dir, "settings.json"))).toBe(false);
  });

  it("migrates the legacy localllm key from settings.json, leaving other keys intact", () => {
    fs.writeFileSync(
      path.join(dir, "settings.json"),
      JSON.stringify({
        defaultModel: "anthropic/claude-sonnet-4-5",
        localllm: { servers: [server] },
      }),
    );

    expect(readSettings(dir)).toEqual({ servers: [server] });

    const migrated = JSON.parse(fs.readFileSync(path.join(dir, "localllm.json"), "utf8"));
    expect(migrated).toEqual({ servers: [server] });

    const settings = JSON.parse(fs.readFileSync(path.join(dir, "settings.json"), "utf8"));
    expect(settings).toEqual({ defaultModel: "anthropic/claude-sonnet-4-5" });
  });

  it("reads localllm.json without touching settings.json once migrated", () => {
    fs.writeFileSync(
      path.join(dir, "settings.json"),
      JSON.stringify({ localllm: { servers: [server] } }),
    );
    expect(readSettings(dir)).toEqual({ servers: [server] });

    // Second read must come from localllm.json; the (now clean) settings.json
    // must not change.
    const before = fs.readFileSync(path.join(dir, "settings.json"), "utf8");
    expect(readSettings(dir)).toEqual({ servers: [server] });
    expect(fs.readFileSync(path.join(dir, "settings.json"), "utf8")).toBe(before);
  });

  it("falls back to the legacy key when localllm.json is corrupt", () => {
    fs.writeFileSync(path.join(dir, "localllm.json"), "{ not json");
    fs.writeFileSync(
      path.join(dir, "settings.json"),
      JSON.stringify({ localllm: { servers: [server] } }),
    );
    expect(readSettings(dir)).toEqual({ servers: [server] });
    expect(fs.readFileSync(path.join(dir, "localllm.json"), "utf8"))
      .toBe(JSON.stringify({ servers: [server] }, null, 2));
  });

  it("does not migrate a legacy entry without a servers array", () => {
    fs.writeFileSync(
      path.join(dir, "settings.json"),
      JSON.stringify({ localllm: { somethingElse: true } }),
    );
    expect(readSettings(dir)).toEqual({ servers: [] });
    expect(fs.existsSync(path.join(dir, "localllm.json"))).toBe(false);
    expect(JSON.parse(fs.readFileSync(path.join(dir, "settings.json"), "utf8"))).toEqual({
      localllm: { somethingElse: true },
    });
  });
});
