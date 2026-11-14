/**
 * The table of wire cases: values of NEMAR_LAUNCH_SPEC, each with the answer nemar-cli's contract gives.
 *
 * The launcher's decoder (launcher/nemar_launch/spec.py, `decode_wire`) mirrors nemar-cli's
 * `decodeLaunchSpecEnv` rule for rule. tests/launcher/test_wire_cases.py replays the table this script
 * writes through `decode_wire`, and tests/launcher/test_drift.py regenerates it from a checkout of
 * nemar-cli at the pinned commit and requires the bytes to be the committed file's, so the seven rules
 * cannot drift apart between the two repositories without a test failing on one side.
 *
 * Usage. Bun only. The checkout must be at the pinned commit (tests/contract/nemar-cli-pin.json) and
 * have its production dependencies, which the contract imports (zod); `--ignore-scripts` keeps the
 * install from running the checkout's own lifecycle scripts (husky among them):
 *
 *   (cd <nemar-cli checkout> && bun install --frozen-lockfile --ignore-scripts --production)
 *   bun run --no-install scripts/gen_wire_cases.ts <nemar-cli checkout> \
 *     > tests/launcher/fixtures/wire_cases.json
 *
 * Deterministic: no clock, no environment, no randomness (a fixed linear congruential generator makes
 * the pseudo-random texts), one row a line, ASCII only. Exit 2: usage or not a git checkout. Exit 3:
 * the dependencies are not installed there (checked first, so Bun can never auto-install other
 * versions of them, and the answer cannot depend on the network).
 *
 * A row is `{id, value, expect}`. `value` is a JSON string, except for the rows that stand for a
 * variable that is not text (JSON null is the unset variable). `expect` is one of
 *   {ok: true, text}                       the value is accepted, and this is the text it decodes to;
 *   {ok: true, text_length, text_sha256}   the same for a text over 4096 bytes, by its length and digest;
 *   {ok: false, words}                     the value is refused, with one of these words. One word, except
 *                                          for a value with a character outside ASCII that is over the
 *                                          bound in some unit (code units, code points or bytes): languages
 *                                          count a length differently, so both too_long and not_base64url
 *                                          conform, and the contract says to compare the word only for ASCII.
 */

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

type Answer = { ok: true; text: string } | { ok: false; reason: string };

interface Contract {
  decodeLaunchSpecEnv: (value: unknown) => Answer;
  encodeLaunchSpec: (spec: unknown) => string;
  encodeLaunchSpecEnv: (text: string) => string;
  LAUNCH_SPEC_MAX_BYTES: number;
  LAUNCH_SPEC_ENV_MAX_CHARS: number;
  LAUNCH_SPEC_ENV_REFUSALS: readonly string[];
  SHELL_INERT_ENV_ALPHABET: string;
}

type Expect =
  | { ok: true; text: string }
  | { ok: true; text_length: number; text_sha256: string }
  | { ok: false; words: string[] };

interface Row {
  id: string;
  value: unknown;
  expect: Expect;
}

// ---- the checkout and the contract --------------------------------------------------------------

const argument = process.argv[2];
if (argument === undefined) {
  process.stderr.write(
    "usage: bun run --no-install scripts/gen_wire_cases.ts <nemar-cli checkout>\n",
  );
  process.exit(2);
}
const checkout = resolve(argument);
const head = spawnSync("git", ["-C", checkout, "rev-parse", "HEAD"], { encoding: "utf8" });
if (head.status !== 0) {
  process.stderr.write(`not a git checkout: ${checkout}\n`);
  process.exit(2);
}
const commit = head.stdout.trim();
if (!existsSync(join(checkout, "node_modules", "zod", "package.json"))) {
  process.stderr.write(
    `DEPENDENCIES MISSING: run \`bun install --frozen-lockfile --ignore-scripts --production\` in ${checkout}\n`,
  );
  process.exit(3);
}
const contract = (await import(
  pathToFileURL(join(checkout, "shared", "contract", "compute-launch.ts")).href
)) as Contract;

const MAX = contract.LAUNCH_SPEC_ENV_MAX_CHARS;
const MAX_BYTES = contract.LAUNCH_SPEC_MAX_BYTES;
const ALPHABET = contract.SHELL_INERT_ENV_ALPHABET;
const WORDS = contract.LAUNCH_SPEC_ENV_REFUSALS;
const LONG_TEXT = 4096;

// ---- small helpers -------------------------------------------------------------------------------

const sha256 = (text: string): string => createHash("sha256").update(text, "latin1").digest("hex");
const hex = (code: number, width = 2): string => code.toString(16).padStart(width, "0");
/** The base64url value (no padding) of the bytes. */
const encode = (bytes: Uint8Array | number[]): string => Buffer.from(bytes).toString("base64url");
/** The base64url value of a text, one byte to a character (the texts here are ASCII or Latin-1). */
const encodeText = (text: string): string => Buffer.from(text, "latin1").toString("base64url");
function isAscii(value: string): boolean {
  for (let i = 0; i < value.length; i++) {
    if (value.charCodeAt(i) > 0x7f) return false;
  }
  return true;
}

/** A fixed linear congruential generator: the same numbers on every run and every platform. */
function sequence(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state >>> 8;
  };
}

/** Printable ASCII (0x20 to 0x7e) of the given length. */
function printable(next: () => number, length: number): string {
  let text = "";
  for (let i = 0; i < length; i++) text += String.fromCharCode(0x20 + (next() % 95));
  return text;
}

/** Whether a string is over the bound in any unit a language may count its length in. */
function overInSomeUnit(value: string): boolean {
  return Math.max(value.length, [...value].length, Buffer.byteLength(value, "utf8")) > MAX;
}

function expectation(value: unknown): Expect {
  const answer = contract.decodeLaunchSpecEnv(value);
  if (answer.ok) {
    if (answer.text.length > LONG_TEXT) {
      return { ok: true, text_length: answer.text.length, text_sha256: sha256(answer.text) };
    }
    return { ok: true, text: answer.text };
  }
  if (!WORDS.includes(answer.reason))
    throw new Error(`a word the contract does not list: ${answer.reason}`);
  if (typeof value === "string" && !isAscii(value) && overInSomeUnit(value)) {
    if (answer.reason !== "too_long" && answer.reason !== "not_base64url") {
      throw new Error(`an over-length non-ASCII value was refused as ${answer.reason}`);
    }
    return { ok: false, words: ["not_base64url", "too_long"] };
  }
  return { ok: false, words: [answer.reason] };
}

const rows: Row[] = [];
const seen = new Set<string>();

function add(id: string, value: unknown): void {
  if (seen.has(id)) throw new Error(`a row id is used twice: ${id}`);
  seen.add(id);
  rows.push({ id, value, expect: expectation(value) });
}

// ---- the golden vectors: built by the contract, and held to the digests it publishes -----------------

const image = {
  uri: `docker://ghcr.io/example/runtime@sha256:${"b".repeat(64)}`,
  digest: `sha256:${"b".repeat(64)}`,
  sif: `${"b".repeat(64)}.sif`,
};
const GOLDEN_SPEC = {
  job_id: "job_0123456789abcdef01234567",
  runtime: "python",
  entry: "run.py",
  args: ["--label", 'caf\u00e9 "q" \u{1F600}'],
  bundle: { sha256: "a".repeat(64), size: 1234 },
  datasets: [{ id: "nm000104", version: "v1.0.2" }],
  api_base: "https://api.example.test",
  image,
  env: { OMP_NUM_THREADS: "4" },
  resources: { gpus: 0 },
  limits: { max_files: 38000, max_total_bytes: 1073741824 },
  job_token: "t".repeat(43),
};
const URLSAFE_SPEC = {
  job_id: "job_0123456789abcdef01234567",
  runtime: "python",
  command: ["python3", "-c"],
  args: ["print($(id))", "`x`;&|<>", "a\\b", "\n\t", ">>>???~~~"],
  datasets: [],
  api_base: "https://api.example.test",
  image,
  env: {},
  resources: { gpus: 0 },
  limits: { max_files: 38000, max_total_bytes: 1073741824 },
  job_token: "t".repeat(43),
};
const goldenText = contract.encodeLaunchSpec(GOLDEN_SPEC);
const urlsafeText = contract.encodeLaunchSpec(URLSAFE_SPEC);
if (sha256(goldenText) !== "88fcf9a4d6e63ae7bbedc9be536989fd7f2c263bb2b1c2fb6cfd706f49efe857") {
  throw new Error("the golden text is not the published one");
}
if (sha256(urlsafeText) !== "501b92700038336d2579133ce14f3664bca06827f120be07d079ace09722f91e") {
  throw new Error("the second vector's text is not the published one");
}
const goldenEnv = contract.encodeLaunchSpecEnv(goldenText);
const urlsafeEnv = contract.encodeLaunchSpecEnv(urlsafeText);

// ---- the rows ------------------------------------------------------------------------------------------

// The cases the other rows are read against: the order of the rules, and the two sides of a boundary.
add("key-QQ-is-accepted", "QQ");
add("key-QR-is-A-with-spare-bits", "QR");
add("key-AB-is-NUL-with-spare-bits", "AB");
add("key-a-canonical-value-and-a-newline", "QQ\n");
add("key-padding-is-not-written", "QQ==");

// 1. not a string; JSON null is the variable that is not set
add("r1-null-the-unset-variable", null);
add("r1-number", 7);
add("r1-true", true);
add("r1-empty-array", []);
add("r1-array-holding-a-good-value", ["QUJD"]);
add("r1-empty-object", {});

// 2. empty
add("r2-empty", "");

// 3. the bound: 87,382 characters are the encoding of 64 KiB, and one more is too long
const capText = printable(sequence(3), MAX_BYTES);
add("r3-cap-ok-87382-characters", encodeText(capText));
add("r3-cap-plus-one-87383-characters", encodeText(`${capText} `));
add("r3-over-and-not-in-the-alphabet", "{".repeat(MAX + 1));
add("r3-over-and-length-1-mod-4", "A".repeat(MAX + 3));
// Over the bound in bytes (UTF-8) and not in code units or code points: a language that counts those
// answers not_base64url, one that counts bytes answers too_long, and both conform.
add("r3-non-ascii-over-in-bytes-only", `${"A".repeat(MAX - 1)}\u00e9`);

// 4. a character outside A-Za-z0-9-_
for (let code = 0; code < 0x80; code++) {
  const char = String.fromCharCode(code);
  if (!ALPHABET.includes(char)) add(`r4-ascii-0x${hex(code)}`, char);
}
for (const code of [0x00, 0x09, 0x0a, 0x0d, 0x20, 0x22, 0x27, 0x2b, 0x2f, 0x3d, 0x5c, 0x7f]) {
  add(`r4-after-QUJD-0x${hex(code)}`, `QUJD${String.fromCharCode(code)}`);
}
for (const code of [0x80, 0xe9, 0x2ff, 0xffff]) {
  add(`r4-non-ascii-U+${hex(code, 4)}`, String.fromCharCode(code));
}
add("r4-non-ascii-astral", "\u{1F600}");
add("r4-non-ascii-after-QQ", "QQ\u00e9");
add("r4-non-ascii-inside-abcd", "ab\u00e9d");
add("r4-padding-two-signs", "QQ==");
add("r4-padding-one-sign-too-few", "QQ=");
add("r4-padding-one-sign", "QUI=");
add("r4-padding-after-0-mod-4", "QUJD=");
add("r4-padding-four-signs-after-0-mod-4", "QUJD====");
add("r4-padding-only", "====");
add("r4-padding-on-the-golden-value", `${goldenEnv}==`);
add("r4-standard-plus", "Pj4+");
add("r4-standard-slash", "Pz8/");
add("r4-standard-both", "ab+/");
add("r4-standard-plus-in-a-good-length", "ab+d");
add("r4-standard-slash-in-a-good-length", "ab/d");
add(
  "r4-standard-form-of-the-second-vector",
  Buffer.from(urlsafeText, "latin1").toString("base64").replace(/=+$/, ""),
);
add("r4-QQ-and-a-carriage-return-and-a-newline", "QQ\r\n");
add("r4-QQ-and-a-space", "QQ ");
add("r4-QQ-and-a-tab", "QQ\t");
add("r4-QUI-and-a-newline", "QUI\n");
add("r4-a-leading-space", " QQ");
add("r4-a-space-inside", "Q Q");
add("r4-a-leading-newline", "\nabcd");
add("r4-a-newline-inside", "ab\ncd");
add("r4-a-carriage-return-and-a-newline-after-four", "abcd\r\n");
add("r4-a-tab-inside", "ab\tcd");
add("r4-raw-json-empty-object", "{}");
add("r4-raw-json-no-space", '{"a":1}');
add("r4-raw-json-one-brace", "{");
add("r4-raw-json-array", "[]");
add("r4-raw-json-the-golden-text", goldenText);
add("r4-raw-json-the-second-vectors-text", urlsafeText);

// 5. a length of 1 modulo 4, and the three that are not
for (const size of [1, 5, 9, 13, 101, 4001]) add(`r5-length-${size}-is-1-mod-4`, "A".repeat(size));
for (let size = 1; size <= 9; size++) {
  add(`r5-a-text-of-${size}-bytes`, encodeText("abcdefghi".slice(0, size)));
}
add("r5-a-good-length-and-three-NUL-bytes", "AAAA");

// 6. every byte alone, so every edge of 0x20 to 0x7e is a row, then a bad byte inside a longer text
for (let byte = 0; byte < 256; byte++) add(`r6-byte-0x${hex(byte)}`, encode([byte]));
for (const bad of [0x00, 0x0a, 0x1f, 0x7f, 0x80, 0xff]) {
  for (const at of [0, 29]) {
    const bytes = new Array<number>(30).fill(0x61);
    bytes[at] = bad;
    add(`r6-30-bytes-with-0x${hex(bad)}-at-${at}`, encode(bytes));
  }
}
add("r6-a-newline-at-the-end-of-a-document", encodeText('{"a":1}\n'));
add("r6-a-carriage-return-and-a-newline-at-the-end", encodeText('{"a":1}\r\n'));
add("r6-a-tab-in-a-document", encodeText('{"a":\t1}'));
add("r6-a-NUL-in-a-string", encodeText('{"a":"b\u0000c"}'));
add("r6-a-DEL-in-a-string", encodeText('{"a":"b\u007fc"}'));
add("r6-e-acute-as-two-utf-8-bytes", encode(Buffer.from('{"a":"\u00e9"}', "utf8")));
add("r6-e-acute-as-one-latin-1-byte", encodeText('{"a":"caf\u00e9"}'));

// 7. canonical: the spare bits of the last character take each value and only zero is the encoder's
for (const byte of [0x00, 0x41, 0x7e]) {
  const canonical = encode([byte]);
  const last = ALPHABET.indexOf(canonical[1]);
  for (let bits = 0; bits < 16; bits++) {
    add(`r7-one-byte-0x${hex(byte)}-spare-bits-${bits}`, canonical[0] + ALPHABET[last + bits]);
  }
}
for (const pair of ["AB"]) {
  const canonical = encodeText(pair);
  const last = ALPHABET.indexOf(canonical[2]);
  for (let bits = 0; bits < 4; bits++) {
    const name = `${hex(pair.charCodeAt(0))}${hex(pair.charCodeAt(1))}`;
    add(`r7-two-bytes-0x${name}-spare-bits-${bits}`, canonical.slice(0, 2) + ALPHABET[last + bits]);
  }
}

// 8. a value cut short: at a group boundary it is the canonical encoding of a printable prefix
for (const cut of [4, 8, 400, 1020])
  add(`r8-golden-cut-at-${cut}-is-0-mod-4`, goldenEnv.slice(0, cut));
for (const cut of [5, 1021]) add(`r8-golden-cut-at-${cut}-is-1-mod-4`, goldenEnv.slice(0, cut));
for (const cut of [402, 1018]) add(`r8-golden-cut-at-${cut}-is-2-mod-4`, goldenEnv.slice(0, cut));
for (const cut of [403, 1019]) add(`r8-golden-cut-at-${cut}-is-3-mod-4`, goldenEnv.slice(0, cut));

// 9. the two golden vectors
add("r9-the-golden-vector", goldenEnv);
add("r9-the-second-vector-with-both-url-safe-characters", urlsafeEnv);

// 10. pseudo-random texts: printable ones are accepted, bytes are mostly not
const texts = sequence(0x1234abcd);
for (let length = 1; length <= 8; length++) {
  add(`r10-printable-text-of-${length}-bytes`, encodeText(printable(texts, length)));
}
const noise = sequence(7);
for (let length = 1; length <= 4; length++) {
  add(`r10-random-bytes-${length}`, encode(Array.from({ length }, () => noise() & 0xff)));
}

// ---- the file -----------------------------------------------------------------------------------------------

/** JSON with every character from U+007F up as an escape, so the file is ASCII whatever the rows hold. */
const ascii = (value: unknown): string =>
  JSON.stringify(value).replace(
    /[\u007f-\uffff]/g,
    (char) => `\\u${char.charCodeAt(0).toString(16).padStart(4, "0")}`,
  );

const provenance = {
  repository: "nemarOrg/nemar-cli",
  commit,
  source: "shared/contract/compute-launch.ts",
  function: "decodeLaunchSpecEnv",
  generator: "scripts/gen_wire_cases.ts",
  regenerate:
    "bun run --no-install scripts/gen_wire_cases.ts <nemar-cli checkout at that commit> > tests/launcher/fixtures/wire_cases.json",
  alphabet: ALPHABET,
  max_chars: MAX,
  words: WORDS,
  rows: rows.length,
};

const output = [
  "{",
  `  "provenance": ${JSON.stringify(provenance, null, 2).split("\n").join("\n  ")},`,
  '  "cases": [',
  rows.map((row) => `    ${ascii(row)}`).join(",\n"),
  "  ]",
  "}",
  "",
].join("\n");

if (!isAscii(output)) throw new Error("the table is not ASCII");
process.stdout.write(output);
