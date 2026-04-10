import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const defaultOutputDir = path.join(repoRoot, "artifacts", "llm-context");

const targets = [
  {
    key: "brain",
    source: "sim-core/src/brain.rs",
    output: "brain_context.rs",
    title: "sim-core brain module",
  },
  {
    key: "genome",
    source: "sim-core/src/genome.rs",
    output: "genome_context.rs",
    title: "sim-core genome module",
  },
  {
    key: "turn",
    source: "sim-core/src/turn.rs",
    output: "turn_context.rs",
    title: "sim-core turn module",
  },
  {
    key: "evaluation",
    source: "sim-evaluation/src/main.rs",
    output: "evaluation_context.rs",
    title: "sim-evaluation module",
  },
];

async function main() {
  const outputDirArg = process.argv[2];
  const outputDir = path.resolve(repoRoot, outputDirArg ?? defaultOutputDir);

  await mkdir(outputDir, { recursive: true });

  const writtenFiles = [];
  for (const target of targets) {
    const sections = await collectModuleTree(target.source);
    const outputPath = path.join(outputDir, target.output);
    const body = renderBundle(target, sections);
    await writeFile(outputPath, body, "utf8");
    writtenFiles.push(path.relative(repoRoot, outputPath));
  }

  process.stdout.write(
    `${writtenFiles.length} stitched context files written to ${path.relative(repoRoot, outputDir)}\n`,
  );
  for (const writtenFile of writtenFiles) {
    process.stdout.write(`- ${writtenFile}\n`);
  }
}

async function collectModuleTree(entryRelativePath) {
  const visited = new Set();
  const sections = [];
  await visitModule(entryRelativePath, sections, visited);
  return sections;
}

async function visitModule(moduleRelativePath, sections, visited) {
  const normalizedPath = normalizeRelativePath(moduleRelativePath);
  if (visited.has(normalizedPath)) {
    return;
  }
  visited.add(normalizedPath);

  const absolutePath = path.join(repoRoot, normalizedPath);
  const content = await readFile(absolutePath, "utf8");
  sections.push({
    path: normalizedPath,
    content,
  });

  const childModules = findFileModuleDeclarations(content);
  for (const childModule of childModules) {
    const childPath = await resolveChildModulePath(normalizedPath, childModule);
    await visitModule(childPath, sections, visited);
  }
}

function findFileModuleDeclarations(source) {
  const moduleNames = [];
  const moduleDeclarationPattern = /^\s*mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;/gm;
  for (const match of source.matchAll(moduleDeclarationPattern)) {
    moduleNames.push(match[1]);
  }
  return moduleNames;
}

async function resolveChildModulePath(parentRelativePath, moduleName) {
  const parentDirectory = path.dirname(parentRelativePath);
  const parentBaseName = path.basename(parentRelativePath, path.extname(parentRelativePath));

  const nestedModulePath = path.join(parentDirectory, parentBaseName, `${moduleName}.rs`);
  const siblingModulePath = path.join(parentDirectory, `${moduleName}.rs`);

  if (await fileExists(nestedModulePath)) {
    return nestedModulePath;
  }
  if (await fileExists(siblingModulePath)) {
    return siblingModulePath;
  }
  throw new Error(
    `Unable to resolve module '${moduleName}' declared from '${parentRelativePath}'`,
  );
}

function normalizeRelativePath(relativePath) {
  return path.normalize(relativePath).replaceAll(path.sep, "/");
}

async function fileExists(relativePath) {
  try {
    await access(path.join(repoRoot, relativePath));
    return true;
  } catch {
    return false;
  }
}

function renderBundle(target, sections) {
  const headerLines = [
    `# ${target.title} stitched context`,
    `# Source root: ${target.source}`,
    `# Included files: ${sections.length}`,
    "",
  ];
  const renderedSections = sections.map(({ path: sourcePath, content }) => {
    const trimmedContent = content.endsWith("\n") ? content : `${content}\n`;
    return [
      `// ===== BEGIN FILE: ${sourcePath} =====`,
      trimmedContent.trimEnd(),
      `// ===== END FILE: ${sourcePath} =====`,
      "",
    ].join("\n");
  });

  return `${headerLines.join("\n")}${renderedSections.join("")}`;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
