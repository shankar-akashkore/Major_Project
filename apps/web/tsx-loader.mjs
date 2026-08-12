/**
 * A module loader that teaches `node --test` to read `.tsx`.
 *
 * Node 22 strips TypeScript *types* natively, which is what lets `lib/*.test.ts`
 * run with no framework and no build step. It does not transform JSX — `<div/>` is
 * a syntax error, not an annotation to erase — so the components were untestable
 * for the same reason they were cheap to write.
 *
 * The alternatives were a test framework with its own transform (jest/vitest, plus
 * a config file and a second module graph that can disagree with the one Next
 * builds) or this: about thirty lines that hand the file to the TypeScript compiler
 * that is already installed. Same trade as `adml.figures` drawing SVG rather than
 * installing matplotlib, and the same reason — this machine has under 20 GB free.
 *
 * Two hooks, doing the two things Next does for free at build time:
 *
 * - `resolve` maps the `@/` alias from `tsconfig.json`. Without it every component
 *   import of `@/lib/presentation.ts` fails, and the test would be measuring the
 *   loader rather than the component.
 * - `load` transpiles `.tsx` with `jsx: react-jsx`, which is what emits the
 *   `react/jsx-runtime` calls `renderToStaticMarkup` needs.
 *
 * Types are *erased*, not checked: `pnpm typecheck` is what checks them, and doing
 * it twice would make the tests slow at something already covered.
 */

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import ts from "typescript";

const ROOT = path.dirname(fileURLToPath(import.meta.url));

export async function resolve(specifier, context, next) {
  if (specifier.startsWith("@/")) {
    return next(pathToFileURL(path.join(ROOT, specifier.slice(2))).href, context);
  }
  // `next/link` is a CJS file with no export-map entry, so bare-specifier
  // resolution cannot find it. Pointed at the real file rather than replaced with
  // a stub `<a>`: a fake Link would render whatever the fake renders, and the
  // navigation markup is part of what these tests read.
  if (specifier.startsWith("next/") && !specifier.endsWith(".js")) {
    return next(`${specifier}.js`, context);
  }
  return next(specifier, context);
}

export async function load(url, context, next) {
  if (!url.endsWith(".tsx")) {
    const result = await next(url, context);
    // Node strips the types off a `.ts` file itself, then warns that it had to
    // reparse because the package has no `"type": "module"`. Saying the format
    // outright silences a warning printed twice per run; adding `"type": "module"`
    // to package.json would silence it too and would also make every `.js` config
    // file in this app ESM, which is a build change for a log line.
    if (url.endsWith(".ts") && result.format === "commonjs") {
      return { ...result, format: "module" };
    }
    return result;
  }
  const filename = fileURLToPath(url);
  const { outputText } = ts.transpileModule(await readFile(filename, "utf8"), {
    fileName: filename,
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
      jsx: ts.JsxEmit.ReactJSX,
      // Keeps the emitted `import "./ui.tsx"` specifiers exactly as written, so
      // they come back through `load` instead of being rewritten to `.js` paths
      // that do not exist on disk.
      verbatimModuleSyntax: false,
    },
  });
  return { format: "module", shortCircuit: true, source: outputText };
}
