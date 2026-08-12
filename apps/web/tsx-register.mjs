/**
 * Registers `tsx-loader.mjs` for the test run.
 *
 * A separate one-line file because `--import` needs a module that performs the
 * registration, while `register()` needs the loader to be a different module: the
 * hooks run on their own thread and cannot be defined in the file that installs
 * them.
 */

import { register } from "node:module";

register("./tsx-loader.mjs", import.meta.url);
