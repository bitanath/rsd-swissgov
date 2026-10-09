// XXX Only for `bun run dev` has no appended model and no build step, so point elthur at the repo's GGUF directly
import { dlopen } from "bun:ffi";
import { fileURLToPath } from "node:url";

const DEV_MODEL = (() => {
  try {
    return fileURLToPath(new URL("../../core/assets/apertus-v1.5-8b-text-Q5_K_M.gguf", import.meta.url));
  } catch {
    return "";
  }
})();

function setEnv(name: string, value: string): void {
  const lib = process.platform === "darwin" ? "/usr/lib/libSystem.B.dylib" : "libc.so.6";
  const { setenv } = dlopen(lib, {
    setenv: { args: ["cstring", "cstring", "i32"], returns: "i32" },
  }).symbols;
  
  (setenv as unknown as (n: string, v: string, o: number) => number)(name, value, 0); // FIXME overwrite = 0: an explicit ELTHUR_MODEL always wins.
}

if (DEV_MODEL) {
  try {
    setEnv("ELTHUR_MODEL", DEV_MODEL);
  } catch {
    console.log("Errored out while tring to setenv")
  }
}
