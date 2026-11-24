# Flume Interface: project conventions
# Read by coding agents working in this directory. Notes about our own machines are in
# `AGENTS.local.md`, which git ignores. Read it too if it is there.

Static site, no build step. Three files load in order and share one script scope:

| File | Layer | What it is |
|---|---|---|
| `index.html` | interface | Canvases, view, drawing tools, shapes, bed, material raster, scene save/open, undo, sidebar. Knows nothing about how water moves. |
| `model.js` | model | The water, on the GPU (WebGPU): MAC grid, PLIC volume of fluid, MG-preconditioned CG pressure with a ghost-fluid free surface, river and wave boundaries, the water renderer. |
| `site.js` | site | What this repo does with the scene (stage 2: the OpenFOAM case export). |

Rules:

- **Nothing about water goes in `index.html`.** The interface talks to the model only through the
  MODEL EXTENSION POINT (comment near the end of the script in `index.html`, which is the reference
  for `registerModel` and for the raster and view contracts).
- **Shapes are the source of truth**, in metres. The raster is rebuilt from them (`rasterize()`),
  so resolution and scale changes lose nothing. Every edit ends in `sceneChanged()`.
- **The model must run in a browser and in Node** (Dawn, the `webgpu` npm package). No DOM in the
  model except inside `controls` and `syncControls`.
- **A frame never queues more than about 40 ms of GPU work** (a slow GPU must not trip the browser
  watchdog). A lost device is rebuilt (`M.recover`).
- Tests: `dev/interface_checks.js` (interface, stub model, no GPU), `dev/model_checks.js` (the model
  on a real GPU), `dev/browser_check.js` (the page in Chrome with a screenshot; use `?readback` when
  the browser cannot present a WebGPU canvas, as in headless Chrome).
- Checks that fail silently are worse than none: a check that can not run must fail.

Stages and design: see the design notes kept with the project (not in this repo).
