# Flume Interface

A flume in section, in the browser. Draw any shape in the flume (solid, porous, plants, or
sediment), shape the bed and lay down sediment, then run a river or a wave flume over it and
watch the water move.

It belongs to the same family as the [Sediment Interface](https://bradcantrell.github.io/sediment-interface/).
The water is computed on the GPU with WebGPU, so it needs a browser with WebGPU (recent Chrome
or Edge, or Safari 26 and later).

## What it does now (stage 2)

- A 2D section of a lab flume (20 m by 1 m) or a field reach (400 m by 16 m), at draft, normal, or fine grid resolution.
- Drawing tools are a brush, line, polygon, rectangle, ellipse, eraser, and a bed profile editor. Shapes stay editable (select, move, resize, reorder, delete, and undo).
- Materials are solid, porous, plants, and three grain classes of sediment. Porous areas and plants fill only water cells, so they stand on the bed.
- The water is a free-surface solver on the GPU (staggered grid, volume of fluid, and multigrid pressure with the free surface placed inside the cell). Momentum is advected conservatively in the bulk of the flow, so hydraulic jumps take the right depths.
- Porous areas slow the flow with Darcy-Forchheimer drag (van Gent coefficients), and plants with stem drag (half of Cd a u squared). Turbulence is a Smagorinsky eddy viscosity, and the bed and walls have log-law friction set by the grain size.
- A river flume (set discharge in, tailwater level out) or a wave flume (piston paddle, absorbing far end).
- Scenes save to and open from a `scene.json` file, and the last scene is kept in the browser.
- **OpenFOAM export.** The flume you drew becomes a 2D OpenFOAM case (OpenFOAM 12 or later, `foamRun` with `incompressibleVoF`), with the same porous and plant drag, k-epsilon turbulence with rough walls, and probes of the water surface along the flume.

Sediment transport comes in stage 4.

## How it compares

The model is checked against textbook hydraulics and against OpenFOAM runs of its own exported cases.

| Case | Browser | Reference |
|---|---|---|
| Waves from the paddle, 1 to 2 wavelengths out | 5.9 to 6.1 cm | 6.0 cm (linear wavemaker theory) |
| Broad-crested weir, energy over the crest / 1.5 critical depth | 1.110 | 1.099 (OpenFOAM), about 1.11 for a discharge coefficient of 0.85 |
| Hydraulic jump, depth after the jump | 182 mm | 189 mm (Belanger) |
| Porous block, head drop | 33.3 mm | 33.9 mm (OpenFOAM) |
| Plant patch, head drop | 21.7 mm | 21.7 mm (half of Cd a u squared) |

## Run it

It is a static site with no build step. Serve the folder over http on localhost (WebGPU needs a
secure context) and open `index.html`.

    python3 -m http.server 8000
    # then open http://localhost:8000/

`dev/bench.html` times the GPU step on your computer.

## Checks

    npm install                    # dev only, installs Dawn WebGPU for Node
    node dev/interface_checks.js   # the interface, no GPU needed
    node dev/site_checks.js        # the OpenFOAM case writer, no GPU needed
    node dev/model_checks.js       # the model on a GPU (FLUME_ONLY=weir,jump runs a few)

To compare with OpenFOAM, write a test case with `node dev/openfoam_case.js <dir> weir|porous|wave`,
run its `Allrun`, then `node dev/openfoam_compare.js weir|porous|wave <dir>`.

With several GPUs, `FLUME_ADAPTER` set to part of a GPU's name picks that one.

## License

MIT, see [LICENSE](LICENSE).
