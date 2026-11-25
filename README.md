# Flume Interface

A flume in section, in the browser. Draw any shape in the flume (solid, porous, plants, or
sediment), shape the bed and lay down sediment, then run a river or a wave flume over it and
watch the water move.

It belongs to the same family as the [Sediment Interface](https://bradcantrell.github.io/sediment-interface/).
The water is computed on the GPU with WebGPU, so it needs a browser with WebGPU (recent Chrome
or Edge, or Safari 26 and later).

## What it does now (stage 3)

- A 2D section of a lab flume (20 m by 1 m) or a field reach (400 m by 16 m), at draft, normal, or fine grid resolution.
- Drawing tools are a brush, line, polygon, rectangle, ellipse, eraser, and a bed profile editor. Shapes stay editable (select, move, resize, reorder, delete, and undo).
- Materials are solid, porous, plants, and three grain classes of sediment. Porous areas and plants fill only water cells, so they stand on the bed.
- The water is a free-surface solver on the GPU (staggered grid, volume of fluid, and multigrid pressure with the free surface placed inside the cell). Momentum is advected conservatively in the bulk of the flow, so hydraulic jumps take the right depths.
- Porous areas slow the flow with Darcy-Forchheimer drag (van Gent coefficients), and plants with stem drag (half of Cd a u squared). Turbulence is a Smagorinsky eddy viscosity, and the bed and walls have log-law friction set by the grain size.
- A river flume (set discharge in, tailwater level out) or a wave flume. The wave paddle is a piston set by linear wavemaker theory and can absorb waves coming back from the flume, so reflections from a structure do not build up. The far end is an absorbing beach or a reflecting wall.
- Scenes save to and open from a `scene.json` file, and the last scene is kept in the browser.
- **OpenFOAM export.** The flume you drew becomes a 2D OpenFOAM case (OpenFOAM 12 or later, `foamRun` with `incompressibleVoF`), with the same porous and plant drag, probes of the water surface along the flume, and k-epsilon turbulence with rough walls for a river or laminar flow for waves (k-epsilon is the choice when waves break).

Sediment transport comes in stage 4.

## How it compares

The model is checked against textbook hydraulics and against OpenFOAM runs of its own exported cases.

| Case | Browser | Reference |
|---|---|---|
| Small waves, wavelength | 2.836 m | 2.836 m (linear theory) |
| Waves from the paddle, 1 to 2 wavelengths out | 5.9 to 6.0 cm | 6.0 cm (linear wavemaker theory) |
| Reflection from the absorbing far end | 3% | |
| Waves against a wall, with paddle absorption | incident wave holds at 3.7 to 3.9 cm | grows to 7.6 cm without absorption |
| Waves over a submerged bar (3 cm, not breaking) | 5.3 mm RMS difference along the flume | OpenFOAM, laminar |
| Broad-crested weir, energy over the crest / 1.5 critical depth | 1.06 | 1.10 (OpenFOAM), 1.03 to 1.11 for discharge coefficients of 0.95 to 0.85 |
| Hydraulic jump, depth after the jump | 180 mm | 188 mm (Belanger) |
| Porous block, head drop | 33.3 mm | 33.9 mm (OpenFOAM) |
| Plant patch, head drop | 21.7 mm | 21.7 mm (half of Cd a u squared) |

Known limits. Waves lose a little height with distance (about 7% over 11 m at 2 cm cells, half that at 1 cm), and 4 cm waves at 2 cm cells run 1.5 to 3% longer than linear theory. Behind a bar, the shorter harmonic waves are damped more than in OpenFOAM, so the wave height there varies less along the flume.

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
