# Flume Interface

A flume in section, in the browser. Draw any shape in the flume (solid, porous, plants, or
sediment), shape the bed and lay down sediment, then run a river or a wave flume over it and
watch the water move.

It belongs to the same family as the [Sediment Interface](https://bradcantrell.github.io/sediment-interface/).
The water is computed on the GPU with WebGPU, so it needs a browser with WebGPU (recent Chrome
or Edge, or Safari 26 and later).

## What it does now (stage 5)

- A 2D section of a lab flume (20 m by 1 m) or a field reach (400 m by 16 m), at draft, normal, or fine grid resolution.
- Drawing tools are a brush, line, polygon, rectangle, ellipse, eraser, and a bed profile editor. Shapes stay editable (select, move, resize, reorder, delete, and undo).
- Materials are solid, porous, plants, and three grain classes of sediment. Porous areas and plants fill only water cells, so they stand on the bed.
- The water is a free-surface solver on the GPU (staggered grid, volume of fluid, and multigrid pressure with the free surface placed inside the cell). Momentum is advected conservatively in the bulk of the flow, so hydraulic jumps take the right depths.
- Porous areas slow the flow with Darcy-Forchheimer drag (van Gent coefficients), and plants with stem drag (half of Cd a u squared). Turbulence is a Smagorinsky eddy viscosity, and the bed and walls have log-law friction set by the grain size.
- A river flume (set discharge in, tailwater level out) or a wave flume. The wave paddle is a piston set by linear wavemaker theory and can absorb waves coming back from the flume, so reflections from a structure do not build up. The far end is an absorbing beach or a reflecting wall.
- **A moving bed.** Sediment moves as bedload (Meyer-Peter and Mueller, from the bed shear stress) and in suspension (carried by the flow, settling at each grain's fall velocity, picked up from the bed by van Rijn's reference concentration). The bed follows the Exner equation, slopes steeper than 32 degrees slump, and sediment under a solid or below a fixed sill stays put. Three grain classes sort themselves. The top layer of the bed changes its mix with what leaves and arrives, so a sediment-starved river armours its bed with gravel. Deposits are drawn banded by when they were laid down. A bed speed-up factor lets the bed change faster than the water, and a river brings in sediment at equilibrium or runs clear.
- **Flexible plants.** Plants have a stiffness (flexural rigidity EI). They bend in the flow as Luhar and Nepf (2011) describe, drag only over their bent length, and sway with each wave. A large stiffness makes them rigid. The OpenFOAM export treats them as rigid.
- **Gauges.** Place water-level gauges anywhere along the flume. Each shows its level over the last 30 seconds, its mean level and wave height, and with two or more gauges each one's wave height relative to the first (wave transmission behind a reef or breakwater).
- **Dye.** Click or drag in the water to release dye and watch the flow carry it.
- **The look.** Beds and shapes are drawn with smooth outlines, suspended sediment shows as a sandy cloud, and foam gathers where fast water meets the surface.
- **Images and video.** Save the view as a PNG, or record it as a video for a presentation or a reel.
- Scenes save to and open from a `scene.json` file, and the last scene is kept in the browser.
- **OpenFOAM export.** The flume you drew becomes a 2D OpenFOAM case (OpenFOAM 12 or later, `foamRun` with `incompressibleVoF`), with the same porous and plant drag, probes of the water surface along the flume, and k-epsilon turbulence with rough walls for a river or laminar flow for waves (k-epsilon is the choice when waves break).

Sediment does not move in the OpenFOAM export (stock OpenFOAM has no morphodynamics). The gauges are exported as probe points.

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
| Bedload rate | Meyer-Peter and Mueller for the computed bed shear, exactly | |
| Still water over sand | bed does not move | |
| Sediment in a closed wave flume | conserved to 0.03% | |
| A 45 degree mound | slumps to 32 degrees | angle of repose |
| Settling column, sand left at t = h / (2 ws) | 50.0% | 50% |
| Suspended profile in a steady river | 11 to 20% above Rouse | Rouse profile |
| Clear water over a 50/50 sand and gravel bed | surface coarsens to 76% gravel | armouring |
| Flexible meadow, effective length | 0.211 of its length | 0.211 (Luhar and Nepf) |

Known limits. The suspended profile carries more sand high in the water at fine resolution, where the flow resolves eddies that mix on top of the modelled mixing. Ripples and dunes are smaller than the cells and are not formed. Waves lose a little height with distance (about 7% over 11 m at 2 cm cells, half that at 1 cm), and 4 cm waves at 2 cm cells run 1.5 to 3% longer than linear theory. Behind a bar, the shorter harmonic waves are damped more than in OpenFOAM, so the wave height there varies less along the flume.

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
