# Flume Interface

A flume in section, in the browser. Draw any shape in the flume (solid, porous, plants, or
sediment), shape the bed and lay down sediment, then run a river or a wave flume over it and
watch the water move.

It belongs to the same family as the [Sediment Interface](https://bradcantrell.github.io/sediment-interface/).
The water is computed on the GPU with WebGPU, so it needs a browser with WebGPU (recent Chrome
or Edge, or Safari 26 and later).

## What it does now (stage 1)

- A 2D section of a lab flume (20 m by 1 m) or a field reach (400 m by 16 m), at draft, normal, or fine grid resolution.
- Drawing tools are a brush, line, polygon, rectangle, ellipse, eraser, and a bed profile editor. Shapes stay editable (select, move, resize, reorder, delete, and undo).
- Materials are solid, porous, plants, and three grain classes of sediment. Porous areas and plants fill only water cells, so they stand on the bed. They do not slow the water yet (drag arrives in stage 2).
- The water is a free-surface solver (staggered grid, volume of fluid, and multigrid pressure) with a river inflow and tailwater, or a piston wave paddle and an absorbing far end.
- Scenes save to and open from a `scene.json` file, and the last scene is kept in the browser.

Drag in porous areas and plants, turbulence and bed friction, sediment transport, and the OpenFOAM export come in the next stages.

## Run it

It is a static site with no build step. Serve the folder over http on localhost (WebGPU needs a
secure context) and open `index.html`.

    python3 -m http.server 8000
    # then open http://localhost:8000/

`dev/bench.html` times the GPU step on your computer.

## Checks

    npm install                    # dev only, installs Dawn WebGPU for Node
    node dev/interface_checks.js   # the interface, no GPU needed
    node dev/model_checks.js       # the model on a GPU

With several GPUs, `FLUME_ADAPTER` set to part of a GPU's name picks that one.

## License

MIT, see [LICENSE](LICENSE).
