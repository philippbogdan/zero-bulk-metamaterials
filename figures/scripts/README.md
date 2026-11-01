# Regenerate the figures

Run these commands from the repository root. `make_data.py` reads only the JSON evidence in `data/` and writes the eight inputs in [figures/data](../data/README.md). It takes seconds and needs only NumPy. `figures.py` writes PNG and SVG versions of the stiffness map, corner-gap comparison and wall-convergence chart directly to `figures/`.

```sh
python -m pip install -r requirements-figures.txt
python figures/scripts/make_data.py
MPLBACKEND=Agg python figures/scripts/figures.py
```

The plots use Arial with STIX Sans mathematical symbols. Install Arial to match the supplied typography; Matplotlib otherwise substitutes an available font. Package versions and fonts can affect image bytes. Selection and display transformations are saved in `figures/tmp/figures_info.json`.

`wallrender.py` traces the exact stored voxels. Its `DESIGN` settings fix the crop, cutaway, lighting, camera path and sampling. The wall image uses 192 samples per pixel at 2400 by 1350; the orbit uses 56 at 1280 by 720. These calculations take substantially longer than table generation. Numba compiles the tracing functions on their initial use.

```sh
python figures/scripts/wallrender.py hero --parts 2 --part 0
python figures/scripts/wallrender.py hero --parts 2 --part 1
python figures/scripts/wallrender.py assemble --parts 2
python figures/scripts/wallrender.py frames --start 0 --stop 144
python figures/scripts/wallrender.py encode --crf 18
```

Assembly writes `figures/wall.png` and removes the temporary image strips. Frame generation can be split into consecutive `--start` and `--stop` ranges, with the stop excluded. Frames stay in `figures/tmp/frames/`; encoding writes `figures/wall.mp4`. Add `--clean` to the encoding command to remove those PNG frames afterwards.

Install FFmpeg separately with libx264 available on `PATH`. The encoding command is:

```sh
ffmpeg -y -loglevel error -framerate 24 \
  -i figures/tmp/frames/f_%04d.png \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \
  -profile:v high -level 4.1 -tune film \
  -movflags +faststart -an figures/wall.mp4
```

This produces the six-second H.264 orbit with 144 frames, 24 frames per second, yuv420p colour and faststart metadata. It moves the camera around the fixed geometry. The [geometry description](../../docs/finite-scale.md) explains the two views and orientation colours.

For the inline animation, convert the MP4 to a 128-colour GIF at 8 frames per second, then optimise it with a separately installed gifsicle:

```sh
ffmpeg -i figures/wall.mp4 -vf "fps=8,scale=1100:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];[b][p]paletteuse=dither=none" -loop 0 figures/wall-full.gif
gifsicle -O3 --lossy=80 figures/wall-full.gif -o figures/wall.gif
```

The reduced frame rate and lossy optimisation keep the supplied `wall.gif` under 10 MB: 8,634,889 bytes, 1100 by 619 pixels and 48 frames, looping continuously. `wall-full.gif` is the intermediate file; `wall.png` remains the high-resolution still and poster. See [dependency licences](../../DEPENDENCIES.md).
