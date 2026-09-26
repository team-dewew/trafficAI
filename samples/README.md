# Samples

Put the four sample videos from the organizers here (`C3896.MP4`, `C3897.MP4`,
`C3902.MP4`, `C3905.MP4`, download links in the task's `Videos.pdf`). Video files
are git-ignored.

`previews/` holds the annotated sample videos shown on the website. They are
rendered with the same registration / perception / signal code as Part A:

```bash
python -m src.annotate --video samples/C3896.MP4 --out samples/previews/C3896_preview.mp4 \
    --events predictions_samples.json --stride 2 --width 960
```

The organizers confirmed that `camera.md` is not provided. The scene layout is
described in `docs/scene.md`.
