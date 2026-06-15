# Eval report assets (vendored, embedded inline)

`evals/report.py` inlines these into every generated report so the HTML stays
**fully self-contained and offline** — a hard requirement because the report
contains PHI and must never fetch external resources at view time.

| File | What | Source |
|------|------|--------|
| `echarts.min.js` | Apache ECharts 5 UMD bundle (charts) | `npm i echarts@5` → `node_modules/echarts/dist/echarts.min.js` |
| `fonts.css` | `@font-face` rules with base64-embedded IBM Plex Mono + Sans (subset to Latin + common punctuation) | `npm i @ibm/plex-mono @ibm/plex-sans`, subset with `fonttools` |

## Regenerating the fonts

```bash
npm i @ibm/plex-mono @ibm/plex-sans
UNI="U+0020-007E,U+00A0-00FF,U+2013,U+2014,U+2018-201A,U+201C-201E,U+2022,U+2026,U+2192,U+2212,U+25CF,U+00B7"
for f in PlexMono-Regular PlexMono-SemiBold PlexSans-Regular PlexSans-SemiBold; do
  fam=$(echo $f | sed 's/-.*//'); wt=$(echo $f | sed 's/.*-//')
  python -m fontTools.subset "node_modules/@ibm/${fam,,}/fonts/complete/woff2/IBM${f/-/-}.woff2" \
    --unicodes="$UNI" --layout-features=kern,liga,calt,tnum --flavor=woff2 --output-file="sub/$f.woff2"
done
# then base64-embed each sub/*.woff2 into @font-face src:url(data:font/woff2;base64,...)
```

Keep the subset small: these bytes ship in every report. If you add glyphs to
the UI (new symbols, box-drawing, etc.), extend `UNI` and regenerate.
