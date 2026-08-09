# Overleaf package

Copy `assets/` and `figures/` into the Overleaf project root, and either use
`rebuttal.tex` as-is or add these two lines to your existing file:

    \usepackage{subcaption}        % in the preamble
    \input{figures/figure_block}   % right after \maketitle

## Layout

Two rows across the full page width, at the top. Plots first, images second:
the plot row is shorter, so leading with it puts the densest quantitative
content directly under the opening paragraph.

    row 1   fig1 noise+failures (0.40)  fig4 unbolted (0.30)  fig5 unified (0.30)
    row 2   fig3 novel geometries (0.62)              fig2 end-to-end (0.38)

Figures occupy ~2.85 in of a ~9 in text block, leaving ~37 lines for prose.
Each PDF is generated at its final on-page size, so `width=\linewidth` neither
scales nor resamples it -- do not change the subfigure widths without
regenerating, or the type will no longer be at its intended point size.

## Files

    assets/fig{1..5}_*.pdf      vector figures, drawn at final size
    assets/fig{1..5}_*.png      same figures as 400 dpi raster

`\includegraphics` picks the PDF automatically (graphicx prefers it), so the
PNGs are there for slides, email and quick previews rather than for the build.
If you ever need to force one, name the extension explicitly.
    figures/fig{1..5}_*.tex     one subfigure each, with caption and label
    figures/figure_block.tex    composes the five into the two-row block

Labels: `fig:noise_failures`, `fig:end_to_end`, `fig:novel_geometries`,
`fig:unbolted`, `fig:unified`.

## Regenerating

    python plot_figures/rebuttal/make_final_figures.py --textwidth 5.5

`--textwidth` must match the template's `\textwidth`. Everything is sized
relative to it; at 6.9 in the tighter panels stop being marginal.

## Status of the numbers

Real: fig2 (16/20, 12/20 and the video frames), fig4, fig5, fig1's failure pie
(14 logged failures), and fig3's dimension panels (measured off the assets).

Placeholder: fig3's four rollout tiles, pending the plug and fork teachers
finishing training. Fig1's sweep is real but currently a SINGLE task
(Screw-Leg) -- the peg and beam sweep JSONs no longer exist on disk, so adding
those series needs the sweeps re-run.
