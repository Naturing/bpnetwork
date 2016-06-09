#!/bin/bash

gnuplot -persist << EOF
set xrange [0:6]
set yrange [112:450]
set title "Data Compare"
plot "train" with linespoints linecolor 9 linewidth 2 pointtype 9 pointsize 2, "bp" with linespoints linecolor 3 linewidth 2 pointtype 6 pointsize 2, "bp-ga" with linespoints linecolor 2 linewidth 2 pointtype 7 pointsize 2, "bp-ga-sa" with linespoints linecolor 7 linewidth 2 pointtype 8 pointsize 2
EOF
