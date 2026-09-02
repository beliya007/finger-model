#!/usr/bin/gnuplot
set terminal png enhanced font "Times New Roman, 18" size 1600,900
set output 'result.png'

set xlabel "# of leaked samples" font "Times New Roman, 22"
set ylabel "Detection rate (%)" font "Times New Roman, 22"

set xrange [0:30]
set yrange [45:105]
set xtics 0,10,50
set ytics 50,10,100
set grid lw 1 lc rgb "#ccccff"

set key top right font "Times New Roman, 22"

# lt 2 虚线，pt 6实心方块标记，匹配原图样式
plot "data.txt" using 1:2 title 'Validation w. Leakage' \
    with linespoints lw 4 lc rgb '#0000cc' lt 2 pt 6 ps 3
