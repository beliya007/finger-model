#!/usr/bin/gnuplot
set terminal png enhanced font "Times New Roman, 18" size 1600,900
set output 'PublicCheck.png'

set xlabel "# of training epoch" font "Times New Roman, 22"
set ylabel "Detection rate (%)" font "Times New Roman, 22"

set xrange [0:10]
set yrange [0:100]
set xtics 0,2,10
set ytics 0,20,100
set grid lw 1 lc rgb "#ccccff"
set key bottom right font "Times New Roman, 20"

plot "publiccheck_data.txt" using 1:2 title 'Tampering' with linespoints lw 3 lc rgb '#dd2222' lt 2 pt 4 ps 2.5,\
"publiccheck_data.txt" using 1:3 title 'Validation w. Leakage' with linespoints lw 3 lc rgb '#0022cc' lt 2 pt 6 ps 2.5,\
"publiccheck_data.txt" using 1:4 title 'Validation w.o. Leakage' with linespoints lw 3 lc rgb '#000000' lt 2 pt 7 ps 2.5
