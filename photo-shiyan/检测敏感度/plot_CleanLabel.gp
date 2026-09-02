# Gnuplot script for CleanLabel attack curve (with VMIISF)

set terminal pngcairo enhanced font "Times New Roman,16" size 1200,900
set output "CleanLabel_attack_plot.png"

set lmargin 10
set rmargin 5
set tmargin 3
set bmargin 4

set xlabel "# of training steps" font "Times New Roman,20"
set ylabel "Detection rate (%)" font "Times New Roman,20" rotate by 90

set xrange [0:400]
set yrange [0:100]
set xtics 0,100,400 font "Times New Roman,16"
set ytics 0,10,100 font "Times New Roman,16"
set grid ytics lw 1 lc rgb "#cccccc"

# ns=1 uses dashed lines; ns=5 uses solid lines.
set style line 1 lc rgb '#009933' lw 4 dt 2
set style line 2 lc rgb '#000000' lw 4 dt 2
set style line 3 lc rgb '#4477cc' lw 4 dt 2
set style line 4 lc rgb '#009933' lw 4 dt 1
set style line 5 lc rgb '#000000' lw 4 dt 1
set style line 6 lc rgb '#4477cc' lw 4 dt 1
set style line 7 lc rgb '#ff0000' lw 4 dt 2
set style line 8 lc rgb '#ff0000' lw 4 dt 1

set key right bottom font "Times New Roman,18"

plot "CleanLabel_data.txt" using 1:2 title "MISentry-1" with lines linestyle 1, \
     "" using 1:3 title "SSF-1" with lines linestyle 2, \
     "" using 1:4 title "PubCheck-1" with lines linestyle 3, \
     "" using 1:5 title "MISentry-5" with lines linestyle 4, \
     "" using 1:6 title "SSF-5" with lines linestyle 5, \
     "" using 1:7 title "PubCheck-5" with lines linestyle 6, \
     "" using 1:8 title "VMIISF-1" with lines linestyle 7, \
     "" using 1:9 title "VMIISF-5" with lines linestyle 8

set label "(d) CleanLabel attack" at 200,-8 centre font "Times New Roman,22"
replot