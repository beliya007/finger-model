# Gnuplot script for Backdoor attack curve (with IVSF)

# =========================
# Terminal and output
# =========================
set terminal pngcairo enhanced font "Times New Roman,16" size 1200,900
set output "TrojanNN_attack_plot.png"


# =========================
# Margins
# =========================
set lmargin 10
set rmargin 5
set tmargin 3
set bmargin 4


# =========================
# Axis settings
# =========================
set xlabel "# of training steps" font "Times New Roman,20"

set ylabel "Detection rate (%)" \
    font "Times New Roman,20" \
    rotate by 90

set xrange [0:400]
set yrange [0:100]

set xtics 0,100,400 font "Times New Roman,16"
set ytics 0,10,100 font "Times New Roman,16"

set grid ytics lw 1 lc rgb "#cccccc"


# =========================
# Line styles
# =========================

# -------- "-1" group: DASHED --------
# dt 2 = dashed

# MISentry-1: green dashed
set style line 1 lc rgb '#009933' lw 4 dt 2

# SSF-1: black dashed
set style line 2 lc rgb '#000000' lw 4 dt 2

# PubCheck-1: blue dashed
set style line 3 lc rgb '#4477cc' lw 4 dt 2

# IVSF-1: red dashed
set style line 7 lc rgb '#ff0000' lw 4 dt 2


# -------- "-5" group: SOLID --------
# dt 1 = solid

# MISentry-5: green solid
set style line 4 lc rgb '#009933' lw 4 dt 1

# SSF-5: black solid
set style line 5 lc rgb '#000000' lw 4 dt 1

# PubCheck-5: blue solid
set style line 6 lc rgb '#4477cc' lw 4 dt 1

# IVSF-5: red solid
set style line 8 lc rgb '#ff0000' lw 4 dt 1


# =========================
# Legend
# =========================
set key right bottom font "Times New Roman,18"


# =========================
# Plot
# =========================
plot "TrojanNN_data.txt" using 1:2 title "MISentry-1" with lines linestyle 1, \
     "" using 1:3 title "SSF-1"      with lines linestyle 2, \
     "" using 1:4 title "PubCheck-1" with lines linestyle 3, \
     "" using 1:5 title "MISentry-5" with lines linestyle 4, \
     "" using 1:6 title "SSF-5"      with lines linestyle 5, \
     "" using 1:7 title "PubCheck-5" with lines linestyle 6, \
     "" using 1:8 title "VMIISF-1"     with lines linestyle 7, \
     "" using 1:9 title "VMIISF-5"     with lines linestyle 8


# =========================
# Figure caption
# =========================
set label "(c) Backdoor attack" \
    at 200,-8 \
    centre \
    font "Times New Roman,22"

replot