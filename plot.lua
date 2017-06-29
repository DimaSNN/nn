--plot

require 'gnuplot'
gnuplot.figure(1)
gnuplot.title('original')
gnuplot.plot(output[1])

gnuplot.figure(2)
gnuplot.title('from nn')
gnuplot.plot(mlp:forward(input[1]))