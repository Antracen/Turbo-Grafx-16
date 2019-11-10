# CPU EXECUTION

for PARTICLES in 20000 40000 60000 80000 100000
do
	echo $PARTICLES
	srun -n 1 ./exercise_3.out $PARTICLES 10000 1
done

# GPU EXECUTION

#for THREADS in 16 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256
#do
#	echo $THREADS
#	for PARTICLES in 20000 40000 60000 80000 100000
#	do
#		echo $PARTICLES
#		srun -n 1 ./exercise_3.out $PARTICLES 10000 $THREADS
#	done
#done
