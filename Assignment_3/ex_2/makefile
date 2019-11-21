exercise_2a_nopin:
	nvcc -arch=sm_30 exercise_2a.cu -o exercise_2a_nopin.out -D PINNED=false

exercise_2a_pinned:
	nvcc -arch=sm_30 exercise_2a.cu -o exercise_2a_pinned.out -D PINNED=true

exercise_2b:
	nvcc -arch=sm_30 exercise_2b.cu -o exercise_2b.out

all: exercise_2a_nopin exercise_2a_pinned exercise_2b