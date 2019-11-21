edge:
	nvcc -O3 -arch=sm_30 hw3_ex1.cu tools.cu -o hw3_ex1.out

run:
	optirun ./hw3_ex1.out ./images/hw3.bmp
	optirun ./hw3_ex1.out ./images/hk.bmp
	optirun ./hw3_ex1.out ./images/nyc.bmp
	optirun ./hw3_ex1.out ./images/rome.bmp

display:
	# display -resize 1280x720 images/hw3_result_1.bmp
	# montage -tile 2x1 -crop 320x180+512+512 -geometry 640x360 images/hw3_result_1.bmp images/hw3_result_2.bmp images/hw3_result_2_comp.jpg
	# display images/hw3_result_2_comp.jpg
	montage -border 0 -geometry 640x640 -tile 3x2 images/hw3_result_cpu_1.bmp images/hw3_result_cpu_2.bmp images/hw3_result_cpu_3.bmp images/hw3_result_gpu_1.bmp images/hw3_result_gpu_2.bmp images/hw3_result_gpu_3.bmp images/hw3_montage.bmp
	display images/hw3_montage.bmp
	# display -resize 1280x720 images/hw3_result_3.bmp

compare:
	diff images/hw3_result_cpu_1.bmp images/hw3_result_gpu_1.bmp && echo "Compare successful"
	diff images/hw3_result_cpu_2.bmp images/hw3_result_gpu_2.bmp && echo "Compare successful"
	diff images/hw3_result_cpu_3.bmp images/hw3_result_gpu_3.bmp && echo "Compare successful"

all: clean edge run compare display

clean:
	rm -f ./images/hw3_*