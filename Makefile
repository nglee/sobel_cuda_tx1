#
# Author : Lee Namgoo
# E-Mail : lee.namgoo@sualab
#

INCLUDE_DIRS=-I$(HOME)/tegra_multimedia_api/include -I/usr/local/cuda-8.0/targets/aarch64-linux/include

LIB_DIRS=-L/usr/lib/aarch64-linux-gnu/tegra -L/usr/local/cuda-8.0/lib64

LIBS=-largus -lopencv_core -lopencv_highgui -lnvbuf_utils -lm

OBJS=

NVCC_OPTIONS=-arch=sm_53 -ccbin=g++ -std=c++11

all: main

main: main.cpp kernel.cu $(OBJS)
	nvcc $(NVCC_OPTIONS) $(INCLUDE_DIRS) $(LIB_DIRS) -o $@ $^ $(LIBS)

clean:
	rm main *.jpg *.bmp

distclean: clean
	rm cscope.out tags
