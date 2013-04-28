# Makefile for Modules

include ../../make.sys

# location of needed modules
MODFLAGS= $(MOD_FLAG)../iotk/src $(MOD_FLAG)../../Modules $(MOD_FLAG)/../../ELPA/ $(MOD_FLAG).

MODGPUOBJS = \
cuda_mem_alloc.o \
cuda_header.o \
cuda_env_para.o \
cuda_init.o
#fft_stick_cuda.o \
#fft_scalar.o

all : version libqemodgpu.a 

## Mimic CPU-only version
version:
	- ( if test -x ../install/update_gpu_version ; then \
	../install/update_gpu_version; \
	else if test ! -f ../include/gpu-version.h ; then \
	cat ../include/gpu-version.h.in > ../include/gpu-version.h ; fi ; fi )	

libqemodgpu.a: $(MODGPUOBJS)
	$(AR) $(ARFLAGS) $@ $? $(MODULES)     
	$(RANLIB) $@    

clean :
	- /bin/rm -f *.o *.a *.d *.i *~ *.F90 *.mod *.L *.lst 
	- /bin/rm ../include/.gpu-version.h.tmp ../include/gpu-version.h 

#include make.depend
