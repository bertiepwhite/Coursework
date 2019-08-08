CC	= gcc
LIBS    = -lOpenCL -lm -lcurl
LIBPATH = -L${HOME}/cfitsio/lib
INCPATH = -I${HOME}/cfitsio/include

all: Gravity_me_cl

Gravity_me_cl.o: Gravity_me_cl.c
		$(CC) -c Gravity_me_cl.c
Gravity_me_cl: Gravity_me_cl.o
		$(CC) Gravity_me_cl.o $(LIBS) -o Gravity_me_cl
 

