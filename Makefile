CC = gcc
CFLAGS = -Wall -Wextra

ifndef build
	build=release
endif

ifeq ($(build),release)
	CFLAGS += -O3 -fomit-frame-pointer -ffast-math -march=pentium3 -mfpmath=sse
endif

ifeq ($(build),profile)
	CFLAGS += -g3 -pg -fprofile-arcs -ftest-coverage
endif

ifeq ($(build),debug)
	CFLAGS += -g3
endif

%.o: %.c
	$(CC) $(CFLAGS) -c $<

all: rflearn rfclassify

debug: 
	make build=debug

profile:
	make build=profile

rflearn: tree.o common.o
	$(CC) $(CFLAGS) -o rflearn tree.o common.o

rfclassify: tree.o common.o
	$(CC) $(CFLAGS) -o rfclassify tree.o common.o

tree.o: tree.c tree.h dataset.h
common.o: common.c dataset.h

clean:
	/bin/rm -f svn-commit* *.o *.gcov *.gcda *.gcno gmon.out vplearn vpclassify
