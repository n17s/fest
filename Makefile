CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -lm

ifndef build
	build=release
endif

ifeq ($(build),release)
	CFLAGS += -O3 -fomit-frame-pointer -ffast-math
endif

ifeq ($(build),profile)
	CFLAGS += -g3 -pg -fprofile-arcs -ftest-coverage
endif

ifeq ($(build),debug)
	CFLAGS += -g3
endif

%.o: %.c
	$(CC) $(CFLAGS) -c $<

all: festlearn festclassify

debug: 
	make build=debug

profile:
	make build=profile

festlearn: tree.o forest.o learn.o dataset.o
	$(CC) $(CFLAGS) -o festlearn tree.o forest.o learn.o dataset.o $(LDFLAGS)

festclassify: tree.o forest.o classify.o dataset.o 
	$(CC) $(CFLAGS) -o festclassify tree.o forest.o classify.o dataset.o $(LDFLAGS)

tree.o: tree.c tree.h dataset.h
dataset.o: dataset.c dataset.h
learn.o: learn.c
classify.o: classify.c
forest.o: tree.h forest.c forest.h

clean:
	/bin/rm -f svn-commit* *.o *.gcov *.gcda *.gcno gmon.out festlearn festclassify
