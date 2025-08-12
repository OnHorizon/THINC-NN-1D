CXX=g++
CXXFLAGS=-O3 -Wall -Wextra -MD
SRC := $(wildcard *.cc)
LIBS = -lm

OBJ=${SRC:.cc=.o}

%.o: %.cc
	${CXX} ${CXXFLAGS} -o  $@  -c $< ${LIBS}
a.out: ${OBJ}
	${CXX} ${CXXFLAGS} -o $@  ${OBJ} ${LIBS}
clean:
	@rm -f *.o *.d a.out
distclean:
	@rm -f *.o *.d *.csv *.png a.out
scrub:
	@rm -f *.csv

-include $(OBJ:.o=.d)
