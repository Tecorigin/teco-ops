SRC = $(wildcard *.scpp)
OBJ = $(SRC:%.scpp=%.out)

all : $(OBJ)

./%.out:%.scpp
	tecocc $^ ../../../thirdparty/accuracy/lib/libtecoaccuracy.a -o $@

clean:
	rm -rf *.out

