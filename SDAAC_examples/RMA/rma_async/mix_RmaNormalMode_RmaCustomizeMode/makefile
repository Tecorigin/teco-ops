SRC = $(wildcard *.scpp)
OBJ = $(SRC:%.scpp=%.out)

all : $(OBJ)

./%.out:%.scpp
	tecocc $^ -o $@

clean:
	rm -rf *.out

