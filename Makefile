CXX = g++

CXXFLAGS = -O3 -march=native -flto -fno-plt -fno-rtti -fdevirtualize-speculatively

INCLUDES = -Isrc -Isrc/io -Isrc/reductions

LIBS = -pthread

TARGET = pace_solver
SRCS = src/main.cpp
HEADERS = $(shell find src -name '*.hpp')

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
