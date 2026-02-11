CXX = g++
CXXFLAGS = -std=c++17 -O2
INCLUDES = -I./genesis/lib
LIBS = -L./genesis/bin -lgenesis -lpthread -lz -Wl,-rpath,'$$ORIGIN/genesis/bin'

solver: solver.cpp
	$(CXX) $(CXXFLAGS) solver.cpp $(INCLUDES) $(LIBS) -o solver

clean:
	rm -f solver
