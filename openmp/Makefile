# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -std=c++11

# Target executable name
TARGET = main

# Source files
SRC = main.c

# Input Image
INP_IMAGE = "input.jpg"

# Compile and link
$(TARGET): $(SRC)
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(shell pkg-config --cflags --libs opencv4) -lgomp -fopenmp -lgomp

# Clean
clean:
	rm -f $(TARGET)

run:
	./$(TARGET) $(INP_IMAGE) 
