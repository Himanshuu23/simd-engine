CC = gcc
CFLAGS = -O3 -mavx2 -msse2 -mfma -Wall -Wextra -Iinclude
LDFLAGS = -lm

SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build

SOURCES = $(SRC_DIR)/simd_abstraction.c
TEST_SOURCES = $(TEST_DIR)/test_simd.c
TARGET = $(BUILD_DIR)/test_simd

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(SOURCES) $(TEST_SOURCES) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SOURCES) $(TEST_SOURCES) $(LDFLAGS) -o $(TARGET)
	@echo "Build complete! Run with: ./$(TARGET)"

clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory"

test: $(TARGET)
	@echo ""
	./$(TARGET)

.PHONY: all clean test
