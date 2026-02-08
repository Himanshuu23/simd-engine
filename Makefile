CC = gcc
CFLAGS = -O3 -mavx2 -msse2 -mfma -Wall -Wextra -Iinclude
LDFLAGS = -lm

SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build

SIMD_SRC = $(SRC_DIR)/simd_abstraction.c
ARRAY_SRC = $(SRC_DIR)/array.c

TEST_SIMD = $(BUILD_DIR)/test_simd
TEST_ARRAY = $(BUILD_DIR)/test_array

all: $(TEST_SIMD) $(TEST_ARRAY)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_SIMD): $(SIMD_SRC) $(TEST_DIR)/test_simd.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SIMD_SRC) $(TEST_DIR)/test_simd.c $(LDFLAGS) -o $(TEST_SIMD)
	@echo "SIMD test built"

$(TEST_ARRAY): $(SIMD_SRC) $(ARRAY_SRC) $(TEST_DIR)/test_array.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SIMD_SRC) $(ARRAY_SRC) $(TEST_DIR)/test_array.c $(LDFLAGS) -o $(TEST_ARRAY)
	@echo "Array test built"

clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned"

test-simd: $(TEST_SIMD)
	@echo "\n Running SIMD Tests \n"
	./$(TEST_SIMD)

test-array: $(TEST_ARRAY)
	@echo "\nRunning Array Tests \n"
	./$(TEST_ARRAY)

test: test-simd test-array

.PHONY: all clean test test-simd test-array
