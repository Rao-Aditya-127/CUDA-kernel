NVCC     = nvcc
CFLAGS   = -O2 -arch=sm_80   # change sm_80 to match your GPU (e.g. sm_75 for Turing)
TARGET   = vec_add

all: $(TARGET)

$(TARGET): vec_add.cu
	$(NVCC) $(CFLAGS) -o $@ $<

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)
