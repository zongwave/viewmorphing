CC := g++
LD := ld
SOURCES := $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
DEPENDS = $(SOURCES:.cpp=.d)

OCV_VERSION = $(shell pkg-config --modversion opencv4)
ifneq ($(OCV_VERSION), )
    $(info OpenCV $(OCV_VERSION) is installed)
    INC_PATH += $(shell pkg-config --cflags opencv4)
    LDFLAGS += $(shell pkg-config --libs opencv4)
    CXXFLAGS += -D OCV_VERSION_4
else
    OCV_VERSION = $(shell pkg-config --modversion opencv)
    ifneq ($(OCV_VERSION), )
        $(info OpenCV $(OCV_VERSION) is installed)
        INC_PATH += $(shell pkg-config --cflags opencv)
        LDFLAGS += $(shell pkg-config --libs opencv)
        CXXFLAGS += -D OCV_VERSION_3
    endif
endif

INC_PATH += $(shell pkg-config --cflags libxcam)
LDFLAGS += $(shell pkg-config --libs libxcam)

AVX512_INSTRUCTION = $(shell grep -c avx512 /proc/cpuinfo)
ifeq ($(AVX512_INSTRUCTION), 0)
CREATE_MAP_OPT := 0
else
CREATE_MAP_OPT := 1
endif

SOFLAGS = -shared
CXXFLAGS += -fPIC -g -Ofast
ifneq ($(CREATE_MAP_OPT), 0)
CXXFLAGS += -mavx512f -DCREATE_MAP_USE_SIMD
else
CXXFLAGS += -DCREATE_MAP_USE_PLAIN_CODE
endif
CXXFLAGS += $(INC_PATH)

ifeq ($(DEBUG), 1)
CXXFLAGS += -DDEBUG
endif

ifeq ($(DEBUG_MORPH), 1)
CXXFLAGS += -DDEBUG_MORPH
endif

#TARGET_LIB := libfreeview.so
TARGET_EXE := test-freeview

.PHONY: all clean

all: $(TARGET_LIB) $(TARGET_EXE)

all:

$(TARGET_EXE): $(OBJECTS)
	$(CC) $^ $(LDFLAGS) -o $@

$(TARGET_LIB): $(OBJECTS)
	$(CC) $(SOFLAGS) -o $@ $^ $(LDFLAGS)

%.d: %.cpp
	$(CC) $(CXXFLAGS) -M $< > $@

%.o: %.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@


.PHONY: clean
clean:
	rm -f $(OBJECTS) $(TARGET_EXE) $(TARGET_LIB)
