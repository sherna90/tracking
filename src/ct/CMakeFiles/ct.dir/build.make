# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sergio/code/cpp/VOTR/CT

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sergio/code/cpp/VOTR/CT

# Include any dependencies generated for this target.
include CMakeFiles/ct.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ct.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ct.dir/flags.make

CMakeFiles/ct.dir/CompressiveTracker.cpp.o: CMakeFiles/ct.dir/flags.make
CMakeFiles/ct.dir/CompressiveTracker.cpp.o: CompressiveTracker.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sergio/code/cpp/VOTR/CT/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/ct.dir/CompressiveTracker.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/ct.dir/CompressiveTracker.cpp.o -c /home/sergio/code/cpp/VOTR/CT/CompressiveTracker.cpp

CMakeFiles/ct.dir/CompressiveTracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ct.dir/CompressiveTracker.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sergio/code/cpp/VOTR/CT/CompressiveTracker.cpp > CMakeFiles/ct.dir/CompressiveTracker.cpp.i

CMakeFiles/ct.dir/CompressiveTracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ct.dir/CompressiveTracker.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sergio/code/cpp/VOTR/CT/CompressiveTracker.cpp -o CMakeFiles/ct.dir/CompressiveTracker.cpp.s

CMakeFiles/ct.dir/CompressiveTracker.cpp.o.requires:
.PHONY : CMakeFiles/ct.dir/CompressiveTracker.cpp.o.requires

CMakeFiles/ct.dir/CompressiveTracker.cpp.o.provides: CMakeFiles/ct.dir/CompressiveTracker.cpp.o.requires
	$(MAKE) -f CMakeFiles/ct.dir/build.make CMakeFiles/ct.dir/CompressiveTracker.cpp.o.provides.build
.PHONY : CMakeFiles/ct.dir/CompressiveTracker.cpp.o.provides

CMakeFiles/ct.dir/CompressiveTracker.cpp.o.provides.build: CMakeFiles/ct.dir/CompressiveTracker.cpp.o

CMakeFiles/ct.dir/RunTracker.cpp.o: CMakeFiles/ct.dir/flags.make
CMakeFiles/ct.dir/RunTracker.cpp.o: RunTracker.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sergio/code/cpp/VOTR/CT/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/ct.dir/RunTracker.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/ct.dir/RunTracker.cpp.o -c /home/sergio/code/cpp/VOTR/CT/RunTracker.cpp

CMakeFiles/ct.dir/RunTracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ct.dir/RunTracker.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sergio/code/cpp/VOTR/CT/RunTracker.cpp > CMakeFiles/ct.dir/RunTracker.cpp.i

CMakeFiles/ct.dir/RunTracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ct.dir/RunTracker.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sergio/code/cpp/VOTR/CT/RunTracker.cpp -o CMakeFiles/ct.dir/RunTracker.cpp.s

CMakeFiles/ct.dir/RunTracker.cpp.o.requires:
.PHONY : CMakeFiles/ct.dir/RunTracker.cpp.o.requires

CMakeFiles/ct.dir/RunTracker.cpp.o.provides: CMakeFiles/ct.dir/RunTracker.cpp.o.requires
	$(MAKE) -f CMakeFiles/ct.dir/build.make CMakeFiles/ct.dir/RunTracker.cpp.o.provides.build
.PHONY : CMakeFiles/ct.dir/RunTracker.cpp.o.provides

CMakeFiles/ct.dir/RunTracker.cpp.o.provides.build: CMakeFiles/ct.dir/RunTracker.cpp.o

# Object files for target ct
ct_OBJECTS = \
"CMakeFiles/ct.dir/CompressiveTracker.cpp.o" \
"CMakeFiles/ct.dir/RunTracker.cpp.o"

# External object files for target ct
ct_EXTERNAL_OBJECTS =

ct: CMakeFiles/ct.dir/CompressiveTracker.cpp.o
ct: CMakeFiles/ct.dir/RunTracker.cpp.o
ct: CMakeFiles/ct.dir/build.make
ct: /usr/local/lib/libopencv_xphoto.so.3.0.0
ct: /usr/local/lib/libopencv_xobjdetect.so.3.0.0
ct: /usr/local/lib/libopencv_ximgproc.so.3.0.0
ct: /usr/local/lib/libopencv_xfeatures2d.so.3.0.0
ct: /usr/local/lib/libopencv_tracking.so.3.0.0
ct: /usr/local/lib/libopencv_text.so.3.0.0
ct: /usr/local/lib/libopencv_surface_matching.so.3.0.0
ct: /usr/local/lib/libopencv_saliency.so.3.0.0
ct: /usr/local/lib/libopencv_rgbd.so.3.0.0
ct: /usr/local/lib/libopencv_reg.so.3.0.0
ct: /usr/local/lib/libopencv_optflow.so.3.0.0
ct: /usr/local/lib/libopencv_line_descriptor.so.3.0.0
ct: /usr/local/lib/libopencv_latentsvm.so.3.0.0
ct: /usr/local/lib/libopencv_face.so.3.0.0
ct: /usr/local/lib/libopencv_datasets.so.3.0.0
ct: /usr/local/lib/libopencv_ccalib.so.3.0.0
ct: /usr/local/lib/libopencv_bioinspired.so.3.0.0
ct: /usr/local/lib/libopencv_bgsegm.so.3.0.0
ct: /usr/local/lib/libopencv_adas.so.3.0.0
ct: /usr/local/lib/libopencv_videostab.so.3.0.0
ct: /usr/local/lib/libopencv_videoio.so.3.0.0
ct: /usr/local/lib/libopencv_video.so.3.0.0
ct: /usr/local/lib/libopencv_superres.so.3.0.0
ct: /usr/local/lib/libopencv_stitching.so.3.0.0
ct: /usr/local/lib/libopencv_shape.so.3.0.0
ct: /usr/local/lib/libopencv_photo.so.3.0.0
ct: /usr/local/lib/libopencv_objdetect.so.3.0.0
ct: /usr/local/lib/libopencv_ml.so.3.0.0
ct: /usr/local/lib/libopencv_imgproc.so.3.0.0
ct: /usr/local/lib/libopencv_imgcodecs.so.3.0.0
ct: /usr/local/lib/libopencv_highgui.so.3.0.0
ct: /usr/local/lib/libopencv_hal.a
ct: /usr/local/lib/libopencv_flann.so.3.0.0
ct: /usr/local/lib/libopencv_features2d.so.3.0.0
ct: /usr/local/lib/libopencv_core.so.3.0.0
ct: /usr/local/lib/libopencv_calib3d.so.3.0.0
ct: /usr/local/lib/libopencv_text.so.3.0.0
ct: /usr/local/lib/libopencv_face.so.3.0.0
ct: /usr/local/lib/libopencv_xobjdetect.so.3.0.0
ct: /usr/local/lib/libopencv_xfeatures2d.so.3.0.0
ct: /usr/local/lib/libopencv_shape.so.3.0.0
ct: /usr/local/lib/libopencv_video.so.3.0.0
ct: /usr/local/lib/libopencv_calib3d.so.3.0.0
ct: /usr/local/lib/libopencv_features2d.so.3.0.0
ct: /usr/local/lib/libopencv_ml.so.3.0.0
ct: /usr/local/lib/libopencv_highgui.so.3.0.0
ct: /usr/local/lib/libopencv_videoio.so.3.0.0
ct: /usr/local/lib/libopencv_imgcodecs.so.3.0.0
ct: /usr/local/lib/libopencv_imgproc.so.3.0.0
ct: /usr/local/lib/libopencv_flann.so.3.0.0
ct: /usr/local/lib/libopencv_core.so.3.0.0
ct: /usr/local/lib/libopencv_hal.a
ct: /usr/local/share/OpenCV/3rdparty/lib/libippicv.a
ct: CMakeFiles/ct.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ct"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ct.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ct.dir/build: ct
.PHONY : CMakeFiles/ct.dir/build

CMakeFiles/ct.dir/requires: CMakeFiles/ct.dir/CompressiveTracker.cpp.o.requires
CMakeFiles/ct.dir/requires: CMakeFiles/ct.dir/RunTracker.cpp.o.requires
.PHONY : CMakeFiles/ct.dir/requires

CMakeFiles/ct.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ct.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ct.dir/clean

CMakeFiles/ct.dir/depend:
	cd /home/sergio/code/cpp/VOTR/CT && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sergio/code/cpp/VOTR/CT /home/sergio/code/cpp/VOTR/CT /home/sergio/code/cpp/VOTR/CT /home/sergio/code/cpp/VOTR/CT /home/sergio/code/cpp/VOTR/CT/CMakeFiles/ct.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ct.dir/depend
