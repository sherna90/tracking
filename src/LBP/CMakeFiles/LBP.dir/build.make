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
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sergio/code/cpp/naive_bayes_multinomial/src/LBP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sergio/code/cpp/naive_bayes_multinomial/src/LBP

# Include any dependencies generated for this target.
include CMakeFiles/LBP.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/LBP.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/LBP.dir/flags.make

CMakeFiles/LBP.dir/LBP.cpp.o: CMakeFiles/LBP.dir/flags.make
CMakeFiles/LBP.dir/LBP.cpp.o: LBP.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sergio/code/cpp/naive_bayes_multinomial/src/LBP/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/LBP.dir/LBP.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/LBP.dir/LBP.cpp.o -c /home/sergio/code/cpp/naive_bayes_multinomial/src/LBP/LBP.cpp

CMakeFiles/LBP.dir/LBP.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LBP.dir/LBP.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sergio/code/cpp/naive_bayes_multinomial/src/LBP/LBP.cpp > CMakeFiles/LBP.dir/LBP.cpp.i

CMakeFiles/LBP.dir/LBP.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LBP.dir/LBP.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sergio/code/cpp/naive_bayes_multinomial/src/LBP/LBP.cpp -o CMakeFiles/LBP.dir/LBP.cpp.s

CMakeFiles/LBP.dir/LBP.cpp.o.requires:
.PHONY : CMakeFiles/LBP.dir/LBP.cpp.o.requires

CMakeFiles/LBP.dir/LBP.cpp.o.provides: CMakeFiles/LBP.dir/LBP.cpp.o.requires
	$(MAKE) -f CMakeFiles/LBP.dir/build.make CMakeFiles/LBP.dir/LBP.cpp.o.provides.build
.PHONY : CMakeFiles/LBP.dir/LBP.cpp.o.provides

CMakeFiles/LBP.dir/LBP.cpp.o.provides.build: CMakeFiles/LBP.dir/LBP.cpp.o

# Object files for target LBP
LBP_OBJECTS = \
"CMakeFiles/LBP.dir/LBP.cpp.o"

# External object files for target LBP
LBP_EXTERNAL_OBJECTS =

libLBP.so: CMakeFiles/LBP.dir/LBP.cpp.o
libLBP.so: CMakeFiles/LBP.dir/build.make
libLBP.so: /usr/local/lib/libopencv_xphoto.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_xobjdetect.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_ximgproc.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_xfeatures2d.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_tracking.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_text.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_surface_matching.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_structured_light.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_stereo.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_saliency.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_rgbd.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_reg.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_plot.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_optflow.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_line_descriptor.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_fuzzy.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_face.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_dpm.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_dnn.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_datasets.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_ccalib.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_bioinspired.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_bgsegm.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_aruco.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_videostab.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_videoio.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_video.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_superres.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_stitching.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_shape.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_photo.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_objdetect.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_ml.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_imgproc.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_highgui.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_flann.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_features2d.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_core.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_calib3d.so.3.1.0
libLBP.so: /usr/lib/x86_64-linux-gnu/libfftw3.so
libLBP.so: /usr/local/lib/libopencv_text.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_face.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_ximgproc.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_xfeatures2d.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_shape.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_video.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_objdetect.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_calib3d.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_features2d.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_ml.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_highgui.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_videoio.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_imgproc.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_flann.so.3.1.0
libLBP.so: /usr/local/lib/libopencv_core.so.3.1.0
libLBP.so: CMakeFiles/LBP.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library libLBP.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LBP.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/LBP.dir/build: libLBP.so
.PHONY : CMakeFiles/LBP.dir/build

CMakeFiles/LBP.dir/requires: CMakeFiles/LBP.dir/LBP.cpp.o.requires
.PHONY : CMakeFiles/LBP.dir/requires

CMakeFiles/LBP.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/LBP.dir/cmake_clean.cmake
.PHONY : CMakeFiles/LBP.dir/clean

CMakeFiles/LBP.dir/depend:
	cd /home/sergio/code/cpp/naive_bayes_multinomial/src/LBP && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sergio/code/cpp/naive_bayes_multinomial/src/LBP /home/sergio/code/cpp/naive_bayes_multinomial/src/LBP /home/sergio/code/cpp/naive_bayes_multinomial/src/LBP /home/sergio/code/cpp/naive_bayes_multinomial/src/LBP /home/sergio/code/cpp/naive_bayes_multinomial/src/LBP/CMakeFiles/LBP.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/LBP.dir/depend
