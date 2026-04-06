Project structure:
project/
├── eigen_opencv.cpp
├── fisher_opencv.cpp
├── dataset/
   ├──An
   ├──Edward
   ├──Nam
   ├──modified
├── README.md

Compile the file:
!git clone https://github.com/Illgetgood/EE4228_Assignment_1_2026.git
%cd EE4228_Assignment_1_2026
!g++ fisher_opencv.cpp -o fisher_opencv `pkg-config --cflags --libs opencv4`
!./eigen_opencv
