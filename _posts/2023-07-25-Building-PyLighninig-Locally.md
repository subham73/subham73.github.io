

basically i was trying to contribute to pytorch lightinig 
and i found one good first issue on changing some legacy code to new code.

for that i need to build it locally to test it.

wsl2 + Ubuntu 22.04.2 LTS 

clone it from github

creating a conda environment

see locally cmake, make, gcc, g++ is installed or not
-
-
-
-
-

then install  
apt-get install libprotobuf-dev protobuf-compiler
export CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON" 







