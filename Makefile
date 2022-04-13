unit_test:
	pytest

 compile_SIR:
	 cd cpp_src && \
 	 g++ -c -fPIC generic_SIR.cpp -o generic_SIR.o && \
 	 g++ -shared -Wl,-soname,libSIR.so -o libSIR.so  generic_SIR.o
