# Quantum
## Summary
This is an implementation of a quantum computer emulator that aims to bridge the gap between proof-of-concept implementations and libraries like Q#, Cirq or Qiskit, by optimizing each part of the emulator without adding unnecessary boilerplate code.
## Project 
### Structure
- Examples are provided in [src/examples](https://github.com/Doge815/quantum/tree/main/src/examples).
- To write your own code, modify [src/main.c](https://github.com/Doge815/quantum/tree/main/src/main.c) or edit the examples.
- The code of the emulator can be found in [src/Simulation](https://github.com/Doge815/quantum/tree/main/src/Simulation).
  - [Math.h](https://github.com/Doge815/quantum/tree/main/src/Simulation/Math.h): Definition of vectors and matrices 
  - [Simulation.h](https://github.com/Doge815/quantum/tree/main/src/Simulation/Simulation.h): Definition of the quantum register and quantum operations
  - [Input.c](https://github.com/Doge815/quantum/tree/main/src/Simulation/Input.c): Creation and destruction of quantum registers
  - [Constants.c](https://github.com/Doge815/quantum/tree/main/src/Simulation/Constants.c): Definition of basic gates
  - [Gate.c](https://github.com/Doge815/quantum/tree/main/src/Simulation/Gate.c): Functions to apply a gate to a single qubit
  - [Controlled_Gate.c](https://github.com/Doge815/quantum/tree/main/src/Simulation/Controlled_Gate.c): Functions to apply controlled gates
  - [Measure.c](https://github.com/Doge815/quantum/tree/main/src/Simulation/Measure.c): Functions to measure a single or all qubits
### Compilation on GNU/Linux
#### Requirements
- CPU with AVX2 support
- GCC 13.2.1
- CMake 3.28
- Make
- git

Clone the repository.

`git clone https://github.com/Doge815/quantum.git`

Navigate to the root directory.

`cd quantum`

Create the Makefile.

`cmake .`

Compile the code.

`make`

Navigate to the output directory.

`cd bin`

