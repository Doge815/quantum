#include "Simulation.h"

//Used to convert a real matrix to complex matrix
const double *restrict ZERO = (double[]) {0.0, 0.0, 0.0, 0.0};

///Vector representation of the base state |1>, 1, or true
const ColumnVectorReal QG_One = {2, (double[]) {0.0, 1.0}};

///Vector representation of the base state |0>, 0, or false
const ColumnVectorReal QG_Zero = {2, (double[]) {1.0, 0.0}};

const qfloat inv_sqrt2 = 0.7071067811865475244008443621048490392848359376884740365883398689;

///Identity Gate
///does not modify the state of a qubit
const MatrixReal QG_I = {2, (double[]) {1.0, 0.0, 0.0, 1.0}};

///X Gate
///is used to rotate a qubit pi radians around the x axis and flips the qubit
const MatrixReal QG_X = {2, (double[]) {0.0, 1.0, 1.0, 0.0}};

///Z Gate
///is used to rotate a qubit pi radians around the z axis
const MatrixReal QG_Z = {2, (double[]) {1.0, 0.0, 0.0, -1.0}};

///H Gate
///is used to bring a qubit into a superposiotion
const MatrixReal QG_H = {2, (double[]) {inv_sqrt2, inv_sqrt2, inv_sqrt2, -inv_sqrt2}};


///H Gate
///is used to bring a qubit into a superposiotion
const MatrixComplex QG_Y = {2, (double[]) {0.0, 0.0, 0.0, 0.0}, (double[]) {0, -1, 1, 0}};

///S Gate
///shifts the phase of a qubit by exp(i*pi/2)
const MatrixComplex QG_S = {2, (double[]) {1.0, 0.0, 0.0, 0.0}, (double[]) {0, 0, 0, 1}};

///T Gate
///shifts the phase of a qubit by exp(i*pi/4)
const MatrixComplex QG_T = {2, (double[]) {1.0, 0.0, 0.0, inv_sqrt2}, (double[]) {0, 0, 0, inv_sqrt2}};
