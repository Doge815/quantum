#ifndef MATH_MATH_H
#define MATH_MATH_H

#include <stdint.h>

#define qfloat double
#define qint uint64_t //supports 64 qubits

#include <stdlib.h>
#include <string.h>

#define MULREAL1(m1, i1, m2, i2) ((m1->real[i1] * m2->real[i2]) - (m1->imag[i1] * m2->imag[i2]))
#define MULIMAG1(m1, i1, m2, i2) ((m1->real[i1] * m2->imag[i2]) + (m1->imag[i1] * m2->real[i2]))
#define MULREAL2(realVal, imagVal, m2, i2) ((realVal * m2->real[i2]) - (imagVal * m2->imag[i2]))
#define MULIMAG2(realVal, imagVal, m2, i2) ((realVal * m2->imag[i2]) + (imagVal * m2->real[i2]))

///a matrix that contains complex entries
typedef struct MatrixComplex {
    qint n;
    qfloat* restrict real;
    qfloat* restrict imag;
}MatrixComplex;

struct VectorComplex {
    qint n;
    qfloat* real;
    qfloat* imag;
};

///a vector that contains complex entries
typedef struct VectorComplex ColumnVectorComplex;

///a matrix that contains real entries
typedef struct MatrixReal {
    qint n;
    qfloat* restrict real;
}MatrixReal;

struct VectorReal {
    qint n;
    qfloat* real;
};

///a vector that contains real entries
typedef struct VectorReal ColumnVectorReal;


/// Creates a complex matrix with all elements set to zero.
/// \param n Size of the matrix.
/// \return The matrix.
static inline MatrixComplex MatrixComplexCreateZero(const qint n) {
    return (MatrixComplex) {.n = n, .real = calloc(n * n, sizeof(qfloat)), .imag = calloc(n * n, sizeof(qfloat))};
}

/// Creates an uninitialized complex matrix
/// \param n Size of the matrix.
/// \return The matrix.
static inline MatrixComplex MatrixComplexCreateEmpty(const qint n){
    return (MatrixComplex) {.n = n, .real = malloc(n * n * sizeof(qfloat)), .imag = malloc(n * n * sizeof(qfloat))};

}

/// Frees the memory allocated for the complex matrix.
/// \param matrix MatrixComplex which should be freed.
static inline void MatrixComplexFree(const MatrixComplex matrix){
    free(matrix.real);
    free(matrix.imag);
}

/// Creates a complex vector with all elements set to zero.
/// \param n Size of the vector.
/// \return The vector.
static inline struct VectorComplex VectorComplexCreateZero(const qint n) {
    return (struct VectorComplex) {.n = n, .real = calloc(n, sizeof(qfloat)), .imag = calloc(n, sizeof(qfloat))};
}

/// Creates an uninitialized complex vector
/// \param n Size of the vector.
/// \return The vector.
static inline struct VectorComplex VectorComplexCreateEmpty(const qint n){
    return (struct VectorComplex) {.n = n, .real = malloc( n * sizeof(qfloat)), .imag = malloc( n * sizeof(qfloat))};
}

/// Frees the memory allocated for the complex vector.
/// \param vector vector which should be freed.
static inline void VectorComplexFree(const struct VectorComplex vector){
    free(vector.real);
    free(vector.imag);
}

/// Creates a REAL matrix with all elements set to zero.
/// \param n Size of the matrix.
/// \return The matrix.
static inline MatrixReal MatrixRealCreateZero(const qint n) {
    return (MatrixReal) {.n = n, .real = calloc(n * n, sizeof(qfloat))};
}

/// Creates an uninitialized REAL matrix
/// \param n Size of the matrix.
/// \return The matrix.
static inline MatrixReal MatrixRealCreateEmpty(const qint n){
    return (MatrixReal) {.n = n, .real = malloc(n * n * sizeof(qfloat))};

}

/// Frees the memory allocated for the REAL matrix.
/// \param matrix MatrixReal which should be freed.
static inline void MatrixRealFree(const MatrixReal matrix){
    free(matrix.real);
}


/// Creates a REAL vector with all elements set to zero.
/// \param n Size of the vector.
/// \return The vector.
static inline struct VectorReal VectorRealCreateZero(const qint n) {
    return (struct VectorReal) {.n = n, .real = calloc(n, sizeof(qfloat))};
}

/// Creates an uninitialized REAL vector
/// \param n Size of the vector.
/// \return The vector.
static inline struct VectorReal VectorRealCreateEmpty(const qint n){
    return (struct VectorReal) {.n = n, .real = malloc( n * sizeof(qfloat))};
}

#endif //MATH_MATH_H
