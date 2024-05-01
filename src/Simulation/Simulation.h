#ifndef SIMULATION_SIMULATION_H
#define SIMULATION_SIMULATION_H
#pragma clang diagnostic push
#pragma ide diagnostic ignored "bugprone-reserved-identifier"

#include "Math.h"

extern const double* restrict ZERO;

extern const ColumnVectorReal QG_One;
extern const ColumnVectorReal QG_Zero;

extern const MatrixReal QG_I;
extern const MatrixReal QG_X;
extern const MatrixReal QG_Z;
extern const MatrixReal QG_H;

extern const MatrixComplex QG_Y;
extern const MatrixComplex QG_S;
extern const MatrixComplex QG_T;

enum QuantumRegisterType {
    REAL,
    COMPLEX,
};

typedef struct QuantumRegister {
    uint qubit_count;
    enum QuantumRegisterType qubit_type;
    union {
        ColumnVectorReal real_reg;
        ColumnVectorComplex complex_reg;
    } q_register;

} QuantumRegister;

QuantumRegister create_register(const ColumnVectorReal* const base_states[], int count);
void delete_register(QuantumRegister const* restrict reg);
void apply_gate_real(QuantumRegister* reg, MatrixReal const* restrict gate, uint target);
void apply_gate_complex(QuantumRegister* reg, MatrixComplex const* restrict gate, uint target);
void apply_controlled_gate_real(QuantumRegister *restrict reg, MatrixReal const *restrict gate, const uint controls[], uint controls_count, uint target);
void apply_controlled_gate_complex(QuantumRegister *restrict reg, MatrixComplex const *restrict gate, const uint controls[], uint controls_count, uint target);
void apply_controlled_X(QuantumRegister *restrict reg, const uint controls[], uint controls_count, uint target);

/// apply_c_gate_real calls apply_controlled_gate_real, but accepts only one control qubit
/// \param reg the quantum register that contains the qubits
/// \param gate the gate that will be applied
/// \param control the index of the control qubit
/// \param target the target qubit
__attribute__((always_inline)) static inline void apply_c_gate_real(QuantumRegister *restrict reg, MatrixReal const *restrict gate, const uint control, uint target) {
    apply_controlled_gate_real(reg, gate, (const uint[]){control}, 1, target);
}

/// apply_c_gate_real calls apply_controlled_gate_real, but accepts only one control qubit
/// \param reg the quantum register that contains the qubits
/// \param gate the gate that will be applied
/// \param control the index of the control qubit
/// \param target the target qubit
__attribute__((always_inline)) static inline void apply_c_gate_complex(QuantumRegister *restrict reg, MatrixComplex const *restrict gate, const uint control, uint target) {
    apply_controlled_gate_complex(reg, gate, (const uint[]){control}, 1, target);
}

/// apply_cX calls apply_controlled_X, but accepts only one control qubit
/// \param reg the quantum register that contains the qubits
/// \param control the index of the control qubit
/// \param target the target qubit
__attribute__((always_inline)) static inline void apply_cX(QuantumRegister *restrict reg, const uint control, uint target) {
    apply_controlled_X(reg, (const uint[]){control}, 1, target);
}

qint measure_all(QuantumRegister* reg);

ColumnVectorReal const * measure_single(QuantumRegister* restrict reg, int index);
bool const * measurement_to_booleans(QuantumRegister *restrict req, qint measurement);
ColumnVectorReal const ** measurement_to_vectors(QuantumRegister *restrict req, qint measurement);

/// measure_to_booleans measures all qubits and creates a vector of booleans that represents the state of the register
/// \param reg the quantum register to be measured
/// \return vector of booleans that represents the state of the register
__attribute__((always_inline)) static inline bool const * measure_to_booleans(QuantumRegister *restrict reg) {
    return measurement_to_booleans(reg, measure_all(reg));
}

/// measure_to_qubits measures all qubits and creates a vector of ColumnVectorReal references that represents the state of the register
/// \param reg the quantum register to be measured
/// \return vector of ColumnVectorReal references that represents the state of the register
__attribute__((always_inline)) static inline ColumnVectorReal const ** measure_to_qubits(QuantumRegister *restrict reg) {
    return measurement_to_vectors(reg, measure_all(reg));
}

#endif //SIMULATION_SIMULATION_H

#pragma clang diagnostic pop
