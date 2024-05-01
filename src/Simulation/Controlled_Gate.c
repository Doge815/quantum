#include <stdio.h>
#include "Simulation.h"

static inline void __attribute__((always_inline)) _apply_controlled_gate_real(ColumnVectorReal *restrict const input, const qint count, const uint controls[], const uint controlCount, const uint target, MatrixReal const* restrict const gate) {
    //mask to check if all control bits are set for a given index
    qint controlBitsMask = 0;
    //iterate over all control qubits
    for (int i = 0; i < controlCount; i++) {
        //set the bit of the mask
        controlBitsMask |= 1 << (count - 1 - controls[i]);
    }
    //used to get the index of the probability, that the qubit is measured as |1>
    const qint bitFlip = 1 << (count - target - 1);
    //apply the gate without parallelization for less than 22 qubits
    if(count < 21) {
        //iterate over all indices
        for(qint i = 0; i < input->n; i++) {
            //skip the index, if not all control qubits are |1>
            if((i & controlBitsMask ) == controlBitsMask) {
                qint other = i ^ bitFlip;
                //skip the index, if the target qubit is |1>
                if ( i < other) {
                    //probability that the qubit is |0>
                    const qfloat iVal = input->real[i];
                    //probability that the qubit is |1>
                    const qfloat otherVal = input->real[other];
                    //apply the gate
                    input->real[i] = iVal * gate->real[0] + otherVal * gate->real[1];
                    input->real[other] = iVal * gate->real[2] + otherVal * gate->real[3];
                }
            }
        }
    } else {
        //the same but parallelized
#pragma omp parallel for
        for(qint i = 0; i < input->n; i++) {
            if((i & controlBitsMask ) == controlBitsMask) {
                qint other = i ^ bitFlip;
                if ( i < other) {
                    qfloat iVal = input->real[i];
                    qfloat otherVal = input->real[other];
                    input->real[i] = iVal * gate->real[0] + otherVal * gate->real[1];
                    input->real[other] = iVal * gate->real[2] + otherVal * gate->real[3];
                }
            }
        }
    }
}

static inline void __attribute__((always_inline)) _apply_controlled_gate_complex(ColumnVectorComplex *restrict const input, const qint count, const uint controls[], const uint controlCount, const uint target, MatrixComplex const* restrict const gate) {
    //the same as the real version, just with complex probabilities
    qint controlBitsMask = 0;
    for (int i = 0; i < controlCount; i++) {
        controlBitsMask |= 1 << (count - 1 - controls[i]);
    }
    const qint bitFlip = 1 << (count - target - 1);
    if(count < 22) {
        for(qint i = 0; i < input->n; i++) {
            if((i & controlBitsMask ) == controlBitsMask) {
                qint other = i ^ bitFlip;
                if ( i < other) {
                    const qfloat iVal_real = input->real[i];
                    const qfloat iVal_imag = input->imag[i];
                    const qfloat otherVal_real = input->real[other];
                    const qfloat otherVal_imag = input->imag[other];

                    input->real[i] = MULREAL2(iVal_real, iVal_imag, gate, 0) + MULREAL2(otherVal_real, otherVal_imag, gate, 1);
                    input->imag[i] = MULIMAG2(iVal_real, iVal_imag, gate, 0) + MULIMAG2(otherVal_real, otherVal_imag, gate, 1);

                    input->real[other] = MULREAL2(iVal_real, iVal_imag, gate, 2) + MULREAL2(otherVal_real, otherVal_imag, gate, 3);
                    input->imag[other] = MULIMAG2(iVal_real, iVal_imag, gate, 2) + MULIMAG2(otherVal_real, otherVal_imag, gate, 3);
                }
            }
        }
    } else {
#pragma omp parallel for
        for(qint i = 0; i < input->n; i++) {
            if((i & controlBitsMask ) == controlBitsMask) {
                qint other = i ^ bitFlip;
                if ( i < other) {
                    const qfloat iVal_real = input->real[i];
                    const qfloat iVal_imag = input->imag[i];
                    const qfloat otherVal_real = input->real[other];
                    const qfloat otherVal_imag = input->imag[other];

                    input->real[i] = MULREAL2(iVal_real, iVal_imag, gate, 0) + MULREAL2(otherVal_real, otherVal_imag, gate, 1);
                    input->imag[i] = MULIMAG2(iVal_real, iVal_imag, gate, 0) + MULIMAG2(otherVal_real, otherVal_imag, gate, 1);

                    input->real[other] = MULREAL2(iVal_real, iVal_imag, gate, 2) + MULREAL2(otherVal_real, otherVal_imag, gate, 3);
                    input->imag[other] = MULIMAG2(iVal_real, iVal_imag, gate, 2) + MULIMAG2(otherVal_real, otherVal_imag, gate, 3);
                }
            }
        }
    }
}

static inline void __attribute__((always_inline)) _apply_controlled_X_real(ColumnVectorReal *restrict const input, const qint count, const uint controls[], const uint controlCount, const uint target) {
    //applies the x gate without the need to actually multiply and add values, saving a few clock cycles
    qint controlBitsMask = 0;
    for (int i = 0; i < controlCount; i++) {
        controlBitsMask |= 1 << (count - 1 - controls[i]);
    }
    const qint bitFlip = 1 << (count - target - 1);
    if(count < 21) {
        for(qint i = 0; i < input->n; i++) {
            if((i & controlBitsMask ) == controlBitsMask) {
                qint other = i ^ bitFlip;
                if ( i < other) {
                    //swaps the values without adding oder multiplying
                    const qfloat temp = input->real[other];
                    input->real[other] = input->real[i];
                    input->real[i] = temp;
                }
            }
        }
    } else {
#pragma omp parallel for
        for(qint i = 0; i < input->n; i++) {
            if((i & controlBitsMask ) == controlBitsMask) {
                qint other = i ^ bitFlip;
                if ( i < other) {
                    const qfloat temp = input->real[other];
                    input->real[other] = input->real[i];
                    input->real[i] = temp;
                }
            }
        }
    }
}

static inline void __attribute__((always_inline)) _apply_controlled_X_complex(ColumnVectorComplex *restrict const input, const qint count, const uint controls[], const uint controlCount, const uint target) {
    //the same as the real function, just with complex probabilities
    qint controlBitsMask = 0;
    for (int i = 0; i < controlCount; i++) {
        controlBitsMask |= 1 << (count - 1 - controls[i]);
    }
    const qint bitFlip = 1 << (count - target - 1);
    if(count < 22) { //use linear direct in-place linear aka applyDirectInplace
        for(qint i = 0; i < input->n; i++) {
            if((i & controlBitsMask ) == controlBitsMask) {
                qint other = i ^ bitFlip;
                if ( i < other) {
                    qfloat iVal_real = input->real[i];
                    qfloat iVal_imag = input->imag[i];
                    qfloat otherVal_real = input->real[other];
                    qfloat otherVal_imag = input->imag[other];

                    input->real[i] = otherVal_real;
                    input->imag[i] = otherVal_imag;
                    input->real[other] = iVal_real;
                    input->imag[other] = iVal_imag;
                }
            }
        }
    } else {
#pragma omp parallel for
        for(qint i = 0; i < input->n; i++) {
            if((i & controlBitsMask ) == controlBitsMask) {
                qint other = i ^ bitFlip;
                if ( i < other) {
                    qfloat iVal_real = input->real[i];
                    qfloat iVal_imag = input->imag[i];
                    qfloat otherVal_real = input->real[other];
                    qfloat otherVal_imag = input->imag[other];

                    input->real[i] = otherVal_real;
                    input->imag[i] = otherVal_imag;
                    input->real[other] = iVal_real;
                    input->imag[other] = iVal_imag;
                }
            }
        }
    }
}

/// apply_controlled_gate_real applies a gate to the target qubit, if all control qubits are |1>
/// \param reg the quantum register that contains the qubits
/// \param gate the gate that will be applied
/// \param controls the indices of the control qubits
/// \param controls_count the number of control qubits
/// \param target the index of the target qubit
void apply_controlled_gate_real(QuantumRegister *restrict const reg, MatrixReal const *restrict const gate, const uint controls[], const uint controls_count, const uint target) {
    if (reg->qubit_type == REAL) {
        _apply_controlled_gate_real(&reg->q_register.real_reg, reg->qubit_count, controls, controls_count, target, gate);
    } else {
        const MatrixComplex gate_complex = {gate->n, gate->real, (double *restrict) ZERO};
        _apply_controlled_gate_complex(&reg->q_register.complex_reg, reg->qubit_count, controls, controls_count, target, &gate_complex);
    }
}

/// apply_controlled_gate_complex applies a gate to the target qubit, if all control qubits are |1>
/// \param reg the quantum register that contains the qubits
/// \param gate the gate that will be applied
/// \param controls the indices of the control qubits
/// \param controls_count the number of control qubits
/// \param target the index of the target qubit
void apply_controlled_gate_complex(QuantumRegister *restrict const reg, MatrixComplex const *restrict const gate, const uint controls[], const uint controls_count, const uint target) {
    if (reg->qubit_type == REAL) {
        const ColumnVectorReal old = reg->q_register.real_reg;
        const ColumnVectorComplex new = (ColumnVectorComplex){old.n, old.real, calloc(old.n, sizeof(qfloat))};
        if(new.imag == nullptr) {
            fprintf(stderr, "Out of memory.");
            exit(-1);
        }
        reg->q_register.complex_reg = new;
        reg->qubit_type = COMPLEX;
    }
    _apply_controlled_gate_complex(&reg->q_register.complex_reg, reg->qubit_count, controls, controls_count, target, gate);
}

/// apply_controlled_X applies the X gate to the target qubit, if all control qubits are |1>
/// \param reg the quantum register that contains the qubits
/// \param controls the indices of the control qubits
/// \param controls_count the number of control qubits
/// \param target the index of the target qubit
void apply_controlled_X(QuantumRegister *restrict const reg, const uint controls[], const uint controls_count, const uint target) {
    if (reg->qubit_type == REAL) {
        _apply_controlled_X_real(&reg->q_register.real_reg, reg->qubit_count, controls, controls_count, target);
    } else {
        _apply_controlled_X_complex(&reg->q_register.complex_reg, reg->qubit_count, controls, controls_count, target);
    }
}
