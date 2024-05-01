#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include "Simulation.h"

static inline void __attribute__((always_inline)) _apply_gate_real(ColumnVectorReal *restrict const input, MatrixReal const *restrict const gate, const uint target, const uint count) {
    //the index of the target bit, but from the back
    const qint shifts_to_target_bit = (count - target - 1);

    //linearly iterate over half the indices and calculate the other half using a bit mask
    if (count < 17) {

        //a bit mask that masks only the bit that describes the target qubit
        //if the masked bit isn't set for an index, the qubit is |0>
        const qint bitFlip = (1ULL << shifts_to_target_bit);

        //masks all bits right of the target qubit
        const qint rightBitsMask = (1ULL << shifts_to_target_bit) - 1ULL;

        //masks all bits left of the target qubit
        const qint leftBitsMask = ~rightBitsMask;

        //iterate over half of all possible indices
        for (qint i = 0; i < input->n / 2; i++) {

            //get the bits on the right
            const qint rightBits = i & rightBitsMask;

            //get the bits on the left and move them to the left
            const qint leftBits = (i & leftBitsMask) << 1;

            //put both halfs back together, effectively adding a 0 into 'i' at the position of the target qubit
            const qint index = leftBits | rightBits;

            //flip the bit that represents the target qubit
            const qint other = index ^ bitFlip;

            //get the probabilities that the qubit is measured as |0> and |1>
            const qfloat combinedVal = input->real[index];
            const qfloat otherVal = input->real[other];

            //apply the matrix
            input->real[index] = combinedVal * gate->real[0] + otherVal * gate->real[1];
            input->real[other] = combinedVal * gate->real[2] + otherVal * gate->real[3];
        }
    } else {
        //try to allocate output memory
        double* output = malloc(input->n * sizeof(qfloat));
        if (output != nullptr) {
            //iterate over all indices in parallel with manually optimized SIMD instructions
            //register to flip the bit that represents the target qubit
            const __m256i bitFlip = _mm256_set1_epi64x((1LL << shifts_to_target_bit));
#pragma omp parallel for

            //iterate over all indices in steps of 4
            //since there are > 17 qubits and the vector contains 2^(qubit count) values, the number of indices is always dividable by 4
            for (qint i = 0; i < input->n; i += 4) {

                //store the indices in a simd register
                const __m256i indices = _mm256_set_epi64x((int64_t) i + 3, (int64_t) i + 2, (int64_t) i + 1, (int64_t) i);

                //flip the bits to get the state, where the target qubit is flipped
                const __m256i other = _mm256_xor_si256(indices, bitFlip);

                //store the indices with a flipped bit in an array
                qint otherArray[4];
                _mm256_storeu_si256((__m256i_u *) otherArray, other);

                //iterate over the 4 indices
                for (int j = 0; j < 4; ++j) {

                    //if the index is smaller than the index with the flipped bit, use the first row of the matrix
                    //save the result in the output matrix
                    if (i + j < otherArray[j]) {
                        output[i + j] = input->real[i + j] * gate->real[0] + input->real[otherArray[j]] * gate->real[1];
                    }
                    //use the second row of the matrix
                    else {
                        output[i + j] = input->real[i + j] * gate->real[3] + input->real[otherArray[j]] * gate->real[2];
                    }
                }
            }
            //free the original vector
            free(input->real);

            //set the output vector as the vector of the quantum register
            input->real = output;

        } else {
            //this is only executed, if the allocation was not successful

            //register to mark all bits to the right of the bit that represents the target qubit
            const __m256i rightBitsMask = _mm256_set1_epi64x((1LL << shifts_to_target_bit) - 1LL);
            //register to mark all bits to the left of the bit that represents the target qubit
            const __m256i leftBitsMask = _mm256_set1_epi64x(~((1LL << shifts_to_target_bit) - 1LL));
            //register to flip the bit that represents the target qubit
            const __m256i bitFlip = _mm256_set1_epi64x((1LL << shifts_to_target_bit));

#pragma omp parallel  for
            //input-> is always dividable by 4
            for (qint i = 0; i < input->n / 2; i += 4) {

                //The following few lines are faster than loading the values with _mm256_set_epi64x for some reason,
                //but only for this scenario.

                //create a 32-bit aligned array on the stack that contains the indices
                alignas(32) const qint iArray[4] = {i, i + 1, i + 2, i + 3};

                //bring the cache line of the array to all levels of the cache
                _mm_prefetch((const char *) iArray, _MM_HINT_T0);

                //load the array into a register
                const __m256i iRegister = _mm256_load_si256((const __m256i_u *) iArray);

                //get the right bits
                const __m256i rightBits = _mm256_and_si256(iRegister, rightBitsMask);

                //get the left bits
                __m256i leftBits = _mm256_and_si256(iRegister, leftBitsMask);

                //shift left bits to the left by 1
                leftBits = _mm256_slli_epi64(leftBits, 1);

                //combine both halves, effectively adding a '0' at the position of the target qubit
                const __m256i indices = _mm256_or_si256(leftBits, rightBits);

                //flip the target qubit to get the indices where the target_qubit is |1>
                const __m256i other = _mm256_xor_si256(indices, bitFlip);

                //save the indices to arrays
                qint combinedArray[4];
                qint otherArray[4];
                _mm256_storeu_si256((__m256i_u *) combinedArray, indices);
                _mm256_storeu_si256((__m256i_u *) otherArray, other);

                //iterate over the indices
                for (int j = 0; j < 4; ++j) {
                    //save temporary copies of the indices and probabilities, gets optimized by the compiler
                    const qint combinedIndex = combinedArray[j];
                    const qint otherIndex = otherArray[j];
                    const qfloat combinedVal = input->real[combinedIndex];
                    const qfloat otherVal = input->real[otherIndex];

                    //apply the matrix
                    input->real[combinedIndex] = combinedVal * gate->real[0] + otherVal * gate->real[1];
                    input->real[otherIndex] = combinedVal * gate->real[2] + otherVal * gate->real[3];
                }

            }
        }
    }
}

static inline void __attribute__((always_inline)) _apply_gate_complex(ColumnVectorComplex *restrict const input, MatrixComplex const *restrict const gate, const uint target, const uint count) {
    //the same as the real function, just with complex probabilities
    const qint shifts_to_target_bit = (count - target - 1);
    if (count < 24) {
        const qint bitFlip = (1ULL << shifts_to_target_bit);
        const qint rightBitsMask = (1ULL << shifts_to_target_bit) - 1LL;
        const qint leftBitsMask = ~rightBitsMask;
        for (qint i = 0; i < input->n / 2; i++) {
            const qint rightBits = i & rightBitsMask;

            const qint leftBits = (i & leftBitsMask) << 1;

            const qint index = leftBits | rightBits;
            const qint other = index ^ bitFlip;

            const qfloat indexVal_real = input->real[index];
            const qfloat indexVal_imag = input->imag[index];

            const qfloat otherVal_real = input->real[other];
            const qfloat otherVal_imag = input->imag[other];

            input->real[index] =
                    MULREAL2(indexVal_real, indexVal_imag, gate, 0) + MULREAL2(otherVal_real, otherVal_imag, gate, 1);
            input->imag[index] =
                    MULREAL2(indexVal_real, indexVal_imag, gate, 0) + MULREAL2(otherVal_real, otherVal_imag, gate, 1);

            input->real[other] =
                    MULREAL2(indexVal_real, indexVal_imag, gate, 2) + MULREAL2(otherVal_real, otherVal_imag, gate, 3);
            input->imag[other] =
                    MULREAL2(indexVal_real, indexVal_imag, gate, 2) + MULREAL2(otherVal_real, otherVal_imag, gate, 3);
        }
    } else {
        double* output_real = malloc(input->n * sizeof(qfloat));
        double* output_imag = malloc(input->n * sizeof(qfloat));
        if (output_real != nullptr && output_imag != nullptr) {
            const __m256i otherMask = _mm256_set1_epi64x((1L << shifts_to_target_bit));
#pragma omp parallel for
            for (qint i = 0; i < input->n; i+=4) {
                const __m256i indices = _mm256_set_epi64x(i + 3, i + 2, i + 1, i);
                const __m256i other = _mm256_xor_si256(indices, otherMask);
                qint otherArray[4];
                _mm256_storeu_si256((__m256i_u *) otherArray, other);

                for (int j = 0; j < 4; ++j) {
                    if (i + j < otherArray[j]) {
                        output_real[i] = MULREAL1(input, i + j, gate, 0) + MULREAL1(input, otherArray[j], gate, 1);
                        output_imag[i] = MULIMAG1(input, i + j, gate, 0) + MULIMAG1(input, otherArray[j], gate, 1);
                    } else {
                        output_real[i] = MULREAL1(input, i + j, gate, 3) + MULREAL1(input, otherArray[j], gate, 2);
                        output_imag[i] = MULIMAG1(input, i + j, gate, 3) + MULIMAG1(input, otherArray[j], gate, 2);
                    }
                }
            }
            free(input->real);
            free(input->imag);
            input->real = output_real;
            input->imag = output_imag;
        } else {
            if(output_real != nullptr) {
                free(output_real);
            } else if (output_imag != nullptr) {
                free(output_imag);
            }
            const __m256i rightBitsMask = _mm256_set1_epi64x((1LL << shifts_to_target_bit) - 1LL);
            const __m256i leftBitsMask = _mm256_set1_epi64x(~((1LL << shifts_to_target_bit) - 1LL));
            const __m256i bitFlip = _mm256_set1_epi64x((1LL << shifts_to_target_bit));

#pragma  omp parallel for
            for (qint i = 0; i < input->n / 2; i += 4) {

                alignas(32) qint iArray [4] = {i, i + 1, i + 2, i + 3};
                _mm_prefetch((const char *) iArray, _MM_HINT_T0);
                const __m256i iRegister = _mm256_load_si256((const __m256i_u *) iArray);
                const __m256i rightBits = _mm256_and_si256(iRegister, rightBitsMask);
                __m256i leftBits = _mm256_and_si256(iRegister, leftBitsMask);
                leftBits = _mm256_slli_epi64(leftBits, 1);
                const __m256i indices = _mm256_or_si256(leftBits, rightBits);
                const __m256i other = _mm256_xor_si256(indices, bitFlip);

                qint combinedArray[4];
                qint otherArray[4];

                _mm256_storeu_si256((__m256i_u *) combinedArray, indices);
                _mm256_storeu_si256((__m256i_u *) otherArray, other);

                for (int j = 0; j < 4; ++j) {
                    const qint combinedIndex = combinedArray[j];
                    const qint otherIndex = otherArray[j];

                    const qfloat combinedVal_real = input->real[combinedIndex];
                    const qfloat combinedVal_imag = input->imag[combinedIndex];
                    const qfloat otherVal_real = input->real[otherIndex];
                    const qfloat otherVal_imag = input->imag[otherIndex];

                    input->real[combinedIndex] = MULREAL2(combinedVal_real, combinedVal_imag, gate, 0) + MULREAL2(otherVal_real, otherVal_imag, gate, 1);
                    input->imag[combinedIndex] = MULIMAG2(combinedVal_real, combinedVal_imag, gate, 0) + MULIMAG2(otherVal_real, otherVal_imag, gate, 1);

                    input->real[otherIndex] = MULREAL2(combinedVal_real, combinedVal_imag, gate, 2) + MULREAL2(otherVal_real, otherVal_imag, gate, 3);
                    input->imag[otherIndex] = MULIMAG2(combinedVal_real, combinedVal_imag, gate, 2) + MULIMAG2(otherVal_real, otherVal_imag, gate, 3);
                }

            }
        }
    }
}

/// apply_gate_real applies a gate to the target qubit
/// \param reg the quantum register that contains the qubits
/// \param gate the gate that will be applied
/// \param target the index of the target qubit
void apply_gate_real(QuantumRegister *restrict const reg, MatrixReal const *restrict const gate, const uint target) {
    if (reg->qubit_type == REAL) {
        _apply_gate_real(&reg->q_register.real_reg, gate, target, reg->qubit_count);
    } else {
        const MatrixComplex gate_complex = {gate->n, gate->real, (double *restrict) ZERO};
        _apply_gate_complex(&reg->q_register.complex_reg, &gate_complex, target, reg->qubit_count);
    }
}

/// apply_gate_complex applies a gate to the target qubit
/// \param reg the quantum register that contains the qubits
/// \param gate the gate that will be applied
/// \param target the index of the target qubit
void apply_gate_complex(QuantumRegister *restrict const reg, MatrixComplex const *restrict const gate, const uint target) {
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
    _apply_gate_complex(&reg->q_register.complex_reg, gate, target, reg->qubit_count);
}
