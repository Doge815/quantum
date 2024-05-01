#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include "Simulation.h"

static inline qint __attribute__((always_inline)) _measure_all_to_output_real(ColumnVectorReal *restrict const input) {
    //pick a random number between 0 and 1
    //the first index for which the sum of all previous probabilities is greater than the random number will be the new state of the register.
    qfloat random = (qfloat) rand() / (qfloat) RAND_MAX;
    qfloat sum = 0;

    //if the quantum register has more than 19 qubits, iterate over the indices in parallel
    if (input->n > 524288) { //more than 19 qubits
        const int threads = omp_get_max_threads();

        //to prevent multiple threads from writing to the same variable, create one for each thread
        qfloat sums[threads];

#pragma omp parallel for
        for (int thread = 0; thread < threads; ++thread) {
            //set the sum for the current thread to 0
            sums[thread] = 0;

            //iterate over all the states that are assigned to the current thread
            //since this code uses simd, each step contains 4 probabilities
            for (int i = thread * 4; i < input->n; i += (4 * threads)) {

                //if there are 4 or more indices left
                if (i + 4 < input->n) {

                    //load the 4 probabilities into a register
                    const __m256d real = _mm256_loadu_pd(&(input->real[i]));
                    //calculate the square
                    //since the number is real, it is it's own abs after squaring
                    const __m256d squared = _mm256_mul_pd(real, real);

                    //horizontally add the squares together
                    __m128d low = _mm256_castpd256_pd128(squared);
                    __m128d high = _mm256_extractf128_pd(squared, 1); // high 128
                    low = _mm_add_pd(low, high);     // reduce down to 128
                    __m128d high64 = _mm_unpackhi_pd(low, low);

                    //add the sum of the 4 squares to the sum of the thread
                    sums[thread] += _mm_cvtsd_f64(_mm_add_sd(low, high64));  // reduce to scalar

                } else {
                    //add the remaining indices
                    for (int j = i; j < input->n; ++j) {
                        //the compiler optimizes pow(x, 2) to x * x
                        sums[thread] += pow(input->real[j], 2);
                    }
                }
            }
        }
        //iterate over the sums calculated by the threads
        for (int i = 0; i < threads; ++i) {
            //add the sum of the thread to the global sum
            sum += sums[i];

            //if the sum is greater than the random value, remove the last sum again and add each contained squared probability one by one
            if (sum > random) {
                sum -= sums[i];

                //use simd again
                for (int j = i * 4; j < input->n; j += (4 * threads)) {
                    if (j + 4 < input->n) {

                        const __m256d real = _mm256_loadu_pd(&(input->real[j]));
                        const __m256d squared = _mm256_mul_pd(real, real);

                        //save the values to an array instead of calculating the horizontal sum
                        double out[4];
                        _mm256_storeu_pd(out, squared);
                        //add the squared probabilities
                        for (int k = 0; k < 4; ++k) {
                            sum += out[k];
                            if (sum >= random) {
                                //if the sum is greater than the random number, the right index was found

                                //set the register to zeros
                                memset(input->real, 0, sizeof(qfloat) * input->n);

                                //set a single '1' to the current position to normalize the vector
                                input->real[j + k] = 1;
                                return j + k;
                            }
                        }
                    } else {
                        //iterate over the remaining probabilities
                        for (; j < input->n; ++j) {
                            sum += pow(input->real[j], 2);
                            if (sum >= random) {
                                //index was found
                                memset(input->real, 0, sizeof(qfloat) * input->n);
                                input->real[j] = 1;
                                return j;
                            }
                        }
                    }
                }
            }
        }
    } else {
        //if there are 19 qubits or less
        //iterate over all probabilities without parallelization
        for (qint i = 0; i < input->n; ++i) {
            //add the squared probability
            sum += pow(input->real[i], 2);
            if (sum >= random) {
                //index was found
                memset(input->real, 0, sizeof(qfloat) * input->n);
                input->real[i] = 1; //normalize

                return i;
            }
        }
    }
    //if this event occurs, the vector length was less than 1
    return -1;
}

static inline qint __attribute__((always_inline)) _measure_all_to_output_complex(ColumnVectorComplex *restrict const input) {
    //the same as the real function, just with complex probabilities
    qfloat random = (qfloat) rand() / (qfloat) RAND_MAX;
    qfloat sum = 0;
    if (input->n > 524288) { //more than 19 qubits
        const int threads = omp_get_max_threads();
        qfloat sums[threads];
#pragma omp parallel for
        for (int thread = 0; thread < threads; ++thread) {

            sums[thread] = 0;

            for (int i = thread * 4; i < input->n; i += (4 * threads)) {
                if (i + 4 < input->n) {
                    const __m256d real = _mm256_loadu_pd(&(input->real[i]));
                    const __m256d real_squared = _mm256_mul_pd(real, real);
                    const __m256d imag = _mm256_loadu_pd(&(input->imag[i]));
                    const __m256d imag_squared = _mm256_mul_pd(imag, imag);
                    //add the real and imaginary square together to get the abs of the square of the probability
                    const __m256d squared_sum = _mm256_add_pd(real_squared, imag_squared);

                    __m128d low = _mm256_castpd256_pd128(squared_sum);
                    __m128d high = _mm256_extractf128_pd(squared_sum, 1); // high 128
                    low = _mm_add_pd(low, high);     // reduce down to 128
                    __m128d high64 = _mm_unpackhi_pd(low, low);
                    sums[thread] += _mm_cvtsd_f64(_mm_add_sd(low, high64));  // reduce to scalar
                } else {
                    for (int j = i; j < input->n; ++j) {
                        sums[thread] += pow(input->imag[j], 2) + pow(input->real[j], 2);
                    }
                }
            }
        }
        for (int i = 0; i < threads; ++i) {
            sum += sums[i];
            if (sum > random) {
                sum -= sums[i];
                for (int j = i * 4; j < input->n; j += (4 * threads)) {
                    if (j + 4 < input->n) {

                        const __m256d real = _mm256_loadu_pd(&(input->real[j]));
                        const __m256d real_squared = _mm256_mul_pd(real, real);
                        const __m256d imag = _mm256_loadu_pd(&(input->real[j]));
                        const __m256d imag_squared = _mm256_mul_pd(imag, imag);
                        const __m256d squared_summed = _mm256_add_pd(imag_squared, real_squared);

                        double out[4];
                        _mm256_storeu_pd(out, squared_summed);
                        for (int k = 0; k < 4; ++k) {
                            sum += out[k];
                            if (sum >= random) {
                                memset(input->real, 0, sizeof(qfloat) * input->n);
                                memset(input->imag, 0, sizeof(qfloat) * input->n);
                                input->real[j + k] = 1; //normalize
                                return j + k;
                            }
                        }
                    } else {
                        for (; j < input->n; ++j) {
                            sum += pow(input->imag[j], 2) + pow(input->real[j], 2);
                            if (sum >= random) {
                                memset(input->real, 0, sizeof(qfloat) * input->n);
                                memset(input->imag, 0, sizeof(qfloat) * input->n);
                                input->real[j] = 1; //normalize
                                return j;
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (qint i = 0; i < input->n; ++i) {
            sum += pow(input->real[i], 2) + pow(input->imag[i], 2);
            if (sum >= random) {
                memset(input->real, 0, sizeof(qfloat) * input->n);
                memset(input->imag, 0, sizeof(qfloat) * input->n);
                input->real[i] = 1; //normalize
                return i;
            }
        }
    }
    return -1;
}

static inline ColumnVectorReal const * __attribute__((always_inline)) _measure_single_real(ColumnVectorReal *restrict const input, const uint count, const int index) {
    //the idea of measuring a single qubit is very similar to the measurement of all qubits
    //but instead of iterating over all probabilities, only the indices where the measured qubit is |0> are used
    //if the sum of those probabilities is greater than a random number, than the qubit is measured as |0>

    qfloat sum = 0;
    qfloat random = (qfloat) rand() / (qfloat) RAND_MAX;

    //the number of indices where the measured qubit |0> that lie in a row
    const qint stepSize = (1 << (count - index - 1));

    //the number of times a row of probability is added and a row is skipped
    const qint steps = input->n / (2 * stepSize);
    {
        qint ind = 0;
        for (int i = 0; i < steps; ++i) {
            //add the row of probabilities
            for (qint j = 0; j < stepSize; ++j) {
                sum += pow(input->real[ind], 2);
                ind++;
            }
            //skip a row
            ind += stepSize;
        }
    }

    //if the sum is greater than the random value, the qubit is measured as |0>
    if (sum > random)
    {
        //iterate over the probabilities again
        qint ind = 0;
        for (int i = 0; i < steps; ++i) {
            for (int j = 0; j < stepSize; ++j) {
                //the sum is the length of the vector, to normalize it, every entry needs to be divided by the length
                input->real[ind] /= sum;
                ind++;
            }
            //set the probabilities of the indices where the measured qubit is |1> to 0
            memset(&input->real[ind], 0, sizeof(qfloat) * stepSize);
            ind += stepSize;
        }
        //return the state of the qubit
        return &QG_Zero;
    }
    //if the sum is greater than the random value, the qubit is measured as |0>
    else {
        //iterate over the probabilities again
        qint ind = 0;
        for (qint i = 0; i < steps; ++i) {
            //set the probabilities of the indices where the measured qubit is |0> to 0
            memset(&input->real[ind], 0, sizeof(qfloat) * stepSize);
            ind += stepSize;
            for (qint j = 0; j < stepSize; ++j) {
                //the sum is the length of the vector, to normalize it, every entry needs to be divided by the length
                input->real[ind] /= (1-sum);
                ind++;
            }
        }
        return &QG_One;
    }
}

static inline ColumnVectorReal const * __attribute__((always_inline)) _measure_single_complex(ColumnVectorComplex *restrict const input, const uint count, const int index) {
    //the same as the real function, just with complex probabilities
    qfloat sum = 0;
    qfloat random = (qfloat) rand() / (qfloat) RAND_MAX;

    const qint stepSize = (1 << (count - index - 1));
    const qint steps = input->n / (2 * stepSize);
    {
        qint ind = 0;
        for (int i = 0; i < steps; ++i) {
            for (qint j = 0; j < stepSize; ++j) {
                sum += pow(input->imag[ind], 2) + pow(input->real[ind], 2);
                ind++;
            }
            ind += stepSize;
        }
    }
    if (sum > random) //qubit is 0 {
    {
        qint ind = 0;
        for (int i = 0; i < steps; ++i) {
            for (int j = 0; j < stepSize; ++j) {
                input->real[ind] /= sum;
                input->imag[ind] /= sum;
                ind++;
            }
            memset(&input->real[ind], 0, sizeof(qfloat) * stepSize);
            memset(&input->imag[ind], 0, sizeof(qfloat) * stepSize);
            ind += stepSize;
        }
        return &QG_Zero;
    } else {
        qint ind = 0;
        for (qint i = 0; i < steps; ++i) {
            memset(&input->real[ind], 0, sizeof(qfloat) * stepSize);
            memset(&input->imag[ind], 0, sizeof(qfloat) * stepSize);
            ind += stepSize;
            for (qint j = 0; j < stepSize; ++j) {
                input->real[ind] /= (1-sum);
                input->real[ind] /= (1-sum);
                ind++;
            }
        }
        return &QG_One;
    }
}

static inline bool const * __attribute__((always_inline)) _measured_to_booleans(qint index, const uint size) {
    //crates an array of booleans that represents the state of a quantum register after a measurement

    //allocate an array with a boolean for each qubit
    bool * const output = malloc(size * sizeof(bool));

    //iterate over the bits of the index, that was returned by the measurement
    for (int l = 0; l < size; ++l) {

        //if the last bit of the index is set, set the current boolean of the output array to true
        if (index & 0x01)
        {
            output[size - 1 - l] = true;
        } else {

            output[size - 1 - l] = false;
        }

        //shift the index to the right
        index >>= 1;
    }
    return output;
}

static inline ColumnVectorReal const ** __attribute__((always_inline)) _measured_to_qubits(qint index, const uint size) {
    //the same as the boolean function, just with ColumnVectorReal references
    const ColumnVectorReal ** output = malloc(size * sizeof(ColumnVectorReal*));
    for (int l = 0; l < size; ++l) {
        if (index & 0x01) //get the last bit
        {
            output[size - 1 - l] = &QG_One;
        } else {

            output[size - 1 - l] = &QG_Zero;
        }
        index >>= 1; //move to next bit
    }
    return output;
}

/// measure_all measures all the qubits and forces them in a base state
/// \param reg quantum register to be measured
/// \return index of the state of the quantum register, can be converted to an array of vectors or booleans
qint measure_all(QuantumRegister *restrict const reg) {
    if (reg->qubit_type == REAL) {
        return _measure_all_to_output_real(&reg->q_register.real_reg);
    } else {
        return _measure_all_to_output_complex(&reg->q_register.complex_reg);
    }
}

/// measure_to_booleans allocates an array of booleans that represents the states of all the qubits of a quantum register
/// \param req quantum register that was measured
/// \param measurement index returned by the measurement
/// \return array of booleans with one boolean for each qubit of the register, where true represents a qubit measured as |1> and false represents a qubit measured as |0>
bool const * measurement_to_booleans(QuantumRegister * restrict const req, qint measurement) {
    return _measured_to_booleans(measurement, req->qubit_count);
}

/// measure_to_vectors allocates an array of ColumnVectorReal references that represents the states of all the qubits of a quantum register
/// \param req quantum register that was measured
/// \param measurement index returned by the measurement
/// \return array of ColumnVectorReal references with one ColumnVectorReal reference for each qubit of the register, where &QG_One represents a qubit measured as |1> and &QG_Zero represents a qubit measured as |0>
ColumnVectorReal const ** measurement_to_vectors(QuantumRegister * restrict const req, qint measurement) {
    return _measured_to_qubits(measurement, req->qubit_count);
}

/// measure_single measures a single qubit and forces it into a base state
/// \param reg the quantum register to be measured
/// \param index index of the qubit to be measured
/// \return ColumnVectorReal reference, where &QG_One represents that the qubit was measured as |1> and &QG_Zero represents that the qubit was measured as |0>
ColumnVectorReal const * measure_single(QuantumRegister* restrict const reg, int index) {
    if (reg->qubit_type == REAL) {
        return _measure_single_real(&reg->q_register.real_reg, reg->qubit_count, index);
    } else {
        return _measure_single_complex(&reg->q_register.complex_reg, reg->qubit_count, index);
    }
}
