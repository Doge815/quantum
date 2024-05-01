#include "Simulation.h"

static inline ColumnVectorReal __attribute__((always_inline)) _create_input(const ColumnVectorReal *const QG[], int count) {
    //calculate the index of the only '1' in the vector
    qint index = 0;

    //iterate over all qubits
    for (qint i = 0; i < count; ++i) {
        //for each qubit, the number of zeros in front of the 1 doubles
        index *= 2;
        //if the current qubit is set to |1>, increment the index
        if (QG[i] == &QG_One) {
            index += 1;
        }
    }

    //allocate the vector
    ColumnVectorReal new = VectorRealCreateZero(1L << count);

    //set the '1'
    new.real[index] = 1;
    return new;
}

/// create_register allocates a quantum register and initializes all qubits with the provided base states
/// \param base_states the base states of the qubits
/// \param count the number of qubits
/// \return the created quantum register
QuantumRegister create_register(const ColumnVectorReal *const base_states[], int count) {
    return (QuantumRegister) {count, REAL, {_create_input(base_states, count)}};
}

/// delete_register destroys a quantum register and frees all it's allocated resources
/// \param req the quantum register to be deleted
void delete_register(QuantumRegister const * restrict const req) {
    if(req->qubit_type == REAL)
    {
        free(req->q_register.real_reg.real);
    } else {
        free(req->q_register.complex_reg.real);
        free(req->q_register.complex_reg.imag);
    }
}
