#include "../Simulation/Simulation.h"
#include <stdio.h>
#include <time.h>

int main(void) {
    //seed the random number generator with the current time
    srand(time(NULL));

    //initialize a single qubit as |0>
    ColumnVectorReal const* initial_state [1] = {&QG_Zero};

    //create a quantum register
    QuantumRegister qReg = create_register(initial_state, 1);

    //apply the hadamard-gate to the qubit
    apply_gate_real(&qReg, &QG_H, 0);
    //the qubit is now in a superposition, the chances of measuring |0> and |1> are both 50%

    //measure the qubit
    ColumnVectorReal const ** measured = measure_to_qubits(&qReg);

    //print the state of the qubit
    if(measured[0] == &QG_Zero)
    {
        printf("Head!\n");
    } else {
        printf("Tails!\n");
    }

    //delete the quantum register
    delete_register(&qReg);

    //delete the measurement output
    free(measured);
    return 0;
}
