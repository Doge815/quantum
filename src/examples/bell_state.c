#include "../Simulation/Simulation.h"
#include <stdio.h>
#include <time.h>

int main(void) {
    //seed the random number generator with the current time
    srand(time(NULL));

    //initialize 2 qubits, both as |0>
    ColumnVectorReal const* initial_state [2] = {&QG_Zero, &QG_Zero};

    //create the quantum register
    QuantumRegister qReg = create_register(initial_state, 2);

    //apply the hadamard-gate to the first qubit
    //the first qubit is now in a superposition
    apply_gate_real(&qReg, &QG_H, 0);

    //apply the controlled-not-gate to the second qubit, with the first as the control-qubit
    //the second qubit will be flipped, if the second qubit is |1>
    apply_cX(&qReg, 0, 1);

    //the chances of measuring |00> and |11> are both 50%
    //measure both qubits
    ColumnVectorReal const ** measured = measure_to_qubits(&qReg);

    //print the state of the first qubit
    if(measured[0] == &QG_Zero)
    {
        printf("0\n");
    } else {
        printf("1\n");
    }
    //print the state of the second qubit
    if(measured[1] == &QG_Zero)
    {
        printf("0\n");
    } else {
        printf("1\n");
    }

    //delete the quantum register
    delete_register(&qReg);

    //delete the measurement output
    free(measured);
    return 0;
}

