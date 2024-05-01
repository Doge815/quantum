#include "../Simulation/Simulation.h"
#include <stdio.h>
#include <time.h>


void deutsch_jozsa(int input_qubits, void(*oracle)(QuantumRegister*)) {

    //create a register with 'input_qubits' input qubits plus an output qubit
    const ColumnVectorReal* initial_state [input_qubits + 1];

    //set the input qubits to |0>
    for (int i = 0; i < input_qubits; ++i) {
        initial_state[i] = &QG_Zero;
    }

    //set the output qubit to |1>
    initial_state[input_qubits] = &QG_One;

    //create the quantum register
    QuantumRegister qReg = create_register(initial_state, input_qubits + 1);

    //bring all qubits into a superposition
    for (int i = 0; i < input_qubits + 1; ++i) {
        apply_gate_real(&qReg, &QG_H, i);
    }

    //apply the provided oracle
    oracle(&qReg);

    //bring all input qubits into a base position
    for (int i = 0; i < input_qubits; ++i) {
        apply_gate_real(&qReg, &QG_H, i);
    }

    //measure all qubits
    ColumnVectorReal const ** measured = measure_to_qubits(&qReg);

    //if all (or a single) input qubits are |0>, the oracle is constant
    if(measured[0] == &QG_Zero) {
        printf("constant\n");
    }
    //if all (or a single) input qubits are |1>, the oracle is balanced
    else {
        printf("balanced\n");
    }

    //delete the quantum register
    delete_register(&qReg);

    //delete the measurement output
    free(measured);
}

void oracle_constant1(__attribute__((unused)) QuantumRegister* qReg){
    //don't modify the output qubit
}

void oracle_constant2(QuantumRegister* qReg){
    //flip the output qubit
    apply_gate_real(qReg, &QG_X, qReg->qubit_count - 1);
}

void oracle_balanced1(QuantumRegister* qReg){
    //flip the output qubit for half of all possible input bit combinations
    for (int i = 0; i < qReg->qubit_count; ++i) {
        apply_cX(qReg, i, qReg->qubit_count - 1);
    }
}

void oracle_balanced2(QuantumRegister* qReg){
    //flip the output qubit for half of all possible input bit combinations
    for (int i = 0; i < qReg->qubit_count; ++i) {
        apply_cX(qReg, i, qReg->qubit_count - 1);
    }

    //flip the output qubit again
    apply_gate_real(qReg, &QG_X, qReg->qubit_count - 1);
}

int main(void) {
    //seed the random number generator with the current time
    srand(time(NULL));

    const int input_qubits = 5;

    //test the 4 oracles
    deutsch_jozsa(input_qubits, oracle_constant1);
    deutsch_jozsa(input_qubits, oracle_constant2);
    deutsch_jozsa(input_qubits, oracle_balanced1);
    deutsch_jozsa(input_qubits, oracle_balanced2);

    return 0;
}


