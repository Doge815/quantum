#include "Simulation/Simulation.h"
#include <time.h>

int main(void) {
    srand(time(NULL));
    ColumnVectorReal const* initial_state [1] = {&QG_Zero};
    QuantumRegister qReg = create_register(initial_state, 1);

    //
    //TODO
    //

    ColumnVectorReal const ** measured = measure_to_qubits(&qReg);

    delete_register(&qReg);
    free(measured);
    return 0;
}
