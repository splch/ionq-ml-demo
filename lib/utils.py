import numpy as np
from io import BytesIO
from PIL import Image
from IPython.display import display, Latex
from sympy import Matrix, latex
from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.visualization import plot_bloch_multivector, plot_histogram


def random_angle():
    """Generate a random angle between -2π and 2π."""
    return np.random.uniform(-2 * np.pi, 2 * np.pi)


def plot_circuit(circuit: QuantumCircuit):
    temp_circuit = circuit.remove_final_measurements(inplace=False)
    state = Statevector.from_instruction(temp_circuit)

    # Capture circuit plot
    circuit_img = BytesIO()
    fig1 = circuit.draw("mpl")
    fig1.savefig(circuit_img)
    circuit_img.seek(0)

    # Capture Bloch sphere plot
    bloch_img = BytesIO()
    fig2 = plot_bloch_multivector(state)
    fig2.savefig(bloch_img)
    bloch_img.seek(0)

    # Display side by side
    circuit_pil = Image.open(circuit_img)
    bloch_pil = Image.open(bloch_img)
    new_im = Image.new(
        "RGB",
        (
            circuit_pil.width + bloch_pil.width,
            max(circuit_pil.height, bloch_pil.height),
        ),
        (255, 255, 255),  # White background
    )
    new_im.paste(circuit_pil, (0, (bloch_pil.height - circuit_pil.height) // 2))
    new_im.paste(bloch_pil, (circuit_pil.width, 0))
    display(new_im)
    display(Latex(f"$${latex(Matrix(Operator(temp_circuit)))}$$"))


def is_measured(circuit: QuantumCircuit) -> bool:
    for op, _, _ in circuit.data:
        if op.name == "measure":
            return True
    return False


def run_circuit(circuit: QuantumCircuit, simulator, **kwargs) -> dict:
    temp_circuit = circuit.copy()
    if not is_measured(temp_circuit):
        # Measure the circuit
        temp_circuit.measure_all()

    # Transpile for simulator
    circuit = transpile(temp_circuit, simulator)

    # Run and get counts
    result = simulator.run(temp_circuit, **kwargs).result()
    counts = result.get_counts(temp_circuit)
    display(plot_histogram(counts))
