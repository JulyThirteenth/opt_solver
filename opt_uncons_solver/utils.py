from typing import List

def _plot_convergence(costs: List[float], method_name: str) -> None:
    """Plots convergence curve for optimization results."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(costs)), costs, marker="o", markersize=4)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title(f"Convergence Curve: {method_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()