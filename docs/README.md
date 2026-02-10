# Quantifying Resilience in the Amazon Rainforest

**Authors:** Theresa Wenzel & Tim Renner  
**Course:** Computational Earth System Science Seminar  
**Date:** February 10, 2026

---

## 1. Project Overview
This project investigates the stability and resilience of the Amazon Rainforest using non-linear dynamic models, motivated by recent discussions on early-warning signals of ecosystem tipping points. We compare traditional **Linear Stability Analysis** with the global **Basin Stability** concept. The comparison shows that basin stability provides a more intuitive measure of resilience in multi-stable systems than local linear analysis alone.


The core of the project is the implementation and comparison of two models:
* **Conceptual Model (Menck et al., 2013):** A bistable 1D-model driven by aridity.
* **Coupled Model (Van Nes et al., 2014):** A complex model including vegetation-atmosphere feedback, showing tristability (Forest, Savanna, Treeless).

We chose a **Monte Carlo approach** because a full grid scan of the coupled model became computationally infeasible at higher resolutions. This also allows more efficient computation if the model gets expanded to include more variables.


## 2. Project Structure
The project is organized as follows:

    .
    ├── main.py                # Main Python script (Simulation & Plotting)
    ├── README.md              # Project Documentation
    ├── results/               # Output directory for generated plots
    │   ├── tests.png          # Grid Scan ("Ground Truth")
    │   ├── monte_carlo_resilience.png  # Monte Carlo Results
    │   ├── trajectories.png   # Time series visualization
    │   └── ...
    └── presentation/          # LaTeX source code for the slides
        ├── main.tex
        └── ...

## 3. Requirements & Installation
To run the simulations, a Python environment with the following scientific libraries is required:

* `numpy`
* `matplotlib`
* `scipy`
* `tqdm` (for progress bars)

You can install the dependencies via pip:

    pip install numpy matplotlib scipy tqdm

## 4. How to Run
Execute the main script to run all simulations and generate the plots in the `results/` folder.

    python main.py

*Note: The script utilizes `multiprocessing` to parallelize the Monte Carlo sampling. Depending on your CPU, the run might take a few moments.*

## 5. Implementation Details & Extensions
We based our code on the mathematical models described in **Menck et al. (2013)** and **Van Nes et al. (2014)**. However, we extended the implementation significantly:

### A. Extensions
1.  **Monte Carlo Basin Estimation:** Instead of relying solely on analytical solutions or expensive grid scans, we implemented a statistical Monte Carlo approach to estimate the volume of the basin of attraction.
2.  **Validation via "Ground Truth":** We implemented a high-resolution grid scan (see `grid_scan.png`) to validate that our Monte Carlo approximation converges to the correct stability landscape.
3.  **Visualizations:** We added custom plotting functions for vector fields (quiver plots) and time-series trajectories to visualize physical phenomena like **Critical Slowing Down** and **Hysteresis**.

### B. Numerical Methods
* **Solver:** `scipy.integrate.odeint` (LSODA) for integrating the coupled ODEs.
* **Root Finding:** `scipy.optimize.brentq` for bifurcation analysis.