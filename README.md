# simul_fit: Bayesian Isochrone Fitting for Stellar Clusters

`simul_fit.py` is a high-performance Python tool designed for the simultaneous multi-filter isochrone fitting of stellar clusters. By employing a fully vectorized grid-search combined with Bayesian parameter estimation, the software determines the fundamental astrophysical parameters of a cluster:

- **Age** ($\log t$)
- **Metallicity** ($Z$)
- **Distance Modulus** ($DM$)
- **Reddening** ($E(B-V)$)

The code is optimized for computational efficiency and statistical robustness, making it suitable for large stellar samples and modern survey data like Gaia DR3.

---

## Contributors

- **Sait Sarrajoğlu** [https://github.com/speletry] – Independent Researcher
*   ORCID: https://orcid.org/0009-0009-5906-9313

* **Zahra Al** [https://github.com/zahraa1996al] – Istanbul University, Department of Astronomy and Space Sciences
*   ORCID: https://orcid.org/0009-0004-4134-3752  

**License:** MIT

---

## 🌌 Features

- **Dual Photometric Support**
  - **Gaia System:** Optimized for $G$, $G_{BP}$, and $G_{RP}$ filters.
  - **Standard UBVRI System:** Supports $U, B, V, R, I$ filters with error-weighting.
- **High-Performance Optimization**
  - Fully vectorized $\chi^2$ minimization using NumPy broadcasting.
  - "Point-to-Pipe" distance matching for thousands of stars across model grids.
- **Robust Bayesian Framework**
  - Likelihood-based posterior construction ($P \propto \exp(-\Delta\chi^2/2)$).
  - **Weighted Resampling:** Uses Monte Carlo-style resampling (20,000 samples) to simulate continuous probability distributions.
  - **Anti-Aliasing Jitter:** Applies Gaussian noise to resampled points for smooth, publication-ready confidence contours.
- **Automated Workflow**
  - **Smart Loading:** Automatically detects and processes the newest `.csv` file in the input directory.
  - **Path Agnostic:** Runs anywhere without hardcoded user directories.

---

## 🛠 Requirements

Ensure you have Python 3.9 or higher installed. Install dependencies via pip:

```bash
pip install numpy pandas matplotlib corner

---

📂 Project Structure
Maintain the following directory structure for the script to function correctly:

Plaintext
IsochroneProject/
├── simul_fit.py          # Main analysis script
├── config.txt            # (Optional) Grid boundary settings
├── input/                # Place observational .csv files here (Newest is processed)
├── output/               # Results and plots are saved here automatically
└── isochrone/            # Place PARSEC library files here:
    ├── isochronesG.txt   # Required for Gaia mode
    └── isochronesJ.txt   # Required for UBVRI mode

---

📥 Input Data & Configuration

1. Observational Data

Place your cluster data in input/.

Gaia Mode: Requires columns G, BP, RP.

UBVRI Mode: Requires columns U, B, V, R, I and their corresponding error columns (e.g., eU, eB, eV).

2. Grid Tuning (config.txt)

Adjust the search space without editing the source code. If this file is missing, the script defaults to internal presets.

Parameter     Description                        Range  #Example!

DM            Distance Modulus $(m-M_0)$         7.0 to 10.0

E_MIN/MAX     Interstellar Reddening $E(B-V)$    0.1 to 0.30

AGE_MIN/MAX   Age in $\log(\text{years})$        7.0 to 9.9

Z_MIN/MAX     Metallicity (Mass fraction)        0.001 to 0.030  

---

Running the Code

Open a terminal in the project directory and run the command for your preferred system:

Bash
# For Gaia Data
python simul_fit.py gaia

# For Johnson-Cousins Data
python simul_fit.py ubv

---

📊 Output & Visualization
The script generates three primary files in the output/ folder, prefixed by your input filename:

Statistical Results (_results.csv): Contains the 16th, 50th (median), and 84th percentiles. It also calculates the Physical Distance (pc).

Covariance Matrix (_covariance.csv): Details the mathematical correlation between parameters.

Corner Plot (_corner.png): A high-resolution visualization showing 1D histograms and 2D confidence contours (68% and 95%) with a custom Cyan Median Overlay.

---

🔬 Method Summary
Minimization: Calculates the minimum χ 
2
  distance between each star and the model isochrone.

Error Floor: Applies a 0.1 mag floor to ensure the fit is not biased by underestimated instrumental errors.

Resampling: Draws 20,000 samples from the Likelihood surface to generate marginalized distributions.

Jittering: Applies Gaussian noise to resampled points to "smooth" the discrete grid boundaries for visual analysis in the corner plot.

---

📌 Citation

If you use this software in your research, please cite it as:

Al, Z., & Sarrajoğlu, S. (2026). simul_fit: A Bayesian Isochrone Fitting Tool for Stellar Clusters (v1.0.0). Zenodo. [![DOI](https://zenodo.org/badge/1197720435.svg)](https://doi.org/10.5281/zenodo.19360379)

The development version is available on GitHub at: [https://github.com/speletry/simul_fit]
