import pandas as pd
import numpy as np
import corner
import matplotlib.pyplot as plt
from io import StringIO
import os
import sys

# === 1. Load Data (Updated for Newest File) ===

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(BASE_DIR, "input")
output_dir = os.path.join(BASE_DIR, "output")
os.makedirs(output_dir, exist_ok=True)

# 1. Get all CSV files with their full paths
files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]

if not files:
    raise FileNotFoundError(f"❌ No CSV files found in: {input_dir}")

# 2. Sort files by modification time (mtime) - Newest last
files.sort(key=os.path.getmtime)

# 3. Pick the last one in the sorted list (the newest)
input_file = files[-1]
file_name_only = os.path.basename(input_file)

print(f"📅 Newest file detected: {file_name_only}")
print(f"📂 Automatically loading: {file_name_only}")

# Extract base name for outputs
base_name = os.path.splitext(file_name_only)[0]

obs_df = pd.read_csv(input_file)

# ==========================================================
# 2. USER SYSTEM SELECTION
# ==========================================================

if len(sys.argv) != 2:
    print("Usage: python simul_fit.py [ubv | gaia]")
    sys.exit()

mode = sys.argv[1].lower()

if mode == "ubv":
    print("📘 UBVRI system selected")

    library_path = os.path.join(BASE_DIR, "isochrone", "isochronesJ.txt")

    ISO_COLS = ['Umag','Bmag','Vmag','Rmag','Imag']
    OBS_COLS = ['U','B','V','R','I']
    ERR_COLS = ['eU','eB','eV','eR','eI']
   
    obs_df = obs_df.dropna(subset=OBS_COLS + ERR_COLS).reset_index(drop=True)

    A_coeff = np.array([4.82, 4.10, 3.10, 2.54, 1.85])
    use_errors = True

elif mode == "gaia":
    print("🌌 Gaia system selected")

    library_path = os.path.join(BASE_DIR, "isochrone", "isochronesG.txt")

    ISO_COLS = ['Gmag','G_BPmag','G_RPmag']
    OBS_COLS = ['G','BP','RP']
    ERR_COLS = None   # Gaia → no magnitude errors
    
    obs_df = obs_df.dropna(subset=OBS_COLS).reset_index(drop=True)

    # Gaia extinction coefficients (PARSEC typical values)
    A_coeff = np.array([2.74, 3.374, 2.035])
    use_errors = False

else:
    raise ValueError("❌ Invalid choice. Type UBV or Gaia.")

# === 3. Grid Settings (Refined for Symmetry & Centering) ===
# === 3. Grid Settings (Loaded from config.txt) ===

def load_config(filepath):
    # Standard defaults in case the file is missing or broken
    conf = {
        'DM_MIN': 7.0, 'DM_MAX': 14.0, 'DM_STEP': 0.5,
        'E_MIN': 0.001, 'E_MAX': 0.10, 'E_STEP': 0.01,
        'AGE_MIN': 7.0, 'AGE_MAX': 10.0,
        'Z_MIN': 0.001, 'Z_MAX': 0.040
    }
    
    if not os.path.exists(filepath):
        print("⚠️ config.txt not found. Using internal defaults.")
        return conf

    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Ignore comments and empty lines
                if ':' in line and not line.strip().startswith('#'):
                    key, val = line.split(':')
                    conf[key.strip()] = float(val.strip())
        print("⚙️ Configuration loaded successfully from config.txt")
    except Exception as e:
        print(f"⚠️ Error reading config.txt: {e}. Using defaults.")
        
    return conf

# Execute the loader
config = load_config(os.path.join(BASE_DIR, "config.txt"))

# Assign to variables
DM_MIN, DM_MAX, DM_STEP = config['DM_MIN'], config['DM_MAX'], config['DM_STEP']
E_MIN, E_MAX, E_STEP = config['E_MIN'], config['E_MAX'], config['E_STEP']
AGE_MIN, AGE_MAX = config['AGE_MIN'], config['AGE_MAX']
Z_MIN, Z_MAX = config['Z_MIN'], config['Z_MAX']

# Generate the actual numerical grids for the math loops
dm_grid = np.arange(DM_MIN, DM_MAX + DM_STEP, DM_STEP)
e_grid = np.arange(E_MIN, E_MAX + E_STEP, E_STEP)

print(f"✅ Search Grid: DM({DM_MIN}-{DM_MAX}), E(B-V)({E_MIN}-{E_MAX}), Age({AGE_MIN}-{AGE_MAX})")

# === 4. Functions ===
def load_parsec_library(filepath, z_min, z_max, age_min, age_max):
    print(f"📂 Reading library: {filepath}")
    clean_lines = []
    header = None
    with open(filepath, 'r') as f:
        for line in f:
            if 'Zini' in line and 'logAge' in line:
                if header is None:
                    header = line.replace('#', '').strip().split()
                continue 
            if line.startswith('#') or not line.strip():
                continue
            clean_lines.append(line)
    
    data_str = "".join(clean_lines)
    full_df = pd.read_csv(StringIO(data_str), sep=r'\s+', names=header)
    mask = (full_df['Zini'] >= z_min) & (full_df['Zini'] <= z_max) & \
           (full_df['logAge'] >= age_min) & (full_df['logAge'] <= age_max)
    filtered_df = full_df[mask]
    isochrone_groups = filtered_df.groupby(['Zini', 'logAge'])
    
    library = []
    for (z, age), group in isochrone_groups:
        library.append({'Z': z, 'logAge': age, 'df': group.reset_index(drop=True)})
    return library

# === 5. Load Library ===

isochrone_library = load_parsec_library(library_path, Z_MIN, Z_MAX, AGE_MIN, AGE_MAX)
print(f"✅ Loaded {len(isochrone_library)} isochrones.")

# === 6. Execution (Ultra-Fast Vectorized) ===
print(f"🚀 Starting High-Res Grid Search...")
obs_mags_val = obs_df[OBS_COLS].values

error_floor = 0.1
if use_errors:
    obs_errs_val = np.sqrt(obs_df[ERR_COLS].values**2 + error_floor**2)
else:
    # Gaia case: assume uniform uncertainty floor
    obs_errs_val = np.ones_like(obs_mags_val) * 0.01

# Pre-calculate extinction offsets for the entire E-grid at once
# Shape: (E_grid_size, 5 filters)
ext_offsets = e_grid[:, np.newaxis] * A_coeff 

results = []
for i, iso in enumerate(isochrone_library):
    Z, logAge, iso_mags = iso['Z'], iso['logAge'], iso['df'][ISO_COLS].values
    best_chi2 = np.inf
    print(f"📊 Progress: {(i+1)/len(isochrone_library)*100:5.1f}%", end="\r")

    for dm in dm_grid:
        # Calculate models for all E values simultaneously
        # Shape: (E_grid_size, stars, filters)
        models = iso_mags[np.newaxis, :, :] + dm + ext_offsets[:, np.newaxis, :]
        
        # Calculate Chi2 for all E values using broadcasting
        deltas = models[:, :, np.newaxis, :] - obs_mags_val[np.newaxis, np.newaxis, :, :]
        norm_sq = np.sum((deltas / obs_errs_val)**2, axis=3)
        chi2_per_e = np.sum(np.min(norm_sq, axis=1), axis=1)
        
        # Find best E for this DM
        min_idx = np.argmin(chi2_per_e)
        if chi2_per_e[min_idx] < best_chi2:
            best_chi2 = chi2_per_e[min_idx]
            best_dm, best_e = dm, e_grid[min_idx]

    results.append({"Z": Z, "logAge": logAge, "DM": best_dm, "E(B-V)": best_e, "chi2": best_chi2})
    
import pandas as pd
import numpy as np
import corner
import matplotlib.pyplot as plt

# === 7. Statistical Processing (Back to DM and E) ===
df_results = pd.DataFrame(results)
chi2_min = df_results["chi2"].min()

# 1. Calculate weights based on Chi2
# Divisor of 2.0 keeps the contours statistically significant
df_results["weight"] = np.exp(-(df_results["chi2"] - chi2_min) / 2.0)

# 2. RESAMPLE to create MCMC-like distribution for corner.py
num_samples = 20000
sample_indices = np.random.choice(
    df_results.index, 
    size=num_samples, 
    replace=True, 
    p=df_results["weight"] / df_results["weight"].sum()
)

# Column Order: logAge, Z, DM, E(B-V)
plot_samples = df_results.loc[sample_indices, ["logAge", "Z", "DM", "E(B-V)"]].values

# 3. JITTER: Spreads the points out so they aren't stuck exactly on grid lines
# This prevents the "singular transformation" error
jitter = np.random.normal(0, 1.0, size=plot_samples.shape)
jitter[:, 0] *= 0.01  # logAge jitter
jitter[:, 1] *= 0.001 # Z jitter
jitter[:, 2] *= 0.05  # DM jitter (since step 0.4 is large)
jitter[:, 3] *= 0.02  # E(B-V) jitter (since step 0.1 is large)
plot_samples += jitter

# === 8. Final Plotting (Using your Grid Boundaries) ===
labels = [r"$\log(t)$", r"$Z$", r"$DM$", r"$E(B-V)$"]

# MATCHING SECTION 3 BOUNDARIES EXACTLY
plot_ranges = [
    (AGE_MIN, AGE_MAX), 
    (Z_MIN, Z_MAX),     
    (DM_MIN, DM_MAX),   
    (E_MIN, E_MAX)      
]

fig = corner.corner(
    plot_samples,
    labels=labels,
    range=plot_ranges,
    show_titles=True,
    title_fmt=".3f",
    quantiles=[0.16, 0.5, 0.84],
    smooth=1.2,           
    smooth1d=0,           
    bins=35,              
    plot_datapoints=False,
    fill_contours=True,
    levels=(0.68, 0.95),
    color="black",
    hist_kwargs={"linewidth": 2} 
)

# Custom Cyan Overlay
axes = np.array(fig.axes).reshape((4, 4))
medians = np.percentile(plot_samples, 50, axis=0)
q16 = np.percentile(plot_samples, 16, axis=0)
q84 = np.percentile(plot_samples, 84, axis=0)

for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        if i == j: # Histograms
            ax.axvline(medians[i], color='deepskyblue', linestyle='-', linewidth=2)
            ax.axvline(q16[i], color='deepskyblue', linestyle='--', linewidth=1.5)
            ax.axvline(q84[i], color='deepskyblue', linestyle='--', linewidth=1.5)
        elif i > j: # 2D plots
            ax.axvline(medians[j], color='deepskyblue', linewidth=1)
            ax.axhline(medians[i], color='deepskyblue', linewidth=1)
            ax.plot(medians[j], medians[i], 's', color='deepskyblue', markersize=4)

plt.show()

# === 9. Quantitative Summary ===
print("\n" + "="*45)
print("📊 STATISTICAL RESULTS (16th, 50th, 84th Percentiles)")
print("="*45)

stat_labels = ["logAge", "Z", "DM", "E(B-V)"]
final_results = {}

for i, label in enumerate(stat_labels):
    q_16, q_50, q_84 = np.percentile(plot_samples[:, i], [16, 50, 84])
    upper_err, lower_err = q_84 - q_50, q_50 - q_16
    final_results[label] = q_50
    print(f"📏 {label:7}: {q_50:.3f} (+{upper_err:.3f} / -{lower_err:.3f})")

# Final Distance calc based on the Median DM
dist_pc = 10**((final_results['DM'] + 5) / 5)
print(f"\n📍 Median Distance: {dist_pc:.1f} pc")
print("="*45)


# === 10. Covariance Matrix ===
cov_matrix = np.cov(plot_samples.T)

print("\n" + "="*45)
print("📐 COVARIANCE MATRIX")
print("="*45)

param_names = ["logAge", "Z", "DM", "E(B-V)"]

cov_df = pd.DataFrame(cov_matrix, index=param_names, columns=param_names)
print(cov_df)

# --- Save Results ---
# These paths now all use {base_name} and {mode}
results_file = os.path.join(output_dir, f"{base_name}_{mode}_results.csv")
cov_file = os.path.join(output_dir, f"{base_name}_{mode}_covariance.csv")
plot_file = os.path.join(output_dir, f"{base_name}_{mode}_corner.png")

# Save Statistical Percentiles
df_final_stats = pd.DataFrame([final_results]) # Convert dict to DF for saving
df_final_stats.to_csv(results_file, index=False)

# Save Covariance Matrix
cov_df.to_csv(cov_file)

# Save Corner Plot
fig.savefig(plot_file, dpi=300)

print(f"\n💾 Results successfully saved to: {output_dir}")
print(f"📄 Data: {os.path.basename(results_file)}")
print(f"📄 Covariance: {os.path.basename(cov_file)}")
print(f"🖼️ Plot: {os.path.basename(plot_file)}")

