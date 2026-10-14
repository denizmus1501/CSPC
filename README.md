# CSPC - Computer Science for Physics and Chemistry
My coursework repository. Each practical is under PW<n>/Lab <X>/.
## Setup
Create the environment for a given lab:
conda env create -f PW<n>/Lab\ <X>/environment.yml
conda activate cspc
---
## PW1 - Lab A: Reproducible Foundations

**What I built:**
- added automated tests with pytest and compared the performance of the two implementations
**Speed comparison (loop vs NumPy):**
- loop : 5.162205 s
- numpy : 0.000188 s
- speed-up: 27464.53 x faster
**Tests:** all passing?
(yes / no) yes
**Conclusion:**
- the simulation works correctly and all tests pass successfully. the numpy implem-n is faster than the pure python loop, especially for a large number of atoms. i learned how to use pytest for automized testing and how numpy vectorization can improve the performance of numerical simulations.

---
## PW1 - Lab B: Data, Plotting, and Automation

**What I built:**
- read the observed decay data from a CSV file using NumPy
- plotted the observed data and the analytical decay law using Matplotlib
- created a Snakemake pipeline to automate the generation of the figure

**Data and analytical comparison:**
- the observed count decreases over time, following an exponential decay pattern
- the observed data matches the analytical decay law reasonably well

**Snakemake:**
- the pipeline takes `decay_observed.csv` as input and runs `plot.py` to generate `figure.png`
- Snakemake only reruns the plotting step when the input or required files have changed

**Conclusion:**
- the observed decay data follows the expected exponential decay behavior. i learned how to read numerical data with NumPy, create plots with Matplotlib, and use Snakemake to automate a data processing workflow.

---
## PW2 - Lab A: Motion from Tracking Data

**What I built:**
- read the free-fall position data from a CSV file using NumPy
- calculated velocity and acceleration using numerical differentiation with `np.gradient`
- integrated acceleration and velocity numerically to recover velocity and position
- plotted position, velocity, and acceleration using Matplotlib

**Results:**
- mean acceleration: -8.5797 m/s²
- acceleration standard deviation: 28.7161 m/s²
- maximum difference between the original and recovered position: 0.7846 m

**Noise observation:**
- differentiation amplifies measurement noise
- applying differentiation twice makes the acceleration much noisier than the original position data

**Integration:**
- numerical integration was used to recover velocity from acceleration and position from velocity (`cumulative_trapezoid`)
- the recovered position differed from the original position by at most 0.7846 m
- integration partly suppresses the noise introduced by differentiation, because it's a sum and some errors get cancelled

**Conclusion:**
- the motion data was successfully processed to obtain velocity and acceleration from position measurements. the acceleration is noisy because numerical differentiation amplifies measurement noise. integrating the data back showed that the original position can be recovered reasonably well, with a maximum difference of about 0.78 m. i learned how to use NumPy for numerical differentiation and integration and how noise affects numerical calculations.

---
## PW2 - Lab B: Optimization in Chemistry

**What I built:**

- compared three optimization methods: Gradient Descent, Newton's method, and SLSQP
- used optimization methods for reaction rate fitting and chemical equilibrium
- analyzed titration data to find the equivalence point

**Optimization methods:**

- tested the three methods first on a simple convex function
- all three methods found the same minimum at `x = 3`
- then tested the methods on a harder function with several stationary points
- Gradient Descent starting from `x = 0` converged to approximately `x = -1.30`
- Gradient Descent starting from `x = 2` also converged to approximately `x = -1.30`
- Newton's method starting from `x = 0` converged to approximately `x = 0.17`
- this point is a local maximum because `d2g < 0`
- Newton's method starting from `x = 2` converged to approximately `x = 1.13`
- this point is a local minimum because `d2g > 0`
- SLSQP starting from `x = 0` converged to approximately `x = -1.30`
- SLSQP starting from `x = 2` also converged to approximately `x = -1.30`
- the methods did not always agree because the harder function has several stationary points
- the starting point changed the result of Newton's method: starting from `x = 0` led to a local maximum, while starting from `x = 2` led to a local minimum

**Reaction rate fitting:**

- used the first-order reaction model `C(t) = C0 * exp(-k*t)` to fit noisy concentration data
- created a total squared error function to measure the difference between the measured and predicted concentrations
- used SLSQP to find the value of `k` that minimized the total error
- the fitted rate constant was approximately `k = 0.262`
- plotted the measured concentration data together with the fitted curve in `kinetics.png`

**Chemical equilibrium:**

- studied the reaction `H2 + I2 <=> 2HI`, starting with `1 mol` of `H2` and `1 mol` of `I2`
- represented the reaction using the extent `x`, where `H2 = 1-x`, `I2 = 1-x`, and `HI = 2x`
- solved the equilibrium condition in two ways: Newton root-finding and SLSQP by minimizing the squared imbalance
- both methods gave approximately `x = 0.66`
- this gives equilibrium amounts of approximately `0.34 mol H2`, `0.34 mol I2`, and `1.33 mol HI`
- plotted how the amounts change with reaction extent and marked the equilibrium point in `equilibrium.png`

**Titration:**

- analyzed titration data by plotting pH against the volume of titrant added
- calculated the slope of the titration curve to identify the equivalence point
- the equivalence point was found at `50 mL`
- at the equivalence point, the pH changes most rapidly and the slope reaches its maximum
- created two plots side by side: pH vs volume and slope vs volume
- marked the equivalence point on both plots
- saved the result as `titration.png`

**Conclusion:**

- this lab showed that different optimization methods can behave differently when a function has several stationary points
- I learned that Newton's method finds stationary points, so it can converge to either a minimum or a maximum depending on the starting point
- I also learned that the starting point can strongly affect the result of an optimization method
- SLSQP was useful for constrained optimization because bounds could be placed on the variables
- optimization methods can be applied to real chemistry problems such as fitting reaction rate constants, finding chemical equilibrium, and identifying the equivalence point of a titration

---
# PW3 - Data, Distributions, Testing, and Causation

**Data**

- Dataset 1: Heart Disease (Cleveland), Kaggle / UCI, 303 patients (164 healthy, 139 with heart disease)
- Dataset 2: `chemicals_cancer.csv`, synthetic, 1000 patients (benzene, cadmium, pollution index, age, malignancy)

**What I built**

- `heart_disease.py`: loaded the heart data with pandas, plotted and tested the distributions, compared the two groups and measured the age / heart rate relationship
- `chemicals_cancer.py`: measured the association of each chemical with malignancy and tested whether a third variable creates it
- bonus: entropy of the `target` column

**Distributions and normality (Session 1)**

- histograms of `age`, `chol`, `trestbps` and `thalach` are saved in `distributions.png`
- `age` and `thalach` are roughly bell-shaped, with a slight tail to the left (skewness -0.21 and -0.53)
- `trestbps` has a tail to the right (skewness 0.70)
- `chol` is clearly skewed to the right, with a few very high values up to 564 (skewness 1.13)
- normality was checked with Q-Q plots (`qq_plots.png`) and with the Shapiro-Wilk test (null hypothesis: the data is normal, so p < 0.05 means evidence against normality)
- Shapiro-Wilk p-values: `age` 0.006, `chol` ~0, `trestbps` 2e-6, `thalach` 7e-5
- verdict: none of the four variables is normal at the 5% level
- `age` and `thalach` are only mildly non-normal (the p-values are not extreme and the histograms are close to a bell), while `chol` and `trestbps` are clearly not normal
- because of these verdicts I chose a test that does not assume normality (Mann-Whitney U) in Session 2

**Heart rate comparison (Task 4)**

- split `thalach` into healthy (`target = 0`, n = 164) and heart disease (`target = 1`, n = 139)
- test: Mann-Whitney U, because `thalach` is not normal and the test compares the ranks of the values
- p-value = 1.9e-13, far below 0.05
- mean maximum heart rate with the standard error of the mean (SEM = std / sqrt(n)):
  - healthy: 158.4 ± 1.5 beats/min
  - heart disease: 139.3 ± 1.9 beats/min
- plot: bar chart of the two means with SEM error bars in `task4_heart_rate.png`
- the bars differ by about 19 beats/min, which is much larger than the error bars, so the picture agrees with the test
- conclusion: patients with heart disease reach a lower maximum heart rate than healthy patients, and the difference is statistically significant

**Age and heart rate (Task 5)**

- Pearson r = -0.39, which indicates a negative correlation
- scatter plot in `task5_correlation.png`
- older patients tend to have a lower maximum heart rate, but the relationship is only moderate because the points are quite spread out


**Chemical mystery**

**Naive analysis (Task 6)**

- benzene vs malignancy: r = 0.385
- cadmium vs malignancy: r = 0.880
- scatter plots in `task6_scatterplots.png`
- at face value cadmium looks like the cause, because its correlation with malignancy is much stronger than benzene's

**Looking again (Task 7)**

- suspected third variable: `pollution_index`
- the correlation matrix shows that the pollution index is almost identical to cadmium (r = 0.98) and strongly related to malignancy (r = 0.90), while it is almost unrelated to benzene (r = 0.05)
- so pollution could be influencing both cadmium and malignancy
- I kept only the patients with a similar pollution index (between 40 and 60, 199 patients) and recomputed the correlations:
  - benzene: r went from 0.385 to 0.702, the association remained and became stronger
  - cadmium: r went from 0.880 to 0.316, the association became much weaker
- possible explanation: in the full dataset, pollution and cadmium vary together, while pollution is also associated with malignancy. This may partly explain why cadmium initially appeared so strongly associated with malignancy
- when the pollution range is restricted, the cadmium association becomes weaker
- benzene is almost unrelated to pollution in the full dataset, and its association with malignancy remains strong in the selected group
- cadmium keeps a smaller positive correlation (0.32), because the pollution index still varies within the 40–60 range
- result: the analysis suggests that the original cadmium association may partly reflect confounding by pollution, while the benzene association remains strong in the selected group
- this analysis alone cannot prove that benzene is the real cause of malignancy; further analysis would be needed to establish causation

**Bonus (Entropy of `target`)**

- proportions: 54.1% healthy, 45.9% heart disease
- Shannon entropy H = -sum(p * log2(p)) = 0.995 bits, where the maximum for two categories is 1 bit
- the column is almost perfectly balanced, so guessing the class without additional information is similar to flipping a coin

**Conclusion**

- heart disease patients have a clearly lower maximum heart rate than healthy patients (about 19 beats/min), and maximum heart rate decreases with age
- in the chemical dataset, the initial analysis pointed to cadmium (r = 0.88 against 0.38 for benzene), but after restricting the pollution index to a similar range, the cadmium association became much weaker while the benzene association remained strong
- the results suggest that pollution may partly explain the original cadmium correlation, but they do not prove that benzene causes malignancy
- I learned how to check normality before choosing a statistical test, why a mean should be considered together with its uncertainty, and why correlation does not necessarily mean causation because a third variable can influence the observed relationship between two variables