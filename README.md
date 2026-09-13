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