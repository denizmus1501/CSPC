"""
PW2 Lab B Part 2 -- three routes to a minimum.

Compare gradient descent, Newton, and SLSQP on two functions:
  2A: f(x) = (x-3)**2 + 1          (easy, one minimum at x=3)
  2B: g(x) = x**4 - 3*x**2 + x + 5 (harder, several stationary points)
Run:  python warmup.py
"""
import numpy as np
from scipy.optimize import newton, minimize

# ---------- 2A: easy convex function ----------
def f(x):   return (x-3)**2 + 1
def df(x):  return 2*(x-3)
def d2f(x): return 2.0

# TODO 2A: minimise f three ways from x0=0 and print each result:
#   (1) gradient descent by hand (loop x = x - lr*df(x) until the step is tiny)
#   (2) scipy.optimize.newton(df, x0, fprime=d2f)
#   (3) scipy.optimize.minimize(f, x0, method="SLSQP")

x0 = 0 # (1)
lr = 0.1
iterations = 100
for i in range(iterations):
    gradient = df(x0)
    next_x = x0 - (lr * gradient)

    if abs(next_x - x0) < 1e-8:
        break

    x0 = next_x

print(f"(1a)Gradient descent x = {x0}")

x0 = 0 # (2)
x = newton(df, x0, fprime = d2f)
print(f"(2a)Newton x = {x}")

x0 = 0 # (3)
x = minimize(f, x0, method = "SLSQP")
print(f"(3a)SLSQP x = {x.x}")

# ---------- 2B: harder landscape ----------
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6

# TODO 2B: run the same three methods on g, from x0=0 AND from x0=2.
#   For Newton (which solves dg(x)=0), also check the sign of d2g at the answer:
#   d2g > 0 means a minimum, d2g < 0 means a maximum.
#   In your README note: do the methods agree? did Newton find a minimum or
#   another stationary point? how did the starting point change the result?

print()
x0 = 0 # (1)
iterations = 100
for i in range(iterations):
    gradient = dg(x0)
    next_x = x0 - (lr * gradient)

    if abs(next_x - x0) < 1e-8:
        break

    x0 = next_x

print(f"(1b)Gradient descent x = {x0}")

x0 = 0 # (2)
x = newton(dg, x0, fprime = d2g)
print(f"(2b)Newton x = {x}")

x0 = 0 # (3)
x = minimize(g, x0, method = "SLSQP")
print(f"(3b)SLSQP x = {x.x}")

print()
#now using x0 = 2
x0 = 2 # (1)
iterations = 100
for i in range(iterations):
    gradient = dg(x0)
    next_x = x0 - (lr * gradient)

    if abs(next_x - x0) < 1e-8:
        break

    x0 = next_x

print(f"(1c)Gradient descent x = {x0}")

x0 = 2 # (2)
x = newton(dg, x0, fprime = d2g)
print(f"(2c)Newton x = {x}")
if d2g(x) > 0:
    print("minimum")
elif d2g(x) < 0:
    print("maximum")

x0 = 2 # (3)
x = minimize(g, x0, method = "SLSQP")
print(f"(3c)SLSQP x = {x.x}")