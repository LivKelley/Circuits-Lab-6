#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

"""
For each of the four transistors on your ALD1106 chip, measure channel current as a function
of gate voltage with the source voltage at ground and the drain voltage at Vdd. Fit the EKV
model to each of these characteristics and extract a value of Is, κ, and VT0. 

In your report, include a table showing these extracted parameter values for all four transistors. 

Also, make a single semilog plot for your report showing all four current–voltage characteristics along with the fits. 

Also, make a semilog (i.e., make the x-axis log) plot showing the percentage difference between each transistor’s channel current and the mean value of all four channel currents as a function of the mean value of all four channel currents. 

How well do the
transistors match each other? Do you notice any systematic trends? For instance, do the
devices match better as a function of the level of inversion? Do certain devices match each
other better than others?
"""


def clip(xs, ys, xbounds, ybounds):
  pairs = [(x, y) for (x, y) in zip(xs, ys) if (xbounds[0] <= x) and (x <= xbounds[1]) and (ybounds[0] <= y) and (y <= ybounds[1])]
  out = list(zip(*pairs))
  return np.array(out[0]), np.array(out[1])

def clip_range(xs, ys, bounds):
  return clip(xs, ys, (-np.inf, np.inf), bounds)

def fit(xs, ys, model, initial_params):
  def err_f(params): return np.mean(np.power(np.log(ys) - np.log(model(xs, params)), 2))
  res = minimize(err_f, x0 = initial_params, method='Nelder-Mead')
  print(res)
  return res.x

# From ekvfit
def ekv(vg, params):
  Is, Vt, kappa = params
  return Is * np.power(np.log(1 + np.exp(kappa*(vg - Vt)/(2*0.0258))), 2)


vg = [[],[],[],[]]
isat = [[],[],[],[]]
params = [[],[],[],[]]
for i in range(4):
  with open("data/exp1_nmos_%s.csv" % [1,2,3,4][i]) as f:
    c = csv.reader(f, delimiter=",")
    next(c) # Throw away the header
    for row in c:
      vg[i] += [float(row[0])]
      isat[i] += [-float(row[1])] 

  # vg[i], isat[i] = clip_range(vg[i], isat[i], (1e-9, np.inf)) # Clip to reasonable
  vg[i] = vg[i][50:] # Do it uniformly so we can subtract things
  isat[i] = isat[i][50:]

  params[i] = fit(vg[i], isat[i], ekv, [7.2197482429849523e-08, 0.50917435504354447, 3.3782458038598859])
  print("Transistor %d: Is = %g, Vt = %g, κ = %g" % (i, params[i][0], params[i][1], params[i][2]))



imean = np.array([np.mean(i) for i in zip(*isat)])
idiff = [np.array(i) - imean for i in isat]


# Plot things
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111)

for (n, v, i, p) in zip(range(4), vg, isat, params):
  ax.semilogy(v, i, ['r.', 'y.', 'g.', 'b.'][n], markersize=1, label="Actual current (%d)" % (n+1))
  ax.semilogy(v, ekv(v, p), ['r-', 'y-', 'g-', 'b-'][n], linewidth=0.6, label="Theoretical fit (%d) (Is = %g A, Vt0 = %g V, κ = %g)" %  (n+1, p[0], p[1], p[2]))

plt.title("Saturation current-voltage characteristics")
plt.xlabel("Gate voltage (V)")
plt.ylabel("Current (A)")
plt.grid(True)
ax.legend()
plt.savefig("exp1-vi-semilog.pdf")
plt.cla()


for (n, i) in zip(range(4), idiff):
  ax.semilogx(imean, i * 100, ['r.', 'y.', 'g.', 'b.'][n], markersize=1, label="Percent difference in current (%d)" % (n+1))
plt.title("Current differences between transistors")
plt.xlabel("Mean current (A)")
plt.ylabel("Difference in current (%)")
plt.grid(True)
ax.legend()
plt.savefig("exp1-diffs.pdf")
plt.cla()
