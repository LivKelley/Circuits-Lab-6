#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

"""
For each of the four transistors on your ALD1106 chip, measure channel current as a function
of gate voltage with the source voltage at ground and the drain voltage at Vdd. Fit the EKV
model to each of these characteristics and extract a value of Is, κ, and VT0. In your report,
include a table showing these extracted parameter values for all four transistors. Also, make
a single semilog plot for your report showing all four current–voltage characteristics along
with the fits. Also, make a semilog (i.e., make the x-axis log) plot showing the percentage
difference between each transistor’s channel current and the mean value of all four channel
currents as a function of the mean value of all four channel currents. How well do the
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


for i in range(3):
  vg = []
  isat = []

  with open("data/exp1_nmos_%s.csv" % [1,2,3,4][i]) as f:
    c = csv.reader(f, delimiter=",")
    next(c) # Throw away the header
    for row in c:
      vg += [float(row[0])]
      isat += [float(row[1])] 

  vg, isat = clip_range(vg, isat, (np.finfo(float).eps, np.inf)) # Clip to positive

  params = fit(vg, isat, ekv, [7.2197482429849523e-08, 0.50917435504354447, 3.3782458038598859])


  # Calculate incremental transconductance gains
  gm_en = np.diff(vg) / np.diff(isat)
  gm_vn = np.arange(min(vg), max(vg), (max(vg) - min(vg))/len(vg))
  gm_tn_both = [[],[]]
  gm_tn_both[0] = [ekv(v, params) for v in gm_vn][:-1]
  gm_tn_both[1] = np.diff(gm_vn) / np.diff([ekv(v, params) for v in gm_vn])

  # Plot things
  fig = plt.figure(figsize=(8,6))
  ax = plt.subplot(111)

  ax.semilogy(vg, isat, 'b.', label="N-type current (experimental)")
  ax.semilogy(vg, ekv(vg, params), 'g-', label="N-type current (theoretical, Is = %g A, Vt0 = %g V, κ = %g)" %  (params[0], params[1], params[2]))

  plt.title("Saturation current-voltage characteristics")
  plt.xlabel("Gate voltage (V)")
  plt.ylabel("Current (A)")
  plt.grid(True)
  ax.legend()
  plt.savefig("exp1-vi-semilog.pdf")
  plt.cla()

  ax.loglog(isat[:-1], gm_en, 'b.', label="Inc. transconductance gain (experimental)")
  ax.loglog(gm_tn_both[0], gm_tn_both[1], 'g-', label="Inc. transconductance gain (theoretical)")

  plt.title("N-type incremental transconductance gain")
  plt.xlabel("Current (A)")
  plt.ylabel("Gm (℧)")
  plt.grid(True)
  ax.legend()
  plt.savefig("exp1-gm-n.pdf")
  plt.cla()
