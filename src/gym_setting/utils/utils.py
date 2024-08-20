#!/usr/bin/env python3
from sys import stdout
import functools
from numpy import pi, allclose, sin, cos, arctan2, sqrt, mean, argmin, argmax
from numpy import min as npmin, max as npmax
from numpy.core import overrides
from numpy.lib.npyio import _savez

__all__ = ['normalize_angle', 'RK2', 'RK4', 'Verbose', 'ArrEq', 'savez']



def normalize_angle(angle):

  return arctan2(sin(angle), cos(angle))


def RK2(function, duration, state, *args):

  k1 = duration * function(state, *args)
  k2 = duration * function(state + k1, *args)
  sol = state + (k1 + k2) / 2

  return sol


def RK4(function, duration, state, *args):

  k1 = duration * function(state, *args)
  k2 = duration * function(state + k1 / 2, *args)
  k3 = duration * function(state + k2 / 2, *args)
  k4 = duration * function(state + k3, *args)
  sol = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

  return sol


class Verbose:

  def __init__(self, *args):
    self.verbose = args[0]
    self.lenStr = 0

  def __call__(self, string):
    if self.verbose:
      stdout.write('\r'+' '*self.lenStr+'\r')
      stdout.flush()
      self.lenStr = 0 if string[-2:]=='\n' else len(string)
      stdout.write(string)
      stdout.flush()



class ArrEq:

  def __init__(self, obj):
    self.obj = obj

  def __eq__(self, other):
    if self.obj.shape == other.shape:
      return allclose(self.obj, other)
    else:
      return False


array_function_dispatch = functools.partial(
  overrides.array_function_dispatch,
  module='numpy'
)


def _savez_dispatcher(file, *args, **kwds):
  yield from args
  yield from kwds.values()


@array_function_dispatch(_savez_dispatcher)
def savez(file, *args, **kwds):
  _savez(file, args, kwds, False, allow_pickle=False)


def print_test(data, dim):
  err = []
  time = []
  success_count = 0.0
  if dim==2:
    for _, _, traj, success in data:
      err.append(traj[-1][0]**2)
      time.append(len(traj) * 0.05)
      if success is not None:
        success_count += float(success)
  elif dim==3:
    for _, _, _, traj, success in data:
      err.append(traj[-1][0]**2 + traj[-1][-1]**2)
      time.append(len(traj) * 0.05)
      if success is not None:
        success_count += float(success)
  print('  success rate:', success_count / float(len(data)))
  print('  avg err:', mean(sqrt(err)))
  print('  min err:', sqrt(npmin(err)))
  print('  min err state:', data[argmin(err)][:dim])
  print('  max err:', sqrt(npmax(err)))
  print('  max err state:', data[argmax(err)][:dim])
  print('  avg time:', mean(time))
  print('  min time:', npmin(time))
  print('  min time state:', data[argmin(time)][:dim])
  print('  max time:', npmax(time))
  print('  max time state:', data[argmax(time)][:dim])