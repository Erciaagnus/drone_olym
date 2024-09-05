#!/usr/bin/env python3
from multiprocessing import Manager, Pool, cpu_count
from time import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
import numpy as np
from scipy import sparse as sp
from utils import ArrEq, Verbose, savez
import traceback
__all__ = [
    "States",
    "Actions",
    "Surveillance_Actions",
    "Rewards",
    "Policy",
    "StateTransitionProbability",
    "MarkovDecisionProcess",
    "MarkovDecisionProcessTerminalCondition"
]

from multiprocessing import Array, Process, Value, cpu_count, Lock, Event, Semaphore
from time import time

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import bicgstab, lgmres, spsolve
from utils import Verbose


__all__ = ["ValueIteration", "PolicyIteration"]


class ValueIteration:
    def __init__(self, mdp, values=None):
        self.mdp = mdp
        if values is None:
            self.values = np.max(self.mdp.rewards.toarray(), axis=1) # initialize values by max_R(t=1)
        else:
            self.values = values

    def save(self, filename):
        np.savez(filename, values=self.values, policy=self.mdp.policy.toarray())

    def solve(
        self,
        max_iteration=1e3,
        tolerance=1e-8,
        earlystop=100,
        verbose=True,
        callback=None,
        parallel=True,
        save_name=None,
    ):
        self.verbose = Verbose(verbose)
        self.verbose("solving with Value Iteration...")
        best_iter = 0
        min_val_diff = np.inf
        start_time = time()
        last_time = time()
        current_time = time()
        result_filename = save_name + "_val_iter" #"UAV1Target1_value_iter_result"

        if parallel:
            self.shared = np.frombuffer(
                Array(
                    np.ctypeslib.as_ctypes_type(self.mdp.rewards.dtype),
                    int(self.mdp.states.num_states * self.mdp.actions.num_actions),
                ).get_obj(),
                dtype=self.mdp.rewards.dtype,
            )
            self.values = np.frombuffer(
                Array(
                    np.ctypeslib.as_ctypes_type(self.mdp.rewards.dtype), self.values
                ).get_obj(),
                dtype=self.mdp.rewards.dtype,
            )

            def _worker(P, key, flag):
                while True:
                    if flag.value > 0:
                        self.shared[key] = P.dot(self.values)
                        flag.value = 0
                    elif flag.value < 0:
                        break

            chunksize = np.ceil(
                self.mdp.states.num_states * self.mdp.actions.num_actions / cpu_count()
            )
            self.workers = []
            for pid in range(cpu_count()):
                key = slice(
                    int(chunksize * pid),
                    int(
                        min(
                            chunksize * (pid + 1),
                            self.mdp.states.num_states * self.mdp.actions.num_actions,
                        )
                    ),
                    None,
                )
                flag = Value("i", 0)
                self.workers.append(
                    (
                        Process(
                            target=_worker,
                            args=(
                                self.mdp.state_transition_probability[key],
                                key,
                                flag,
                            ),
                        ),
                        flag,
                    )
                )
                self.workers[-1][0].start()

        divergence_flag = 0
        for iter in range(int(max_iteration)):
            value_diff = self.update(parallel=parallel)

            if value_diff < tolerance:
                self.save(result_filename)
                self.verbose(f"possible best value, policy saved as {result_filename}...")
                break

            if min_val_diff - value_diff > tolerance*0.1:
                min_val_diff = value_diff
                best_iter = iter
                # save the best result
                if value_diff < 1e-6:
                    self.save(result_filename)

            delta = iter - best_iter
            if delta >= earlystop:
                self.save(result_filename)
                self.verbose(
                    f"Stopping training early as no improvement observed in last {earlystop} epochs. "
                    f"Best results observed at iter {best_iter+1}, best model saved as {result_filename}.npz.\n"
                )
                break

            current_time = time()
            self.verbose(
                "Iter.: %d, Value diff.: %.7f, Step time: %f (sec).\n"
                % (iter + 1, value_diff, current_time - last_time)
            )
            last_time = current_time

            if callback is not None:
                callback(self)

            if np.isnan(value_diff) or value_diff == np.inf:
                divergence_flag +=1
                if divergence_flag > 10:
                    for p, flag in self.workers:
                        flag.value = -1
                        p.join()
                    self.verbose("Divergence detected.")
                    break
                    # raise OverflowError("Divergence detected.")
            else:
                divergence_flag = 0
        if parallel:
            for p, flag in self.workers:
                flag.value = -1
                p.join()
        self.verbose("Time elapsed: %f (sec).\n" % (current_time - start_time))
        del self.verbose

    def update(self, parallel):

        self.verbose("Computing action values...")
        if parallel:
            for _, flag in self.workers:
                flag.value = 1
            for _, flag in self.workers:
                while True:
                    if flag.value == 0:
                        break
            q = self.mdp.rewards.toarray() + np.multiply(
                self.mdp.discount, self.shared.reshape(self.mdp.rewards.shape)
            )
        else:
            q = self.mdp.rewards.toarray() + np.multiply(
                self.mdp.discount,
                self.mdp.state_transition_probability.dot(self.values).reshape(
                    self.mdp.rewards.shape
                ),
            )

        # self.verbose("Updating policy...")
        policy = np.argmax(q, axis=1).astype(self.mdp.policy.dtype)
        new_values = np.take_along_axis(q, policy[:, np.newaxis], axis=1).ravel()
        self.mdp.policy.update(policy)

        value_diff = self.values[:] - new_values[:]
        value_diff = np.sqrt(
            np.dot(value_diff, value_diff) / self.mdp.states.num_states
        )

        self.values[:] = new_values.copy()

        return value_diff

class PolicyIteration:
    def __init__(self, mdp, values=None):
        self.result_filename = None
        self.mdp = mdp
        if values is None:
            self.values = np.max(self.mdp.rewards.toarray(), axis=1) # initialize values by max_R(t=1)
        else:
            self.values = values
        self.min_val_diff = np.inf
        self.not_improve = 0
        # Identity matrix $I_{|s|}$ and $I_{|a|}$ for computation
        self.terminal_state = False
        self.__I = sp.identity(
            self.mdp.states.num_states, dtype=np.float32, format="csr"
        )
        self.__innerloop_maxiter = int(self.mdp.states.num_states)

    def save(self, filename):
        np.savez(filename, values=self.values, policy=self.mdp.policy.toarray())

    def solve(
        self,
        max_iteration=1e3,
        tolerance=1e-8,
        earlystop=100,
        verbose=True,
        callback=None,
        parallel=True,
        save_name=None,
    ):

        self.verbose = Verbose(verbose)
        self.verbose("solving with Policy Iteration...")
        start_time = time()
        last_time = time()
        current_time = time()
        self.result_filename = save_name + "_pol_iter" #UAV1Target1_policy_iter_result"

        if parallel:
            self.shared = np.frombuffer(
                Array(
                    np.ctypeslib.as_ctypes_type(self.mdp.rewards.dtype),
                    int(self.mdp.states.num_states * self.mdp.actions.num_actions),
                ).get_obj(),
                dtype=self.mdp.rewards.dtype,
            )
            self.values = np.frombuffer(
                Array(
                    np.ctypeslib.as_ctypes_type(self.mdp.rewards.dtype), self.values
                ).get_obj(),
                dtype=self.mdp.rewards.dtype,
            )
            self.lock =Lock()
            self.sem = Semaphore(0)

            def _worker(P, key, event, exit_event):
                while True:
                    event.wait()  # Block until the main thread sets the event
                    if exit_event.is_set():  # If exit_event is set, break out of the loop
                        break
                    # Perform the update
                    with self.lock:
                        self.shared[key] = P.dot(self.values)
                    # Clear the event so the main thread knows this worker is done
                    event.clear()
                    # Signal the main process that this worker has finished
                    self.sem.release()

            chunksize = np.ceil(
                self.mdp.states.num_states * self.mdp.actions.num_actions / cpu_count()
            )
            self.workers = []
            for pid in range(cpu_count()):
                key = slice(
                    int(chunksize * pid),
                    int(
                        min(
                            chunksize * (pid + 1),
                            self.mdp.states.num_states * self.mdp.actions.num_actions,
                        )
                    ),
                    None,
                )
                event = Event()
                exit_event = Event()
                self.workers.append(
                    (
                        Process(
                            target=_worker,
                            args=(
                                self.mdp.state_transition_probability[key],
                                key,
                                event,
                                exit_event,
                            ),
                        ),
                        event,
                        exit_event,
                    )
                )
                self.workers[-1][0].start()

        for iter in range(int(max_iteration)):
            self.policy_eval(parallel=parallel, tolerance=tolerance, earlystop=earlystop)
            policy_stable = self.policy_improve(parallel=parallel)
            self.not_improve += 1
            if policy_stable:
                self.verbose(f"policy stable. policy saved as {self.result_filename}...\n")
                self.save(self.result_filename)
                break
            # if iter % 10 == 0:
            #     self.verbose(f"iteration: {iter}, policy not stable. policy saved as {self.result_filename}...")
            #     self.save(self.result_filename)
            if self.not_improve >= 10:
                self.verbose(f"policy not stable. policy saved as {self.result_filename} before {self.not_improve - 1} iterations...\n")
                break
            current_time = time()
            self.verbose(
                "Iter.: %d, Step time: %f (sec).\n"
                % (iter + 1, current_time - last_time)
            )
            last_time = current_time
        if parallel:
            # Recommendation 3: Use Semaphore for efficient waiting
            for p, event, exit_event in self.workers:
                exit_event.set()  # Signal the worker to exit
                event.set()  # Wake up the worker so it can check the exit_event
            self.sem.acquire(len(self.workers))  # Wait for all workers to finish
            for p, _, _ in self.workers:
                p.join()
        current_time = time()
        self.verbose("Time elapsed: %f (sec).\n" % (current_time - start_time))
        del self.verbose

    def policy_eval(self, parallel, tolerance, earlystop, direct_method=False, use_linalg=False):
        value_diff = np.inf
        min_val_diff = np.inf
        iter = 0
        best_iter = 0
        divergence_flag = 0
        while True:
            iter += 1
            if use_linalg:
                # Compute the value $V(s)$ via solving the linear system $(I-\gamma P^{\pi}), R^{\pi}$
                self.verbose("Constructing linear system...")
                A = self.__I - self.mdp.discount * sp.vstack(
                    [
                        self.mdp.state_transition_probability[s, a, :]
                        for s, a in enumerate(self.mdp.policy)
                    ],
                    format="csr",
                )
                if np.all(A.diagonal()):
                    b = self.mdp.rewards[self.mdp.policy.one_hot()]
                    if self.mdp.rewards.issparse:
                        b = b.T
                    if direct_method:
                        self.verbose("Solving linear system (SuperLU)...")
                        new_values = spsolve(A, b)
                    else:
                        self.verbose("Solving linear system (BiCGstab)...")
                        new_values, info = bicgstab(
                            A, b, x0=self.values, tol=1e-8, maxiter=self.__innerloop_maxiter
                        )
                        if info < 0:
                            self.verbose("BiCGstab failed. Call LGMRES...")
                            new_values, info = lgmres(
                                A,
                                b,
                                x0=new_values,
                                tol=1e-8,
                                maxiter=int(max(np.sqrt(self.__innerloop_maxiter), 10)),
                            )
                else:
                    self.verbose("det(A) is zero. Use policy evaluation update instead...")
                    if parallel:
                        # Signal workers to start
                        for _, event, _ in self.workers:
                            event.set()

                        # Wait for all workers to finish using the semaphore
                        self.sem.acquire(len(self.workers))

                        with self.lock:
                            q = self.mdp.rewards.toarray() + np.multiply(
                                self.mdp.discount, self.shared.reshape(self.mdp.rewards.shape)
                            )
                    else:
                        q = self.mdp.rewards.toarray() + np.multiply(
                            self.mdp.discount,
                            self.mdp.state_transition_probability.dot(self.values).reshape(
                                self.mdp.rewards.shape
                            ),
                        )

                    # new_values = np.sum(np.multiply(
                    #     self.mdp.policy.toarray(),  # This assumes the policy is a probability distribution over actions
                    #     q
                    # ), axis=1)
                    # ValueError: operands could not be broadcast together with shapes (172800,) (172800,2)
                    new_values = np.take_along_axis(q, self.mdp.policy[:, np.newaxis], axis=1).ravel()
            else:
                if parallel:
                    for _, event, _ in self.workers:
                        event.set()
                    # Wait for all workers to finish using the semaphore
                    self.sem.acquire(len(self.workers))

                    with self.lock:
                        q = self.mdp.rewards.toarray() + np.multiply(
                            self.mdp.discount, self.shared.reshape(self.mdp.rewards.shape)
                        )
                else:
                    q = self.mdp.rewards.toarray() + np.multiply(
                        self.mdp.discount,
                        self.mdp.state_transition_probability.dot(self.values).reshape(
                            self.mdp.rewards.shape
                        ),
                    )
                new_values = np.take_along_axis(q, self.mdp.policy[:, np.newaxis], axis=1).ravel()
            value_diff = self.values[:] - new_values[:]
            value_diff = np.sqrt(
                np.dot(value_diff, value_diff) / self.mdp.states.num_states
            )
            self.values = new_values.copy()
            self.verbose(
                "Policy evaluation: Iter.: %d, Value diff.: %f.\n"
                % (iter + 1, value_diff)
            )
            # divergence condition
            if np.isnan(value_diff) or value_diff == np.inf:
                divergence_flag +=1
                if divergence_flag > 10:
                    self.verbose("Divergence detected.")
                    break
                    # raise OverflowError("Divergence detected.")
            else:
                divergence_flag = 0
            # convergence condition
            if value_diff < tolerance:
                break
            # early stopping condition
            if min_val_diff - value_diff > tolerance*0.1:
                min_val_diff = value_diff
                if min_val_diff < self.min_val_diff:
                    self.min_val_diff = min_val_diff
                    self.not_improve = 0
                best_iter = iter
                best_values = self.values.copy()
            delta = iter - best_iter
            if delta >= earlystop:
                self.verbose(
                    f"Stopping training early as no improvement observed in last {earlystop} epochs. "
                    f"Best results observed at iter {best_iter+1}\n"
                )
                self.values = best_values
                break

    def policy_improve(self, parallel):
        policy_stable = False
        current_policy = self.mdp.policy.toarray()
        self.verbose("Policy improvement...")
        if parallel:
            # Signal workers to start
            for _, event, _ in self.workers:
                event.set()

            # Wait for all workers to finish using the semaphore
            self.sem.acquire(len(self.workers))

            with self.lock:
                q = self.mdp.rewards.toarray() + np.multiply(
                    self.mdp.discount, self.shared.reshape(self.mdp.rewards.shape)
                )
        else:
            q = self.mdp.rewards.toarray() + np.multiply(
                self.mdp.discount,
                self.mdp.state_transition_probability.dot(self.values).reshape(
                    self.mdp.rewards.shape
                ),
            )

        self.mdp.policy.update(
            np.argmax(q, axis=1).astype(self.mdp.policy.dtype)
        )
        if not self.not_improve:
            self.save(self.result_filename)
        new_policy = self.mdp.policy.toarray()
        policy_stable = np.array_equal(current_policy, new_policy)
        return policy_stable
class States: # Distance, Heading Angle, Age, Battery
    def __init__(
        self, *state_lists, cycles=None, terminal_states=None, n_alpha=None, dtype=np.float32
    ):
        self.__data = None
        self.__cycles =None
        self.__n_alpha = None
        self.update(
            *state_lists, cycles=cycles, terminal_states=terminal_states, n_alpha=n_alpha, dtype=dtype
        )

    def update(self, *state_lists, cycles=None, terminal_states=None, n_alpha=None, dtype=np.float32):
        self.__data = [np.array(state_list, dtype=dtype) for state_list in state_lists] # in state lists,,, -> Formation?? [a, b, c], [a, b, c] Form??
        if cycles is None:
            self.__cycles = [np.inf]*len(state_lists)
        elif len(cycles) == len(state_lists):
            self.__cycles = np.array(cycles, dtype=dtype)
        else:
            raise ValueError(
                "opernads could not be broadcast together with shapes ({},) ({},)".format(
                    len(state_lists), len(cycles)
                )
            )
        if terminal_states is None:
            self.__terminal_states = []
        else:
            self.__terminal_states = list(
                [np.array(state, dtype=self.dtype) for state in terminal_states]
            )
        if n_alpha is None:
            self.__n_alpha = None
        else:
            self.__n_alpha = n_alpha

    def __getitem__(self, key):

        if len(self.__data)==1:
            return np.array(self.__data[0][key])
        if isinstance(key, int):
            indices = np.unravel_index(key, shape=self.shape)
            return np.array(
                [state_list[idx] for (state_list, idx) in zip(self.__data, indices)], dtype=self.dtype,
            )
        elif isinstance(key, tuple):
            if np.any([isinstance(slice, k) for k in key]):
                raise KeyError("class States does not support slice method.")
            if len(key) == 1:
                indices = np.unravel_index(key[0], shape=self.shape)
                return np.array(
                    [
                        state_list[idx]
                        for (state_list, idx) in zip(self.__data, indices)
                    ],
                    dtype=self.dtype,
                )
            elif len(key) == len(self.shape):
                return np.array(
                    [state_list[k] for (state_list, k) in zip(self.__data, key)],
                    dtype=self.dtype,
                )
            else:
                raise KeyError("Number of indices mismatch.")
        else:
            raise KeyError("Unsupported key type.")
    def __iter__(self):
        class SubIterator:
            def __init__(cls, data, shape, num_states, dtype):
                cls.dtype = dtype
                cls.__data = data
                cls.__shape = shape
                cls.__num_states = num_states
                cls.__dim = len(cls.__shape)
                cls.__counters = [0]*cls.__dim
                cls.__total_count = 0

            def __next__(cls):
                cls.__total_count += 1
                if cls.__total_count > cls.__num_states:
                    raise StopIteration()
                else:
                    ret=[
                        state_list[count]
                        for count, state_list in zip(cls.__counters, cls.__data)
                    ]
                for d in range(cls.__dim)[::-1]:
                    if cls.__counters[d] < (cls.__shape[d]-1):
                        cls.__counters[d] += 1
                        break
                    else:
                        cls.__counters[d] = 0
                return np.array(ret, dtype=cls.dtype)
        return SubIterator(self.__data, self.shape, self.num_states, self.dtype)
    @property
    def shape(self):
        return tuple([len(state_list) for state_list in self.__data])

    @property
    def dtype(self):
        return self.__data[0].dtype

    @property
    def num_states(self):
        return np.prod(self.shape)

    def computeBarycentric(self, item):
        # concatenate dict into numpy array
        if isinstance(item, dict):
            # Select specific values from the dictionary
            selected_values = [item['uav1_target1'][0], item['uav1_target1'][1],
                            item['uav1_charge_station'][0], item['uav1_charge_station'][1],
                            item['battery'], item['age']]

            # Convert the selected values to numpy array
            item = np.array(selected_values)
            # print('concatenated item: ',item)
        i = []
        p = []
        for (state_list, cycle, x) in zip(self.__data, self.__cycles, item):
            in_range = False
            # loop = 0
            while not in_range:
                # loop += 1
                # if loop > 10:
                #     print('looping over 10 times..')
                idx = np.searchsorted(state_list, x)
                if idx < len(state_list): # x < max(state_list)
                    if x == state_list[idx]:
                    # if x - state_list[idx] < 1e-7: # -pi -> pi -> -pi...
                        i.append(np.array([idx], dtype=int))
                        p.append(np.ones((1,), dtype=self.dtype))
                        in_range = True
                    else:
                        d1 = state_list[idx] - x
                        if idx == 0: # x < min(state_list)
                            if cycle == np.inf: # was is
                                i.append(np.array([idx], dtype=int))
                                p.append(np.ones((1,), dtype=self.dtype))
                                in_range = True
                            else:
                                if d1 < np.pi/self.__n_alpha: # for accuray(avoid extrapolating when interpolating is available for 2 nearer points) # 10 for 1uav1target
                                    d2 = x + cycle - state_list[-1] #seems to need fix
                                    i.append(np.array([idx -1, 0]))
                                    p.append(np.array([d1, d2]) / (d1 + d2))
                                    in_range = True
                                else:
                                    x += cycle
                        else: # min(sate_list) < x < max(state_list)
                            d2 = x - state_list[idx - 1]
                            i.append(np.array([idx - 1, idx], dtype=int))
                            p.append(np.array([d1, d2]) / (d1 + d2))
                            in_range = True
                else: # x > max(state_list)
                    if cycle == np.inf:
                        i.append(np.array([idx - 1], dtype=int))
                        p.append(np.ones((1,), dtype=self.dtype))
                        in_range = True
                    else: # need fix
                        d2 = x - state_list[-1]
                        if d2 < np.pi/self.__n_alpha: # for accuray(avoid extrapolating when interpolating is available for 2 nearer points) # 10 for 1uav1target
                            d1 = state_list[0] + cycle - x
                            i.append(np.array([idx - 1, 0]))
                            p.append(np.array([d1, d2]) / (d1 + d2))
                            in_range = True
                        else:
                            x -= cycle
        indices = np.zeros((1,), dtype=int)
        probs = np.ones((1,), dtype=self.dtype)
        for dim, (idx, prob) in enumerate(zip(i, p)):
            indices = np.repeat(indices, len(idx)) + np.tile(idx, len(indices))
            if dim < len(self.shape) - 1: # multi-dimension indices into 1d array index
                indices *= self.shape[dim + 1]
            probs = np.repeat(probs, len(idx)) * np.tile(prob, len(probs))
        return indices, probs

        # for dim, (idx, prob) in enumerate(zip(i, p)):
        #     # Indices
        #     #print("indices:", indices, "shape:", indices.shape)
        #     #print("idx:", idx, "shape:", idx.shape)
        #     #print("prob:", prob, "shape:", prob.shape)
        #     indices = indices.flatten()
        #     idx = idx.flatten()
        #     prob = prob.flatten()
        #     # Repeat and tile with compatible shape
        #     repeated_indices = np.repeat(indices, len(idx))
        #     tiled_idx = np.tile(idx, len(indices))
        #     #print("repeated_indices:", repeated_indices, "shape:", repeated_indices.shape)
        #     #print("tiled_idx:", tiled_idx, "shape", tiled_idx.shape)

        #     if repeated_indices.shape == tiled_idx.shape:
        #         indices = repeated_indices + tiled_idx
        #     else:
        #         raise ValueError(f"Shapes are not compatible for broadcasting: {repeated_indices.shape}, {tiled_idx.shape}")
        #     #indices = np.repeat(indices, len(idx)) + np.tile(idx, len(indices))
        #     # probability
        #     if dim < len(self.shape) - 1:
        #         indices *= self.shape[dim + 1]
        #     repeated_probs = np.repeat(probs, len(idx))
        #     tiled_prob = np.tile(prob, len(probs))
        #     if repeated_probs.shape == tiled_prob.shape:
        #         probs = repeated_probs * tiled_prob
        #     else:
        #         raise ValueError(f"Shapes are not compatible for broadcasting:  {repeated_probs.shape}, {tiled_prob.shape}")
        #     #probs = np.repeat(probs, len(idx)) * np.tile(prob, len(probs))
        # return indices, probs

    def index(self, state):

        if len(state) == len(self.__data):
            n = 0
            for idx, x in enumerate(state):
                n += np.argmin(np.abs(self.__data[idx] - x)) * np.prod(
                    self.shape[idx + 1 :], dtype=int
                )
            return n
        else:
            raise ValueError(
                "operands could not be broadcast together with shapes ({},) ({},)".format(
                    len(self.__data), len(state)
                )
            )

    def info(self, return_data=False, return_cycles=False):

        if return_data:
            if return_cycles:
                return self.__data, self.__cycles
            else:
                return self.__data
        else:
            if return_cycles:
                return self.__cycles
            else:
                return None

    # End of class States

class Actions:
    def __init__(self, action_list, dtype=np.float32):

        self.__data = None
        self.__data_ndarr = None
        self.update(action_list, dtype)

    def index(self, item):

        if isinstance(item, np.ndarray):
            return self.__data_ndarr.index(item)
        else:
            return self.__data.index(item)

    def __getitem__(self, key):

        return self.__data[key]

    def __iter__(self):
        return self.__data.__iter__()

    @property
    def dtype(self):
        return self.__data[0].dtype

    @property
    def shape(self):
        return (len(self.__data),) + self.__data[0].shape

    @property
    def num_actions(self):
        return len(self.__data)

    def update(self, action_list, dtype=np.float32):
        self.__data = list()
        self.__data_ndarr = list()
        for item in action_list:
            self.__data.append(np.array(item, dtype=dtype))
            self.__data_ndarr.append(ArrEq(self.__data[-1]))

    def tolist(self):
        return self.__data

    def toarray(self):
        return np.array(self.__data)

    # End of class Actions

class Surveillance_Actions:
    def __init__(self, action_lists, dtype=np.float32):

        self.__data = None
        self.__data_ndarr = None
        self.update(action_lists, dtype)

    def index(self, item):

        if isinstance(item, np.ndarray):
            return self.__data_ndarr.index(item)
        else:
            return self.__data.index(item)

    def __getitem__(self, key):

        return self.__data[key]

    def __iter__(self):
        return self.__data.__iter__()

    @property
    def dtype(self):
        return self.__data[0].dtype

    @property
    def shape(self):
        return (len(self.__data),) + self.__data[0].shape

    @property
    def num_actions(self):
        return len(self.__data)

    def update(self, action_lists, dtype=np.float32):
        self.__data = list()
        self.__data_ndarr = list()
        for item in action_lists:
            self.__data.append(np.array(item))
            self.__data_ndarr.append(ArrEq(self.__data[-1]))

    def tolist(self):
        return self.__data

    def toarray(self):
        return np.array(self.__data)

    # End of class Actions

class Rewards:
    def __init__(self, states, actions, dtype=np.float32, sparse=False):
        shape = (states.num_states, actions.num_actions)
        if sparse:
            self.__data = sp.dok_matrix(shape, dtype=dtype)
            print('sp matrix of Rewards made...')
        else:
            self.__data = np.zeros(shape, dtype=dtype)

    def __setitem__(self, key, val):

        self.__data[key] = val

    def __getitem__(self, key):

        return self.__data[key]

    def __iter__(self):

        return self.__data.__iter__()

    def __eq__(self, other):
        # Check if both are instances of Rewards class
        if not isinstance(other, Rewards):
            return False

        # Check if both are sparse or both are dense
        if self.issparse != other.issparse:
            return False

        # If both are sparse, convert to CSR format and compare
        if self.issparse:
            return (self.tocsr() != other.tocsr()).nnz == 0

        # If both are dense, simply compare the numpy arrays
        return np.array_equal(self.toarray(), other.toarray())

    @property
    def dtype(self):
        return self.__data.dtype

    @property
    def shape(self):
        return self.__data.shape

    @property
    def issparse(self):
        return isinstance(self.__data, sp.spmatrix)

    def tocsr(self):

        if self.issparse:
            if not isinstance(self.__data, sp.csr_matrix):
                self.__data = self.__data.tocsr()
        else:
            self.__data = sp.csr_matrix(self.__data, dtype=self.dtype)
        return self.__data

    def todok(self):

        if self.issparse:
            if not isinstance(self.__data, sp.dok_matrix):
                self.__data = self.__data.todok()
        else:
            self.__data = sp.dok_matrix(self.__data, dtype=self.dtype)
        return self.__data

    def toarray(self, copy=False):

        if self.issparse:
            return self.__data.toarray()
        else:
            if copy:
                return self.__data.copy()
            else:
                return self.__data

    def update(self, data):
        self.__data = data

    def load(self, filename):
        filetype = filename.split(".")[-1]
        if filetype == "npz":
            self.__data = sp.load_npz(filename)
        elif filetype == "npy":
            self.__data = np.load(filename)

    def save(self, filename):
        if self.issparse:
            self.__data = sp.save_npz(filename, self.__data)
        else:
            self.__data = np.save(filename, self.__data)

    # End of class Rewards

class StateTransitionProbability:
    def __init__(self, states, actions, dtype=np.float32):
        print(
            "states.num_states * actions.num_actions: ",
            states.num_states * actions.num_actions,
        )
        self.__data = sp.dok_matrix(
            (states.num_states * actions.num_actions, states.num_states), dtype=dtype
        )
        print('sp matrix of P made..')

    def __setitem__(self, key, val):

        if isinstance(key, tuple):
            if len(key) == 1:
                return self.__data[key[0]]
            elif len(key) == 2:
                self.__data[key] = val
            elif len(key) == 3:
                if isinstance(key[0], slice):
                    raise IndexError("First index does not supports slice method.")
                if isinstance(key[1], slice):
                    start = key[1].start
                    stop = key[1].stop
                    step = key[1].step
                    shape = self.shape
                    self.__data[
                        slice(
                            np.ravel_multi_index((key[0], start), shape[:2]),
                            np.ravel_multi_index((key[0], stop), shape[:2]),
                            step,
                        ),
                        key[2],
                    ] = val
                else:
                    self.__data[
                        np.ravel_multi_index(key[:2], self.shape[:2]), key[2]
                    ] = val
            else:
                raise IndexError("Indices mismatch.")
        elif isinstance(key, int) or isinstance(key, slice):
            self.__data[key] = val
        else:
            raise IndexError("Indices mismatch.")

    def __getitem__(self, key):

        if isinstance(key, tuple):
            if len(key) == 1:
                return self.__data[key[0]]
            elif len(key) == 2:
                return self.__data[key]
            elif len(key) == 3:
                if isinstance(key[0], slice):
                    raise IndexError("First index does not supports slice method.")
                if isinstance(key[1], slice):
                    start = key[1].start
                    stop = key[1].stop
                    step = key[1].step
                    shape = self.shape
                    return self.__data[
                        slice(
                            np.ravel_multi_index((key[0], start), shape[:2]),
                            np.ravel_multi_index((key[0], stop), shape[:2]),
                            step,
                        ),
                        key[2],
                    ]
                else:
                    return self.__data[
                        np.ravel_multi_index(key[:2], self.shape[:2]), key[2]
                    ]
        elif isinstance(key, int) or isinstance(key, slice):
            return self.__data[key]
        else:
            raise IndexError("Indices mismatch.")

    def __iter__(self):
        return self.__data.__iter__()

    def __eq__(self, other):
        # Check if the other object is an instance of StateTransitionProbability
        if not isinstance(other, StateTransitionProbability):
            return NotImplemented

        # Compare the __data attributes of both instances
        # The expression (self.__data != other.__data) returns a matrix of boolean values indicating element-wise inequality.
        # The .nnz property returns the number of non-zero entries in the matrix.
        # If there are no non-zero entries, it means all elements are equal, and hence the matrices are equal.
        return (self.__data != other.__data).nnz == 0

    @property
    def dtype(self):
        return self.__data.dtype

    @property
    def shape(self):
        sa, s = self.__data.shape
        return (s, sa // s, s)

    def dot(self, other):
        return self.__data.dot(other)

    def tocsr(self):
        if not sp.isspmatrix_csr(self.__data):
            self.__data = self.__data.tocsr()

    def todok(self):
        if not sp.isspmatrix_dok(self.__data):
            self.__data = self.__data.todok()

    def tospmat(self):
        return self.__data

    def toarray(self):
        return self.__data.toarray()

    def update(self, data):
        self.__data = data

    def load(self, filename):
        self.__data = np.load(filename, allow_pickle=True)

    def save(self, filename):
        np.save(filename, self.__data, allow_pickle=True)

    # End of class StateTransitionProbability

class Policy:
    def __init__(self, states, actions, dtype=int):
        self.__states = states
        self.__actions = actions
        self.__data = None
        self.__I = np.eye(self.__actions.num_actions, dtype=bool)
        self.reset(dtype=dtype)

    def __setitem__(self, key, val):
        self.__data[key] = min(val, self.__actions.num_actions - 1)

    def __getitem__(self, key):
        return self.__data[key]

    def __iter__(self):
        return self.__data.__iter__()

    def __str__(self):
        return self.__data.__str__()

    @property
    def dtype(self):
        return self.__data.dtype

    @property
    def shape(self):
        return self.__data.shape

    def one_hot(self):
        return self.__I[self.__data]

    def get_action(self, state):
        S, P = self.__states.computeBarycentric(state)
        action = 0
        for s, p in zip(S, P):
            action += p * self.__actions[int(self.__data[s])]
        return action

    def update(self, data):
        self.__data = data

    def reset(self, dtype=int):
        self.__data = np.random.randint(
            0, self.__actions.num_actions, self.__states.num_states, dtype=dtype
        )

    def toarray(self, copy=False):
        if copy:
            return self.__data.copy()
        else:
            return self.__data

    def load(self, filename):
        self.__data = np.load(filename)

    def save(self, filename):
        np.save(filename, self.__data)

    # End of class Policy

# MDP without terminal states
class MarkovDecisionProcess:
    def __init__(
        self,
        states=None,
        actions=None,
        rewards=None,
        state_transition_probability=None,
        policy=None,
        discount=0,
    ):

        self.states = States([]) if states is None else states
        self.actions = Actions([]) if actions is None else actions
        self.rewards = (
            Rewards(self.states, self.actions) if rewards is None else rewards
        )
        self.discount = min(
            np.array(discount, dtype=self.rewards.dtype).item(),
            np.array(1, dtype=self.rewards.dtype).item()
            - np.finfo(self.rewards.dtype).eps,
        )
        self.state_transition_probability = (
            StateTransitionProbability(self.states, self.actions)
            if state_transition_probability is None
            else state_transition_probability
        )
        self.policy = Policy(self.states, self.actions) if policy is None else policy
        self.__sampler = None
        self.__sample_reward = False

    def _worker(self, queue, state):
        if self.__sample_reward:
            spmat, arr = self.__sampler(state)
        else:
            spmat = self.__sampler(state)
        if queue is None:
            pass
        else:
            queue.put(1)
        if self.__sample_reward:
            return np.array([spmat.tocsr(), arr], dtype=object)
        else:
            return spmat.tocsr()

    def sample(self, sampler, parallel, sample_reward=False, verbose=True):
        verbose = Verbose(verbose)
        verbose("Start sampling...")
        start_time = time()
        self.__sampler = sampler
        self.__sample_reward = sample_reward
        if parallel:
            queue = Manager().Queue()
            with Pool(cpu_count()) as p:
                data = p.starmap_async(
                    self._worker, [(queue, state) for state in self.states]
                )
                counter = 0
                tic = time()
                while counter < self.states.num_states:
                    counter += queue.get()
                    if time() - tic > 0.1:
                        progress = counter / self.states.num_states
                        rt = (time() - start_time) * (1 - progress) / progress
                        rh = rt // 3600
                        rt %= 3600
                        rm = rt // 60
                        rs = rt % 60
                        progress *= 100
                        verbose(
                            "Sampling progress: %5.1f %%... (%dh %dm %ds rem.)"
                            % (progress, rh, rm, rs)
                        )
                        tic = time()
                if self.__sample_reward:
                    data = np.array(data.get(), dtype=object)
                    self.state_transition_probability.update(sp.vstack(data[:, 0]))
                    self.rewards.update(
                        np.array(data[:, 1].tolist(), dtype=self.rewards.dtype)
                    )
                else:
                    self.state_transition_probability.update(sp.vstack(data.get()))
        else:
            # Modified code to run in a single process for debugging
            data = []
            for state in self.states:
                try:
                    result = self._worker(None, state)  # Replace None with an actual queue if necessary
                    data.append(result)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    traceback.print_exc()
                    sys.exit(1)

            if self.__sample_reward:
                # data = np.array(data.get(), dtype=object)
                data = np.array(data, dtype=object)
                self.state_transition_probability.update(sp.vstack(data[:, 0]))
                self.rewards.update(
                    np.array(data[:, 1].tolist(), dtype=self.rewards.dtype)
                )
            else:
                self.state_transition_probability.update(sp.vstack(data.get()))
        self.__sampler = None
        end_time = time()
        verbose("Sampling is done. %f (sec) elapsed.\n" % (end_time - start_time))

    def load(self, filename):

        data = np.load(filename, allow_pickle=False)
        state_lists = []
        for idx in range(data["states.num_lists"].item()):
            state_lists.append(data["states.data." + str(idx)])
        self.states = States(
            *state_lists,
            cycles=data["states.cycles"],
            terminal_states=data["states.terminal_states"]
        )

        self.actions = Actions(data["actions.data"])

        self.rewards = Rewards(
            self.states, self.actions, sparse=data["rewards.issparse"].item()
        )
        if self.rewards.issparse:
            self.rewards.update(
                sp.csr_matrix(
                    (
                        data["rewards.data"],
                        data["rewards.indices"],
                        data["rewards.indptr"],
                    ),
                    shape=(self.states.num_states, self.actions.num_actions),
                )
            )
        else:
            self.rewards.update(data["rewards.data"])

        self.state_transition_probability = StateTransitionProbability(
            self.states, self.actions
        )
        self.state_transition_probability.update(
            sp.csr_matrix(
                (
                    data["state_transition_probability.data"],
                    data["state_transition_probability.indices"],
                    data["state_transition_probability.indptr"],
                ),
                shape=(
                    self.states.num_states * self.actions.num_actions,
                    self.states.num_states,
                ),
            )
        )
        self.policy = Policy(self.states, self.actions)
        self.policy.update(data["policy.data"])
        self.discount = data["discount"].item()

    def save(self, filename):

        if not sp.isspmatrix_csr(self.state_transition_probability.tospmat()):
            self.state_transition_probability.tocsr()

        kwargs = {
            "states.num_lists": len(self.states.shape),
            "states.cycles": self.states.info(return_cycles=True),
            "actions.data": self.actions.toarray(),
            "rewards.issparse": self.rewards.issparse,
            "state_transition_probability.data": self.state_transition_probability.tospmat().data,
            "state_transition_probability.indices": self.state_transition_probability.tospmat().indices,
            "state_transition_probability.indptr": self.state_transition_probability.tospmat().indptr,
            "policy.data": self.policy.toarray(),
            "discount": self.discount,
        }
        for idx, state_list in enumerate(self.states.info(return_data=True)):
            kwargs["states.data." + str(idx)] = state_list
        if self.rewards.issparse:
            kwargs["rewards.data"] = (self.rewards.tocsr().data,)
            kwargs["rewards.indices"] = (self.rewards.tocsr().indices,)
            kwargs["rewards.indptr"] = (self.rewards.tocsr().indptr,)
        else:
            kwargs["rewards.data"] = self.rewards.toarray()

        savez(filename, **kwargs)

    # End of class MarkovDecisionProcess

# MDP with terminal states
class MarkovDecisionProcessTerminalCondition:
    def __init__(
        self,
        states=None,
        actions=None,
        rewards=None,
        state_transition_probability=None,
        policy=None,
        discount=0,
    ):

        self.states = States([]) if states is None else states
        self.actions = Actions([]) if actions is None else actions
        self.rewards = (
            Rewards(self.states, self.actions) if rewards is None else rewards
        )
        print('discount: ', discount)
        self.discount = min(
            np.array(discount, dtype=self.rewards.dtype).item(),
            np.array(1, dtype=self.rewards.dtype).item()
            - np.finfo(self.rewards.dtype).eps,
        )
        self.state_transition_probability = (
            StateTransitionProbability(self.states, self.actions)
            if state_transition_probability is None
            else state_transition_probability
        )
        self.policy = Policy(self.states, self.actions) if policy is None else policy
        self.__sampler = None
        self.__sample_reward = False

    def _worker(self, queue, state):
        if len(state) == 2: # toc/dkc
            r, _ = state
            if r > 0.001: # if not terminal state: 0.1 if real else 1
                if self.__sample_reward:
                    spmat, arr = self.__sampler(state)
                else:
                    spmat = self.__sampler(state)
            else: # if state is in terminal states
                spmat = sp.dok_matrix((self.actions.num_actions, self.states.num_states), dtype=np.float32)
                state_indice, _ = self.states.computeBarycentric(state)
                spmat[:, state_indice] += 1
        else: # 1u1t
            if self.__sample_reward:
                spmat, arr = self.__sampler(state)
            else:
                spmat = self.__sampler(state)
        if queue is None:
            pass
        else:
            queue.put(1)
        if self.__sample_reward:
            return np.array([spmat.tocsr(), arr], dtype=object)
        else:
            return spmat.tocsr()

    def sample(self, sampler, parallel, sample_reward=False, verbose=True):
        verbose = Verbose(verbose)
        verbose("Start sampling...")
        start_time = time()
        self.__sampler = sampler
        self.__sample_reward = sample_reward
        if parallel:
            queue = Manager().Queue()
            with Pool(cpu_count()) as p:
                data = p.starmap_async(
                    self._worker, [(queue, state) for state in self.states]
                )
                counter = 0
                tic = time()
                while counter < self.states.num_states:
                    counter += queue.get()
                    if time() - tic > 0.1:
                        progress = counter / self.states.num_states
                        rt = (time() - start_time) * (1 - progress) / progress
                        rh = rt // 3600
                        rt %= 3600
                        rm = rt // 60
                        rs = rt % 60
                        progress *= 100
                        verbose(
                            "Sampling progress: %5.1f %%... (%dh %dm %ds rem.)"
                            % (progress, rh, rm, rs)
                        )
                        tic = time()
                if self.__sample_reward:
                    data = np.array(data.get(), dtype=object)
                    self.state_transition_probability.update(sp.vstack(data[:, 0]))
                    self.rewards.update(
                        np.array(data[:, 1].tolist(), dtype=self.rewards.dtype)
                    )
                else:
                    self.state_transition_probability.update(sp.vstack(data.get()))
        else:
            # Modified code to run in a single process for debugging
            data = []
            for state in self.states:
                try:
                    result = self._worker(None, state)  # Replace None with an actual queue if necessary
                    data.append(result)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    traceback.print_exc()
                    sys.exit(1)

            if self.__sample_reward:
                data = np.array(data, dtype=object)
                self.state_transition_probability.update(sp.vstack(data[:, 0]))
                self.rewards.update(
                    np.array(data[:, 1].tolist(), dtype=self.rewards.dtype)
                )
            else:
                self.state_transition_probability.update(sp.vstack(data.get()))
        self.__sampler = None
        end_time = time()
        verbose("Sampling is done. %f (sec) elapsed.\n" % (end_time - start_time))

    def load(self, filename):

        data = np.load(filename, allow_pickle=False)
        state_lists = []
        for idx in range(data["states.num_lists"].item()):
            state_lists.append(data["states.data." + str(idx)])
        self.states = States(
            *state_lists,
            cycles=data["states.cycles"],
            terminal_states=data["states.terminal_states"]
        )

        self.actions = Actions(data["actions.data"])

        self.rewards = Rewards(
            self.states, self.actions, sparse=data["rewards.issparse"].item()
        )
        if self.rewards.issparse:
            self.rewards.update(
                sp.csr_matrix(
                    (
                        data["rewards.data"],
                        data["rewards.indices"],
                        data["rewards.indptr"],
                    ),
                    shape=(self.states.num_states, self.actions.num_actions),
                )
            )
        else:
            self.rewards.update(data["rewards.data"])

        self.state_transition_probability = StateTransitionProbability(
            self.states, self.actions
        )
        self.state_transition_probability.update(
            sp.csr_matrix(
                (
                    data["state_transition_probability.data"],
                    data["state_transition_probability.indices"],
                    data["state_transition_probability.indptr"],
                ),
                shape=(
                    self.states.num_states * self.actions.num_actions,
                    self.states.num_states,
                ),
            )
        )
        self.policy = Policy(self.states, self.actions)
        self.policy.update(data["policy.data"])
        self.discount = data["discount"].item()

    def save(self, filename):

        if not sp.isspmatrix_csr(self.state_transition_probability.tospmat()):
            self.state_transition_probability.tocsr()

        kwargs = {
            "states.num_lists": len(self.states.shape),
            "states.cycles": self.states.info(return_cycles=True),
            "states.terminal_states": self.states.terminal_states,
            "actions.data": self.actions.toarray(),
            "rewards.issparse": self.rewards.issparse,
            "state_transition_probability.data": self.state_transition_probability.tospmat().data,
            "state_transition_probability.indices": self.state_transition_probability.tospmat().indices,
            "state_transition_probability.indptr": self.state_transition_probability.tospmat().indptr,
            "policy.data": self.policy.toarray(),
            "discount": self.discount,
        }
        for idx, state_list in enumerate(self.states.info(return_data=True)):
            kwargs["states.data." + str(idx)] = state_list
        if self.rewards.issparse:
            kwargs["rewards.data"] = (self.rewards.tocsr().data,)
            kwargs["rewards.indices"] = (self.rewards.tocsr().indices,)
            kwargs["rewards.indptr"] = (self.rewards.tocsr().indptr,)
        else:
            kwargs["rewards.data"] = self.rewards.toarray()

        savez(filename, **kwargs)

if __name__ == "__main__":
    # Define States
    states = States(np.linspace(0, 1, 100)) # 100 POINTS in 0 ~ 1 Total 100?, # Input States Lists

    actions = Actions(np.linspace(0, 1, 10)) # 10 POINTS in 0 ~ 1 Total 100

    rewards = Rewards(states, actions) # Get Rewards [ Age ]

    state_transition_prob = StateTransitionProbability(states, actions)
    for s in range(states.num_states):
        for a in range(actions.num_actions):
            state_transition_prob[s, a, s] = 1.0 # Why lists [states, action, states]?

    policy = Policy(states, actions) # Policy
    # Without Terminal Conditions
    mdp = MarkovDecisionProcess(
        states=states,
        actions=actions,
        rewards=rewards,
        state_transition_probability=state_transition_prob,
        policy=policy,
        discount=0.99,
    ) # Create MDP