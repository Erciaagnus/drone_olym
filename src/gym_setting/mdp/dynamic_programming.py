#!/usr/bin/env python3

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