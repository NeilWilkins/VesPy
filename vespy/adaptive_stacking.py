# START:
# INPUT PARAMETERS: a stream
# OUTPUT: a stream that has been shifted differently
# CAN THEN LOOP BACK INTO self
import numpy as np
import itertools
from scipy.optimize import minimize_scalar

def misfit(tr, stack, p=3):

    P_p = np.sum(np.abs(stack - tr) ** p)

    return P_p

def shift_trace(tr, tau):

    tr.data = np.roll(tr.data, int(round(tau)))

def stack(st):

    stack = np.sum([tr.data for tr in st]) / len(st)

    return stack

def optimised_shifts(st):

    stack = stack(st)

    shifts = []
    #Pp_vector = []

    for tr in st:

        def _misfit(tau):

            shift_trace(tr, tau)

            return misfit(tr, stack, p=3)

        res = minimize_scalar(_misfit, bounds=(-40, 40))

        shift = res.x

        shifts.append(shift)
        #Pp_vector.append(_misfit(tau))

    return shifts

def iterative_shifts(st, N):

    overall_shifts = np.zeros(len(st))

    for _ in itertools.repeat(None, N):

        shifts = optimised_shifts(st)

        for i, tr in enumerate(st):

            shift_trace(tr, shifts[i])

        overall_shifts += shifts

    return overall_shifts
