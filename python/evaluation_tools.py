import numpy as np

def unproper_order(x, y, z, verbose=False):
    # Evaluate the order of convergence for when the theoretical error does not approach 0
    # This is useful for when we have Δx→0, but Δt is fixed, for instance.

    # Given:
    #   x:: List[float]: the inputs
    #   y:: List[float]: the outputs (for instance, the resulting errors)
    #   z:: List[float]: the outputs, with commonly-scaled input values
    # Return:
    #   float: The order of convergence

    if verbose:
        print(f"Our errors were {y}")
        print(f"and {z}")

    # First, log the squared-difference terms
    log_out = np.log((y - z) ** 2)
    log_in = np.log(x)

    # Return (half) the slope of the line-of-best-fit
    return np.polyfit(log_in, log_out, 1)[0] / 2

def unproper_order_func(xfunc, yfunc, Ns, common_scale, verbose=False):
    # Evaluate the order of convergence of the given function yfunc, with input values Ns,
    # against variables xfunc(Ns). 
    #
    # Given:
    #   xfunc:: Function: Convert the input values Ns into a variable h for convergence analysis.
    #   yfunc:: Function: The function whose convergence we analyze.
    #   Ns:: List[int]: The input values for the function yfunc.
    #   common_scale:: float: The common scale for the input values.
    # Return:
    #   float: The order of convergence

    # First, apply yFunc to our input values
    y = np.array([yfunc(N) for N in Ns])

    # Apply the scaled Ns to yFunc
    z = np.array([yfunc(int(N * common_scale)) for N in Ns])

    # Finally, determine the convergence
    return unproper_order(np.vectorize(xfunc)(Ns), y, z, verbose)

def proper_order(x, y, verbose=False):
    # Evaluate the order of convergence for when the theoretical error does converge to 0

    # Given:
    #   x:: List[float]: the inputs
    #   y:: List[float]: the outputs (for instance, the resulting errors)
    # Return:
    #   float: The order of convergence

    if verbose:
        print(f"Our errors were {y}")

    return np.polyfit(np.log(x), np.log(y), 1)[0]

def proper_order_func(xfunc, yfunc, Ns, verbose=False):
    # Evaluate the order of convergence of the given function yfunc, with input values Ns,
    # against variables xfunc(Ns). 
    #
    # Given:
    #   xfunc:: Function: Convert the input values Ns into a variable h for convergence analysis.
    #   yfunc:: Function: The function whose convergence we analyze.
    #   Ns:: List[int]: The input values for the function yfunc.
    # Return:
    #   float: The order of convergence

    # First, apply yFunc to our input values
    y = np.array([yfunc(N) for N in Ns])
    
    # Finally, determine the convergence
    return proper_order(np.vectorize(xfunc)(Ns), y, verbose)