""" Do facet extension tests on a general polytope
"""
import numpy as np
import gurobipy as grb
import subprocess
import time
import os.path


def compute_vertices(model, num_trials=100, verbose=False, time_limit=60):
    """ Stochastically compute the vertices of the covnex hull of the feasible
    region specified by the given model using a "pea-shooter" experiment.

    This function repeatedly solves the given model with objective vectors
    sampled uniformly from the unit sphere, and records all the extreme points
    that it finds.
    
    Args:
        model (gurobipy.Model):
            Gurobi model specifying the constraints of the feasible region. If
            the model has an objective function, it will be overwritten.
        num_trials (int, optional):
            Number of random objective vectors to sample (i.e. number of times
            the model will be solved). Default is 100.
        verbose (bool, optional):
            If True, displays periodic updates on progress of the algorithm.
        time_limit (float, optional):
            Maximum number of seconds to run the algorithm. Default is 60.
    
    Returns:
        ndarray:
            The vertices stored as columns.

    Notes:
    - The returned vertices will be in the order returned by Gurobi's
      model.getVars() function.
    - Will attempt to run the specified number of trials, unless the time limit
      is reached first.
    """
    # Turns off Gurobi output
    # (and restores it to its original state at the end)
    init_flag = model.Params.OutputFlag
    model.Params.OutputFlag = False
    model_vars = model.getVars()
    num_vars = len(model_vars)
    c = _sphere_sample(num_vars, num_points=num_trials)

    vertices = list()

    last_new_vertex = 0
    t0 = time.time()
    elapsed_time = time.time() - t0
    for t in range(num_trials):
        obj = grb.quicksum(model_vars[i] * c[i, t] for i in range(num_vars))
        model.setObjective(obj, sense=grb.GRB.MINIMIZE)
        model.optimize()
        if model.status != grb.GRB.OPTIMAL:
            raise RuntimeError(
                "Model not solved to optimality. "
                "We do not currently support unbounded "
                "polyhedra, which may be the source of this "
                f"error (model status = {model.status})."
            )
        soln_vec = np.array([var.X for var in model_vars])
        if not _is_member(soln_vec, vertices):
            vertices.append(soln_vec)
            last_new_vertex = t
        if verbose and (t - last_new_vertex) % 1000 == 0:
            print("\b" * 80, end="", flush=True)
            if t - last_new_vertex == 0:
                print(
                    f"Last new vertex was found  <1k "
                    f"iterations ago ({len(vertices):6d} total vertices)",
                    end="",
                    flush=True,
                )
            else:
                print(
                    "Last new vertex was found "
                    f"{(t-last_new_vertex)//1000:3d}k "
                    f"iterations ago ({len(vertices):6d} total vertices)",
                    end="",
                    flush=True,
                )
        elapsed_time = time.time() - t0
        if elapsed_time > time_limit:
            if verbose:
                print("Terminating due to time limit")
            break
    if verbose:
        print()

    model.Params.OutputFlag = init_flag

    return np.array(vertices).T


def compute_facets(
    vertices,
    out_fname=None,
    tmp_facet_fname="tmpf",
    tmp_vertex_fname="tmpv",
    qhull_binary_path="qhull",
):
    """ Compute the convex hull of the specified vertices using Qhull.
    
    Args:
        vertices (ndarray):
            Matrix with vertices stored as columns.
        out_fname (str or None, optional):
            If not None, facets are written to the specified file. Otherwise,
            facets are returned (see `Returns`). Default is None.
        tmp_facet_fname (str, optional):
            Name of the temporary file used to store the facets computed by
            Qhull. This file will be deleted before this function returns.
            Default is `tmpf`.
        tmp_vertex_fname (str, optional):
            Name of the temporary file used to store the vertices passed to
            Qhull. This file will be deleted before this function returns.
            Default is `tmpv`.
        qhull_binary_path (str, optional):
            Path to the qhull binary (equivalently, the thing you would type at
            the command line to make qhull go brr). Default is `qhull`.

    Returns:
        If out_fname is None:
            ndarray:
                Matrix A of lhs coefficients of the facet-defining
                inequalities, one per row.
            ndarray:
                Vector b of the rhs coefficients of the facet-defining
                inequalities.
        Else:
            None.

    Raises:
        FileNotFoundError:
            If the specified qhull binary does not exist.

    Notes:
    - This function passes data in and out of qhull via text files.
    - Qhull and file deletion is done using the subprocess.run function.
    """
    if not os.path.isfile(os.path.expanduser(qhull_binary_path)):
        raise FileNotFoundError(f"Could not find qhull binary: {qhull_binary_path}")
    vertices_string = _vert_to_string(vertices.T)  # Original function requires
    # the transpose

    # This currently has to be done by reading in and out of files--really?
    with open(tmp_vertex_fname, "w") as f:
        f.write(vertices_string)

    with open(tmp_facet_fname, "w") as f:
        cmd = f"{qhull_binary_path} n < {tmp_vertex_fname}"
        subprocess.run(cmd, shell=True, stdout=f)  # Security risk w/shell=True?

    A, b = _read_facets(tmp_facet_fname)

    subprocess.run(f"rm {tmp_vertex_fname}", shell=True)
    subprocess.run(f"rm {tmp_facet_fname}", shell=True)

    if out_fname is None:
        return A, b
    else:
        _write_facets(out_fname, A, b)


def compute_dimension(vertices):
    """ Compute the dimension of the convex hull of the given list of vertices

    Args:
        vertices (ndarray):
            Matrix with vertices stored as columns.

    Returns:
        int:
            Dimension of the convex hull of the specified vertices.
    """
    M = np.vstack((vertices, np.ones(vertices.shape[1])))
    return np.linalg.matrix_rank(M) - 1


def inequality_is_valid(lhs, rhs, model, tipe="leq", tol=1e-9):
    """ Determine whether the specified inequality is valid.

    Args:
        lhs (Gurobi LinExpr): 
            Expression for the lhs of the specified constraint (includes
            coefficients and variables--_not_ a vector of coefficients).
        rhs (int or float): 
            Rhs value for the constraint.
        model (gurobipy.Model):
            Gurobi model specifying the constraints of the feasible region. If
            the model has an objective function, it will be overwritten.
        tipe (str, optional):
            Inequality type. Must be `leq` or `geq`. Default is `leq`.
        tol (float, optional):
            Absolute tolerance used to determine whether the constraint is
            valid. Default is 1e-9.

    Returns:
        bool:
            True if the specified constraint is valid.

    Raises:
        RuntimeError:
            If the model cannot be solved to optimality.
    """
    if tipe not in ["leq", "geq"]:
        raise RuntimeError("Inequality tipe must be 'leq' or 'geq'")
    if tipe == "leq":
        model.setObjective(lhs, sense=grb.GRB.MAXIMIZE)
    else:
        model.setObjective(lhs, sense=grb.GRB.MINIMIZE)
    model.optimize()
    if model.status != grb.GRB.OPTIMAL:
        raise RuntimeError("Model not solved to optimality")
    if tipe == "leq":
        return model.objval <= rhs + tol
    else:
        return model.objval >= rhs - tol


def inequality_is_supporting_hyperplane(lhs, rhs, model, tipe="leq", tol=1e-9):
    """ Determine whether the specified inequality is a supporting hyperplane.

    Args:
        lhs (Gurobi LinExpr): 
            Expression for the lhs of the specified constraint (includes
            coefficients and variables--_not_ a vector of coefficients).
        rhs (int or float): 
            Rhs value for the constraint.
        model (gurobipy.Model):
            Gurobi model specifying the constraints of the feasible region. If
            the model has an objective function, it will be overwritten.
        tipe (str, optional):
            Inequality type. Must be `leq` or `geq`. Default is `leq`.
        tol (float, optional):
            Absolute tolerance used to determine whether the constraint is
            valid. Default is 1e-9.

    Returns:
        bool:
            True if the specified constraint defines a supporting hyperplane.

    Raises:
        RuntimeError:
            If the model cannot be solved to optimality, or the specified
            inequality is not valid.

    Notes:
    - This function preserves the objective function of the model.
    """
    is_valid = inequality_is_valid(lhs, rhs, model, tipe=tipe, tol=tol)
    if not is_valid:
        raise RuntimeError("Provided inequality is not valid")

    # Change the objective to prevent unboundedness
    original_obj = model.getObjective()
    original_sense = model.Params.ModelSense
    model.setObjective(0, sense=grb.GRB.MINIMIZE)

    tmp_constr = model.addConstr(lhs == rhs)

    model.optimize()
    if model.status in [grb.GRB.INFEASIBLE, grb.GRB.INF_OR_UNBD]:
        is_sh = False
    elif model.status == grb.GRB.OPTIMAL:
        is_sh = True
    else:
        model.setObjective(original_obj, sense=original_sense)
        model.remove(tmp_constr)
        raise RuntimeError(
            "Could not ascertain whether given inequality is a "
            f"supporting hyperplane (Gurobi status code {model.status})"
        )

    # Reset the model to its original state
    model.setObjective(original_obj, sense=original_sense)
    model.remove(tmp_constr)
    model.update()
    return is_sh


def compute_dimension_of_face(
    lhs, rhs, model, tipe="leq", tol=1e-9, num_trials=100, verbose=False
):
    """ Compute the dimension of the face defined by the specified inequality.

    Args:
        lhs (Gurobi LinExpr): 
            Expression for the lhs of the specified constraint (includes
            coefficients and variables--_not_ a vector of coefficients).
        rhs (int or float): 
            Rhs value for the constraint.
        model (gurobipy.Model):
            Gurobi model specifying the constraints of the feasible region. If
            the model has an objective function, it will be overwritten.
        tipe (str, optional):
            Inequality type. Must be `leq` or `geq`. Default is `leq`.
        tol (float, optional):
            Absolute tolerance used to determine whether the constraint is
            valid. Default is 1e-9.
        num_trials (int, optional):
            Number of trials used to compute the vertices of the face. Default
            is 100.
        verbose (bool, optional):
            If True, displays progress as we compute the vertices of the
            specified face.

    Returns:
        int:
            The dimension of the specified face. If the face is valid but not a
            supporting hyperplane, returns 0.

    Raises:
        RuntimeError:
            If the specified inequality is not valid (or a solve fails).
    """
    if not inequality_is_supporting_hyperplane(lhs, rhs, model, tipe=tipe, tol=tol):
        return 0

    tmp_constr = model.addConstr(lhs == rhs)
    vertices = compute_vertices(model, num_trials=num_trials, verbose=verbose)
    model.remove(tmp_constr)
    model.update()
    return compute_dimension(vertices)


def print_facets(A, b, var_names, num_decimals=1, tol=1e-7):
    """ Print the computed facets in a human-readable format.

    Args:
        A (ndarray):
            lhs coefficients of the facets, one per row.
        b (ndarray):
            rhs values of the facets (vector).
        var_names (list of str):
            String to represent each variable name, assumed to be in the same
            order as the columns of A.
        num_decimals (int, optional):
            Number of decimals to display for each coefficient. Default is 1.
        tol (float, optional):
            Tolerance used to determine whether a float is actually an integer.

    Returns:
        None
    """
    num_facs = A.shape[0]
    num_vars = A.shape[1]
    for i in range(num_facs):
        nz = np.where(np.abs(A[i, :]) > tol)[0]
        if len(nz) == 0:
            print("ZERO??")
        elif len(nz) >= 2:
            first = _get_token_string(
                A[i, nz[0]], var_names[nz[0]], num_decimals, tol, first=True
            )
            rhs = f"{b[i]:.{num_decimals}f}"
            nz = nz[1:]
            signs = ["+" if val >= 0 else "-" for val in A[i, nz]]
            tokens = ["" for _ in range(len(nz))]
            for (k, j) in enumerate(nz):
                tokens[k] = _get_token_string(A[i, j], var_names[j], num_decimals, tol)
            rest = " ".join(tokens)
            print(first, rest, "<=", rhs)
        else:  # Bound inequality
            aye = A[i, nz[0]]
            bee = b[i]
            inq = "<="
            if A[i, nz[0]] < 0:
                aye *= -1
                bee *= -1
                inq = ">="
            first = _get_token_string(
                aye, var_names[nz[0]], num_decimals, tol, first=True
            )
            rhs = f"{bee:.{num_decimals}f}"
            print(first, inq, rhs)


def _is_member(vec, lyst, tol=1e-7):
    """ Check whether the specified vector is in the list of vectors.

    Args:
        vec (ndarray):
            A particular vector.
        lyst (list of ndarray):
            List of vectors.
        tol (float, optional):
            Two float vectors are considered equal if their 2-norm <= tol.
            Default is 1e-7.

    Returns:
        bool:
            True if the vector is contained in the list.
    
    Notes:
    - Assumes that each vector in lyst is of the same length (namely, the
      length of vec).
    """
    if len(lyst) == 0:
        return False
    err = np.min(np.linalg.norm(lyst - vec, axis=1))
    return err < tol


def _vert_to_string(vertices):
    """ Convert a matrix of vertices to a string.

    This function is used to write the vertices to a file which will be read in
    by Qhull.
    
    Args:
        vertices (ndarray):
            Matrix with vertices stored as columns.

    Returns:
        str:
            Vertices formatted nicely for Qhull.
    """
    m, n = vertices.shape
    s = f"{n:d}\n{m:d}\n"
    for row in vertices:
        s += " ".join("{:f}".format(val) for val in row) + "\n"
    return s


def _sphere_sample(dim, num_points=1):
    """ Sample a vector uniformly from the unique sphere.

    Args:
        dim (int):
            Dimension of the sphere to sample from. For example, dim=2
            corresponds to sampling from S^1 (i.e., the unit circle in R^2).
        num_points (int, optional):
            Number of points to sample. If > 1, points are returned as columns.
            Default is 1.

    Returns:
        ndarray:
            Point, or points stored as columns.
    """
    vec = np.random.randn(dim, num_points)
    vec /= np.linalg.norm(vec, axis=0)
    if num_points == 1:
        return np.reshape(vec, (dim,))  # Flatten out the (n,1) to (n,)
    else:
        return vec


def _read_facets(fname):
    """ Read the facets computed by Qhull from a file, and scale them.
    
    Args:
        fname (str):
            File name of the file output by QHull.

    Returns:
        ndarray:
            Matrix A of lhs coefficients of the facet-defining
            inequalities, one per row.
        ndarray:
            Vector b of the rhs coefficients of the facet-defining
            inequalities.

    Notes:
    - Scales the inequalities to be...O(1)-ish?
    """
    with open(fname, "r") as f:
        dim = int(f.readline()) - 1
        n_facets = int(f.readline())
        data = np.array(
            [[float(val) for val in f.readline().split()] for _ in range(n_facets)]
        )
        A, b = data[:, :-1], -data[:, -1]

    for i in range(A.shape[0]):
        ix = np.where(np.abs(A[i, :]) > 1e-7)[0]
        scale = np.min(np.abs(A[i, ix]))
        if np.abs(b[i]) > 1e-7:
            scale = np.minimum(scale, np.abs(b[i]))
        A[i, :] /= scale
        b[i] /= scale
    return A, b


def _write_facets(fname, A, b):
    """ Write the specified facets to a file for later human reading.

    Args:
        fname (str):
            Name of the file to write the facets to.
        A (ndarray):
            Matrix A of lhs coefficients of the facet-defining
            inequalities, one per row.
        b (ndarray):
            Vector b of the rhs coefficients of the facet-defining
            inequalities.
    Returns:
        None

    Notes:
    - Rounds floats to 5 decimal places.
    - If the file already exists, it will be overwritten.
    """
    header = f"{A.shape[1]+1:d}\n{A.shape[0]}\n"
    body = "\n".join(
        " ".join(f"{round(A[i,j],5):+}".replace("+", " ") for j in range(A.shape[1]))
        + f" {round(b[i],5):+}".replace("+", " ")
        for i in range(A.shape[0])
    )
    with open(fname, "w") as f:
        f.write(header + body)


def _get_token_string(coef, vname, num_decimals, tol, first=False):
    """ Convert a (coefficient, variable_name) pair into a nicely-formatted
    string.
    
    Args:
        coef (float):
            Coefficient value.
        vname (str):
            Name of the variable.
        num_decimals (int):
            Number of decimal places to round to.
        tol (float):
            Tolerance used to determine if the coefficient is in {-1,0,1}.
        first (bool, optional):
            If True, assumes that the coefficient is ths first appearing in a
            row, and does not prepend a + sign if the coefficient is positive.
            Default is False (so that  + sign is prepended).

    Returns:
        str of None:
            Resulting formatted string, or None if the coefficient is zero.
            
    Notes:
    - Returns None if the coefficient is zero.
    """
    if np.abs(coef) < tol:  # Coefficient is zero
        return None
    if first:
        sgn = "" if coef > 0 else "-"
    else:
        sgn = "+ " if coef > 0 else "- "
    if np.abs(1 - np.abs(coef)) < tol:  # Coefficient in {-1,1}
        return f"{sgn}{vname}"
    return f"{sgn}{np.abs(coef):.{num_decimals}f} {vname}"


if __name__ == "__main__":
    print("poly.py has no main")
