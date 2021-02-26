# poly
Basic functions to use Gurobi and Qhull to study the polyhedral structure of small mixed-integer programs. Poly functions only for polytopes (i.e. bounded polyhedra). It will crash if the polyhedron in question is unbounded.

## Requirements
- [Gurobi](https://www.gurobi.com/): Gurobi is proprietary, but there are free licensing options available for academics.
- [QHull](http://qhull.org/html/qhull.htm): QHull is freely available. The only binary required for `poly` is `qhull` itself (not `qconvex`, `qdelaunay`, etc.).

## Description

Poly allows the user to specify the feasible region of a mixed-integer program by creating a Gurobi model which encodes the constraints. It provides a series of utility functions to explore the polyhedral structure of the convex hull of the specified constraints. At its core, Poly does the following two things:

1. Computes the vertices (V-representation) of the convex hull of the specified model using a "peashooter experiment". That is, the vertices are computed by repeatedly solving the specified model with objective vectors sampled uniformly from the unit sphere. While this method is not guaranteed to find all the vertices of the convex hull, it is very effective for small (non-pathological) models.
2. Compute the facets (H-representation) of the convex hull of the specified model using Qhull. Because computing convex hulls if very expensive, this is practical only for small models (<= 12-ish variables, maybe).

It also offers several utility functions for checking whether specified inequalities are valid/supporting hyperplanes/facet-defining.

## Example

Consider a knapsack polytope: X=conv{x&isin;{0,1}<sup>5</sup> | 3.5x<sub>1</sub> + 6x<sub>2</sub> + 7.5x<sub>3</sub> + 8x<sub>4</sub> + 9.1x<sub>5</sub> &le; 26}. The following code snippet first computes the vertices of X, then its convex hull using Qhull, and displays the results in a nice, human-readable format.

```python

import poly
import gurobipy as grb

# Data for the knapsack model
n = 5
a = [3.5, 6.0, 7.5, 8.0, 9.1]
b = 26

# Build the knapsack model in Gurobi
model = grb.Model()
x = model.addVars(n, vtype=grb.GRB.BINARY)
model.addConstr(grb.quicksum(a[i] * x[i] for i in range(n)) <= b)
model.Params.OutputFlag = 0
model.update()  # Critical! Make sure to call model.update() before using it.

# Compute the vertices
vertices = poly.compute_vertices(model, num_trials=500)
num_vertices = vertices.shape[1]  # Vertices are stored in columns
print("Found", num_vertices, "vertices")

# You must specify the location of the qhull binary
# (Default assumes `qhull` is on the PATH)
A, b = poly.compute_facets(
    vertices, qhull_binary_path="~/qhull-2015.2/bin/qhull"
)

# Print the results nicely
var_names = [f"x[{i}]" for i in range(n)]
poly.print_facets(A, b, var_names, num_decimals=2)
```
The results:
```
Found 27 vertices
x[0] + x[2] + x[3] + x[4] <= 3.00
x[0] + x[1] + x[3] + x[4] <= 3.00
x[1] + x[2] + x[3] + x[4] <= 3.00
x[0] + x[1] + x[2] + x[3] + 2.00 x[4] <= 4.00
x[4] <= 1.00
x[3] <= 1.00
x[2] >= -0.00
x[0] + x[1] + x[2] + x[4] <= 3.00
x[0] <= 1.00
x[0] >= 0.00
x[1] >= 0.00
x[4] >= 0.00
x[1] <= 1.00
x[2] <= 1.00
x[3] >= 0.00
```
We see that the convex hull is defined by the 10 trivial inequalities x<sub>i</sub> &le; 1 and x<sub>i</sub> &ge; 0 for all i, and 5 additional cover inequalities.

Continuing this example, we can also compute the dimension of the polytope:

```python
dim = poly.compute_dimension(vertices)
print("The polytope is", dim, "dimensional")
```
with the result
```
The polytope is 5 dimensional
```
as expected.

We can check whether a given inequality is valid:
```python
lhs = x[0] + x[3]
rhs = 1
v = poly.inequality_is_valid(lhs, rhs, model, tipe="leq")
print(v)
```
```
False
```

We can check whether a given inequality is a supporting hyperplane:
```python
lhs = x[2]
rhs = 1
v = poly.inequality_is_supporting_hyperplane(lhs, rhs, model, tipe="leq")
print(v)
```
```
True
```

Can we can compute the dimension of a face (which, among other things, can tell us whether an inequality is facet-defining):
```python
lhs = x[0] + x[1] + x[2] + x[3] + 2 * x[4]
rhs = 4
dim = poly.compute_dimension_of_face(lhs, rhs, model, tipe="leq", num_trials=100)
print("Face is", dim, "dimensional")
```
```
Face is 4 dimensional
```

This is everything that Poly can do.
