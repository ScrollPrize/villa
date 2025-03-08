Theseus
Theseus is a library for differentiable nonlinear optimization built on PyTorch.

Theseus is motivated by problems in robotics and computer vision that can be formulated as differentiable nonlinear least squares optimization problems, such as Simultaneous Localization and Mapping (SLAM), motion planning, and bundle adjustment. These problems can be broadly categorized as doing structured learning, where neural components can be modularly mixed with known priors to get the benefit of deep learning in a way that adds value over classical methods. While interest in this area is rapidly increasing, existing work is fragmented and built using application-specific codebases. Theseus fills this gap by providing a problem-agnostic platform for structured learning, letting the user easily combine neural networks with priors represented as differentiable blocks of nonlinear optimization problems, and do end-to-end training over these.

This tutorial introduces the basic building blocks for solving such optimization problems in Theseus; in the following tutorials, we will show how to put together these building blocks to solve optimization problems with various aspects and increasing complexity. We cover six conceptual building blocks in this tutorial:

Variables: named wrappers for torch tensors that form the fundamental data type for defining optimization problems in Theseus. (Section 1)
Cost functions: computes an error term as a function of one or more variables, and which are the functions to be minimized by Theseus optimizers. (Section 2)
Cost weights: computes a weight that modifies the contribution of one or more cost functions to the overall objective. (Section 3)
Objective: compiles multiple cost functions and weights to define the structure of an optimization problem. (Section 4)
Optimizer: implements an optimization algorithm (e.g., Gauss-Newton, LevenbergMarquardt) that can be used to minimize an Objective. (Section 5)
TheseusLayer: groups an objective and optimizer and serves as the interface between torch modules upstream/downstream and differentiable optimization problems. (Section 6)
1. Variables
Optimization objectives in Theseus are a function of th.Variable objects, which are torch.tensor wrappers of different types (e.g., 2D points, rotation groups, etc.) that can be, optionally, associated with a name. In Theseus, we require the first dimension of all variables to be a batch dimension (similar to the convention in PyTorch modules). We describe here two main operations common to all Variables: (1) Creating variables and (2) Updating Variables.

1.1 Creating variables
Variables can be created using the generic th.Variable interface, or through a sub-class with custom functionality. Many Variables used in Theseus applications are manifolds; therefore, Theseus provides several Variable sub-classes supporting commonly used manifolds, such as vectors, 2-D/3-D points, 2-D rotations and 2-D rigid transformations. We show some example usage below:

import torch
import theseus as th
# Create a variable with 3-D random data of batch size = 2 and name "x"
x = th.Variable(torch.randn(2, 3), name="x")
print(f"x: Named variable with 3-D data of batch size 2:\n  {x}\n")

# Create an unnamed variable. A default name will be created for it
y = th.Variable(torch.zeros(1, 1))
print(f"y: Un-named variable:\n  {y}\n")

# Create a named SE2 (2D rigid transformation) specifying data (batch_size=2)
z = th.SE2(x_y_theta=torch.zeros(2, 3).double(), name="se2_1")
print(f"z: Named SE2 variable:\n  {z}")
x: Named variable with 3-D data of batch size 2:
  Variable(tensor=tensor([[-0.5966,  0.7318,  2.2279],
        [ 0.6040,  0.3843, -2.0580]]), name=x)

y: Un-named variable:
  Variable(tensor=tensor([[0.]]), name=Variable__1)

z: Named SE2 variable:
  SE2(xytheta=tensor([[0., 0., 0.],
        [0., 0., 0.]], dtype=torch.float64), name=se2_1)
1.2 Updating variables
After creating a variable, its value can be updated via update() method. Below we show a few examples and possible errors to avoid when updating variables.

# Example usage of `update`
print("Example usage of `update`: ")
print(f"  Original variable: {x}")
x.update(torch.ones(2, 3))
print(f"  Updated variable: {x}\n")

# The following inputs don't work
print("Error inputs for a Variable `update`:")
try:
    # `update` expects input tensor to respect the internal data format
    x.update(torch.zeros(2, 4))
except ValueError as e:
    print(f"  Mismatched internal data format:")
    print(f"    {e}")
try:
    # `update` expects a batch dimension
    x.update(torch.zeros(3))
except ValueError as e:
    print(f"  Missing batch dimension: ")
    print(f"    {e}\n")
    
# However the batch size can be changed via `update`
print("Change variable batch size via `update`:")
x.update(torch.ones(4, 3))
print(f"  New shape: {x.shape}")
Example usage of `update`: 
  Original variable: Variable(tensor=tensor([[-0.5966,  0.7318,  2.2279],
        [ 0.6040,  0.3843, -2.0580]]), name=x)
  Updated variable: Variable(tensor=tensor([[1., 1., 1.],
        [1., 1., 1.]]), name=x)

Error inputs for a Variable `update`:
  Mismatched internal data format:
    Tried to update tensor x with data incompatible with original tensor shape. Given torch.Size([4]). Expected: torch.Size([3])
  Missing batch dimension: 
    Tried to update tensor x with data incompatible with original tensor shape. Given torch.Size([]). Expected: torch.Size([3])

Change variable batch size via `update`:
  New shape: torch.Size([4, 3])
Over the next few sections, we will see the different ways that Variables are used in optimization problems in Theseus.

2. Cost functions
A Theseus cost function represents an error function of one or more Theseus variables. Thus, cost functions capture the core quantities being optimized in Theseus.

For this reason, a cost function needs to know which variables can be optimized, and which variables are not allowed to be optimized. In Theseus, we represent this concept by having two kinds of variables:

optimization variables: variables that can be modified by Theseus optimizers for minimizing the objective.
auxiliary variables: variables that are required to compute the objective, but that remain constant to Theseus optimizers.
In Theseus, a Variable becomes an optimization variable if it is defined as such in the creation of a cost function. All optimization variables must be sub-classes of th.Manifold.

A cost function thus needs to be created with its optimization (required) and auxiliary variables (optional) declared. The core operations provided by a cost function are the computation of the error and the error's Jacobian using the latest values of its variables. The th.CostFunction class is an abstract class, and to instantiate it, one needs to implement the error computation and the Jacobian. A cost function must return a torch tensor as its error.

As a simple example, we will show how to use the th.Difference cost function, which is a concrete sub-class of th.CostFunction. Below, we instantiate this cost function with two Vector variables, one optimization and one auxiliary.

We then show a few useful operations on the cost function: how the cost function can access its optimization and auxiliary variables; the computation of its error, which is defined as optim_var - target for the th.Difference c); how the error changes when an underlying Variable is updated. Lastly, we show the computation of its jacobians: this returns a list of jacobians, with one entry per optimization variable.

# Note: CostWeight is a weighting quantity required for constructing a cost function.
# We explain it in Section 3; for this example, we simply create it but we do not use it.
w1 = th.ScaleCostWeight(2.0)

# Create a Difference cost function
optim_var = th.Vector(tensor=torch.ones(1, 2), name="x1")
target = th.Vector(tensor=torch.zeros(1, 2), name="target")
cf = th.Difference(optim_var, target, w1)

# A cost function can retrieve its optimization and auxiliary variables 
print ("Retrieving the optimization and auxiliary variables from the cost function:")
print("  Optimization variables: ", list(cf.optim_vars))
print("  Auxiliary variables: ", list(cf.aux_vars))
print("")

# Cost functions compute the error using the values of the variables.
error = cf.error()
print(f"Original cost function (unweighted) error:\n  {error} of shape {error.shape}\n")

# Cost functions use the _latest_ values of the variables,
# as shown by the error values after the variable is updated.
print("Updating optimization variables by factor of 2: ")
optim_var.update(2 * torch.ones(1, 2))
print(f"  Updated variables: {optim_var}")
# Error is now twice as large as the one printed above
print(f"  Updated (unweighted) error: {cf.error()}\n")

# Compute the (unweighted) jacobians and error
# This returns a list of jacobians, with one entry per _optimization_ variable.
print("Computing cost function's (unweighted) jacobians:")
jacobians, error = cf.jacobians()  # Note cf.jacobians also returns error 
print(f"  Jacobians: {type(jacobians)} of length {len(jacobians)}")
print(f"    {jacobians[0]}")
# The i-th jacobian has shape (batch_size, cf.dim(), i-th_optim_var.dof())
print(f"    Shape of 0-th Jacobian: {jacobians[0].shape}")
Retrieving the optimization and auxiliary variables from the cost function:
  Optimization variables:  [Vector(dof=2, tensor=tensor([[1., 1.]]), name=x1)]
  Auxiliary variables:  [Vector(dof=2, tensor=tensor([[0., 0.]]), name=target)]

Original cost function (unweighted) error:
  tensor([[1., 1.]]) of shape torch.Size([1, 2])

Updating optimization variables by factor of 2: 
  Updated variables: Vector(dof=2, tensor=tensor([[2., 2.]]), name=x1)
  Updated (unweighted) error: tensor([[2., 2.]])

Computing cost function's (unweighted) jacobians:
  Jacobians: <class 'list'> of length 1
    tensor([[[1., 0.],
         [0., 1.]]])
    Shape of 0-th Jacobian: torch.Size([1, 2, 2])
In Tutorial 3, we will delve into the internals of a cost function and show how to construct custom cost functions.

3. Cost weights
The Theseus cost weight is a weighting function applied to cost functions: it computes a weight as a function of one or more variables, and applies it to the error of one or more cost functions. The cost weights are thus a way of modifying the error of a cost function in the optimization problem. Cost weights add another layer of abstraction that help trade-off between different cost functions in an objective.

The th.CostWeight class is abstract, as any function of Variables can be used to create CostWeight. Theseus provides a number of concrete CostWeight sub-classes currently:

ScaleCostWeight, where the weighting function is a scalar real number,
DiagonalCostWeight, where the the weighting function is a diagonal matrix,
th.eb.GPCostWeight, where the weighting function represents the inverse covariance function of an exactly sparse Gaussian process.
The main use of the CostWeight is to support the weighted_error and weighted_jacobians_and_error functions of the cost functions; so these sub-classes implement their (defined) weighting functions.

The Variables used in a CostWeight may be named or unnamed; however, using a named Variable allows us to update the value of the CostWeight directly; this is especially useful in updating the Objective or the TheseusLayer whenever the cost weight is computed by some external function (e.g., a torch.nn.Module).

We show an example of CostWeight usage below with the ScaleCostWeight class.

print("Scale cost weight creation:")
# Create a scale cost weight from a float
w1 = th.ScaleCostWeight(10.0)
# The weight is wrapped into a default variable
print(f"  w1 (default variable): {w1.scale}")

# A theseus variable can be passed directly
w2 = th.ScaleCostWeight(th.Variable(2 * torch.ones(1, 1), name="scale"))
print(f"  w2 (named variable): {w2.scale}\n")

# Weighting errors and jacobians with a ScaleCostWeight
print("Weighting errors/jacobian directly with a ScaleCostWeight:")
weighted_jacobians, weighted_error = w1.weight_jacobians_and_error(jacobians, error)
print(f"  jacobians:\n     weighted: {weighted_jacobians}\n     original: {jacobians}")
print(f"  error:\n    weighted: {weighted_error}\n    original: {error}\n")

# If the ScaleCostWeight is included in the cost function, we can directly
# use the `weight_errors` and `weight_jacobians_and_error` of the cost function.
print("Using the `weighted_error` function of the previous cost function:") 
print(f"  weighted cost function error: {cf.weighted_error()} vs unweighted error: {cf.error()}")
Scale cost weight creation:
  w1 (default variable): Variable(tensor=tensor([[10.]]), name=Variable__17)
  w2 (named variable): Variable(tensor=tensor([[2.]]), name=scale)

Weighting errors/jacobian directly with a ScaleCostWeight:
  jacobians:
     weighted: [tensor([[[10.,  0.],
         [ 0., 10.]]])]
     original: [tensor([[[1., 0.],
         [0., 1.]]])]
  error:
    weighted: tensor([[20., 20.]])
    original: tensor([[2., 2.]])

Using the `weighted_error` function of the previous cost function:
  weighted cost function error: tensor([[4., 4.]]) vs unweighted error: tensor([[2., 2.]])
4. Objective
A th.Objective defines the structure of an optimization problem, by adding one or more cost functions to it, each with associated cost weights and variables. The th.Objective will combine them into a global error function, with an internal structure that can be used by a Theseus optimizer to minimize the global error via changes in the optimization variables.

Currently, th.Objective supports nonlinear sum of squares objectives, where the global error is the sum of the squares of each of its cost function errors, weighted by their corresponding cost weights. We plan to extend to other optimization structures in the future. A critical point in the creation of the objective is that Theseus assumes that cost weights provided will also be squared in the final the objective. Formally, we currently support objectives of the form

Theseus Objective

where v represents the set of variables, fi is a cost function error, and wi its associated cost weight.

Below we show a simple example of creating an objective. We will want to minimize the following function (x - a)2 + 4(y - b)2, where a and b as constants, x and y as variables. Below, we first create (1) the optimization and auxiliary variables, (2) cost weights, (3) cost functions, (4) objective.

Then, to evaluate the Objective, we will use its error_metric function, which evaluates the squared norm of the error vector, divided by 2. Before we can evaluate it, however, we must use the Objective.update function at least once (so that the internal data structures are correctly set up). In general, the update function is used to easily change the values of all variables registered with the Objective. This function receives a dictionary that maps variable names to torch tensors to which the corresponding variables should be updated.

We finally show that the current objective is computed correctly for this function. (In the next section, we optimize the objective to its minimum value)

# Step 1: Construct optimization and auxiliary variables.
# Construct variables of the function: these the optimization variables of the cost functions. 
x = th.Vector(1, name="x")
y = th.Vector(1, name="y")

# Construct auxiliary variables for the constants of the function.
a = th.Vector(tensor=torch.randn(1,1), name="a")
b = th.Vector(tensor=torch.randn(1,1), name="b")

# Step 2: Construct cost weights
# For w1, let's use a named variable
w1 = th.ScaleCostWeight(th.Variable(tensor=torch.ones(1, 1), name="w1_sqrt"))
w2 = th.ScaleCostWeight(2.0)  # we provide 2, as sqrt of 4 for the (y-b)^2 term

# Step 3: Construct cost functions representing each error term
# First term
cf1 = th.Difference(x, a, w1, name="term_1")
# Second term
cf2 = th.Difference(y, b, w2, name="term_2")

# Step 4: Create the objective function and add the error terms
objective = th.Objective()
objective.add(cf1)
objective.add(cf2)

# Step 5: Evaluate objective under current values
# Note this needs to be preceded by a call to `objective.update`
# Here we use the update function to set values of all variables
objective.update({"a": torch.ones(1,1), "b": 2 * torch.ones(1, 1), 
                  "x": 0.5 * torch.ones(1,1), "y": 3 * torch.ones(1, 1)})
# Weighted error should be: cost_weight * weighted_error 
print(f"Error term 1: unweighted: {cf1.error()} weighted: {cf1.weighted_error()}")
print(f"Error term 2: unweighted: {cf2.error()} weighted: {cf2.weighted_error()}")
# Objective value should be: (error1)^2 + (error2)^2 
print(f"Objective value: {objective.error_metric()}")
Error term 1: unweighted: tensor([[-0.5000]]) weighted: tensor([[-0.5000]])
Error term 2: unweighted: tensor([[1.]]) weighted: tensor([[2.]])
Objective value: tensor([4.2500])
Adding cost functions to the objective registers all of its optimization and auxiliary variables (and those of its cost weights, if present). th.Objective also checks that names are not overloaded by different variable or cost function objects

try:
    objective.add(th.Difference(y, b, w2, name="term_1"))
except ValueError as e:
    print(e)
    
try:
    obj2 = th.Objective()
    obj2.add(th.Difference(x, a, w1, name="term_1"))
    fake_x1 = th.Vector(1, name="x")
    obj2.add(th.Difference(fake_x1, b, w2, name="fake_term"))
except ValueError as e:
    print(e)
Two different cost function objects with the same name (term_1) are not allowed in the same objective.
Two different variable objects with the same name (x) are not allowed in the same objective.
5. Optimizers
Theseus provides a set of linear and nonlinear optimizers for minimizing problems described as th.Objective. The objective can be solved by calling optimizer.optimize(), which will change the values of optimization variables to minimize its associated objective. optimize leaves the optimization variables at the final values found, and returns an info object about the optimization (which contains the best solution and optimization statistics).

# Recall that our objective is (x - a)^2 + 4 (y - b)^2
# which is minimized at x = a and y = b
# Let's start by assigning random values to them
objective.update({
    "x": torch.randn(1, 1),
    "y": torch.randn(1, 1)
})

# Now let's use the optimizer. Because this problem is minimizing a
# quadratic form, a linear optimizer can solve for the optimal solution
optimizer = th.LinearOptimizer(objective, th.CholeskyDenseSolver)
info = optimizer.optimize()

# Now let's check the values of x and y 
# Here we print only the Vectors' tensor attributes for ease of understanding
print(f"x: {x.tensor} vs a: {a.tensor}")  # Matches a = 1
print(f"y: {y.tensor} vs b: {b.tensor}")  # Matches b = 2
print(f"Objective after optimization: {objective.error_metric()}")
x: tensor([[1.]]) vs a: tensor([[1.]])
y: tensor([[2.]]) vs b: tensor([[2.]])
Objective after optimization: tensor([0.])
/private/home/lep/code/theseus/theseus/optimizer/optimizer.py:42: UserWarning: Vectorization is off by default when not running from TheseusLayer. Using TheseusLayer is the recommended way to run our optimizers.
  warnings.warn(
6. TheseusLayer
As the warning above indicates, the recommended way to run our optimizers is via TheseusLayer. The TheseusLayer provides an interface between torch code upstream/downstream, and Theseus objectives and optimizers. The forward() method combines the functionality of Objective.update() and Optimizer.optimizer() into a single call. It receives an update dictionary as input, and returns a dictionary with the torch data of optimization variables after optimization, as well as the optimizer's output info.

layer = th.TheseusLayer(optimizer)
values, info = layer.forward({
    "x": torch.randn(1, 1),
    "y": torch.randn(1, 1),
    "a": torch.ones(1, 1),
    "b": 2 * torch.ones(1, 1),
    "w1_sqrt": torch.ones(1, 1)
})
print(f"After calling TheseusLayer's forward():")
print(f"  Values: {values}")
print(f"  Info: {info}")
print(f"  Optimized objective: {objective.error_metric()}")
After calling TheseusLayer's forward():
  Values: {'x': tensor([[1.]]), 'y': tensor([[2.]])}
  Info: OptimizerInfo(best_solution={'x': tensor([[1.]]), 'y': tensor([[2.]])}, status=array([<LinearOptimizerStatus.CONVERGED: 1>], dtype=object))
  Optimized objective: tensor([0.])
The TheseusLayer allows for backpropagation, and is semantically similar to a layer in a PyTorch neural network. Backpropagating through the TheseusLayer allows for learning of any necessary quantities of the problem, such as cost weights, initial values for the optimization variables, and other parameters for the optimization. The following tutorials will illustrate several applications for learning with a TheseusLayer.

To distinguish between the optimization done by the Theseus optimizers, and those done outside the Theseus optimizers (e.g., by PyTorch's autograd during learning), we will refer to them as inner loop optimization and outer loop optimization, respectively. Note that the inner loop optimization optimizes only the optimization variables, and the outer loop optimization can optimize torch tensors associated with selected variables provided to the PyTorch autograd optimizers. A call to TheseusLayer forward() performs only inner loop optimization; typically the PyTorch autograd learning steps will perform the outer loop optimizations. We will see examples of this in the following tutorials.

During the outer loop, we will commonly want to update Theseus variables before running inner loop optimization; for example, to set initial values for optimization variables, or to update auxiliary variables with tensors learned by the outer loop. We recommend that such updates to Theseus variables are done via TheseusLayer.forward(). While variables and objectives can be updated independently without going through TheseusLayer.forward(), following this convention makes it explicitly what the latest inputs to the TheseusLayer are, helping to avoid hidden errors and unwanted behavior. Therefore, we recommend that any updates during learning be performed only via the TheseusLayer.

Creating Custom Cost Functions
In this tutorial, we show how to create a custom cost function that might be needed for an application. While we can always use the AutoDiffCostFunction by simply writing an error function, it is often more efficient for compute-intensive applications to derive a new CostFunction subclass and use closed-form Jacobians.

We will show how to write a custom VectorDifference cost function in this tutorial. This cost function provides the difference between two Vectors as the error.

Note: VectorDifference is a simplified version of the Difference cost function already provided in the Theseus library, and shown in Tutorial 0. Difference can be used on any LieGroup, while VectorDifference can only be used on Vectors.

Initialization
Any CostFunction subclass should be initialized with a CostWeight and all arguments needed to compute the cost function. In this example, we set up __init__ function for VectorDifference to require as input the two Vectors whose difference we wish to compute: the Vector to be optimized, var, and the Vector that is the reference for comparison, target.

In addition, the __init__ function also needs to register the optimization variables and all the auxiliary variables. In this example, optimization variable var is registered with register_optim_vars. The other input necessary to evaluate the cost, target is registered with register_aux_vars. This is required for the nonlinear optimizers to work correctly: these functions register the optimization and auxiliary variables into internal lists, and then are easily used by the relevant Objective to add them, ensure no name collisions, and to update them with new values.

The CostWeight is used to weight the errors and jacobians, and is required by every CostFunction sub-class (the error and jacobian weighting functions are inherited from the parent CostFunction class.)

from typing import List, Optional, Tuple
import theseus as th

class VectorDifference(th.CostFunction):
    def __init__(
        self,
        cost_weight: th.CostWeight,
        var: th.Vector,
        target: th.Vector,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name) 

        # add checks to ensure the input arguments are of the same class and dof:
        if not isinstance(var, target.__class__):
            raise ValueError(
                "Variable for the VectorDifference inconsistent with the given target."
            )
        if not var.dof() == target.dof():
            raise ValueError(
                "Variable and target in the VectorDifference must have identical dof."
            )

        self.var = var
        self.target = target

        # register variable and target
        self.register_optim_vars(["var"])
        self.register_aux_vars(["target"])
Implement abstract functions
Next, we need to implement the abstract functions of CostFunction: dim, error, jacobians, and _copy_impl:

dim: returns the degrees of freedom (dof) of the error; in this case, this is the dof of the optimization variable var
error: returns the difference of Vectors i.e. var - target
jacobian: returns the Jacobian of the error with respect to the var
_copy_impl: creates a deep copy of the internal class members
We illustrate these below (including once again the __init__ function from above, so the class is fully defined.)

import torch 

class VectorDifference(th.CostFunction):
    def __init__(
        self,
        cost_weight: th.CostWeight,
        var: th.Vector,
        target: th.Vector,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name) 
        self.var = var
        self.target = target
        # to improve readability, we have skipped the data checks from code block above
        self.register_optim_vars(["var"])
        self.register_aux_vars(["target"])

    def error(self) -> torch.Tensor:
        return (self.var - self.target).tensor

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return [
            # jacobian of error function wrt var is identity matrix I
            torch.eye(self.dim(), dtype=self.var.dtype)  
            # repeat jacobian across each element in the batch
            .repeat(self.var.shape[0], 1, 1)  
            # send to variable device
            .to(self.var.device)  
        ], self.error()

    def dim(self) -> int:
        return self.var.dof()

    def _copy_impl(self, new_name: Optional[str] = None) -> "VectorDifference":
        return VectorDifference(  # type: ignore
            self.var.copy(), self.weight.copy(), self.target.copy(), name=new_name
        )
Usage
We show now that the VectorDifference cost function works as expected.

For this, we create a set of VectorDifference cost functions each over a pair of Vectors a_i and b_i, and add them to an Objective. We then create the data for each Vector a_i and b_i of the VectorDifference cost functions, and update the Objective with it. The code snippet below shows that the Objective error is correctly computed.

We use a ScaleCostWeight as the input CostWeight here: this is a scalar real-valued CostWeight used to weight the CostFunction; for simplicity we use a fixed value of 1. in this example.

cost_weight = th.ScaleCostWeight(1.0)

# construct cost functions and add to objective
objective = th.Objective()
num_test_fns = 10
for i in range(num_test_fns):
    a = th.Vector(2, name=f"a_{i}")
    b = th.Vector(2, name=f"b_{i}")
    cost_fn = VectorDifference(cost_weight, a, b)
    objective.add(cost_fn)
    
# create data for adding to the objective
theseus_inputs = {}
for i in range(num_test_fns):
    # each pair of var/target has a difference of [1, 1]
    theseus_inputs.update({f"a_{i}": torch.ones((1,2)), f"b_{i}": 2 * torch.ones((1,2))})

objective.update(theseus_inputs)
# sum of squares of errors [1, 1] for 10 cost fns: the result should be 20
error_sq = objective.error_metric()
print(f"Sample error squared norm: {error_sq.item()}")




another example 

Creating Custom Cost Functions
In this tutorial, we show how to create a custom cost function that might be needed for an application. While we can always use the AutoDiffCostFunction by simply writing an error function, it is often more efficient for compute-intensive applications to derive a new CostFunction subclass and use closed-form Jacobians.

We will show how to write a custom VectorDifference cost function in this tutorial. This cost function provides the difference between two Vectors as the error.

Note: VectorDifference is a simplified version of the Difference cost function already provided in the Theseus library, and shown in Tutorial 0. Difference can be used on any LieGroup, while VectorDifference can only be used on Vectors.

Initialization
Any CostFunction subclass should be initialized with a CostWeight and all arguments needed to compute the cost function. In this example, we set up __init__ function for VectorDifference to require as input the two Vectors whose difference we wish to compute: the Vector to be optimized, var, and the Vector that is the reference for comparison, target.

In addition, the __init__ function also needs to register the optimization variables and all the auxiliary variables. In this example, optimization variable var is registered with register_optim_vars. The other input necessary to evaluate the cost, target is registered with register_aux_vars. This is required for the nonlinear optimizers to work correctly: these functions register the optimization and auxiliary variables into internal lists, and then are easily used by the relevant Objective to add them, ensure no name collisions, and to update them with new values.

The CostWeight is used to weight the errors and jacobians, and is required by every CostFunction sub-class (the error and jacobian weighting functions are inherited from the parent CostFunction class.)

from typing import List, Optional, Tuple
import theseus as th

class VectorDifference(th.CostFunction):
    def __init__(
        self,
        cost_weight: th.CostWeight,
        var: th.Vector,
        target: th.Vector,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name) 

        # add checks to ensure the input arguments are of the same class and dof:
        if not isinstance(var, target.__class__):
            raise ValueError(
                "Variable for the VectorDifference inconsistent with the given target."
            )
        if not var.dof() == target.dof():
            raise ValueError(
                "Variable and target in the VectorDifference must have identical dof."
            )

        self.var = var
        self.target = target

        # register variable and target
        self.register_optim_vars(["var"])
        self.register_aux_vars(["target"])
Implement abstract functions
Next, we need to implement the abstract functions of CostFunction: dim, error, jacobians, and _copy_impl:

dim: returns the degrees of freedom (dof) of the error; in this case, this is the dof of the optimization variable var
error: returns the difference of Vectors i.e. var - target
jacobian: returns the Jacobian of the error with respect to the var
_copy_impl: creates a deep copy of the internal class members
We illustrate these below (including once again the __init__ function from above, so the class is fully defined.)

import torch 

class VectorDifference(th.CostFunction):
    def __init__(
        self,
        cost_weight: th.CostWeight,
        var: th.Vector,
        target: th.Vector,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name) 
        self.var = var
        self.target = target
        # to improve readability, we have skipped the data checks from code block above
        self.register_optim_vars(["var"])
        self.register_aux_vars(["target"])

    def error(self) -> torch.Tensor:
        return (self.var - self.target).tensor

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return [
            # jacobian of error function wrt var is identity matrix I
            torch.eye(self.dim(), dtype=self.var.dtype)  
            # repeat jacobian across each element in the batch
            .repeat(self.var.shape[0], 1, 1)  
            # send to variable device
            .to(self.var.device)  
        ], self.error()

    def dim(self) -> int:
        return self.var.dof()

    def _copy_impl(self, new_name: Optional[str] = None) -> "VectorDifference":
        return VectorDifference(  # type: ignore
            self.var.copy(), self.weight.copy(), self.target.copy(), name=new_name
        )
Usage
We show now that the VectorDifference cost function works as expected.

For this, we create a set of VectorDifference cost functions each over a pair of Vectors a_i and b_i, and add them to an Objective. We then create the data for each Vector a_i and b_i of the VectorDifference cost functions, and update the Objective with it. The code snippet below shows that the Objective error is correctly computed.

We use a ScaleCostWeight as the input CostWeight here: this is a scalar real-valued CostWeight used to weight the CostFunction; for simplicity we use a fixed value of 1. in this example.

cost_weight = th.ScaleCostWeight(1.0)

# construct cost functions and add to objective
objective = th.Objective()
num_test_fns = 10
for i in range(num_test_fns):
    a = th.Vector(2, name=f"a_{i}")
    b = th.Vector(2, name=f"b_{i}")
    cost_fn = VectorDifference(cost_weight, a, b)
    objective.add(cost_fn)
    
# create data for adding to the objective
theseus_inputs = {}
for i in range(num_test_fns):
    # each pair of var/target has a difference of [1, 1]
    theseus_inputs.update({f"a_{i}": torch.ones((1,2)), f"b_{i}": 2 * torch.ones((1,2))})

objective.update(theseus_inputs)
# sum of squares of errors [1, 1] for 10 cost fns: the result should be 20
error_sq = objective.error_metric()
print(f"Sample error squared norm: {error_sq.item()}")
Sample error squared norm: 20.0

and one more 
Motion Planning Part 1: motion planning as nonlinear least squares optimization
In this tutorial, we will learn how to implement the GPMP2 (Mukadam et al, 2018) motion planning algorithm, for a 2D robot in a planar environment.

The goal is to find the trajectory (pose and velocity) of the robot given a start and goal pose and some representation of the environment. This can be solved as an optimization problem where the variables to be optimized for are the 2D pose and 2D velocity of the robot along a trajectory of some total time steps (at some fixed time interval). In this example, we formulate the objective of the optimization with the following cost terms that are balanced by their respective weights:

Boundary condition: the trajectory should begin at the start pose with zero velocity and end at the goal pose with zero velocity.
Collision avoidance: the trajectory should avoid colliding with obstacles in the environment (we use a signed distance fields).
Smoothness: the trajectory should be smooth (we use a zero acceleration prior).
import random

import matplotlib as mpl
import numpy as np
import torch
import torch.utils.data

import theseus as th
import theseus.utils.examples as theg

%load_ext autoreload
%autoreload 2

torch.set_default_dtype(torch.double)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
seed = 0
torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["font.size"] = 16
1. Loading and visualizing the trajectory data
First, let's load some motion planning problems from a dataset of maps and trajectories generated using the code in dgpmp2.

dataset_dir = "data/motion_planning_2d"
dataset = theg.TrajectoryDataset(True, 2, dataset_dir, map_type="tarpit")
data_loader = torch.utils.data.DataLoader(dataset, 2)

batch = next(iter(data_loader))
The batch is a dictionary of strings to torch.Tensor containing the following keys:

for k, v in batch.items():
    if k != "file_id":
        print(f"{k:20s}: {v.shape}")
map_tensor          : torch.Size([2, 128, 128])
sdf_origin          : torch.Size([2, 2])
cell_size           : torch.Size([2, 1])
sdf_data            : torch.Size([2, 128, 128])
expert_trajectory   : torch.Size([2, 4, 101])
Let's plot the maps and trajectories loaded. th.eb.SignedDistanceField2D is a signed distance field object, which includes a function to convert x,y-coordinates to map cells that we use here for plotting. For completeness, we show the expert trajectories loaded, although we won't use them in this example (we will do so in Part 2 of this tutorial). We also illustrate the signed distance field for each map.

sdf = th.eb.SignedDistanceField2D(
    th.Point2(batch["sdf_origin"]),
    th.Variable(batch["cell_size"]),
    th.Variable(batch["sdf_data"]),
)
figs = theg.generate_trajectory_figs(
    batch["map_tensor"], 
    sdf, 
    [batch["expert_trajectory"]], 
    robot_radius=0.4, 
    labels=["expert trajectory"], 
    fig_idx_robot=0,
    figsize=(10, 4),
    plot_sdf=True,
)
figs[0].show()
figs[1].show()


The following are some constants that we will use throughout the example

trajectory_len = batch["expert_trajectory"].shape[2]
num_time_steps = trajectory_len - 1
map_size = batch["map_tensor"].shape[1]
safety_distance = 0.4
robot_radius = 0.4
total_time = 10.0
dt_val = total_time / num_time_steps
Qc_inv = [[1.0, 0.0], [0.0, 1.0]]
collision_w = 20.0
boundary_w = 100.0
2. Modeling the problem
2.1. Defining Variable objects
Our goal in this example will be to use Theseus to produce plans for the maps loaded above. As mentioned in the introduction, we need a 2D pose and a 2D velocity for each point along the trajectory to be optimized. For this, we will create a set of th.Point2 variables with individual names, and store them in two lists so that they can be later passed to the appropriate cost functions.

# Create optimization variables
poses = []
velocities = []
for i in range(trajectory_len):
    poses.append(th.Point2(name=f"pose_{i}", dtype=torch.double))
    velocities.append(th.Point2(name=f"vel_{i}", dtype=torch.double))
In addition to the optimization variables, we will also need a set of auxiliary variables to wrap map-dependent quantities involved in cost function computation, but that are constant throughout the optimization. This includes start/goal target values, as well as parameters for collision and dynamics cost functions.

# Targets for pose boundary cost functions
start_point = th.Point2(name="start")
goal_point = th.Point2(name="goal")

# For collision avoidance cost function
sdf_origin = th.Point2(name="sdf_origin")
cell_size = th.Variable(torch.empty(1, 1), name="cell_size")
sdf_data = th.Variable(torch.empty(1, map_size, map_size), name="sdf_data")
cost_eps = th.Variable(torch.tensor(robot_radius + safety_distance).view(1, 1), name="cost_eps")

# For GP dynamics cost function
dt = th.Variable(torch.tensor(dt_val).view(1, 1), name="dt")
2.2. Cost weights
Next we will create a series of cost weights to use for each of the cost functions involved in our optimization problem.

# Cost weight to use for all GP-dynamics cost functions
gp_cost_weight = th.eb.GPCostWeight(torch.tensor(Qc_inv), dt)

# Cost weight to use for all collision-avoidance cost functions
collision_cost_weight = th.ScaleCostWeight(th.Variable(torch.tensor(collision_w)))

# For all hard-constraints (end points pos/vel) we use a single scalar weight
# with high value
boundary_cost_weight = th.ScaleCostWeight(boundary_w)
2.3. Cost functions
In this section, we will now create a Theseus objective and add the GPMP2 cost functions for motion planning. First, we create the objective:

objective = th.Objective(dtype=torch.double)
Boundary cost functions
Here we create cost functions for the boundary conditions, assign names to them, and add them to the Objective. For boundaries, we need four cost functions, and for each we use a cost function of type th.Difference. This cost function type takes as input an optimization variable, a cost weight, a target auxiliary variable, and a name. Its error function is the local difference between the optimization variable and the target.

For example, consider the first Difference added below (with name pose_0). This cost function will tell the optimizer to try to bring the value of the variable at poses[0] close to that of auxiliary variable start_point (which is also a named variable, as explained in Section 2.1). On the other hand, for velocity constraints (for vel_0), we don't need to pass a named auxiliary variable for the target, since we know that we will want it to be a torch.zeros(1, 2), no matter what the map data is (the robot start with zero velocity).

Finally, all of these cost functions share the same boundary_cost_weight, which as you may recall, is a ScaleCostWeight(100.0).

# Fixed starting position
objective.add(
    th.Difference(poses[0], start_point, boundary_cost_weight, name="pose_0")
)
# Fixed initial velocity
objective.add(
    th.Difference(
        velocities[0],
        th.Point2(tensor=torch.zeros(1, 2)),
        boundary_cost_weight,
        name="vel_0",
    )
)
objective.add(
    th.Difference(
        poses[-1], goal_point, boundary_cost_weight, name="pose_N"
    )
)
objective.add(
    th.Difference(
        velocities[-1],
        th.Point2(tensor=torch.zeros(1, 2)),
        boundary_cost_weight,
        name="vel_N",
    )
)
Collision cost functions
For collision avoidance, we use a th.eb.Collision2D cost function type, which receives the following inputs:

A single th.Point2 optimization variable.
Auxiliary variables:
Three representing signed distance field data (sdf_origin, sdf_data, cell_size).
The distance within which collision cost is incurred (cost_eps).
A cost weight.
Since we need one such cost function for each internal point in the trajectory, we create the cost functions in a loop and pass the corresponding pose variable defined above.

for i in range(1, trajectory_len - 1):
    objective.add(
        th.eb.Collision2D(
            poses[i],
            sdf_origin,
            sdf_data,
            cell_size,
            cost_eps,
            collision_cost_weight,
            name=f"collision_{i}",
        )
    )
GP-dynamics cost functions
For ensuring smooth trajectories, we use a th.eb.GPMotionModel cost function, which receives the following inputs:

Four th.Point2 optimization variables: pose at time t-1, velocity at time t-1, pose at time t, velocity at time t.
One auxiliary variable describing the time differential between consecutive time steps.
A cost weight (typically of type th.eb.GPCostWeight).
We need one such cost function for each pair of consecutive states (pose and velocity), so we add these in a loop and pass the appropriate optimization variables from the lists created above.

for i in range(1, trajectory_len):
    objective.add(
        (
            th.eb.GPMotionModel(
                poses[i - 1],
                velocities[i - 1],
                poses[i],
                velocities[i],
                dt,
                gp_cost_weight,
                name=f"gp_{i}",
            )
        )
    )
Creating the TheseusLayer for motion planning
For this example, we will use Levenberg-Marquardt as the non-linear optimizer, coupled with a dense linear solver based on Cholesky decomposition.

optimizer = th.LevenbergMarquardt(
    objective,
    th.CholeskyDenseSolver,
    max_iterations=50,
    step_size=1.0,
)
motion_planner = th.TheseusLayer(optimizer)
motion_planner.to(device=device, dtype=torch.double)
3. Running the optimizer
Finally, we are ready to generate some optimal plans. We first initialize all auxiliary variables whose values are map dependent (e.g., start and goal positions, or SDF data). We also provide some sensible initial values for the optimization variables; in this example, we will initialize the optimizaton variables to be on a straight line from start to goal. The following helper function will be useful for this:

def get_straight_line_inputs(start, goal):
    # Returns a dictionary with pose and velocity variable names associated to a 
    # straight line trajectory between start and goal
    start_goal_dist = goal - start
    avg_vel = start_goal_dist / total_time
    unit_trajectory_len = start_goal_dist / (trajectory_len - 1)
    input_dict = {}
    for i in range(trajectory_len):
        input_dict[f"pose_{i}"] = start + unit_trajectory_len * i
        if i == 0 or i == trajectory_len - 1:
            input_dict[f"vel_{i}"] = torch.zeros_like(avg_vel)
        else:
            input_dict[f"vel_{i}"] = avg_vel
    return input_dict
Now, let's pass the motion planning data to our TheseusLayer and start create some trajectories; note that we can solve for both trajectories simultaneously by taking advantage of Theseus' batch support. For initializing variables, we create a dictionary mapping strings to torch.Tensor, where the keys are th.Variable names, and the values are the tensors that should be used for their initial values.

start = batch["expert_trajectory"][:, :2, 0].to(device)
goal = batch["expert_trajectory"][:, :2, -1].to(device)
planner_inputs = {
    "sdf_origin": batch["sdf_origin"].to(device),
    "start": start.to(device),
    "goal": goal.to(device),
    "cell_size": batch["cell_size"].to(device),
    "sdf_data": batch["sdf_data"].to(device),
}
planner_inputs.update(get_straight_line_inputs(start, goal))    
with torch.no_grad():        
    final_values, info = motion_planner.forward(
        planner_inputs,
        optimizer_kwargs={
            "track_best_solution": True,
            "verbose": True,
            "damping": 0.1,
        }
    )
Nonlinear optimizer. Iteration: 0. Error: 3905.6714736579306
Nonlinear optimizer. Iteration: 1. Error: 2282.251394473349
Nonlinear optimizer. Iteration: 2. Error: 136.22542880573795
Nonlinear optimizer. Iteration: 3. Error: 50.78312661084182
Nonlinear optimizer. Iteration: 4. Error: 4.300118887813326
Nonlinear optimizer. Iteration: 5. Error: 14.84557244071468
Nonlinear optimizer. Iteration: 6. Error: 2.3698518778509152
Nonlinear optimizer. Iteration: 7. Error: 2.1764857434368388
Nonlinear optimizer. Iteration: 8. Error: 163.46637040887495
Nonlinear optimizer. Iteration: 9. Error: 2.213071348805293
Nonlinear optimizer. Iteration: 10. Error: 181.50298247241238
Nonlinear optimizer. Iteration: 11. Error: 2.2187748890968124
Nonlinear optimizer. Iteration: 12. Error: 169.89215894334885
Nonlinear optimizer. Iteration: 13. Error: 3.805443232691481
Nonlinear optimizer. Iteration: 14. Error: 132.1010533389907
Nonlinear optimizer. Iteration: 15. Error: 1.830013377706033
Nonlinear optimizer. Iteration: 16. Error: 137.40931841459334
Nonlinear optimizer. Iteration: 17. Error: 2.3630850988848158
Nonlinear optimizer. Iteration: 18. Error: 127.7326186066636
Nonlinear optimizer. Iteration: 19. Error: 1.6962584333307351
Nonlinear optimizer. Iteration: 20. Error: 133.13349342246815
Nonlinear optimizer. Iteration: 21. Error: 2.3779312169651745
Nonlinear optimizer. Iteration: 22. Error: 100.65554701995993
Nonlinear optimizer. Iteration: 23. Error: 1.5484209548141115
Nonlinear optimizer. Iteration: 24. Error: 79.11945618450362
Nonlinear optimizer. Iteration: 25. Error: 1.5620953068549461
Nonlinear optimizer. Iteration: 26. Error: 60.766726980963455
Nonlinear optimizer. Iteration: 27. Error: 1.4306538819862062
Nonlinear optimizer. Iteration: 28. Error: 1.3989819422747745
Nonlinear optimizer. Iteration: 29. Error: 1.1621472075446577
Nonlinear optimizer. Iteration: 30. Error: 1.1368880251022904
Nonlinear optimizer. Iteration: 31. Error: 1.1085143552454548
Nonlinear optimizer. Iteration: 32. Error: 1.0931716928138968
Nonlinear optimizer. Iteration: 33. Error: 1.0802935750315232
Nonlinear optimizer. Iteration: 34. Error: 1.0712316967724411
Nonlinear optimizer. Iteration: 35. Error: 1.0644195614703251
Nonlinear optimizer. Iteration: 36. Error: 1.0603135136138264
Nonlinear optimizer. Iteration: 37. Error: 1.0569988398213055
Nonlinear optimizer. Iteration: 38. Error: 1.0549630384559077
Nonlinear optimizer. Iteration: 39. Error: 1.0530582611647743
Nonlinear optimizer. Iteration: 40. Error: 1.0515998330960024
Nonlinear optimizer. Iteration: 41. Error: 1.050424648528516
Nonlinear optimizer. Iteration: 42. Error: 1.049765287190003
Nonlinear optimizer. Iteration: 43. Error: 1.0490200351376027
Nonlinear optimizer. Iteration: 44. Error: 1.048551093917665
Nonlinear optimizer. Iteration: 45. Error: 1.048095388794331
Nonlinear optimizer. Iteration: 46. Error: 1.04779239077781
Nonlinear optimizer. Iteration: 47. Error: 1.0475756086931332
Nonlinear optimizer. Iteration: 48. Error: 1.0474201617339547
Nonlinear optimizer. Iteration: 49. Error: 1.0473066061954055
Nonlinear optimizer. Iteration: 50. Error: 1.0472228497233835
4. Results
After the optimization is completed, we can query the optimization variables to obtain the final trajectory and visualize the result. The following function creates a trajectory tensor from the output dictionary of TheseusLayer.

def get_trajectory(values_dict):
    trajectory = torch.empty(values_dict[f"pose_0"].shape[0], 4, trajectory_len, device=device)
    for i in range(trajectory_len):
        trajectory[:, :2, i] = values_dict[f"pose_{i}"]
        trajectory[:, 2:, i] = values_dict[f"vel_{i}"]
    return trajectory
Let's now plot the final trajectories

trajectory = get_trajectory(info.best_solution).cpu()

sdf = th.eb.SignedDistanceField2D(
    th.Point2(batch["sdf_origin"]),
    th.Variable(batch["cell_size"]),
    th.Variable(batch["sdf_data"]),
)
figs = theg.generate_trajectory_figs(
    batch["map_tensor"], 
    sdf, 
    [trajectory], 
    robot_radius=robot_radius, 
    labels=["solution trajectory"], 
    fig_idx_robot=0,
    figsize=(6, 6)
)
figs[0].show()
figs[1].show()

