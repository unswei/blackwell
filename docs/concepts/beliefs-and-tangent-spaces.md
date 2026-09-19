# Beliefs and tangent spaces

## Euclidean states

For a vector state $x\in\mathbb{R}^n$, adding an error $\delta$ is ordinary
addition:

$$
x \boxplus \delta = x + \delta,
\qquad
y \boxminus x = y - x.
$$

The mean and covariance therefore use the same coordinates.

## Why poses are different

An SE(2) pose combines a planar translation and a periodic heading:

$$
X = [x, y, \theta].
$$

Adding pose coordinates component-wise causes trouble around $\pm\pi$ and does
not respect body-frame motion. Blackwell represents the nominal pose as
`[x, y, heading]`, but uncertainty as a local tangent vector
`[forward, lateral, turn]`.

The right retraction is

$$
X \boxplus \delta = X\,\operatorname{Exp}(\delta),
$$

and the difference is

$$
Y \boxminus X = \operatorname{Log}(X^{-1}Y).
$$

This convention means an SE(2) covariance is body-frame and attached to its
belief mean. When the mean moves, Blackwell transports the covariance into the
new tangent coordinate system.

## Spatial poses

SE(3) uses the same right/body retraction with state encoding
`[x, y, z, qx, qy, qz, qw]`. The Hamilton quaternion is scalar-last (`xyzw`).
The tangent is `[rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]`, with body
translation first and rotation in radians. A pose Gaussian therefore has a
`(7,)` mean but a `(6, 6)` covariance. Quaternion signs are equivalent;
`local_coordinates` accounts for this and selects a principal rotation.

See [SE(3) conventions](../reference/spaces/se3.md) and the
[correlated uncertainty example](../examples/se3-uncertainty.md). Separate
belief containers store marginal covariances only; correlations between poses
or between a pose and point must be carried explicitly by the application.

## Belief containers

`GaussianBelief` contains a mean state and its local covariance. It does not
store a state-space object, keeping the belief itself a simple JAX PyTree.

`ParticleBelief` contains states and normalised weights. Particles live on the
state space directly, so no Gaussian covariance assumption is required.

| Container | Leading shape | Invariant |
| --- | --- | --- |
| `GaussianBelief.mean` | state shape | Valid state representation |
| `GaussianBelief.covariance` | `(tangent_dim, tangent_dim)` | Symmetric local covariance |
| `ParticleBelief.particles` | `(particle_count, *state_shape)` | Each row is a state |
| `ParticleBelief.weights` | `(particle_count,)` | Non-negative and sums to one |

!!! important "Reference and target order"

    `local_coordinates(reference, state)` returns the tangent displacement that
    retracts from `reference` towards `state`.

See the [SE(2) state-space reference](../reference/spaces/se2.md) for every group
operation.
