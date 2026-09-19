# SE(3) uncertainty propagation

Compose uncertain transforms, invert them, and transform uncertain 3D points:

```console
uv run python examples/se3_uncertainty.py
```

This example needs Blackwell 0.0.2 or its source checkout. It uses only public
operations and prints a reproducible Monte Carlo comparison without plotting
dependencies. Read the [encoding and frame conventions](../reference/spaces/se3.md)
before adapting it to a sensor or calibration problem.

## Linearise in tangent coordinates

Write an uncertain transform as $T=\bar T\operatorname{Exp}(\delta)$, with
$\delta=[\rho,\phi]$ and a $6\times6$ covariance. Its seven stored coordinates
are not the covariance axes. For a transform-valued function $f$, differentiate
the local output error at $\bar Y=f(\bar T)$:

```python
jacobian = jax.jacfwd(
    lambda delta: se3.local_coordinates(
        output_mean, function(se3.retract(input_mean, delta))
    )
)(jnp.zeros(6, dtype=input_mean.dtype))
```

`jax.jacrev` also works away from the principal-logarithm cut. Differentiating
the raw quaternion components produces a different, redundant coordinate
Jacobian and cannot directly propagate a six-dimensional pose covariance.

## Composition and inversion

Let $C=AB$. For right perturbations about $\bar A,\bar B,\bar C$,

$$
J_A=\operatorname{Ad}_{\bar B^{-1}},\qquad J_B=I_6.
$$

With $P_{AB}=\operatorname{Cov}(\delta_A,\delta_B)$, the first-order result is

$$
P_C=J_A P_A J_A^T+P_B+J_A P_{AB}+P_{AB}^T J_A^T.
$$

The cross-covariance is between the respective **body tangent coordinates**
of A and B; shared map, calibration or odometry errors can make it non-zero.
Two `GaussianBelief` objects store only marginals, so they cannot encode or
reconstruct this cross-covariance. Supply it from the joint model. Zero is an
explicit independence assumption.

For inversion $Y=A^{-1}$, the right-tangent Jacobian is
$J_A=-\operatorname{Ad}_{\bar A}$ and $P_Y=J_A P_A J_A^T$. If Y is used
together with A, retain their cross-covariance $P_{AY}=P_AJ_A^T$. For example,
$AA^{-1}$ has zero uncertainty; treating those inputs as independent loses
this cancellation. A regression test checks this singular joint distribution.

The script constructs the Jacobians using `retract`, `local_coordinates` and
`jax.jacfwd`; no operation-specific Jacobian helper is needed.

## Transforming correlated uncertain points

For $y=Tp=Rp+t$, let the point perturbation be additive in T's input frame.
At the nominal pose and point,

$$
J_T=\begin{bmatrix}R&-R[p]_\times\end{bmatrix},\qquad J_p=R.
$$

Let $P_{Tp}=\operatorname{Cov}(\delta_T,\delta_p)$ have shape $(6,3)$.
Propagate the full joint covariance:

$$
P_y=
\begin{bmatrix}J_T&J_p\end{bmatrix}
\begin{bmatrix}P_T&P_{Tp}\\P_{Tp}^T&P_p\end{bmatrix}
\begin{bmatrix}J_T&J_p\end{bmatrix}^T.
$$

Equivalently, this is $J_TP_TJ_T^T+J_pP_pJ_p^T$ plus the two cross terms
$J_TP_{Tp}J_p^T+J_pP_{Tp}^TJ_T^T$. The joint matrix must be symmetric
positive semidefinite; arbitrary cross blocks need not define a valid model.
The example builds it from a shared latent factor, so this condition holds.

The Monte Carlo calculation draws **paired errors from that same joint
distribution**, retracts each pose error, adds each point error and transforms
each paired sample. Sampling the marginals independently would silently remove
the correlation being tested.

The script compares the first-order covariance with 50,000 transformed samples
and separately reports the error caused by discarding the cross-covariance.
This tests the small-noise approximation. For large errors or rotations near
the logarithm's cut, use the particle representation to inspect the resulting
distribution. The nominal transformed point need not equal its nonlinear
expectation; a small second-order mean shift is expected.

## Belief contracts and filters

- A pose `GaussianBelief` has a `(7,)` mean and `(6, 6)` local covariance.
- A pose `ParticleBelief` has `(N, 7)` states and `(N,)` weights. Draw six
  tangent errors and retract them; do not add noise to quaternion components.
- A transformed point is Euclidean: its Gaussian mean/covariance are `(3,)`
  and `(3, 3)`, and its particle states are `(N, 3)`.
- Neither container retains cross-covariance with another belief. Keep joint
  uncertainty in application data and pass it explicitly when propagating.

The existing `ExtendedKalmanFilter` and `BootstrapParticleFilter` accept
`se3` as their state-space module when supplied with compatible dynamics and
observation operations. Integration tests exercise prediction, correction,
particle initialisation/resampling, `jit`, `vmap`, `scan` and EKF gradients.
This release adds geometry; application-specific 3D sensors and IMU models
remain custom models.

## Complete example

```python
--8<-- "examples/se3_uncertainty.py"
```

The geometry and tangent-linearisation background is described in
[Solà, Deray and Atchuthan, A micro Lie theory for state estimation in robotics](https://arxiv.org/abs/1812.01537)
and [Eade, Lie Groups for 2D and 3D Transformations](https://ethaneade.com/lie.pdf).
The formulas above use Blackwell's right/body perturbation convention.
