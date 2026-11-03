"""Condensed linear means and derivatives against a recurrent reference."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from blackwell.models import linear


@pytest.fixture(params=[jnp.float32, jnp.float64], ids=["float32", "float64"])
def precision(request):
    original = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", request.param == jnp.float64)
    try:
        yield request.param, 5e-6 if request.param == jnp.float32 else 2e-12
    finally:
        jax.config.update("jax_enable_x64", original)


def scenario(dtype=jnp.float32):
    # Unequal dimensions and dense, non-symmetric A expose axis/order mistakes.
    model = linear.LinearDynamics(
        jnp.array([[0.8, 0.2, -0.1], [0.0, 0.7, 0.3], [0.1, -0.2, 0.9]], dtype),
        jnp.array([[0.2, -0.3], [0.5, 0.1], [-0.2, 0.4]], dtype),
        jnp.eye(3, dtype=dtype) * 0.1,
    )
    projection = jnp.array([[1.0, 0.0, -0.5], [0.3, 0.2, 0.7]], dtype)
    a = jnp.array([0.1, -0.2, 0.3], dtype)
    d = jnp.array([-0.4, 0.5], dtype)
    state = jnp.array([0.3, -0.7, 1.2], dtype)
    controls = jnp.array([[0.2, 0.8], [-0.4, 0.1], [0.7, -0.6], [0.5, 0.3]], dtype)
    return model, projection, a, d, state, controls


def scan_predict(model, state, controls, projection, a, d):
    def step(state, control):
        state = linear.propagate(state, control, model) + a
        return state, projection @ state + d

    return jax.lax.scan(step, state, controls)[1]


@pytest.mark.parametrize("horizon", [1, 4])
@pytest.mark.parametrize("selected", [False, True], ids=["full", "selected"])
@pytest.mark.parametrize("affine", [False, True], ids=["linear", "affine"])
def test_prediction_matches_scan_and_preserves_dtype(
    precision, horizon, selected, affine
):
    dtype, tol = precision
    model, projection, a, d, state, controls = scenario(dtype)
    controls = controls[:horizon]
    if not selected:
        projection = jnp.eye(3, dtype=dtype)
        d = jnp.array([-0.4, 0.5, 0.2], dtype)
    if not affine:
        a, d = jnp.zeros_like(a), jnp.zeros_like(d)
    kwargs = {}
    if selected:
        kwargs["output_matrix"] = projection
    if affine:
        kwargs.update(state_offset=a, output_offset=d)
    response = linear.prepare_prediction(model, horizon, **kwargs)
    compiled = jax.jit(linear.prepare_prediction, static_argnames="horizon")(
        model, horizon, **kwargs
    )
    p = projection.shape[0]
    assert response.state_response.shape == (horizon, p, 3)
    assert response.command_response.shape == (horizon, p, horizon, 2)
    assert response.offset.shape == (horizon, p)
    for eager, jit in zip(response, compiled, strict=True):
        assert eager.dtype == dtype
        assert_allclose(eager, jit, rtol=tol, atol=tol)
    expected = scan_predict(model, state, controls, projection, a, d)
    for predict in (linear.predict_sequence, jax.jit(linear.predict_sequence)):
        outputs = predict(response, state, controls)
        assert outputs.dtype == dtype
        assert_allclose(outputs, expected, rtol=tol, atol=tol)
        assert_allclose(
            outputs[0],
            projection @ (model.transition @ state + model.control @ controls[0] + a)
            + d,
            rtol=tol,
            atol=tol,
        )


def test_response_is_causal_and_matches_both_jacobians(precision):
    dtype, tol = precision
    model, projection, a, d, state, controls = scenario(dtype)
    response = linear.prepare_prediction(
        model, 4, output_matrix=projection, state_offset=a, output_offset=d
    )
    for differentiate in (jax.jacfwd, jax.jacrev):
        state_jac, command_jac = jax.jit(
            differentiate(linear.predict_sequence, argnums=(1, 2))
        )(response, state, controls)
        expected_state, expected_command = differentiate(scan_predict, argnums=(1, 2))(
            model, state, controls, projection, a, d
        )
        assert_allclose(state_jac, response.state_response, rtol=tol, atol=tol)
        assert_allclose(command_jac, response.command_response, rtol=tol, atol=tol)
        assert_allclose(state_jac, expected_state, rtol=tol, atol=tol)
        assert_allclose(command_jac, expected_command, rtol=tol, atol=tol)
    for t in range(4):
        assert_array_equal(response.command_response[t, :, t + 1 :, :], 0)
        assert_allclose(
            response.command_response[t, :, t, :],
            projection @ model.control,
            rtol=tol,
            atol=tol,
        )


def test_vmap_over_controls_states_and_models(precision):
    dtype, tol = precision
    model, projection, a, d, state, controls = scenario(dtype)
    def prepare(model):
        return linear.prepare_prediction(
            model, 4, output_matrix=projection, state_offset=a, output_offset=d
        )

    response = prepare(model)
    states = jnp.stack((state, -state))
    candidates = jnp.stack((controls, -controls))
    for axes, states_, controls_ in [
        ((None, None, 0), state, candidates),
        ((None, 0, None), states, controls),
        ((None, 0, 0), states, candidates),
    ]:
        expected = jax.vmap(
            scan_predict, in_axes=(None, axes[1], axes[2], None, None, None)
        )(model, states_, controls_, projection, a, d)
        batched = jax.vmap(linear.predict_sequence, in_axes=axes)
        assert_allclose(
            batched(response, states_, controls_), expected, rtol=tol, atol=tol
        )
        assert_allclose(
            jax.jit(batched)(response, states_, controls_), expected, rtol=tol, atol=tol
        )
    models = jax.tree.map(lambda x: jnp.stack((x, x * 0.8)), model)
    responses = jax.jit(jax.vmap(prepare))(models)
    outputs = jax.jit(jax.vmap(linear.predict_sequence, in_axes=(0, None, None)))(
        responses, state, controls
    )
    expected = jax.vmap(scan_predict, in_axes=(0, None, None, None, None, None))(
        models, state, controls, projection, a, d
    )
    assert_allclose(outputs, expected, rtol=tol, atol=tol)


def test_model_and_objective_gradients_include_preparation(precision):
    dtype, tol = precision
    model, projection, a, d, state, controls = scenario(dtype)

    def objective(transition, control, projection, a, d, controls, *, condensed):
        dynamics = model._replace(transition=transition, control=control)
        if condensed:
            response = linear.prepare_prediction(
                dynamics, 4, output_matrix=projection, state_offset=a, output_offset=d
            )
            outputs = linear.predict_sequence(response, state, controls)
        else:
            outputs = scan_predict(dynamics, state, controls, projection, a, d)
        return jnp.sum(outputs**2) + 0.2 * jnp.sum(controls**2)

    gradient = jax.jit(
        jax.grad(objective, argnums=(0, 1, 2, 3, 4, 5)), static_argnames="condensed"
    )
    args = (model.transition, model.control, projection, a, d, controls)
    for actual, expected in zip(
        gradient(*args, condensed=True), gradient(*args, condensed=False), strict=True
    ):
        assert_allclose(actual, expected, rtol=tol, atol=tol)


def test_response_is_immutable_pytree_and_ignores_process_covariance():
    model, _, _, _, state, controls = scenario()
    response = linear.prepare_prediction(model, 4)
    leaves, tree = jax.tree.flatten(response)
    assert len(leaves) == 3
    assert isinstance(jax.tree.unflatten(tree, leaves), linear.PredictionResponse)
    with pytest.raises(AttributeError):
        response.offset = jnp.zeros_like(response.offset)
    other = linear.prepare_prediction(
        model._replace(process_covariance=jnp.full((3, 3), jnp.nan)), 4
    )
    assert_array_equal(
        linear.predict_sequence(response, state, controls),
        linear.predict_sequence(other, state, controls),
    )


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True, jnp.array(2)])
def test_prepare_rejects_invalid_horizons(horizon):
    with pytest.raises(ValueError, match="horizon"):
        linear.prepare_prediction(scenario()[0], horizon)


@pytest.mark.parametrize(
    "field,shape",
    [
        ("transition", (3,)),
        ("transition", (3, 2)),
        ("control", (3,)),
        ("control", (2, 2)),
    ],
)
def test_prepare_rejects_invalid_model_shapes(field, shape):
    model = scenario()[0]._replace(**{field: jnp.zeros(shape)})
    with pytest.raises(ValueError, match=field):
        linear.prepare_prediction(model, 4)


@pytest.mark.parametrize(
    "field,shape",
    [
        ("output_matrix", (3,)),
        ("output_matrix", (2, 2)),
        ("state_offset", (3, 1)),
        ("state_offset", (2,)),
        ("output_offset", (3, 1)),
        ("output_offset", (2,)),
    ],
)
def test_prepare_rejects_invalid_optional_shapes(field, shape):
    with pytest.raises(ValueError, match=field):
        linear.prepare_prediction(scenario()[0], 4, **{field: jnp.zeros(shape)})


@pytest.mark.parametrize(
    "field,shape",
    [
        ("initial_state", (3, 1)),
        ("initial_state", (2,)),
        ("controls", (4,)),
        ("controls", (3, 2)),
        ("controls", (4, 1)),
        ("state_response", (4, 3)),
        ("state_response", (0, 3, 3)),
        ("command_response", (4, 3, 2)),
        ("command_response", (4, 3, 3, 2)),
        ("command_response", (4, 2, 4, 2)),
        ("offset", (4, 1)),
    ],
)
def test_predict_rejects_incompatible_shapes_including_under_jit(field, shape):
    model, _, _, _, state, controls = scenario()
    response = linear.prepare_prediction(model, 4)
    if field == "initial_state":
        state = jnp.zeros(shape)
    elif field == "controls":
        controls = jnp.zeros(shape)
    else:
        response = response._replace(**{field: jnp.zeros(shape)})
    for predict in (linear.predict_sequence, jax.jit(linear.predict_sequence)):
        with pytest.raises(ValueError, match=field):
            predict(response, state, controls)


def test_mixed_preparation_dtypes_use_jax_promotion(precision):
    dtype, _ = precision
    model, projection, a, d, state, controls = scenario(dtype)
    model = model._replace(control=model.control.astype(jnp.float32))
    response = linear.prepare_prediction(
        model, 4, output_matrix=projection, state_offset=a, output_offset=d
    )
    assert all(leaf.dtype == dtype for leaf in response)
    assert linear.predict_sequence(response, state, controls).dtype == dtype
