"""Compare candidate inputs and exact sensitivities for linear motion."""

import jax
import jax.numpy as jnp

from blackwell.models import linear


def run() -> tuple[jax.Array, jax.Array]:
    model = linear.LinearDynamics(
        transition=jnp.array([[1.0, 1.0], [0.0, 1.0]]),
        control=jnp.array([[0.5], [1.0]]),
        process_covariance=jnp.diag(jnp.array([0.02, 0.04])),
    )
    response = linear.prepare_prediction(
        model,
        horizon=4,
        output_matrix=jnp.array([[1.0, 0.0]]),  # Select position.
        state_offset=jnp.array([0.0, -0.05]),
        output_offset=jnp.array([0.1]),
    )
    initial_state = jnp.array([0.0, 0.2])
    candidates = jnp.array(
        [
            [[0.2], [0.1], [0.0], [-0.1]],
            [[0.0], [0.0], [0.0], [0.0]],
            [[-0.1], [0.0], [0.1], [0.2]],
        ]
    )
    predict_many = jax.jit(jax.vmap(linear.predict_sequence, in_axes=(None, None, 0)))
    outputs = predict_many(response, initial_state, candidates)  # (3, 4, 1)

    def objective(controls):
        predicted = linear.predict_sequence(response, initial_state, controls)
        return jnp.sum((predicted - 1.0) ** 2) + 0.1 * jnp.sum(controls**2)

    gradients = jax.jit(jax.vmap(jax.grad(objective)))(candidates)  # (3, 4, 1)
    # This Jacobian is available directly, with no differentiation required.
    assert response.command_response.shape == (4, 1, 4, 1)
    return outputs, gradients


def main() -> None:
    outputs, gradients = run()
    print("Predicted positions, one row per candidate:\n", outputs[:, :, 0])
    print("Objective gradients with respect to each input:\n", gradients[:, :, 0])


if __name__ == "__main__":
    main()
