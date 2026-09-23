# JAX and Flax (new NNX API)
import jax.numpy as jnp  # Numpy for JAX
from flax import nnx

# SIMPLEST
        
class Phi(nnx.Module):
    def __init__(
        self,
        Nsize_p,
        n_cols,
        n_params,
        drop_rate,
        n_hidden_layers,
        *,
        rngs,
    ):
        self.input = nnx.Linear(
            n_cols,
            Nsize_p,
            use_bias=True,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

        self.hidden = [
            nnx.Linear(
                Nsize_p,
                Nsize_p,
                use_bias=True,
                kernel_init=nnx.initializers.lecun_normal(),
                rngs=rngs,
            )
            for _ in range(n_hidden_layers)
        ]

        self.output = nnx.Linear(
            Nsize_p,
            Nsize_p,
            use_bias=True,
            kernel_init=nnx.initializers.normal(stddev=0.01),
            rngs=rngs,
        )

        self.dropout = nnx.Dropout(drop_rate, rngs=rngs)

    def __call__(self, data, mask):
        h = data

        h = self.input(h)
        h = nnx.gelu(h)
        h = self.dropout(h)

        for layer in self.hidden:
            h = layer(h)
            h = nnx.gelu(h)
            h = self.dropout(h)

        return self.output(h)


class Rho(nnx.Module):
    def __init__(
        self,
        Nsize_p,
        Nsize_r,
        n_params,
        drop_rate,
        n_hidden_layers,
        *,
        rngs,
    ):
        self.input = nnx.Linear(
            Nsize_p + n_params + 1,
            Nsize_r,
            use_bias=True,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

        self.hidden = [
            nnx.Linear(
                Nsize_r,
                Nsize_r,
                use_bias=True,
                kernel_init=nnx.initializers.lecun_normal(),
                rngs=rngs,
            )
            for _ in range(n_hidden_layers)
        ]

        self.output = nnx.Linear(
            Nsize_r,
            1,
            use_bias=True,
            kernel_init=nnx.initializers.normal(stddev=0.01),
            rngs=rngs,
        )

        self.dropout = nnx.Dropout(drop_rate, rngs=rngs)

    def __call__(self, pooled_features, params):
        x = jnp.concatenate([pooled_features, params], axis=-1)

        x = self.input(x)
        x = nnx.gelu(x)
        x = self.dropout(x)

        for layer in self.hidden:
            x = layer(x)
            x = nnx.gelu(x)
            x = self.dropout(x)

        return self.output(x)


class DeepSetClassifier(nnx.Module):
    def __init__(
        self,
        phi_drop_rate,
        rho_drop_rate,
        Nsize_p,
        Nsize_r,
        depth_p,
        depth_r,
        n_cols,
        n_params,
        # val_idx,
        # err_idx,
        *,
        rngs,
    ):
        self.n_cols = n_cols
        self.n_params = n_params

        self.phi = Phi(
            Nsize_p,
            self.n_cols,
            self.n_params,
            phi_drop_rate,
            depth_p,
            rngs=rngs,
        )

        self.rho = Rho(
            Nsize_p,
            Nsize_r,
            n_params,
            rho_drop_rate,
            depth_r,
            rngs=rngs,
        )

        self.norm = nnx.LayerNorm(Nsize_p, rngs=rngs)

    def __call__(self, input_data):
        if input_data.ndim == 1:
            input_data = input_data[None, :]

        N = input_data.shape[0]
        input_dim = input_data.shape[1]

        M = (input_dim - self.n_params - 1) // (self.n_cols + 1)

        data = input_data[:, :M * self.n_cols].reshape(
            N, M, self.n_cols
        )

        mask = input_data[
            :, M * self.n_cols : M * self.n_cols + M
        ]

        M_norm = input_data[
            :, M * self.n_cols + M : M * self.n_cols + M + 1
        ]

        theta = input_data[:, -self.n_params:]

        features = self.phi(
            data,
            mask[..., None],
        )

        mask_exp = mask[..., None]

        features = features * mask_exp

        mask_sum = jnp.sum(mask_exp, axis=1)
        mask_sum = jnp.maximum(mask_sum, 1.0)

        pooled = self.norm(
            jnp.sum(features, axis=1) / mask_sum
        )

        pooled_M = jnp.concatenate(
            [pooled, M_norm],
            axis=-1,
        )

        return self.rho(pooled_M, theta)