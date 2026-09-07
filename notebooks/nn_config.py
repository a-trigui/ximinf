nn_path = '../data/NNs/nn_model_priors_uniform_SM_N_100000_1_M_1000_batch_500_lr_1e-4_decay_1e-3_relu_isup-0_5_rng_mabs_beta_alpha_gamma_dropout_0_0_phi_0_2_rho_values_err_deepset_256_depth_3_patience_30_equivariant_cut_layernorm_pool_planckBAO18'

# JAX and Flax (new NNX API)
import jax.numpy as jnp  # Numpy for JAX
from flax import nnx

class EquivariantBlock(nnx.Module):
    def __init__(self, Nsize, drop_rate, *, rngs):
        self.local = nnx.Linear(
            Nsize,
            Nsize,
            use_bias=True,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.global_proj = nnx.Linear(
            Nsize,
            Nsize,
            use_bias=True,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        # self.norm = nnx.LayerNorm(Nsize, rngs=rngs)
        self.dropout = nnx.Dropout(drop_rate, rngs=rngs)


    def __call__(self, h, mask):
        # mask: [N, M, 1]
        local = self.local(h)
        denom = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
        global_feat = jnp.sum(h * mask, axis=1, keepdims=True) / denom
        global_feat = self.global_proj(global_feat)
        global_feat = jnp.broadcast_to(global_feat, h.shape)

        h = local + global_feat
        # h = self.norm(h)
        h = nnx.gelu(h)
        h = self.dropout(h)
        return h

class Phi(nnx.Module):
    def __init__(
        self,
        Nsize_p,
        n_cols_val,
        n_cols_err,
        n_params,
        drop_rate,
        n_hidden_layers,
        *,
        rngs,
    ):
        self.input = nnx.Linear(
            n_cols_val + n_cols_err, # + n_cols_err, + n_params
            Nsize_p,
            use_bias=True,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

        self.equiv_layers = [
            EquivariantBlock(Nsize_p, drop_rate, rngs=rngs)
            for _ in range(n_hidden_layers)
        ]

        # self.hidden_norms = [nnx.LayerNorm(Nsize, rngs=rngs) for _ in range(n_hidden_layers)]

        self.output = nnx.Linear(
            Nsize_p,
            Nsize_p,
            use_bias=True,
            kernel_init=nnx.initializers.normal(stddev=0.01),
            rngs=rngs,
        )

        # self.output_norm = nnx.LayerNorm(Nsize, rngs=rngs)

        self.dropout = nnx.Dropout(drop_rate, rngs=rngs)

    def __call__(self, values, errors, mask): #, theta, mask
        # h = values
        # h = jnp.concatenate([values, theta], axis=-1)
        h = jnp.concatenate([values, errors], axis=-1)
        # h = jnp.concatenate([values, errors, theta], axis=-1)

        h = self.input(h)
        h = nnx.gelu(h)
        h = self.dropout(h)

        # equivariant refinement before pooling
        for layer in self.equiv_layers:
            h = layer(h, mask)

        # for layer, norm in zip(self.equiv_layers, self.hidden_norms):
        #     h = layer(h)
        #     h = norm(h)

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
            Nsize_p + n_params + 1, # + 1
            Nsize_r,
            use_bias=True,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

        # self.input_norm = nnx.LayerNorm(Nsize_r, rngs=rngs)

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
        
        # self.hidden_norms = [nnx.LayerNorm(Nsize_r, rngs=rngs) for _ in range(n_hidden_layers)]

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
        # x = self.input_norm(x)
        x = nnx.gelu(x)
        x = self.dropout(x)

        for layer in self.hidden:
            x = layer(x)
            x = nnx.gelu(x)
            x = self.dropout(x)

        # for linear, norm in zip(self.hidden, self.hidden_norms):
        #     x = linear(x)
        #     x = norm(x)
        #     x = nnx.gelu(x)
        #     x = self.dropout(x)

        return self.output(x)


class DeepSetClassifier(nnx.Module):
    def __init__(self, phi_drop_rate, rho_drop_rate,
                 Nsize_p, Nsize_r,
                 depth_p, depth_r, depth_w,
                 n_cols, n_params, val_idx, err_idx, *, rngs):

        self.n_cols = n_cols
        self.n_params = n_params

        self.val_idx = jnp.asarray(val_idx)
        self.err_idx = jnp.asarray(err_idx)

        self.n_cols_val = len(val_idx)
        self.n_cols_err = len(err_idx)

        self.phi = Phi(Nsize_p, self.n_cols_val, self.n_cols_err, self.n_params, phi_drop_rate, depth_p, rngs=rngs)
        self.rho = Rho(Nsize_p, Nsize_r, n_params, rho_drop_rate, depth_r, rngs=rngs)
        # self.weights = Weights(Nsize_p, self.n_cols_val, self.n_cols_err, self.n_params, phi_drop_rate, depth_w, rngs=rngs)

        self.norm = nnx.LayerNorm(Nsize_p, rngs=rngs)

    def __call__(self, input_data): # Add parameters

        if input_data.ndim == 1:
            input_data = input_data[None, :]

        N = input_data.shape[0]
        input_dim = input_data.shape[1]

        M = (input_dim - self.n_params - 1) // (self.n_cols + 1)

        data = input_data[:, :M * self.n_cols].reshape(N, M, self.n_cols)

        values = data[..., self.val_idx]
        errors = data[..., self.err_idx]

        mask = input_data[:, M * self.n_cols : M * self.n_cols + M]
        
        M_norm = input_data[:, M * self.n_cols + M : M * self.n_cols + M + 1]
        
        theta = input_data[:, -self.n_params:]
        # theta_exp = jnp.broadcast_to(theta[:, None, :], (N, M, self.n_params))

        # element-wise representation
        features = self.phi(values, errors, mask[..., None])
        # features = self.phi(values, errors, theta_exp, mask[..., None])

        # weights = self.weights(values, errors)
        # weights = self.weights(values, errors, theta_exp)

        # masked pooling
        mask_exp = mask[..., None]
        
        features = features * mask_exp
        # weights = weights * mask_exp

        mask_sum = jnp.sum(mask_exp, axis=1)
        mask_sum = jnp.maximum(mask_sum, 1.0)

        # pooled = jnp.sum(features, axis=1)/mask_sum
        pooled = self.norm(jnp.sum(features, axis=1)/mask_sum)
        # pooled = self.norm(jnp.sum(features * weights, axis=1)/jnp.sum(weights*mask_exp, axis=1)) #/mask_sum)   # sum pooling  #* weights #sum(weights*mask, axis=1)
        pooled_M = jnp.concatenate([pooled, M_norm], axis=-1)

        return self.rho(pooled_M, theta)