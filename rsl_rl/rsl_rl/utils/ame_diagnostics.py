"""Read-only AME training diagnostics, independent of the optimizer and checkpoint.

Only detached summaries survive a minibatch. Loss attribution reuses its graph;
it never runs another policy forward or accumulates into parameter gradients.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
import warnings


def terrain_column_names(generator):
    """Match Isaac Lab's curriculum column allocation; random cells are unknown."""
    if generator is None or not generator.curriculum:
        return None
    names = list(generator.sub_terrains)
    proportions = [float(generator.sub_terrains[name].proportion) for name in names]
    total = sum(proportions)
    if total <= 0:
        return None
    cumulative = []
    value = 0.0
    for proportion in proportions:
        value += proportion / total
        cumulative.append(value)
    return [
        names[next(i for i, bound in enumerate(cumulative) if col / generator.num_cols + 0.001 < bound)]
        for col in range(generator.num_cols)
    ]


class TerrainDiagnostics:
    """Accumulate pre-step motion and post-step episode outcomes by terrain type."""

    def __init__(self, env):
        self.env = getattr(env, "unwrapped", env)
        self.available = False
        self.names = ["unknown"]
        self.env_names = ["unknown"] * env.num_envs
        self.dt = float(getattr(self.env, "step_dt", 0.02))
        self.device = env.device
        self.stall_time = torch.zeros(env.num_envs, device=self.device)
        self.group_ids = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        scene = getattr(self.env, "scene", None)
        if scene is not None and hasattr(self.env, "command_manager"):
            self.robot = scene["robot"]
            self.terrain = getattr(scene, "terrain", None)
            generator = getattr(getattr(self.terrain, "cfg", None), "terrain_generator", None)
            columns = terrain_column_names(generator)
            types = getattr(self.terrain, "terrain_types", None)
            if columns is not None and types is not None:
                self.env_names = [columns[i] for i in types.detach().cpu().tolist()]
                self.names = list(dict.fromkeys(self.env_names))
                self.group_ids = torch.tensor([self.names.index(n) for n in self.env_names], device=self.device)
            self.available = True
        # exposure, error, directed speed, moving exposure, stalled exposure,
        # level sum, completed episodes, failures, classified outcomes.
        self.sums = torch.zeros(len(self.names), 9, device=self.device, dtype=torch.float64)

    @torch.no_grad()
    def before_step(self):
        if not self.available:
            return
        command = self.env.command_manager.get_command("base_velocity")[:, :2]
        velocity = self.robot.data.root_lin_vel_b[:, :2]
        speed = command.norm(dim=-1)
        directed = (velocity * command).sum(-1) / speed.clamp_min(1e-8)
        moving = speed > 0.2
        stalled = moving & (directed < 0.1)
        self.stall_time.copy_(torch.where(stalled, self.stall_time + self.dt, 0.0))
        levels = getattr(self.terrain, "terrain_levels", None)
        if levels is None:
            levels = torch.zeros_like(speed)
        values = torch.stack(
            (
                torch.ones_like(speed),
                (command - velocity).norm(dim=-1),
                directed,
                moving.float(),
                (self.stall_time >= 1.0 - 1e-6).float(),
                levels.float(),
            ),
            dim=-1,
        )
        self.sums[:, :6].index_add_(0, self.group_ids, values.double())

    @torch.no_grad()
    def after_step(self, dones, extras):
        done = dones.flatten().bool()
        self.stall_time[done] = 0.0
        if not self.available:
            return
        timeout = extras.get("time_outs")
        known = timeout is not None
        failures = done & ~timeout.flatten().bool() if known else torch.zeros_like(done)
        values = torch.stack(
            (done.float(), failures.float(), done.float() if known else torch.zeros_like(done).float()), -1
        )
        self.sums[:, 6:].index_add_(0, self.group_ids, values.double())

    def flush(self):
        metrics = {}
        if self.available:
            for name, values in zip(self.names, self.sums.cpu().tolist()):
                samples, error, speed, moving, stalled, levels, completed, failures, known = values
                prefix = f"AME/Terrain/{name}"
                metrics[f"{prefix}/samples"] = samples
                metrics[f"{prefix}/moving_samples"] = moving
                metrics[f"{prefix}/completed_episodes"] = completed
                metrics[f"{prefix}/classified_episodes"] = known
                if samples:
                    metrics[f"{prefix}/velocity_error"] = error / samples
                    metrics[f"{prefix}/directed_velocity"] = speed / samples
                    metrics[f"{prefix}/terrain_level"] = levels / samples
                if moving:
                    metrics[f"{prefix}/stall_fraction"] = stalled / moving
                if known:
                    metrics[f"{prefix}/non_timeout_termination_fraction"] = failures / known
        self.sums.zero_()
        return metrics


def attention_statistics(weights):
    """Return per-head entropy/peak and mean off-diagonal head cosine similarity."""
    p = weights.detach().float().squeeze(-2)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)
    entropy = entropy / math.log(p.shape[-1]) if p.shape[-1] > 1 else torch.zeros_like(entropy)
    normalized = F.normalize(p, dim=-1)
    similarity = normalized @ normalized.transpose(-2, -1)
    heads = p.shape[1]
    off_diagonal = similarity.sum((-2, -1)) - similarity.diagonal(dim1=-2, dim2=-1).sum(-1)
    similarity = off_diagonal.mean() / (heads * (heads - 1)) if heads > 1 else p.new_tensor(float("nan"))
    return entropy.mean(0), p.amax(-1).mean(0), similarity


def attention_js(actor, critic):
    """Jensen-Shannon divergence between paired actor/critic attention distributions."""
    p, q = actor.detach().float(), critic.detach().float()
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
    middle = (p + q) * 0.5
    return (
        0.5
        * (
            p * (p.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())
            + q * (q.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())
        ).sum(-1)
    ).mean()


def _colorize(values):
    """Blue-to-yellow heatmap; x goes right and y up in TensorBoard."""
    x = values.float().clamp(0, 1).flip(-2)
    return torch.stack((x, x.sqrt(), 1.0 - x), dim=0)


def attention_image(height, weights, grid_shape):
    """Height, mean overlay, then all heads; each tile uses the same input grid."""
    height = height.cpu().float()
    heads = weights.cpu().float().reshape(-1, *grid_shape)
    # Conv stride-2 centers are 0, 2, ..., not a transposed/flattened ray grid.
    target_h, target_w = height.shape
    if heads.shape[-2:] != height.shape:
        center_shape = (2 * heads.shape[-2] - 1, 2 * heads.shape[-1] - 1)
        heads = F.interpolate(heads[:, None], size=center_shape, mode="bilinear", align_corners=True)[:, 0]
        heads = F.pad(heads, (0, target_w - center_shape[1], 0, target_h - center_shape[0]), mode="replicate")
    ground = _colorize((height + 1.2) / 1.2)
    # Fixed scale relative to uniform attention, shared across heads and updates.
    heat = [_colorize(h * math.prod(grid_shape) / 4.0) for h in heads]
    mean = _colorize(heads.mean(0) * math.prod(grid_shape) / 4.0)
    tiles = [ground, 0.5 * ground + 0.5 * mean, *heat]
    columns = 4
    tiles += [torch.zeros_like(ground)] * ((-len(tiles)) % columns)
    rows = [torch.cat(tiles[i : i + columns], dim=-1) for i in range(0, len(tiles), columns)]
    image = torch.cat(rows, dim=-2)
    return F.interpolate(image[None], scale_factor=4, mode="nearest")[0]


class AMEDiagnostics:
    """Collector owned by OnPolicyRunner, with no serialized model state."""

    def __init__(self, policy, cfg, env=None):
        self.policy = policy
        self.scalar_interval = int(cfg.get("scalar_interval", 10))
        self.attribution_interval = int(cfg.get("attribution_interval", 50))
        self.image_interval = int(cfg.get("image_interval", 200))
        self.max_images = int(cfg.get("max_images", 4))
        if min(self.scalar_interval, self.attribution_interval, self.image_interval) < 1 or self.max_images < 0:
            raise ValueError("AME diagnostic intervals must be positive and max_images must be nonnegative.")
        self.params = tuple(p for p in policy.parameters() if p.requires_grad)
        indices = {id(p): i for i, p in enumerate(self.params)}
        self.groups = {}
        for label, attribute in (
            ("cnn", "map_cnn"),
            ("mha", "mha"),
            ("actor_query", "actor_proprio_embedding"),
            ("critic_query", "critic_proprio_embedding"),
            ("actor", "actor"),
            ("critic", "critic"),
            ("global_encoder", "global_encoder"),
            ("query_projector", "query_projector"),
        ):
            module = getattr(policy, attribute, None)
            if module is not None:
                self.groups[label] = [indices[id(p)] for p in module.parameters() if p.requires_grad]
        noise = getattr(policy, "std", getattr(policy, "log_std", None))
        if noise is not None and noise.requires_grad:
            self.groups["action_noise"] = [indices[id(noise)]]
        self.terrain = TerrainDiagnostics(env) if env is not None else None
        self.scalars = {}
        self.optional = {}
        self.images = {}
        self._hooks = []
        self._attention = {}
        self._before = None
        self._loss_samples = []
        self._clip_samples = []
        self.capture_features = False
        self.capture_images = False
        self.forward_role = "actor"
        self._legend_written = False
        self.begin_iteration(0)

    @property
    def capture_encoder(self):
        return self.capture_features or self.capture_images

    def begin_iteration(self, iteration):
        self.iteration = iteration
        self.scalar_due = iteration % self.scalar_interval == 0
        self.attribution_due = iteration % self.attribution_interval == 0
        self.image_due = iteration % self.image_interval == 0 and self.max_images > 0
        self.capture_features = self.capture_images = False
        self._rollout_steps = 0
        self._batch_index = -1
        self._loss_samples.clear()
        self._clip_samples.clear()

    @torch.no_grad()
    def before_step(self, obs):
        if self.terrain is not None:
            self.terrain.before_step()
        self.capture_images = self.image_due and self._rollout_steps == 0
        if not self.capture_images:
            return
        scans = self.policy.get_actor_obs(obs)[:, -math.prod(self.policy.map_scan_dim) :]
        heights = scans.reshape(scans.shape[0], self.policy.W, self.policy.L, 3)[..., 2]
        variation = heights.flatten(1).amax(1) - heights.flatten(1).amin(1)
        names = self.terrain.env_names if self.terrain is not None else ["unknown"] * len(heights)
        selected = []
        order = list(dict.fromkeys(("hf_steppingstones", "hf_gaps", *names)))
        for name in order:
            candidates = [i for i, n in enumerate(names) if n == name]
            if candidates and len(selected) < self.max_images:
                ids = torch.tensor(candidates, device=variation.device)
                selected.append(candidates[variation[ids].argmax().item()])
        for i in variation.argsort(descending=True).tolist():
            if len(selected) >= self.max_images:
                break
            if i not in selected:
                selected.append(i)
        self._image_ids = selected
        self._image_names = [names[i] for i in selected]

    def after_step(self, dones, extras):
        self.capture_images = False
        self._rollout_steps += 1
        if self.terrain is not None:
            self.terrain.after_step(dones, extras)

    def begin_minibatch(self, index):
        self._batch_index = index
        self.capture_features = index == 0 and self.scalar_due
        if self.capture_features:
            for name, module in self.policy.map_cnn.named_modules():
                if isinstance(module, torch.nn.ReLU):
                    self._hooks.append(
                        module.register_forward_hook(lambda _m, _a, output, name=name: self._record_relu(name, output))
                    )

    @torch.no_grad()
    def _record_relu(self, name, output):
        if not self.capture_features:
            return
        self.scalars[f"AME/Features/{self.forward_role}/relu_{name}/zero_fraction"] = (output == 0).float().mean()

    @torch.no_grad()
    def _feature(self, name, value):
        value = value.detach().float()
        finite = torch.isfinite(value)
        safe = torch.where(finite, value, 0.0)
        count = finite.sum().clamp_min(1)
        mean = safe.sum() / count
        square = safe.square().sum() / count
        self.scalars[f"{name}/nonfinite_fraction"] = (~finite).float().mean()
        for suffix, statistic in (
            ("mean", mean),
            ("std", (square - mean.square()).clamp_min(0).sqrt()),
            ("rms", square.sqrt()),
        ):
            self.scalars[f"{name}/{suffix}"] = statistic
            self.optional[f"{name}/{suffix}"] = finite.any()

    @torch.no_grad()
    def record_encoder(self, role, scan, cnn, query, output, attention, grid_shape):
        if self.capture_features:
            for name, value in (("cnn", cnn), ("query", query), ("mha", output)):
                self._feature(f"AME/Features/{role}/{name}", value)
            entropy, peak, similarity = attention_statistics(attention)
            for head in range(attention.shape[1]):
                self.scalars[f"AME/Attention/{role}/head_{head:02d}/entropy"] = entropy[head]
                self.scalars[f"AME/Attention/{role}/head_{head:02d}/max_weight"] = peak[head]
            self.scalars[f"AME/Attention/{role}/head_similarity_valid"] = float(attention.shape[1] > 1)
            if attention.shape[1] > 1:
                self.scalars[f"AME/Attention/{role}/head_similarity"] = similarity
            self._attention[role] = attention.detach()
            if "actor" in self._attention and "critic" in self._attention:
                self.scalars["AME/Attention/actor_critic_js"] = attention_js(
                    self._attention["actor"], self._attention["critic"]
                )
                self._attention.clear()
                # Auxiliary symmetry forwards must not overwrite the PPO pair.
                self.capture_features = False
        if self.capture_images:
            for slot, (i, name) in enumerate(zip(self._image_ids, self._image_names)):
                self.images[f"AME/Attention/{name}/sample_{slot}/{role}"] = (
                    scan[i, ..., 2].detach().to(device="cpu", copy=True),
                    attention[i].detach().to(device="cpu", copy=True),
                    grid_shape,
                )

    @staticmethod
    def _norm(values, reference):
        squares = [v.detach().float().square().sum() for v in values if v is not None]
        return torch.stack(squares).sum().sqrt() if squares else reference.new_zeros(())

    def _attribution(self, policy_loss, value_loss):
        gp = torch.autograd.grad(policy_loss, self.params, retain_graph=True, allow_unused=True)
        gv = torch.autograd.grad(value_loss, self.params, retain_graph=True, allow_unused=True)
        with torch.no_grad():
            for name, indices in self.groups.items():
                a, b = [gp[i] for i in indices], [gv[i] for i in indices]
                na, nb = self._norm(a, policy_loss), self._norm(b, value_loss)
                prefix = f"AME/Attribution/{name}"
                self.scalars[f"{prefix}/policy_grad_l2"] = na
                self.scalars[f"{prefix}/value_grad_l2"] = nb
                self.scalars[f"{prefix}/policy_connected"] = float(any(g is not None for g in a))
                self.scalars[f"{prefix}/value_connected"] = float(any(g is not None for g in b))
                valid = (na > 0) & (nb > 0) & torch.isfinite(na) & torch.isfinite(nb)
                dots = [
                    (x.detach().float() * y.detach().float()).sum()
                    for x, y in zip(a, b)
                    if x is not None and y is not None
                ]
                dot = torch.stack(dots).sum() if dots else na.new_zeros(())
                self.scalars[f"{prefix}/cosine_valid"] = valid.float()
                self.scalars[f"{prefix}/cosine"] = dot / (na * nb).clamp_min(1e-30)
                self.optional[f"{prefix}/cosine"] = valid
                ratio_valid = (na > 0) & torch.isfinite(na) & torch.isfinite(nb)
                self.scalars[f"{prefix}/value_policy_ratio_valid"] = ratio_valid.float()
                self.scalars[f"{prefix}/value_policy_ratio"] = nb / na.clamp_min(1e-30)
                self.optional[f"{prefix}/value_policy_ratio"] = ratio_valid

    def before_backward(
        self, policy_loss, value_loss, total_loss, ratio, returns, values, old_mu, old_sigma, mu, sigma, clip_param
    ):
        if self.attribution_due and self._batch_index == 0:
            self._attribution(policy_loss, value_loss)
        if not self.scalar_due:
            return
        with torch.no_grad():
            old_sigma, sigma = old_sigma.detach().clamp_min(1e-12), sigma.detach().clamp_min(1e-12)
            kl = (
                (
                    torch.log(sigma / old_sigma)
                    + (old_sigma.square() + (old_mu.detach() - mu.detach()).square()) / (2 * sigma.square())
                    - 0.5
                )
                .sum(-1)
                .mean()
            )
            if self._batch_index == 0:
                self.scalars["AME/PPO/pre_update_kl"] = kl
            target = returns.detach().double()
            residual = target - values.detach().double()
            self._loss_samples.append(
                torch.stack((
                    policy_loss.detach(),
                    value_loss.detach(),
                    total_loss.detach(),
                    kl,
                    ((ratio.detach() - 1).abs() > clip_param).float().mean(),
                    target.new_tensor(target.numel()),
                    target.sum(),
                    target.square().sum(),
                    residual.sum(),
                    residual.square().sum(),
                ))
            )

    @torch.no_grad()
    def after_backward(self):
        if self._batch_index != 0 or not (self.scalar_due or self.attribution_due):
            return
        self._before = [p.detach().clone() for p in self.params]
        for name, indices in self.groups.items():
            params = [self.params[i] for i in indices]
            gradients = [p.grad for p in params]
            norm = self._norm(gradients, params[0])
            prefix = f"AME/Modules/{name}"
            self.scalars[f"{prefix}/grad_l2"] = norm
            self.scalars[f"{prefix}/grad_rms"] = norm / math.sqrt(sum(p.numel() for p in params))
            self.scalars[f"{prefix}/grad_connected"] = float(any(g is not None for g in gradients))
            self.scalars[f"{prefix}/grad_nonzero"] = (norm > 0).float()
            self.scalars[f"{prefix}/parameter_l2"] = self._norm(params, params[0])
        for name, module in self.policy.map_cnn.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                self._feature(f"AME/Features/bn_{name}/running_variance", module.running_var)
                self.scalars[f"AME/Features/bn_{name}/running_variance/min"] = module.running_var.detach().min()
                self.scalars[f"AME/Features/bn_{name}/running_variance/max"] = module.running_var.detach().max()

    @torch.no_grad()
    def record_clipping(self, norm, max_norm):
        if self.scalar_due:
            self._clip_samples.append(torch.stack(((norm > max_norm).float(), (max_norm / (norm + 1e-6)).clamp(max=1))))

    @torch.no_grad()
    def after_optimizer_step(self):
        if self._before is not None:
            for name, indices in self.groups.items():
                before = [self._before[i] for i in indices]
                delta = [self.params[i].detach() - self._before[i] for i in indices]
                norm = self._norm(before, before[0])
                prefix = f"AME/Modules/{name}"
                self.scalars[f"{prefix}/update_relative"] = self._norm(delta, before[0]) / norm.clamp_min(1e-30)
                self.scalars[f"{prefix}/update_relative_valid"] = (norm > 0).float()
                self.optional[f"{prefix}/update_relative"] = norm > 0
            self._before = None
        self.capture_features = False
        self._attention.clear()
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def write(self, writer):
        if self._loss_samples:
            losses = torch.stack(self._loss_samples)
            for name, value in zip(
                ("weighted_policy_loss", "weighted_value_loss", "total_loss", "update_kl", "clip_fraction"),
                losses[:, :5].mean(0),
            ):
                self.scalars[f"AME/PPO/{name}"] = value
            count, target, target2, residual, residual2 = losses[:, 5:].sum(0)
            var_target = (target2 / count - (target / count).square()).clamp_min(0)
            var_error = (residual2 / count - (residual / count).square()).clamp_min(0)
            self.scalars["AME/PPO/explained_variance_valid"] = (var_target > 1e-12).float()
            self.scalars["AME/PPO/explained_variance"] = 1 - var_error / var_target.clamp_min(1e-12)
            self.optional["AME/PPO/explained_variance"] = var_target > 1e-12
        if self._clip_samples:
            clips = torch.stack(self._clip_samples).mean(0)
            self.scalars["AME/PPO/gradient_clip_fraction"] = clips[0]
            self.scalars["AME/PPO/gradient_clip_scale"] = clips[1]
        # One transfer for scalars and validity, rather than a GPU sync per tag.
        if self.scalars:
            device = self.params[0].device
            keys = list(self.scalars)
            packed = (
                torch.stack([
                    torch.stack((
                        torch.as_tensor(self.scalars[k], device=device).double(),
                        torch.as_tensor(self.optional.get(k, True), device=device).double(),
                    ))
                    for k in keys
                ])
                .detach()
                .cpu()
                .tolist()
            )
            invalid = 0
            for name, (value, valid) in zip(keys, packed):
                if valid:
                    if math.isfinite(value):
                        writer.add_scalar(name, value, self.iteration)
                    else:
                        invalid += 1
            writer.add_scalar("AME/Logging/nonfinite_scalar_count", invalid, self.iteration)
        if self.scalar_due and self.terrain is not None:
            for name, value in self.terrain.flush().items():
                writer.add_scalar(name, value, self.iteration)
        for name, (height, attention, grid) in self.images.items():
            writer.add_image(name, attention_image(height, attention, grid), self.iteration)
        if self.images and not self._legend_written:
            writer.add_text(
                "AME/Attention/legend",
                "Tiles: input height, mean-attention overlay, heads in order. "
                "x right, y up. Height colors: -1.2 to 0 m. Attention colors: 0 to 4 times uniform. "
                "These are rollout samples, not causal attributions.",
                self.iteration,
            )
            self._legend_written = True
        self.scalars.clear()
        self.optional.clear()
        self.images.clear()
        self._loss_samples.clear()
        self._clip_samples.clear()


def create_ame_diagnostics(policy, cfg, env, is_distributed=False):
    """Enable only supported configurations; keep ordinary training operational."""
    if not cfg or not cfg.get("enabled", False):
        return None
    if is_distributed:
        warnings.warn(
            "AME diagnostics support single-GPU training only; diagnostics disabled for this distributed run."
        )
        return None
    if not hasattr(policy, "map_cnn") or not hasattr(policy, "mha"):
        warnings.warn("AME diagnostics require ActorCriticEncoder; diagnostics disabled for this policy.")
        return None
    return AMEDiagnostics(policy, cfg, env)
