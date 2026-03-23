"""Trainer distribuido para DRM Transformer.

Suporta single GPU, DDP e FSDP. Inclui:
- Mixed precision (bf16/fp16)
- Gradient accumulation
- Cosine LR schedule com warmup
- Checkpointing
- Logging estruturado
- Losses DRM (metric_reg + metric_diversity)
- NaN gradient skip
"""

import math
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)


class DRMTrainer:
    """Trainer distribuido para DRM Transformer.

    Args:
        config: Dict de configuracao.
        model: Modelo wrapped (DDP/FSDP ou raw).
        train_loader: DataLoader de treino.
        eval_loader: DataLoader de avaliacao (opcional).
    """

    def __init__(
        self,
        config: dict,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.device = next(model.parameters()).device
        self.global_step = 0
        self.best_loss = float("inf")

        self.raw_model = (
            model.module if hasattr(model, "module") else model
        )

        self._eval_no_improve = 0
        self._early_stopped = False

        self._setup_optimizer()
        self._setup_precision()
        self._setup_dirs()

    def _setup_optimizer(self) -> None:
        """Configura optimizer AdamW com weight decay."""
        lr = self.config.get("learning_rate", 3e-4)
        wd = self.config.get("weight_decay", 0.1)
        betas = (
            self.config.get("adam_beta1", 0.9),
            self.config.get("adam_beta2", 0.95),
        )

        decay_params = []
        no_decay_params = []
        for name, param in self.raw_model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "weight" in name and param.dim() < 2:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": wd},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=betas,
        )

    def _setup_precision(self) -> None:
        """Configura mixed precision."""
        mp = self.config.get("mixed_precision", "none")
        self.use_amp = mp in ("bf16", "fp16")
        if mp == "bf16":
            self.amp_dtype = torch.bfloat16
            self.scaler = None
        elif mp == "fp16":
            self.amp_dtype = torch.float16
            self.scaler = GradScaler()
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

    def _setup_dirs(self) -> None:
        """Cria diretorios de checkpoints e logs."""
        self.save_dir = Path(self.config.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(self.config.get("log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.save_dir / "train.log"
        self._log_history = []

    def _lr_schedule(self, step: int) -> float:
        """Cosine schedule com warmup linear.

        Args:
            step: Step atual.

        Returns:
            Fator multiplicativo do LR.
        """
        warmup = self.config.get("warmup_steps", 2000)
        total = self._total_steps()

        if step < warmup:
            return step / max(warmup, 1)

        progress = (step - warmup) / max(total - warmup, 1)
        min_ratio = self.config.get("min_lr_ratio", 0.1)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )

    def _total_steps(self) -> int:
        """Calcula total de optimization steps."""
        total_tokens = self.config.get("total_tokens", 1_000_000_000)
        batch_size = self.config.get("batch_size", 16)
        seq_len = self.config.get("max_seq_len", 1024)
        accum = self.config.get("gradient_accumulation_steps", 1)
        tokens_per_step = batch_size * seq_len * accum
        return total_tokens // max(tokens_per_step, 1)

    def _optimizer_step(self) -> float:
        """Executa optimizer step com NaN guard.

        Returns:
            Gradient norm (ou -1 se skip).
        """
        if self.scaler:
            self.scaler.unscale_(self.optimizer)

        gn = torch.nn.utils.clip_grad_norm_(
            self.raw_model.parameters(),
            self.config.get("max_grad_norm", 1.0),
        ).item()

        if not math.isfinite(gn):
            logger.warning(
                "[TRAINER] SKIP-GRAD step=%d gn=%s",
                self.global_step, gn,
            )
            self.optimizer.zero_grad(set_to_none=True)
            return -1.0

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        return gn

    @torch.no_grad()
    def _extract_drm_metrics(self) -> Dict[str, float]:
        """Extrai metricas DRM para logging.

        Returns:
            Dict com metric_G_var, metric_G_diag_mean, metric_G_frob,
            gamma_mean, mass_mean, temperature, diversity_active, etc.
        """
        metrics = {}
        model = self.raw_model

        metric_net = getattr(model, "metric_net", None)
        if metric_net is None:
            return metrics

        # Coordenadas de amostra para avaliar G(x)
        d = metric_net.dim
        coords = torch.rand(1, 16, d, device=self.device)
        U = metric_net(coords)  # [1, 16, D, r]

        # U(x) estatisticas (low-rank)
        metrics["metric_U_norm_mean"] = U.pow(2).sum(dim=-2).sqrt().mean().item()
        metrics["metric_U_rank"] = float(metric_net.rank)
        # Eixos: norma media por eixo semantico
        axis_norms = U.pow(2).sum(dim=-2).sqrt().mean(dim=(0, 1))  # [r]
        for i, n in enumerate(axis_norms.tolist()):
            metrics[f"metric_axis{i}_norm"] = n

        # Gamma
        anchors = getattr(model, "anchors", None)
        if anchors is not None:
            from ..manifold import gamma_scale
            gamma_c = self.config.get("gamma_c", 4.0)
            gamma = gamma_scale(coords, anchors, c_param=gamma_c)
            metrics["gamma_mean"] = gamma.mean().item()

        # Mass
        gravity = getattr(model, "gravity_field", None)
        if gravity is not None:
            mass = gravity.compute_mass(coords)
            metrics["mass_mean"] = mass.mean().item()

        # Temperatures (media dos heads do block 0)
        if hasattr(model, "blocks") and len(model.blocks) > 0:
            temp = model.blocks[0].attn.temperature
            metrics["temperature"] = temp.item()

        # Diversity ativo?
        warmup_div = self.config.get("metric_diversity_warmup_steps", 5000)
        metrics["diversity_active"] = 1.0 if self.global_step >= warmup_div else 0.0

        # DimGate (se habilitado)
        dim_gate = getattr(model, "dim_gate", None)
        if dim_gate is not None:
            x_sample = torch.randn(1, 4, model.config.d_model, device=self.device)
            _, dimD = dim_gate(x_sample)
            metrics["dimD_mean"] = dimD.mean().item()

        return metrics

    def _compute_drm_losses(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Computa losses DRM (metric regularization + diversity).

        Usa coordenadas reais do batch (via q_to_manifold do block 0)
        em vez de coords aleatorias. O .detach() nas coords impede
        gradiente de fluir de volta para o caminho de atencao.

        Args:
            input_ids: [B, T] tokens do batch actual.

        Returns:
            Loss escalar adicional.
        """
        drm_loss = torch.tensor(0.0, device=self.device)

        model = self.raw_model
        metric_net = getattr(model, "metric_net", None)
        if metric_net is None:
            return drm_loss

        from ..losses import metric_regularization, metric_diversity_loss

        lambda_reg = self.config.get("lambda_metric_reg", 0.001)
        lambda_div = self.config.get("lambda_metric_diversity", 0.05)
        warmup_div = self.config.get("metric_diversity_warmup_steps", 5000)

        # Extrair coords reais do batch via block 0
        with torch.no_grad():
            x = model.token_emb(input_ids)
            if model.dim_gate is not None:
                x, _ = model.dim_gate(x)
            block0 = model.blocks[0]
            x_normed = block0.norm1(x)
            B, T = input_ids.shape
            q = block0.attn.q_proj(x_normed)
            q = q.view(B, T, block0.attn.n_heads, block0.attn.d_head)
            q = q.transpose(1, 2)
            coords = torch.sigmoid(block0.attn.q_to_manifold(q[:, 0]))

        # coords.detach() ja garantido pelo no_grad acima
        # Mas U precisa de grad para MetricNet aprender
        U = metric_net(coords.detach().reshape(-1, metric_net.dim))
        U = U.view(B, T, metric_net.dim, metric_net.rank)

        drm_loss = drm_loss + lambda_reg * metric_regularization(U)

        # Ortogonalidade dos eixos semanticos
        from ..losses import orthogonality_loss
        lambda_ortho = self.config.get("lambda_ortho", 0.01)
        drm_loss = drm_loss + lambda_ortho * orthogonality_loss(U)

        if self.global_step >= warmup_div:
            target_var = self.config.get("target_metric_var", 0.001)
            drm_loss = drm_loss + lambda_div * metric_diversity_loss(U, target_var)

        return drm_loss

    def train(self) -> Dict[str, Any]:
        """Loop principal de treinamento.

        Returns:
            Dict com historico: total_time, final_loss, steps.
        """
        total_steps = self._total_steps()
        accum = self.config.get("gradient_accumulation_steps", 1)
        eval_interval = self.config.get("eval_interval", 500)
        save_interval = self.config.get("save_interval", 1000)
        log_interval = self.config.get("log_interval", 10)
        base_lr = self.config.get("learning_rate", 3e-4)

        is_main = self.config.get("_is_main", True)

        if is_main:
            logger.info(
                "[TRAINER] Iniciando treino: %d steps, accum=%d",
                total_steps, accum,
            )

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        t0 = time.time()
        running_loss = 0.0
        micro_step = 0
        skip_grads = 0
        epoch = 0

        while self.global_step < total_steps:
            epoch += 1
            if hasattr(self.train_loader, "sampler") and hasattr(
                self.train_loader.sampler, "set_epoch"
            ):
                self.train_loader.sampler.set_epoch(epoch)

            for batch in self.train_loader:
                if self.global_step >= total_steps:
                    break

                input_ids = batch["input_ids"].to(self.device)
                targets = batch["targets"].to(self.device)

                with torch.autocast(
                    device_type="cuda",
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    logits, loss = self.model(input_ids, targets)
                    loss = loss / accum

                    drm_loss = self._compute_drm_losses(input_ids) / accum
                    loss = loss + drm_loss

                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss += loss.item() * accum
                micro_step += 1

                if micro_step % accum != 0:
                    continue

                lr_factor = self._lr_schedule(self.global_step)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = base_lr * lr_factor

                gn = self._optimizer_step()
                if gn < 0:
                    skip_grads += 1

                self.global_step += 1

                if is_main and self.global_step % log_interval == 0:
                    avg_loss = running_loss / (log_interval * accum)
                    elapsed = time.time() - t0
                    batch_sz = self.config.get("batch_size", 16)
                    seq_len = self.config.get("max_seq_len", 1024)
                    tokens_seen = self.global_step * batch_sz * seq_len * accum
                    tokens_per_sec = tokens_seen / max(elapsed, 1)
                    current_lr = self.optimizer.param_groups[0]["lr"]

                    # Metricas DRM
                    metrics = self._extract_drm_metrics()

                    # Log line (pipe-separated)
                    parts = [
                        f"step={self.global_step}",
                        f"loss={avg_loss:.4f}",
                        f"lr={current_lr:.2e}",
                        f"gn={gn if gn >= 0 else 0:.2f}",
                        f"tok/s={tokens_per_sec:.0f}",
                        f"tokens={tokens_seen}",
                        f"skip={skip_grads}",
                    ]
                    for k, v in metrics.items():
                        parts.append(f"{k}={v:.6f}")

                    log_line = " | ".join(parts)
                    logger.info(log_line)

                    # Salvar em train.log
                    with open(self._log_path, "a") as f:
                        f.write(log_line + "\n")

                    # Historico para JSON
                    entry = {
                        "step": self.global_step,
                        "loss": avg_loss,
                        "lr": current_lr,
                        "grad_norm": gn if gn >= 0 else 0,
                        "tokens_per_sec": tokens_per_sec,
                        "tokens_seen": tokens_seen,
                        "skip_grads": skip_grads,
                    }
                    entry.update(metrics)
                    self._log_history.append(entry)

                    running_loss = 0.0

                if is_main and self.global_step % save_interval == 0:
                    self.save_checkpoint()

                if (
                    is_main
                    and self.eval_loader
                    and self.global_step % eval_interval == 0
                ):
                    self.evaluate()
                    self.model.train()
                    if self._early_stopped:
                        break

            if self._early_stopped:
                break

        total_time = time.time() - t0
        batch_sz = self.config.get("batch_size", 16)
        seq_len = self.config.get("max_seq_len", 1024)
        total_tokens = self.global_step * batch_sz * seq_len * accum
        avg_tok_s = total_tokens / max(total_time, 1)

        if is_main:
            self.save_checkpoint(tag="final")

            # Salvar training_log.json
            import json
            log_path = self.save_dir / "training_log.json"
            with open(log_path, "w") as f:
                json.dump(self._log_history, f, indent=2)
            logger.info("[LOG] Salvo: %s (%d entries)", log_path, len(self._log_history))

            # Salvar metrics.json (sumario final)
            metrics_summary = {
                "final_loss": running_loss,
                "best_val_loss": self.best_loss if self.best_loss < float("inf") else None,
                "best_val_ppl": math.exp(min(self.best_loss, 20)) if self.best_loss < float("inf") else None,
                "total_steps": self.global_step,
                "total_tokens": total_tokens,
                "total_time_s": round(total_time, 1),
                "avg_tokens_per_s": round(avg_tok_s),
                "skip_grads": skip_grads,
                "early_stopped": self._early_stopped,
                "seed": self.config.get("seed", "unknown"),
            }
            metrics_path = self.save_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics_summary, f, indent=2)
            logger.info("[METRICS] Salvo: %s", metrics_path)

            logger.info(
                "[TRAINER] Concluido: %d steps em %.0fs (%.0f tok/s)",
                self.global_step, total_time, avg_tok_s,
            )

        return {
            "total_time": total_time,
            "final_loss": running_loss,
            "steps": self.global_step,
            "skip_grads": skip_grads,
            "early_stopped": self._early_stopped,
        }

    @torch.no_grad()
    def evaluate(self) -> float:
        """Avalia no eval set.

        Returns:
            Loss medio.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)

            with torch.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                _, loss = self.model(input_ids, targets)

            total_loss += loss.item()
            n_batches += 1
            if n_batches >= 50:
                break

        avg = total_loss / max(n_batches, 1)
        ppl = math.exp(min(avg, 20))
        logger.info(
            "[EVAL] step=%d | val_loss=%.4f | ppl=%.2f",
            self.global_step, avg, ppl,
        )

        # Salvar no historico de log
        self._log_history.append({
            "step": self.global_step,
            "val_loss": avg,
            "val_ppl": ppl,
            "type": "eval",
        })

        if avg < self.best_loss:
            self.best_loss = avg
            self._eval_no_improve = 0
            self.save_checkpoint(tag="best")
        else:
            self._eval_no_improve += 1
            patience = self.config.get("early_stop_patience", 0)
            if patience > 0 and self._eval_no_improve >= patience:
                logger.info(
                    "[EARLY STOP] %d evals sem melhora (patience=%d)",
                    self._eval_no_improve, patience,
                )
                self._early_stopped = True

        return avg

    def save_checkpoint(self, tag: Optional[str] = None) -> None:
        """Salva checkpoint.

        Args:
            tag: Nome especial (best, final) ou None para step_N.
        """
        name = tag if tag else f"step_{self.global_step}"
        path = self.save_dir / f"{name}.pt"

        state = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "config": self.config,
        }
        if self.scaler:
            state["scaler"] = self.scaler.state_dict()

        torch.save(state, path)
        logger.info("[CHECKPOINT] Salvo: %s", path)

        # Manter apenas os N checkpoints de step mais recentes
        if tag is None:
            limit = self.config.get("save_total_limit", 5)
            step_files = sorted(
                self.save_dir.glob("step_*.pt"),
                key=lambda p: p.stat().st_mtime,
            )
            while len(step_files) > limit:
                old = step_files.pop(0)
                old.unlink()
                logger.info("[CHECKPOINT] Removido: %s", old.name)

    def load_checkpoint(self, path: str) -> None:
        """Carrega checkpoint.

        Args:
            path: Caminho do .pt.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)

        model_state = state.get("model", {})
        cleaned = {}
        for k, v in model_state.items():
            k = k.replace("module.", "").replace("_orig_mod.", "")
            cleaned[k] = v
        self.raw_model.load_state_dict(cleaned, strict=False)

        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scaler and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

        self.global_step = state.get("global_step", 0)
        self.best_loss = state.get("best_loss", float("inf"))

        logger.info(
            "[CHECKPOINT] Carregado: %s (step=%d)",
            path, self.global_step,
        )
