from typing import Optional

import einops
import torch
from todd.models.modules.transformer import Block
from todd.patches.torch import Sequential
from torch import nn

from .enviroment.enviroment import Observation, TorqueAction
from .utils import continuous_actions_sample

TASK_DIM = 6
SATELLITE_DIM = 35
MAX_TORQUE = 0.2


class Encoder(nn.Module):

    def __init__(
        self,
        *args,
        data_dim: int = TASK_DIM,
        width: int,
        depth: int,
        num_heads: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._num_heads = num_heads

        self._in_projector = nn.Linear(
            data_dim,
            width,
        )
        self._blocks = Sequential(
            *[Block(width=width, num_heads=num_heads) for _ in range(depth)], )
        self._norm = nn.LayerNorm(width)

    def forward(
        self,
        task_data: torch.Tensor,
        tasks_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self._in_projector(task_data)
        attn_mask = torch.einsum(
            '...i, ...j -> ...ij',
            tasks_mask,
            tasks_mask,
        )
        attn_mask = einops.repeat(
            attn_mask,
            'b nt nt_prime -> (b nh) nt nt_prime',
            nh=self._num_heads,
        )
        attn_mask = torch.where(attn_mask, 0, float('-inf'))

        x = self._blocks(x, attn_mask=attn_mask)
        x = self._norm(x)
        return x


class DecoderBlock(Block):

    def __init__(
        self,
        *args,
        width: int,
        num_heads: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, width=width, num_heads=num_heads, **kwargs)
        self._norm3 = nn.LayerNorm(width, 1e-6)
        self._cross_attention = nn.MultiheadAttention(
            width,
            num_heads,
            batch_first=True,
        )

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        *,
        hidden_states: torch.Tensor,
        cross_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = super().forward(x, attn_mask=attn_mask)

        norm = self._norm3(x)
        cross_attention, _ = self._cross_attention(
            norm,
            hidden_states,
            hidden_states,
            need_weights=False,
            attn_mask=cross_attention_mask,
        )
        x = x + cross_attention

        return x


class Decoder(nn.Module):

    def __init__(
        self,
        *args,
        data_dim: int = SATELLITE_DIM,
        width: int,
        depth: int,
        num_heads: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._num_heads = num_heads
        self._in_proj = nn.Linear(data_dim, width)

        self._blocks = Sequential(
            *[
                DecoderBlock(width=width, num_heads=num_heads)
                for _ in range(depth)
            ], )
        self._norm = nn.LayerNorm(width)

    def forward(
        self,
        satellites_data: torch.Tensor,
        satellites_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        tasks_mask: torch.Tensor,
    ) -> torch.Tensor:

        x = self._in_proj(satellites_data)

        attn_mask = torch.einsum(
            '...i, ...j -> ...ij',
            satellites_mask,
            satellites_mask,
        )
        attn_mask = einops.repeat(
            attn_mask,
            'b nt nt_prime -> (b nh) nt nt_prime',
            nh=self._num_heads,
        )
        attn_mask = torch.where(
            attn_mask,
            0.,
            float('-inf'),
        )

        cross_attention_mask = einops.repeat(
            tasks_mask,
            'b nt -> (b nh) ns nt',
            ns=satellites_mask.shape[1],
            nh=self._num_heads,
        )
        cross_attention_mask = torch.where(
            cross_attention_mask,
            0.,
            float('-inf'),
        )

        x = self._blocks(
            x,
            attn_mask=attn_mask,
            hidden_states=hidden_states,
            cross_attention_mask=cross_attention_mask,
        )
        x = self._norm(x)

        return x


class Transformer(nn.Module):

    def __init__(
        self,
        *args,
        encoder_width: int,
        encoder_depth: int,
        encoder_num_heads: int,
        decoder_width: int,
        decoder_depth: int,
        decoder_num_heads: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._encoder = Encoder(
            width=encoder_width,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
        )
        self._decoder = Decoder(
            width=decoder_width,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
        )

    def forward(
        self,
        constellation_data: torch.Tensor,
        constellation_mask: torch.Tensor,
        tasks_data: torch.Tensor,
        tasks_mask: torch.Tensor,
    ) -> torch.Tensor:

        hidden_states = self._encoder(
            tasks_data,
            tasks_mask,
        )
        outputs = self._decoder(
            constellation_data,
            constellation_mask,
            hidden_states,
            tasks_mask,
        )

        return outputs


class GRUDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        time_step: int,
        output_dim: int,
        deterministic: bool,
    ):
        super(GRUDecoder, self).__init__()
        self._hidden_dim = hidden_dim
        self._time_step = time_step
        self._output_dim = output_dim
        self._deterministic = deterministic

        self._gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self._h0 = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self._mu_projection = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, output_dim),
        )
        if not self._deterministic:
            self._sigma_projection = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Linear(4 * hidden_dim, output_dim),
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, ns, _ = x.shape
        x = einops.rearrange(x, 'b ns d -> (b ns) 1 d')

        h0 = einops.repeat(
            self._h0,
            '1 1 d -> 1 (b ns) d',
            b=batch_size,
            ns=ns,
        )
        x = einops.repeat(
            x,
            'b 1 d -> b ts d',
            ts=self._time_step,
        )
        x, _ = self._gru(x, h0)
        mu = self._mu_projection(x)

        mu = einops.rearrange(
            mu,
            '(b ns) ts nd -> b ns ts nd',
            b=batch_size,
            ns=ns,
        )
        if self._deterministic:
            return mu

        sigma = self._sigma_projection(x)
        sigma = einops.rearrange(
            sigma,
            '(b ns) ts nd -> b ns ts nd',
            b=batch_size,
            ns=ns,
        )
        return mu, sigma


class Model(nn.Module):

    def __init__(
        self,
        *args,
        encoder_width: int = 512,
        encoder_depth: int = 12,
        encoder_num_heads: int = 16,
        decoder_width: int = 512,
        decoder_depth: int = 12,
        decoder_num_heads: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._transformer = Transformer(
            encoder_width=encoder_width,
            encoder_depth=encoder_depth,
            encoder_num_heads=encoder_num_heads,
            decoder_width=decoder_width,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
        )

    def forward(self, observation: Observation) -> torch.Tensor:
        batch_size, decoder_seq_len, _ = observation.constellation_data.shape
        constellation_mask = torch.zeros(
            batch_size,
            decoder_seq_len,
            dtype=torch.bool,
        )
        for batch_idx, num_satellite in enumerate(observation.num_satellites):
            constellation_mask[batch_idx, :num_satellite] = True

        logits = self._transformer(
            constellation_data=observation.constellation_data,
            constellation_mask=constellation_mask,
            tasks_data=observation.tasks_data,
            tasks_mask=observation.tasks_visibility,
        )
        return logits


class Actor(Model):

    def __init__(
        self,
        *args,
        decoder_width: int = 512,
        time_step: int,
        deterministic: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, decoder_width=decoder_width, **kwargs)
        self._actions_decoder = GRUDecoder(
            hidden_dim=decoder_width,
            time_step=time_step,
            output_dim=3,
            deterministic=deterministic,
        )
        self._deterministic = deterministic

    def forward(
        self,
        *args,
        **kwargs,
    ) -> TorqueAction:
        logits = super().forward(*args, **kwargs)
        actions_prob = self._actions_decoder(logits)
        if self._deterministic:
            return actions_prob.unbind(dim=-2)

        mu, sigma = actions_prob
        actions = continuous_actions_sample(mu, sigma).unbind(dim=-2)
        return actions


class Critic(Model):

    def __init__(
        self,
        *args,
        decoder_width: int = 512,
        **kwargs,
    ) -> None:
        super().__init__(*args, decoder_width=decoder_width, **kwargs)
        self._mlp = nn.Sequential(
            nn.Linear(decoder_width, 4 * decoder_width),
            nn.GELU(),
            nn.Linear(4 * decoder_width, 1),
        )

    def forward(self, observation: Observation) -> torch.Tensor:
        batch_size, decoder_seq_len, _ = observation.constellation_data.shape
        constellation_mask = torch.zeros(
            batch_size,
            decoder_seq_len,
            dtype=torch.bool,
        )
        for batch_idx, num_satellite in enumerate(observation.num_satellites):
            constellation_mask[batch_idx, :num_satellite] = True

        logits: torch.Tensor = self._transformer(
            constellation_data=observation.constellation_data,
            constellation_mask=constellation_mask,
            tasks_data=observation.tasks_data,
            tasks_mask=observation.tasks_visibility,
        )

        constellation_mask = constellation_mask.unsqueeze(-1)
        sum_logits = (constellation_mask * logits).sum(dim=1)

        sum_count = constellation_mask.int().sum(dim=1)
        mean_state = sum_logits / (sum_count + 1e-8)

        return self._mlp(mean_state).squeeze(-1)  # [b]


class AttitudeControlMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self._input_projection = nn.Linear(input_dim, hidden_dim)
        self._mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, 3),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self._input_projection(x)
        x = self._mlp(x)
        torque = torch.tanh(x) * MAX_TORQUE
        if return_logits:
            return x, torque
        return torque
