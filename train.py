
import jax
import jax.numpy as jnp
from helper import (
    TraceDatasetKey,
    TraceNormalizer,
    get_trace_normal_and_denormal,
)
from jax import Array


def fft_on_trace(trace: Array) -> Array:
    return jnp.fft.rfft(trace)


def get_stim_window(datagen_config: dict, trace_len: int) -> tuple[int, int]:
    """Sample indices ``[start, end)`` of a trace where the stimulus current is on.

    Traces are trimmed by ``t_filter`` on both ends and averaged by ``avg_scale``
    (see ``compress_traces``), so the current onset at ``delay`` ms lands at
    sample ``(delay - t_filter) / (dt * avg_scale)``.
    """
    eff_dt = datagen_config["dt"] * datagen_config["avg_scale"]
    t_filter = datagen_config["t_filter"]
    current = datagen_config["current"]

    start = int((current["delay"] - t_filter) / eff_dt)
    end = int((current["delay"] + current["duration"] - t_filter) / eff_dt)

    return max(start, 0), min(end, trace_len)


def get_mask_width(mask_width_ms: float, datagen_config: dict) -> int:
    """Mask width in ms converted to a number of (compressed) trace samples."""
    eff_dt = datagen_config["dt"] * datagen_config["avg_scale"]
    return max(int(round(mask_width_ms / eff_dt)), 1)


def make_keep_mask(
    key,
    n_traces: int,
    trace_len: int,
    stim_start: int,
    stim_end: int,
    mask_width: int,
) -> Array:
    """A (n_traces, trace_len) 0/1 mask, one random masked-out window per trace.

    Each window has a fixed width of ``mask_width`` samples and is placed
    uniformly at random so that it lies fully inside ``[stim_start, stim_end)``.
    Entries are 1.0 where the trace is kept and 0.0 where it is masked out.
    """
    starts = jax.random.randint(
        key, (n_traces, 1), stim_start, stim_end - mask_width + 1
    )
    positions = jnp.arange(trace_len)[None, :]

    return ((positions < starts) | (positions >= starts + mask_width)).astype(
        jnp.float32
    )


def dataloader(
    voltage_traces,
    batch_size,
    key,
    normalizer_type: TraceNormalizer,
    indices: Array | None = None,
    shuffle: bool = True,
    data_type: TraceDatasetKey | None = None,
):
    if indices is None:
        dataset_size = voltage_traces.shape[0]
        indices = jnp.arange(dataset_size)
    else:
        dataset_size = len(indices)
    normalizer, _ = get_trace_normal_and_denormal(normalizer_type, data_type)
    normalizer_vmap = jax.vmap(normalizer)
    if shuffle:
        perm = jax.random.permutation(key, indices)
    else:
        perm = indices
    start = 0
    end = batch_size

    while end <= dataset_size:
        batch_perm = perm[start:end]
        yield normalizer_vmap(voltage_traces[batch_perm])
        start = end
        end = start + batch_size


def split_voltage_data(voltage_data, train_split, data_key):
    dataset_size = voltage_data.shape[0]
    indices = jax.random.permutation(data_key, jnp.arange(dataset_size))
    bound = int(dataset_size * train_split)
    train_indices = indices[:bound]
    val_indices = indices[bound:]
    return train_indices, val_indices
