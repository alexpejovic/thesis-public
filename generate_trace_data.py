import argparse
import os
from functools import partial
from itertools import batched, product
from pathlib import Path
from typing import get_args

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
from helper import (
    ChannelType,
    TraceDatasetKey,
    compress_traces,
    get_datagen_config,
    get_param_bounds,
    insert_channels,
    set_const_params,
)
from jax import Array
from jaxley import Compartment

jax.config.update("jax_enable_x64", True)


def _create_param_sweep_iter(param_sweep: dict, amps: Array, channel_type: ChannelType):
    if channel_type == "HH":
        return product(*param_sweep.values(), amps)
    elif channel_type == "Pospischil":
        # Note there is a lot of hard coding here. If anything changes in the
        # Pospischil datagen config, this should be changed too most likely.
        # This is to filter out gK and gNa values that are very far from each other
        # max_distance_factor = 30
        # return filter(
        #     lambda x: 1 / max_distance_factor < x[2] / x[3] < max_distance_factor,
        #     product(*param_sweep.values(), amps),
        # )
        return product(*param_sweep.values(), amps)
    else:
        raise ValueError("Could not get valid parameter sweep iterator")


def _gen_data_from_channel(
    channel_type: ChannelType,
    *,
    default_params,
    param_bounds,
    param_consts,
    dt,
    t_max,
    t_filter,
    avg_scale,
    n_param_sweep,
    save_path,
    current_params,
) -> None:
    if channel_type not in get_args(ChannelType):
        ValueError("Attempted to generate channel data with an invalid channel.")

    jax.config.update("jax_enable_x64", True)

    comp = Compartment()

    comp = insert_channels(comp, channel_type)

    param_bounds = get_param_bounds(param_bounds, channel_type)
    if channel_type == "Pospischil-bad":
        comp = set_const_params(param_consts, "Pospischil", comp)
    else:
        comp = set_const_params(param_consts, channel_type, comp)

    comp.record("v")

    @jax.jit
    @partial(jax.vmap, in_axes=(0,))
    def simulate_network(amp_and_params):
        """Simulates the network, returning the voltage traces of the output neurons"""
        params = amp_and_params[0:-1]
        amp = amp_and_params[-1]
        current = jx.step_current(
            i_delay=current_params["delay"],
            i_dur=current_params["duration"],
            i_amp=amp,
            delta_t=dt,
            t_max=t_max,
        )

        param_state = None
        for i, param in enumerate(param_bounds.keys()):
            param_state = comp.data_set(param, params[i], param_state)

        data_stimuli = comp.data_stimulate(current, None)

        return jnp.array(
            jx.integrate(
                comp,
                data_stimuli=data_stimuli,
                param_state=param_state,
                delta_t=dt,
                t_max=t_max,
            ).flatten()
        )

    amps = jnp.arange(
        current_params["amp_min"], current_params["amp_max"], current_params["amp_step"]
    )
    param_sweep = {
        param: jnp.linspace(p0, p1, n_param_sweep, endpoint=True)
        if any([(p0 := param_bounds[param][0]) < 0, (p1 := param_bounds[param][1]) < 0])
        else jnp.logspace(jnp.log10(p0), jnp.log10(p1), n_param_sweep, endpoint=True)
        for param in param_bounds
    }
    n_param_cur_sweep = len(amps) * (n_param_sweep ** len(param_bounds))
    max_batch_size = 2**12 * len(amps)  # needs to be divsible by len(amps)
    batch_size = min(max_batch_size, n_param_cur_sweep)
    n_batches = n_param_cur_sweep // batch_size
    if channel_type == "Pospischil-bad":
        param_sweep_iter = _create_param_sweep_iter(param_sweep, amps, "Pospischil")
    else:
        param_sweep_iter = _create_param_sweep_iter(param_sweep, amps, channel_type)

    dir_path = Path(save_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    comp.init_states(delta_t=dt)

    for i, amp_and_param_sweep in enumerate(batched(param_sweep_iter, batch_size)):
        amp_and_param_sweep = jnp.array(amp_and_param_sweep)
        voltage_array = simulate_network(amp_and_param_sweep)

        voltage_array = compress_traces(voltage_array, t_filter, dt, avg_scale)

        with open(dir_path / f"{channel_type}{i}.npz", "wb") as f:
            np.savez(f, voltage_array)

        with open(dir_path / f"{channel_type}_params{i}.npz", "wb") as f:
            np.savez(f, amp_and_param_sweep[:: len(amps), 0:-1])

        print(f"Finished batch {i + 1} out of (estimated) {n_batches + 1}", flush=True)

    total_batches = i + 1
    combine_temps(channel_type, total_batches, dir_path)

    with open(dir_path / f"{channel_type}_amps.npz", "wb") as f:
        np.savez(f, amps)


def combine_temps(channel_type: ChannelType, total_batches: int, dir_path: Path):
    total_params = []
    # Write to param file
    for i in range(total_batches):
        with open(dir_path / f"{channel_type}_params{i}.npz", "rb") as f:
            param_array = np.load(f)["arr_0"]
            total_params.extend(param_array)
    with open(dir_path / f"{channel_type}_params.npz", "wb") as f:
        np.savez(f, np.array(total_params))

    # First, calculate total shape
    x = 0
    y = 0
    for i in range(total_batches):
        with open(dir_path / f"{channel_type}{i}.npz", "rb") as f:
            v_array = np.load(f)["arr_0"]
            x += v_array.shape[0]
            y = v_array.shape[1]

    final_shape = (x, y)

    fp = np.memmap(
        dir_path / f"{channel_type}.dat",
        dtype="float32",
        mode="w+",
        shape=final_shape,
    )

    # Then, write to .dat file
    x_cur = 0
    for i in range(total_batches):
        with open(dir_path / f"{channel_type}{i}.npz", "rb") as f:
            v_array = np.load(f)["arr_0"]
            x_size = v_array.shape[0]
            fp[x_cur : x_cur + x_size, :] = v_array
            x_cur += x_size
            fp.flush()

    # Delete all temporary files
    for i in range(total_batches):
        os.remove(dir_path / f"{channel_type}{i}.npz")
        os.remove(dir_path / f"{channel_type}_params{i}.npz")


def _sine_wave(
    frequency: float,
    duration: float,
    dt: float,
    amplitude: float,
    phase: float,
    shift: float,
):
    t = jnp.arange(0, duration, dt)
    y = amplitude * jnp.sin(2 * jnp.pi * frequency * t + phase) + shift

    return y


def _gen_nonchannel_data(data_type: TraceDatasetKey):
    if data_type in get_args(ChannelType):
        ValueError("Error generating data.")

    config = get_datagen_config()
    dt = config["dt"]
    t_max = config["t_max"]
    t_filter = config["t_filter"]
    avg_scale = config["avg_scale"]
    dir_path = Path(config["save_path"])

    if data_type == "Sines":
        duration = t_max - (2 * t_filter)
        dt = dt * avg_scale
        n_cycles = jnp.linspace(start=3.0, stop=10.0, num=5)
        amplitudes = jnp.linspace(start=0.5, stop=2.0, num=5)
        phases = jnp.linspace(0, jnp.pi, num=4)
        shifts = jnp.linspace(0.25, 1, num=4)

        traces = jnp.array(
            [
                _sine_wave(n_cycle / duration, duration, dt, amplitude, phase, shift)
                for n_cycle, amplitude, phase, shift in product(
                    n_cycles, amplitudes, phases, shifts
                )
            ]
        )
    elif data_type == "Pospischil-small":
        y_axis = int((t_max - 2 * t_filter) / (dt * avg_scale))
        traces = np.memmap(
            dir_path / "Pospischil.dat",
            dtype="float32",
            mode="r",
        ).reshape((-1, y_axis))
        traces = traces[:400]

    dir_path.mkdir(parents=True, exist_ok=True)

    fp = np.memmap(
        dir_path / f"{data_type}.dat",
        dtype="float32",
        mode="w+",
        shape=traces.shape,
    )
    fp[:] = traces
    fp.flush()


def _parse_args(config: dict) -> dict:
    parser = argparse.ArgumentParser(description="Process cell type input.")
    for k, v in config.items():
        parser.add_argument(f"--{k}", type=type(v), default=v, help=f"Default: {v}")

    parser.add_argument(
        "data_type",
        type=str,
        help=f"Type of data to generate: Possible values: {get_args(TraceDatasetKey)}",
    )
    return parser.parse_args()


def main():
    config = get_datagen_config()
    args = _parse_args(config)

    data_type = args.data_type
    channel_types = get_args(ChannelType)

    if data_type in channel_types:
        _gen_data_from_channel(
            data_type,
            default_params=args.default_params,
            param_bounds=args.param_bounds,
            param_consts=args.param_consts,
            dt=args.dt,
            t_max=args.t_max,
            t_filter=args.t_filter,
            avg_scale=args.avg_scale,
            n_param_sweep=args.n_param_sweep,
            save_path=args.save_path,
            current_params=args.current,
        )
    else:
        _gen_nonchannel_data(data_type)


if __name__ == "__main__":
    main()
