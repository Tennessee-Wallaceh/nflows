import numpy as np
import torch
from torch.nn import functional as F

from nflows.transforms.base import InputOutsideDomain
from nflows.utils import torchutils

DEFAULT_MIN_BIN_WIDTH = 1e-4
DEFAULT_MIN_BIN_HEIGHT = 1e-4
DEFAULT_MIN_DERIVATIVE = 1e-3


def unconstrained_rational_quadratic_spline_forward(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    tails="linear",
    left_tail_bound=-1.0,
    right_tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    # Both tails are linear, so we add extra derivatives to either side
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    # compute the spline
    (
        spline_outputs,
        spline_logabsdet,
    ) = forward_rational_quadratic_spline(
        inputs=torch.clamp(inputs, left_tail_bound, right_tail_bound),
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        left=left_tail_bound,
        right=right_tail_bound,
        bottom=left_tail_bound,
        top=right_tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    # only use spline where valid
    inside_interval_mask = (inputs >= left_tail_bound) & (inputs <= right_tail_bound)
    outputs = torch.where(
        inside_interval_mask,
        spline_outputs,
        inputs,
    )
    logabsdet = torch.where(
        inside_interval_mask, spline_logabsdet, torch.zeros_like(inputs)
    )
    return outputs, logabsdet


def unconstrained_rational_quadratic_spline_inverse(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    tails="linear",
    left_tail_bound=-1.0,
    right_tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    # Both tails are linear, so we add extra derivatives to either side
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    # compute the spline
    (
        spline_outputs,
        spline_logabsdet,
    ) = inverse_rational_quadratic_spline(
        inputs=torch.clamp(inputs, left_tail_bound, right_tail_bound),
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        left=left_tail_bound,
        right=right_tail_bound,
        bottom=left_tail_bound,
        top=right_tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    # only use spline where valid
    inside_interval_mask = (inputs >= left_tail_bound) & (inputs <= right_tail_bound)
    outputs = torch.where(
        inside_interval_mask,
        spline_outputs,
        inputs,
    )
    logabsdet = torch.where(
        inside_interval_mask, spline_logabsdet, torch.zeros_like(inputs)
    )
    return outputs, logabsdet


def forward_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    num_bins = unnormalized_widths.shape[-1]

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    widths *= right - left
    cumwidths = F.pad(widths, pad=(1, 0), mode="constant", value=left)
    cumwidths = torch.cumsum(cumwidths, dim=-1)
    cumwidths[..., -1] = right

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_idx = torchutils.searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)

    numerator = input_heights * (
        input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
    )
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        * theta_one_minus_theta
    )
    outputs = input_cumheights + numerator / denominator

    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, logabsdet


def inverse_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    num_bins = unnormalized_widths.shape[-1]

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    widths *= right - left
    cumwidths = F.pad(widths, pad=(1, 0), mode="constant", value=left)
    cumwidths = torch.cumsum(cumwidths, dim=-1)
    cumwidths[..., -1] = right

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_idx = torchutils.searchsorted(cumheights, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    a = (inputs - input_cumheights) * (
        input_derivatives + input_derivatives_plus_one - 2 * input_delta
    ) + input_heights * (input_delta - input_derivatives)
    b = input_heights * input_derivatives - (inputs - input_cumheights) * (
        input_derivatives + input_derivatives_plus_one - 2 * input_delta
    )
    c = -input_delta * (inputs - input_cumheights)

    discriminant = b.pow(2) - 4 * a * c
    assert (discriminant >= 0).all()

    root = (2 * c) / (-b - torch.sqrt(discriminant))
    # root = (- b + torch.sqrt(discriminant)) / (2 * a)
    outputs = root * input_bin_widths + input_cumwidths

    theta_one_minus_theta = root * (1 - root)
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        * theta_one_minus_theta
    )
    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * root.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - root).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, -logabsdet
