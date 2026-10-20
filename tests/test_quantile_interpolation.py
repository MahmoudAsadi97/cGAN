"""Synthetic regressions for the shared empirical inverse-quantile primitive."""
import numpy as np

from fluvius_cgan.calibration import fit
from fluvius_cgan.copula import _inverse_quantiles, fit_conditional_copula


def _compressed_tied_source():
    rng = np.random.default_rng(7312)
    source = np.repeat(rng.normal(size=(8, 24)), 512, axis=0)
    uniform = np.broadcast_to(np.linspace(0., 1., 1025)[:, None], (1025, 24))
    return source, uniform


def test_compressing_float64_plateaus_preserves_ties_and_monotonicity():
    source, uniform = _compressed_tied_source()
    sorted_source = np.sort(source, axis=0)
    result = _inverse_quantiles(uniform, sorted_source)
    positions = uniform * (len(source) - 1)
    lower = np.minimum(positions.astype(np.int64), len(source) - 2)
    columns = np.arange(source.shape[1])
    left, right = sorted_source[lower, columns], sorted_source[lower + 1, columns]
    tied = left == right
    np.testing.assert_array_equal(result[tied], left[tied])
    assert np.all(result[1:] >= result[:-1])
    np.testing.assert_array_equal(result[[0, -1]], sorted_source[[0, -1]])


def test_calibration_accepts_compressed_float64_source_atoms():
    source, _ = _compressed_tied_source()
    train = np.broadcast_to(np.array([-2., 3.])[:, None], (2, 24)).copy()
    model = fit(train, np.zeros(2, dtype=np.int64), source,
                np.zeros(len(source), dtype=np.int64), quantile_knots=1025)
    mapped = model.transform(np.sort(source, axis=0), np.zeros(len(source), dtype=np.int64))
    assert np.isfinite(mapped).all()
    assert np.all(mapped[1:] >= mapped[:-1])
    assert np.all((mapped >= -2.) & (mapped <= 3.))


def test_copula_float64_tie_knots_save_and_reload(tmp_path):
    source, _ = _compressed_tied_source()
    model = fit_conditional_copula(source, np.zeros(len(source), dtype=np.int64),
                                   quantile_knots=1025)
    path = tmp_path / "copula.npz"
    model.save(path)
    restored = type(model).load(path)
    np.testing.assert_array_equal(restored.models[0].quantiles, model.models[0].quantiles)


def test_interpolation_remains_finite_bounded_and_monotone_at_float64_extremes():
    maximum = np.finfo(np.float64).max
    tiny = np.nextafter(0., 1.)
    pairs = np.array([
        [-maximum, maximum], [-maximum, -tiny], [tiny, maximum],
        [np.nextafter(maximum, 0.), maximum], [-maximum, -np.nextafter(maximum, 0.)],
        [-tiny, tiny], [-maximum, 0.], [0., maximum],
        [.12345678912345678, .12345678912345678],
        [-.12345678912345678, -.12345678912345678], [0., 0.],
    ]).T
    probabilities = np.unique(np.r_[-1., np.nextafter(0., 1.), np.linspace(0., 1., 4097),
                                   np.nextafter(1., 0.), 2.])
    uniform = np.broadcast_to(probabilities[:, None], (len(probabilities), pairs.shape[1]))
    with np.errstate(over="raise", invalid="raise"):
        values = _inverse_quantiles(uniform, pairs)
    assert np.isfinite(values).all()
    assert np.all(values[1:] >= values[:-1])
    assert np.all((values >= pairs[0]) & (values <= pairs[1]))
    np.testing.assert_array_equal(values[probabilities <= 0.], np.broadcast_to(pairs[0], values[probabilities <= 0.].shape))
    np.testing.assert_array_equal(values[probabilities >= 1.], np.broadcast_to(pairs[1], values[probabilities >= 1.].shape))
    np.testing.assert_array_equal(values[:, 8:], np.broadcast_to(pairs[0, 8:], values[:, 8:].shape))
    assert values[np.flatnonzero(probabilities == .5)[0], 0] == 0.


def test_random_finite_knots_are_monotone_across_segment_boundaries():
    rng = np.random.default_rng(20261020)
    mantissa = rng.uniform(-1., 1., size=(33, 64))
    exponents = rng.integers(-1020, 1020, size=mantissa.shape)
    knots = np.sort(np.ldexp(mantissa, exponents), axis=0)
    probabilities = np.unique(np.r_[np.linspace(0., 1., 8193),
                                   np.nextafter(np.linspace(0., 1., 33), 0.),
                                   np.nextafter(np.linspace(0., 1., 33), 1.)])
    uniform = np.broadcast_to(probabilities[:, None], (len(probabilities), knots.shape[1]))
    with np.errstate(over="raise", invalid="raise"):
        values = _inverse_quantiles(uniform, knots)
    assert np.isfinite(values).all()
    assert np.all(values[1:] >= values[:-1])
    assert np.all((values >= knots[0]) & (values <= knots[-1]))
    np.testing.assert_array_equal(values[np.searchsorted(probabilities, np.linspace(0., 1., 33))], knots)


def test_linear_quantile_weights_and_exact_knots_have_manual_oracle():
    knots = np.array([[-4., 2., -14.], [4., 10., -6.], [12., 18., 2.]])
    uniform = np.broadcast_to(np.arange(9)[:, None] / 8, (9, 3))
    expected = knots[0] + np.arange(9)[:, None] * 2
    np.testing.assert_array_equal(_inverse_quantiles(uniform, knots), expected)


def test_negative_interval_does_not_undershoot_at_tiny_positive_probability():
    # A right anchor rounded the second output one ULP below the first knot.
    # Preserve that exact case; a tolerance would hide the support violation.
    a, b = -(1.5 + 2**-52), -(3 * 2**-53)
    knots = np.array([[a], [b]], dtype=np.float64)
    probabilities = np.array([0., np.nextafter(0., 1.), 1e-30,
                              np.nextafter(.5, 0.), .5, np.nextafter(.5, 1.),
                              np.nextafter(1., 0.), 1.], dtype=np.float64)[:, None]
    values = _inverse_quantiles(probabilities, knots)
    np.testing.assert_array_equal(values[:3], np.full((3, 1), a))
    assert np.all(values[1:] >= values[:-1])
    assert np.all((values >= a) & (values <= b))
    np.testing.assert_array_equal(values[[0, -1]], knots)
