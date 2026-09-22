"""
Analytical TM forward model for a dielectric cylinder with one internal line source.

This module implements the canonical physics-based forward solver used throughout
the inverse-source-em framework. It provides closed-form expressions for the
tangential electric and magnetic fields on the boundary of a dielectric cylinder
excited by a single internal TM line source.

The solver is fully vectorized (NumPy + SciPy) and exposes a unified forward API:

    Esurf(rho_s, phi_s, theta_or_num_angles)
    Hsurf(rho_s, phi_s, theta_or_num_angles)

where the third argument may be:
- int: number of observation angles → uniform grid in [0, 2π)
- float: single observation angle
- array-like: explicit array of observation angles

All returned fields are complex-valued arrays representing the TM boundary fields
(E_z and H_φ components) evaluated at the cylinder surface r = R.

This analytical solver is used for:
- validating surrogate models
- generating surrogate training datasets
- benchmarking physical consistency
- providing ground-truth forward fields for regression pipelines
"""

import numpy as np
import math
from scipy.special import jv, jvp, hankel1, h1vp


class PhysicsTM:
    """
    Analytical TM forward solver for cylindrical scattering with one internal line source.

    This class computes the boundary electric and magnetic fields on a dielectric
    cylinder of radius R, excited by a single TM line source located at
    (rho_s, phi_s). The solution is expressed as a truncated cylindrical-harmonic
    expansion using Bessel and Hankel functions.

    Parameters
    ----------
    R : float, optional
        Cylinder radius.
    OMEGA : float, optional
        Angular frequency ω.
    EPS0 : float, optional
        Permittivity of the exterior medium.
    MU0 : float, optional
        Permeability of the exterior medium.
    EPS1 : float, optional
        Permittivity of the interior medium.
    MU1 : float, optional
        Permeability of the interior medium.
    N : int, optional
        Truncation order of the cylindrical-harmonic expansion.
        Total number of modes is 2N + 1.
    I0 : float, optional
        Source strength of the internal line source.

    Notes
    -----
    - All computations are fully vectorized.
    - Returned fields are complex-valued.
    - Observation angles θ are always interpreted in radians.
    - The solver is used as the ground-truth forward model throughout the project.
    """

    def __init__(self,
                 R=1.0,
                 OMEGA=4.0,
                 EPS0=1.0,
                 MU0=1.0,
                 EPS1=1.3225,
                 MU1=1.0,
                 N=20,
                 I0=1.0):

        self.R     = float(R)
        self.OMEGA = float(OMEGA)
        self.EPS0  = float(EPS0)
        self.MU0   = float(MU0)
        self.EPS1  = float(EPS1)
        self.MU1   = float(MU1)
        self.N     = int(N)
        self.I0    = float(I0)

        self.K0   = self.OMEGA * math.sqrt(self.EPS0 * self.MU0)
        self.ETA  = math.sqrt(self.EPS1 * self.MU1)
        self.K1   = self.K0 * self.ETA
        self.A    = -self.OMEGA * self.MU0 * self.MU1 / 4.0

        self.nvals = np.arange(-self.N, self.N + 1, dtype=np.int64)

    def make_obs(self, num_angles):
        """
        Generate a uniform grid of observation angles in [0, 2π).

        Parameters
        ----------
        num_angles : int
            Number of observation angles.

        Returns
        -------
        ndarray of shape (num_angles,)
            Uniformly spaced angles in radians.
        """
        return np.linspace(0.0, 2*np.pi, int(num_angles), endpoint=False)

    def alpha_n_vec(self, rho_s):
        """
        Compute the cylindrical-harmonic expansion coefficients α_n for a source
        located at radial coordinate rho_s.

        Parameters
        ----------
        rho_s : float
            Radial coordinate of the internal line source.

        Returns
        -------
        ndarray of shape (2N+1,)
            Complex-valued expansion coefficients α_n.
        """
        rho_s = float(rho_s)
        n = self.nvals
        R = self.R

        num = -2j * jv(n, self.K1 * rho_s)

        Jn_prime_K1R = jvp(n, self.K1 * R, 1)
        Hn_K0R       = hankel1(n, self.K0 * R)
        Hn_prime_K0R = h1vp(n, self.K0 * R, 1)
        Jn_K1R       = jv(n, self.K1 * R)

        den_inner = self.ETA * Jn_prime_K1R * Hn_K0R - self.MU1 * Hn_prime_K0R * Jn_K1R
        den = self.K0 * R * np.pi * den_inner

        return num / den

    def Esurf_theta(self, rho_s, phi_s, theta_array):
        """
        Compute the boundary electric field E_surf(θ) for an explicit array of angles.

        Parameters
        ----------
        rho_s : float
            Radial coordinate of the source.
        phi_s : float
            Angular coordinate of the source.
        theta_array : array-like
            Observation angles in radians.

        Returns
        -------
        ndarray of complex128, shape (len(theta_array),)
            Complex electric field values at r = R.
        """
        rho_s = float(rho_s)
        phi_s = float(phi_s)

        obs = np.asarray(theta_array, dtype=float)
        n = self.nvals

        coeffs = self.alpha_n_vec(rho_s)
        H = hankel1(n, self.K0 * self.R).astype(np.complex128)

        phase = np.exp(1j * n[:, None] * (obs[None, :] - phi_s))
        field = np.sum(coeffs[:, None] * H[:, None] * phase, axis=0)

        return np.complex128(self.A * self.I0) * field

    def Hsurf_theta(self, rho_s, phi_s, theta_array):
        """
        Compute the boundary magnetic field H_surf(θ) for an explicit array of angles.

        Parameters
        ----------
        rho_s : float
            Radial coordinate of the source.
        phi_s : float
            Angular coordinate of the source.
        theta_array : array-like
            Observation angles in radians.

        Returns
        -------
        ndarray of complex128, shape (len(theta_array),)
            Complex magnetic field values at r = R.
        """
        rho_s = float(rho_s)
        phi_s = float(phi_s)

        obs = np.asarray(theta_array, dtype=float)
        n = self.nvals

        coeffs = self.alpha_n_vec(rho_s)
        Hprime = h1vp(n, self.K0 * self.R, 1).astype(np.complex128)

        phase = np.exp(1j * n[:, None] * (obs[None, :] - phi_s))
        field = np.sum(coeffs[:, None] * Hprime[:, None] * phase, axis=0)

        return np.complex128(self.A * self.I0 / self.MU0) * field

    def Esurf(self, rho_s, phi_s, theta_or_num_angles=720):
        """
        Unified forward API for the boundary electric field.

        Parameters
        ----------
        rho_s : float
            Radial coordinate of the source.
        phi_s : float
            Angular coordinate of the source.
        theta_or_num_angles : int, float, or array-like
            - int: number of observation angles → uniform grid
            - float: single angle
            - array-like: explicit angle array

        Returns
        -------
        complex or ndarray of complex
            Electric field values at r = R.
        """
        if isinstance(theta_or_num_angles, (int, np.integer)):
            obs = self.make_obs(int(theta_or_num_angles))
            return self.Esurf_theta(rho_s, phi_s, obs)

        if np.isscalar(theta_or_num_angles):
            obs = np.array([theta_or_num_angles], dtype=float)
            out = self.Esurf_theta(rho_s, phi_s, obs)
            return out[0]

        obs = np.asarray(theta_or_num_angles, dtype=float)
        return self.Esurf_theta(rho_s, phi_s, obs)

    def Hsurf(self, rho_s, phi_s, theta_or_num_angles=720):
        """
        Unified forward API for the boundary magnetic field.

        Parameters
        ----------
        rho_s : float
            Radial coordinate of the source.
        phi_s : float
            Angular coordinate of the source.
        theta_or_num_angles : int, float, or array-like
            - int: number of observation angles → uniform grid
            - float: single angle
            - array-like: explicit angle array

        Returns
        -------
        complex or ndarray of complex
            Magnetic field values at r = R.
        """
        if isinstance(theta_or_num_angles, (int, np.integer)):
            obs = self.make_obs(int(theta_or_num_angles))
            return self.Hsurf_theta(rho_s, phi_s, obs)

        if np.isscalar(theta_or_num_angles):
            obs = np.array([theta_or_num_angles], dtype=float)
            out = self.Hsurf_theta(rho_s, phi_s, obs)
            return out[0]

        obs = np.asarray(theta_or_num_angles, dtype=float)
        return self.Hsurf_theta(rho_s, phi_s, obs)
