from utilities.models.low_level_controllers import LowLevelControllerBase

import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import torch


def roots(p):
    """ p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n] """
    d = p.shape[0]
    if p.ndim == 3:
        c = p.view(d, -1)
    else:
        c = p
    n = c.shape[1]

    found_roots = torch.zeros(n, d - 1, dtype=torch.complex64).to(p.device)

    # identify leading zeros
    lz = c[0] == 0
    if d == 2:
        found_roots[~lz] = -( c[1, ~lz, None] / c[0, ~lz, None]).to(torch.complex64)
    else:
        if torch.any(lz):
            found_roots[lz, 1:] = roots(c[1:, lz])
            found_roots[lz, 0] = float('nan')  # For usability afterwards
        if torch.any(~lz):
            new_p = c[:, ~lz] / c[0, ~lz]

            new_n = torch.sum(~lz)
            e = torch.cat((torch.zeros(1, d-2), torch.eye(d-2)))[None].repeat(new_n, 1, 1).to(p.device)
            comp = torch.cat((e, -new_p[1:].flip(0).T[..., None]), dim=-1)
            # Float 64 needed to avoid numerical errors
            found_roots[~lz] = torch.linalg.eigvals(comp.to(torch.float64)).to(torch.complex64)

    if p.ndim == 3:
        found_roots = found_roots.T.reshape(d-1, *p.shape[1:])
    return found_roots


def calculate_polynomial_trajectory(t, p0, v0, a0, j):
    at = a0 + j * t
    vt = v0 + a0 * t + 0.5 * j * t ** 2
    pt = p0 + v0 * t + 0.5 * a0 * t ** 2 + (1 / 6) * j * t ** 3
    return pt, vt, at


@torch.jit.script
def evaluate_trajectory(t, j, p0, v0, a0):
    p, v, a = p0, v0, a0
    for i in range(t.shape[0]): # Iterates through phases
        p, v, a = calculate_polynomial_trajectory(t[i], p, v, a, j[i])
    return p, v, a


@torch.jit.script
def phase_cumsum_deterministic(t: torch.Tensor) -> torch.Tensor:
    """ Deterministic replacement for cumsum. """
    # t: [7, N, D]

    e0 = t[0]
    e1 = e0 + t[1]
    e2 = e1 + t[2]
    e3 = e2 + t[3]
    e4 = e3 + t[4]
    e5 = e4 + t[5]
    e6 = e5 + t[6]

    # [N, D, 7]
    return torch.stack((e0, e1, e2, e3, e4, e5, e6), dim=-1)


@torch.jit.script
def evaluate_trajectory_stepwise(steps: int, dt: float, t: torch.Tensor, j: torch.Tensor, p0: torch.Tensor,
                                 v0: torch.Tensor, a0: torch.Tensor) -> torch.Tensor:
    """Vectorized trajectory sampling at dt, 2*dt, ..., steps*dt. """
    phases = t.shape[0]
    num_envs = p0.shape[0]
    dofs = p0.shape[1]

    # State at the beginning of every phase, plus final state.
    ps = torch.empty((phases + 1, num_envs, dofs), device=p0.device, dtype=p0.dtype)
    vs = torch.empty_like(ps)
    accs = torch.empty_like(ps)
    ps[0] = p0
    vs[0] = v0
    accs[0] = a0

    p, v, a = p0, v0, a0
    for i in range(phases):
        p, v, a = calculate_polynomial_trajectory(t[i], p, v, a, j[i])
        ps[i + 1] = p
        vs[i + 1] = v
        accs[i + 1] = a

    # [N, D, 7]
    phase_ends = phase_cumsum_deterministic(t)
    phase_starts = torch.cat((torch.zeros_like(phase_ends[..., :1]), phase_ends), dim=-1)
    ps = ps.permute(1, 2, 0).contiguous()
    vs = vs.permute(1, 2, 0).contiguous()
    accs = accs.permute(1, 2, 0).contiguous()

    # Extra zero-jerk phase holds the final state after trajectory completion.
    j_ext = torch.cat((j, torch.zeros_like(j[:1])), dim=0).permute(1, 2, 0).contiguous()

    sample_t = torch.arange(1, steps + 1, device=p0.device, dtype=p0.dtype) * dt
    sample_t = sample_t.view(1, 1, -1).expand(num_envs, dofs, steps)
    sample_t = torch.minimum(sample_t, phase_ends[..., -1:])

    # right=True moves an exact phase-boundary sample to the next phase
    phase_idx = torch.searchsorted(phase_ends, sample_t, right=True)
    phase_idx = torch.clamp(phase_idx, max=phases)

    p = torch.gather(ps, 2, phase_idx)
    v = torch.gather(vs, 2, phase_idx)
    a = torch.gather(accs, 2, phase_idx)
    ji = torch.gather(j_ext, 2, phase_idx)
    local_t = sample_t - torch.gather(phase_starts, 2, phase_idx)
    local_t2 = local_t * local_t

    pt = p + v * local_t + 0.5 * a * local_t2 + (1.0 / 6.0) * ji * local_t2 * local_t
    vt = v + a * local_t + 0.5 * ji * local_t2
    at = a + ji * local_t
    return torch.stack((pt, vt, at)).permute(3, 0, 1, 2)


class SCurve(LowLevelControllerBase):

    def __init__(self, num_envs, dof, device='cpu'):
        LowLevelControllerBase.__init__(
            self, control_mode='velocity', target_mode='position'
        )

        # Placeholders
        self.p0 = torch.zeros((num_envs, dof), device=device)
        self.pT = torch.zeros((num_envs, dof), device=device)
        self.v0 = torch.zeros((num_envs, dof), device=device)
        self.a0 = torch.zeros((num_envs, dof), device=device)

        self.vmax = 0.0
        self.amax = 0.0
        self.jmax = 0.0

        self.t = torch.zeros((7, num_envs, dof), device=device)
        self.j = torch.zeros((7, num_envs, dof), device=device)

        self.t_stop = torch.zeros((3, num_envs, dof), device=device)
        self.j_stop = torch.zeros((3, num_envs, dof), device=device)

        self.step = 0

    @property
    def target(self):
        return self.pT

    def set_limits(self, vmax, amax, jmax):
        self.vmax = vmax
        self.amax = amax
        self.jmax = jmax

        # precalculate for efficiency
        self.amax2 = self.amax * self.amax # TODO: Faster than torch.pow or **2
        self.jmax2 = self.jmax * self.jmax

        self.jmax3 = self.jmax2 * self.jmax
        self.amax3 = self.amax2 * self.amax

        self.amax4 = self.amax3 * self.amax
        self.jmax4 = self.jmax3 * self.jmax

    def initialize(self, p0, pT, v0, a0):
        self.p0[:] = p0
        self.pT[:] = pT
        self.v0[:] = v0
        self.a0[:] = a0

        # precalculate for efficiency
        self.v02 = v0 * v0
        self.a02 = a0 * a0

        self.v03 = self.v02 * v0
        self.a03 = self.a02 * a0

        self.a04 = self.a03 * a0

        self.t[:] = 0
        self.j[:] = 0
        self.t_stop[:] = 0
        self.j_stop[:] = 0

        self.pStop = self.compute_fast_stop()
        self.s = self.compute_s()
        self.step = 0

    def compute_fast_stop(self):
        # per robot and joint: s = sign(a0) if a0 != 0 else sign(v0)
        # s defines sign of applied jerks
        s = torch.where(self.a0 != 0, torch.sign(self.a0), torch.sign(self.v0))

        # Time of constant jerk to zero acceleration/deceleration
        tj0 = torch.abs(self.a0 / self.jmax)

        # calculate peak velocity vp
        vp = self.v0 + s * 0.5 * self.a02 / self.jmax

        # Time of constant jerk to peak acceleration/deceleration
        tja = torch.zeros_like(tj0)
        # Time of constant acceleration/deceleration
        ta = torch.zeros_like(tj0)

        # Compute case 1: Any of conditions c1, c2 or c3
        c1 = torch.sign(self.v0) == torch.sign(self.a0)
        c2 = self.a0 == 0
        c3 = torch.sign(self.v0) != torch.sign(vp)
        case1 = c1 | c2 | c3

        tja[case1] = torch.sqrt(torch.abs(vp) / self.jmax)[case1]

        strict_case1 = case1 & ((self.jmax * tja) > self.amax)
        tja[strict_case1] = self.amax[strict_case1] / self.jmax[strict_case1]
        ta[strict_case1] = (torch.abs(vp) / self.amax - tja)[strict_case1]

        # Compute case2:
        case2 = ~case1 & (torch.sign(self.v0) == torch.sign(vp))
        if torch.any(case2):
            p1 = s[case2] * vp[case2] / self.jmax[case2]
            p2 = 2 * s[case2] * self.a0[case2] / self.jmax[case2]
            p3 = torch.ones_like(p1)
            tja_candidates = roots(torch.stack((p3, p2, p1)))

            # Remove negative and complex roots since time must be a real positive number
            # invalid roots marked as 'inf' to maintain tensor structure
            tja_candidates[tja_candidates.imag.abs() > 1e-5] = float('inf')
            tja_candidates[tja_candidates.real < 0] = float('inf')
            tja_candidates[torch.isnan(tja_candidates.real)] = float('inf')
            tja[case2] = tja_candidates.real.min(dim=-1).values

            strict_case2 = torch.minimum((self.jmax * tja + s * self.a0) > self.amax, case2)
            tja[strict_case2] = ((self.amax - (s*self.a0)) / self.jmax)[strict_case2]
            ta[strict_case2] = (torch.abs(vp) / self.amax - tja)[strict_case2]

        self.t_stop[0, case1] = tja[case1] + tj0[case1]
        self.j_stop[0, case1] = - s[case1] * self.jmax[case1]
        self.t_stop[0, case2] = tja[case2]
        self.j_stop[0, case2] = s[case2] * self.jmax[case2]

        self.t_stop[1] = ta
        self.j_stop[1] = torch.zeros_like(ta)

        self.t_stop[2, case1] = tja[case1]
        self.j_stop[2, case1] = s[case1] * self.jmax[case1]
        self.t_stop[2, case2] = tja[case2] + tj0[case2]
        self.j_stop[2, case2] = - s[case2] * self.jmax[case2]

        return evaluate_trajectory(self.t_stop, self.j_stop, self.p0, self.v0, self.a0)[0]

    def evaluate_trajectory(self, t, j, p0, v0, a0):
        p, v, a = p0, v0, a0
        for i in range(t.shape[0]): # Iterates through phases
            p, v, a = calculate_polynomial_trajectory(t[i], p, v, a, j[i])
        return p, v, a

    def evaluate_t_steps(self, steps, dt):
        return evaluate_trajectory_stepwise(steps, dt, self.t, self.j, self.p0, self.v0, self.a0)

    def evaluate_at_t(self, t):
        # Assuming t as scalar
        p, v, a, j = self.p0.clone(), self.v0.clone(), self.a0.clone(), self.j[0].clone()
        final_t = torch.ones_like(p) * t
        f = torch.ones_like(final_t, dtype=torch.bool)
        for i in range(self.t.shape[0]):
            f = f & (self.t[i] < final_t)
            p[f], v[f], a[f] = calculate_polynomial_trajectory(
                self.t[i, f], p[f], v[f], a[f], self.j[i, f]
            )
            if i < self.t.shape[0]-1:
                j[f] = self.j[i+1, f].clone()
            else:  # In case t is larger than time to reach goal
                j[f] = 0.0
            final_t[f] -= self.t[i, f]

        final_t2 = final_t * final_t
        p = p + v * final_t + 0.5 * a * final_t2 + (1 / 6) * j * final_t2 * final_t
        v = v + a * final_t + 0.5 * j * final_t2
        a = a + j * final_t
        return p, v, a

    def compute_s(self):
        return torch.sign(self.pT - self.pStop)

    def compute_zero_cruise_profile(self):
        # 'cruise' refers to constant velocity phase
        vp = self.s * self.vmax

        # estimate peak acceleration ap
        v_isclose = torch.isclose(vp, self.v0, atol=1e-4)  # Check for small numerical errors to avoid NaN
        self.v0[v_isclose] = vp[v_isclose]
        ap = self.s * torch.sqrt((vp - self.v0) * self.s * self.jmax + 0.5 * self.a02)
        t2 = torch.zeros_like(ap)
        c1 = torch.abs(ap) > self.amax
        ap[c1] = self.s[c1] * self.amax[c1]
        t2[c1] = ((vp - self.v0 + (0.5 * self.a02 - ap*ap) / (self.s * self.jmax)) / ap)[c1]

        # estimate peak deceleration dp
        dp = - self.s * torch.sqrt(torch.abs(vp) * self.jmax)
        t6 = torch.zeros_like(dp)
        c2 = torch.abs(dp) > self.amax
        dp[c2] = -self.s[c2] * self.amax[c2]
        t6[c2] = torch.abs(vp / dp)[c2] - torch.abs(dp / self.jmax)[c2]
        t5 = torch.abs(dp) / self.jmax

        self.t[0] = torch.abs(ap - self.a0) / self.jmax
        self.j[0] = self.s * self.jmax
        self.t[1] = t2
        self.t[2] = torch.abs(ap) / self.jmax
        self.j[2] = -self.s * self.jmax

        self.t[4] = t5
        self.j[4] = -self.s * self.jmax
        self.t[5] = t6
        self.t[6] = t5
        self.j[6] = self.s * self.jmax

        return evaluate_trajectory(self.t, self.j, self.p0, self.v0, self.a0)[0]

    def compute_profile(self):
        stop_case = torch.isclose(self.pStop, self.pT, atol=1e-3)

        # calculate final trajectory position tpT
        tpT = self.compute_zero_cruise_profile()
        vp = self.s * self.vmax  # peak velocity
        t4 = (self.pT - tpT) / vp
        cruise_case = t4 >= 0
        if torch.any(cruise_case):
            self.t[3, cruise_case] = t4[cruise_case]

        others = ~stop_case & ~cruise_case
        overshoot = self.s * torch.sign(tpT - self.pT) == 1
        reduction = others & overshoot
        tpT = self.compute_profile_type_reduction(reduction)

        adjust = ~torch.isclose(tpT, self.pT, atol=1e-3)
        t2 = self.t[1]
        t6 = self.t[5]
        ww_filter = adjust & (t2 == 0) & (t6 == 0) & others
        tw_filter = adjust & (t2 > 0) & (t6 == 0) & others
        wt_filter = adjust & (t2 == 0) & (t6 > 0) & others
        tt_filter = adjust & (t2 > 0) & (t6 > 0) & others
        if torch.any(ww_filter):
            self.optimize_ww_type(ww_filter, tpT)
        if torch.any(tw_filter):
            self.optimize_tw_type(tw_filter, tpT)
        if torch.any(wt_filter):
            self.optimize_wt_type(wt_filter, tpT)
        if torch.any(tt_filter):
            self.optimize_tt_type(tt_filter, tpT)

        # unsolvable trajectories go through unchanged
        # Apply stop trajectories in these cases instead
        tpT = evaluate_trajectory(self.t, self.j, self.p0, self.v0, self.a0)[0]
        invalid = ~torch.isclose(tpT, self.pT, atol=1e-3)
        stop_case = stop_case | invalid
        if torch.any(stop_case):
            self.t[:3, stop_case] = self.t_stop[:, stop_case]
            self.j[:3, stop_case] = self.j_stop[:, stop_case]
            self.t[3:, stop_case] = 0
            self.j[3:, stop_case] = 0

    def compute_profile_type_reduction(self, filter):
        new_t = self.t.clone()
        t1, t2, t3, t4, t5, t6, t7 = self.t

        # Case 1: WW-Profiles -> Do Nothing
        ww_profiles = filter & (t2 == 0) & (t6 == 0)
        new_filter = filter & ~ww_profiles

        # Case 2: TT-Profiles
        tt_profiles = new_filter & (t2 > 0) & (t6 > 0)
        if tt_profiles.any():
            dt = torch.min(t2, t6)[tt_profiles]
            new_t[1, tt_profiles] -= dt
            new_t[5, tt_profiles] -= dt

        # Case 3: WT-Profiles
        wt_profiles = new_filter & (t2 == 0) & (t6 > 0)
        if wt_profiles.any():
            area_w_max = self.jmax * t3*t3
            area_w_max = torch.where(t1 < t3, area_w_max-(0.5*self.a02)/self.jmax, area_w_max)
            area_t_max = t6 * self.amax
            cutable = wt_profiles & (area_w_max > area_t_max)
            new_t[5, cutable] = 0
            a1 = self.a0 + self.j[0] * t1
            c = (a1*a1 - area_t_max * self.jmax)[cutable]
            dt = (a1[cutable].abs() - torch.sqrt(c)) / self.jmax[cutable]
            new_t[4, cutable] -= dt
            new_t[6, cutable] -= dt
            new_filter[new_filter & wt_profiles & ~cutable] = False

        # Case 4: TW-Profiles
        tw_profiles = new_filter & (t2 > 0) & (t6 == 0)
        if tw_profiles.any():
            a5 = self.j[4] * t5
            area_w_max = torch.abs(t5 * a5)
            area_t_max = t2 * self.amax
            cutable = tw_profiles & (area_w_max > area_t_max)
            new_t[1, cutable] = 0
            c = (area_w_max - area_t_max)[cutable]
            dt = torch.sqrt(c) / self.jmax[cutable]
            new_t[4, cutable] = dt
            new_t[6, cutable] = dt
            new_filter[new_filter & tw_profiles & ~cutable] = False

        new_pT = evaluate_trajectory(new_t, self.j, self.p0, self.v0, self.a0)[0]
        overshoot = self.s * torch.sign(new_pT - self.pT) == 1
        new_filter = torch.minimum(new_filter, overshoot)
        if torch.any(new_filter):
            self.t[:, new_filter] = new_t[:, new_filter]
            new_pT[new_filter] = self.compute_profile_type_reduction(new_filter)[new_filter]

        return new_pT

    def optimize_ww_type(self, filter, best_tpT):
        s = self.s
        p0, pT = self.p0, self.pT
        p02, pT2 = p0 * p0, pT * pT
        L = pT - p0
        v0, a0, jm = self.v0, self.a0, self.jmax
        v02, a02, jm2 = self.v02, self.a02, self.jmax2
        v03, a03, jm3 = self.v03, self.a03, self.jmax3
        a04, jm4 = self.a04, self.jmax4
        a05 = a04 * a0
        a06 = a05 * a0
        condition = self.j[0] != self.j[2]
        da = torch.where(condition, torch.ones_like(p0), -torch.ones_like(p0))

        c4 = (-18*a02 + 36*s*v0*jm)*(1.0+da)
        c3 = (72*s*v0*a0*jm - 72*jm2*L - 48*a03)*(1.0+da)
        c2 = (-27*a04 - 216*a0*jm2*L + 36*v02*jm2 - 36*s*v0*a02*jm)*(1.0+da)
        c1 = (-144*s*v0*jm3*L - 144*a02*jm2*L - 72*v02*a0*jm2 - 6*a05 - 24*s*v0*a03*jm)*(1.0+da)
        c0 = - 6*s*v0*a04*jm - 144*jm4*p0*pT - 144*s*v0*a0*jm3*L - 72*s*v03*jm3\
             - 48*a03*jm2*L + 72*jm4*(pT2+p02) - 36*v02*a02*jm2 - a06

        rts = roots(torch.stack([c4, c3, c2, c1, c0]))
        rts[abs(rts.imag) > 1e-5] = float('nan')  # Mark all imaginary roots
        rts = rts.real

        best_t = torch.sum(self.t, dim=0)
        for root in rts:
            valid = filter & ~torch.isnan(root)
            if valid.any():
                root2 = root * root

                t1 = s * root / jm
                t3 = (a03 + 3 * a02 * da * root - 6 * jm2 * p0
                      - 6 * s * v0 * jm * root + 6 * jm2 * pT) \
                     / (6 * v0 * jm2 + 3 * s * a02 * jm
                        + s * (3 * jm * root2 + 6 * a0 * jm * root) * (1 + da))
                t7 = (-2 * a03 - 3 * root * (2 * a02 + 3 * a0 * root + root2) * (1 + da)
                      + 6 * jm2 * L - 6 * s * v0 * jm * ( a0 + root * (1 + da))) \
                     / (6 * v0 * jm2 + 3 * s * jm * (a02 + (2 * a0 * root + root2) * (1 + da)))

                valid = valid & (t1 > 0) & (t3 > 0) & (t7 > 0)
                if torch.any(valid):
                    new_t = self.t.clone()
                    new_t[0, valid] = t1[valid]
                    new_t[1, valid] = 0
                    new_t[2, valid] = t3[valid]
                    new_t[3, valid] = 0
                    new_t[4, valid] = 0
                    new_t[5, valid] = 0
                    new_t[6, valid] = t7[valid]
                    new_tpT = evaluate_trajectory(new_t, self.j, self.p0, self.v0, self.a0)[0]
                    valid_tpT = torch.isclose(new_tpT, self.pT, atol=1e-3)
                    if torch.any(valid_tpT):
                        # current best is not a valid solution but new profile is
                        improved = valid_tpT & ~torch.isclose(best_tpT, self.pT, atol=1e-3)
                        # or new solution is faster and valid
                        faster = valid_tpT & (t1+t3+t7 < best_t)
                        overwrite = improved | faster
                        self.t[:, overwrite] = new_t[:, overwrite]
                        # Set new best case
                        best_t[overwrite] = (t1+t3+t7)[overwrite]
                        best_tpT[overwrite] = new_tpT[overwrite]

    def optimize_tw_type(self, filter, best_tpT):
        s = self.s
        p0, pT = self.p0, self.pT
        L = pT - p0
        v0, a0, am, jm = self.v0, self.a0, self.amax, self.jmax
        v02, a02, am2, jm2 = self.v02, self.a02, self.amax2, self.jmax2
        a03, a04 = self.a03, self.a04
        condition = self.j[0] != self.j[2]
        da = torch.where(condition, torch.ones_like(p0), -torch.ones_like(p0))

        c4 = 12.0 * torch.ones_like(p0)
        c3 = -24.0*s*am
        c2 = 12.0*am2
        c1 = torch.zeros_like(p0)
        c0 = 12.0*s*v0*jm*da*(a02+am2) + 8.0*s*a03*am - 24.0*s*am*jm2*L \
             - 24.0*v0*a0*am*jm*da - 3.0*a04 - 6.0*a02*am2 - 12.0*v02*jm2

        rts = roots(torch.stack([c4, c3, c2, c1, c0]))
        rts[abs(rts.imag) > 1e-5] = float('nan')  # Mark all imaginary roots
        rts = rts.real

        best_t = torch.sum(self.t, dim=0)
        for root in rts:
            valid = filter & ~torch.isnan(root)
            if valid.any():
                root2 = root * root

                t1 = s*da * (s*am - a0) / jm
                t2 = da * (a02 - am2 + da*(am2 + 2*root2 - s*(4*root*am + 2*v0*jm))) / (2*am*jm)
                t3 = s * root / jm
                t7 = s * (root - s*am) / jm

                valid = valid & (t1 > 0) & (t2 > 0) & (t3 > 0) & (t7 > 0)
                if torch.any(valid):
                    new_t = self.t.clone()
                    new_t[0, valid] = t1[valid]
                    new_t[1, valid] = t2[valid]
                    new_t[2, valid] = t3[valid]
                    new_t[3, valid] = 0
                    new_t[4, valid] = 0
                    new_t[5, valid] = 0
                    new_t[6, valid] = t7[valid]
                    new_tpT = evaluate_trajectory(new_t, self.j, self.p0, self.v0, self.a0)[0]
                    valid_tpT = torch.isclose(new_tpT, self.pT, atol=1e-3)
                    if torch.any(valid_tpT):
                        # current best is not a valid solution but new profile is
                        improved = valid_tpT & ~torch.isclose(best_tpT, self.pT, atol=1e-3)
                        # or new solution is faster and valid
                        faster = valid_tpT & (t1+t2+t3+t7 < best_t)
                        overwrite = improved | faster
                        self.t[:, overwrite] = new_t[:, overwrite]
                        # Set new best case
                        best_t[overwrite] = (t1+t2+t3+t7)[overwrite]
                        best_tpT[overwrite] = new_tpT[overwrite]

    def optimize_wt_type(self, filter, best_tpT):
        s = self.s
        p0, pT = self.p0, self.pT
        L = pT - p0
        v0, a0, am, jm = self.v0, self.a0, self.amax, self.jmax
        v02, a02, am2, jm2 = self.v02, self.a02, self.amax2, self.jmax2
        a03, a04 = self.a03, self.a04
        condition = self.j[0] != self.j[2]
        da = torch.where(condition, torch.ones_like(p0), -torch.ones_like(p0))

        c4 = 6.0*(1+da)
        c3 = (24.0*a0 + 12.0*s*am)*(1.0+da)
        c2 = (36.0*s*a0*am + 12.0*s*v0*jm + 6.0*am2 + 30.0*a02)*(1.0+da)
        c1 = (12.0*a0*am2 + 24.0*v0*am*jm + 24.0*s*v0*a0*jm + 12.0*a03 + 24.0*s*am*a02)*(1.0+da)
        c0 = 12.0*s*v0*jm*(a02 + am2) - 24.0*s*am*jm2*L + 8.0*s*a03*am \
             + 6.0*a02*am2 + 24.0*v0*a0*am*jm + 12.0*v02*jm2 + 3.0*a04

        rts = roots(torch.stack([c4, c3, c2, c1, c0]))
        rts[abs(rts.imag) > 1e-5] = float('nan')  # Mark all imaginary roots
        rts = rts.real

        best_t = torch.sum(self.t, dim=0)
        for root in rts:
            valid = filter & ~torch.isnan(root)
            if valid.any():
                root2 = root * root

                t1 = s * root / jm
                t3 = s * (s*am + a0 + da*root) / jm
                t6 = (a02 - 2 * am2 + 2 * s * v0 * jm + (2 * a0 * root + root2) * (1 + da)) / jm / am / 2
                t7 = am / jm  # Always >0

                valid = valid & (t1 > 0) & (t3 > 0) & (t6 > 0)
                if torch.any(valid):
                    new_t = self.t.clone()
                    new_t[0, valid] = t1[valid]
                    new_t[1, valid] = 0
                    new_t[2, valid] = t3[valid]
                    new_t[3, valid] = 0
                    new_t[4, valid] = 0
                    new_t[5, valid] = t6[valid]
                    new_t[6, valid] = t7[valid]
                    new_tpT = evaluate_trajectory(new_t, self.j, self.p0, self.v0, self.a0)[0]
                    valid_tpT = torch.isclose(new_tpT, self.pT, atol=1e-3)
                    if torch.any(valid_tpT):
                        # current best is not a valid solution but new profile is
                        improved = valid_tpT & ~torch.isclose(best_tpT, self.pT, atol=1e-3)
                        # or new solution is faster and valid
                        faster = valid_tpT & (t1+t3+t6+t7 < best_t)
                        overwrite = improved | faster
                        self.t[:, overwrite] = new_t[:, overwrite]
                        # Set new best case
                        best_t[overwrite] = (t1+t3+t6+t7)[overwrite]
                        best_tpT[overwrite] = new_tpT[overwrite]

    def optimize_tt_type(self, filter, best_tpT):
        s = self.s
        p0, pT = self.p0, self.pT
        L = pT - p0
        v0, a0, am, jm = self.v0, self.a0, self.amax, self.jmax
        v02, a02, am2, jm2 = self.v02, self.a02, self.amax2, self.jmax2
        am4 = self.amax4
        a03, a04 = self.a03, self.a04
        condition = self.j[0] != self.j[2]
        da = torch.where(condition, torch.ones_like(p0), -torch.ones_like(p0))

        c2 = 24.0 * torch.ones_like(p0)
        c1 = -24*a02 + 48*s*v0*jm*da + 24*am2*(1.0+2*da)
        c0 = 24*am4*(1.0+da) + 12*s*v0*am2*jm*(4.0+3*da) - 24*v0*a0*am*jm*da - 12*s*v0*a02*jm*da\
             + 8*s*a03*am + 3*a04 - 24*s*am*jm2*L + 12*v02*jm2 - 6*a02*am2*(3.0+4*da)

        rts = roots(torch.stack([c2, c1, c0]))
        rts[abs(rts.imag) > 1e-5] = float('nan')  # Mark all imaginary roots
        rts = rts.real

        best_t = torch.sum(self.t, dim=0)
        for root in rts:
            valid = filter & ~torch.isnan(root)
            if valid.any():
                t1 = da * (am - s*a0) / jm
                t2 = da * root / (am*jm)
                t3 = am / jm  # Always > 0
                t6 = da * (-am2*da + 2*root + 2*s*v0*jm*da - a02 + am2) / (2*am*jm)

                valid = valid & (t1 > 0) & (t2 > 0) & ( t6 > 0)
                if torch.any(valid):
                    new_t = self.t.clone()
                    new_t[0, valid] = t1[valid]
                    new_t[1, valid] = t2[valid]
                    new_t[2, valid] = t3[valid]
                    new_t[3, valid] = 0
                    new_t[4, valid] = t3[valid]
                    new_t[5, valid] = t6[valid]
                    new_t[6, valid] = t3[valid]
                    new_tpT = evaluate_trajectory(new_t, self.j, self.p0, self.v0, self.a0)[0]
                    valid_tpT = torch.isclose(new_tpT, self.pT, atol=1e-3)
                    if torch.any(valid_tpT):
                        # current best is not a valid solution but new profile is
                        improved = valid_tpT & ~torch.isclose(best_tpT, self.pT, atol=1e-3)
                        # or new solution is faster and valid
                        faster = valid_tpT & (t1+t2+3*t3+t6 < best_t)
                        overwrite = improved | faster
                        self.t[:, overwrite] = new_t[:, overwrite]
                        # Set new best case
                        best_t[overwrite] = (t1+t2+3*t3+t6)[overwrite]
                        best_tpT[overwrite] = new_tpT[overwrite]

    def step_controller(self, count):
        p, v, a = self.trajectory[count]
        self.a0 = a
        self.apply_control_target(v)

    def set_target(self, target):
        p0, v0 = self.get_robot_states()
        self.initialize(p0, target, v0, self.a0)
        self.compute_profile()
        self.trajectory = self.evaluate_t_steps(
                self._task.control_frequency_inv, self.dt
        )


if __name__ == '__main__':
    device = "cpu"
    n = 2
    d = 2

    t = 2 * 0.01108 * np.pi  # Belt drive transmission factor (r*2*pi)
    vmax = torch.tensor([50 * t, 100 * np.pi], device=device)
    amax = torch.tensor([1500 * t, 3000 * np.pi], device=device)
    jmax = torch.tensor([5_000 * t, 50_000 * np.pi], device=device)

    vmax = vmax.expand(n, -1)
    amax = amax.expand(n, -1)
    jmax = jmax.expand(n, -1)

    planner = SCurve(n, d, device)
    planner.set_limits(vmax, amax, jmax)

    for i in range(100):
        p0 = (2 * torch.rand(n*d).reshape(n, d).to(device) - 1)
        p0[:, 0] *= 0.12
        p0[:, 1] *= 2*np.pi
        pT = (2 * torch.rand(n*d).reshape(n, d).to(device) - 1)
        pT[:, 0] *= 0.12
        pT[:, 1] *= 2*np.pi

        v0 = (2 * torch.rand(n*d).reshape(n, d).to(device) - 1) * 0.1*vmax
        a0 = (2 * torch.rand(n*d).reshape(n, d).to(device) - 1) * 0.0*amax
        dt = 1/1000

        planner.initialize(p0, pT, v0, a0)
        planner.compute_profile()
        tpT, tvT, taT = planner.evaluate_trajectory(
            planner.t, planner.j, p0, v0, a0
        )
        planner.evaluate_at_t(1/60)

    print("Done.")
