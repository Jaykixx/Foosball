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

    def set_limits(self, vmax, amax, jmax):
        self.vmax = vmax
        self.amax = amax
        self.jmax = jmax

    def initialize(self, p0, pT, v0, a0):
        self.p0[:] = p0
        self.pT[:] = pT
        self.v0[:] = v0
        self.a0[:] = a0

        self.t[:] = 0
        self.j[:] = 0
        self.t_stop[:] = 0
        self.j_stop[:] = 0

        self.pStop = self.compute_fast_stop()
        self.s = self.compute_s()

    def compute_fast_stop(self):
        s = torch.where(self.a0 != 0, torch.sign(self.a0), torch.sign(self.v0))
        tj0 = torch.abs(self.a0 / self.jmax)
        vr = self.v0 + s * 0.5 * self.a0 ** 2 / self.jmax

        tja = torch.zeros_like(tj0)
        ta = torch.zeros_like(tj0)

        # Compute case 1: Either of c1, c2 or c3
        c1 = torch.sign(self.v0) == torch.sign(self.a0)
        c2 = self.a0 == 0
        c3 = torch.sign(self.v0) != torch.sign(vr)
        case1 = c1 | c2 | c3

        tja[case1] = torch.sqrt(torch.abs(vr) / self.jmax)[case1]

        strict_case1 = case1 & ((self.jmax * tja) > self.amax)
        tja[strict_case1] = self.amax[strict_case1] / self.jmax[strict_case1]
        ta[strict_case1] = (torch.abs(vr) / self.amax - tja)[strict_case1]

        # Compute case2:
        case2 = ~case1 & (torch.sign(self.v0) == torch.sign(vr))
        if torch.any(case2):
            p1 = s[case2] * vr[case2] / self.jmax[case2]
            p2 = 2 * s[case2] * self.a0[case2] / self.jmax[case2]
            p3 = torch.ones_like(p1)
            tja_candidates = roots(torch.stack((p3, p2, p1)))

            # Remove negative and complex roots since time must be a real positive number
            tja_candidates[tja_candidates.imag.abs() > 1e-5] = float('inf')
            tja_candidates[tja_candidates.real < 0] = float('inf')
            tja_candidates[torch.isnan(tja_candidates.real)] = float('inf')
            tja[case2] = tja_candidates.real.min(dim=-1).values

            strict_case2 = torch.minimum((self.jmax * tja + s * self.a0) > self.amax, case2)
            tja[strict_case2] = ((self.amax - (s*self.a0)) / self.jmax)[strict_case2]
            ta[strict_case2] = (torch.abs(vr) / self.amax - tja)[strict_case2]

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

        return self.evaluate_trajectory(self.t_stop, self.j_stop, self.p0, self.v0, self.a0)[0]

    def evaluate_trajectory(self, t, j, p0, v0, a0):
        p, v, a = p0, v0, a0
        for i in range(t.shape[0]):
            p, v, a = calculate_polynomial_trajectory(t[i], p, v, a, j[i])
        return p, v, a

    def evaluate_t_steps(self, steps, dt):
        num_envs, dofs = self.p0.shape[0], self.p0.shape[1]
        counts = torch.zeros(num_envs, dofs, dtype=torch.int, device=self.p0.device)
        cum_t = torch.zeros(num_envs, dofs, device=self.p0.device)
        pt = torch.zeros((num_envs, dofs, steps), device=self.p0.device)
        vt = torch.zeros((num_envs, dofs, steps), device=self.p0.device)
        at = torch.zeros((num_envs, dofs, steps), device=self.p0.device)
        p, v, a = self.p0.clone(), self.v0.clone(), self.a0.clone()
        for i in range(self.t.shape[0]):
            save = ((cum_t + dt) < self.t[i]) & (counts < steps)
            updated = save.clone()  # Store for use below since save will be altered
            while save.any():
                cum_t[save] += dt
                pt[save, counts[save]] = (p + v * cum_t + 0.5 * a * cum_t ** 2 + (1 / 6) * self.j[i] * cum_t ** 3)[save]
                vt[save, counts[save]] = (v + a * cum_t + 0.5 * self.j[i] * cum_t ** 2)[save]
                at[save, counts[save]] = (a + self.j[i] * cum_t)[save]
                counts[save] += 1
                save = ((cum_t + dt) < self.t[i]) & (counts < steps)

            p, v, a = calculate_polynomial_trajectory(self.t[i], p, v, a, self.j[i])
            cum_t[updated] -= self.t[i][updated]

        fills = counts < steps
        while fills.any():
            pt[fills, counts[fills]] = p[fills]
            vt[fills, counts[fills]] = v[fills]
            at[fills, counts[fills]] = a[fills]
            counts[fills] += 1
            fills = counts < steps

        return torch.stack((pt, vt, at)).permute(3, 0, 1, 2)

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

        p = p + v * final_t + 0.5 * a * final_t ** 2 + (1 / 6) * j * final_t ** 3
        v = v + a * final_t + 0.5 * j * final_t ** 2
        a = a + j * final_t
        return p, v, a

    def compute_s(self):
        return torch.sign(self.pT - self.pStop)

    def compute_zero_cruise_profile(self):
        vr = self.s * self.vmax

        # estimate acceleration
        ar = self.s * torch.sqrt((vr - self.v0) * self.s * self.jmax + 0.5 * self.a0 ** 2)
        t2 = torch.zeros_like(ar)
        c1 = torch.abs(ar) > self.amax
        ar[c1] = self.s[c1] * self.amax[c1]
        t2[c1] = ((vr - self.v0 + (0.5 * self.a0 ** 2 - ar ** 2) / (self.s * self.jmax)) / ar)[c1]

        # estimate deceleration
        dr = - self.s * torch.sqrt(torch.abs(vr) * self.jmax)
        t6 = torch.zeros_like(dr)
        c2 = torch.abs(dr) > self.amax
        dr[c2] = -self.s[c2] * self.amax[c2]
        t6[c2] = torch.abs(vr / dr)[c2] - torch.abs(dr / self.jmax)[c2]
        t5 = torch.abs(dr) / self.jmax

        self.t[0] = torch.abs(ar - self.a0) / self.jmax
        self.j[0] = self.s * self.jmax
        self.t[1] = t2
        self.t[2] = torch.abs(ar) / self.jmax
        self.j[2] = -self.s * self.jmax

        self.t[4] = t5
        self.j[4] = -self.s * self.jmax
        self.t[5] = t6
        self.t[6] = t5
        self.j[6] = self.s * self.jmax

        return self.evaluate_trajectory(self.t, self.j, self.p0, self.v0, self.a0)[0]

    def compute_profile(self):
        stop_case = torch.isclose(self.pStop, self.pT, atol=1e-3)

        tpT = self.compute_zero_cruise_profile()
        vr = self.s * self.vmax
        t4 = (self.pT - tpT) / vr
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
            self.optimize_ww_type(ww_filter)
        if torch.any(tw_filter):
            self.optimize_tw_type(tw_filter)
        if torch.any(wt_filter):
            self.optimize_wt_type(wt_filter)
        if torch.any(tt_filter):
            self.optimize_tt_type(tt_filter)

        # unsolvable trajectories go through unchanged
        # Apply stop trajectories in these cases instead
        tpT = self.evaluate_trajectory(self.t, self.j, self.p0, self.v0, self.a0)[0]
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
        dt = torch.min(t2, t6)[tt_profiles]
        new_t[1, tt_profiles] -= dt
        new_t[5, tt_profiles] -= dt

        # Case 3: WT-Profiles
        wt_profiles = new_filter & (t2 == 0) & (t6 > 0)
        area_w_max = self.jmax * t3 ** 2
        area_w_max = torch.where(t1 < t3, area_w_max-(0.5*self.a0**2)/self.jmax, area_w_max)
        area_t_max = t6 * self.amax
        cutable = wt_profiles & (area_w_max > area_t_max)
        new_t[5, cutable] = 0
        a1 = self.a0 + self.j[0] * t1
        c = (a1**2 - area_t_max * self.jmax)[cutable]
        dt = (a1[cutable].abs() - torch.sqrt(c)) / self.jmax[cutable]
        new_t[4, cutable] -= dt
        new_t[6, cutable] -= dt
        new_filter[new_filter & wt_profiles & ~cutable] = False

        # Case 4: TW-Profiles
        tw_profiles = new_filter & (t2 > 0) & (t6 == 0)
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

        new_pT = self.evaluate_trajectory(new_t, self.j, self.p0, self.v0, self.a0)[0]
        overshoot = self.s * torch.sign(new_pT - self.pT) == 1
        new_filter = torch.minimum(new_filter, overshoot)
        if torch.any(new_filter):
            self.t[:, new_filter] = new_t[:, new_filter]
            new_pT[new_filter] = self.compute_profile_type_reduction(new_filter)[new_filter]

        return new_pT

    def optimize_ww_type(self, filter):
        # s = self.s[filter]
        # p0, pT = self.p0[filter], self.pT[filter]
        s = self.s
        p0, pT = self.p0, self.pT
        p02, pT2 = torch.pow(p0, 2), torch.pow(pT, 2)
        L = pT - p0
        # v0, a0, jm = self.v0[filter], self.a0[filter], self.jmax[filter]
        v0, a0, jm = self.v0, self.a0, self.jmax
        v02, a02, jm2 = torch.pow(v0, 2), torch.pow(a0, 2), torch.pow(jm, 2)
        v03, a03, jm3 = torch.pow(v0, 3), torch.pow(a0, 3), torch.pow(jm, 3)
        a04, jm4 = torch.pow(a0, 4), torch.pow(jm, 4)
        condition = self.j[0] != self.j[2]
        da = torch.where(condition, torch.ones_like(p0), -torch.ones_like(p0))

        c4 = (-18*a02 + 36*s*v0*jm)*(1.0+da)
        c3 = (72*s*v0*a0*jm - 72*jm2*L - 48*a03)*(1.0+da)
        c2 = (-27*a04 - 216*a0*jm2*L + 36*v02*jm2 - 36*s*v0*a02*jm)*(1.0+da)
        c1 = (-144*s*v0*jm3*L - 144*a02*jm2*L - 72*v02*a0*jm2 - 6*a0**5 - 24*s*v0*a03*jm)*(1.0+da)
        c0 = - 6*s*v0*a04*jm - 144*jm4*p0*pT - 144*s*v0*a0*jm3*L - 72*s*v03*jm3\
             - 48*a03*jm2*L + 72*jm4*(pT2+p02) - 36*v02*a02*jm2 - a03**2

        rts = roots(torch.stack([c4, c3, c2, c1, c0]))
        rts[abs(rts.imag) > 1e-5] = float('nan')  # Mark all imaginary roots
        rts = rts.real

        best_t = torch.sum(self.t, dim=0)
        for root in rts:
            root2, root3 = torch.pow(root, 2), torch.pow(root, 3)

            t1 = s * root / jm
            t3 = (a03 + 3 * a02 * da * root - 6 * jm2 * p0
                  - 6 * s * v0 * jm * root + 6 * jm2 * pT) \
                 / (6 * v0 * jm2 + 3 * s * a02 * jm
                    + s * (3 * jm * root2 + 6 * a0 * jm * root) * (1 + da))
            t7 = (-2 * a03 - 3 * root * (2 * a02 + 3 * a0 * root + root2) * (1 + da)
                  + 6 * jm2 * L - 6 * s * v0 * jm * ( a0 + root * (1 + da))) \
                 / (6 * v0 * jm2 + 3 * s * jm * (a02 + (2 * a0 * root + root2) * (1 + da)))

            valid = torch.stack(
                (~torch.isnan(root), t1 > 0, t3 > 0, t7 > 0, t1+t3+t7 < best_t)
            )
            valid = torch.minimum(filter, torch.min(valid, dim=0).values)
            if torch.any(valid):
                self.t[0, valid] = t1[valid]
                self.t[1, valid] = 0
                self.t[2, valid] = t3[valid]
                self.t[3, valid] = 0
                self.t[4, valid] = 0
                self.t[5, valid] = 0
                self.t[6, valid] = t7[valid]
                best_t[valid] = (t1+t3+t7)[valid]

    def optimize_tw_type(self, filter):
        s = self.s
        p0, pT = self.p0, self.pT
        L = pT - p0
        v0, a0, am, jm = self.v0, self.a0, self.amax, self.jmax
        v02, a02 = torch.pow(v0, 2), torch.pow(a0, 2)
        am2, jm2 = torch.pow(am, 2), torch.pow(jm, 2)
        a03, a04 = torch.pow(a0, 3), torch.pow(a0, 4)
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
            root2 = torch.pow(root, 2)

            t1 = s*da * (s*am - a0) / jm
            t2 = da * (a02 - am2 + da*(am2 + 2*root2 - s*(4*root*am + 2*v0*jm))) / (2*am*jm)
            t3 = s * root / jm
            t7 = s * (root - s*am) / jm

            valid = torch.stack(
                (~torch.isnan(root), t1 > 0, t2 > 0, t3 > 0, t7 > 0, t1+t2+t3+t7 < best_t)
            )
            valid = torch.minimum(filter, torch.min(valid, dim=0).values)
            if torch.any(valid):
                self.t[0, valid] = t1[valid]
                self.t[1, valid] = t2[valid]
                self.t[2, valid] = t3[valid]
                self.t[3, valid] = 0
                self.t[4, valid] = 0
                self.t[5, valid] = 0
                self.t[6, valid] = t7[valid]
                best_t[valid] = (t1 + t2 + t3 + t7)[valid]

    def optimize_wt_type(self, filter):
        s = self.s
        p0, pT = self.p0, self.pT
        L = pT - p0
        v0, a0, am, jm = self.v0, self.a0, self.amax, self.jmax
        v02, a02 = torch.pow(v0, 2), torch.pow(a0, 2)
        am2, jm2 = torch.pow(am, 2), torch.pow(jm, 2)
        a03, a04 = torch.pow(a0, 3), torch.pow(a0, 4)
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
            root2 = torch.pow(root, 2)

            t1 = s * root / jm
            t3 = s * (s*am + a0 + da*root) / jm
            t6 = (a02 - 2 * am2 + 2 * s * v0 * jm + (2 * a0 * root + root2) * (1 + da)) / jm / am / 2
            t7 = am / jm  # Always >0

            valid = torch.stack(
                (~torch.isnan(root), t1 > 0, t3 > 0, t6 > 0, t1+t3+t6+t7 < best_t)
            )
            valid = torch.minimum(filter, torch.min(valid, dim=0).values)
            if torch.any(valid):
                self.t[0, valid] = t1[valid]
                self.t[1, valid] = 0
                self.t[2, valid] = t3[valid]
                self.t[3, valid] = 0
                self.t[4, valid] = 0
                self.t[5, valid] = t6[valid]
                self.t[6, valid] = t7[valid]
                best_t[valid] = (t1 + t3 + t6 + t7)[valid]

    def optimize_tt_type(self, filter):
        s = self.s
        p0, pT = self.p0, self.pT
        L = pT - p0
        v0, a0, am, jm = self.v0, self.a0, self.amax, self.jmax
        v02, a02, jm2 = torch.pow(v0, 2), torch.pow(a0, 2), torch.pow(jm, 2)
        am2, am4 = torch.pow(am, 2), torch.pow(am, 4)
        a03, a04 = torch.pow(a0, 3), torch.pow(a0, 4)
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
            t1 = da * (am - s*a0) / jm
            t2 = da * root / (am*jm)
            t3 = am / jm  # Always > 0
            t6 = da * (-am2*da + 2*root + 2*s*v0*jm*da - a02 + am2) / (2*am*jm)

            valid = torch.stack(
                (~torch.isnan(root), t1 > 0, t2 > 0, t6 > 0, t1+t2+3*t3+t6 < best_t)
            )
            valid = torch.minimum(filter, torch.min(valid, dim=0).values)
            if torch.any(valid):
                self.t[0, valid] = t1[valid]
                self.t[1, valid] = t2[valid]
                self.t[2, valid] = t3[valid]
                self.t[3, valid] = 0
                self.t[4, valid] = t3[valid]
                self.t[5, valid] = t6[valid]
                self.t[6, valid] = t3[valid]
                best_t[valid] = (t1 + t2 + 3*t3 + t6)[valid]

    def step_controller(self, count):
        p, v, a = self.trajectory[count]
        self.apply_control_target(v)
        if count == self.max_steps - 1:
            self.a0 = a

    def set_target(self, target):
        p0, v0 = self.get_robot_states()
        self.initialize(p0, target, v0, self.a0)
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
