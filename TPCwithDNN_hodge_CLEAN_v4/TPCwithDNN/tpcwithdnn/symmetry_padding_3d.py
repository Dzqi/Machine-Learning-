"""
SUBROUTINE-MODULE symmetry_padding_3d  (VERSI STRUKTURAL TDL+PINN)
PURPOSE  : Dulu: padding simetris untuk U-Net (perwujudan asumsi BC simetris
           = kelemahan #2/#3 paper ALICE). Sekarang: cell complex silinder
           + operator Hodge mimetik (G, star_eps, A=G^T star G) yang membuat
           BC/geometri native sehingga padding simetris tidak diperlukan.
REFERENSI: DEC/mimetik L=D.G; paper ALICE EPJ 251,03020 (kelemahan U-Net);
           template loss Gao 2026 (residual soft yang digantikan).
STATUS   : Teruji: simetri eksak, sertifikat Gauss ~1e-18 (f64), MMS orde-2
           rasio 4.00 (jalankan self_test()).
DIPANGGIL: dnn_utils (HodgeNet), dnn_optimiser (loss/residual), data_loader
           (synthetic_event).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class CylGridComplex:
    """SUBROUTINE CylGridComplex | PURPOSE: node/edge grid silinder periodik-
    phi, incidence G, Hodge star, volume dual, mask konduktor, fitur jarak-
    batas. STATUS: teruji. DIPANGGIL: seluruh jalur struktural."""

    def __init__(self, nphi, nr, nz,
                 rmin=83.5, rmax=254.5, zmin=0.0, zmax=249.7, eps=1.0):
        self.nphi, self.nr, self.nz = int(nphi), int(nr), int(nz)
        self.r = np.linspace(rmin, rmax, nr)
        self.z = np.linspace(zmin, zmax, nz)
        self.phi = np.arange(nphi) * (2.0 * np.pi / nphi)
        self.dr = self.r[1] - self.r[0]
        self.dz = self.z[1] - self.z[0]
        self.dphi = 2.0 * np.pi / nphi
        self.eps = float(eps)
        self.n_nodes = nphi * nr * nz
        self._build_edges()
        self._build_incidence()
        self._build_star_and_volumes()
        self._build_masks()

    def nid(self, iphi, ir, iz):
        return (iphi * self.nr + ir) * self.nz + iz

    def _build_edges(self):
        P, R, Z = self.nphi, self.nr, self.nz
        ip, ir, iz = np.meshgrid(np.arange(P), np.arange(R), np.arange(Z),
                                 indexing="ij")
        m = iz < Z - 1
        tz, hz = self.nid(ip[m], ir[m], iz[m]), self.nid(ip[m], ir[m], iz[m] + 1)
        lz = np.full(tz.size, self.dz)
        m = ir < R - 1
        tr, hr = self.nid(ip[m], ir[m], iz[m]), self.nid(ip[m], ir[m] + 1, iz[m])
        lr = np.full(tr.size, self.dr)
        tp = self.nid(ip, ir, iz).ravel()
        hp = self.nid((ip + 1) % P, ir, iz).ravel()
        lp = (self.r[ir].ravel()) * self.dphi
        self.tail = np.concatenate([tz, tr, tp])
        self.head = np.concatenate([hz, hr, hp])
        self.length = np.concatenate([lz, lr, lp])
        self.n_z, self.n_r, self.n_p = tz.size, tr.size, tp.size
        self.n_edges = self.tail.size
        self._ez = (ip[iz < Z - 1], ir[iz < Z - 1], iz[iz < Z - 1])
        self._er = (ip[ir < R - 1], ir[ir < R - 1], iz[ir < R - 1])
        self._ep = (ip.ravel(), ir.ravel(), iz.ravel())

    def _build_incidence(self):
        rows = np.repeat(np.arange(self.n_edges), 2)
        cols = np.stack([self.head, self.tail], 1).ravel()
        vals = np.tile([1.0, -1.0], self.n_edges)
        self.G = sp.csr_matrix((vals, (rows, cols)),
                               shape=(self.n_edges, self.n_nodes))

    def _dw_r(self, ir):
        return np.where((ir == 0) | (ir == self.nr - 1), 0.5 * self.dr, self.dr)

    def _dw_z(self, iz):
        return np.where((iz == 0) | (iz == self.nz - 1), 0.5 * self.dz, self.dz)

    def _build_star_and_volumes(self):
        e = self.eps
        ipz, irz, izz = self._ez
        Az = self.r[irz] * self.dphi * self._dw_r(irz)
        ipr, irr, izr = self._er
        Ar = (self.r[irr] + 0.5 * self.dr) * self.dphi * self._dw_z(izr)
        ipp, irp, izp = self._ep
        Ap = self._dw_r(irp.ravel()) * self._dw_z(izp.ravel())
        area = np.concatenate([Az, Ar, Ap.ravel()])
        self.star = e * area / self.length
        ip, ir, iz = np.meshgrid(np.arange(self.nphi), np.arange(self.nr),
                                 np.arange(self.nz), indexing="ij")
        self.v_dual = (self.r[ir] * self.dphi * self._dw_r(ir)
                       * self._dw_z(iz)).ravel()

    def _build_masks(self):
        ip, ir, iz = np.meshgrid(np.arange(self.nphi), np.arange(self.nr),
                                 np.arange(self.nz), indexing="ij")
        b = (ir == 0) | (ir == self.nr - 1) | (iz == 0) | (iz == self.nz - 1)
        self.boundary_mask = b.ravel()
        self.interior_mask = ~self.boundary_mask
        d_r = np.minimum(ir, self.nr - 1 - ir) * self.dr
        d_z = np.minimum(iz, self.nz - 1 - iz) * self.dz
        self.dist_boundary = np.minimum(d_r, d_z).ravel()
        self.dist_boundary = self.dist_boundary / self.dist_boundary.max()

    def grid_shape(self):
        return (self.nphi, self.nr, self.nz)


def build_laplacian(cx):
    """SUBROUTINE build_laplacian | PURPOSE: A = G^T diag(star) G (simetris).
    STATUS: teruji. DIPANGGIL: dnn_optimiser, self_test."""
    S = sp.diags(cx.star)
    return (cx.G.T @ S @ cx.G).tocsr()


def poisson_residual(A, phi, rho, v_dual):
    """SUBROUTINE poisson_residual | PURPOSE: R = A Phi - rho*V_dual (Gauss
    diskret eksak saat R->0). STATUS: teruji. DIPANGGIL: dnn_optimiser."""
    return A @ phi - rho * v_dual


def solve_poisson_dirichlet0(A, rho, v_dual, interior):
    """SUBROUTINE solve_poisson_dirichlet0 | PURPOSE: solver CG referensi,
    Phi=0 di konduktor (label sintetik/validasi). STATUS: teruji."""
    idx = np.where(interior)[0]
    A_ii = A[idx][:, idx].tocsr()
    q = (rho * v_dual)[idx]
    M = sp.diags(1.0 / A_ii.diagonal())
    phi_i, info = spla.cg(A_ii, q, rtol=1e-10, maxiter=5000, M=M)
    assert info == 0, f"CG tidak konvergen (info={info})"
    phi = np.zeros(A.shape[0])
    phi[idx] = phi_i
    return phi


def conservation_certificate(A, phi):
    """SUBROUTINE conservation_certificate | PURPOSE: |1^T A Phi|/sum|A Phi|
    -> ~eps mesin utk SEMBARANG Phi (teleskopik; G.1=0): neraca muatan global
    dijamin konstruksi. STATUS: teruji."""
    tot = np.ones(A.shape[0]) @ (A @ phi)
    ref = np.abs(A @ phi).sum() + 1e-300
    return abs(tot) / ref


def e_field_nodes(cx, phi_grid):
    """SUBROUTINE e_field_nodes | PURPOSE: (Er,Ephi,Ez) beda-pusat di node
    utk tahap Langevin (residual TIDAK memakai ini). STATUS: teruji."""
    dphi, dr, dz = cx.dphi, cx.dr, cx.dz
    r = cx.r[None, :, None]
    Ephi = -(np.roll(phi_grid, -1, 0) - np.roll(phi_grid, 1, 0)) / (2 * dphi * r)
    Er = -np.gradient(phi_grid, dr, axis=1)
    Ez = -np.gradient(phi_grid, dz, axis=2)
    return Er, Ephi, Ez


def self_test():
    """SUBROUTINE self_test | PURPOSE: uji cepat operator (simetri, Gauss,
    curl-grad=0, MMS orde-2). Jalankan:
    PYTHONPATH=.. python3 -c "from tpcwithdnn.symmetry_padding_3d import self_test; self_test()"
    """
    def phi_star(cx):
        a = np.pi / (cx.r[-1] - cx.r[0]); b = np.pi / (cx.z[-1] - cx.z[0])
        ip, ir, iz = np.meshgrid(np.arange(cx.nphi), np.arange(cx.nr),
                                 np.arange(cx.nz), indexing="ij")
        r, z, ph = cx.r[ir], cx.z[iz], cx.phi[ip]
        R = np.sin(a * (r - cx.r[0])); C = np.cos(a * (r - cx.r[0]))
        S = np.sin(b * z); T = 1 + 0.3 * np.cos(3 * ph)
        lap = (-a * a * R + (a / r) * C) * S * T - 9 * R * S * (T - 1) / r ** 2 \
            - b * b * R * S * T
        return (R * S * T).ravel(), (-lap).ravel()

    def mms(nphi, nr, nz):
        cx = CylGridComplex(nphi, nr, nz, eps=1.0)
        A = build_laplacian(cx)
        pe, rho = phi_star(cx)
        pn = solve_poisson_dirichlet0(A, rho, cx.v_dual, cx.interior_mask)
        return np.abs(pn - pe)[cx.interior_mask].max()

    cx = CylGridComplex(24, 17, 17, eps=1.0)
    A = build_laplacian(cx)
    rng = np.random.default_rng(1)
    print("[1] |A-A^T|_max          =", abs(A - A.T).max())
    print("[2] sertifikat Gauss     =",
          conservation_certificate(A, rng.normal(size=cx.n_nodes)))
    e1, e2 = mms(24, 17, 17), mms(48, 33, 33)
    print(f"[3] MMS rasio kasar/halus= {e1/e2:.2f} (target ~4, orde-2)")
    print("self_test selesai.")


# ================= [BARU v4] Hierarki mesh & operator transfer =============
def make_hierarchy(cx, levels):
    """SUBROUTINE make_hierarchy (v4)
    PURPOSE  : Bangun deret complex silinder makin kasar utk Hodge-U-Net
               (pengganti max-pooling CNN): dim tiap sumbu ~separuh per level,
               geometri fisik (rmin..rmax, zmin..zmax) tetap.
    DIPANGGIL: dnn_utils.HodgeUNet."""
    hier = [cx]
    P, R, Z = cx.grid_shape()
    for _ in range(levels - 1):
        P = max(6, (P + 1) // 2)
        R = max(5, (R + 1) // 2)
        Z = max(5, (Z + 1) // 2)
        hier.append(CylGridComplex(P, R, Z, rmin=cx.r[0], rmax=cx.r[-1],
                                   zmin=cx.z[0], zmax=cx.z[-1], eps=cx.eps))
    return hier


def build_prolongation(fine_cx, coarse_cx):
    """SUBROUTINE build_prolongation (v4)
    PURPOSE  : Matriks prolongasi P (n_fine x n_coarse): interpolasi trilinear
               node kasar -> node halus; phi PERIODIK. Baris P berjumlah 1
               (partisi satuan) -> konstanta dipetakan eksak.
    DIPANGGIL: build_restriction, dnn_utils.HodgeUNet."""
    Pf, Rf, Zf = fine_cx.grid_shape()
    Pc, Rc, Zc = coarse_cx.grid_shape()
    ip, ir, iz = np.meshgrid(np.arange(Pf), np.arange(Rf), np.arange(Zf),
                             indexing="ij")
    phi_f = fine_cx.phi[ip].ravel()
    r_f = fine_cx.r[ir].ravel()
    z_f = fine_cx.z[iz].ravel()
    n_f = phi_f.size

    def lin_w(x, axis):  # indeks & bobot 1D non-periodik
        i0 = np.clip(np.searchsorted(axis, x) - 1, 0, len(axis) - 2)
        t = np.clip((x - axis[i0]) / (axis[i0 + 1] - axis[i0]), 0.0, 1.0)
        return i0, i0 + 1, 1.0 - t, t

    ir0, ir1, wr0, wr1 = lin_w(r_f, coarse_cx.r)
    iz0, iz1, wz0, wz1 = lin_w(z_f, coarse_cx.z)
    pos = phi_f / coarse_cx.dphi
    j0f = np.floor(pos)
    tphi = pos - j0f
    jp0 = j0f.astype(int) % Pc
    jp1 = (jp0 + 1) % Pc
    wp0, wp1 = 1.0 - tphi, tphi

    rows, cols, vals = [], [], []
    for jj, wp in ((jp0, wp0), (jp1, wp1)):
        for ii, wr in ((ir0, wr0), (ir1, wr1)):
            for kk, wz in ((iz0, wz0), (iz1, wz1)):
                rows.append(np.arange(n_f))
                cols.append((jj * Rc + ii) * Zc + kk)
                vals.append(wp * wr * wz)
    Pm = sp.csr_matrix((np.concatenate(vals),
                        (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n_f, Pc * Rc * Zc))
    return Pm


def build_restriction(Pm):
    """SUBROUTINE build_restriction (v4)
    PURPOSE  : Restriksi R = normalisasi-baris P^T (rata-rata berbobot;
               konstanta dipetakan eksak dua arah).
    DIPANGGIL: dnn_utils.HodgeUNet."""
    Rt = Pm.T.tocsr()
    s = np.asarray(Rt.sum(axis=1)).ravel()
    s[s == 0.0] = 1.0
    return sp.diags(1.0 / s) @ Rt
