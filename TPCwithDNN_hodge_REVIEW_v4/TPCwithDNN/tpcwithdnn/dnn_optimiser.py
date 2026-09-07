"""
[REVIEW] FILE DIGANTI TOTAL oleh jalur struktural TDL+PINN.
Kode ASLI = baris awal file s.d. penanda KODE-BARU (207 baris, tiap baris berawalan "# OLD| ").
KODE BARU dimulai setelah penanda.
"""

# OLD| """
# OLD| Deep neural network for 3D IDC distortion correction.
# OLD| 
# OLD| NOTE: This code is based on old data, it needs to be adjusted to IDC.
# OLD| """
# OLD| # pylint: disable=protected-access
# OLD| # pylint: disable=consider-using-f-string
# OLD| 
# OLD| import matplotlib.pyplot as plt
# OLD| 
# OLD| import numpy as np
# OLD| 
# OLD| from keras.callbacks import TensorBoard
# OLD| from keras.optimizers import Adam
# OLD| from keras.metrics import RootMeanSquaredError
# OLD| from keras.models import model_from_json
# OLD| from keras.utils import plot_model
# OLD| 
# OLD| from ROOT import TFile # pylint: disable=import-error, no-name-in-module
# OLD| 
# OLD| from tpcwithdnn import plot_utils
# OLD| from tpcwithdnn.optimiser import Optimiser
# OLD| from tpcwithdnn.symmetry_padding_3d import SymmetryPadding3d
# OLD| from tpcwithdnn.fluctuation_data_generator import FluctuationDataGenerator
# OLD| from tpcwithdnn.dnn_utils import u_net
# OLD| from tpcwithdnn.data_loader import load_train_apply
# OLD| 
# OLD| class DnnOptimiser(Optimiser):
# OLD|     """
# OLD|     DNN optimizer class, with the interface defined by the Optimiser parent class
# OLD|     """
# OLD|     name = "dnn"
# OLD| 
# OLD|     def __init__(self, config):
# OLD|         """
# OLD|         Initialize the optimizer. No more action needed that in the base class.
# OLD| 
# OLD|         :param CommonSettings config: a singleton settings object
# OLD|         """
# OLD|         super().__init__(config)
# OLD|         self.config.logger.info("DnnOptimiser::Init")
# OLD| 
# OLD|     def train(self):
# OLD|         """
# OLD|         Train the optimizer.
# OLD|         """
# OLD|         self.config.logger.info("DnnOptimiser::train")
# OLD| 
# OLD|         training_generator = FluctuationDataGenerator(self.config.partition['train'],
# OLD|                                                       dirinput=self.config.dirinput_train,
# OLD|                                                       **self.config.params)
# OLD|         validation_generator = FluctuationDataGenerator(self.config.partition['validation'],
# OLD|                                                         dirinput=self.config.dirinput_validation,
# OLD|                                                         **self.config.params)
# OLD|         model = u_net((self.config.grid_phi, self.config.grid_r, self.config.grid_z,
# OLD|                        self.config.dim_input),
# OLD|                       depth=self.config.depth, batchnorm=self.config.batch_normalization,
# OLD|                       pool_type=self.config.pool_type, start_channels=self.config.filters,
# OLD|                       dropout=self.config.dropout)
# OLD|         if self.config.metrics == "root_mean_squared_error":
# OLD|             metrics = RootMeanSquaredError()
# OLD|         else:
# OLD|             metrics = self.config.metrics
# OLD|         model.compile(loss=self.config.lossfun, optimizer=Adam(lr=self.config.adamlr),
# OLD|                       metrics=[metrics]) # Mean squared error
# OLD| 
# OLD|         model.summary()
# OLD|         plot_model(model, to_file='%s/model_%s_nEv%d.png' % \
# OLD|                    (self.config.dirplots, self.config.suffix, self.config.train_events),
# OLD|                    show_shapes=True, show_layer_names=True)
# OLD| 
# OLD|         #log_dir = "logs/" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
# OLD|         log_dir = 'logs/' + '%s_nEv%d' % (self.config.suffix, self.config.train_events)
# OLD|         tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
# OLD| 
# OLD|         model._get_distribution_strategy = lambda: None
# OLD|         his = model.fit(training_generator,
# OLD|                         validation_data=validation_generator,
# OLD|                         use_multiprocessing=False,
# OLD|                         epochs=self.config.epochs, callbacks=[tensorboard_callback])
# OLD| 
# OLD|         self.__plot_train(his)
# OLD|         self.save_model(model)
# OLD| 
# OLD|     def apply(self):
# OLD|         """
# OLD|         Apply the optimizer.
# OLD|         """
# OLD|         self.config.logger.info("DnnOptimiser::apply, input size: %d", self.config.dim_input)
# OLD|         loaded_model = self.load_model()
# OLD| 
# OLD|         myfile = TFile.Open("%s/output_%s_nEv%d.root" % \
# OLD|                             (self.config.dirapply, self.config.suffix, self.config.train_events),
# OLD|                             "recreate")
# OLD|         h_dist_all_events, h_deltas_all_events, h_deltas_vs_dist_all_events =\
# OLD|                 plot_utils.create_apply_histos(self.config, self.config.suffix, infix="all_events_")
# OLD| 
# OLD|         for indexev in self.config.partition['apply']:
# OLD|             inputs_, exp_outputs_ = load_train_apply(self.config.dirinput_apply, indexev,
# OLD|                                                      self.config.z_range,
# OLD|                                                      self.config.grid_r, self.config.grid_phi,
# OLD|                                                      self.config.grid_z,
# OLD|                                                      self.config.opt_train,
# OLD|                                                      self.config.opt_predout)
# OLD|             inputs_single = np.empty((1, self.config.grid_phi, self.config.grid_r,
# OLD|                                       self.config.grid_z, self.config.dim_input))
# OLD|             exp_outputs_single = np.empty((1, self.config.grid_phi, self.config.grid_r,
# OLD|                                            self.config.grid_z, self.config.dim_output))
# OLD|             inputs_single[0, :, :, :, :] = inputs_
# OLD|             exp_outputs_single[0, :, :, :, :] = exp_outputs_
# OLD| 
# OLD|             distortion_predict_group = loaded_model.predict(inputs_single)
# OLD| 
# OLD|             distortion_numeric_flat_m, distortion_predict_flat_m, deltas_flat_a, deltas_flat_m =\
# OLD|                 plot_utils.get_apply_results_single_event(distortion_predict_group,
# OLD|                                                           exp_outputs_single)
# OLD|             plot_utils.fill_apply_tree_single_event(self.config, indexev,
# OLD|                                                     distortion_numeric_flat_m,
# OLD|                                                     distortion_predict_flat_m,
# OLD|                                                     deltas_flat_a, deltas_flat_m)
# OLD|             plot_utils.fill_apply_tree(h_dist_all_events, h_deltas_all_events,
# OLD|                                        h_deltas_vs_dist_all_events,
# OLD|                                        distortion_numeric_flat_m, distortion_predict_flat_m,
# OLD|                                        deltas_flat_a, deltas_flat_m)
# OLD| 
# OLD|         for hist in (h_dist_all_events, h_deltas_all_events, h_deltas_vs_dist_all_events):
# OLD|             hist.Write()
# OLD|         plot_utils.fill_profile_apply_hist(h_deltas_vs_dist_all_events, self.config.profile_name,
# OLD|                                            self.config.suffix)
# OLD|         plot_utils.fill_std_dev_apply_hist(h_deltas_vs_dist_all_events, self.config.h_std_dev_name,
# OLD|                                            self.config.suffix, "all_events_")
# OLD| 
# OLD|         myfile.Close()
# OLD|         self.config.logger.info("Done apply")
# OLD| 
# OLD|     def search_grid(self):
# OLD|         """
# OLD|         Perform grid search to find the best model configuration.
# OLD| 
# OLD|         :raises NotImplementedError: the method not implemented yet for DNN
# OLD|         """
# OLD|         raise NotImplementedError("Search grid method not implemented yet")
# OLD| 
# OLD|     def bayes_optimise(self):
# OLD|         """
# OLD|         Perform Bayesian optimization to find the best model configuration.
# OLD| 
# OLD|         :raises NotImplementedError: the method not implemented yet for DNN
# OLD|         """
# OLD|         raise NotImplementedError("Bayes optimise method not implemented yet")
# OLD| 
# OLD|     def save_model(self, model):
# OLD|         """
# OLD|         Save the model to a JSON file, and the weights to a h5 file.
# OLD| 
# OLD|         :param tf.keras.Model model: the tf.keras model to be saved
# OLD|         """
# OLD|         model_json = model.to_json()
# OLD|         with open("%s/model_%s_nEv%d.json" % (self.config.dirmodel, self.config.suffix,
# OLD|                                               self.config.train_events), "w", encoding="utf-8") \
# OLD|             as json_file:
# OLD|             json_file.write(model_json)
# OLD|         model.save_weights("%s/model_%s_nEv%d.h5" % (self.config.dirmodel, self.config.suffix,
# OLD|                                                      self.config.train_events))
# OLD|         self.config.logger.info("Saved trained DNN model to disk")
# OLD| 
# OLD|     def load_model(self):
# OLD|         """
# OLD|         Load the DNN model from a JSON file, with weights from a h5 file
# OLD| 
# OLD|         :return: the loaded model
# OLD|         :rtype: tf.keras.Model
# OLD|         """
# OLD|         with open("%s/model_%s_nEv%d.json" % \
# OLD|                   (self.config.dirmodel, self.config.suffix, self.config.train_events), "r",
# OLD|                   encoding="utf-8") as f:
# OLD|             loaded_model_json = f.read()
# OLD|         loaded_model = \
# OLD|             model_from_json(loaded_model_json, {'SymmetryPadding3d' : SymmetryPadding3d})
# OLD|         loaded_model.load_weights("%s/model_%s_nEv%d.h5" % \
# OLD|                                   (self.config.dirmodel, self.config.suffix,
# OLD|                                    self.config.train_events))
# OLD|         return loaded_model
# OLD| 
# OLD|     def __plot_train(self, his):
# OLD|         """
# OLD|         Plot the learning curve for 3D calibration.
# OLD|         Function used internally.
# OLD| 
# OLD|         :param tf.keras.History his: a history object of the network training,
# OLD|                                      returned by model.fit()
# OLD|         """
# OLD|         plt.style.use("ggplot")
# OLD|         plt.figure()
# OLD|         plt.yscale('log')
# OLD|         plt.plot(np.arange(0, self.config.epochs), his.history["loss"], label="train_loss")
# OLD|         plt.plot(np.arange(0, self.config.epochs), his.history["val_loss"], label="val_loss")
# OLD|         plt.plot(np.arange(0, self.config.epochs), his.history[self.config.metrics],
# OLD|                  label="train_" + self.config.metrics)
# OLD|         plt.plot(np.arange(0, self.config.epochs), his.history["val_" + self.config.metrics],
# OLD|                  label="val_" + self.config.metrics)
# OLD|         plt.title("Training Loss and Accuracy on Dataset")
# OLD|         plt.xlabel("Epoch #")
# OLD|         plt.ylabel("Loss/Accuracy")
# OLD|         plt.legend(loc="lower left")
# OLD|         plt.savefig("%s/learning_plot_%s_nEv%d.png" % (self.config.dirplots, self.config.suffix,
# OLD|                                                        self.config.train_events))

# ======================= KODE BARU =======================
"""
SUBROUTINE-MODULE dnn_optimiser  (VERSI STRUKTURAL TDL+PINN, v4 - Hodge-U-Net + epoch sejati)
PURPOSE  : Dulu: training U-Net Keras + keluaran ROOT via PyROOT.
           Sekarang: DnnOptimiser struktural - HodgeNet + loss template Gao
           (residual Poisson via operator Hodge, BUKAN autodiff) + Langevin
           terdiferensialkan. v2 menambah: (a) create-data MC statistik
           (Pers.1 ALICE), (b) plotting matplotlib (kurva belajar + peta
           distorsi pred/label/error), (c) keluaran .root via uproot
           (TANPA butuh instalasi ROOT penuh; opsional).
REFERENSI: Gao 2026 Pers.14-17; paper ALICE Sec.2.2 (Pers.1) & Sec.3;
           keluaran asli: model + output_*.root + learning_plot_*.png.
STATUS   : Kalibrasi satuan data riil, E0 & omega_tau = PLACEHOLDER;
           skema branch .root kita (TTree 'validation') TIDAK meniru persis
           skema internal ROOT asli - didokumentasikan.
DIPANGGIL: steer_analysis.init_models.
"""
import os
import json
import numpy as np
import scipy.sparse as sp
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tpcwithdnn.logger import get_logger
from tpcwithdnn.symmetry_padding_3d import (CylGridComplex, build_laplacian,
                                            solve_poisson_dirichlet0,
                                            e_field_nodes)
from tpcwithdnn.dnn_utils import HodgeNet, HodgeUNet, sp2torch
from tpcwithdnn.data_loader import (load_event_grid, synthetic_event,
                                    synthetic_event_mc)

RMIN_CM, RMAX_CM = 83.5, 254.5   # geometri TPC (IFC/OFC)


def _coeffs(omega_tau):
    c0 = 1.0 / (1.0 + omega_tau ** 2)
    c1 = omega_tau / (1.0 + omega_tau ** 2)
    return c0, c1


def distortions_numpy(cx, Er, Ephi, Ez, omega_tau, e0):
    """SUBROUTINE distortions_numpy | PURPOSE: twin numpy integral distorsi
    (label sintetik/validasi). PLACEHOLDER konvensi dZ/tanda B."""
    c0, c1 = _coeffs(omega_tau)
    fr = (c0 * Er + c1 * Ephi) / e0
    fp = (-c1 * Er + c0 * Ephi) / e0
    fz = Ez / e0
    def cumint(f):
        rev = np.flip(f, axis=2)
        acc = np.cumsum(0.5 * (rev + np.roll(rev, 1, axis=2)), axis=2) * cx.dz
        acc[:, :, 0] = 0.0
        return np.ascontiguousarray(np.flip(acc, axis=2))
    return cumint(fr), cumint(fp), cumint(fz)


def make_torch_distorter(cx, omega_tau, e0):
    """SUBROUTINE make_torch_distorter | PURPOSE: Langevin terdiferensialkan
    Phi->E->(dR,dRPhi,dZ)."""
    c0, c1 = _coeffs(omega_tau)
    dphi, dr, dz = cx.dphi, cx.dr, cx.dz
    r = torch.tensor(cx.r, dtype=torch.float32)[None, :, None]

    def forward(phi_grid):
        Ephi = -(torch.roll(phi_grid, -1, 0) - torch.roll(phi_grid, 1, 0)) \
               / (2 * dphi * r)
        Er = -torch.gradient(phi_grid, spacing=dr, dim=1)[0]
        Ez = -torch.gradient(phi_grid, spacing=dz, dim=2)[0]
        def cumint(f):
            rev = torch.flip(f, dims=[2])
            acc = torch.cumsum(0.5 * (rev + torch.roll(rev, 1, dims=2)),
                               dim=2) * dz
            acc = acc.clone(); acc[:, :, 0] = 0.0
            return torch.flip(acc, dims=[2])
        dR = cumint((c0 * Er + c1 * Ephi) / e0)
        dP = cumint((-c1 * Er + c0 * Ephi) / e0)
        dZ = cumint(Ez / e0)
        return dR, dP, dZ
    return forward


def _fd_laplacian(cx):
    """SUBROUTINE _fd_laplacian | PURPOSE: Laplacian silinder 7-titik NON-
    mimetik = residual 'soft' ala Gao (mode ablasi fd_soft)."""
    P, R, Z = cx.nphi, cx.nr, cx.nz
    rows, cols, vals = [], [], []
    def nid(ip, ir, iz): return (ip * R + ir) * Z + iz
    for ipp in range(P):
        for irr in range(1, R - 1):
            rr_ = cx.r[irr]
            for izz in range(1, Z - 1):
                i = nid(ipp, irr, izz)
                cr, cz = 1.0 / cx.dr ** 2, 1.0 / cx.dz ** 2
                cp = 1.0 / (rr_ * cx.dphi) ** 2
                rr1 = 1.0 / (2 * rr_ * cx.dr)
                for j, v in ((nid(ipp, irr + 1, izz), cr + rr1),
                             (nid(ipp, irr - 1, izz), cr - rr1),
                             (nid(ipp, irr, izz + 1), cz),
                             (nid(ipp, irr, izz - 1), cz),
                             (nid((ipp + 1) % P, irr, izz), cp),
                             (nid((ipp - 1) % P, irr, izz), cp),
                             (i, -2 * (cr + cz + cp))):
                    rows.append(i); cols.append(j); vals.append(v)
    n = cx.n_nodes
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


class StructuralPINNLoss(torch.nn.Module):
    """SUBROUTINE StructuralPINNLoss | PURPOSE: L total template Gao dengan
    residual struktural Hodge + diagnostik sertifikat Gauss."""

    def __init__(self, cx, distorter, lam_data=1.0, lam_pde=0.1,
                 residual_mode="structural_hodge"):
        super().__init__()
        self.cx, self.distorter = cx, distorter
        self.lam_data, self.lam_pde = lam_data, lam_pde
        self.mode = residual_mode
        self.A_t = sp2torch(build_laplacian(cx))
        self.Afd_t = sp2torch(_fd_laplacian(cx)) if residual_mode == "fd_soft" \
            else None
        self.register_buffer("vdual", torch.tensor(cx.v_dual,
                                                   dtype=torch.float32))
        self.register_buffer("imask", torch.tensor(
            cx.interior_mask.astype(np.float32)))
        self.register_buffer("ones", torch.ones(cx.n_nodes))

    def forward(self, phi_flat, rho_flat, labels):
        cx = self.cx
        if self.mode == "structural_hodge":
            R = torch.sparse.mm(self.A_t, phi_flat[:, None]).squeeze(1) \
                - rho_flat * self.vdual
            R = R / self.vdual.mean()
        else:
            R = torch.sparse.mm(self.Afd_t, phi_flat[:, None]).squeeze(1) \
                + rho_flat
        l_pde = ((R * self.imask) ** 2).sum() / self.imask.sum()
        phi_g = phi_flat.view(*cx.grid_shape())
        dR, dP, dZ = self.distorter(phi_g)
        lr, lp, lz = labels
        l_data = ((dR - lr) ** 2).mean() + ((dP - lp) ** 2).mean() \
            + ((dZ - lz) ** 2).mean()
        total = self.lam_data * l_data + self.lam_pde * l_pde
        with torch.no_grad():
            aphi = torch.sparse.mm(self.A_t, phi_flat[:, None]).squeeze(1)
            cert = (self.ones @ aphi).abs() / (aphi.abs().sum() + 1e-30)
        return total, {"L_data": l_data.detach().item(),
                       "L_pde": l_pde.detach().item(),
                       "gauss_cert": cert.item()}


# ---------------------- [BARU v2] plotting matplotlib ----------------------
def plot_learning(history, path):
    """SUBROUTINE plot_learning | PURPOSE: kurva belajar (L_data, L_pde,
    sertifikat Gauss) -> PNG; padanan learning_plot_*.png repo asli."""
    idx = [h.get("epoch", i) for i, h in enumerate(history)]
    ld = [h["L_data"] for h in history]
    lp = [h["L_pde"] for h in history]
    gc = [h["gauss_cert"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].semilogy(idx, ld, label="L_data")
    ax[0].semilogy(idx, lp, label="L_pde (Hodge)")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
    ax[0].legend(); ax[0].set_title("Kurva belajar")
    ax[1].semilogy(idx, gc, color="tab:green")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("|1^T A phi| / sum")
    ax[1].set_title("Sertifikat konservasi (float32)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_maps(cx, preds, labels, path, iphi=0):
    """SUBROUTINE plot_maps | PURPOSE: peta (r,z) irisan phi: label vs
    prediksi vs |error| untuk dR/dRPhi/dZ -> satu PNG."""
    names = ("dR", "dRPhi", "dZ")
    ext = [cx.z[0], cx.z[-1], cx.r[0], cx.r[-1]]
    fig, ax = plt.subplots(3, 3, figsize=(12, 9))
    for i, nm in enumerate(names):
        lab = labels[i][iphi]; pre = preds[i][iphi]
        vmax = max(1e-12, np.abs(lab).max())
        for j, (arr, ttl, vm) in enumerate((
                (lab, f"label {nm}", vmax), (pre, f"pred {nm}", vmax),
                (np.abs(pre - lab), f"|err| {nm}", None))):
            im = ax[i, j].imshow(arr, origin="lower", aspect="auto",
                                 extent=ext, cmap="RdBu_r" if vm else "magma",
                                 vmin=-vm if vm else None, vmax=vm)
            ax[i, j].set_title(ttl, fontsize=9)
            ax[i, j].set_xlabel("z [cm]"); ax[i, j].set_ylabel("r [cm]")
            fig.colorbar(im, ax=ax[i, j], shrink=0.85)
    fig.suptitle(f"Irisan phi indeks {iphi} (data sintetik = uji mekanika)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------------- [BARU v2] keluaran .root (uproot) ------------------
def save_root_output(cx, rho, preds, labels, path, logger):
    """SUBROUTINE save_root_output
    PURPOSE  : Tulis keluaran .root (TTree 'validation': phi,r,z, rho_fluc,
               pred_*, label_*) via uproot - TANPA instalasi ROOT penuh;
               bisa dibuka di ROOT/TBrowser milik pengguna.
    CATATAN  : skema branch TIDAK meniru persis file internal repo asli."""
    try:
        import uproot
    except ImportError:
        logger.warning("uproot tidak terpasang -> keluaran .root dilewati "
                       "(pip install uproot).")
        return
    P, R, Z = cx.grid_shape()
    ip, ir, iz = np.meshgrid(np.arange(P), np.arange(R), np.arange(Z),
                             indexing="ij")
    br = {"phi": cx.phi[ip].ravel().astype(np.float32),
          "r": cx.r[ir].ravel().astype(np.float32),
          "z": cx.z[iz].ravel().astype(np.float32),
          "rho_fluc": np.asarray(rho).ravel().astype(np.float32)}
    for nm, p, l in zip(("dR", "dRPhi", "dZ"), preds, labels):
        br["pred_" + nm] = np.asarray(p).ravel().astype(np.float32)
        br["label_" + nm] = np.asarray(l).ravel().astype(np.float32)
    with uproot.recreate(path) as f:
        f["validation"] = br
    logger.info("Keluaran ROOT: %s (TTree 'validation')", path)


class DnnOptimiser:
    """SUBROUTINE DnnOptimiser (STRUKTURAL v2)
    PURPOSE  : Antarmuka kompatibel steer_analysis (train/apply/...).
    DIPANGGIL: steer_analysis.init_models."""
    name = "dnn"

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()
        self.logger.info("DnnOptimiser STRUKTURAL v2 (HodgeNet + Hodge-PINN "
                         "+ MC + plot + root)")
        zr = config.z_range
        self.cx = CylGridComplex(config.grid_phi, config.grid_r,
                                 config.grid_z, rmin=RMIN_CM, rmax=RMAX_CM,
                                 zmin=float(zr[0]),
                                 zmax=min(float(zr[1]), 249.7), eps=1.0)
        if getattr(config, "network", "hodge_unet") == "hodge_unet":
            # v4: bentuk autoencoder U (encoder-decoder + skip) lapisan Hodge
            self.model = HodgeUNet(self.cx, hidden=config.hodge_hidden,
                                   levels=config.unet_levels,
                                   blocks=config.unet_blocks,
                                   growth=config.unet_channel_growth)
            self.logger.info("Arsitektur: HodgeUNet (v4) levels=%d blocks=%d "
                             "growth=%d", config.unet_levels,
                             config.unet_blocks, config.unet_channel_growth)
        else:  # 'hodge_flat' = arsitektur datar v3 (untuk ABLASI v3 vs v4)
            self.model = HodgeNet(self.cx, hidden=config.hodge_hidden,
                                  nblocks=config.hodge_nblocks)
            self.logger.info("Arsitektur: HodgeNet datar (v3/ablasi)")
        self.e0 = 1.0 if config.synthetic else config.e0_vcm
        self.distorter = make_torch_distorter(self.cx, config.omega_tau,
                                              self.e0)
        self.crit = StructuralPINNLoss(self.cx, self.distorter,
                                       lam_data=config.lam_data,
                                       lam_pde=config.lam_pde,
                                       residual_mode=config.residual_mode)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=config.adamlr)
        self.model_path = os.path.join(config.dirmodel,
                                       "hodgenet_struktural.pt")
        # n_ev utk penamaan file; mode riil diisi saat train() berjalan
        self.n_ev = int(config.synthetic_events) if config.synthetic else 0

    def _make_rho(self, seed):
        cfg = self.config
        if cfg.synthetic_mode == "mc":
            return synthetic_event_mc(self.cx, seed=seed,
                                      n_pileup=cfg.mc_pileup,
                                      mult_rel_sigma=cfg.mc_mult_rel_sigma,
                                      charge_rel_sigma=cfg.mc_charge_rel_sigma)
        return synthetic_event(self.cx, seed=seed)

    def _get_event(self, index, synthetic_seed=None):
        cfg = self.config
        if cfg.synthetic:
            rho = self._make_rho(synthetic_seed)
            A = build_laplacian(self.cx)
            phi_ref = solve_poisson_dirichlet0(A, rho.ravel(), self.cx.v_dual,
                                               self.cx.interior_mask)
            s = phi_ref[self.cx.interior_mask].std() + 1e-30
            rho, phi_ref = rho / s, phi_ref / s
            Er, Ep, Ez = e_field_nodes(self.cx,
                                       phi_ref.reshape(self.cx.grid_shape()))
            labels = distortions_numpy(self.cx, Er, Ep, Ez,
                                       cfg.omega_tau, self.e0)
        else:
            _, rho, labels = load_event_grid(cfg.dirinput_train, index)
            self.logger.warning("Data riil: kalibrasi satuan rho/E0 masih "
                                "PLACEHOLDER - verifikasi vs O2.")
        rho_t = torch.tensor(np.ascontiguousarray(rho.ravel()),
                             dtype=torch.float32)
        lab_t = [torch.tensor(np.ascontiguousarray(l), dtype=torch.float32)
                 for l in labels]
        return rho, rho_t, lab_t

    def train(self):
        """SUBROUTINE train (v3 - EPOCH SEJATI)
        PURPOSE  : Setara semantik kode asli (dnn_optimiser.py:80 Keras
                   model.fit(epochs=config.epochs)): satu epoch = satu pass
                   SELURUH dataset event. Dataset dibangun SEKALI (cache RAM)
                   lalu diulang config.epochs kali; urutan event diacak tiap
                   epoch (meniru shuffle default Keras). Kunci steps_per_event
                   v2 DIHAPUS demi kesetaraan parameter dengan kode asli.
        CATATAN  : cache RAM ~ 4 array x n_node x 4 B per event; grid produksi
                   180x33x33 -> ~3 MB/event. History & plot per-EPOCH sehingga
                   kurva sebanding sumbu-x dengan Fig.3 paper ALICE.
        DIPANGGIL: steer_analysis (dotrain)."""
        cfg = self.config
        if cfg.synthetic:
            indices = list(range(cfg.synthetic_events)); tag = "SINTETIK"
        else:
            lo, hi = cfg.range_rnd_index_train
            n = min(int(cfg.train_events), hi + 1 - lo)
            indices = [lo + i for i in range(n)]; tag = "RIIL"
        dataset = []
        for idx in indices:
            _, rho_t, lab_t = self._get_event(idx, synthetic_seed=idx)
            dataset.append((idx, rho_t, lab_t))
        self.logger.info("Dataset siap: %d event (%s); epochs=%d",
                         len(dataset), tag, cfg.epochs)
        rng = np.random.default_rng(12345)
        order = np.arange(len(dataset))
        hist = []
        for ep in range(cfg.epochs):
            rng.shuffle(order)               # shuffle per epoch (ala Keras)
            sum_d = sum_p = 0.0; last_cert = 0.0
            for k in order:
                _, rho_t, lab_t = dataset[k]
                self.opt.zero_grad()
                phi = self.model(rho_t)
                loss, logs = self.crit(phi, rho_t, lab_t)
                loss.backward()
                self.opt.step()
                sum_d += logs["L_data"]; sum_p += logs["L_pde"]
                last_cert = logs["gauss_cert"]
            nb = max(1, len(dataset))
            rec = {"epoch": ep, "L_data": sum_d / nb, "L_pde": sum_p / nb,
                   "gauss_cert": last_cert}
            hist.append(rec)
            if ep % max(1, cfg.epochs // 10) == 0 or ep == cfg.epochs - 1:
                self.logger.info("[epoch %d/%d] L_data=%.4e L_pde=%.4e "
                                 "gauss=%.1e", ep + 1, cfg.epochs,
                                 rec["L_data"], rec["L_pde"],
                                 rec["gauss_cert"])
        os.makedirs(cfg.dirmodel, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        with open(os.path.join(cfg.dirmodel, "hodgenet_history.json"),
                  "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=1)
        self.logger.info("Model tersimpan: %s", self.model_path)
        if cfg.make_plots:
            os.makedirs(cfg.dirplots, exist_ok=True)
            p = os.path.join(cfg.dirplots,
                             f"learning_plot_hodgenet_nEv{self.n_ev}.png")
            plot_learning(hist, p)
            self.logger.info("Plot kurva belajar (per-epoch): %s", p)

    def apply(self):
        cfg = self.config
        if os.path.isfile(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path,
                                                  weights_only=True))
        else:
            self.logger.warning("Model terlatih tidak ditemukan; memakai "
                                "bobot saat ini.")
        idx = 9999 if cfg.synthetic else cfg.range_rnd_index_nd_val[0]
        rho, rho_t, lab_t = self._get_event(idx, synthetic_seed=idx)
        with torch.no_grad():
            phi = self.model(rho_t)
            dR, dP, dZ = self.distorter(phi.view(*self.cx.grid_shape()))
        preds = [t.numpy() for t in (dR, dP, dZ)]
        labels = [t.numpy() for t in lab_t]
        os.makedirs(cfg.dirapply, exist_ok=True)
        out = {}
        for nm, p, l in zip(("dR", "dRPhi", "dZ"), preds, labels):
            np.save(os.path.join(cfg.dirapply, f"pred_{nm}.npy"), p)
            np.save(os.path.join(cfg.dirapply, f"label_{nm}.npy"), l)
            out[f"rmse_{nm}"] = float(np.sqrt(((p - l) ** 2).mean()))
        with open(os.path.join(cfg.dirapply, "apply_metrics.json"),
                  "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        self.logger.info("Apply selesai: %s", out)
        if cfg.make_plots:
            os.makedirs(cfg.dirplots, exist_ok=True)
            p = os.path.join(cfg.dirplots,
                             f"maps_hodgenet_nEv{self.n_ev}.png")
            plot_maps(self.cx, preds, labels, p, iphi=0)
            self.logger.info("Plot peta distorsi: %s", p)
        if cfg.save_root:
            rp = os.path.join(cfg.dirapply,
                              f"output_hodgenet_nEv{self.n_ev}.root")
            save_root_output(self.cx, rho, preds, labels, rp, self.logger)

    def plot(self):
        self.logger.warning("plot() jalur validasi asli (ROOT/RootInteractive)"
                            " tidak diporting; pakai PNG matplotlib & .root "
                            "uproot dari train()/apply().")

    def draw_profile(self, events_counts):
        self.logger.warning("draw_profile() tidak diporting di versi "
                            "struktural v2.")

    def search_grid(self):
        raise NotImplementedError("search_grid belum diporting (v2).")

    def bayes_optimise(self):
        raise NotImplementedError("bayes_optimise belum diporting (v2).")
