"""
[REVIEW] FILE DIGANTI TOTAL oleh jalur struktural TDL+PINN.
Kode ASLI = baris awal file s.d. penanda KODE-BARU (117 baris, tiap baris berawalan "# OLD| ").
KODE BARU dimulai setelah penanda.
"""

# OLD| """
# OLD| Utilities for constructing U-Net.
# OLD| """
# OLD| # pylint: disable=too-many-arguments, invalid-name
# OLD| # pylint: disable=consider-using-f-string
# OLD| 
# OLD| from keras.models import Model
# OLD| from keras.layers import Input, concatenate, UpSampling3D
# OLD| from keras.layers import AveragePooling3D, Conv3DTranspose
# OLD| from keras.layers import Dropout
# OLD| from keras.layers import BatchNormalization
# OLD| from keras.layers import Conv3D, MaxPooling3D
# OLD| 
# OLD| from tpcwithdnn.symmetry_padding_3d import SymmetryPadding3d
# OLD| 
# OLD| #https://github.com/mimrtl/DeepRad-Tools/blob/master/Examples/Unet.py
# OLD| def conv_block(m, dim, activation, batchnorm, residual, dropout=0):
# OLD|     """
# OLD|     Create a convolutional block of layers.
# OLD| 
# OLD|     ;param keras.Layer m: the previous layer
# OLD|     :param int dim: dimension of the convolutional layer,
# OLD|                     the convolution kernel will have dimension dim x dim x dim
# OLD|     :param str activation: name of the activation function
# OLD|     :param bool batchnorm: whether to apply batch normalization
# OLD|     :param bool residual: whether to apply residual connection
# OLD|     :param double dropout: dropout magnitude from the range [0, 1), 0 means no dropout. default: 0
# OLD|     :return: a convolutional block
# OLD|     """
# OLD|     n = Conv3D(dim, 3, activation=activation, padding='same', kernel_initializer="normal")(m)
# OLD|     n = BatchNormalization()(n) if batchnorm else n
# OLD|     n = Dropout(dropout)(n) if dropout else n
# OLD|     n = Conv3D(dim, 3, activation=activation, padding='same', kernel_initializer="normal")(n)
# OLD|     n = BatchNormalization()(n) if batchnorm else n
# OLD|     return concatenate([m, n]) if residual else n
# OLD| 
# OLD| def level_block(m, dim, depth, inc_rate, activation, dropout, batchnorm, pool_type,
# OLD|                 upconv, residual):
# OLD|     """
# OLD|     Recursively create U-Net blocks, starting from the top level (network input / output)
# OLD|     and increasing depth by one with each recursive call.
# OLD| 
# OLD|     :param keras.Layer m: the network input layer
# OLD|     :param int dim: dimension of the convolutional layer,
# OLD|                     the convolution kernel will have dimension dim x dim x dim
# OLD|     :param int depth: number of convolutional blocks
# OLD|     :param double inc_rate: rate of dimension increment for the convolutional block
# OLD|                             E.g., with inc_rate 2, if the first conv block has layers 4x4x4,
# OLD|                             the next blocks will have layers 8x8x8, 16x16x16, and so on.
# OLD|     :param str activation: name of the activation function in the convolutional layers
# OLD|     :param double dropout: dropout magnitude from the range [0, 1), 0 means no dropout
# OLD|     :param bool batchnorm: whether to apply batch normalization
# OLD|     :param str pool_type: short name of the pooling function, from: max, avg, conv
# OLD|     :param bool upconv: whether to use the upsampling or transposed convolution
# OLD|     :param bool residual: whether to apply residual connection
# OLD|     :return: U-Net network model without the very first input and the very last output layer
# OLD|     :rtype: keras.Model
# OLD|     """
# OLD|     if depth > 0:
# OLD|         n = conv_block(m, dim, activation, batchnorm, residual)
# OLD|         if pool_type == "max":
# OLD|             m = MaxPooling3D(pool_size=(2, 2, 2))(n)
# OLD|         elif pool_type == "avg":
# OLD|             m = AveragePooling3D(pool_size=(2, 2, 2))(n)
# OLD|         else: # pool_type == "conv"
# OLD|             Conv3D(dim, 3, strides=2, padding='same')(n)
# OLD| 
# OLD|         m = level_block(m, int(inc_rate*dim), depth-1, inc_rate, activation, dropout, batchnorm,
# OLD|                         pool_type, upconv, residual)
# OLD| 
# OLD|         if upconv:
# OLD|             m = UpSampling3D(size=(2, 2, 2))(m)
# OLD|             diff_phi = n.shape[1] - m.shape[1]
# OLD|             diff_r = n.shape[2] - m.shape[2]
# OLD|             diff_z = n.shape[3] - m.shape[3]
# OLD|             padding = [[int(diff_phi), 0], [int(diff_r), 0], [int(diff_z), 0]]
# OLD|             if diff_phi != 0:
# OLD|                 m = SymmetryPadding3d(padding=padding, mode="SYMMETRIC")(m)
# OLD|             elif (diff_r != 0 or diff_z != 0):
# OLD|                 m = SymmetryPadding3d(padding=padding, mode="CONSTANT")(m)
# OLD|         else:
# OLD|             m = Conv3DTranspose(dim, 3, strides=2, activation=activation,
# OLD|                                 padding='same')(m)
# OLD|         n = concatenate([n, m])
# OLD|         m = conv_block(n, dim, activation, batchnorm, residual)
# OLD|     else:
# OLD|         m = conv_block(m, dim, activation, batchnorm, residual, dropout)
# OLD|     return m
# OLD| 
# OLD| def u_net(input_shape, start_channels=4, depth=4, inc_rate=2.0, activation="relu", dropout=0.2,
# OLD|           batchnorm=False, pool_type="max", upconv=True, residual=False):
# OLD|     """
# OLD|     Build an U-Net network of given size and type.
# OLD| 
# OLD|     :param int input_shape: shape of the input vector (phi x r x z x dim_input from the config file)
# OLD|     :param int start_channels: size of the initial convolutional layer, default: 4
# OLD|                                the convolution kernel will have dimension dim x dim x dim
# OLD|     :param int depth: number of convolutional blocks, default: 4
# OLD|     :param double inc_rate: rate of dimension increment for the convolutional block, default: 2.0
# OLD|                             E.g., if the first conv block has layers 4x4x4, the next blocks
# OLD|                             will have layers 8x8x8, 16x16x16, and so on.
# OLD|     :param str activation: name of the activation function in the convolutional layers,
# OLD|                            default: relu
# OLD|     :param double dropout: dropout magnitude from the range [0, 1), 0 means no dropout,
# OLD|                            default: 0.2
# OLD|     :param bool batchnorm: whether to apply batch normalization, default: False
# OLD|     :param str pool_type: short name of the pooling function, from: max, avg, conv, default: max
# OLD|     :param bool upconv: whether to use the upsampling or transposed convolution, default: True
# OLD|     :param bool residual: whether to apply residual connection, default: False
# OLD|     :return: Full U-Net network model
# OLD|     :rtype: keras.Model
# OLD|     """
# OLD|     i = Input(shape=input_shape)
# OLD|     output = level_block(i, start_channels, depth, inc_rate, activation, dropout, batchnorm,
# OLD|                          pool_type, upconv, residual)
# OLD|     output = Conv3D(1, 1, activation="linear", padding="same", kernel_initializer="normal")(output)
# OLD|     return Model(inputs=i, outputs=output)

# ======================= KODE BARU =======================
"""
SUBROUTINE-MODULE dnn_utils  (VERSI STRUKTURAL TDL+PINN, v4 - Hodge-U-Net)
PURPOSE  : Dulu: arsitektur U-Net Keras (CNN encoder-decoder + skip).
           v3: HodgeNet datar (message passing node<->edge, lebar tetap).
           v4: HodgeUNet - BENTUK autoencoder encoder-decoder ala U-Net asli
           tapi dengan LAPISAN Hodge/simplicial: encoder me-restriksi fitur
           ke mesh makin kasar (menangkap struktur GLOBAL jarak-jauh),
           decoder mem-prolongasi kembali + skip-concat (memulihkan detail
           LOKAL) - menyerang jeratan global-vs-lokal yang membuat U-Net
           asli gagal pada uji line-charge (Fig.5 paper ALICE).
           HodgeNet (v3) DIPERTAHANKAN untuk ablasi (config network:
           hodge_flat vs hodge_unet).
REFERENSI: bentuk-U dari kode asli (dnn_utils Keras); transfer antar level =
           prolongasi trilinear periodik-phi + restriksi ternormalisasi
           (gaya multigrid; partisi satuan eksak - lihat symmetry_padding_3d);
           pelebaran kanal per level ala U-Net asli (unet_channel_growth).
STATUS   : v4 BELUM dieksekusi (hanya py_compile + cek konstruksi operator)
           sesuai instruksi; operator sparse disimpan sebagai atribut biasa
           (CPU; tidak ikut state_dict - dibangun ulang dari geometri).
DIPANGGIL: dnn_optimiser (dipilih via config.network).
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from tpcwithdnn.symmetry_padding_3d import (make_hierarchy,
                                            build_prolongation,
                                            build_restriction)


def sp2torch(S):
    """SUBROUTINE sp2torch | PURPOSE: scipy.sparse -> torch.sparse_coo.
    STATUS: stabil. DIPANGGIL: HodgeNet, HodgeUNet, dnn_optimiser."""
    S = S.tocoo()
    idx = torch.tensor(np.vstack([S.row, S.col]), dtype=torch.long)
    val = torch.tensor(S.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, S.shape).coalesce()


class HodgeBlock(nn.Module):
    """SUBROUTINE HodgeBlock | PURPOSE: satu blok tukar node->edge->node:
    e = G h ; e' = akt(Lin(e)) ; h <- h + akt(Lin(G^T star_n e')) + Lin(h).
    STATUS: teruji (v2). DIPANGGIL: HodgeNet, HodgeUNet."""

    def __init__(self, G, GtS, hidden):
        super().__init__()
        self.G, self.GtS = G, GtS
        self.lin_e = nn.Linear(hidden, hidden)
        self.lin_n = nn.Linear(hidden, hidden)
        self.lin_s = nn.Linear(hidden, hidden)
        self.act = nn.SiLU()

    def forward(self, h):
        e = torch.sparse.mm(self.G, h)
        e = self.act(self.lin_e(e))
        m = torch.sparse.mm(self.GtS, e)
        return h + self.act(self.lin_n(m)) + self.lin_s(h)


def _level_ops(cx):
    s = cx.star / cx.star.mean()
    return sp2torch(cx.G), sp2torch((cx.G.T @ sp.diags(s)).tocsr())


def _geo_features(cx):
    ip, ir, iz = np.meshgrid(np.arange(cx.nphi), np.arange(cx.nr),
                             np.arange(cx.nz), indexing="ij")
    return np.stack([(cx.r[ir] / cx.r.max()).ravel(),
                     (cx.z[iz] / cx.z.max()).ravel(),
                     cx.dist_boundary], 1)


class HodgeNet(nn.Module):
    """SUBROUTINE HodgeNet (v3, DATAR - dipertahankan utk ablasi)
    PURPOSE  : rho (+ fitur r, z, jarak-batas) -> Phi pada node;
               Phi = 0 keras di konduktor (mask) = BC struktural.
    DIPANGGIL: dnn_optimiser (config network: hodge_flat)."""

    def __init__(self, cx, hidden=32, nblocks=4):
        super().__init__()
        self.cx = cx
        G, GtS = _level_ops(cx)
        geo = _geo_features(cx)
        self.register_buffer("geo", torch.tensor(geo, dtype=torch.float32))
        self.register_buffer("mask", torch.tensor(
            cx.interior_mask.astype(np.float32))[:, None])
        self.inp = nn.Linear(1 + geo.shape[1], hidden)
        self.blocks = nn.ModuleList(HodgeBlock(G, GtS, hidden)
                                    for _ in range(nblocks))
        self.out = nn.Linear(hidden, 1)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, rho_flat):
        x = torch.cat([rho_flat[:, None], self.geo], 1)
        h = torch.tanh(self.inp(x))
        for b in self.blocks:
            h = b(h)
        phi = self.out(h) * self.scale
        return (phi * self.mask).squeeze(1)


class HodgeUNet(nn.Module):
    """SUBROUTINE HodgeUNet (v4 - BENTUK AUTOENCODER ala U-Net asli)
    PURPOSE  : Encoder: [HodgeBlock x b] -> simpan skip -> restriksi R ke
               mesh kasar (+Linear pelebaran kanal). Bottleneck: HodgeBlock
               di mesh terkasar. Decoder: prolongasi P (+Linear) -> concat
               skip -> Linear merge -> [HodgeBlock x b]. Kepala: Linear ->
               Phi; Dirichlet-0 keras (mask) di konduktor.
    CATATAN  : bila grid kecil, level terdalam bisa mentok di floor dimensi
               (transfer mendekati identitas) - sah tapi boros; pada grid
               produksi 180x33x33: (180,33,33)->(90,17,17)->(45,9,9).
    DIPANGGIL: dnn_optimiser (config network: hodge_unet)."""

    def __init__(self, cx, hidden=32, levels=3, blocks=1, growth=2):
        super().__init__()
        self.cx = cx
        hier = make_hierarchy(cx, max(2, int(levels)))
        self.nlev = len(hier)
        widths = [int(hidden) * int(growth) ** l for l in range(self.nlev)]
        ops = [_level_ops(h) for h in hier]
        self.P_ops, self.R_ops = [], []
        for l in range(self.nlev - 1):
            Pm = build_prolongation(hier[l], hier[l + 1])
            self.P_ops.append(sp2torch(Pm))
            self.R_ops.append(sp2torch(build_restriction(Pm)))
        geo = _geo_features(cx)
        self.register_buffer("geo", torch.tensor(geo, dtype=torch.float32))
        self.register_buffer("mask", torch.tensor(
            cx.interior_mask.astype(np.float32))[:, None])
        self.inp = nn.Linear(1 + geo.shape[1], widths[0])

        def make_blocks(l):
            G, GtS = ops[l]
            return nn.Sequential(*[HodgeBlock(G, GtS, widths[l])
                                   for _ in range(max(1, int(blocks)))])
        self.enc = nn.ModuleList(make_blocks(l) for l in range(self.nlev))
        self.down = nn.ModuleList(nn.Linear(widths[l], widths[l + 1])
                                  for l in range(self.nlev - 1))
        self.up = nn.ModuleList(nn.Linear(widths[l + 1], widths[l])
                                for l in range(self.nlev - 1))
        self.merge = nn.ModuleList(nn.Linear(2 * widths[l], widths[l])
                                   for l in range(self.nlev - 1))
        self.dec = nn.ModuleList(make_blocks(l) for l in range(self.nlev - 1))
        self.out = nn.Linear(widths[0], 1)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.act = nn.SiLU()

    def forward(self, rho_flat):
        x = torch.cat([rho_flat[:, None], self.geo], 1)
        h = torch.tanh(self.inp(x))
        skips = []
        for l in range(self.nlev - 1):                       # ENCODER
            h = self.enc[l](h)
            skips.append(h)
            h = self.act(self.down[l](torch.sparse.mm(self.R_ops[l], h)))
        h = self.enc[self.nlev - 1](h)                       # BOTTLENECK
        for l in reversed(range(self.nlev - 1)):             # DECODER
            h = self.act(self.up[l](torch.sparse.mm(self.P_ops[l], h)))
            h = self.act(self.merge[l](torch.cat([h, skips[l]], 1)))
            h = self.dec[l](h)
        phi = self.out(h) * self.scale
        return (phi * self.mask).squeeze(1)                  # Dirichlet-0
