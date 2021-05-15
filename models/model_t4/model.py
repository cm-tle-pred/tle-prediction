import torch
import torch.nn as nn

class NNModelEx(nn.Module):
    def __init__(self, inputSize, outputSize, *args):
        super().__init__()
        network = []
        p = inputSize
        for k,v in args:
            if k.startswith('l'):
                network.append(nn.Linear(in_features=p, out_features=v))
                p=v
            elif k.startswith('d'):
                network.append(nn.Dropout(v))
            elif k.startswith('t'):
                network.append(nn.Tanh())
            elif k.startswith('s'):
                network.append(nn.Sigmoid())
            elif k.startswith('r'):
                network.append(nn.ReLU())
        network.append(nn.Linear(in_features=p, out_features=outputSize))
        self.net = nn.Sequential(*network)

    def forward(self, X):
        out = self.net(X)
        return out

class NNBiEpochBiasModel(nn.Module):
    def __init__(self, inputSize):
        super().__init__()
        self.linear_in = nn.Linear(in_features=inputSize, out_features=1)
        self.bilinear_in_e = nn.Bilinear(in1_features=inputSize, in2_features=1, out_features=1)
        self.bilinear_in = nn.Bilinear(in1_features=1, in2_features=1, out_features=1)

    def forward(self, X, X_e, X_f, X_c=None):
        in_o = self.linear_in(X)
        in_e = self.bilinear_in_e(X, X_e)
        if not X_c == None:
            in_e[:,0] = in_e[:,0] * X_c
        in_o = self.bilinear_in(in_e, in_o)
        in_o[:,0] += X_f
        return in_o


class NNBig(nn.Module):
    def __init__(self, inputSize, base_head_out, feature_head_out, base_model_config, model_config):
        super().__init__()

        self.X_cols = ['X_delta_EPOCH', 'X_EPOCH_JD_1', 'X_EPOCH_FR_1', 'X_EPOCH_JD_2',
                       'X_EPOCH_FR_2', 'X_MEAN_MOTION_DOT_1', 'X_BSTAR_1', 'X_INCLINATION_1',
                       'X_RA_OF_ASC_NODE_1', 'X_ECCENTRICITY_1', 'X_ARG_OF_PERICENTER_1',
                       'X_MEAN_ANOMALY_1', 'X_MEAN_MOTION_1', 'X_MEAN_ANOMALY_COS_1',
                       'X_MEAN_ANOMALY_SIN_1', 'X_INCLINATION_COS_1', 'X_INCLINATION_SIN_1',
                       'X_RA_OF_ASC_NODE_COS_1', 'X_RA_OF_ASC_NODE_SIN_1',
                       'X_SEMIMAJOR_AXIS_1', 'X_PERIOD_1', 'X_APOAPSIS_1', 'X_PERIAPSIS_1',
                       'X_RCS_SIZE_1', 'X_SAT_RX_1', 'X_SAT_RY_1', 'X_SAT_RZ_1', 'X_SAT_VX_1',
                       'X_SAT_VY_1', 'X_SAT_VZ_1', 'X_YEAR_1', 'X_DAY_OF_YEAR_COS_1',
                       'X_DAY_OF_YEAR_SIN_1', 'X_SUNSPOTS_1D_1', 'X_SUNSPOTS_3D_1',
                       'X_SUNSPOTS_7D_1', 'X_AIR_MONTH_AVG_TEMP_1',
                       'X_WATER_MONTH_AVG_TEMP_1'
                      ]
        self.incl_idx = self.X_cols.index('X_INCLINATION_1')
        self.ecc_idx = self.X_cols.index('X_ECCENTRICITY_1')
        self.mm_idx = self.X_cols.index('X_MEAN_MOTION_1')
        self.peri_idx = self.X_cols.index('X_ARG_OF_PERICENTER_1')
        self.raan_idx = self.X_cols.index('X_RA_OF_ASC_NODE_1')
        self.ma_idx = self.X_cols.index('X_MEAN_ANOMALY_1')
        self.epoch_diff_idx = self.X_cols.index('X_delta_EPOCH')

        self.base_out_count = base_head_out
        self.f_head_out_count = feature_head_out
        self.base_head = NNModelEx(inputSize, self.base_out_count, *base_model_config)

        self.incl_head = NNModelEx(inputSize, self.f_head_out_count, *model_config)
        self.incl_model = NNBiEpochBiasModel(self.base_out_count + self.f_head_out_count)

        self.ecc_head = NNModelEx(inputSize, self.f_head_out_count, *model_config)
        self.ecc_model = NNBiEpochBiasModel(self.base_out_count + self.f_head_out_count)

        self.mm_head = NNModelEx(inputSize, self.f_head_out_count, *model_config)
        self.mm_model = NNBiEpochBiasModel(self.base_out_count + self.f_head_out_count)

        self.raan_head = NNModelEx(inputSize, self.f_head_out_count, *model_config)
        self.raan_model = NNBiEpochBiasModel(self.base_out_count + self.f_head_out_count)

        self.peri_head = NNModelEx(inputSize, self.f_head_out_count, *model_config)
        self.peri_model = NNBiEpochBiasModel(self.base_out_count + self.f_head_out_count)

        self.ma_head = NNModelEx(inputSize, self.f_head_out_count, *model_config)
        self.ma_model = NNBiEpochBiasModel(self.base_out_count + self.f_head_out_count)

        
    def forward(self, X):
        
        base_head_out = self.base_head(X)

        incl_head_out = self.incl_head(X)
        incl_target_out = self.incl_model(
            torch.cat((base_head_out, incl_head_out), 1),
            X[:,self.epoch_diff_idx:self.epoch_diff_idx+1],
            X[:,self.incl_idx],
            None,
        )

        ecc_head_out = self.ecc_head(X)
        ecc_target_out = self.ecc_model(
            torch.cat((base_head_out, ecc_head_out), 1),
            X[:,self.epoch_diff_idx:self.epoch_diff_idx+1],
            X[:,self.ecc_idx],
            None,
        )

        mm_head_out = self.mm_head(X)
        mm_target_out = self.mm_model(
            torch.cat((base_head_out, mm_head_out), 1),
            X[:,self.epoch_diff_idx:self.epoch_diff_idx+1],
            X[:,self.mm_idx],
            None,
        )

        raan_head_out = self.raan_head(X)
        raan_target_out = self.raan_model(
            torch.cat((base_head_out, raan_head_out), 1),
            X[:,self.epoch_diff_idx:self.epoch_diff_idx+1],
            X[:,self.raan_idx],
            None,
        )

        peri_head_out = self.peri_head(X)
        peri_target_out = self.peri_model(
            torch.cat((base_head_out, peri_head_out), 1),
            X[:,self.epoch_diff_idx:self.epoch_diff_idx+1],
            X[:,self.peri_idx],
            None,
        )

        ma_head_out = self.ma_head(X)
        ma_target_out = self.ma_model(
            torch.cat((base_head_out, ma_head_out), 1),
            X[:,self.epoch_diff_idx:self.epoch_diff_idx+1],
            X[:,self.ma_idx],
            X[:,self.mm_idx],
        )
        
        return torch.cat((
            incl_target_out,
            ecc_target_out,
            mm_target_out,
            raan_target_out,
            peri_target_out,
            ma_target_out,
        ), 1)
    
    
#     ['y_INCLINATION', 'y_ECCENTRICITY', 'y_MEAN_MOTION', 'y_RA_OF_ASC_NODE_REG', 'y_ARG_OF_PERICENTER_REG', 'y_REV_MA_REG']