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
        self.linear_alt = nn.Linear(in_features=1, out_features=1)

    def forward(self, X, X_e, X_f, X_c=None):
        
        in_o = self.linear_in(X)
        
        if X_c == None:
            in_e = self.bilinear_in_e(X, X_e)
        else:
            alt_e = (X_c * X_e).unsqueeze(0)
            in_e = self.linear_alt(alt_e)
        in_o = self.bilinear_in(in_e, in_o)
        in_o[:,0] += X_f
        return in_o

class NNSingleFeatureModel(nn.Module):
    def __init__(self, inputSize, feature_head_out, feature_index, epoch_diff_index, mag_index, model_config):
        super().__init__()
        self.head_out_count = feature_head_out
        self.feature_index = feature_index
        self.epoch_diff_index = epoch_diff_index
        self.mag_index = mag_index
        
        self.head = NNModelEx(inputSize, self.head_out_count, *model_config)
        self.model = NNBiEpochBiasModel(self.head_out_count)

    def forward(self, X):
        head_out = self.head(X)
        target_out = self.model(
            head_out,
            X[:,self.epoch_diff_index:self.epoch_diff_index+1],
            X[:,self.feature_index],
            (X[:,self.mag_index] if not self.mag_index == None else None),
        )
        return target_out

class NNBig(nn.Module):
    def __init__(self, inputSize, feature_head_out, model_config):
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

        self.incl_model = NNSingleFeatureModel(inputSize, feature_head_out, self.ma_idx, self.epoch_diff_idx, None, model_config)


        
    def forward(self, X):
        incl_out = self.incl_model(X)
        return incl_out
#         incl_head_out = self.incl_head(X)
#         incl_target_out = self.incl_model(
#             incl_head_out,
#             X[:,self.epoch_diff_idx:self.epoch_diff_idx+1],
#             X[:,self.incl_idx],
#             None,
#         )
#         return incl_target_out
#         ecc_head_out = self.ecc_head(X)
#         ecc_target_out = self.ecc_model(
#             ecc_head_out,
#             X[:,self.epoch_diff_idx:self.epoch_diff_idx+1],
#             X[:,self.ecc_idx],
#             None,
#         )

#         mm_head_out = self.mm_head(X)
#         mm_target_out = self.mm_model(
#             mm_head_out,
#             X[:,self.epoch_diff_idx:self.epoch_diff_idx+1],
#             X[:,self.mm_idx],
#             None,
#         )

#         raan_head_out = self.raan_head(X)
#         raan_target_out = self.raan_model(
#             raan_head_out,
#             X[:,self.epoch_diff_idx:self.epoch_diff_idx+1],
#             X[:,self.raan_idx],
#             None,
#         )

#         peri_head_out = self.peri_head(X)
#         peri_target_out = self.peri_model(
#             peri_head_out,
#             X[:,self.epoch_diff_idx:self.epoch_diff_idx+1],
#             X[:,self.peri_idx],
#             None,
#         )

#         ma_head_out = self.ma_head(X)
#         ma_target_out = self.ma_model(
#             ma_head_out,
#             X[:,self.epoch_diff_idx:self.epoch_diff_idx+1],
#             X[:,self.ma_idx],
#             X[:,self.mm_idx],
#         )
        
#         return torch.cat((
#             incl_target_out,
#             ecc_target_out,
#             mm_target_out,
#             raan_target_out,
#             peri_target_out,
#             ma_target_out,
#         ), 1)
    
    
# #     ['y_INCLINATION', 'y_ECCENTRICITY', 'y_MEAN_MOTION', 'y_RA_OF_ASC_NODE_REG', 'y_ARG_OF_PERICENTER_REG', 'y_REV_MA_REG']