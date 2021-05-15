import torch
import torch.nn as nn

class NNModelEx(nn.Module):
    def __init__(self, inputSize, outputSize, *args):
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
        self.epoch_diff_idx = self.X_cols.index('X_delta_EPOCH')

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

#         network.append(nn.Linear(in_features=p, out_features=outputSize))

        self.net = nn.Sequential(*network)

        self.linear_incl = nn.Linear(in_features=p, out_features=1)
        self.bilinear_incl_e = nn.Bilinear(in1_features=p, in2_features=1, out_features=1)
        self.bilinear_incl = nn.Bilinear(in1_features=1, in2_features=1, out_features=1)

        self.linear_ecc = nn.Linear(in_features=p, out_features=1)
        self.bilinear_ecc_e = nn.Bilinear(in1_features=p, in2_features=1, out_features=1)
        self.bilinear_ecc = nn.Bilinear(in1_features=1, in2_features=1, out_features=1)

        self.linear_mm = nn.Linear(in_features=p, out_features=1)
        self.bilinear_mm_e = nn.Bilinear(in1_features=p, in2_features=1, out_features=1)
        self.bilinear_mm = nn.Bilinear(in1_features=1, in2_features=1, out_features=1)

    def forward(self, X):
        out = self.net(X)
        
        incl = self.linear_incl(out)
        incle = self.bilinear_incl_e(out, X[:,self.epoch_diff_idx:self.epoch_diff_idx+1])
        incl = self.bilinear_incl(incle, incl)
        incl[:,0] += X[:,self.incl_idx]
        
        ecc = self.linear_ecc(out)
        ecce = self.bilinear_ecc_e(out, X[:,self.epoch_diff_idx:self.epoch_diff_idx+1])
        ecc = self.bilinear_ecc(ecce, ecc)
        ecc[:,0] += X[:,self.ecc_idx]
        
        mm = self.linear_mm(out)
        mme = self.bilinear_mm_e(out, X[:,self.epoch_diff_idx:self.epoch_diff_idx+1])
        mm = self.bilinear_mm(mme, mm)
        mm[:,0] += X[:,self.mm_idx]
        
        return torch.cat((incl, ecc, mm), 1)