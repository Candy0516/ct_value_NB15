from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.squeeze(1)
        return x
    
def Get_Scale_Pos_Weight(data, sensitivity):
    Nn = len(data[data['label'] == 0])
    Np = len(data[data['label'] == 1])
    SPW = sensitivity * (Nn / Np)
    return SPW

def Get_Model(data, model_type):
    if model_type == 'ada':
        model = AdaBoostClassifier(
            n_estimators=150,
            learning_rate=0.1,
            random_state=42
        )
    elif model_type == 'sxgb':
        model = XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        scale_pos_weight=Get_Scale_Pos_Weight(data, sensitivity=1.5),
        eval_metric='logloss'
    )
    elif model_type == 'usxgb':
        model = XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        scale_pos_weight=Get_Scale_Pos_Weight(data, sensitivity=0.85),
        eval_metric='logloss'
    )
    elif model_type == 'rf':
        model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42, 
        #class_weight='balanced' #peerj
    )
    else:
        raise NameError('model_type should be \'ada\', \'sxgb\', \'usxgb\', \'rf\'')
        
    return model