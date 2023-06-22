from models import U2NET
import os
from config import *

model = U2NET(3, 1).to(DEVICE)
checkpoint = U2NET_SALIENCY_MAP_CHECKPOINT_FILE
state_dict = torch.load(os.path.join(CHECKPOINT_DIR, checkpoint))
print(state_dict.keys())
model.load_state_dict(state_dict)

