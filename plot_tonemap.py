# Standard Library
import torch
import numpy as np
import matplotlib.pyplot as plt

# Custom
from src.stylegan3.training.utils import log_transform, invert_log_transform

# Plot error
error = 0.01
data_HDR = torch.arange(0.01, 20000., 1.)
try:
    data_LDR = log_transform(data_HDR.clone())
    data_HDR_new = invert_log_transform(data_LDR)
    # data_HDR_new = invert_log_transform(data_LDR-error)
    result = torch.abs(data_HDR - data_HDR_new)
    plt.plot(data_HDR, data_LDR, color='blue', label='Log Transform')
    # plt.plot(data_HDR, result, color='red', label='Error')
except:
    print(f"Tonemapper failed!")


plt.title(f"Inverse Tone-Mapping for $LDR' = LDR-{error}$")
plt.xlabel("HDR Intensity")
plt.ylabel(f"$\Delta HDR'$ Intensity")
plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.savefig(
    "plot_tonemapping_error.png", dpi=180, metadata=None,
    bbox_inches=None, pad_inches=0.1,
    facecolor='auto', edgecolor='auto',
    backend=None
)
exit()