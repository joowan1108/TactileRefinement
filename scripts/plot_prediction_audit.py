"""Plot already-computed tactile maps using the benchmark environment."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

output = Path(sys.argv[1])
panels = np.load(output/'prediction_panels.npz')
bound = max(float(np.abs(panels[k]).max()) for k in ('target','prediction'))
fig, axes = plt.subplots(len(panels['groups']), 2, figsize=(10,12), squeeze=False)
for i, group in enumerate(panels['groups']):
    for j, key in enumerate(('target','prediction')):
        im = axes[i,j].imshow(np.concatenate(panels[key][i],axis=1),
                             cmap='coolwarm',vmin=-bound,vmax=bound)
        axes[i,j].set_title(f'{group}: {key} (left | right)')
        axes[i,j].axis('off')
fig.colorbar(im,ax=axes.ravel().tolist(),label='Signed normalized depth change',shrink=.7)
fig.savefig(output/'change_predictions.png',dpi=140,bbox_inches='tight')
plt.close(fig)
