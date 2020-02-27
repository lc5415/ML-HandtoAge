# visdom remote tryout

import visdom
import numpy as np

viz = vidsom.Visdom()

textwindow = viz.text('Hello World!')

viz.images(np.random.randn(20, 3, 64, 64),opts=dict(title='Random images', caption='How random.'))

viz.image(np.random.rand(3, 512, 256),
        opts=dict(title='Random image as jpg!', caption='How random as jpg.', jpgquality=50))