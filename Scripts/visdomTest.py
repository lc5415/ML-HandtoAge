# visdom remote tryout
# PROBLEM IS: by submitting a job to set the visdom.sever, i don't know where that is running hence I lose that information 
import visdom
import numpy as np
import sys

port_in = int(sys.argv[1])

print(port_in)

viz = visdom.Visdom(port = port_in)

textwindow = viz.text('Hello World!')

viz.images(np.random.randn(20, 3, 64, 64),opts=dict(title='Random images', caption='How random.'))

viz.image(np.random.rand(3, 512, 256),
        opts=dict(title='Random image as jpg!', caption='How random as jpg.', jpgquality=50))

        ######### WE ARE NOT USING VISDOM FOR NOW
#         # plot loss to visdom object
#         viz.line(X = np.array([epoch]),
#                  Y = np.array([train_loss.cpu().detach().numpy() if use_cuda else train_loss]),
#                  win="Train", update = "append",
#                  opts=dict(xlabel='Epochs', ylabel='Loss', title='Training Loss', legend=['Loss']))

#         # plot loss to visdom object
#         viz.line(X=np.array([epoch]),
#                  Y=np.array([test_loss]),
#                  win="Test", update="append",
#                 opts=dict(xlabel='Epochs', ylabel='Loss', title='Training Loss', legend=['Loss']))