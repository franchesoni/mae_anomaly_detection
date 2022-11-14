# [Masked Autoencoder demo](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=5555531082047)

This code uses Masked Autoencoder of He et al. for impainting a masked region of an image. [The demo is available on IPOL](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=5555531082047), where the mask can be drawn over the image. One can try the models (ViT-Large) trained with the MSE loss or with the GAN loss.

### To run locally (with a given `mask_0.png`)
Install packages (PIL, numpy, pytorch, timm) and run:
```
python main.py --input 000019.jpg --loss GAN
```

## Credit
Demo based on the official repo https://github.com/facebookresearch/mae which is under the CC-BY-NC 4.0 license. Check the official repo for more details.
