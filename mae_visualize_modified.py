import torch
import numpy as np

from PIL import Image

import models_mae

"""### Define utils"""

# define the utils

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def per_patch_loss(self, imgs, pred, mask):
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove,
    """
    target = self.patchify(imgs)
    if self.norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = loss * mask  # put loss to zero in visible patches
    return loss


def prepare_to_save(image):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    return np.array(
        torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255)
    ).astype(np.uint8)


def prepare_model(chkpt_dir, arch="mae_vit_large_patch16"):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print(msg)
    return model


def prepare_model_dummy(arch="mae_vit_large_patch16"):
    # build model
    model = getattr(models_mae, arch)()
    return model


def run_one_image(img, mask, model):
    x = torch.tensor(img)
    mask = torch.tensor(mask)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum("nhwc->nchw", x)
    mask = mask.unsqueeze(dim=0)
    mask = torch.einsum("nhwc->nchw", mask)

    # run MAE
    loss, y, vector_mask = model(
        x.float(), mask_ratio=mask
    )  # we use mask_ratio to pass the mask
    y = model.unpatchify(y)
    y = torch.einsum("nchw->nhwc", y).detach().cpu()

    # visualize the mask
    mask = vector_mask.detach().clone()
    mask = mask.unsqueeze(-1).repeat(
        1, 1, model.patch_embed.patch_size[0] ** 2 * 3
    )  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum("nchw->nhwc", mask).detach().cpu()

    x = torch.einsum("nchw->nhwc", x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    original = prepare_to_save(x[0])
    masked = prepare_to_save(im_masked[0])
    reconstruction = prepare_to_save(y[0])
    reconstructionplusvisible = prepare_to_save(im_paste[0])
    return original, masked, reconstruction, reconstructionplusvisible, loss, vector_mask


"""### Load an image"""

# # load an image
# img = Image.open(input_img_path)
# img = img.resize((224, 224))
# img = np.array(img) / 255.

# assert img.shape == (224, 224, 3)

# normalize by ImageNet mean and std
def imagenet_normalize(img):
    img = (img - imagenet_mean) / imagenet_std
    return img


def patchify_mask(self, mask):
    """
    mask: (N, 1, H, W)
    x: (N, L, patch_size**2 *1)
    """
    p = self.patch_embed.patch_size[0]
    assert mask.shape[2] == mask.shape[3] and mask.shape[2] % p == 0

    h = w = mask.shape[2] // p
    x = mask.reshape(shape=(mask.shape[0], 1, h, p, w, p))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(mask.shape[0], h * w, p**2 * 1))
    return x


def masking_from_mask(self, x, mask):
    """follow original random_masking implementation
    mask: (N, 1, H, W)"""
    N, L, D = x.shape  # batch, length, dim
    mask = self.patchify(mask, C=1)  # (N, H*W, p*p*3)
    noise = mask.any(dim=-1)  # ascend: small is keep, large is remove
    len_keep = (~noise).sum()  # number

    # sort noise for each sample
    # npn = 1.*noise+np.arange(noise.shape[1])*0.001
    npn = noise
    npns, ids_shuffle = torch.sort(
        npn, dim=1
    )  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
    )

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


if __name__ == "__main__":
    """### Load a pre-trained MAE model"""

    # This is an MAE model trained with pixels as targets for visualization (ViT-Large, training mask ratio=0.75)

    # # download checkpoint if not exist
    # !wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth

    chkpt_dir = "mae_visualize_vit_large.pth"
    model_mae = prepare_model(chkpt_dir, "mae_vit_large_patch16")
    print("Model loaded.")

    """### Run MAE on the image"""

    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    print("MAE with pixel reconstruction:")
    run_one_image(img, model_mae)

    """### Load another pre-trained MAE model"""

    # This is an MAE model trained with an extra GAN loss for more realistic generation (ViT-Large, training mask ratio=0.75)

    # # download checkpoint if not exist
    # !wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth

    chkpt_dir = "mae_visualize_vit_large_ganloss.pth"
    model_mae_gan = prepare_model(
        "mae_visualize_vit_large_ganloss.pth", "mae_vit_large_patch16"
    )
    print("Model loaded.")

    """### Run MAE on the image"""

    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    print("MAE with extra GAN loss:")
    run_one_image(img, model_mae_gan)
