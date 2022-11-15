import os
from types import MethodType
from PIL import Image
import numpy as np
import time

from mae_visualize_modified import (
    imagenet_normalize,
    per_patch_loss,
    masking_from_mask,
    patchify_mask,
    prepare_model,
    prepare_model_dummy,
    run_one_image,
)

# if you need to access a file next to the source code, use the variable ROOT
# for example:
#    torch.load(os.path.join(ROOT, 'weights.pth'))
ROOT = os.path.dirname(os.path.realpath(__file__))


def minmaxnorm(x):
    return (x - x.min()) / (x.max() - x.min())


def tic_tac_toe_masks():
    mask0, mask1, mask2, mask3 = np.zeros((4, 224, 224, 1))
    for i in range(14):
        for j in range(14):
            if i % 2 == 0 and j % 2 == 0:
                mask0[i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16, 0] = 1
            elif i % 2 == 0 and j % 2 == 1:
                mask1[i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16, 0] = 1
            elif i % 2 == 1 and j % 2 == 0:
                mask2[i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16, 0] = 1
            elif i % 2 == 1 and j % 2 == 1:
                mask3[i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16, 0] = 1
    return mask0, mask1, mask2, mask3


def detect_anomaly(img_path, loss):
    """Detects anomalies by measuring the difference between the original and reconstructed image."""
    img, size = load_img(img_path)
    tictactoemasks = tic_tac_toe_masks()
    model_mae = get_model_mae(loss)
    reconstruction, losses = [], []
    for i, mask in enumerate(tictactoemasks):
        (
            original,
            masked,
            reconstruction_m,
            reconstructionplusvisible,
            size,
            loss_per_patch,
            vector_mask,
        ) = reconstruct_mask(img, mask, model_mae, size)
        reconstruction.append(reconstruction_m * mask)
        losses.append((loss_per_patch * vector_mask).detach().numpy())
    reconstruction = np.array(reconstruction).sum(0)
    losses = np.array(losses).sum(0).reshape(14, 14)
    return original, reconstruction, losses, size


def load_img(img_path):
    img = Image.open(img_path).convert("RGB")
    size = img.size
    img = np.array(img.resize((224, 224))) / 255.0
    assert img.shape == (
        224,
        224,
        3,
    ), f"Expected image to be (224, 224, 3) instead of {img.shape}"
    img = imagenet_normalize(img)
    return img, size


def load_mask(mask_path):
    mask = (
        np.array(Image.open(mask_path).resize((224, 224), Image.NEAREST))[
            ..., -1
        ]
        > 0
    ).astype(bool)[..., None]
    assert mask.shape == (
        224,
        224,
        1,
    ), f"Expected mask to be (224, 224, 1) instead of {mask.shape}"
    return mask


def reconstruct_mask0(img_path, loss):
    """Runs MAE model with given loss over `input` image using `mask_0.png` as the part to ignore."""
    img, size = load_img(img_path)
    mask = load_mask(os.path.join(ROOT, "mask_0.png"))
    model_mae = get_model_mae(loss)
    return reconstruct_mask(img, mask, model_mae, size)


def get_model_mae(loss):
    st = time.time()
    mse_ckpt = os.path.join(ROOT, "mae_visualize_vit_large.pth")
    gan_ckpt = os.path.join(ROOT, "mae_visualize_vit_large_ganloss.pth")
    if loss == "MSE" and os.path.exists(mse_ckpt):
        model_mae = prepare_model(mse_ckpt, "mae_vit_large_patch16")
    elif loss == "GAN" and os.path.exists(gan_ckpt):
        model_mae = prepare_model(gan_ckpt, "mae_vit_large_patch16")
    else:
        print("No model found for loss type " + loss)
        print(f"Directory contains: {os.listdir(ROOT)}")
        print("Loading model with random weights")
        model_mae = prepare_model_dummy("mae_vit_large_patch16")

    model_mae.patchify_mask = MethodType(patchify_mask, model_mae)
    model_mae.random_masking = MethodType(masking_from_mask, model_mae)
    model_mae.forward_loss = MethodType(per_patch_loss, model_mae)
    print(f"Model loaded in {time.time()-st}s.")
    return model_mae


def reconstruct_mask(img, mask, model_mae, size):
    st = time.time()
    (
        original,
        masked,
        reconstruction,
        reconstructionplusvisible,
        loss_per_patch,
        vector_mask,
    ) = run_one_image(img, mask, model_mae)
    print("Reconstruction done in " + str(time.time() - st) + "s.")
    return (
        original,
        masked,
        reconstruction,
        reconstructionplusvisible,
        size,
        loss_per_patch,
        vector_mask,
    )


def save_outputs(size, **kwargs):
    for k, v in kwargs.items():
        Image.fromarray(v).resize(size).save(f"{k}.png")
        Image.fromarray(v).save(f"{k}rs.png")


def main_reconstruct(img_path, loss):

    (
        original,
        masked,
        reconstruction,
        reconstructionplusvisible,
        size,
        loss_per_patch,
        vector_mask,
    ) = reconstruct_mask0(img_path, loss)
    save_outputs(
        size=size,
        original=original,
        masked=masked,
        reconstruction=reconstruction,
        reconstructionplusvisible=reconstructionplusvisible,
    )


def main_anomaly(img_path, loss):
    original, reconstruction, losses, size = detect_anomaly(img_path, loss)
    save_outputs(
        size=size,
        original=original,
        reconstruction=reconstruction.astype(np.uint8),
        error=(minmaxnorm(losses) * 255).astype(np.uint8),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--loss", type=str, required=True)

    args = parser.parse_args()
    main_anomaly(args.input, args.loss)
