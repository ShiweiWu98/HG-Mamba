import random
from torchvision.transforms import functional as TFF
from torchvision import transforms as TF


def paired_transform(
    img1,
    img2,
    crop_size=128,
    hflip=True,
    rotation=True,
    crop_prob=0.7,
):
    """
    Apply identical random augmentations to a paired image set (e.g. smoky / clear).
    All spatial decisions (crop params, flip flags) are sampled once and shared,
    guaranteeing pixel-level alignment is preserved after augmentation.
    """
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    # Decide once whether to random-crop or resize for this sample.
    # If the image is smaller than the requested crop size, fall back to resize
    # to avoid out-of-bounds crop params.
    use_crop = random.random() < crop_prob
    if use_crop:
        if img1.height < crop_size[0] or img1.width < crop_size[1]:
            use_crop = False
    if use_crop:
        # Sample crop coordinates once; the same (i, j, h, w) is reused for both images
        i, j, h, w = TF.RandomCrop.get_params(img1, output_size=crop_size)

    # Sample all augmentation flags upfront so every branch of `apply` is deterministic
    do_hflip = hflip and random.random() < 0.5
    do_vflip = rotation and random.random() < 0.5
    do_rot90 = rotation and random.random() < 0.5

    def apply(img):
        # Spatial extent: crop to a random patch, or resize the full image if crop is unavailable
        if use_crop:
            img = TFF.crop(img, i, j, h, w)
        else:
            img = TFF.resize(img, crop_size)

        if do_hflip:
            img = TFF.hflip(img)
        if do_vflip:
            img = TFF.vflip(img)
        if do_rot90:
            img = TFF.rotate(img, 90)
        img = TFF.to_tensor(img)  # convert PIL to [0, 1] float tensor
        return img

    # Apply the same transformation to both images to preserve spatial correspondence
    img1 = apply(img1)
    img2 = apply(img2)
    return img1, img2