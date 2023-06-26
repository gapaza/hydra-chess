from keras import layers
import tensorflow as tf
import config





class ShiftedPatchTokenization(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_size = config.vt_img_size
        self.patch_size = config.vt_patch_size
        self.half_patch = self.patch_size // 2
        self.flatten_patches = layers.Reshape((config.vt_num_patches, -1))
        self.projection = layers.Dense(units=config.embed_dim)
        self.layer_norm = layers.LayerNormalization(epsilon=config.vt_epsilon)


    def __call__(self, images):

        # 1. Shift original board position (8x8x12) in 4 diagonal directions
        # - left-up, left-down, right-up, right-down
        # - each shift produces a 8x8x12 board tensor
        # - these tensors are concatenated to the original board tensor along the last dimension
        # - input shape: (batch, 8, 8, 12)
        # - output shape: (batch, 8, 8, 60)
        images = tf.concat(
            [
                images,
                self.crop_shift_pad(images, mode="left-up"),
                self.crop_shift_pad(images, mode="left-down"),
                self.crop_shift_pad(images, mode="right-up"),
                self.crop_shift_pad(images, mode="right-down"),
            ],
            axis=-1,
        )



        # 2. Extract patches from the 8x8x60 tensor
        # - input shape: (batch, 8, 8, 60)
        # - output shape: (batch, 4, 4, 240)
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],    # (1, 2, 2, 1)
            strides=[1, self.patch_size, self.patch_size, 1],  # (1, 2, 2, 1)
            rates=[1, 1, 1, 1],
            padding="VALID",
            name="extract_patches",
        )



        # 3. Flatten patches
        # - input shape: (batch, 4, 4, 240)
        # - output shape: (batch, 16, 240)
        flat_patches = self.flatten_patches(patches)



        # 4. Normalize and project patches
        # - input shape: (batch, 16, 240)
        # - output shape: (batch, 16, embed_dim)
        tokens = self.layer_norm(flat_patches)
        tokens = self.projection(tokens)
        return tokens


    def crop_shift_pad(self, images, mode):
        # Build the diagonally shifted images
        if mode == "left-up":
            crop_height = self.half_patch
            crop_width = self.half_patch
            shift_height = 0
            shift_width = 0
        elif mode == "left-down":
            crop_height = 0
            crop_width = self.half_patch
            shift_height = self.half_patch
            shift_width = 0
        elif mode == "right-up":
            crop_height = self.half_patch
            crop_width = 0
            shift_height = 0
            shift_width = self.half_patch
        else:
            crop_height = 0
            crop_width = 0
            shift_height = self.half_patch
            shift_width = self.half_patch

        # Crop the shifted images and pad them
        crop = tf.image.crop_to_bounding_box(
            images,
            offset_height=crop_height,
            offset_width=crop_width,
            target_height=self.image_size - self.half_patch,
            target_width=self.image_size - self.half_patch,
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=shift_height,
            offset_width=shift_width,
            target_height=self.image_size,
            target_width=self.image_size,
        )
        return shift_pad

