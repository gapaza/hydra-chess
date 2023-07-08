from keras import layers
import keras
import config
import numpy as np
import tensorflow as tf


class BoardEmbedding(layers.Layer):

    def __init__(self, name):
        super(BoardEmbedding, self).__init__(name=name)

        self.image_size = config.vt_img_size
        self.patch_size = config.vt_patch_size
        # self.half_patch = self.patch_size // 2
        self.half_patch = 1
        self.flatten_patches = layers.Reshape((config.vt_num_patches, -1))
        self.projection = layers.Dense(units=config.embed_dim)
        self.layer_norm = layers.LayerNormalization(epsilon=config.vt_epsilon)

    def __call__(self, images):

        # Create forth dimension for flattened board
        if config.mode in ['pt3', 'ft2']:
            images = tf.expand_dims(images, axis=-1)

        # 1. Shift original board position (8x8x12) in 4 diagonal directions
        # - left-up, left-down, right-up, right-down
        # - each shift produces a 8x8x12 board tensor
        # - these tensors are concatenated to the original board tensor along the last dimension
        # - input shape: (batch, 8, 8, 12)
        # - output shape: (batch, 8, 8, 60)
        # - output shape2: (batch, 8, 8, 108) ~ with 4 additional shifts
        # - output shape2: (batch, 8, 8, 300) ~ with 4 additional shifts
        # images = tf.concat(
        #     [
        #         images,
        #         self.crop_shift_pad2(images, mode="up"),
        #         self.crop_shift_pad2(images, mode="left"),
        #         self.crop_shift_pad2(images, mode="right"),
        #         self.crop_shift_pad2(images, mode="down"),
        #         self.crop_shift_pad2(images, mode="left-up"),
        #         self.crop_shift_pad2(images, mode="left-down"),
        #         self.crop_shift_pad2(images, mode="right-up"),
        #         self.crop_shift_pad2(images, mode="right-down"),
        #     ],
        #     axis=-1,
        # )
        images = self.get_image_stack(images)


        # 2. Extract patches from the 8x8x60 tensor
        # - input shape: (batch, 8, 8, 60)
        # - output shape: (batch, 4, 4, 240)
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],  # (1, 1, 1, 1)
            strides=[1, self.patch_size, self.patch_size, 1],  # (1, 1, 1, 1)
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


    def get_image_stack(self, images):

        def get_shift(images, vars, psize):
            crop_height, crop_width, shift_height, shift_width = vars
            crop = tf.image.crop_to_bounding_box(
                images,
                offset_height=crop_height,
                offset_width=crop_width,
                target_height=self.image_size - psize,
                target_width=self.image_size - psize,
            )
            shift_pad = tf.image.pad_to_bounding_box(
                crop,
                offset_height=shift_height,
                offset_width=shift_width,
                target_height=self.image_size,
                target_width=self.image_size,
            )
            return shift_pad


        stack = []
        for shift in self.get_box_shifts():
            stack.append(get_shift(images, shift, 1))
        # for shift in self.get_outer_box_shifts():
        #     stack.append(get_shift(images, shift, 2))
        return tf.concat(stack, axis=-1)


    def get_box_shifts(self):
        return [
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
                (1, 1, 0, 0),
                (0, 1, 1, 0),
                (1, 0, 0, 1),
                (0, 0, 1, 1)
            ]

    def get_outer_box_shifts(self):
        return [
            (0, 2, 0, 0),
            (1, 2, 0, 0),
            (2, 2, 0, 0),
            (2, 1, 0, 0),
            (2, 0, 0, 0),
            (2, 0, 0, 1),
            (2, 0, 0, 2),
            (1, 0, 0, 2),
            (0, 0, 0, 2),
            (0, 0, 1, 2),
            (0, 0, 2, 2),
            (0, 0, 2, 1),
            (0, 0, 2, 0),
            (0, 1, 2, 0),
            (0, 2, 2, 0),
            (0, 2, 1, 0)
        ]




    def crop_shift_pad(self, images, mode):
        if mode == "left":
            crop_height = 0
            crop_width = self.half_patch
            shift_height = 0
            shift_width = 0
        elif mode == "up":
            crop_height = self.half_patch
            crop_width = 0
            shift_height = 0
            shift_width = 0
        elif mode == "right":
            crop_height = 0
            crop_width = 0
            shift_height = 0
            shift_width = self.half_patch
        elif mode == "down":
            crop_height = 0
            crop_width = 0
            shift_height = self.half_patch
            shift_width = 0
        elif mode == "left-up":
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

