import tensorflow as tf
import config
import math


class FinetuningScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps=4000):
        super().__init__()

        self.warmup_steps = warmup_steps
        self.target_warmup = 0.0001
        self.initial_learning_rate = 0.0
        self.decay_steps = 100000.
        self.alpha = 0.000000001


    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        return tf.cond(
            step < self.warmup_steps,
            lambda: self.warmup_learning_rate(step),
            lambda: self.decay_learning_rate(step)
        )

    def warmup_learning_rate(self, step):
        completed_fraction = step / self.warmup_steps
        total_delta = self.target_warmup - self.initial_learning_rate
        return completed_fraction * total_delta

    def decay_learning_rate(self, step):
        step = tf.cond(
            step < self.decay_steps,
            lambda: step,
            lambda: self.decay_steps
        )
        cos_val = math.pi * step / self.decay_steps
        cos = tf.math.cos(cos_val)
        cosine_decay = 0.5 * (1 + cos)
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.target_warmup * decayed


    def get_config(self):
        return {
            'warmup_steps': self.warmup_steps,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)



