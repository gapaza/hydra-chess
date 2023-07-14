import tensorflow as tf
import config
import math


class LinearWarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps=1000., decay_steps=20000.):
        super().__init__()

        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

        self.initial_learning_rate = 0.0
        self.target_warmup = 0.0002
        self.target__decay = 0.00002


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
        decayed = (1 - self.target__decay) * cosine_decay + self.target__decay
        return self.target_warmup * decayed


    def get_config(self):
        return {
            'warmup_steps': self.warmup_steps,
            'decay_steps': self.decay_steps
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)



