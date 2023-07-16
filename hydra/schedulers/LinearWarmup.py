import tensorflow as tf
import config
import math


class LinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps=1000., target_warmup=0.001, initial_learning_rate=0.0):
        super().__init__()

        self.warmup_steps = warmup_steps
        self.initial_learning_rate = initial_learning_rate
        self.target_warmup = target_warmup


    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        return tf.cond(
            step < self.warmup_steps,
            lambda: self.warmup_learning_rate(step),
            lambda: self.target_learning_rate(step)
        )

    def warmup_learning_rate(self, step):
        completed_fraction = step / self.warmup_steps
        total_delta = self.target_warmup - self.initial_learning_rate
        return completed_fraction * total_delta

    def target_learning_rate(self, step):
        return self.target_warmup


    def get_config(self):
        return {
            'warmup_steps': self.warmup_steps,
            'target_warmup': self.target_warmup,
            'initial_learning_rate': self.initial_learning_rate
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)



