import tensorflow as tf
import config


class PretrainingScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps=4000, hold_steps=40000):
        super().__init__()

        self.d_model = config.embed_dim
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)
        self.hold_steps = hold_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)

        # Warm-up Args
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        # return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


        # Hold Args
        arg1_hold = tf.math.rsqrt(self.warmup_steps)
        arg2_hold = self.warmup_steps * (self.warmup_steps ** -1.5)

        # Converge Args
        staggered_step = step - self.hold_steps
        arg1_converge = tf.math.rsqrt(staggered_step)
        arg2_converge = staggered_step * (self.warmup_steps ** -1.5)


        return tf.cond(
            step < self.warmup_steps,
            lambda: tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2),
            lambda: tf.cond(
                step < self.warmup_steps + self.hold_steps,
                lambda: tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1_hold, arg2_hold),
                lambda: tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1_converge, arg2_converge)
            )
        )



