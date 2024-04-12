import math

log_prob_new = -3.09e-03
log_prob_old = -1.09e+01

prob_new = math.exp(log_prob_new)
prob_old = math.exp(log_prob_old)

print(f"Initial probability from new policy: {prob_new}")
print(f"Initial probability from old policy: {prob_old}")