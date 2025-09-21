import numpy as np

x_seq = [0.1, 0.5, -0.3, 0.7]

W_xh = np.array([[0.5]])
W_hh = np.array([[0.8]])
W_hy = np.array([[1.0]])
h = np.array([[0.0]])

print("========== RNN sederhana (manual) ==========")
for t, x in enumerate(x_seq, 1):
    print(f"--- Time step {t} ---")
    
    x_t = np.array([[x]])
    h = np.tanh(np.dot(W_xh, x_t) + np.dot(W_hh, h)) # update hidden state
    y = np.dot(W_hy, h) # output

    print(f"Step {t}: Input={x:.2f}, Hidden={h.flatten()[0]:.2f}, Output={y.flatten()[0]:.2f}")
print("============================================")