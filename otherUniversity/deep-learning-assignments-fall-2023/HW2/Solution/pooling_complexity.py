import numpy as np


def pooling(input, p_w, p_h, s_w, s_h):
    calculation_counter = 0
    w, h, c = input.shape
    w_out = 1 + (w - p_w) // s_w
    h_out = 1 + (h - p_h) // s_h
    output = np.zeros((w_out, h_out, c))
    calculation_counter += w_out * h_out * c
    for i in range(c):
        for j in range(w_out):
            for k in range(h_out):
                output[j, k, i] = np.max(
                    input[j * s_w : j * s_w + p_w, k * s_h : k * s_h + p_h, i]
                )
                calculation_counter += p_w * p_h
    return output, calculation_counter


if __name__ == "__main__":
    input = np.random.rand(1000, 1000, 3)
    p_w, p_h, s_w, s_h = 8, 6, 4, 3
    output, complexity = pooling(input, p_w, p_h, s_w, s_h)
    theoretical_complexity = float(1000 * 1000 * 3 * p_w * p_h) / float(s_w * s_h)
    print(
        output.shape,
        complexity,
        theoretical_complexity,
        float(complexity) / theoretical_complexity,
    )
