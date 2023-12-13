"""
Find scale factor for target aspect ratio
"""

def aspect_ratios(multiple: float = 32, target_ratio: float = 16/9, threshold: float = 0.02, num_iters: int = 32):

    for i in range(1, num_iters):

        for j in range(1, i):

            high = i * multiple
            low = j * multiple

            ratio = high / low

            if abs(ratio - target_ratio) < threshold:
                print(f'{high}x{low} with ratio: {ratio}')



if __name__ == '__main__':

    aspect_ratios()