import os
from PIL import Image
import numpy as np

# if you need to access a file next to the source code, use the variable ROOT
# for example:
#    torch.load(os.path.join(ROOT, 'weights.pth'))
ROOT = os.path.dirname(os.path.realpath(__file__))

def minmaxnorm(x):
    return (x - x.min()) / (x.max() - x.min())

def main(input, loss, output):
    u = np.array(Image.open(input))
    print("hello world", u.shape)
    print(os.listdir('.'))
    print(loss)

    v = u + np.random.randn(*u.shape) * 30

    out = (minmaxnorm(v)*255).astype(np.uint8)
    Image.fromarray(out).save(output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--loss", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    main(args.input, args.loss, args.output)
