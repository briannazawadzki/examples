import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description="Combine numpy files")
    parser.add_argument("baselines")
    parser.add_argument("img")
    parser.add_argument("output")
    args = parser.parse_args()

    b = np.load(args.baselines)
    img = np.load(args.img)
    
    np.savez(args.output, img=img, uu=b["uu"], vv=b["vv"], weight=b["weight"])


if __name__ == "__main__":
    main()
