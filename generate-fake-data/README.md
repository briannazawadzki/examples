# Generate Fake Data

This example generates fake visibility data from a mock image.

* start with the butterfly image, downsample / resize as with PIL. Export raw image, but also compact commands for processing as data factory in tests.

Expects that you will run commands from this directory.

- [ ] script to download image and convert to nufft-ready format
- [ ] script to combine npy baselines and numpy image into single .npz archive.

Specify the frequency as 1.3mm, no need to get it from dataset.

Yes, we took 5% of all baselines *after* channel averaging, and the datasize is 2.3 Mb w/ float64 and 1.2mb with float32. So that's acceptable and we're done now.

Assumes that from this directory, you will run

```
$ snakemake -c 1 all
```