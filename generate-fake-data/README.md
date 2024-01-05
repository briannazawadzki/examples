# Generate Fake Data

This example generates fake visibility data from a mock image and a set of example baselines. It uses the IM Lup DSHARP dataset for realistic baseline and weight values and source flux. This particular fake dataset is used as a test fixture within the MPoL test suite.

* `create_butterfly.py` downloads a nice looking image from the `ceyda/smithsonian_butterflies` collection, uses PIL to greyscale and crop it, adjusts the flux value to match DSHARP IM Lup, then saves it as a numpy array.
* `export_baselines.py` uses MPoL-dev/visread and casatools to extract real baselines from the IM Lup measurement set, and saves them as a numpy array. To save space, we take <5% of the visibilities.
* `package_data.py` combines the two numpy arrays into a single archive, saved as `float32` to save space.
* `Snakefile` is a snakemake file setting up the workflow. From this directory, you can run 

```
$ snakemake -c 1 all
```