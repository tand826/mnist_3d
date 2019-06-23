# mnist_3d
- Plot mnist data in 3d space.

# Example
![2D plot](images/2d.png "2d")

![3D plot](images/3d.png "3d")

# Requirements
python version >= 3.6
```
pip install -r requirements.txt
```

# Run
### (1) Save mnist data as png
```
python mnist_to_png.py
```

### (2) Plot mnist data
```
python mnist_to_3dplot.py
```

# What does it do?
- Truncated SVD
- T-SNE

# Arguments
| Arguments | Default | Example |
|:-----------|:------------|:------------|
| sample_number | 100 | 1000 |
| phase | train | test |
| svd_dimension | 100 | 500 |
| out_dimension | 2 | 3 |
| sampled_count | False | (nothing) |
