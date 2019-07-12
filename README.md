# mnist_3d
- Plot mnist data in 3d space.
- Visualization Method
	- Normal Supervised Training -> T-SNE
	- Between Class Learning -> T-SNE
	- TruncatedSVD -> T-SNE

# Example
![2D plot](images/lenet2d.png "lenet2d Normal")

![3D plot](images/lenet3d.png "lenet3d Normal")

![2D plot](images/lened2d_bc.png "lenet2d BC-Learning")

![3d plot](images/lened3d_bc.png "lenet3d BC-Leaerning")

![2D plot](images/tsvd2d.png "tsvd2d")

![3D plot](images/tsvd3d.png "tsvd3d")


# Requirements
python version >= 3.6
```
pip install -r requirements.txt
```

# Run
### (1) Save mnist data as png
```
python save_mnist.py
```

### (2) Plot mnist data
```
python plot.py
```

# What does it do?
### Between Class Learning
[paper](https://arxiv.org/abs/1711.10284)
### Truncated SVD
t-SNE reduces the dimension of the input tensor quite well, but takes a long time when its dimension is high too much. Truncated SVD is recommended to use in that case for speedy processing.
### t-SNE
t-SNE is a tool to visualize high-dimensional data. See [here](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) for more information.

# Arguments
| Arguments | Default | Example | Description
|:-----------|:------------|:------------|:----------|
| sample_number | 100 | 1000 | Number of samples extracted from the whole mnist data. |
| phase | train | test | Phase of samples. |
| neural_net | False | lenet | Network to use or not to use |
| mid_dimension | 100 | 500 | Output dimension of TruncatedSVD or LeNet |
| out_dimension | 2 | 3 | Output dimension of t-SNE |
| sampled_count | False | (nothing) | Show number of each handwriting nubmers. |

```python
python plot.py -sn 1000 -p test -sd 500 -od 3 -sc
```