# Dictionary-based Robust PCA
Code corresponding to the theoretical and experimental analysis in the papers


## Phase Transition Plots in the Theory Paper

`lr` : the vector of rank values, `k` : vector of sparsity values, `d`: the size of the dictionary, and `out_folder`: the path of the output folder. 
### Entry-wise Sparsity 
```python
    run_lr_dict_ent_sp(lr, k, d, out_folder)
```
### Column-wise Sparsity
```python
	run_lr_dict_col_sp(lr, k, d, out_folder)
```
## Applications to Target Localization in Hyperspectral Imaging
For Indian Pines:
```python
load('S.mat'); load('gt.mat'); 

```
For Pavia University:
```python
load('S_pavia.mat'); load('gt_pavia.mat'); 
```

For entry-wise sparsity and column-wise sparsity run the appropriate function as:
```python
id_par = []; 
hyperSpec_func(clss, dict_lam, dict_size, out_folder, gt, S, id_par);
```

Here, the function will sweep across `100` values of regularization parameters in the the available range. 
TO selectively run specific values select the indices (between `1` and `100`) that need to be run via `id_par`.

`clss`: the class #, `dict_lam` : regularization parameter for dictionary learning step, `dict_size`: the size of the dictionary to be learned, `out_folder`: the path of the output folder, `gt`: the ground truth matrix, and `S`: the 3D-scene matrix. 

If `dict_size = 0` then the code will load the static dictionary `R.mat` in case of Pavia University and pick up voxels for the Indian Pines dataset. You will need to uncomment the matrix `R` in this case. 

