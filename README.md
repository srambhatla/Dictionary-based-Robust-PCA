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


## Copyright & License Notice
DRPCA is copyrighted by the Regents of the University of Minnesota. It can be freely used for educational and research purposes by non-profit institutions and US government agencies only. Other organizations are allowed to use DRPCA only for evaluation purposes, and any further uses will require prior approval. The software may not be sold or redistributed without prior approval. One may make copies of the software for their use provided that the copies, are not sold or distributed, are used under the same terms and conditions.
As unestablished research software, this code is provided on an "as is" basis without warranty of any kind, either expressed or implied. The downloading, or executing any part of this software constitutes an implicit agreement to these terms. These terms and conditions are subject to change at any time without prior notice. 
 The software is also available via a standard negotiated license agreement. Contact umotc@umn.edu for specific details.
