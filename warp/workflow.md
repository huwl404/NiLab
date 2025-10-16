newest protocol can be found here:

https://hku.notion.site/Using-Warp-M-after-Relion4-28c8022e7dd680c08928c4966a405bd0

# Prerequisite

After the routine workflow in Dr. Ni’s lab, now you have aligned tilt series from IMOD (containing ***.xf*** and ***.tlt*** files), order list from emClarity (***.csv*** files), particles from other software (***.star*** file, no matter what binning factor you are using).

# Project Hierarchy

project_folder/

--job051/ (Refine3D job from Relion4)

--script/ (***.py*** script that would be used)

--tomograms/

----SMV031825_43002/ (IMOD-processed)

------SMV031825_43002.mrc (tilt series stack)

------SMV031825_43002.xf

------SMV031825_43002.tlt

------SMV031825_43002_test.csv (data collection order list, other suffix rather than *_test* is OK)

----SMV031825_43003/

......

# Workflow

`#` warp/2.0.0dev31 has severe bugs when coming to use ***M***. Check ***Ref no.8***

```shell
srun -c96 -p normal --gres=gpu:4 --x11 --pty bash
ml warp/2.0.0dev36
```

`#` The following two commands should be run in the other terminal. Replace the ***jobid*** with yours.

```shell
srun --pty --overlap --jobid 25525 bash
```

`#` use this terminal to monitor your GPU usage to adjust ***--perdevice*** and ***--perdevice_refine*** parameters.

```shell
watch -n1 nvidia-smi
```

`#` Split IMOD-processed tilt series stacks into individual MRC images using .tlt angle files.

`#` Algorithm adapted from ***Ref no.2***

```shell
python script/split_tiltstack.py -i tomograms/ -o frames/ --workers 16 --recursive
```

`#` ***--folder_data*** should point to the folder where the separated MRC images saved.

`#` after this, ***warp_frameseries.settings*** would be generated and related output would be saved in ***warp_frameseries***

```shell
WarpTools create_settings \
--folder_data frames \
--folder_processing warp_frameseries \
--output warp_frameseries.settings \
--extension "*.mrc" \
--angpix 1.571 \
--exposure 3
```

`#` use ***fs_ctf*** instead of ***fs_motion_and_ctf***, you cannot correct motion because you don’t have raw frames!

```shell
WarpTools fs_ctf \
--settings warp_frameseries.settings \
--grid 2x2x1 \
--range_max 7 \
--defocus_max 8 \
--use_sum \
--perdevice 4
```

`#` check processing statistics

```shell
WarpTools filter_quality --settings warp_frameseries.settings --histograms
```

`#` symlink motion corrected averages into an ***average*** directory inside your frame series processing directory

```shell
cd warp_frameseries
ln -s ../frames average
cd ..
```

`#` Generate .tomostar files for Warp from IMOD-processed tilt series using order CSV and frame averages.

`#` Algorithm Adapted from ***Ref no.3***

`#` Many default parameters are highly customized for Xiang Fang’s data, you really need to check them.

`#` For example, ***--csv-suffix*** and ***other data collection parameters*** can be very different.

```shell
python script/generate_tomostar.py -i tomograms/ -f frames/ -o tomostar/ --recursive --workers 16
```

`#` Override tilt angles to opposite numbers to generate tomgrams matching with Relion’s, check ***Ref no.4***

```shell
python script/invert_tlt.py -i tomograms/ --recursive
```

`#` After this, you may find ***some folder processed unsuccessfully***, which is often because of your order list is wrong.

`#` For example, you deleted some low-angle tilts and emClarity defined the first angle wrongly. To address this issue,

`#` you need to use ***generate_orderlist.py*** to generate correct order list ***.csv*** file and redo ***generate_tomostar***.

`#` PLEASE **record ahead** which tomograms you deleted low-angle tilts to help DEBUG.

`#` Tomograms without second tilt will **generate** ***matching_tomograms.star*** **with wrong** ***_rlnTomoImportFractionalDose***.

`#` ***--folder_data*** should point to the folder where the separated ***.tomostar*** files saved.

`#` After this, ***warp_tiltseries.settings*** would be generated and related output would be saved in ***warp_******tilt******series***

`#` ***--tomo_dimensions*** is the original dimension defined in emClarity in this case (***convert_emClarity2relion.m***).

`#` However, you can check ***Ref no.5*** for better dimension settings if you want to re-pick particles using Warp.

```shell
WarpTools create_settings \
--output warp_tiltseries.settings \
--folder_processing warp_tiltseries \
--folder_data tomostar \
--extension "*.tomostar" \
--angpix 1.571 \
--exposure 3 \
--tomo_dimensions 4096x4096x3000
```

`#` ***--alignment_angpix*** must be the original pixel size...

```shell
WarpTools ts_import_alignments \
--settings warp_tiltseries.settings \
--alignments tomograms/ \
--alignment_angpix 1.571
```

`#` Check Defocus Handedness.

```shell
WarpTools ts_defocus_hand --settings warp_tiltseries.settings --check
```

`#` If the correlation is **negative**, you need to run the following commented command:

```shell
WarpTools ts_defocus_hand --settings warp_tiltseries.settings --set_flip
```

`#` You should check your reconstruction, ***--angpix*** does not matter here.

`#` You could terminate this job just for checking, as long as there is ***.mrc*** file in ***warp_tiltseries/reconstruction/***.

```shell
WarpTools ts_reconstruct --settings warp_tiltseries.settings --angpix 10
relion_tomo_reconstruct_tomogram --i ImportTomo/job002/optimisation_set.star --tn SMV031825_44002 --bin 6.365 --j 48 --o SMV031825_44002_relion10Apx.mrc
```


`#` If you find **z is opposite** in Warp-reconstructed (left z 172) and Relion-reconstructed (right z 304) tomograms like this:
![warp-relion-tomo-compare.png](img%2Fwarp-relion-tomo-compare.png)
`#` you have to go back to the python script/invert_tlt.py step and try without .tlt flipping.

`#` CTF Estimation

```shell
WarpTools ts_ctf \
--settings warp_tiltseries.settings \
--range_high 7 \
--defocus_max 8 \
--perdevice 4
```

`#` check processing statistics

```shell
WarpTools filter_quality --settings warp_tiltseries.settings --histograms
```

`#` Split particles from a multi-tomogram STAR into per-tomo STARs

`#` after applying recenter and local shift, then bin coordinates. Algorithm Adapted from ***Ref no.6***

`#` In this case, the particles has been refined in bin1, so I use bin4 to match the alignment.

`#` Theoretically, the binning factor here just needs to match the ***--coords_angpix*** in the following step.

```shell
python script/generate_particlestar.py -i job051/run_data.star -o particlestar -b 4
```

`#` ***--input_directory*** should point to the folder where the per-tomo *.star* particles files saved.

`#` ***--box*** is in pixels, ***--diameter*** is in Angstroms. The re-extracted particles are saved in ***warp_tiltseries/particleseries***.

`#` ***--output_angpix*** means that I still use bin4 here to facilitate the computation and **check whether RELION can run smoothly**.

```shell
WarpTools ts_export_particles \
--settings warp_tiltseries.settings \
--input_directory particlestar \
--input_pattern "*.star" \
--coords_angpix 6.284 \
--output_star warp4relion/matching.star \
--output_angpix 6.284 \
--box 60 \
--diameter 240 \
--relative_output_paths \
--2d \
--perdevice 4
```

`#` You should check whether Warp generates ***matching_tomograms.star*** with ***wrong _rlnTomoImportFractionalDose***,

`#` and **manually correct it**. For refine3D, you should assign ***matching_optimisation_set.star*** as input and 

`#` ***copy reference map*** into this folder (no matter what pixel size). Other parameters should base on your data.

```shell
cd warp4relion
cp ../job051/run_class001.mrc ./
ml relion/5.0_gpu_ompi4_cuda111
relion --tomo&
Refine3D:
    Input optimisation set: matching_optimisation_set.star
    Reference map: run_class001.mrc
    Initial low-pass filter 60
    Symmetry C10
    Mask diameter 250
    Initial angular sampling 1.8
    Local searches from auto-sampling 1.8
    Use GPU acceleration
cd ..
```

`#` Now we have finished pre-processing in Warp and achieved 13.0 Å resolution refined bin4 map. Then we transfer to ***M***.

`#` A ***project*** in M is referred to as a ***Population***. A population contains ***at least one Data Source*** and ***at least one Species***.

`#` A ***Data Source*** contains a set of frame series or ***tilt series and their metadata***.

`#` A ***Species*** is a ***map*** that is refined, as well as ***particle metadata*** for the available data sources.

```shell
MTools create_population \
--directory m \
--name 1

MTools create_source \
--name 1 \
--population m/1.population \
--processing_settings warp_tiltseries.settings
```

`#` M requires a binary mask. ***This mask will be expanded and a soft edge will be added automatically*** during refinement.

`#` You are responsible for deciding the threshold and check whether it is right.

```shell
relion_mask_create \
--i warp4relion/Refine3D/job001/run_class001.mrc \
--o m/mask_0p02.mrc \
--ini_threshold 0.02
```

`#` Resampling particles to bin1 by assigning ***--angpix_resample***, you might need to change ***--lowpass***.

`#` The output in the command line interface would tell you the ***path where map and metadata saved***.

`#` ***versions*** folder in the above path will save refined results each time. Be wary of the potential for overfitting parameters!

`#` So you should introduce new parameters ***one by one*** when refining in M.

```shell
MTools create_species \
--population m/1.population \
--name ZIKV \
--diameter 250 \
--sym C10 \
--half1 warp4relion/Refine3D/job001/run_half1_class001_unfil.mrc \
--half2 warp4relion/Refine3D/job001/run_half2_class001_unfil.mrc \
--mask m/mask_0p02.mrc \
--particles_relion warp4relion/Refine3D/job001/run_data.star \
--angpix_resample 1.571 \
--lowpass 15
```

`#` Run an iteration of M ***without any refinements*** to check that everything imported correctly.

`#` You might encounter issue saying ***The method or operation is not implemented***, just need to reduce ***--perdevice_refine***.

```shell
MCore \
--population m/1.population \
--iter 0 \
--perdevice_refine 4
```

`#` Now we achieve ZIKV: **9.80 Å**

`#` Run an iteration of M with 2D image warp refinement, particle pose refinement and CTF refinement.

`#` ***--refine_imagewarp*** is like Relion’s Frame Alignment, aligning each tilt actually. Check ***Ref no.7*** for optimization advise.

```shell
MCore \
--population m/1.population \
--refine_imagewarp 4x4 \
--refine_particles \
--ctf_defocus \
--ctf_defocusexhaustive \
--perdevice_refine 4
```

`#` Now we achieve ZIKV: **7.98 Å**

`#` Second M Refinement with 2D Image Warp, Particle Poses Refinement and CTF Refinement.

```shell
MCore \
--population m/1.population \
--refine_imagewarp 4x4 \
--refine_particles \
--ctf_defocus \
--perdevice_refine 4
```

`#` Now we achieve ZIKV: **7.72 Å**

`#` Stage Angle Refinement

```shell
MCore \
--population m/1.population \
--refine_imagewarp 4x4 \
--refine_particles \
--refine_stageangles \
--perdevice_refine 4
```

`#` Now we achieve ZIKV: **7.60 Å**

`#` Magnification/Cs/Zernike3

```shell
MCore \
--population m/1.population \
--refine_imagewarp 4x4 \
--refine_particles \
--refine_mag \
--ctf_cs \
--ctf_defocus \
--ctf_zernike3 \
--perdevice_refine 4
```

`#` Now we achieve ZIKV: **7.46 Å**

`#` Weights (Per-Tilt Series)

```shell
EstimateWeights \
--population m/1.population \
--source 1 \
--resolve_items

MCore \
--population m/1.population \
--perdevice_refine 2
```

`#` Now we achieve ZIKV: **7.47 Å**

`#` Weights (Per-Tilt, Averaged over all Tilt Series)

```shell
EstimateWeights \
--population m/1.population \
--source 1 \
--resolve_frames

MCore \
--population m/1.population \
--refine_particles \
--perdevice_refine 2
```

`#` Now we achieve ZIKV: **7.43 Å**

`#` Temporal Pose Resolution

```shell
MTools resample_trajectories \
--population m/1.population \
--species m/species/ZIKV_86688757/ZIKV.species \
--samples 2

MCore \
--population m/1.population \
--refine_imagewarp 4x4 \
--refine_particles \
--refine_stageangles \
--refine_mag \
--ctf_cs \
--ctf_defocus \
--ctf_zernike3 \
--perdevice_refine 2
```

#### Finally we achieve ZIKV: 7.35 Å with higher resolution and more density restored!
![relion-warp-reconstruction-compare.png](img%2Frelion-warp-reconstruction-compare.png)

# Reference

1. [https://warpem.github.io/warp/user_guide/warptools/quick_start_warptools_tilt_series/](https://warpem.github.io/warp/user_guide/warptools/quick_start_warptools_tilt_series/)
2. [https://warpem.github.io/warp/reference/warptools/processing_tilt_series_stacks/](https://warpem.github.io/warp/reference/warptools/processing_tilt_series_stacks/)
3. [https://github.com/warpem/warp/blob/main/WarpTools/Commands/Tiltseries/ImportTiltseries.cs](https://github.com/warpem/warp/blob/main/WarpTools/Commands/Tiltseries/ImportTiltseries.cs)
4. [https://groups.google.com/g/warp-em/c/5XIaNcA9ilM/m/E8m7LG89AgAJ](https://groups.google.com/g/warp-em/c/5XIaNcA9ilM/m/E8m7LG89AgAJ)
5. [https://groups.google.com/g/warp-em/c/ehf6icNKSwY](https://groups.google.com/g/warp-em/c/ehf6icNKSwY)
6. [https://gist.github.com/alisterburt/8744accf3f4696dd6d83fc9c4690612c](https://gist.github.com/alisterburt/8744accf3f4696dd6d83fc9c4690612c)
7. [https://groups.google.com/g/warp-em/c/OzZNq0qKZoc](https://groups.google.com/g/warp-em/c/OzZNq0qKZoc)
8. [https://groups.google.com/g/warp-em/c/4shNQFKYnv8/m/hq4bFu8fAQAJ](https://groups.google.com/g/warp-em/c/4shNQFKYnv8/m/hq4bFu8fAQAJ)