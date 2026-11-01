# All headline values

Normalised shear y is the eps=1e-14, delta=1e-60 evaluation. Defect is the relative distance from the isotropic tensor. Gaps use the accepted stored points and the baseline constructions. The complete precision and selection rules are in [methods](methods.md).

| nu_s | Solid fraction | y | Isotropy defect | Gap before | Gap after | Angle after (degrees) | Geometry |
|---|---|---|---|---|---|---|---|
| 0.3 | 0.01 | 0.903210 | 0.0006869 | 0.706853 | 0.093040 | 106 | [JSON](../data/certified/zero_bulk_aniso_nu0.3_th0.01.json) |
| 0.3 | 0.05 | 0.901902 | 0.0004814 | 0.705809 | 0.094298 | 106 | [JSON](../data/certified/zero_bulk_aniso_nu0.3_th0.05.json) |
| 0.3 | 0.1 | 0.899596 | 0.0003588 | 0.704435 | 0.096515 | 106 | [JSON](../data/certified/zero_bulk_aniso_nu0.3_th0.1.json) |
| 0.3 | 0.2 | 0.893809 | 0.0009249 | 0.701434 | 0.101551 | 107 | [JSON](../data/certified/zero_bulk_aniso_nu0.3_th0.2.json) |
| 0.3 | 0.3 | 0.890614 | 0.0009394 | 0.698041 | 0.104606 | 107 | [JSON](../data/certified/zero_bulk_aniso_nu0.3_th0.3.json) |
| 0.3 | 0.5 | 0.880323 | 0.0005866 | 0.689708 | 0.112557 | 108 | [JSON](../data/certified/zero_bulk_aniso_nu0.3_th0.5.json) |
| 0.3 | 0.7 | 0.869344 | 0.0009211 | 0.678393 | 0.117866 | 108 | [JSON](../data/certified/zero_bulk_aniso_nu0.3_th0.7.json) |
| 0.45 | 0.01 | 0.919405 | 0.0007702 | 0.706298 | 0.077472 | 106 | [JSON](../data/certified/zero_bulk_aniso_nu0.45_th0.01.json) |
| 0.45 | 0.05 | 0.917660 | 0.0008127 | 0.702954 | 0.079150 | 106 | [JSON](../data/certified/zero_bulk_aniso_nu0.45_th0.05.json) |
| 0.45 | 0.1 | 0.915568 | 0.0009250 | 0.698511 | 0.080752 | 106 | [JSON](../data/certified/zero_bulk_aniso_nu0.45_th0.1.json) |
| 0.45 | 0.2 | 0.912089 | 0.0009214 | 0.688617 | 0.083608 | 108 | [JSON](../data/certified/zero_bulk_aniso_nu0.45_th0.2.json) |
| 0.45 | 0.3 | 0.908447 | 0.0009256 | 0.677094 | 0.086565 | 109 | [JSON](../data/certified/zero_bulk_aniso_nu0.45_th0.3.json) |
| 0.45 | 0.5 | 0.901013 | 0.0009451 | 0.647090 | 0.089467 | 109 | [JSON](../data/certified/zero_bulk_aniso_nu0.45_th0.5.json) |
| 0.45 | 0.7 | 0.893605 | 0.0009185 | 0.601391 | 0.090807 | 116 | [JSON](../data/certified/zero_bulk_aniso_nu0.45_th0.7.json) |

The normalised shear range is 0.869344 to 0.919405 when rounded to six decimal places. The full 90-digit re-evaluation of all 14 geometries passed; [its complete output](../results/reproduction.json) records a maximum absolute bulk coordinate of 2.84e-17 and a maximum isotropy defect of 0.0009451. The rounded-parameter dilation residual is checked separately: the largest is about 2.06e-13, for nu_s=0.45 and theta=0.7, with the other 13 about 1e-15.

Across these cases, the baseline gap ranges from 0.6013908 to 0.7068529 and the augmented gap from 0.0774724 to 0.1178660. These ranges compare the sampled support of the same outer square with the two sets of constructions; they are not percentages of missing materials or continuous-angle distance bounds.

The gap uses 555 of the 556 records. The excluded record is `tree_iso_nu0.45_th0.1_c3.0.json`, whose finest-wall defect is about 0.001133. It is kept in the dataset and in the complete re-evaluation path.

The solid-base limits in the additional families are included in the map, but are outside this headline table. No result between the seven tested fractions is inferred by interpolation.
