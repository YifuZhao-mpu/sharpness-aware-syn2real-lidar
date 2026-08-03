> **SUPERSESSION NOTICE (2026-07-16, added post-run).** The original language
> below ("SEALED", "before any SemanticPOSS data or label content was read",
> "official test split") is superseded by the Post-run changelog at the bottom of
> this file: the dataset was concurrently in use in this workspace for an
> unrelated project, only the two named contrasts were prespecified, and the
> split is the customary evaluation split. The body below is preserved verbatim
> as the historical record of what was frozen; the changelog governs
> interpretation.

# SemanticPOSS SEALED-EVALUATION FREEZE MANIFEST
Frozen: 2026-07-16 (before any SemanticPOSS data or label content was read).

## Purpose
Round-9 review (P1): KITTI/STF were development targets (method tournament used
mid-training target evals). This manifest freezes a one-shot, prospective evaluation
on SemanticPOSS — verified untouched by this project (zero references in any code,
log, or document before this date; the only access so far is directory listing /
file counts, recorded below).

## Access log (before freeze)
- 2026-07-16: directory listing of sequences 00–05; scan counts (488/500/500/500/500/500).
  No .bin or .label file contents read.
- Declared single pre-eval peek (executed by eval_poss.py at run time, recorded in its
  output): the first scan of seq 03 (000001; sequence numbering starts at 000001) — intensity-channel min/max to set /255 vs raw scale,
  and the label-file BYTE COUNT to infer columns-per-point (semantic content unused).

## Frozen protocol
- Eval split: sequence 03 (official SemanticPOSS test split), all 500 scans, batch 1,
  voxel 0.05 m, final-epoch checkpoints, NO TTA, NO retuning, NO target-based selection.
- Label space: common 11 classes (documentation-derived subsumption):
  person, rider, car, trunk, plants, traffic-sign, pole, building, fence, bike, ground.
  POSS raw→common: 4,5→person; 6→rider; 7→car; 8→trunk; 9→plants; 10,11,12→traffic-sign;
  13→pole; 15→building; 17→fence; 21→bike; 22→ground; 0,14 (trashcan),16 (cone/stone)→ignore.
  Prediction (19-cls KITTI space)→common: car,truck,other-vehicle→car;
  bicycle,motorcycle→bike; bicyclist,motorcyclist→rider; person→person;
  road,parking,sidewalk,other-ground,terrain→ground; building→building; fence→fence;
  vegetation→plants; trunk→trunk; pole→pole; traffic-sign→traffic-sign.
- Metric: point-level mIoU over the 11 common classes (IoUMeter, ignore=255).
- Checkpoints: ALL 44 non-pilot, non-broken final checkpoints (list + SHA256 below).
- Secondary sensitivity: same checkpoints after common BN recalibration on the fixed
  unaugmented source stream (bn_recalib.py, stride 247 = 804 scans, cumulative BN).
- Analysis plan (fixed): report per-model mIoU for ALL checkpoints; per-method
  mean ± sample SD; seed-paired diffs (SAM−source-only, SAM+PolarMix−PolarMix,
  paired t + 95% CI). Results are published REGARDLESS of outcome. No post-hoc
  exclusions. Any code bug discovered after label access will be fixed, rerun, and
  disclosed in a changelog appended to this manifest.

## Code hashes (SHA256)
771983e52c5ce64a6c1cf70f6c57ba38cd37863e3eaba3f2f42712ad827038d6  eval_poss.py
74fd3632eba36314e6fd6366520a2f26638fedd5f0d5658fffee6670e5290306  bn_recalib.py
5a4eeac46878a66aedbd40a879a9620612f801c5891f7b9d1532780c78e5b83d  datasets.py
a1d4986ce62d3aa46e38af305b8831098c547156dd176b29e729a5bb7ef961f0  model.py
a48fe61d3d62b610314d3c336d7a6b46aead222f34172c6d1887ac8042ce4bef  utils.py

## Checkpoint hashes (SHA256, exp/<run>/model.pth)
25641a3dea98d60b4a7bf71be7db64d331c589682fcaff8973ce94b8971bbe21  cr1_none_s1/model.pth
fbe2518e34213c513cf0395c178724f19c58a834d62194a5dc86c81256c519f6  cr1_none_s2/model.pth
e3587a1f128ce08fcfc16b297efa36ce113033e280b6555540c47ef4e3e1c69f  cr1_none_s3/model.pth
3c2c907aad4a5ca665c93e22fec74186341a19e70e22d9cd2700242e0eecaff3  cr1_sam_s1/model.pth
0232d67859c9655a75a31716906f299797e9a6ce0492c0f5db7036902ef44aba  cr1_sam_s2/model.pth
dd20ad5db9a1ddc4d3bf72f4814a351ca6acdf2269aede1323fbd5ec63a07631  dr_full_s1/model.pth
a0606060010284c485652da5e79e4d28dc4ff5138c48655afbd9bda3a6bce7ca  lasermix_s1/model.pth
9b0ebabf86ef537ca52ac944e319a8f0cb3918672276fd44609d34f540a25107  neg_consist_main/model.pth
493bd874500a6a332b99bffa69e2f5474eb706de2bc282e4f60cc4cd51b0e3e0  neg_whiten_main/model.pth
e3513e21f9cab747529780255c35371ea121642cea39cb39cf83b52d6de223c9  none_full/model.pth
8468b009c24e099fbe5e24c526481df32508c927ce0964e0f5edac26a4ef92ea  none_long_s1/model.pth
6548683f02cddc4eda6305db08b3dcda01fc0b4ad265b1424838f9b6be9ecb8c  none_s2/model.pth
428a60368cbc35ce631749630d3a71330e5b02391758a42394112f5c97676bb6  none_s3/model.pth
dd1095e9104e62479ed94c82a4a8ff68a28223001f1b003c24d05ad44d60c450  pointdr_s1/model.pth
416719bd5e95a89bb55fef88b05baab7d7e0022c008c4c6b6e519304447881f3  pointdr_s2/model.pth
a92bd2807662067d6655aeb703afe0a334e4475156eac724f613b188bd3eb5f2  pointdr_s3/model.pth
e411b4edf42ab53b1a917c58a8878cac544a95730f3184acb59fb682ec39a233  polarmix_s1/model.pth
7406bc9a966ba6c7671aa60f49f43922b207e814778d457c426a0d36abc63703  polarmix_s2/model.pth
5f05551f2a7ddc24b182022b7efdaa5033ef51a880d502badb2e8d3f557714b3  polarmix_s3/model.pth
d3385944d650e4c44c1e18aab8e97c8faf2f8e4e6f9cef6965db48da372ffda2  sac05fix_full/model.pth
b3658d876b0aa0f0cf28585140c94b4d836037bd49053770c7662560f12368eb  sac05_full/model.pth
1451d7c85767c55bb791fc4b33d9a601c5e63d0ab81afbaf281c5e678d00fb2f  sam001_s1/model.pth
cd90178911fd0ef4774727b20621d7a9dd8cb463a6c2f2cfb13c52b44f22bc58  sam005_full/model.pth
fd8d872c6a64ff5c4d4acc4e0d8b04ddf94709c942a02885708d30dc75b30119  sam005_s2/model.pth
4beeb4e0a8f8ae4d607a6b243b124397147ae8cd95e97e5ff836ce54374fce5d  sam005_s3/model.pth
e24b577fe1182d5b8f84a1b1ee83a12bbed69281647933c1421e58f1f29cf208  sam010_s1/model.pth
33b8e9556d1777f6a57af192b8dffbb86c3444dd67fce7815292ed4de7c52476  sam020_s1/model.pth
6ef914ba8a6545185214f5d53dae859700bad6010d181f23dd2143ec89246c77  samdr_s1/model.pth
1380f26947fca360f833e1d5d51a024c5ba9897263dfc616e5ab3e3ed7c59b36  sampolar010_s1/model.pth
a4f182e99ee197bd053dae0175d595587211f7d7eea00ad8025e743574a67a92  sampolarlm_s1/model.pth
9f9e3a5d9f8d3cd6c5fd719d3fdac0fc341234376b5537aba5e849a17756af2b  sampolar_s1/model.pth
771fcd166a2abed390247cd6670be61646e828728182388bc46c0242ca318e0a  sampolar_s2/model.pth
df7dc498dd812ccf26eb01295ae4c795c24d8de8afaa975e61d6ed36affe8e05  sampolar_s3/model.pth
82faea5be1d2230d7a61bc8877eb5f7fbe1f8371a878770bbddc27655331dee1  samsens_s1/model.pth
ce3a23f42c0b0078d3b2e4daec24f58a284e08065575a34a81164e5a7a862c66  samsens_s2/model.pth
abdb59d0e66f3872eb0b3fb196e53fb7314b13662533c8bd57a07fd1e11341c4  samwmix_s1/model.pth
f41b4ae542ad8a97f801864eff53ccb8b647a60324a935381a61343777726326  samwmix_s2/model.pth
d7e87faffd277ac26081f6e3ae4e1dfadbf1246b21605077d3a1bb20a42dee02  swa_s1/model.pth
c985ec2c75ee896f7f190f3ba20ed9197dc7a8bfd93557b4101fdb31bd43f398  swa_s2/model.pth
c3ac4350d497ed6fa0dcb8710460f49e34e7eac127abffa576ceb95d78a1f0af  swa_s3/model.pth
46ff0abb2cfe8b1be43186cb4e189e040911fadbe63ce08a7eb8a890bc1ee2bb  weather_s1/model.pth
afda3562c7bd568f28a0b3d4d813e35a4612cbfa07ff2c78bdb75e137793511b  wmix_s1/model.pth
2b10ffe8a0b2d075dc196a3aa4730c0ad62c2d7eb77b7dd284645ddfa254e9b9  wmix_s2/model.pth
099a859b30a85f705ac3a373148d6ba3338aa0f3844711f3058c65f7ff45cb77  wmix_s3/model.pth

## Post-run changelog (2026-07-16)
- Run executed ONCE (06:53–07:13 local; eval_poss.py unchanged from the hashed
  version; log: code/eval_poss.out). Declared intensity peek recorded:
  scan 000001, min 1.0 / max 255.0 -> scale 1/255. (An earlier version of this
  changelog misstated the times as 07:53–08:12; corrected same day.)
- No bugs discovered after label access; no protocol changes; no exclusions.
- Analysis script poss_analysis.py was written before run completion, but only the
  two contrasts named in the frozen plan above (SAM−source-only,
  SAM+PolarMix−PolarMix) count as PRESPECIFIED; the additional contrasts it prints
  (PolarMix−source-only, SAM+PolarMix−SAM, cr1) are POST-HOC and are labelled as
  such in the manuscript. All 44 checkpoints reported in Supplementary Table S5.
- PROVENANCE CORRECTION (same day, found in round-4 external review): although
  SemanticPOSS was never referenced in THIS project before the freeze, the dataset
  was concurrently in use in the same workspace for an unrelated project
  (HybridTM TTA study; label-based seq03 evaluations on 2026-07-13/14 and a rerun
  manifest dated 2026-07-16 00:43, before this freeze). The evaluation is
  therefore characterized in the manuscript as a DEVELOPMENT-INDEPENDENT EXTERNAL
  EVALUATION (checkpoints, mapping, metric frozen before this study's first
  access), NOT as an author-untouched sealed confirmation.
- Headline frozen contrasts (as-is): SAM − source-only = +2.61 (per-seed +2.71/
  +4.22/+0.90); cr1 +2.10; PolarMix − source-only = −0.68; SAM+PolarMix − SAM = −0.38.
