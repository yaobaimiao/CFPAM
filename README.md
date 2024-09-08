# CFPAM
## Environment Requirement
create enviroment and intall as following: `pip install -r requirements.txt`
## Data Format
trainset: CoCo-9k，DUTS class
testset: CoCA, CoSOD3k, Cosal2015
Download the dataset from [Baidu Driver](https://pan.baidu.com/s/1VfqvOvbKPef2X-qPHYGsdQ?pwd=8epc) (8epc) and unzip them to './dataset'. Then the structure of the './dataset' folder will show as following:
```markdown
-- dataset
   |-- train_data
   |   |-- | CoCo9k
   |   |-- | DUTS_class
   |   |-- | DUTS_class_syn
   |   |-- |-- | img_png_seamless_cloning_add_naive
   |   |-- |-- | img_png_seamless_cloning_add_naive_reverse_2
   |-- test_data
   |   |-- | CoCA
   |   |-- | CoSal2015
   |   |-- | CoSOD3k
```
## Training model
1. Download the pretrained [PVTv2](https://github.com/whai362/PVT) model and put it into ./models.
2. Run `python 1train.py`.
## Testing model
1. Download our trained model from [Baidu Driver](https://pan.baidu.com/s/18Ie11b0DJODRhSAt08shaw?pwd=toex) (toex) and put it into ./data1/cfpam.
2. Run python test.py.
3. The prediction images will be saved in ./CoSODmaps/pred and the evaluation scores will be written in ./eval_result/result.txt.
4. The evaluation tool please follow: [eval-co-sod](https://github.com/zzhanghub/eval-co-sod)
## Prediction results
1. The co-saliency maps of CFPAM can be found at [CFPAM](https://pan.baidu.com/s/11Jh1ky5L8_NO-2pzPXdkhQ?pwd=y39s) (y39s).

